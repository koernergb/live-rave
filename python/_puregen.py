"""Module-tree -> straight-line pure function generator for the RAVE encoder.

Translates the encoder's module tree into a sequence of pure statements:
every hoisted CachedConv1d / CachedPadding1d becomes an explicit
variable-updating block (mirroring `hoisted_conv1d` / `hoisted_pad`), and every
container (CachedSequential, Residual, DilatedUnit) is unrolled into plain
sequential statements with monotonically-named locals. No module-global state
and no mutable lists are touched, so at trace time each value has exactly one
producer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import streaming_rave as sr


class PureEncoder:
    """Generate a pure encoder source from a RAVE model."""

    def __init__(self, model, latent_size):
        self.model = model
        self.root = model.encoder
        self.latent_size = latent_size
        self.lines = []
        self._nc = 0
        self._n = 0
        # full model-hoc dotted paths for weight/padding refs
        self.full = dict(model.named_modules())
        # dotted-path (relative to root) -> module, for weight/padding refs
        self.modules = dict(self.root.named_modules())
        # prefix mapping encoder-relative names to model-full names
        self._pfx = ""
        for n, mm in model.named_modules():
            if mm is model.encoder:
                self._pfx = (n + ".") if n else ""
                break
        # slot index per module id, allocated in DFS pre-order
        self.py = {}
        # debug: (kind, module-path-relative-to-root, emitted-name)
        self.call_log = []

    def _full(self, name):
        """Map an encoder-relative dotted path to its model-full path."""
        return self._pfx + name

    # -- helpers -------------------------------------------------------------

    def _fresh(self):
        self._n += 1
        return f'x{self._n}'

    def emit(self, line):
        self.lines.append(line)

    # -- slot pre-allocation pass (mirrors collect_hoisted_modules) ----------

    def _alloc(self):
        i = self._nc
        self._nc += 1
        return i

    def _slots(self, m):
        if m.__class__ is sr.CACHED_CONV1D:
            s = (self._alloc(), self._alloc())   # (downsampling_delay, cache)
        else:
            s = (self._alloc(),)
        self.py[id(m)] = s
        return s

    def _reserve(self, module):
        """DFS pre-order: allocate slots for every hoisted module."""
        for _, child in module.named_children():
            if child.__class__ is sr.CACHED_CONV1D:
                self._slots(child)
            elif child.__class__ is sr.CACHED_PAD:
                self._slots(child)
            else:
                self._reserve(child)

    # -- emission ------------------------------------------------------------

    def _emit_module(self, name, m, cur):
        c = m.__class__
        if c is sr.CACHED_CONV1D:
            ds, cc = self.py[id(m)]
            out = self._fresh()
            self.emit(f'{out}, c{ds}, c{cc} = _conv1d(_M[{self._full(name)!r}], '
                      f'c{ds}, c{cc}, {cur})')
            self.call_log.append(('conv', name, out))
            return out
        if c is sr.CACHED_PAD:
            p = self.py[id(m)][0]
            out = self._fresh()
            self.emit(f'{out}, c{p} = _pad(_M[{self._full(name)!r}], c{p}, {cur})')
            return out
        if isinstance(m, nn.LeakyReLU):
            out = self._fresh()
            self.emit(f'{out} = F.leaky_relu({cur}, 0.2)')
            return out
        if isinstance(m, torch.nn.Identity):
            return cur
        if isinstance(m, sr.Residual):
            return self._emit_residual(name, m, cur)
        # containers: recurse children in order
        for child_name, child in m.named_children():
            cur = self._emit_module(f'{name}.{child_name}', child, cur)
        return cur

    def _emit_residual(self, name, m, cur):
        # AlignBranches pads the *identity* path by the branch's cumulative
        # delay (delays=[module.cd, 0] -> paddings=[0, cd]): the branch runs on
        # the undelayed x (its convs self-compensate), and the delayed identity
        # is added back afterwards.
        v = cur
        cur = self._emit_module(name + '.aligned.branches.0', pad_branch(m), cur)
        pad = m.aligned.paddings[1]
        p = self.py[id(pad)][0]
        delayed = self._fresh()
        self.emit(f'{delayed}, c{p} = _pad(_M[{self._full(name + ".aligned.paddings.1")!r}], '
                  f'c{p}, {v})')
        res = self._fresh()
        self.emit(f'{res} = {cur} + {delayed}')
        self.call_log.append(('residual_in', name, v))
        self.call_log.append(('residual_out', name, res))
        return res

    def generate(self):
        # The PQMF analysis filterbank is stateful (its forward_conv is a
        # CachedConv1d) and owns the first two cache slots.
        self._slots(self.model.pqmf.forward_conv)
        self._reserve(self.root)
        caches = ', '.join(f'c{i}' for i in range(self._nc))
        self.emit('def _fn_(x, eps' + (', ' + caches if caches else '') + '):')
        body = []
        ps, pc = self.py[id(self.model.pqmf.forward_conv)]
        a = self._fresh()
        self.emit(f'{a}, c{ps}, c{pc} = _conv1d(_M["pqmf.forward_conv"], '
                  f'c{ps}, c{pc}, x)')
        xq = self._fresh()
        self.emit(f'{xq} = reverse_half({a})')
        cur = xq
        for child_name, child in self.root.named_children():
            cur = self._emit_module(child_name, child, cur)
        self.emit(f'mean, scale = torch.chunk({cur}, 2, 1)')
        self.emit('std = F.softplus(scale) + 1e-4')
        self.emit('z = eps * std + mean')
        self.emit('z = z - _G.latent_mean.unsqueeze(-1)')
        self.emit('z = F.conv1d(z, _G.latent_pca.unsqueeze(-1))')
        self.emit('z = z[:, :_LS]')
        ret = ', '.join(['z'] + [f'c{i}' for i in range(self._nc)])
        self.emit(f'return ({ret})')
        head, body = self.lines[0], self.lines[1:]
        return head + '\n' + '\n'.join('    ' + l for l in body)


def pad_branch(m):
    """The module consumed by a Residual's branch (delay-compensated)."""
    return m.aligned.branches[0]


def make_encoder_function(model, latent_size):
    """Build, exec and return the pure encoder function (and its config)."""
    pe = PureEncoder(model, latent_size)
    src = pe.generate()
    import rave.pqmf as pqmfmod
    g = {
        "_G": model,
        "_M": dict(model.named_modules()),
        "_LS": latent_size,
        "_pad": sr.hoisted_pad,
        "_conv1d": sr.hoisted_conv1d,
        "reverse_half": pqmfmod.reverse_half,
        "F": F,
        "torch": torch,
        "__name__": "_pureencoder",
    }
    ns = {}
    exec(src, g, ns)
    fn = ns['_fn_']
    fn._pure_src = src
    fn._nc = pe._nc
    fn._call_log = list(pe.call_log)
    return fn, src, pe.modules