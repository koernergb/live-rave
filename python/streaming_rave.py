"""Cache-hoisted streaming RAVE.

Replaces every CachedConv1d / CachedConvTranspose1d / standalone
CachedPadding1d inside a RAVE model with a forward that reads its state from
(and writes the updated state back to) an explicit per-call context, instead
of self-held buffers.

The resulting wrapper graph has the functional signature

    encoder: (x, eps, c0, ..., cN) -> (z, c0', ..., cN')
    decoder: (z, noise, c0, ..., cN) -> (y, c0', ..., cN')

where the caches fully describe the streaming state. The exact computation
mirrors `VariationalScriptedRAVE` (scripts/export.py): encode projects the
reparametrized (sampled) latent through the latent PCA; decode re-injects the
whitened latent with explicit gaussian noise. The states are then explicit
ONNX inputs/outputs, so a JS runtime can thread them across buffers.
"""

import types

import cached_conv as cc
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHED_PAD = cc.CachedPadding1d
CACHED_CONV1D = cc.CachedConv1d
CACHED_CONVT1D = cc.CachedConvTranspose1d

# Active per-call context. Patched module forwards are instance-bound methods,
# so `module(x)` dispatches the underlying function as `fn(module, x)` —
# there is no room for an extra ctx argument. The closures instead read this
# module-global, which the generated wrapper forward sets for the duration of
# each call. Safe because graphs are executed sequentially.
_CTX = None


def _set_ctx(ctx):
    global _CTX
    _CTX = ctx

STATE_FN = {
    "conv1d": lambda m: [
        ("pad", m.downsampling_delay.padding),
        ("pad", m.cache.padding),
    ],
    "convt1d": lambda m: [("cache", 2 * m.padding[0])],
    "pad": lambda m: [("pad", m.padding)],
}


class _Ctx:
    """Per-call cache registry threaded through the patched module tree."""

    __slots__ = ("inputs", "outputs")

    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = list(inputs)

    def set(self, index, tensor):
        self.outputs[index] = tensor


def collect_hoisted_modules(root):
    """Ordered flat list of (path, module, kind) to hoist in `root`.

    Standalone CachedPadding1d instances living *inside* a CachedConv1d (its
    downsampling_delay / cache pads) are handled by the parent conv and are
    excluded here.
    """
    conv_children = set()
    for m in root.modules():
        if isinstance(m, CACHED_CONV1D):
            conv_children.add(id(m.downsampling_delay))
            conv_children.add(id(m.cache))

    out = []
    for name, m in root.named_modules():
        if isinstance(m, CACHED_CONV1D):
            out.append((name, m, "conv1d"))
        elif isinstance(m, CACHED_CONVT1D):
            out.append((name, m, "convt1d"))
        elif isinstance(m, CACHED_PAD) and id(m) not in conv_children:
            out.append((name, m, "pad"))
    return out


def hoist_registry(root, extra_first=(), extra_last=()):
    """Ordered (name, module, kind) list: `extra_first` entries, then every
    hoisted module in `root`, then `extra_last` entries.

    The encoder graph hoists the PQMF analysis filterbank (`extra_first`) ahead
    of the encoder convs so its two cache slots own indices 0/1; the decoder
    graph hoists the PQMF synthesis filterbank (`extra_last`) after the decoder
    convs. Both sets are stateful across buffers and must be threaded like any
    other cache.
    """
    return list(extra_first) + collect_hoisted_modules(root) + list(extra_last)


# ---------------------------------------------------------------------------
# Hoisted per-module ops. `x` and the caches are tensors.
# ---------------------------------------------------------------------------


def hoisted_pad(m, cache, x):
    p = m.padding
    if p:
        y = torch.cat([cache, x], -1)
        new_cache = y[..., -p:]
        if m.crop:
            y = y[..., :-p]
    else:
        y = x
        new_cache = cache
    return y, new_cache


def hoisted_conv1d(m, ds_cache, cache, x):
    pd = m.downsampling_delay.padding
    pc = m.cache.padding
    if pd:
        y = torch.cat([ds_cache, x], -1)
        new_ds = y[..., -pd:]
        y = y[..., :-pd]
    else:
        y = x
        new_ds = ds_cache
    if pc:
        z = torch.cat([cache, y], -1)
        new_c = z[..., -pc:]
    else:
        z = y
        new_c = cache
    out = F.conv1d(z, m.weight, m.bias, m.stride, m.padding, m.dilation,
                   m.groups)
    return out, new_ds, new_c


def hoisted_convt1d(m, cache, x):
    y = F.conv_transpose1d(x, m.weight, None, m.stride, 0, m.output_padding,
                           m.groups, m.dilation)
    padd = 2 * m.padding[0]
    if padd:
        y = torch.add(y, F.pad(cache, (0, y.shape[-1] - cache.shape[-1])))
        new_cache = y[..., -padd:]
        y = y[..., :-padd]
    else:
        new_cache = cache
    if m.bias is not None:
        y = y + m.bias.unsqueeze(-1)
    return y, new_cache


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


def patch_cached_modules(root, extra_first=(), extra_last=()):
    """Replace each hoisted cached module's forward with a context-threaded
    version. Returns a `(records, shapes)` pair.

    Must be called AFTER the warmup pass so the modules' `pad`/`cache` buffers
    are registered (their shapes describe the state tensors).
    """
    mods = hoist_registry(root, extra_first, extra_last)
    return patch_module_list(mods)


def patch_module_list(mods):
    records = []
    offset = 0
    for name, m, kind in mods:
        n_states = len(STATE_FN[kind](m))
        indices = list(range(offset, offset + n_states))
        offset += n_states

        if kind == "conv1d":
            i_ds, i_c = indices

            def forward(_self, x, _m=m, _ds=i_ds, _c=i_c):
                ctx = _CTX
                out, new_ds, new_c = hoisted_conv1d(
                    _m, ctx.inputs[_ds], ctx.inputs[_c], x)
                ctx.set(_ds, new_ds)
                ctx.set(_c, new_c)
                return out

            shapes = [list(m.downsampling_delay.pad.shape[1:]),
                      list(m.cache.pad.shape[1:])]
        elif kind == "convt1d":
            i_cache = indices[0]

            def forward(_self, x, _m=m, _i=i_cache):
                ctx = _CTX
                out, new_cache = hoisted_convt1d(_m, ctx.inputs[_i], x)
                ctx.set(_i, new_cache)
                return out

            shapes = [list(m.cache.shape[1:])]
        else:
            i_cache = indices[0]

            def forward(_self, x, _m=m, _i=i_cache):
                ctx = _CTX
                out, new_cache = hoisted_pad(_m, ctx.inputs[_i], x)
                ctx.set(_i, new_cache)
                return out

            shapes = [list(m.pad.shape[1:])]

        m.forward = types.MethodType(forward, m)
        records.append({"name": name, "kind": kind, "indices": indices})
    return records


def zero_caches(graph):
    """Zero-init state tensors for a CachedGraph (its initial streaming
    state)."""
    return graph.initial_caches()


# ---------------------------------------------------------------------------
# Wrapper graph (generated forward with one argument per cache)
# ---------------------------------------------------------------------------


def _build_forward_source(encoder, n_caches):
    caches = ", ".join(f"c{i}" for i in range(n_caches))
    if encoder:
        params = "self, x, eps" + (", " + caches if caches else "")
    else:
        params = "self, z, noise" + (", " + caches if caches else "")
    lines = [
        f"def forward({params}):",
    ]
    if encoder:
        # The encoder is exported as a pure straight-line function (see
        # `_puregen`): no _Ctx global, no patched module traversal. It applies
        # pqmf, runs the encoder, and projects the reparametrized latent the
        # same way `VariationalScriptedRAVE` does. `self._pure_fn` was built in
        # `__init__` with signature (x, eps, c0..cN) -> (z, c0'..cN').
        lines += [
            "    return self._pure_fn(x, eps, " + caches + ")",
        ]
    else:
        lines += [
            "    inputs = [{caches}]".format(caches=caches),
            "    ctx = _Ctx(inputs)",
            "    set_ctx(ctx)",
            "    z = torch.cat([z, noise], 1)",
            "    z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))",
            "    z = z + self.latent_mean.unsqueeze(-1)",
            "    y = self.model.decoder(z)",
            "    y = self.model.pqmf.inverse(y)",
            "    return (y, *ctx.outputs)",
        ]
    return "\n".join(lines)


class CachedGraph(nn.Module):

    def __init__(self, model, encoder=True, device="cpu", latent_size=None):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.device = device

        root = model.encoder if encoder else model.decoder
        if encoder:
            extra_first = [("pqmf.forward_conv",
                            model.pqmf.forward_conv, "conv1d")]
            extra_last = ()
        else:
            extra_first = ()
            extra_last = [("pqmf.inverse_conv",
                           model.pqmf.inverse_conv, "conv1d")]
        self.registry = hoist_registry(root, extra_first, extra_last)
        self.records = patch_module_list(self.registry)
        self.n_caches = max(
            (r["indices"][-1] + 1 for r in self.records), default=0)

        self.register_buffer("latent_pca", model.latent_pca)
        self.register_buffer("latent_mean", model.latent_mean)
        self.latent_size = (latent_size if latent_size is not None
                            else model.latent_size)
        self.full_latent_size = model.latent_size

        src = _build_forward_source(encoder, self.n_caches)
        g = {"_Ctx": _Ctx, "torch": torch, "F": F, "set_ctx": _set_ctx}

        if encoder:
            # Generate the pure straight-line encoder fn in this graph's
            # namespace, then bind it to the instance. The generated forward
            # simply calls it with the threaded cache arguments.
            import _puregen as pg
            import rave.blocks as rb
            import rave.pqmf as pqmfmod

            pg.sr.Residual = rb.Residual
            g.update({
                "_G": model,
                "_M": dict(model.named_modules()),
                "_LS": self.latent_size,
                "_pad": hoisted_pad,
                "_conv1d": hoisted_conv1d,
                "reverse_half": pqmfmod.reverse_half,
                "__name__": "_pureencoder",
            })
            pure_src = pg.make_encoder_function(model, self.latent_size)[1]
            exec(pure_src, g)
            self._pure_fn = g["_fn_"]

        exec(src, g)
        self.forward = types.MethodType(g["forward"], self)

    def cache_shapes(self):
        shapes = []
        for name, m, kind in self.registry:
            if kind == "conv1d":
                shapes += [list(m.downsampling_delay.pad.shape[1:]),
                           list(m.cache.pad.shape[1:])]
            elif kind == "convt1d":
                shapes += [list(m.cache.shape[1:])]
            else:
                shapes += [list(m.pad.shape[1:])]
        return shapes

    def initial_caches(self):
        shapes = self.cache_shapes()
        return [
            torch.zeros([1] + s, device=self.device, dtype=torch.float32)
            for s in shapes
        ]

    def warmup_caches(self):
        """Current (live) register values in registry order -- the streaming
        state left behind by the warmup pass, which the reference carries.
        Threading these instead of zeros keeps the exported graphs on the same
        trajectory as the pristine eager model.
        """
        frames = []
        for name, m, kind in self.registry:
            if kind == "conv1d":
                frames += [m.downsampling_delay.pad[:1],
                           m.cache.pad[:1]]
            elif kind == "convt1d":
                frames += [m.cache[:1]]
            else:
                frames += [m.pad[:1]]
        return [f.detach().clone() for f in frames]


def export_graph(graph, output_dir, prefix, block_size, latent_size,
                 full_latent_size, ratio):
    """Trace and export the CachedGraph as ONNX. Returns metadata dict."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    if graph.encoder:
        x = torch.zeros(1, 1, block_size, device=graph.device)
        eps = torch.zeros(1, full_latent_size, block_size // ratio,
                          device=graph.device)
        example = (x, eps, *graph.initial_caches())
        input_names = (["x", "eps"] +
                       [f"cache_{i}" for i in range(graph.n_caches)])
        output_names = (["z"] +
                        [f"cache_{i}_out" for i in range(graph.n_caches)])
    else:
        z = torch.zeros(1, latent_size, block_size // ratio,
                        device=graph.device)
        noise = torch.zeros(1, full_latent_size - latent_size,
                            block_size // ratio, device=graph.device)
        example = (z, noise, *graph.initial_caches())
        input_names = (["z", "noise"] +
                       [f"cache_{i}" for i in range(graph.n_caches)])
        output_names = (["y"] +
                        [f"cache_{i}_out" for i in range(graph.n_caches)])

    path = os.path.join(output_dir, f"{prefix}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            graph,
            example,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=False,
        )

    return {
        "prefix": prefix,
        "onnx": os.path.basename(path),
        "n_caches": graph.n_caches,
        "latent_size": latent_size,
        "full_latent_size": full_latent_size,
        "ratio": ratio,
        "cache_shapes": graph.cache_shapes(),
    }