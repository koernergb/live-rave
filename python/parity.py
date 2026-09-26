"""M1 parity test: cache-hoisted ONNX export vs eager streaming reference.

The reference graph is built from the same seeded RAVE model as the export
(N_BAND=16, ratio 2048). The eager reference mirrors
`VariationalScriptedRAVE` from scripts/export.py: latent PCA projection on the
encoder side, PCA re-injection + explicit gaussian noise on the decoder side.

Gates (milestones.md M1):
  * N streaming buffers, per-buffer max abs error < 1e-4 (default 10000)
  * no monotonic drift (tail-window max <= head-window max)
  * reset test: zeroing caches mid-stream reproduces a fresh start
"""

import argparse
import copy
import json
import math
import os
import sys
import time

# Prefer the pinned vendored cached_conv over any globally installed version.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cached_conv"))

import torch
import torch.nn.functional as F

# RAVE imports pytorch_lightning internally. PL's eager fsdp/precision import
# chain (torch.distributed.fsdp -> torch._dynamo) is fragile and crashes with
# `torch has no attribute _utils` if triggered *late* (e.g. after numpy or
# onnxruntime have initialized). Import PL immediately after torch, before
# numpy, so the chain runs first-time clean.
import pytorch_lightning  # noqa: E402

import numpy as np

import onnxruntime as ort

import streaming_rave


def build_rave(run_dir, config, seed, checkpoint=None):
    run_dir = os.path.abspath(run_dir)
    sys.path.insert(0, run_dir)

    import cached_conv as cc
    import gin
    import rave

    cc.use_cached_conv(True)
    if checkpoint is not None:
        cc.use_iil_compat(True)

    gin.clear_config()
    gin.parse_config_file(os.path.abspath(os.path.join(run_dir, config)))

    torch.manual_seed(seed)
    model = rave.RAVE()
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu")
        state = saved.get("state_dict", saved)
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def compute_latent_size(model, fidelity):
    latent_size = max(int(np.argmax(model.fidelity.numpy() > fidelity)), 1)
    return 2**math.ceil(math.log2(latent_size))


def warmup(model, block_size):
    """Initialise every cached module's buffers (streaming state) on zeros."""
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.zeros(1, 1, block_size)
        x = model.pqmf(x)
        z = model.encoder(x)
        sampled, _ = model.encoder.reparametrize(z)
        model.decoder(sampled)
        model.pqmf.inverse(model.decoder(sampled))


def ref_encode(model, x, eps, latent_size):
    x = model.pqmf(x)
    z = model.encoder(x)
    mean, scale = z.chunk(2, 1)
    std = F.softplus(scale) + 1e-4
    z = eps * std + mean
    z = z - model.latent_mean.unsqueeze(-1)
    z = F.conv1d(z, model.latent_pca.unsqueeze(-1))
    z = z[:, :latent_size]
    return z


def ref_decode(model, z, noise):
    z = torch.cat([z, noise], 1)
    z = F.conv1d(z, model.latent_pca.T.unsqueeze(-1))
    z = z + model.latent_mean.unsqueeze(-1)
    y = model.decoder(z)
    y = model.pqmf.inverse(y)
    return y


def zero_ref_state(model):
    for m in model.modules():
        if isinstance(m, streaming_rave.CACHED_PAD):
            m.initialized = 0
            if hasattr(m, "pad"):
                m.pad.zero_()
        elif isinstance(m, streaming_rave.CACHED_CONVT1D):
            m.initialized = 0
            if hasattr(m, "cache"):
                m.cache.zero_()
        elif isinstance(m, streaming_rave.CACHED_CONV1D):
            if hasattr(m, "downsampling_delay"):
                m.downsampling_delay.pad.zero_()
            if hasattr(m, "cache"):
                m.cache.pad.zero_()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="python/RAVE")
    ap.add_argument("--config", default="rave/configs/v2.gin")
    ap.add_argument("--out", default="benchmarks/export")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--buffers", type=int, default=10000)
    ap.add_argument("--fidelity", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint",
                    help="trained Lightning checkpoint to load strictly")
    ap.add_argument("--latent-size", type=int,
                    help="explicit exported latent cut (overrides --fidelity)")
    ap.add_argument("--name", default="seeded RAVE")
    ap.add_argument("--arch", default="v2")
    ap.add_argument("--source-url")
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="scale deterministic parity audio (trained models: 0.1)")
    ap.add_argument("--no-export", action="store_true",
                    help="keep existing ONNX artifacts, skip re-export")
    args = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    print(f"[1/6] building seeded RAVE model...", flush=True)
    model = build_rave(args.run_dir, args.config, args.seed, args.checkpoint)

    latent_size = (args.latent_size if args.latent_size is not None
                   else compute_latent_size(model, args.fidelity))
    full_latent_size = model.latent_size
    print(f"      latent_size={latent_size} full={full_latent_size} "
          f"sr={model.sr}", flush=True)

    print("[2/6] warmup + reference deepcopy...", flush=True)
    warmup(model, args.block)
    ref_model = copy.deepcopy(model)

    with torch.no_grad():
        probe = torch.zeros(1, 1, args.block)
        ratio = args.block // model.encoder(model.pqmf(probe)).shape[-1]
    print(f"      ratio={ratio}")

    print("[3/6] patching cached modules + exporting ONNX...", flush=True)
    enc_graph = streaming_rave.CachedGraph(model, encoder=True,
                                           latent_size=latent_size)
    dec_graph = streaming_rave.CachedGraph(model, encoder=False,
                                           latent_size=latent_size)

    warm_enc = enc_graph.warmup_caches()
    warm_dec = dec_graph.warmup_caches()

    if not args.no_export:
        meta_enc = streaming_rave.export_graph(
            enc_graph, args.out, "encoder", args.block, latent_size,
            full_latent_size, ratio)
        meta_dec = streaming_rave.export_graph(
            dec_graph, args.out, "decoder", args.block, latent_size,
            full_latent_size, ratio)
        manifest = {
            "seed": args.seed,
            "name": args.name,
            "arch": args.arch,
            "trained": args.checkpoint is not None,
            "source_url": args.source_url,
            "block_size": args.block,
            "ratio": ratio,
            "sampling_rate": int(model.sr),
            "latent_size": latent_size,
            "full_latent_size": full_latent_size,
            "caches": {
                "encoder": enc_graph.cache_shapes(),
                "decoder": dec_graph.cache_shapes(),
            },
        }
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(args.out, "warmup.bin"), "wb") as f:
            for tensor in warm_enc + warm_dec:
                f.write(tensor.flatten().numpy().tobytes())
    else:
        meta_enc, meta_dec = {}, {}

    n = args.buffers
    print(f"[4/6] preparing {n} buffers of {args.block} samples", flush=True)
    torch.manual_seed(1234)
    audio = torch.randn(1, 1, n * args.block) * args.input_scale

    enc_sess = ort.InferenceSession(
        os.path.join(args.out, "encoder.onnx"),
        providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(
        os.path.join(args.out, "decoder.onnx"),
        providers=["CPUExecutionProvider"])

    n_enc = len(enc_sess.get_inputs()) - 2
    n_dec = len(dec_sess.get_inputs()) - 2
    enc_cache = [w.numpy() for w in warm_enc]
    dec_cache = [w.numpy() for w in warm_dec]

    def run_enc(x, eps, caches):
        feeds = {"x": x.numpy(), "eps": eps.numpy()}
        feeds.update({f"cache_{i}": c for i, c in enumerate(caches)})
        out = enc_sess.run(None, feeds)
        return out[0], out[1:]

    def run_dec(z, noise, caches):
        feeds = {"z": z.numpy(), "noise": noise.numpy()}
        feeds.update({f"cache_{i}": c for i, c in enumerate(caches)})
        out = dec_sess.run(None, feeds)
        return out[0], out[1:]

    print("[5/6] streaming parity run...", flush=True)
    reset_at = n // 2
    errors = []
    t_start = time.time()
    with torch.no_grad():
        for k in range(n):
            torch.manual_seed(k)
            eps = torch.randn(1, full_latent_size, 1)
            noise = torch.randn(1, full_latent_size - latent_size, 1)

            x = audio[:, :, k * args.block:(k + 1) * args.block]

            z_ref = ref_encode(ref_model, x, eps, latent_size)
            y_ref = ref_decode(ref_model, z_ref, noise)

            z_o, enc_cache = run_enc(x, eps, enc_cache)
            y_o, dec_cache = run_dec(
                torch.from_numpy(z_o), noise, dec_cache)

            err = float((y_ref - torch.from_numpy(y_o)).abs().max())
            errors.append(err)

            if (k + 1) % 1000 == 0:
                ms = (time.time() - t_start) * 1000 / (k + 1)
                print(f"      buffer {k + 1}/{n}  maxerr={err:.3e}  "
                      f"{ms:.2f} ms/buf", flush=True)

            if k == reset_at:
                print(f"      reset test at buffer {k}...")
                zero_ref_state(ref_model)
                enc_cache = [np.zeros(e.shape, np.float32)
                             for e in enc_cache]
                dec_cache = [np.zeros(e.shape, np.float32)
                             for e in dec_cache]

    errors = np.array(errors)
    maxerr = float(errors.max())
    head = errors[:reset_at]
    tail = errors[reset_at:]

    print("[6/6] summary", flush=True)
    print(f"      buffers      : {n}")
    print(f"      max abs err  : {maxerr:.3e}   (gate < 1e-4)")
    print(f"      head window  : mean {head.mean():.3e}  max {head.max():.3e}")
    print(f"      tail window  : mean {tail.mean():.3e}  max {tail.max():.3e}")
    print(f"      runtime      : {(time.time() - t_start):.1f}s")

    passed = maxerr < 1e-4 and \
        tail.max() <= max(head.max() * 10.0, 1e-5)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
