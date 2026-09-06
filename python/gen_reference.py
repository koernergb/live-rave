"""Generate the browser-parity reference bundle.

Runs the SAME seeded sequence as parity.py against the pristine eager model
(with warmup-seeded state) and dumps the inputs it consumed plus its outputs.
The web pipeline feeds these exact inputs into the exported ONNX graphs and
must reproduce `ref_y` bit-for-bit.

Outputs in benchmarks/browser-parity/:
  manifest.json  copy of the export manifest (cache shapes, block size...)
  audio.bin      float32 mono, N*block samples
  eps.bin        per-block epsilon latent, N * full_latent_size floats
  noise.bin      per-block noise latent, N * (full - latent) floats
  warmup.bin     encoder cache tiles then decoder cache tiles (warmup state)
  ref_y.bin      per-block reconstruct, N * block floats (the ground truth)

Usage: python gen_reference.py [--buffers 128] [--out benchmarks/browser-parity]
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RAVE"))
sys.path.insert(0, os.path.dirname(__file__))

import cached_conv as cc
import gin
import rave

import parity
import streaming_rave as sr

torch = __import__("torch")
F = __import__("torch.nn.functional", fromlist=["F"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(os.path.dirname(__file__), "RAVE"))
    ap.add_argument("--config", default="rave/configs/v2.gin")
    ap.add_argument("--out", default="benchmarks/browser-parity")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--buffers", type=int, default=128)
    ap.add_argument("--fidelity", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cc.use_cached_conv(True)
    torch.set_grad_enabled(False)
    gin.clear_config()
    gin.parse_config_file(os.path.join(args.run_dir, args.config))

    model = parity.build_rave(args.run_dir, args.config, args.seed)
    latent_size = parity.compute_latent_size(model, args.fidelity)
    full_latent = model.latent_size

    parity.warmup(model, args.block)
    ref = __import__("copy").deepcopy(model)

    # Patch a copy so we can read the warmup state through the graph registry.
    probe = __import__("copy").deepcopy(model)
    enc_graph = sr.CachedGraph(probe, encoder=True, latent_size=latent_size)
    dec_graph = sr.CachedGraph(probe, encoder=False, latent_size=latent_size)
    warm_enc = enc_graph.warmup_caches()
    warm_dec = dec_graph.warmup_caches()

    os.makedirs(args.out, exist_ok=True)

    manifest = json.load(
        open(os.path.join("benchmarks/export/manifest.json"))
    ) if os.path.exists(os.path.join("benchmarks/export/manifest.json")) else {
        "seed": args.seed, "block_size": args.block, "ratio": args.block,
        "sampling_rate": int(model.sr), "latent_size": latent_size,
        "full_latent_size": full_latent,
        "caches": {
            "encoder": enc_graph.cache_shapes(),
            "decoder": dec_graph.cache_shapes(),
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.out, "warmup.bin"), "wb") as f:
        for t in warm_enc + warm_dec:
            f.write(t.flatten().numpy().tobytes())

    n = args.buffers
    torch.manual_seed(1234)
    audio = torch.randn(1, 1, n * args.block)

    with open(os.path.join(args.out, "audio.bin"), "wb") as f:
        f.write(audio.flatten().numpy().tobytes())
    with open(os.path.join(args.out, "eps.bin"), "wb") as f:
        pass
    with open(os.path.join(args.out, "noise.bin"), "wb") as f:
        pass
    with open(os.path.join(args.out, "ref_y.bin"), "wb") as f:
        with torch.no_grad():
            for k in range(n):
                torch.manual_seed(k)
                eps = torch.randn(1, full_latent, 1)
                noise = torch.randn(1, full_latent - latent_size, 1)
                x = audio[:, :, k * args.block:(k + 1) * args.block]

                z = parity.ref_encode(ref, x, eps, latent_size)
                y = parity.ref_decode(ref, z, noise)

                eps_s = eps[:, :, :1].flatten().cpu().numpy()
                noise_s = noise[:, :, :1].flatten().cpu().numpy()
                y_s = y.flatten().cpu().numpy()
                f.write(y_s.tobytes())

                with open(os.path.join(args.out, "eps.bin"), "ab") as fe:
                    fe.write(eps_s.tobytes())
                with open(os.path.join(args.out, "noise.bin"), "ab") as fn:
                    fn.write(noise_s.tobytes())

    print(f"reference bundle -> {args.out} ({n} x {args.block})")


if __name__ == "__main__":
    main()