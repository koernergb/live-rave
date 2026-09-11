"""M4 model catalog: export a picker-ready set of RAVE variants.

Every variant is built from the pinned RAVE source (python/RAVE @ f048ec4)
with random-init weights and a fixed latent cut of LATENT_SIZE=16, so the
per-dimension latent controls have real dimensions to act on. Weights are
reproducible via (config, seed); artifacts are CC-BY-NC-4.0 and gitignored —
only the models.json catalog is committed.

Layout:
  benchmarks/export/            v2-s0  (the default: also feeds M2/M3 gates)
  benchmarks/models/<key>/      the other picker variants
  benchmarks/models/models.json catalog (labels, byte sizes, lazy-load urls)

Gate: ONNX-vs-eager parity over N buffers, max abs err < 1e-4 (reuses the M1
harness exactly).
"""

import argparse
import json
import os
import sys
import time

import torch

import parity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RAVE"))
sys.path.insert(0, os.path.dirname(__file__))

import streaming_rave as sr

LATENT_SIZE = 16  # explicit latent cut for the picker variants
N_BUFFERS = 64
BLOCK = 2048
RESET_AT = 32

VARIANTS = [
    # Realtime-lean v2 (capacity 48, no FIR noise path) — the live default.
    {"key": "v2rt-s0", "config": "rave/configs/v2_rt.gin", "seed": 0,
     "label": "v2-live · seed 0", "arch": "v2_rt", "default": True},
    {"key": "v2-s0", "config": "rave/configs/v2.gin", "seed": 0,
     "label": "v2 · seed 0", "arch": "v2"},
    {"key": "v2-s1", "config": "rave/configs/v2.gin", "seed": 1,
     "label": "v2 · seed 1", "arch": "v2"},
    {"key": "v2-s2", "config": "rave/configs/v2.gin", "seed": 2,
     "label": "v2 · seed 2", "arch": "v2"},
    {"key": "v2-s3", "config": "rave/configs/v2.gin", "seed": 3,
     "label": "v2 · seed 3", "arch": "v2"},
]


def out_dir(args, key, default=False):
    if default:
        return args.export_out
    return os.path.join(args.models_out, key)


def makedirs(p):
    os.makedirs(p, exist_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(os.path.dirname(__file__), "RAVE"))
    ap.add_argument("--export-out", default="benchmarks/export")
    ap.add_argument("--models-out", default="benchmarks/models")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    catalog = []
    for v in VARIANTS:
        key = v["key"]
        prefix = v["label"]
        config = os.path.join(args.run_dir, v["config"])
        out = out_dir(args, key, v.get("default", False))
        makedirs(out)

        torch.set_grad_enabled(False)
        print(f"\n[{key}] building {v['config']} seed={v['seed']}...", flush=True)
        model = parity.build_rave(args.run_dir, v["config"], v["seed"])
        full = model.latent_size
        ls = LATENT_SIZE
        print(f"      full_latent={full} latent_cut={ls}", flush=True)

        parity.warmup(model, BLOCK)
        ref_model = __import__("copy").deepcopy(model)
        with torch.no_grad():
            probe = torch.zeros(1, 1, BLOCK)
            ratio = BLOCK // model.encoder(model.pqmf(probe)).shape[-1]
            T = BLOCK // ratio
        print(f"      ratio={ratio} latent_steps/block={T}", flush=True)

        enc_graph = sr.CachedGraph(model, encoder=True, latent_size=ls)
        dec_graph = sr.CachedGraph(model, encoder=False, latent_size=ls)
        warm_enc = enc_graph.warmup_caches()
        warm_dec = dec_graph.warmup_caches()

        if not args.no_export:
            print(f"      exporting ONNX graphs -> {out}", flush=True)
            sr.export_graph(enc_graph, out, "encoder", BLOCK, ls, full, ratio)
            sr.export_graph(dec_graph, out, "decoder", BLOCK, ls, full, ratio)
            manifest = {
                "seed": v["seed"],
                "name": v["label"],
                "arch": v["arch"],
                "block_size": BLOCK,
                "ratio": ratio,
                "sampling_rate": int(model.sr),
                "latent_size": ls,
                "full_latent_size": full,
                "caches": {
                    "encoder": enc_graph.cache_shapes(),
                    "decoder": dec_graph.cache_shapes(),
                },
            }
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            with open(os.path.join(out, "warmup.bin"), "wb") as f:
                for t in warm_enc + warm_dec:
                    f.write(t.flatten().numpy().tobytes())

        enc_path = os.path.join(out, "encoder.onnx")
        dec_path = os.path.join(out, "decoder.onnx")
        enc_sess = __import__("onnxruntime", fromlist=["InferenceSession"]).InferenceSession(
            enc_path, providers=["CPUExecutionProvider"])
        dec_sess = __import__("onnxruntime", fromlist=["InferenceSession"]).InferenceSession(
            dec_path, providers=["CPUExecutionProvider"])

        enc_cache = [w.numpy() for w in warm_enc]
        dec_cache = [w.numpy() for w in warm_dec]

        print(f"      parity {N_BUFFERS} buffers...", flush=True)
        torch.manual_seed(1234)
        audio = torch.randn(1, 1, N_BUFFERS * BLOCK)
        errors = []
        with torch.no_grad():
            for k in range(N_BUFFERS):
                torch.manual_seed(k)
                eps = torch.randn(1, full, T)
                noise = torch.randn(1, full - ls, T)
                x = audio[:, :, k * BLOCK:(k + 1) * BLOCK]
                z_ref = parity.ref_encode(ref_model, x, eps, ls)
                y_ref = parity.ref_decode(ref_model, z_ref, noise)

                feeds = {"x": x.numpy(), "eps": eps.numpy()}
                feeds.update({f"cache_{i}": c for i, c in enumerate(enc_cache)})
                z_o, *enc_cache = enc_sess.run(None, feeds)
                dec_feeds = {"z": z_o, "noise": noise.numpy()}
                dec_feeds.update({f"cache_{i}": c for i, c in enumerate(dec_cache)})
                y_o, *dec_cache = dec_sess.run(None, dec_feeds)

                err = float((y_ref - torch.from_numpy(y_o)).abs().max())
                errors.append(err)
                if k + 1 == RESET_AT:
                    print(f"      reset test at buffer {k + 1}...", flush=True)
                    parity.zero_ref_state(ref_model)
                    enc_cache = [n * 0 for n in enc_cache]
                    dec_cache = [n * 0 for n in dec_cache]

        maxerr = max(errors)
        head = max(errors[:RESET_AT])
        tail = max(errors[RESET_AT:])
        ok = maxerr < 1e-4 and tail <= max(head * 10.0, 1e-5)
        print(f"      maxerr={maxerr:.3e} head={head:.3e} tail={tail:.3e} "
              f"{'PASS' if ok else 'FAIL'}")

        enc_bytes = os.path.getsize(enc_path)
        dec_bytes = os.path.getsize(dec_path)
        warm_bytes = os.path.getsize(os.path.join(out, "warmup.bin"))
        rel = "" if v.get("default", False) else key + "/"
        catalog.append({
            "key": key,
            "label": v["label"],
            "arch": v["arch"],
            "seed": v["seed"],
            "block_size": BLOCK,
            "ratio": ratio,
            "latent_steps": T,
            "latent_size": ls,
            "full_latent_size": full,
            "bytes": {"encoder": enc_bytes, "decoder": dec_bytes,
                      "warmup": warm_bytes},
            "urls": {
                "encoder": f"models/{rel}encoder.onnx",
                "decoder": f"models/{rel}decoder.onnx",
                "manifest": f"models/{rel}manifest.json",
                "warmup": f"models/{rel}warmup.bin",
            },
        })
        if not ok:
            raise SystemExit(f"[{key}] parity FAILED — aborting catalog")

    idx = os.path.join(args.models_out, "models.json")
    makedirs(args.models_out)
    with open(idx, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"\ncatalog -> {idx} ({len(catalog)} models, "
          f"{(time.time() - t0):.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())