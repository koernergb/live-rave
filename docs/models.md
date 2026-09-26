# Models & vendored dependencies

## Vendored upstreams (pinned, committed as flat source)

| Path | Upstream | Pinned commit |
|---|---|---|
| `python/RAVE` | https://github.com/acids-ircam/RAVE.git | `f048ec4` |
| `python/cached_conv` | https://github.com/acids-ircam/cached_conv.git | `beb8a8a` |

Nested `.git` directories were removed when vendoring. To refresh, re-clone
the upstream at the pinned commit and strip the `.git` dir again.

## Seeded RAVE (parity/export model)

- Config: `rave/configs/v2.gin` (ratio 2048, PQMF 16, sr 44100, latent 128).
- Seed: `attrdict=dict(seed=0)` in `parity.build_rave`.
- Gate result: max abs err 5.96e-08 over 10,000 buffers (see `milestones.md` M1).
- Exported graphs: `benchmarks/export/{encoder,decoder}.onnx` (gitignored, CC-BY-NC-4.0); the `.onnx` files are NOT committed — regenerate with `python/parity.py`. The manifest (`benchmarks/export/manifest.json`) IS committed as the pointer.

## Trained default: IIL Organ Archive

- Source: `Intelligent-Instruments-Lab/rave-models`, `organ_archive_b2048_r48000`.
- License: CC-BY-NC-4.0; trained on organ recordings from archive.org.
- Checkpoint SHA-256: `f97cbb87f51bbbeeafc91594a590e5ad96c0fe47f9a44d6a556977bb8cbe373c`.
- Runtime format: 48 kHz, 2048-sample block, 16 exported latent dimensions.
- Gate: strict checkpoint load plus 128-buffer ONNX-vs-eager streaming parity,
  max absolute error `4.954e-05` at realistic input scale 0.1.
- Browser gate: file processing passes. The trained capacity-96 model misses
  the sustained WASM realtime deadline (380 underruns over 60 seconds), so the
  UI marks it offline-only and retains `v2rt-s0` as the live default.

The checkpoint and generated ONNX files are intentionally gitignored. Download
the checkpoint/config into `benchmarks/checkpoints/organ-archive-b2048/`, then
regenerate with:

```bash
MPLCONFIGDIR=/tmp/live-rave-mpl .venv/bin/python python/parity.py \
  --config ../../benchmarks/checkpoints/organ-archive-b2048/config.gin \
  --checkpoint benchmarks/checkpoints/organ-archive-b2048/last.ckpt \
  --out benchmarks/models/organ-archive-b2048 --block 2048 --buffers 128 \
  --latent-size 16 --input-scale 0.1 \
  --name 'IIL Organ Archive · trained' --arch organ_archive_b2048 \
  --source-url 'https://huggingface.co/Intelligent-Instruments-Lab/rave-models'
```
