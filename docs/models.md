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