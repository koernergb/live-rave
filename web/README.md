# live-rave-web (M5)

RAVE-Live: ORT in a Web Worker running a **SharedArrayBuffer ring + AudioWorklet
realtime path** (mic → EQ → rings → worker → rings → speakers), a **latent
engine** (per-dim bias/scale, noise amount, wet/dry, gain + 3-band EQ), a
**lazy-loaded model picker**, and the offline file-processing pipeline (M2).
Realtime and offline share one cache-threading pipeline (`src/pipeline.ts`),
bit-comparable to `python/parity.py` headless in Node (onnxruntime-node) and in
the browser (onnxruntime-web/WASM).

## Layout

```
src/ring.ts          lock-free SPSC ring constants + read/write (SAB)
src/pipeline.ts      shared cached-graph streaming pipeline (worker + node)
src/manifest.ts      RaveManifest (+ name/arch) types
src/worker.ts        browser Web Worker: model load, parity run, block process
src/realtime.ts      AudioContext + AudioWorklet node + EQ/gain/dry-wet net
src/realtime-worker.ts  Atomics.wait block loop: encode -> edit -> decode
src/models.ts        loadModelBundle (main-thread preload or worker fetch; provider param for bench cells)
src/main.ts          UI: picker, latent sliders, scope, realtime, file mode
public/bench/*       block-parameterized exports for the M5 sweep (gitignored)
bench.html           M5 bench page entry (src/bench/main.ts)
public/worklet/live-processor.js  AudioWorklet: IO-only, crossfade on underrun
scripts/node-parity.ts   headless parity check with onnxruntime-node
scripts/copy-models.mjs syncs benchmarks/{export,models,browser-parity} -> public/
e2e/browser-parity.spec.ts  Playwright gate (chromium + WASM, offline)
e2e/realtime.spec.ts        Playwright gate: 60 s sustained, fake mic
e2e/live-ui.spec.ts         Playwright gate: M4 controls, picker, scope, rt
e2e/bench.spec.ts           Playwright gate/sweep: M5 cells -> ../benchmarks/m5-results.json
public/models/*      per-model encoder.onnx / decoder.onnx / manifest / warmup
                     + models.json catalog (labels, byte sizes, lazy-load urls)
public/parity/*      reference bundle (audio/eps/noise/ref_y) from Python
public/_headers      COOP/COEP for Cloudflare Pages (required for SAB)
```

## Benchmarks (M5, M-series Chromium, 44.1 kHz)

Full methodology + caveats: `../docs/benchmarks.md`. Raw per-cell JSON:
`../benchmarks/m5-results.json`.

| backend | block | model | avg ms | p99 | **RTF** | cold-start |
|---|---|---|---|---|---|---|
| wasm single-thread | 2048 | v2-live | 7.70 | 10.5 | **0.166** | 2.3 s |
| wasm single-thread | 4096 | v2-live | 8.87 | 14.9 | **0.095** | 2.4 s |
| wasm single-thread | 8192 | v2-live | 12.15 | 14.2 | **0.065** | 2.8 s |
| wasm single-thread | 2048 | v2 studio | 30.56 | 46.4 | **0.658** | 9.3 s |
| wasm threaded ×4 | 2048/4096/8192 | v2-live | — | — | **hang** | — |
| webgpu | 2048/4096/8192 | v2-live | — | — | **no adapter** | — |

Verbatim: `benchmarks/bench/*` onnx (2048/4096/8192) built by
`python/export_models.py --bench` (ONNX-vs-eager parity PASS ~4e-8), copied to
`public/bench/`, then `npx playwright test e2e/bench.spec.ts`.
Kill criterion (RTF > 0.8 everywhere) is **not** triggered — single-thread WASM
runs at ≤ 0.17. Threaded WASM **hangs on session load** under COI (all three
cells, recorded, runtime hard-pins `numThreads = 1`); WebGPU has **no hardware
GPU adapter** in this CI host's headless Chromium (software SwiftShader runs but
RTF ≫ 1 — needs a GPU host to measure properly).

## Model catalog (M4)

`python/export_models.py` builds 5 architecture variants from the pinned RAVE
source (random-init weights, reproducible via `(config, seed)`), forces the
latent cut to **16 dims** so the latent sliders act on real dimensions, and
gates each on 64-buffer ONNX-vs-eager parity (< 1e-4). Only `models.json` (the
catalog) and default `manifest.json` are committed — the `.onnx`/`.bin`
artifacts are gitignored (CC-BY-NC-4.0).

| key | arch | cap | role |
|---|---|---|---|
| `v2rt-s0` | v2_rt (capacity 48, no FIR noise path) | **default live** | ~16 ms/block turnaround |
| `v2-s0…s3` | v2 (capacity 96) | studio picker | ~62 ms/block at 44.1k — too heavy for one-block deadlines |

Notes / disclosure:
- Checkpoints are **architecture variants with random-init weights**, exported
  to ONNX for this demo; swap in any trained RAVE export to use real weights.
- `v2_small` is **not** in the picker: its FIR excitation path uses
  `view_as_complex`/`rfft`/`irfft` plus an internal `rand_like` (→ ONNX
  `RandomNormalLike`), none of which are exportable/deterministic in torch 2.2.2.
  `v2_rt.gin` keeps v2_small's lighter capacity without that path.
- Per-dim latent controls expose the first 8 dims (bias + scale × 8).

## Realtime gate

```bash
npm run build
RT_SECONDS=60 npx playwright test e2e/realtime.spec.ts   # fake mic
npx playwright test                                       # full suite (4 spec)
```

M4 result (M-series, chromium headless, 44.1 kHz, default `v2rt-s0`):

| metric | value |
|---|---|
| underruns / 60 s | **0** |
| blocks processed | ~1300 (450 blocks in 20.9 s probe) |
| avg worker turnaround | **16.2 ms** (budget 46.4 ms/block) |
| max turnaround | 33 ms |
| estimated latency | ~234 ms (default slack=4) |

Design constraints that came out of the build:

- **Worklet does zero inference.** Copy in / copy out / notify only (`Atomics.store` + `Atomics.notify`).
- Ring: `BLOCK=2048`, quantum `128`, ring cap = `slack + 4` blocks; slack = the latency/stability slider (2…10). A `LIVE` control flag gates the underrun counter so init/priming is not counted as streaming.
- ORT must run **single-threaded** (threaded WASM deadlocks on load under COI).
- The ~64 MB `.onnx` fetches fail in a worker context under headless Chrome (`net::ERR_CACHE_WRITE_FAILURE`), so models are preloaded on the **main thread** (`cache: "no-store"`) and transferred to the worker as `ArrayBuffer`.
- SharedArrayBuffers cannot be in a postMessage transfer list — they are shared by reference, always clone-posted.

## Reproducing the artifacts

```bash
# from repo root: export the 5-model catalog (ONNX + manifest + warmup + models.json)
.venv/bin/python python/export_models.py

# generate the Python reference bundle the parity tests compare against
.venv/bin/python python/gen_reference.py --buffers 128

# copy onnx + manifest + catalog + reference bundle into web/public/
cd web && npm run models
```

## Checks

```bash
npm run parity:node     # headless pipeline vs Python reference  (maxerr ~3e-8)
npm run parity:web      # build + Playwright: browser WASM parity + file mode
npm run build           # typecheck + vite build (preview port 4175)
```

Reference numbers (M2-M4, seed 0, default v2_rt):

| check | n | max abs err | gate |
|---|---|---|---|
| node (onnxruntime-node) | 128 | 3.27e-08 | < 1e-4 |
| browser (onnxruntime-web WASM) | 128 | < 1e-4 (PASS) | < 1e-4 |

Both are bit-comparable to the Python reference (`python/gen_reference.py`),
which replays the exact M1-gate inputs. `.onnx`/`.bin` artifacts are gitignored
(CC-BY-NC-4.0); regenerate with the commands above.

## Honest "what doesn't work yet"

- **Browsers beyond Chromium on M-series:** everything above is Chromium
  headless. Safari's COI + WebGPU story and Firefox threading are **unverified**;
  phone (iPhone/Android) cold-load is pending a deploy. That is the biggest
  remaining unknown; the first deploy should re-run `e2e/bench.spec.ts` on a
  production URL and on a real device.
- **WebGPU:** no hardware adapter in the CI host's headless Chromium, so there is
  no real WebGPU RTF number yet — only the negative results above.
- **Threaded WASM:** hangs on load under COI in every attempt; therefore
  single-threaded is the shipped configuration and the only measured RTF.
- **Memory:** `performance.memory` is not a standard and is unavailable in this
  worker context, so peak-memory rows in the M5 table are unimplemented rather
  than estimated.
- **Trained weights:** checkpoints are random-init architecture variants; latent
  knobs act on 16 real dims but there is no musical IR here until a trained
  checkpoint is dropped in.
- **v2_small:** not exportable deterministically (see the Model catalog note).

Latency budget (measured, M4, default slack=4, v2-live @ 2048):

| component | ms |
|---|---|
| model block 2048 @ 44.1 kHz | 46.4 |
| ring read-ahead (slack+2) | ~93 |
| worklet quantum | 2.9 |
| output device buffer | ~10–20 |
| worker turnaround (avg) | ~16 |
| **estimated end-to-end** | **~234** |

## Citations

- RAVE: Caillon & Esling, *RAVE: A Variational Autoencoder for Fast and
  High-Quality Neural Audio Synthesis*, arXiv:2111.05011, ISMIR 2021.
- cached_conv / DAFx 2022: Caillon & Esling, *Streaming Sparse Convolution for
  interactive RAVE*, arXiv:2204.07064.
- Prior art: RAVE.js (2021, non-streaming ONNX export, MIT) and the streaming
  (`--streaming`) TorchScript export in
  [acids-ircam/RAVE](https://github.com/acids-ircam/RAVE). Model assets are
  CC-BY-NC-4.0 © Caillon/ACIDS (IR-CAM)/IIL — non-commercial, and there will
  never be a paid tier for these weights. This demo builds on their work; early
  and unambiguous credit: Caillon/ACIDS.

## Deploy

Cloudflare Pages build from `web/` (`npm run build`, preview port 4175). The
`public/_headers` file carries the required COOP/COEP headers; confirm
`window.crossOriginIsolated === true` on the production URL.

```bash
# needs an authenticated wrangler / CLOUDFLARE_API_TOKEN
npx wrangler pages deploy dist --project-name live-rave
```

Deployment is blocked until a Cloudflare token is provided (checked locally:
**not set** as of M4 commit). Until then the site runs locally (`npm run
preview`), fully COI on `crossOriginIsolated`.