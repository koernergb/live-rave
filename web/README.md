# live-rave-web (M3)

ORT in a Web Worker running a **SharedArrayBuffer ring + AudioWorklet realtime
path** (mic → rings → worker → rings → speakers), plus the offline file-processing
pipeline (M2). Same cache-threading pipeline as `python/parity.py` runs in the
browser (onnxruntime-web/WASM) and headless in Node (onnxruntime-node).

## Layout

```
src/ring.ts          lock-free SPSC ring constants + read/write (SAB)
src/pipeline.ts      shared cached-graph streaming pipeline (worker + node)
src/worker.ts        browser Web Worker: model load, parity run, block process
src/realtime.ts      AudioContext + AudioWorklet node + SAB plumbing (main)
src/realtime-worker.ts  Atomics.wait block loop: encode -> decode -> out ring
src/models.ts        loadModelBundle (main-thread preload or worker fetch)
src/main.ts          UI: file mode, parity button, realtime start/stop
public/worklet/live-processor.js  AudioWorklet: IO-only, crossfade on underrun
scripts/node-parity.ts   headless parity check with onnxruntime-node
e2e/browser-parity.spec.ts  Playwright gate (chromium + WASM, offline)
e2e/realtime.spec.ts     Playwright gate: 60 s sustained, run with fake mic
public/models/*      encoder.onnx / decoder.onnx / manifest.json / warmup.bin
public/parity/*      reference bundle (audio/eps/noise/ref_y) from Python
public/_headers      COOP/COEP for Cloudflare Pages (required for SAB)
```

## Realtime gate

```bash
npm run build
RT_SECONDS=60 npx playwright test e2e/realtime.spec.ts   # fake mic
npx playwright test                                       # full suite
```

M3 result (M-series, chromium headless, 44.1 kHz, seeded v2):

| metric | value |
|---|---|
| underruns / 60 s | **0** |
| blocks processed | 1308 |
| avg worker turnaround | 25.5 ms (budget 46.4 ms/block) |
| max turnaround (p99 window) | 78 ms |
| estimated latency | ~151 ms |

Design constraints that came out of the build:

- **Worklet does zero inference.** Copy in / copy out / notify only (`Atomics.store` + `Atomics.notify`).
- Ring: `BLOCK=2048`, quantum `128`, in/out rings of `RING_BLOCKS=8` blocks; a `LIVE` control flag gates the underrun counter so init/priming is not counted as streaming.
- ORT must run **single-threaded** (threaded WASM deadlocks on load under COI).
- The ~60 MB `.onnx` fetches fail in a worker context under headless Chrome (`net::ERR_CACHE_WRITE_FAILURE`), so models are preloaded on the **main thread** (`cache: "no-store"`) and transferred to the worker as `ArrayBuffer`.
- SharedArrayBuffers cannot be in a postMessage transfer list — they are shared by reference, always clone-posted.

## Reproducing the artifacts

```bash
# from repo root: export the ONNX graphs + manifest (M1)
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, 'python')
import parity; parity.main(['--out', 'benchmarks/export', '--buffers', '8'])
PY

# generate the Python reference bundle the parity tests compare against
.venv/bin/python python/gen_reference.py --buffers 128

# copy onnx + manifest + reference bundle into web/public/
web: npm run models
```

## Checks

```bash
npm run parity:node     # headless pipeline vs Python reference  (maxerr ~4e-8)
npm run parity:web      # build + Playwright: browser WASM parity + file mode
npm run build           # typecheck + vite build (preview port 4175)
```

Reference numbers (M2, seed 0, v2):

| check | n | max abs err | gate |
|---|---|---|---|
| node (onnxruntime-node) | 128 | 3.93e-08 | < 1e-4 |
| browser (onnxruntime-web WASM) | 128 | 4.47e-08 | < 1e-4 |

Both are bit-comparable to the Python reference (`python/gen_reference.py`),
which replays the exact M1-gate inputs. `.onnx`/`.bin` artifacts are gitignored
(CC-BY-NC-4.0); regenerate with the commands above.