# live-rave-web (M2)

ORT in a Web Worker, offline file processing. Same cache-threading pipeline as
`python/parity.py` runs in the browser (onnxruntime-web/WASM) and headless in
Node (onnxruntime-node).

## Layout

```
src/pipeline.ts      shared cached-graph streaming pipeline (worker + node)
src/worker.ts        browser Web Worker: model load, parity run, block process
src/main.ts          UI: file mode (WAV -> blocks -> WAV), parity button
scripts/node-parity.ts   headless parity check with onnxruntime-node
e2e/browser-parity.spec.ts  Playwright gate (chromium + WASM)
public/models/*      encoder.onnx / decoder.onnx / manifest.json / warmup.bin
public/parity/*      reference bundle (audio/eps/noise/ref_y) from Python
public/_headers      COOP/COEP for Cloudflare Pages (SAB later in M3)
```

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
```

Reference numbers (M2, seed 0, v2):

| check | n | max abs err | gate |
|---|---|---|---|
| node (onnxruntime-node) | 128 | 3.93e-08 | < 1e-4 |
| browser (onnxruntime-web WASM) | 128 | 4.47e-08 | < 1e-4 |

Both are bit-comparable to the Python reference (`python/gen_reference.py`),
which replays the exact M1-gate inputs. `.onnx`/`.bin` artifacts are gitignored
(CC-BY-NC-4.0); regenerate with the commands above.