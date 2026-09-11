# RAVE-Live — M5 benchmark sweep

Recorded numbers, the command that produced them, and the honest caveats.
Rules from `milestones.md` §0: every number ships with its producing command;
no screenshots; underruns and errors are metrics.

## Cell definition

Each cell = one `(backend, block-size, model)` tuple, streaming the **shipping
runtime pipeline** (`web/src/pipeline.ts`, same code as the 24/7 demo — nothing
benchmark-special):

- Deterministic stream: mulberry32-seeded audio (fixed at `0xabcd0000 ^ block`),
  zero `eps` latent, seeded residual-noise input (`0xbeef ^ block`) — identical
  stream across every cell.
- `3` warmup blocks, then `100` measured blocks; per-block `performance.now()`
  turnaround → avg / p50 / p95 / p99 / max. RTF = avg / (block/44.1 kHz).
  Cold-start = script start → first block processed.
- Window per block: 2048 → 46.4 ms, 4096 → 92.9 ms, 8192 → 185.8 ms.
- Models: `v2_rt` (v2-live, capacity 48 — the shipping default) at all three
  block sizes; `v2` studio (capacity 96) at 2048 as a contrast cell.

Produced by:

```
python/export_models.py --bench            # exports default variant @ each block, ONNX-vs-eager parity gate PASS (~4e-8)
cp benchmarks/bench/<block>/* web/public/bench/<block>/
npx vite build                             # bench.html + bench/main.ts (COI middlewares), pinned in vite.config.ts
npx playwright test e2e/bench.spec.ts      # 10 cells -> benchmarks/m5-results.json
```

Host: M-series Mac, Chromium headless, 44.1 kHz, `crossOriginIsolated == true`
(COOP/COEP enforced by the Vite middleware + `web/public/_headers`).

## Results (2026-09-10)

Cell -> `benchmarks/m5-results.json` (committed, regenerable, deterministic).

| backend | block | model | avg ms | p50 | p95 | p99 | max | **RTF** | cold-start |
|---|---|---|---|---|---|---|---|---|---|
| wasm (SIMD, single-thread) | 2048 | v2-live | 7.70 | 7.55 | 9.34 | 10.51 | 10.51 | **0.166** | 2.34 s |
| wasm (SIMD, single-thread) | 4096 | v2-live | 8.87 | 8.72 | 9.45 | 14.93 | 14.93 | **0.095** | 2.37 s |
| wasm (SIMD, single-thread) | 8192 | v2-live | 12.15 | 12.08 | 13.17 | 14.17 | 14.17 | **0.065** | 2.84 s |
| wasm (SIMD, single-thread) | 2048 | v2 studio (cap 96) | 30.56 | 29.32 | 40.95 | 46.38 | 46.38 | **0.658** | 9.30 s |
| wasm-threaded (4 threads) | 2048/4096/8192 | v2-live | — | — | — | — | — | — | **hangs on session load (all 3 cells)** |
| webgpu | 2048/4096/8192 | v2-live | — | — | — | — | — | — | **no adapter (see below)** |

## Reading the table

- **Positive result:** single-threaded WASM keeps RTF well under 1 across all
  workable block sizes; the kill criterion (RTF > 0.8 everywhere) does **not**
  trigger. Larger blocks are faster per second of audio (amortized encode/decode
  + fewer cache re-threads), but cost latency per block; the shipped runtime
  balances this at 2048 with ~15 ms+ headroom for transport/device jitter.
- **The studio contrast cell is the honest why for the default.** Bench RTF 0.658
  (engine alone) still fits in 1 budget, but the live loop adds transport +
  device-buffer jitter on top (M4 measured 61.6 ms/block avg under the worklet →
  5136 underruns/60 s). The bench is the engine; the realtime gate in
  `web/e2e/realtime.spec.ts` is the loop. We default to v2-live not because the
  studio "can't bench" but because it can't *stream* on this class of device.
- **wasm-threaded = recorded hang.** All three threaded cells deadlock during
  `InferenceSession.create` (150 s timeout), reproducing the M2/M3 finding
  (threaded jsep + COI). It is a genuine, recorded limitation — that is why the
  runtime hard-pins `numThreads = 1`.
- **webgpu = no hardware adapter in CI host's headless Chromium.** Manually
  launched with `--enable-unsafe-webgpu --use-angle=swiftshader`, the software
  SwiftShader adapter *did* create a session and start processing, but did not
  finish 20 blocks in 60 s (software RTF ≫ 1) — both findings recorded. A real
  WebGPU latency number needs a hardware GPU host; none available at write time.

## Not-yet-measured (honest gaps)

- **Peak memory:** `performance.memory` is not a WebStandard and in this Chromium
  build it is disabled/absent in the worker context; `usedJSHeapSize` deltas were
  0. Reported as unimplemented rather than noise. Nightly/exposed `--enable-precise-memory-info`
  or a native run is required for real numbers.
- **Browsers beyond Chrome**: Safari/WebKit COI + WebGPU status, Firefox
  threading — not exercised. All claims are M-series Chromium. Safari is the
  stated next unknown (see `web/README.md`).
- **Phones / x86 laptop**: cold-load and RTF on-device pending a deploy.

## Reproducing

```
.venv/bin/python python/export_models.py --bench        # re-export + parity gate (no network)
cp benchmarks/bench/{2048,4096,8192}/* web/public/bench/<block>/
cd web && npx vite build && npx playwright test e2e/bench.spec.ts
```

Assumes the M4 preconditions: a running Vite preview on :4175 with COI headers,
`benchmarks/models/v2-s0/{encoder,decoder,warmup}.onnx/.bin` (gitignored, from
`export_models.py`) present for the studio contrast cell.