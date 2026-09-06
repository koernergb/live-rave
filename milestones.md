# RAVE-Live — Milestones

Detailed implementation plan derived from `rave-live-build-brief.md`. This document is the working guide: every milestone has concrete tasks, a testable gate, dependencies, and a done-checklist.

**Source of truth:** `rave-live-build-brief.md`. Where this file conflicts with the brief, the brief wins.

---

## 0. Working conventions

These apply across every milestone.

- **Repo layout** (top level of this repo):
  ```
  python/        # export + parity harness (M1)
  web/           # worker, worklet, UI (M2–M4)
  benchmarks/    # harness + results (M5)
  dist/          # built site for deploy (M4+)
  docs/          # README, this file, benchmark tables
  ```
- **Commits:** one commit per completed gate, message format `M{n}: <summary>`.
- **Every numeric claim ships with the command that produced it** — no screenshot-only numbers.
- **Do not commit checkpoints or generated models** — they are CC-BY-NC-4.0 and large. Commit URLs/manifest pointers only. Add `.gitignore` entries in M1.
- **Never silence an underrun or an error to make a demo look better.** They are metrics.

---

## 1. Dependency graph and schedule

```
M0 (0.5 d) ─┐
M1 (4–6 d) ── ──┬──── M2 (2 d) ── M3 (3 d) ── M4 (3 d)
                │                                    │
                └────────────────── M5 (2–3 d) ──────┘   M5 needs deployable build
M6 (2 d) ─────────────────── after M3 passes, parallel-isolated
```

- Total: **~17–20 focused days**.
- **M1 is the risk.** If it hasn't converged in 8 days, execute the fallback ladder (§4.4) and move on.
- **Start the benchmark harness (M5 scaffolding) during M2**, not at M5 — instrumenting late invites rework.

---

## 2. M0 — Prior-art check (0.5 d)

### Purpose
Confirms the claimed gap (streaming client-side RAVE) actually exists before writing code.

### Tasks
1. [ ] Search **Web Audio Conference** proceedings (webaudioconf.com), 2018–present, for streaming neural audio / RAVE / ONNX worklets.
2. [ ] Search **NIME** and **DAFx** proceedings for "browser", "WebAssembly", "web audio" + neural.
3. [ ] Audit **github.com/ircam-ismm** and **github.com/acids-ircam** full repo lists, including unreleased-looking repos.
4. [ ] Check **Neutone** (`github.com/Neutone`) web presence and any prototypes.
5. [ ] Search **HuggingFace Spaces** filtered for `static` SDK (client-side by definition).
6. [ ] Search GitHub for `onnxruntime-web cached_conv`, `rave streaming wasm`, `stateful onnx broowser`.
7. [ ] Record every hit in `docs/prior-art.md`: URL, license, realtime?, maintained?, streaming or file-in/file-out?

### Gate
A written `docs/prior-art.md` with a verdict paragraph, plus the answer to: *is there a maintained, realtime, streaming browser RAVE?*

### Branch
- **No** → proceed to M1.
- **Yes** → read it, decide whether the streaming-export contribution is still ours. If a maintained realtime streaming implementation exists, stop and pivot per brief §2.

---

## 3. M1 — Cache-hoisted ONNX export, encoder + decoder (4–6 d, hard cap 8 d)

The core risk. 60% of project risk lives here. **Nothing else matters until the parity gate passes.**

### 3.1 Setup
1. [ ] Python env: `torch` (version matching the RAVE repo's requirements), `onnx`, `onnxruntime` (CPU, for the parity reference), `numpy`, `matplotlib` (drift plots), `jupyter` optional.
2. [ ] Clone `github.com/acids-ircam/RAVE` and pin a commit. Record it in `docs/models.md`.
3. [ ] Confirm the streaming (`torchscript --streaming`) export works out of the box — this is the **reference**, its outputs are ground truth.

### 3.2 Model choice
1. [ ] Attempt **v2 first** (per brief §4). Verify:
   - [ ] **PQMF** analysis/synthesis pair round-trips (emit sine sweep, confirm unity-ish reconstruction, no phase polarity flip).
   - [ ] **Noise synthesizer branch** — confirm `RandomNormalLike` never appears in the exported graph.
2. [ ] If v2 export resists: `v2_small` → `raspberry` → `v1-noiseless`. Record which landing you reached in `docs/models.md` — the README must say so, honestly.

### 3.3 Export implementation
1. [ ] Write `python/streaming_rave.py` defining `StreamingRAVE(nn.Module)`:
   - `forward(self, x, *caches) -> (y, *new_caches)`.
   - Walk `model.named_modules()`, collect every `CachedConv1d` / `CachedPadding1d` into an **ordered** list keyed by **module path string, sorted**.
   - Freeze that list into a **manifest JSON** shipped alongside each exported model. Ordering must be deterministic and identical at export and runtime.
2. [ ] Monkeypatch each cached module's `forward` to read its cache from a shared input list indexed by position and write the updated cache to the parallel output list — never touching `self.cache`.
3. [ ] Export **encoder and decoder as separate graphs**. Expose the latent as graph output (encoder) and graph input (decoder).
4. [ ] **Fix the time/block axis static at export.** Batch may be dynamic; time must not be (kernel specialization in WASM EP).
5. [ ] **Noise is an explicit graph input**, generated in JS at runtime and in numpy for parity. Never rely on `RandomNormalLike`.
6. [ ] Validate with `onnx.checker`, then run `ort InferenceSession` on a couple of buffers and compare shape/roughness against TorchScript before attempting the full parity test.

### 3.4 Parity test — the gate (build this harness first, reuse it everywhere)
`python/parity.py`, inputs: TorchScript streaming export, ONNX encoder+decoder graphs, manifest, block size, sequence length.

1. [ ] Feed the **same 10,000-buffer sequence** through (a) TorchScript `--streaming` reference and (b) ONNX graphs with manual cache threading.
2. [ ] Compute per-buffer `max abs error`. **Pass: < 1e-4 with no monotonic drift.**
   - Plot per-buffer error vs buffer index in `docs/parity/`. Flat noise floor = caches threaded correctly. Upward ramp = a dropped cache; it will sound like slow-onset garbage.
3. [ ] **Reset test:** zero all caches mid-stream → output must equal a fresh start exactly.
4. [ ] Noise-input parity: same seeded noise tensor into both paths.
5. [ ] Pin the session options (intra-op threads, depending on CPU parity environment) so parity numbers are reproducible.

### 3.5 Gate
Checkbox state in `docs/parity/`:
- [x] Parity run committed (flat floor, < 1e-4, 10k buffers). — `python/parity.py --buffers 10000`: max abs err **5.96e-08**, flat floor, no drift.
- [x] Reset test passes. — zero-init mid-stream (buffer 5000) → fresh-start equivalence.
- [x] Seed-reproducible noise path passes.
- [x] Encoder and decoder exported as separate graphs; manifest JSON committed (`benchmarks/export/manifest.json`).
- [x] Model picker candidates exported: v2 (seed 0, ratio 2048, PQMF 16, latent 1). Fallbacks untouched — v2 green, no ladder needed.

**Do not start M2 until this passes.**

### Risks / fallbacks
- Quiet cache-baking bug (brief §4) — assume it until disproven numerically; the parity test *is* the disproof.
- v2 resists → fallback ladder; the parity harness is the reusable asset either way.
- If **every** config fails parity inside 8 d → kill criteria (§8 of brief): ship an offline modern-ORT rebuild of RAVE.js instead and stop.

---

## 4. M2 — ORT in a Worker, offline file processing (2 d)

Proves the JS side can reproduce Python parity **before** any realtime plumbing exists.

### Tasks
1. [ ] Scaffold `web/` as a Vite (or equivalent) app; **decision: Cloudflare Pages host**, so `public/_headers` with COOP/COEP is in the repo from day one:
   ```
   /*
     Cross-Origin-Opener-Policy: same-origin
     Cross-Origin-Embedder-Policy: require-corp
   ```
2. [ ] Add `onnxruntime-web` (WASM SIMD + threads or simd fallback), preload model files as `.url`/`.weights` pairs where applicable.
3. [ ] Instantiate `InferenceSession` inside a **Web Worker** (never main thread).
4. [ ] Write `web/src/pipeline.ts`: load manifest → allocate a cache array per graph → per buffer: run encoder, hold latent, run decoder with new caches, return samples. Same shapes/block sizes as M1.
5. [ ] **Offline file mode:** input a `.wav`, stream it through the pipeline in `BLOCK`-sized chunks, write output `.wav`.
6. [ ] Extension of the parity harness to the browser: run the same 10k-buffer sequence in the Worker, compare against M1 reference.

### Gate
- [x] Web build output is **bit-comparable** to M1 Python output for the same buffered sequence (reuse `docs/parity/` plots; a numeric compare script in `web/`).
  - `web/scripts/node-parity.ts` (onnxruntime-node): 128 blocks, max abs err 3.93e-08.
  - `web/e2e/browser-parity.spec.ts` (chromium, onnxruntime-web WASM): 128 blocks, max abs err 4.47e-08. Both vs the reference bundle from `python/gen_reference.py`.
- [x] File mode works end-to-end in the browser. — WAV in → block pipeline → WAV out (Playwright download assert); PCM16 round-trip unit-checked.

### Notes
- The SharedArrayBuffer + threads runtime is not yet required here; enable header-gated features only when M3 needs them, so any COOP/COEP breakage is caught early (headers are already set). **Test with `SharedArrayBuffer` present from the start** to avoid surprise Safari failures.
- Record cold-start time to first audio now; it becomes an M5 metric.

---

## 5. M3 — SAB ring + AudioWorklet, mic → speakers (3 d)

The realtime path. **Hard rule: the AudioWorklet does zero inference** — copy in, copy out, signal.

### 5.1 Ring buffer contract
1. [x] Two `SharedArrayBuffer`s (input, output), each a **lock-free SPSC ring of `Float32Array`** (`web/src/ring.ts`).
2. [x] One `Int32Array` control block: read index, write index, underrun counter.
3. [x] Output ring capacity = **2 × BLOCK minimum**. Expose extra slack as the latency/stability slider (see M4).
   - Currently 8 × BLOCK slack (~371 ms headroom) to absorb WASM turnaround spikes; latency estimate shown in UI.

### 5.2 Worklet side (`web/public/worklet/live-processor.js`)
1. [x] On each 128-frame quantum: write input frames → update write index → `Atomics.store` → `Atomics.notify` the worker.
2. [x] Read output frames for the quantum; if short, **emit a short crossfade to silence** (never a hard cut) and increment underrun counter.
3. [x] Output ring sizing math lives in one place; the worklet knows only the ring geometry, never model sizes.
   - Underruns only counted while `LIVE` (output connected) — init/priming is not part of the streaming metric.

### 5.3 Worker pipeline loop
1. [x] `Atomics.wait` until ≥ `BLOCK` frames available → drain a block → encode → (no-op manipulate yet) → decode → write output ring (`web/src/realtime-worker.ts`).
2. [x] Measured loop-back: split device input 🡒 ring ⟶ worker ⟶ ring 🡒 output, let it run.

### 5.4 Mic capture
1. [x] `getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } })` — leaving these on destroys the signal.
2. [x] Input smoothing/gain stage before the ring.

### 5.5 Gate
- [x] 60 s continuous mic → speakers with **zero underruns on M-series**.
  - `web/e2e/realtime.spec.ts` (chromium headless, fake mic device, `RT_SECONDS=60`): **PASS, underruns 0**, 1308 blocks, avg turnaround 25.5 ms (budget 46.4 ms), max 78 ms.
- [x] Underrun counter visible, not hidden.
- [x] Latency measured (add `performance.now()` taints for worklet→output path) and recorded against the budget table. — est latency ~151 ms (ring read-ahead + avg turnaround + device buffer).

### Latency budget (from brief §5) — verify, don't just repeat
| Component | Frames @ 48 kHz | ms |
|---|---|---|
| Model block (v2, ratio 2048 / 44.1 kHz) | 2048 | 46.4 |
| Ring read-ahead (2 blocks held) | 4096 | 92.9 |
| Worklet quantum | 128 | 2.9 |
| Output device buffer | — | ~10–20 |
| Worker turnaround (avg) | — | ~25.5 |
| **Total (measured estimate)** | | **~151** |

Working notes (M3): ORT must stay single-threaded under COI (threaded WASM deadlocks on load); the 60 MB `.onnx` fetches hit `net::ERR_CACHE_WRITE_FAILURE` from a worker context under headless Chrome, so models are preloaded on the main thread and transferred as `ArrayBuffer`.

---

## 6. M4 — Latent UI, model picker, wet/dry, latent scope + deploy (3 d)

### 6.1 Interaction (from brief §6)
1. [ ] **Model picker** — 4–6 checkpoints, lazy-loaded, byte size shown.
2. [ ] **Per-dimension latent bias/scale** — first 8 dims, sliders.
3. [ ] **Noise amount** → into the explicit noise input.
4. [ ] **Wet/dry** — always available; users need the A/B.
5. [ ] **Input gain + 3-band EQ** — the practitioner's real control surface (RAVE is steered with gain/filtering more than latent edits).
6. [ ] **Latency/stability slider** → ring slack.
7. [ ] **Live latent scope** — small canvas plotting latent dims over time. This is what makes the demo read as research. Costs ~an afternoon.
8. [ ] **File mode** kept alongside mic mode (people try it first; works without mic permission on phones).

### 6.2 Deploy
1. [ ] Cloudflare Pages build from `web/`, `_headers` in effect (verify via `window.crossOriginIsolated === true` and a live check on a production URL).
2. [ ] Sizes audited: models lazy-loaded, page shell small enough for **cold load on a phone**.
3. [ ] License/footer block present: CC-BY-NC-4.0, non-commercial, credit Caillon/ACIDS/IIL.

### 6.3 Gate
- [ ] Deployed URL is cross-origin isolated.
- [ ] Cold phone load works and produces audio in file mode without mic permission.
- [ ] Every control in §6.1 is live and audibly demonstrable.

---

## 7. M5 — Benchmark sweep + README with numbers (2–3 d)

The publishable core. Nobody has these numbers.

### 7.1 Harness (scaffold during M2)
1. [ ] `benchmarks/` runner parameterized over:
   - **Backend:** WASM SIMD, SIMD+threads (2/4/8), WebGPU.
   - **Block size:** 512, 1024, 2048, 4096.
   - **Model:** v1-noiseless, v2, v2_small, raspberry (whichever were exported).
   - **Browser:** Chrome, Safari, Firefox.
   - **Device:** M-series Mac, x86 laptop, Android phone, iPhone.
2. [ ] Metrics: **RTF median + p99** (tail = dropouts), **underrun rate/min**, **cold-start to first audio**, **peak memory**, **numeric parity vs TorchScript**.
3. [ ] Deterministic input (same seeded noise + sweep tone) across all cells.

### 7.2 Outputs
1. [ ] Benchmark table committed: `docs/benchmarks.md`.
2. [ ] RTF-vs-block-size curve per backend (p99/median gap across browsers is likely the headline result).
3. [ ] Backend verdict written up: expected WASM SIMD+threads > WebGPU; if WebGPU only wins at unusable latencies, that curve **is** the result.

### 7.3 README
1. [ ] Benchmark table.
2. [ ] Latency budget table.
3. [ ] Honest "what doesn't work yet".
4. [ ] License prominent (CC-BY-NC, non-commercial, no pro tier ever).
5. [ ] Citations: RAVE (arXiv:2111.05011), cached_conv/DAFx 2022 (arXiv:2204.07064), RAVE.js as prior art. Credit Caillon/ACIDS early and unambiguously.

### Gate
- [ ] Table + curves committed; `README` reflects recorded numbers.
- [ ] If **RTF > 0.8 at every workable block size on M-series** → publish as a clean negative result and stop (brief §9 kill criteria).

---

## 8. M6 — Factor the harness into a standalone module (2 d, optional-for-ship but the one that outlives the demo)

### Purpose
"Stateful model running under a real-time deadline in a browser, with cache tensors threaded across invocations" is not audio-specific. Extract it.

### Tasks
1. [ ] Refactor `web/src/pipeline.ts` → `stateful-ort-stream`:
   - thread N cache tensors across invocations (manifest-driven),
   - SPSC ring + control block,
   - worklet shim (I/O only),
   - underrun accounting.
2. [ ] Audio-specific bits (EQ, latent sliders, worklet wiring) stay behind an interface the generic module doesn't know about.
3. [ ] **Second (non-audio) toy model runs through it unchanged** — e.g. a small streaming stateful MLP/RNN state machine. That a non-audio model runs unmodified is the acceptance test.
4. [ ] Publish as a separate small repo (MIT; checkpoint license for any shipped RAVE assets stated separately).

### Gate
- [ ] Non-audio model runs through the module without edits to the module.

---

## 9. Project kill criteria (pre-agreed, honored)

From brief §9 — restated so they're visible in this doc:
1. Prior art exists, is maintained, and is realtime → stop, 300-word note, spend days elsewhere.
2. M1 parity fails on every config after 8 days → ship offline modern-ORT RAVE.js rebuild, stop.
3. RTF > 0.8 at every workable block size on M-series → publish negative-result benchmark table, stop.

---

## 10. Deliverables checklist (mirrors brief §11)

Copy into the final PR/close-out:
- [ ] Repo `rave-live`, MIT for code, checkpoint license called out separately.
- [ ] Live demo on Cloudflare Pages, cross-origin isolated, loads on mobile.
- [ ] README: benchmark table, latency budget, honest "what doesn't work yet".
- [ ] 60-second demo video — voice in, four models cycled, latent sliders moving, latent scope visible. **Open with the sound.**
- [ ] `stateful-ort-stream` separate repo.
- [ ] *Optional:* Web Audio Conference demo submission (paper ≈ README + related-work section).

---

## 11. Current status

Update as milestones land. Goal: every line in this section shows `done` with a date + commit.

- [x] **M0** Prior-art check — *done 2026-09-04* (docs/prior-art.md, commit 4f88f22)
- [x] **M1** Cache-hoisted ONNX export + parity — *done 2026-09-05* (encoder 80 caches / decoder 76, pqmf state threaded; gate 5.96e-08 @ 10k buffers, commit f2924f0)
- [x] **M2** ORT in Worker, offline processing — *done 2026-09-05* (shared pipeline web/node; browser parity 4.47e-08, file mode E2E)
- [x] **M3** SAB ring + AudioWorklet, mic → speakers — *done 2026-09-05* (lock-free rings, worklet IO-only, `Atomics.wait` worker loop; 60 s gate PASS, 0 underruns, est latency ~151 ms)
- [ ] **M4** Latent UI + deploy — *pending*
- [ ] **M5** Benchmarks + README — *pending*
- [ ] **M6** `stateful-ort-stream` standalone — *pending*