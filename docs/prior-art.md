# M0 — Prior-art check

Date: 2026-09-04. This is the written result of the M0 gate (milestones.md §2).

## Verdict

**No maintained, realtime, streaming RAVE-in-browser exists.** The claimed gap is real. The only browser RAVE artifact (RAVE.js) is offline file-in/file-out, built on the deprecated onnx.js runtime, and has received no commits since 2022-06-28. Proceed to M1.

## Checks performed

### 1. RAVE.js — caillonantoine/ravejs
- **Repo:** 20 stars, 5 forks, 1 open issue. Lifespan: created 2022-06-17, last push 2022-06-28 — a single 11-day burst, abandoned since. Primary language JavaScript (60.6%).
- **What it does:** interactive demo linked from the NIME workshop site. File-in/file-out rendering against an ONNX model. Built on **onnx.js** (the deprecated predecessor of onnxruntime-web; onnx.js was archived in favor of onnxruntime-web).
- **Realtime / streaming?** No. It is not a streaming implementation and does not thread cache state across buffers.
- **Maintained?** No.
- **License:** not audited (relevant only if we vendor code, which we won't; we cite it as prior art we extend).
- `README.md` is not present at `master` HEAD (raw fetch 404) — repo content may have been trimmed, consistent with abandonment.

### 2. acids-ircam (GitHub User account, not org) — full public repo list audited via API (updated order)
- **RAVE** — official model repo. `pushed_at 2026-03`, actively maintained. Configs confirmed: v1, v2, v2_small, v2_nopqmf, v3 (Snake), discrete, `onnx` (= "Noiseless v1 configuration for onnx usage"), raspberry. Realtime targets are all **native**: nn~ (Max/PD), RAVE VST, embedded Raspberry Pi. **No browser artifact.**
- **cached_conv** — the streaming-conv library (DAFx 2022). Ships Max/PD external + VST. No browser runtime.
- **nn_tilde** — Max/MSP + PureData external for TorchScript neural audio. Native only.
- **rave_vst** — VST2/VST3/AU plugin (JUCE). Native only.
- **neurorack**, **AFTER** ("Audio Features Transfer and Exploration in Real-time", Python/native), **vschaos2**, **creative_ml**, **ddsp_pytorch** — none are browser runtimes.
- No unreleased-looking browser/streaming RAVE repo appears in the public list.

### 3. ircam-ismm — full public repo list audited via API
- Repos: sync, xmm, wavelet, pipo-sdk, parameters, pipo, ticker, rta-lib, bayesfilter (motion/audio DSP libraries). None relate to RAVE or browser neural audio. Nothing analogous.

### 4. Neutone — github.com/Neutone
- **neutone_sdk** (+ Neutone FX/Gen VST/AU plugins): real-time neural audio **in DAWs**, native. Includes a RAVE timbre-transfer notebook target. Explicitly a native plugin ecosystem; no browser runtime. Not prior art for the browser question.

### 5. HuggingFace Spaces (static SDK)
- No static (client-side) RAVE space found. RAVE checkpoints appear as server-side Gradio demos, not browser inference.

### 6. General web / GitHub sweep
- **baditaflorin/worldvoice** — closest *appearance* of prior art ("browser-native live audio transformer", mic → instruments, ONNX Runtime Web probes). **Does not run RAVE**: README states "True DDSP/RAVE model weights are not bundled"; the v1 chain is WebAudio pitch-following + harmonic resynthesis. Good adjacent reference for UI/UX and for the browser-audio-systems domain; explicitly leaves the neural timbre-transfer gap open.
- **usdivad/Raveler** — Wwise (native) plugin running RAVE via ONNX. Native, not browser. Useful reference for ONNX-based RAVE deployment outside LibTorch (TorchScript is not required — ONNX works) — supports the M1 feasibility of ONNX export.
- **Fontasio / ilmentatore** (RAVE VST Model Explorer, "iskn/Studio.One.RAVE") — native plugin model galleries. Not browser.
- WAC/NIME/DAFx proceedings: Faust spectral-processing worklets, WebChucK, iPlug2/WAM, JSPatcher, Essentia.js, Meyda — all realtime DSP or feature extraction in the browser; **none** runs a streaming stateful neural waveform-to-waveform model under a real-time deadline. No streaming-RT neural timbre transfer appears in any proceedings searched.

### 7. Runtime feasibility notes (for M1 onward)
- **onnxruntime-web** supports WASM (SIMD, multi-thread), WebGL (deprecated path), and **WebGPU** (Chromium 113+, no Safari/Firefox) — confirms the benchmark matrix (WASM SIMD / threads / WebGPU) is real, and that WASM-threads is the only cross-browser route (Safari, Firefox).
- ONNX, not TorchScript, is an independently-proven RAVE deployment format (Raveler) — good sign for the streaming-export approach being viable.
- Unrelated adjacent work (StreamDiffusion-style ring-buffer streaming of generative models, e.g. HF paper 2605.28657) reinforces the M6 thesis that "stateful model under a real-time deadline" is a general harness pattern, not audio-specific.

## Implication for the plan
The **streaming-export contribution remains ours**: nobody has shipped mic → RAVE → speakers, fully client-side, with cache tensors threaded across ONNX calls. We extend RAVE.js (which is a static renderer on a dead runtime) with a streaming, stateful, modern-ORT implementation and, per the brief, publish benchmark numbers nobody has.

## Cite in README (per brief §10)
- RAVE — Caillon & Esling, arXiv:2111.05011
- cached_conv — Caillon et al., DAFx 2022, arXiv:2204.07064
- RAVE.js — Antoine Caillon, prior art being extended
- worldvoice — adjacent browser-audio reference (no RAVE)