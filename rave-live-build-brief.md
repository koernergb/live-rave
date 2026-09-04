# RAVE-Live — Build Brief

**Realtime neural timbre transfer in a browser tab. No install, no server, no plugin.**

Sing into your laptop mic, come out the other side as a darbouka, a flock of birds, or a saxophone — at interactive latency, entirely client-side.

---

## 1. Why this exists

RAVE (Caillon & Esling, IRCAM) is the standard realtime neural audio autoencoder. Its realtime story today is entirely native: `nn~` externals for Max/PD, the IRCAM RAVE VST, Neutone. The only official browser artifact is **RAVE.js** (`caillonantoine.github.io/ravejs`), which is offline — you hand it a file, it renders — and is built on onnx.js, the deprecated predecessor to onnxruntime-web.

Nobody has shipped **streaming** RAVE in a browser. The gap is not the model and not the audio API. It's that RAVE's realtime mode depends on `cached_conv`, where convolution padding is replaced by a persistent cached state carried across buffers, and an ONNX graph is stateless by construction. Bridging that is the project.

**Secondary thesis, and the part that outlives the demo:** the thing you build to solve this — a stateful model running under a real-time deadline in a browser, with cache tensors threaded across invocations — is not audio-specific. Factor it correctly and it's the harness for any streaming on-device inference on the web.

**Portfolio framing:** breadth. This sits next to the 3D/spatial work rather than inside it, and demonstrates browser-side systems engineering under a hard real-time deadline in a domain that reads as unrelated. It is also, bluntly, the most demo-able thing in the queue — audio transformation is legible to people who will never understand a splat viewer.

---

## 2. Before you write a line of code

Spend two hours confirming this is actually unbuilt. A generic web search does not cover this space. Check, in order:

1. **Web Audio Conference** proceedings (webaudioconf.com) — 2018 through most recent.
2. **NIME** and **DAFx** proceedings, search "browser", "WebAssembly", "web audio" + neural.
3. **github.com/ircam-ismm** and **github.com/acids-ircam** — full repo list, including anything unreleased-looking.
4. **Neutone** (`github.com/Neutone`) — they had a web presence and may have prototyped this.
5. **HuggingFace Spaces**, filter for `static` SDK (a static Space is by definition client-side).

If someone has done it: read what they did, decide whether the streaming-export contribution is still yours, and pivot to "first *maintained* implementation with published numbers" if so. Do not silently discover this at week three.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Main thread                                                 │
│  UI, latent sliders, model picker, mic permission           │
└───────────────┬─────────────────────────────────────────────┘
                │ postMessage (control only, never audio)
┌───────────────▼──────────┐        ┌────────────────────────┐
│ Web Worker               │        │ AudioWorkletProcessor  │
│  onnxruntime-web         │◄──SAB──┤  128-frame quanta      │
│  encoder + decoder       │  ring  │  I/O only, no compute  │
│  cache tensors held here │ buffers│  Atomics.notify        │
└──────────────────────────┘        └────────────────────────┘
                                              │
                                    getUserMedia ──► speakers
```

**The non-negotiable rule:** the AudioWorklet does zero inference. It runs on the audio thread with a hard 128-frame deadline (2.67 ms at 48 kHz). It copies samples into a ring buffer, copies samples out, and signals. Everything else happens in the Worker.

### Why not WebGPU

RAVE is a stack of many small convolutions — memory-bound rather than compute-bound. Dispatch overhead at a 2048-sample block plausibly exceeds the compute saved. **Benchmark both, but expect WASM SIMD + threads to win**, and treat that as a finding worth reporting rather than a disappointment. WebGPU may pull ahead only at larger blocks, which is exactly where latency becomes unusable — that trade-off curve is a result.

### Cross-origin isolation

`SharedArrayBuffer` and ORT's threaded WASM backend both require:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

GitHub Pages cannot set response headers. Options: `coi-serviceworker` shim (works, adds a reload on first visit, fragile on Safari), or host on **Cloudflare Pages / Netlify** with a `_headers` file. Pick the latter. Decide this on day one — retrofitting COOP/COEP breaks every third-party asset you've pulled in by then.

---

## 4. The hard part: stateful ONNX export

This is 60% of the project risk. Budget accordingly.

### The problem

`cached_conv` replaces padding with a cache buffer stored on the module (`self.cache`). TorchScript preserves that. `torch.onnx.export` will trace through it and **bake the cache in as a constant** — you get a graph that runs, produces plausible-sounding audio for one buffer, and is silently wrong on every subsequent one. This failure mode is quiet. Assume it is happening until you have proven otherwise numerically.

### The approach

Hoist every cache to graph I/O. Write a wrapper module whose signature is:

```python
class StreamingRAVE(nn.Module):
    def forward(self, x, *caches):
        # -> (y, *new_caches)
```

Implementation sketch:

1. Walk `model.named_modules()`, collect every `CachedConv1d` / `CachedPadding1d` instance into an **ordered** list. Ordering must be deterministic and identical at export and at runtime — key them by module path string, sort, and freeze that list into a manifest JSON shipped alongside the model.
2. Monkeypatch each cached module's `forward` to read its cache from a shared list indexed by position, and write the updated cache back to a parallel output list, rather than touching `self`.
3. Export **encoder and decoder as separate graphs**. You need the latent exposed in JS for the manipulation UI, and separate graphs let you skip the encoder entirely for prior-driven generation later.
4. Fix the block size at export time. Mark batch as dynamic if you want, but keep the time axis static — dynamic shapes cost you kernel specialization in the WASM EP.

### Parity test — the gate

Do not proceed past this until it passes.

- Feed the same 10,000-buffer sequence through (a) the TorchScript `--streaming` export and (b) your ONNX graphs with manual cache threading.
- **Pass:** max absolute error < 1e-4, and — critically — **no monotonic drift** across the sequence. Plot per-buffer error against buffer index. A flat noise floor means your caches are threaded correctly. An upward ramp means one cache is being dropped, and it will sound like slow-onset garbage that you'd otherwise spend a week blaming on the ring buffer.
- Also test: a full reset (all caches zeroed) mid-stream should produce exactly the same output as starting fresh.

### Model version risk

The RAVE repo ships an `onnx` training config described as a *noiseless v1 configuration for ONNX usage*. That's a generation behind what people actually use, and if you ship it the demo will sound dated to anyone who knows the VST.

Attempt v2 first. Specific hazards:
- **PQMF** — convolutional, should export cleanly. Verify the analysis/synthesis pair round-trips.
- **Noise synthesizer branch** — involves random sampling. Do not rely on `RandomNormalLike`; **feed noise as an explicit graph input** generated in JS. This also gives you a free UI control and makes outputs reproducible for the parity test.
- **Snake activation** (v3) — `x + sin²(ax)/a`, elementwise, exportable, but check op coverage in the WASM EP before committing to v3.

**Fallback ladder** if v2 export resists: `v2_small` → `raspberry` config → v1-noiseless. Ship something rather than nothing.

---

## 5. Ring buffer contract

Two `SharedArrayBuffer`s (input, output), each a lock-free SPSC ring of `Float32Array`, plus an `Int32Array` control block for read/write indices and an underrun counter.

- **Worklet side:** on each 128-frame quantum, write input frames, read output frames, `Atomics.store` the new write index, `Atomics.notify` the worker.
- **Worker side:** `Atomics.wait` until at least `BLOCK` frames are available, drain a block, run encode → manipulate → decode, write to output ring.
- **Underrun policy:** if the output ring is short when the worklet needs frames, emit a short crossfade to silence rather than a hard cut, and **increment a counter**. Underrun rate per minute is a headline metric, not a bug to hide.
- **Sizing:** output ring holds `2 × BLOCK` minimum. Expose the slack as a user-facing "stability vs latency" slider — it's an honest control and it makes the latency trade-off visible in the demo itself.

### Latency budget

| Component | Frames @ 48 kHz | ms |
|---|---|---|
| Model block (v2, ratio 2048) | 2048 | 42.7 |
| Ring slack (1 block) | 2048 | 42.7 |
| Worklet quantum | 128 | 2.7 |
| Output device buffer | — | ~10–20 |
| **Total** | | **~100–110** |

That is playable-but-not-tight. Halving the compression ratio at training time is the documented route to lower latency — out of scope here, but say so in the writeup rather than letting a reader think 100 ms is the floor.

### Mic capture

```js
getUserMedia({ audio: {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
}})
```

Leaving these on will destroy your input signal — AGC in particular fights the model's amplitude sensitivity. RAVE output varies strongly with input amplitude envelope and spectral content, so give the user an input gain control and a bandpass/EQ. Practitioners steer RAVE with gain and filtering more than with latent editing; expose that.

---

## 6. Interaction design

The encode → manipulate → decode split is the whole point. Ship:

- **Model picker** — 4–6 checkpoints, lazy-loaded, with byte size shown.
- **Per-dimension latent bias/scale** — first 8 dims, sliders. This is the "neural" part made tangible.
- **Noise amount** — into the explicit noise input.
- **Wet/dry** — always. Users need the A/B to hear what happened.
- **Input gain + 3-band EQ** — the practitioner's real control surface.
- **Latency/stability slider** — exposes the ring slack.
- **Live latent scope** — a small canvas plotting latent dims over time. Costs an afternoon, and it is the thing that makes the demo video look like research rather than a toy.

Include a **file mode** as well as mic mode. It's ten extra lines, it's what people try first, and it lets the demo work on a phone without mic permission friction.

---

## 7. Benchmarks — the publishable core

This is what separates a demo from a contribution. Nobody has these numbers.

| Axis | Values |
|---|---|
| Backend | WASM (SIMD), WASM (SIMD+threads, 2/4/8), WebGPU |
| Block size | 512, 1024, 2048, 4096 |
| Model | v1-noiseless, v2, v2_small, raspberry |
| Browser | Chrome, Safari, Firefox |
| Device | M-series Mac, x86 laptop, Android phone, iPhone |

**Metrics:** real-time factor (median + p99 — the tail is what causes dropouts), underrun rate per minute, cold-start time to first audio, peak memory, and numeric parity vs the TorchScript reference.

Publish as a table plus an RTF-vs-block-size curve per backend. The p99/median gap across browsers is likely the most interesting single result — GC pauses and thread scheduling differ enormously and nobody has measured it for an audio workload.

---

## 8. Milestones

| # | Deliverable | Gate | Est. |
|---|---|---|---|
| **M0** | Prior-art check complete | Written summary of what exists | 0.5 d |
| **M1** | Cache-hoisted ONNX export, encoder + decoder | **Parity test passes, no drift over 10k buffers** | 4–6 d |
| **M2** | ORT in Worker, offline file processing | Bit-comparable to M1 Python output | 2 d |
| **M3** | SAB ring + AudioWorklet, mic → speakers | 60 s continuous, zero underruns on M-series | 3 d |
| **M4** | Latent UI, model picker, wet/dry, latent scope | Deployed, cross-origin isolated, loads on a cold phone | 3 d |
| **M5** | Benchmark sweep + README with numbers | Table + curves committed | 2–3 d |
| **M6** | Harness factored into a standalone module | Second (non-audio) toy model runs through it unchanged | 2 d |

**~17–20 focused days.** M1 is the one that slips. If it hasn't converged in 8 days, drop to the v1-noiseless config and move on — the systems contribution survives a weaker model, and the parity harness is the reusable asset either way.

M6 is optional for shipping but it's the milestone that makes this more than a one-off. A generic `stateful-ort-stream` module — thread N cache tensors across invocations, SPSC ring, worklet shim, underrun accounting — is genuinely reusable and is the piece of this that transfers to on-device work you actually care about.

---

## 9. Kill criteria

Decide these now, honestly, and honor them:

- **Prior art exists and is maintained and is realtime** → stop, write a 300-word note on what they did, spend the days elsewhere.
- **M1 parity fails on every config after 8 days** → the streaming-export problem is harder than budgeted; ship an offline modern-ORT rebuild of RAVE.js instead (still an improvement on the deprecated onnx.js version) and stop.
- **RTF > 0.8 at every workable block size on an M-series machine** → no realtime story exists on this hardware generation. Publish the benchmark table as a negative result and stop. A clean "here's why this doesn't work yet, with numbers" is a real artifact.

---

## 10. Licensing and attribution

RAVE checkpoints from IRCAM and from the Intelligent Instruments Lab are **CC-BY-NC-4.0**. Non-commercial only. Generated audio is fine for non-commercial use; shipping the models inside anything paid needs permission from IRCAM.

Consequences: no ads, no sponsorship, no "pro tier", ever, on this domain. State the license prominently in the README and in the page footer.

Cite:
- RAVE — Caillon & Esling, arXiv:2111.05011
- Streamable neural audio synthesis with non-causal convolutions (`cached_conv`) — Caillon et al., DAFx 2022, arXiv:2204.07064
- RAVE.js — Antoine Caillon, as prior art you're extending

Credit Caillon and the ACIDS team unambiguously and early. This is their model; you built a runtime for it.

---

## 11. Deliverables

1. **Repo** — `rave-live`, MIT for your code, with the checkpoint license called out separately.
2. **Live demo** on Cloudflare Pages, cross-origin isolated, loads on mobile.
3. **README** with the benchmark table, the latency budget, and an honest "what doesn't work yet."
4. **60-second demo video** — voice in, four models cycled, latent sliders moving, latent scope visible. Lead with the sound, not the architecture diagram.
5. **`stateful-ort-stream`** as a separate small repo (M6).
6. *Optional:* Web Audio Conference demo submission. Low cost once the benchmark table exists — the paper is the README with a related-work section bolted on.

---

## 12. Post framing

Lead the LinkedIn/X post with the audio clip and one sentence: neural timbre transfer, realtime, in a browser tab, nothing installed. Then the technical hook underneath — that the interesting problem was making a stateful convolutional model streamable inside a stateless graph format, and here are the numbers across four browsers.

Do not open with the architecture. Open with the sound.
