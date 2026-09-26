import { decodeWav, encodeWavPcm16 } from "./wav";
import { RaveManifest } from "./manifest";
import { createRealtime, crossOriginIsolated, ModelUrls } from "./realtime";

const statusEl = document.getElementById("status")!;
const runBtn = document.getElementById("run") as HTMLButtonElement;
const dlBtn = document.getElementById("download") as HTMLButtonElement;
const fileEl = document.getElementById("file") as HTMLInputElement;
const resultEl = document.getElementById("result") as HTMLAnchorElement;
const rtStartBtn = document.getElementById("rt-start") as HTMLButtonElement;
const rtStopBtn = document.getElementById("rt-stop") as HTMLButtonElement;
const coiEl = document.getElementById("coi")!;
const rtUnderruns = document.getElementById("rt-underruns")!;
const rtBlocks = document.getElementById("rt-blocks")!;
const rtAvg = document.getElementById("rt-avg")!;
const rtMax = document.getElementById("rt-max")!;
const rtLat = document.getElementById("rt-lat")!;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const modelInfo = document.getElementById("model-info")!;
const slackEl = document.getElementById("slack") as HTMLInputElement;
const slackV = document.getElementById("slack-v")!;
const noiseEl = document.getElementById("noise") as HTMLInputElement;
const noiseV = document.getElementById("noise-v")!;
const gainEl = document.getElementById("gain") as HTMLInputElement;
const gainV = document.getElementById("gain-v")!;
const eqLowEl = document.getElementById("eq-low") as HTMLInputElement;
const eqMidEl = document.getElementById("eq-mid") as HTMLInputElement;
const eqHighEl = document.getElementById("eq-high") as HTMLInputElement;
const eqLowV = document.getElementById("eq-low-v")!;
const eqMidV = document.getElementById("eq-mid-v")!;
const eqHighV = document.getElementById("eq-high-v")!;
const wetEl = document.getElementById("wet") as HTMLInputElement;
const wetV = document.getElementById("wet-v")!;
const scopeEl = document.getElementById("scope") as HTMLCanvasElement;
const scopeN = document.getElementById("scope-n")!;

const log = (line: string, cls = "") => {
  statusEl.textContent += `${line}\n`;
  if (cls) statusEl.innerHTML = statusEl.innerHTML
    .replace(/\n$/, "") + `\n<span class="${cls}">${line}</span>\n`;
};

const fmt = (x: number): string => x.toExponential(2);

interface CatalogEntry {
  key: string;
  label: string;
  arch: string;
  seed: number;
  trained?: boolean;
  realtime?: boolean;
  block_size: number;
  ratio: number;
  latent_steps: number;
  latent_size: number;
  full_latent_size: number;
  bytes: { encoder: number; decoder: number; warmup: number };
  urls: { encoder: string; decoder: string; manifest: string; warmup: string } & ModelUrls;
}

type WorkerEvent =
  | { type: "loaded"; key?: string }
  | { type: "processed"; y: Float32Array }
  | { type: "parity"; ok: boolean; n: number; maxerr: number; bad: number; msPerBuf: number }
  | { type: "reset"; ok: boolean }
  | { type: "error"; message: string };

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});

let outSamples: Float32Array | null = null;
let outRate = 44100;

// -- model catalog ------------------------------------------------
let catalog: CatalogEntry[] = [];
let current: CatalogEntry | null = null;
let pipelineKey: string | null = null;
const keyWaiters = new Map<string, () => void>();
const markLoaded = (key?: string) => {
  if (!key) return;
  pipelineKey = key;
  keyWaiters.get(key)?.();
  keyWaiters.delete(key);
};
const waitForKey = (key: string): Promise<void> =>
  new Promise((res) => {
    if (pipelineKey === key) return res();
    keyWaiters.set(key, res);
  });

const bootLoad = new Promise<void>((resolve) => {
  const h = (e: MessageEvent<WorkerEvent>) => {
    if (e.data.type === "loaded") {
      worker.removeEventListener("message", h);
      runBtn.disabled = false;
      resolve();
    }
  };
  worker.addEventListener("message", h);
});

worker.onmessage = (e: MessageEvent<WorkerEvent>) => {
  switch (e.data.type) {
    case "loaded": {
      const key = e.data.key;
      markLoaded(key);
      const entry = catalog.find((c) => c.key === key);
      log(`models loaded: ${entry?.label ?? key}`, "ok");
      runBtn.disabled = false;
      break;
    }
    case "processed": {
      outSamples = new Float32Array(e.data.y);
      dlBtn.disabled = false;
      log(`block rendered: ${outSamples.length} samples`);
      break;
    }
    case "parity": {
      const d = e.data;
      const cls = d.ok && d.maxerr < 1e-4 && d.bad === 0 ? "ok" : "err";
      const verdict = cls === "ok" ? "PASS" : "FAIL";
      log(
        `browser parity: n=${d.n} maxerr=${fmt(d.maxerr)} ` +
          `bad=${d.bad} ${d.msPerBuf.toFixed(2)} ms/buf ${verdict}`,
        cls,
      );
      break;
    }
    case "reset":
      log("state reset to warmup");
      break;
    case "error":
      log(`worker error: ${e.data.message}`, "err");
      break;
  }
};

const preloadModel = async (entry: CatalogEntry) => {
  const enc = await fetch(entry.urls.encoder, { cache: "no-store" })
    .then((r) => r.arrayBuffer());
  const dec = await fetch(entry.urls.decoder, { cache: "no-store" })
    .then((r) => r.arrayBuffer());
  return { encoder: enc, decoder: dec };
};

const loadIntoWorker = async (entry: CatalogEntry) => {
  if (pipelineKey === entry.key) return;
  const buffers = await preloadModel(entry);
  worker.postMessage(
    {
      type: "load",
      key: entry.key,
      urls: entry.urls as { encoder: string; decoder: string; manifest: string; warmup: string },
      buffers,
    },
    [buffers.encoder, buffers.decoder],
  );
  await waitForKey(entry.key);
};

const renderModelInfo = () => {
  if (!current) return;
  const c = current;
  const mb = ((c.bytes.encoder + c.bytes.decoder) / 1e6).toFixed(1);
  modelInfo.textContent =
    `${c.arch} · ${c.trained ? "trained" : `seed ${c.seed}`} · ${c.block_size}-sample blocks @ ${c.ratio} ratio ` +
    `· latent ${c.latent_size} dims × ${c.latent_steps} steps · ${mb} MB onnx`;
};

void (async () => {
  try {
    catalog = await fetch("models/models.json").then((r) => r.json());
  } catch (err) {
    log(`catalog fetch failed: ${(err as Error).message}`, "err");
    return;
  }
  for (const c of catalog) {
    const mb = ((c.bytes.encoder + c.bytes.decoder) / 1e6).toFixed(1);
    const opt = document.createElement("option");
    opt.value = c.key;
    opt.textContent = `${c.label} — ${mb} MB`;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = catalog[0].key;
  const initial = catalog[0];
  current = initial;
  renderModelInfo();
  log(`catalog: ${catalog.length} models, default ${catalog[0].label}`);
  const b = await preloadModel(initial);
  worker.postMessage(
    {
      type: "load",
      key: initial.key,
      urls: initial.urls as any,
      buffers: b,
    },
    [b.encoder, b.decoder],
  );
  await bootLoad;
})();

modelSelect.onchange = async () => {
  const next = catalog.find((c) => c.key === modelSelect.value);
  if (!next || next.key === current?.key) return;
  current = next;
  renderModelInfo();
  rtStartBtn.disabled = next.realtime === false;
  if (next.realtime === false) {
    log(`${next.label}: file mode only (misses the sustained realtime deadline)`);
  }
  log(`loading ${next.label} (lazy)...`);
  try {
    await loadIntoWorker(next);
  } catch (err) {
    log(`model load failed: ${(err as Error).message}`, "err");
  }
  if (rt) {
    rt.stop();
    rt = null;
    resetRtDisplay();
    rtStartBtn.disabled = next.realtime === false;
    rtStopBtn.disabled = true;
    log("realtime: stopped — restart to apply new model");
  }
};

// -- latent + transport controls -----------------------------------
const LAT_DIMS = 8;
const bias = new Float32Array(LAT_DIMS);
const scale = new Float32Array(LAT_DIMS).fill(1);
let noiseGain = 1;
let wet = 1;
let inputGain = 1;
let eqDb: [number, number, number] = [0, 0, 0];
let slackBlocks = 4;

const pushLatent = () => {
  rt?.setLatent(Array.from(bias), Array.from(scale), noiseGain);
};

let lastLatentPush = 0;
const debouncedLatent = () => {
  const now = Date.now();
  if (now - lastLatentPush < 120) return;
  lastLatentPush = now;
  pushLatent();
};

const latRow = document.getElementById("lat-row")!;
for (let d = 0; d < LAT_DIMS; d++) {
  const wrap = document.createElement("div");
  const b = document.createElement("input");
  b.type = "range"; b.min = "-2"; b.max = "2"; b.step = "0.01"; b.value = "0";
  b.title = `dim-${d}-bias`;
  const s = document.createElement("input");
  s.type = "range"; s.min = "-2"; s.max = "2"; s.step = "0.01"; s.value = "1";
  s.title = `dim-${d}-scale`;
  const label = document.createElement("label");
  label.textContent = `dim ${d + 1} (bias · scale)`;
  wrap.style.cssText = "display:flex;gap:0.5rem;align-items:center;flex:1";
  wrap.append(label, b, s);
  latRow.appendChild(wrap);
  b.oninput = () => { bias[d] = Number(b.value); debouncedLatent(); };
  s.oninput = () => { scale[d] = Number(s.value); debouncedLatent(); };
}

noiseEl.oninput = () => {
  noiseGain = Number(noiseEl.value) / 100;
  noiseV.textContent = noiseGain.toFixed(2);
  debouncedLatent();
};
gainEl.oninput = () => {
  inputGain = Number(gainEl.value) / 100;
  gainV.textContent = inputGain.toFixed(2);
  rt?.setInputGain(inputGain);
};
const setEq = (i: 0 | 1 | 2, db: number) => {
  eqDb[i] = db;
  [eqLowV, eqMidV, eqHighV][i].textContent = `${db} dB`;
  rt?.setEq(eqDb);
};
eqLowEl.oninput = () => setEq(0, Number(eqLowEl.value));
eqMidEl.oninput = () => setEq(1, Number(eqMidEl.value));
eqHighEl.oninput = () => setEq(2, Number(eqHighEl.value));
wetEl.oninput = () => {
  wet = Number(wetEl.value) / 100;
  wetV.textContent = `${Math.round(wet * 100)}% wet`;
  rt?.setMix(wet);
};
slackEl.oninput = () => {
  slackBlocks = Number(slackEl.value);
  slackV.textContent = `${slackBlocks} blocks`;
};
slackEl.onchange = () => applySlack();

// -- latent scope ------------------------------------------------
const SCOPE_N = 240;
const scopeHist: number[][] = [];
let scopeFrames = 0;
let scopeRaf = 0;
const ctx2 = scopeEl.getContext("2d")!;
const onScope = (z: Float32Array) => {
  const row = new Array(LAT_DIMS);
  for (let i = 0; i < LAT_DIMS; i++) row[i] = z[i] ?? 0;
  scopeHist.push(row);
  if (scopeHist.length > SCOPE_N) scopeHist.shift();
  scopeFrames++;
  scopeN.textContent = String(scopeFrames);
  if (!scopeRaf) scopeRaf = requestAnimationFrame(drawScope);
};
const drawScope = () => {
  scopeRaf = 0;
  const { width: W, height: H } = scopeEl;
  ctx2.clearRect(0, 0, W, H);
  if (!scopeHist.length) return;
  let maxA = 1e-3;
  for (const row of scopeHist) for (const v of row) maxA = Math.max(maxA, Math.abs(v));
  ctx2.strokeStyle = "#2a2f3a";
  ctx2.beginPath();
  ctx2.moveTo(0, H / 2); ctx2.lineTo(W, H / 2);
  ctx2.stroke();
  const n = scopeHist.length;
  for (let d = 0; d < LAT_DIMS; d++) {
    const hue = (d / LAT_DIMS) * 200 + 180;
    ctx2.strokeStyle = `hsla(${hue},70%,55%,0.75)`;
    ctx2.lineWidth = 1.2;
    ctx2.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (SCOPE_N - 1)) * W;
      const y = H / 2 - (scopeHist[i][d] / maxA) * (H / 2 - 4);
      if (i === 0) ctx2.moveTo(x, y); else ctx2.lineTo(x, y);
    }
    ctx2.stroke();
  }
};

// -- realtime -------------------------------------------------------
const resetRtDisplay = () => {
  rtUnderruns.textContent = "0";
  rtBlocks.textContent = "0";
  rtAvg.textContent = "–";
  rtMax.textContent = "–";
  rtLat.textContent = "–";
  scopeHist.length = 0;
  scopeN.textContent = "0";
  drawScope();
};

const renderRt = () => {
  if (!rt) return;
  rtUnderruns.textContent = String(rt.underruns());
  rtBlocks.textContent = String(rt.blocksProcessed());
  rtAvg.textContent = `${rt.avgTurnaroundMs().toFixed(1)} ms`;
  rtMax.textContent = `${rt.maxTurnaroundMs().toFixed(1)} ms`;
  rtLat.textContent = `${rt.latencyMs()} ms`;
};

let rt: Awaited<ReturnType<typeof createRealtime>> | null = null;
let restarting = false;

const makeRealtime = async (): Promise<void> => {
  if (!current) throw new Error("no model selected");
  resetRtDisplay();
  rt?.stop();
  rt = await createRealtime({
    log,
    tick: renderRt,
    urls: current.urls,
    buffers: await preloadModel(current),
    slackBlocks,
    onScope,
    onError: () => {},
  });
  // inherit current control positions on (re)start
  rt.setMix(wet);
  rt.setInputGain(inputGain);
  rt.setEq(eqDb);
  rt.setLatent(Array.from(bias), Array.from(scale), noiseGain);
};

const applySlack = async () => {
  if (!rt?.running()) return;
  if (restarting) return;
  restarting = true;
  log(`realtime: slack -> ${slackBlocks} blocks (restarting)`);
  try {
    await makeRealtime();
    await rt.start();
  } catch (err) {
    log(`realtime: ${(err as Error).message}`, "err");
    rtStartBtn.disabled = false;
    rtStopBtn.disabled = true;
  } finally {
    restarting = false;
  }
};

rtStartBtn.onclick = async () => {
  try {
    if (!rt) await makeRealtime();
    await rt!.start();
    rtStartBtn.disabled = true;
    rtStopBtn.disabled = false;
  } catch (err) {
    log(`realtime: ${(err as Error).message}`, "err");
  }
};

rtStopBtn.onclick = () => {
  rt?.stop();
  rtStartBtn.disabled = false;
  rtStopBtn.disabled = true;
};

// -- file mode ------------------------------------------------------
const mulberry32 = (seed: number) => {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

runBtn.onclick = async () => {
  try {
    await waitForKey(current!.key);
  } catch {
    return log("models not ready", "err");
  }
  const file = fileEl.files?.[0];
  if (!file) return log("choose a WAV first", "err");
  log(`loading ${file.name}...`);
  const wav = decodeWav(await file.arrayBuffer());
  log(
    `wav: ${wav.sampleRate} Hz, ${wav.numChannels} ch, ` +
      `${wav.samples.length} samples`,
  );
  if (wav.numChannels > 1) log("downmixing channel 0");

  const manifest = await fetch(current!.urls.manifest).then(
    (r) => r.json(),
  ) as RaveManifest;
  const B = manifest.block_size;
  const fl = manifest.full_latent_size;
  const ls = manifest.latent_size;
  const T = Math.max(1, Math.round(B / manifest.ratio));
  const mono = wav.numChannels === 1
    ? wav.samples
    : wav.samples.filter((_, i) => i % wav.numChannels === 0);

  const nBlocks = Math.ceil(mono.length / B);
  const padded = new Float32Array(nBlocks * B);
  padded.set(mono.subarray(0, mono.length));

  const rng = mulberry32(0);
  const out = new Float32Array(nBlocks * B);
  const eps = new Float32Array(fl * T);
  const noise = new Float32Array((fl - ls) * T);
  runBtn.disabled = true;
  for (let k = 0; k < nBlocks; k++) {
    for (let i = 0; i < fl * T; i++) eps[i] = rng() * 2 - 1;
    for (let i = 0; i < (fl - ls) * T; i++) noise[i] = rng() * 2 - 1;
    const y = await new Promise<Float32Array>((resolve, reject) => {
      const handler = (e: MessageEvent<WorkerEvent>) => {
        if (e.data.type === "processed") {
          worker.removeEventListener("message", handler);
          resolve(new Float32Array(e.data.y));
        } else if (e.data.type === "error") {
          worker.removeEventListener("message", handler);
          reject(new Error(e.data.message));
        }
      };
      worker.addEventListener("message", handler);
      worker.postMessage({
        type: "process",
        x: padded.subarray(k * B, (k + 1) * B),
        eps,
        noise,
        bias,
        scale,
        noiseGain,
      });
    });
    out.set(y, k * B);
    if ((k + 1) % 100 === 0) log(`  ${k + 1}/${nBlocks} blocks`);
  }
  outSamples = out.subarray(0, mono.length);
  outRate = wav.sampleRate;
  dlBtn.disabled = false;
  runBtn.disabled = false;
  log("done.", "ok");
};

dlBtn.onclick = () => {
  if (!outSamples) return;
  const blob = new Blob([encodeWavPcm16(outSamples, outRate)], {
    type: "audio/wav",
  });
  const url = URL.createObjectURL(blob);
  resultEl.href = url;
  resultEl.download = "rave_live_out.wav";
  resultEl.textContent = "save rave_live_out.wav";
  resultEl.style.display = "block";
  resultEl.click();
};

document.getElementById("parity")!.onclick = async () => {
  // The committed Python reference bundle belongs to the original v2rt gate.
  const def = catalog.find((m) => m.key === "v2rt-s0") ?? catalog[0];
  try {
    if (def.key !== pipelineKey) await loadIntoWorker(def);
  } catch (err) {
    return log(`models: ${(err as Error).message}`, "err");
  }
  log("browser parity vs Python reference (128 blocks)...");
  worker.postMessage({
    type: "parity",
    urls: {
      audio: "parity/audio.bin",
      eps: "parity/eps.bin",
      noise: "parity/noise.bin",
      refY: "parity/ref_y.bin",
    },
  });
};

document.getElementById("reset")!.onclick = () => {
  worker.postMessage({ type: "reset" });
};

coiEl.textContent = crossOriginIsolated()
  ? "cross-origin isolated: yes (SAB / threads available)"
  : "cross-origin isolated: NO — realtime unavailable (see web/README)";
coiEl.className = crossOriginIsolated() ? "ok" : "err";
