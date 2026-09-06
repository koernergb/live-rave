import { decodeWav, encodeWavPcm16 } from "./wav";
import { RaveManifest } from "./manifest";
import { createRealtime, crossOriginIsolated } from "./realtime";

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

const log = (line: string, cls = "") => {
  statusEl.textContent += `${line}\n`;
  if (cls) statusEl.innerHTML = statusEl.innerHTML
    .replace(/\n$/, "") + `\n<span class="${cls}">${line}</span>\n`;
};

const fmt = (x: number): string => x.toExponential(2);

type LogMsg =
  | { type: "loaded" }
  | { type: "processed"; y: Float32Array }
  | { type: "parity"; ok: boolean; n: number; maxerr: number; bad: number; msPerBuf: number }
  | { type: "reset"; ok: boolean }
  | { type: "error"; message: string };

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});

let outSamples: Float32Array | null = null;
let outRate = 44100;

let resolveLoaded: () => void = () => {};
const loadDone = new Promise<void>((r) => {
  resolveLoaded = r;
});
const awaitLoaded = async () => {
  await Promise.race([
    loadDone,
    new Promise<never>((_, rej) =>
      setTimeout(() => rej(new Error("timed out waiting for models")), 90_000)),
  ]);
};

worker.onmessage = (e: MessageEvent<LogMsg>) => {
  switch (e.data.type) {
    case "loaded":
      resolveLoaded();
      log("models loaded (encoder 80 / decoder 76 caches)");
      runBtn.disabled = false;
      break;
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

const preloadModels = async () => {
  const enc = await fetch("models/encoder.onnx", { cache: "no-store" }).then((r) => r.arrayBuffer());
  const dec = await fetch("models/decoder.onnx", { cache: "no-store" }).then((r) => r.arrayBuffer());
  const warm = await fetch("models/warmup.bin", { cache: "no-store" }).then((r) => r.arrayBuffer());
  return { encoder: enc, decoder: dec, warmup: warm };
};

void (async () => {
  try {
    const b = await preloadModels();
    worker.postMessage(
      {
        type: "load",
        urls: {
          encoder: "models/encoder.onnx",
          decoder: "models/decoder.onnx",
          manifest: "models/manifest.json",
          warmup: "models/warmup.bin",
        },
        buffers: { encoder: b.encoder, decoder: b.decoder },
      },
      [b.encoder, b.decoder],
    );
  } catch (err) {
    log(`model preload failed: ${(err as Error).message}`, "err");
  }
})();

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
    await awaitLoaded();
  } catch (err) {
    return log(`models: ${(err as Error).message}`, "err");
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

  const manifest = await fetch("models/manifest.json").then(
    (r) => r.json(),
  ) as RaveManifest;  const B = manifest.block_size;
  const fl = manifest.full_latent_size;
  const ls = manifest.latent_size;
  const mono = wav.numChannels === 1
    ? wav.samples
    : wav.samples.filter((_, i) => i % wav.numChannels === 0);

  const nBlocks = Math.ceil(mono.length / B);
  const padded = new Float32Array(nBlocks * B);
  padded.set(mono.subarray(0, mono.length));

  const rng = mulberry32(0);
  const out = new Float32Array(nBlocks * B);
  runBtn.disabled = true;
  for (let k = 0; k < nBlocks; k++) {
    const eps = new Float32Array(fl);
    const noise = new Float32Array(fl - ls);
    for (let i = 0; i < fl; i++) eps[i] = rng() * 2 - 1;
    for (let i = 0; i < fl - ls; i++) noise[i] = rng() * 2 - 1;
    out.set(
      await new Promise<Float32Array>((resolve, reject) => {
        const handler = (e: MessageEvent<LogMsg>) => {
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
        });
      }),
      k * B,
    );
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
  try {
    await awaitLoaded();
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

let rt: Awaited<ReturnType<typeof createRealtime>> | null = null;

const renderRt = () => {
  if (!rt) return;
  rtUnderruns.textContent = String(rt.underruns());
  rtBlocks.textContent = String(rt.blocksProcessed());
  rtAvg.textContent = `${rt.avgTurnaroundMs().toFixed(1)} ms`;
  rtMax.textContent = `${rt.maxTurnaroundMs().toFixed(1)} ms`;
  rtLat.textContent = `${rt.latencyMs()} ms`;
};

rtStartBtn.onclick = async () => {
  try {
    rt ??= await createRealtime(log, renderRt);
    await rt.start();
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