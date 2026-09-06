/// <reference lib="webworker" />
import { RavePipeline } from "./pipeline";
import { RaveManifest } from "./manifest";
import { loadModelBundle } from "./models";

export interface LoadMsg {
  type: "load";
  urls: {
    encoder: string;
    decoder: string;
    manifest: string;
    warmup: string;
  };
  buffers?: { encoder: ArrayBuffer; decoder: ArrayBuffer };
}
export interface ProcessMsg {
  type: "process";
  x: Float32Array;
  eps: Float32Array;
  noise: Float32Array;
}
export interface ParityMsg {
  type: "parity";
  urls: {
    audio: string;
    eps: string;
    noise: string;
    refY: string;
  };
}
export interface ResetMsg {
  type: "reset";
}
export type WorkerMsg = LoadMsg | ProcessMsg | ParityMsg | ResetMsg;

let pipeline: RavePipeline | null = null;
let manifest: RaveManifest | null = null;
let busy = false;
const queue: WorkerMsg[] = [];

const post = (msg: unknown, transfer?: Transferable[]) =>
  (postMessage as (m: unknown, t?: Transferable[]) => void)(
    msg,
    transfer,
  );

async function load(msg: LoadMsg): Promise<void> {
  const bundle = await loadModelBundle(msg.urls, { buffers: msg.buffers });
  manifest = bundle.manifest;
  pipeline = bundle.pipeline;
  post({ ok: true, type: "loaded" });
}

async function parity(msg: ParityMsg): Promise<void> {
  if (!pipeline || !manifest) throw new Error("models not loaded");
  const [audio, eps, noise, refY] = await Promise.all([
    fetch(new URL(msg.urls.audio, self.location.origin).href).then((r) => r.arrayBuffer()),
    fetch(new URL(msg.urls.eps, self.location.origin).href).then((r) => r.arrayBuffer()),
    fetch(new URL(msg.urls.noise, self.location.origin).href).then((r) => r.arrayBuffer()),
    fetch(new URL(msg.urls.refY, self.location.origin).href).then((r) => r.arrayBuffer()),
  ]);
  const parityBundle = {
    audio: new Float32Array(audio),
    eps: new Float32Array(eps),
    noise: new Float32Array(noise),
    refY: new Float32Array(refY),
  };
  const n =
    parityBundle.audio.length / manifest.block_size;
  const ls = manifest.latent_size;
  const fl = manifest.full_latent_size;
  let maxerr = 0;
  let bad = 0;
  const t0 = performance.now();
  for (let k = 0; k < n; k++) {
    const x = parityBundle.audio.subarray(
      k * manifest.block_size,
      (k + 1) * manifest.block_size,
    );
    const epsK = parityBundle.eps.subarray(k * fl, (k + 1) * fl);
    const noiseK = parityBundle.noise.subarray(k * (fl - ls), (k + 1) * (fl - ls));
    const y = await pipeline.process(x, epsK, noiseK);
    const ref = parityBundle.refY.subarray(
      k * manifest.block_size,
      (k + 1) * manifest.block_size,
    );
    let m = 0;
    for (let i = 0; i < y.length; i++) {
      const d = Math.abs(y[i] - ref[i]);
      if (d > m) m = d;
    }
    if (m > 1e-4) bad++;
    if (m > maxerr) maxerr = m;
  }
  const ms = (performance.now() - t0) / n;
  post({ type: "parity", ok: true, n, maxerr, bad, msPerBuf: ms });
}

async function process(msg: ProcessMsg): Promise<void> {
  if (!pipeline) throw new Error("models not loaded");
  const y = await pipeline.process(msg.x, msg.eps, msg.noise);
  post({ type: "processed", y }, [y.buffer as Transferable]);
}

async function reset(): Promise<void> {
  if (!pipeline) throw new Error("no pipeline");
  pipeline.enc.reset();
  pipeline.dec.reset();
  post({ type: "reset", ok: true });
}

self.onmessage = async (e: MessageEvent<WorkerMsg>) => {
  queue.push(e.data);
  if (busy) return;
  busy = true;
  try {
    while (queue.length) {
      const m = queue.shift()!;
      if (m.type === "load") await load(m);
      else if (m.type === "parity") await parity(m);
      else if (m.type === "process") await process(m);
      else if (m.type === "reset") await reset();
    }
  } catch (err) {
    post({ type: "error", message: String(err) });
  } finally {
    busy = false;
  }
};