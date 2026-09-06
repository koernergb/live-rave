/// <reference lib="webworker" />
import { loadModelBundle, ModelUrls } from "./models";
import { RavePipeline } from "./pipeline";
import {
  BLOCK,
  IR,
  IW,
  OW,
  OR,
  UNDERRUN,
  WORK_EV,
  readRing,
  writeRing,
} from "./ring";

export interface RealtimeStartMsg {
  type: "start";
  control: SharedArrayBuffer;
  rings: SharedArrayBuffer;
  cap: number;
  urls: ModelUrls;
  buffers?: { encoder: ArrayBuffer; decoder: ArrayBuffer };
}
export interface RealtimeStopMsg {
  type: "stop";
}
export type RealtimeMsg = RealtimeStartMsg | RealtimeStopMsg;

const mulberry32 = (seed: number): (() => number) => {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

let ctrl: Int32Array | null = null;
let data: Float32Array | null = null;
let cap = 0;
let pipeline: RavePipeline | null = null;
let stopped = false;

const post = (msg: unknown) =>
  (postMessage as (m: unknown) => void)(msg);

self.addEventListener("error", (e) =>
  post({ type: "rt-error", message: String(e.message) }));
self.addEventListener("unhandledrejection", (e) =>
  post({
    type: "rt-error",
    message: `unhandledrejection: ${String(e.reason)}`,
  }));

self.onmessage = async (e: MessageEvent<RealtimeMsg>) => {
  const m = e.data;
  if (m.type === "stop") {
    stopped = true;
    post({ type: "stopped" });
    return;
  }
  if (m.type !== "start") return;
  stopped = false;
  ctrl = new Int32Array(m.control);
  data = new Float32Array(m.rings);
  cap = m.cap;
  if (!cap) cap = data.length / 2;

  const bundle = await loadModelBundle(m.urls, {
    threads: 1,
    buffers: m.buffers,
  });
  pipeline = bundle.pipeline;
  if (stopped) return;

  const fl = bundle.manifest.full_latent_size;
  const ls = bundle.manifest.latent_size;

  let rng = mulberry32(0x5eed);
  const eps = new Float32Array(fl);
  const noise = new Float32Array(fl - ls);
  const buf = new Float32Array(BLOCK);

  // Prime: run warmup blocks so the output ring starts ahead and first connect
  // never underruns (4 blocks ~186 ms read-ahead).
  const prime = await pipeline.process(buf, eps, noise);
  for (let b = 0; b < 4; b++) {
    writeRing(data as Float32Array, cap, b * BLOCK, prime, 0, BLOCK);
  }
  Atomics.store(ctrl as Int32Array, OW, 4 * BLOCK);

  post({ type: "ready", ok: true });
  post({ type: "rt-log", message: `models loaded + primed (cap=${cap})` });

  let blocks = 0;
  let sumMs = 0;
  let maxMs = 0;
  let lastEv = 0;
  let slowCount = 0;
  const slowSamples: number[] = [];

  while (!stopped) {
    const iw = Atomics.load(ctrl as Int32Array, IW);
    const ir = Atomics.load(ctrl as Int32Array, IR);
    const ow = Atomics.load(ctrl as Int32Array, OW);
    const orr = Atomics.load(ctrl as Int32Array, OR);
    if (iw - ir < BLOCK || ow - orr >= cap - BLOCK + 1) {
      // Need a full input block AND output space; park until the worklet
      // signals progress.
      lastEv = Atomics.load(ctrl as Int32Array, WORK_EV);
      Atomics.wait(ctrl as Int32Array, WORK_EV, lastEv);
      continue;
    }

    const t0 = Date.now();
    readRing(data as Float32Array, cap, ir, BLOCK, buf);
    // No-op latent manipulation for M3: zero eps (mean latent), fresh noise.
    for (let i = 0; i < fl; i++) eps[i] = 0;
    for (let i = 0; i < fl - ls; i++) noise[i] = rng() * 2 - 1;

    const y = await pipeline.process(buf, eps, noise);
    writeRing(data as Float32Array, cap, ow, y, 0, BLOCK);
    Atomics.store(ctrl as Int32Array, OW, ow + BLOCK);
    Atomics.store(ctrl as Int32Array, IR, ir + BLOCK);

    const ms = Date.now() - t0;
    blocks++;
    sumMs += ms;
    if (ms > maxMs) maxMs = ms;
    if (ms > 45) {
      slowCount++;
      if (slowSamples.length >= 20) slowSamples.shift();
      slowSamples.push(ms);
    }
    if (blocks % 2 === 0) {
      const underruns = Atomics.load(ctrl as Int32Array, UNDERRUN);
      post({
        type: "metrics",
        blocks,
        avgMs: sumMs / blocks,
        maxMs,
        underruns,
      });
    }
  }

  post({
    type: "rt-log",
    message: `loop ended: total=${blocks} slow(${slowCount}) samples=${slowSamples.join(",")}`,
  });
};