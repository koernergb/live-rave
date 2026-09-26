/// <reference lib="webworker" />
import { loadModelBundle, ModelUrls } from "./models";
import { RavePipeline } from "./pipeline";
import {
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
  blockSize: number;
  slackBlocks: number;
  urls: ModelUrls;
  buffers?: { encoder: ArrayBuffer; decoder: ArrayBuffer };
}
export interface RealtimeParamMsg {
  type: "params";
  bias?: number[];
  scale?: number[];
  noiseGain?: number;
}
export interface RealtimeStopMsg {
  type: "stop";
}
export type RealtimeMsg = RealtimeStartMsg | RealtimeParamMsg | RealtimeStopMsg;

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
let slack = 4;
let pipeline: RavePipeline | null = null;
let stopped = false;
let bias = new Float32Array(0);
let scale = new Float32Array(0);
let noiseGain = 1;

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
  if (m.type === "params") {
    if (m.bias) for (let i = 0; i < Math.min(m.bias.length, bias.length); i++) bias[i] = m.bias[i];
    if (m.scale) for (let i = 0; i < Math.min(m.scale.length, scale.length); i++) scale[i] = m.scale[i];
    if (m.noiseGain !== undefined) noiseGain = m.noiseGain;
    return;
  }
  if (m.type !== "start") return;
  stopped = false;
  ctrl = new Int32Array(m.control);
  data = new Float32Array(m.rings);
  cap = m.cap;
  if (!cap) cap = data.length / 2;
  slack = m.slackBlocks;
  bias = new Float32Array(0);
  scale = new Float32Array(0);
  noiseGain = 1;

  const bundle = await loadModelBundle(m.urls, {
    threads: 1,
    buffers: m.buffers,
  });
  pipeline = bundle.pipeline;
  if (stopped) return;

  const fl = bundle.manifest.full_latent_size;
  const ls = bundle.manifest.latent_size;
  const T = pipeline.latentSteps;
  const block = m.blockSize;
  bias = new Float32Array(ls).fill(0);
  scale = new Float32Array(ls).fill(1);

  let rng = mulberry32(0x5eed);
  const eps = new Float32Array(fl * T);
  const noise = new Float32Array((fl - ls) * T);
  const buf = new Float32Array(block);

  // Prime: warm a block so the output ring starts `slack` blocks ahead; first
  // connect never underruns. Read-ahead equals the stability slider (+ the
  // worker's fixed processing lag).
  for (let i = 0; i < fl * T; i++) eps[i] = 0;
  for (let i = 0; i < (fl - ls) * T; i++) noise[i] = rng() * 2 - 1;
  const { y: primeY } = await pipeline.process(buf, eps, noise, {
    bias,
    scale,
  });
  for (let b = 0; b < slack; b++) {
    writeRing(data as Float32Array, cap, b * block, primeY, 0, block);
  }
  Atomics.store(ctrl as Int32Array, OW, slack * block);

  post({ type: "ready", ok: true });
  post({ type: "rt-log", message: `models loaded + primed (slack=${slack})` });

  let blocks = 0;
  let sumMs = 0;
  let maxMs = 0;
  let lastEv = 0;

  while (!stopped) {
    const iw = Atomics.load(ctrl as Int32Array, IW);
    const ir = Atomics.load(ctrl as Int32Array, IR);
    const ow = Atomics.load(ctrl as Int32Array, OW);
    const orr = Atomics.load(ctrl as Int32Array, OR);
    if (iw - ir < block || ow - orr >= cap - block + 1) {
      lastEv = Atomics.load(ctrl as Int32Array, WORK_EV);
      Atomics.wait(ctrl as Int32Array, WORK_EV, lastEv);
      continue;
    }

    const t0 = Date.now();
    readRing(data as Float32Array, cap, ir, block, buf, 0);
    // Sampling latent: tightened around the mean (zero eps) by default.
    for (let i = 0; i < fl * T; i++) eps[i] = 0;
    for (let i = 0; i < (fl - ls) * T; i++) noise[i] = (rng() * 2 - 1) * noiseGain;

    const { y, z } = await pipeline.process(buf, eps, noise, { bias, scale });
    writeRing(data as Float32Array, cap, ow, y, 0, block);
    Atomics.store(ctrl as Int32Array, OW, ow + block);
    Atomics.store(ctrl as Int32Array, IR, ir + block);

    post({ type: "scope", z: z.slice() });

    const ms = Date.now() - t0;
    blocks++;
    sumMs += ms;
    if (ms > maxMs) maxMs = ms;
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
    message: `loop ended: total=${blocks}`,
  });
};
