/** M5 benchmark cell: deterministic streaming RTF for one
 * (backend × block-size × model) cell, driven by e2e/bench.spec.ts.
 * Reuses the exact runtime pipeline (RavePipeline) — benchmark what ships. */

import { loadModelBundle, ModelUrls } from "../models";
import { RavePipeline } from "../pipeline";

export type BenchBackend = "wasm" | "wasm-threaded" | "webgpu";

export interface BenchResult {
  backend: BenchBackend;
  block: number;
  ratio: number;
  model: string;
  warmBlocks: number;
  measuredBlocks: number;
  warmupMs: number;
  coldStartMs: number;
  avgMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
  rtf: number;
  memDeltaMB: number;
  bytesLoadedMB: number;
  error?: string;
}

const mulberry32 = (seed: number) => {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const percentile = (sorted: number[], q: number): number => {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[i];
};

const memMB = (): number => {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number } })
    .memory;
  return m ? m.usedJSHeapSize / 1e6 : 0;
};

async function run(cfg: {
  base: string;
  block: number;
  backend: BenchBackend;
  model: string;
  warm: number;
  measure: number;
}): Promise<BenchResult> {
  const urls: ModelUrls = {
    encoder: `${cfg.base}encoder.onnx`,
    decoder: `${cfg.base}decoder.onnx`,
    manifest: `${cfg.base}manifest.json`,
    warmup: `${cfg.base}warmup.bin`,
  };
  const mem0 = memMB();
  const t0 = performance.now();

  let bundle;
  try {
    bundle = await loadModelBundle(urls, {
      provider: cfg.backend,
      threads: 4,
    });
  } catch (err) {
    return {
      backend: cfg.backend,
      block: cfg.block,
      ratio: 0,
      model: cfg.model,
      warmBlocks: cfg.warm,
      measuredBlocks: 0,
      warmupMs: performance.now() - t0,
      coldStartMs: 0,
      avgMs: 0, p50Ms: 0, p95Ms: 0, p99Ms: 0, maxMs: 0,
      rtf: 0,
      memDeltaMB: 0,
      bytesLoadedMB: 0,
      error: `session create failed: ${String(err).slice(0, 200)}`,
    };
  }

  const { manifest, pipeline } = bundle;
  const B = manifest.block_size;
  const fl = manifest.full_latent_size;
  const ls = manifest.latent_size;
  const T = pipeline.latentSteps;
  const blockMs = (B / manifest.sampling_rate) * 1000;

  // Deterministic stream across every cell: seeded audio + zero eps + seeded
  // residual noise, sliced per block.
  const N = cfg.warm + cfg.measure;
  const rng = mulberry32(0xabcd0000 ^ cfg.block);
  const audio = new Float32Array(N * B);
  for (let i = 0; i < audio.length; i++) audio[i] = rng() * 2 - 1;
  const eps = new Float32Array(fl * T);
  const noise = new Float32Array((fl - ls) * T);
  const nrng = mulberry32(0xbeef ^ cfg.block);
  for (let i = 0; i < noise.length; i++) noise[i] = nrng() * 2 - 1;

  const warmBlocks: number[] = [];
  const measured: number[] = [];
  for (let k = 0; k < N; k++) {
    const ts = performance.now();
    await pipeline.process(audio.subarray(k * B, (k + 1) * B), eps, noise);
    const el = performance.now() - ts;
    (k < cfg.warm ? warmBlocks : measured).push(el);
  }
  measured.sort((a, b) => a - b);
  const avg = measured.reduce((s, v) => s + v, 0) / measured.length;

  return {
    backend: cfg.backend,
    block: B,
    ratio: manifest.ratio,
    model: cfg.model,
    warmBlocks: cfg.warm,
    measuredBlocks: cfg.measure,
    warmupMs: performance.now() - t0 - avg,
    coldStartMs: performance.now() - t0,
    avgMs: avg,
    p50Ms: percentile(measured, 0.5),
    p95Ms: percentile(measured, 0.95),
    p99Ms: percentile(measured, 0.99),
    maxMs: measured[measured.length - 1] ?? 0,
    rtf: avg / blockMs,
    memDeltaMB: memMB() - mem0,
    bytesLoadedMB: (pipeline.enc.byteLength + pipeline.dec.byteLength) / 1e6,
  };
}

const out = document.getElementById("out")!;
const cfgEl = document.getElementById("cfg")!;

const q = new URLSearchParams(location.search);
const cfg = {
  base: q.get("base") ?? "/bench/2048/",
  block: Number(q.get("block") ?? 2048),
  backend: (q.get("backend") ?? "wasm") as BenchBackend,
  model: q.get("model") ?? "v2-live · seed 0",
  warm: Number(q.get("warm") ?? 3),
  measure: Number(q.get("measure") ?? 100),
};
cfgEl.textContent =
  `${cfg.model} · block ${cfg.block} (${(cfg.block / 44100 * 1000).toFixed(1)} ms) · ` +
  `${cfg.backend} · ${cfg.warm} warm + ${cfg.measure} measured`;

(async () => {
  const result = await run(cfg);
  (window as unknown as Record<string, unknown>).__benchResult = result;
  out.textContent = `${JSON.stringify(result, null, 2)}\n` +
    `RTF ${result.rtf.toFixed(3)} (${result.avgMs.toFixed(1)} ms / ` +
    `${(cfg.block / 44100 * 1000).toFixed(1)} ms window) — ` +
    `${result.rtf < 1 ? "REALTIME" : "NOT realtime"}`;
})();