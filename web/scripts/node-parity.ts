/** Headless parity: runs web/src/pipeline.ts against the exported ONNX graphs
 * with onnxruntime-node, on the reference bundle from python/gen_reference.py.
 * Proves the shared pipeline logic is bit-comparable to the Python reference
 * before it ever runs in a browser. */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { RavePipeline } from "../src/pipeline";
import type { RaveManifest } from "../src/manifest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const models = resolve(root, "public", "models");
const parity = resolve(root, "public", "parity");

const f32 = (p: string): Float32Array => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

async function main(): Promise<number> {
  const manifest = JSON.parse(
    readFileSync(resolve(models, "manifest.json"), "utf8"),
  ) as RaveManifest;

  const warmup = f32(resolve(models, "warmup.bin"));
  const enc = await ort.InferenceSession.create(
    resolve(models, "encoder.onnx"),
    { executionProviders: ["cpu"] },
  );
  const dec = await ort.InferenceSession.create(
    resolve(models, "decoder.onnx"),
    { executionProviders: ["cpu"] },
  );

  const pipeline = new RavePipeline(enc, dec, ort, {
    blockSize: manifest.block_size,
    latentSize: manifest.latent_size,
    fullLatentSize: manifest.full_latent_size,
    encCacheShapes: manifest.caches.encoder,
    decCacheShapes: manifest.caches.decoder,
    warmup,
  });

  const audio = f32(resolve(parity, "audio.bin"));
  const eps = f32(resolve(parity, "eps.bin"));
  const noise = f32(resolve(parity, "noise.bin"));
  const refY = f32(resolve(parity, "ref_y.bin"));

  const B = manifest.block_size;
  const fl = manifest.full_latent_size;
  const ls = manifest.latent_size;
  const n = audio.length / B;

  let maxerr = 0;
  let bad = 0;
  const t0 = performance.now();
  for (let k = 0; k < n; k++) {
    const y = await pipeline.process(
      audio.subarray(k * B, (k + 1) * B),
      eps.subarray(k * fl, (k + 1) * fl),
      noise.subarray(k * (fl - ls), (k + 1) * (fl - ls)),
    );
    const ref = refY.subarray(k * B, (k + 1) * B);
    let m = 0;
    for (let i = 0; i < y.length; i++) {
      const d = Math.abs(y[i] - ref[i]);
      if (d > m) m = d;
    }
    if (m > 1e-4) bad++;
    if (m > maxerr) maxerr = m;
  }
  const ms = (performance.now() - t0) / n;

  const pass = maxerr < 1e-4;
  console.log(
    `node parity: n=${n} maxerr=${maxerr.toExponential(2)} ` +
      `bad=${bad} ${ms.toFixed(2)} ms/buf`,
  );
  console.log(pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

process.exit(await main());