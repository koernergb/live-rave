#!/usr/bin/env node
// Copy exported artifacts into web/public/models/.
// - default model (v2-s0) from benchmarks/export -> public/models/ root
// - picker variants from benchmarks/models/<key> -> public/models/<key>/
// - browser parity reference bundle -> public/parity/
// - models.json catalog from benchmarks/models/ -> public/models/
// The .onnx / .warmup.bin artifacts are gitignored (CC-BY-NC-4.0); regenerate
// them with python/export_models.py + python/gen_reference.py.
import { copyFileSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const srcDefault = resolve(root, "benchmarks", "export");
const srcModels = resolve(root, "benchmarks", "models");
const dst = resolve(root, "web", "public", "models");

mkdirSync(dst, { recursive: true });

for (const f of ["encoder.onnx", "decoder.onnx", "manifest.json"]) {
  copyFileSync(resolve(srcDefault, f), resolve(dst, f));
  console.log(`copied ${f}`);
}
const w = resolve(srcDefault, "warmup.bin");
copyFileSync(w, resolve(dst, "warmup.bin"));
console.log("copied warmup.bin");

mkdirSync(resolve(srcModels), { recursive: true });
for (const key of readdirSync(srcModels)) {
  const dir = resolve(srcModels, key);
  if (!existsSync(dir) || key === "models.json") continue;
  const dd = resolve(dst, key);
  mkdirSync(dd, { recursive: true });
  for (const f of ["encoder.onnx", "decoder.onnx", "manifest.json", "warmup.bin"]) {
    copyFileSync(resolve(dir, f), resolve(dd, f));
  }
  console.log(`copied models/${key}/`);
}

copyFileSync(resolve(srcModels, "models.json"), resolve(dst, "models.json"));
console.log("copied models.json");

// Browser parity reference bundle -> public/parity/
const pdst = resolve(root, "web", "public", "parity");
mkdirSync(pdst, { recursive: true });
for (const f of ["audio.bin", "eps.bin", "noise.bin", "ref_y.bin"]) {
  copyFileSync(resolve(root, "benchmarks", "browser-parity", f),
    resolve(pdst, f));
}
console.log("copied parity bundle");