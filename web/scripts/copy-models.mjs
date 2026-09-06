#!/usr/bin/env node
// Copy exported artifacts from benchmarks/export into web/public/models/.
// The .onnx files are gitignored (CC-BY-NC-4.0); this regenerates them locally.
import { copyFileSync, mkdirSync, cp } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const src = resolve(root, "benchmarks", "export");
const dst = resolve(root, "web", "public", "models");

mkdirSync(dst, { recursive: true });
for (const f of ["encoder.onnx", "decoder.onnx", "manifest.json"]) {
  copyFileSync(resolve(src, f), resolve(dst, f));
  console.log(`copied ${f}`);
}
const w = resolve(root, "benchmarks", "browser-parity", "warmup.bin");
copyFileSync(w, resolve(dst, "warmup.bin"));
console.log("copied warmup.bin");

// Browser parity reference bundle -> public/parity/
const pdst = resolve(root, "web", "public", "parity");
mkdirSync(pdst, { recursive: true });
for (const f of ["audio.bin", "eps.bin", "noise.bin", "ref_y.bin"]) {
  copyFileSync(resolve(root, "benchmarks", "browser-parity", f),
    resolve(root, "web", "public", "parity", f));
}
console.log("copied parity bundle");