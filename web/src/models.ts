import * as ort from "onnxruntime-web";
import {
  RavePipeline,
  adaptSession,
  OrtLike,
} from "./pipeline";
import { RaveManifest } from "./manifest";

export interface ModelUrls {
  encoder: string;
  decoder: string;
  manifest: string;
  warmup?: string;
}

export interface ModelBundle {
  manifest: RaveManifest;
  pipeline: RavePipeline;
}

export interface LoadOptions {
  threads?: number;
  buffers?: { encoder: ArrayBuffer; decoder: ArrayBuffer };
}

/** Absolute fetch URL from the worker/main origin. */
const abs = (p: string): string => new URL(p, self.location.origin).href;

export async function loadModelBundle(
  urls: ModelUrls,
  opts: LoadOptions = {},
): Promise<ModelBundle> {
  ort.env.wasm.wasmPaths = "";
  // Single-threaded WASM only: threads deadlock on load under COI and gain
  // little here (M2 measured ~30ms/buf single-threaded, < 46ms block budget).
  ort.env.wasm.numThreads = 1;
  const fetchText = async (p: string): Promise<string> => {
    const r = await fetch(abs(p));
    if (!r.ok) throw new Error(`fetch ${p}: HTTP ${r.status}`);
    return r.text();
  };
  const fetchBytes = async (p: string): Promise<ArrayBuffer> => {
    const r = await fetch(abs(p));
    if (!r.ok) throw new Error(`fetch ${p}: HTTP ${r.status}`);
    return r.arrayBuffer();
  };
  const [m, warm] = await Promise.all([
    fetchText(urls.manifest).then((t) => JSON.parse(t)) as Promise<RaveManifest>,
    urls.warmup ? fetchBytes(urls.warmup) : Promise.resolve(undefined),
  ]);
  const [enc, dec] = opts.buffers
    ? await Promise.all([
        ort.InferenceSession.create(opts.buffers.encoder, {
          executionProviders: ["wasm"],
        }).catch((err) => {
          throw new Error(`session create encoder: ${String(err)}`);
        }),
        ort.InferenceSession.create(opts.buffers.decoder, {
          executionProviders: ["wasm"],
        }).catch((err) => {
          throw new Error(`session create decoder: ${String(err)}`);
        }),
      ])
    : await Promise.all([
        ort.InferenceSession.create(abs(urls.encoder), {
          executionProviders: ["wasm"],
        }).catch((err) => {
          throw new Error(`session create ${urls.encoder}: ${String(err)}`);
        }),
        ort.InferenceSession.create(abs(urls.decoder), {
          executionProviders: ["wasm"],
        }).catch((err) => {
          throw new Error(`session create ${urls.decoder}: ${String(err)}`);
        }),
      ]);
  const pipeline = new RavePipeline(
    adaptSession(enc),
    adaptSession(dec),
    ort as unknown as OrtLike,
    {
      blockSize: m.block_size,
      ratio: m.ratio,
      latentSize: m.latent_size,
      fullLatentSize: m.full_latent_size,
      encCacheShapes: m.caches.encoder,
      decCacheShapes: m.caches.decoder,
      warmup: warm ? new Float32Array(warm) : undefined,
    },
  );
  return { manifest: m, pipeline };
}