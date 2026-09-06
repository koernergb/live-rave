/** Manifest layout produced by python/parity.py (see benchmarks/export/manifest.json). */
export interface RaveManifest {
  seed: number;
  block_size: number;
  ratio: number;
  sampling_rate: number;
  latent_size: number;
  full_latent_size: number;
  caches: {
    encoder: number[][];
    decoder: number[][];
  };
}

/** A cache tensor shape is [1, ...dims]; dims[0] can be 0 (empty state). */
export interface CacheSpec {
  dims: number[];
}

export const cacheSpecs = (shapes: number[][]): CacheSpec[] =>
  shapes.map((dims) => ({ dims }));