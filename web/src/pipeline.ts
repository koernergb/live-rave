/** onnxruntime-agnostic streaming pipeline: threads stateful caches across
 * encoder/decoder invocations, exactly mirroring python/parity.py. Runs in a
 * browser Worker (onnxruntime-web) or Node (onnxruntime-node).
 */

export interface OrtTensor {
  data: Float32Array;
  dims: number[];
}

/** Minimal surface of an InferenceSession we rely on. */
export interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

export interface OrtLike {
  Tensor: new (data: Float32Array, dims: number[]) => OrtTensor;
}

/** Wraps a real onnxruntime InferenceSession (web or node) whose run() accepts
 * a richer value map. The pipeline only ever passes real ort.Tensor instances
 * built via ort.Tensor, so the cast is safe. */
export const adaptSession = <F, R>(
  s: { run: (feeds: F) => Promise<R> },
): OrtSession => ({
  run: (feeds) =>
    s.run(feeds as unknown as F) as unknown as Promise<
      Record<string, OrtTensor>
    >,
});

export interface RaveConfig {
  blockSize: number;
  latentSize: number;
  fullLatentSize: number;
  encCacheShapes: number[][];
  decCacheShapes: number[][];
  /** Consecutive float32 tiles: [encoder caches..., decoder caches...].
   * Omitted -> cold-start zeros. */
  warmup?: Float32Array;
}

/** One stateful graph's cache tensors, threaded out-of-place per block. */
export class StreamingGraph {
  readonly cacheShapes: number[][];
  readonly caches: Float32Array[];
  readonly byteLength: number;

  constructor(readonly shapes: number[][], warmup?: Float32Array) {
    this.cacheShapes = shapes;
    this.byteLength = shapes.reduce((s, dims) => s + product(dims), 0);
    this.caches = shapes.map((dims) => new Float32Array(product(dims)));
    if (warmup) this.seed(warmup);
  }

  seed(all: Float32Array): void {
    let off = 0;
    for (const c of this.caches) {
      c.set(all.subarray(off, off + c.length));
      off += c.length;
    }
  }

  reset(): void {
    for (const c of this.caches) c.fill(0);
  }
}

export const product = (dims: number[]): number =>
  dims.reduce((a, b) => a * b, 1);

export const encByteLength = (shapes: number[][]): number =>
  shapes.reduce((s, d) => s + product(d), 0);

/**
 * Threads the cached encoder + decoder graphs. Each process() call runs both
 * graphs on one block, feeding cache tensors as inputs and updating them from
 * the `cache_{i}_out` outputs.
 */
export class RavePipeline {
  readonly enc: StreamingGraph;
  readonly dec: StreamingGraph;

  constructor(
    private readonly encSess: OrtSession,
    private readonly decSess: OrtSession,
    private readonly ort: OrtLike,
    readonly cfg: RaveConfig,
  ) {
    this.enc = new StreamingGraph(
      cfg.encCacheShapes,
      cfg.warmup?.subarray(0, encByteLength(cfg.encCacheShapes)),
    );
    this.dec = new StreamingGraph(
      cfg.decCacheShapes,
      cfg.warmup?.subarray(encByteLength(cfg.encCacheShapes)),
    );
  }

  /** Run one block. x must be `blockSize` samples; eps/noise are single latent
   * time steps of length fullLatentSize / (fullLatentSize - latentSize).
   * Returns decoded y. */
  async process(
    x: Float32Array,
    eps: Float32Array,
    noise: Float32Array,
  ): Promise<Float32Array> {
    const T = this.ort.Tensor;
    const encOut = await this.encSess.run({
      x: new T(x.slice(), [1, 1, this.cfg.blockSize]),
      eps: new T(eps.slice(), [1, this.cfg.fullLatentSize, 1]),
      ...cacheFeeds(this.enc, this.cfg.encCacheShapes, T),
    });
    this.updateCaches(this.enc, encOut);

    const z = new Float32Array(encOut.z.data.subarray(0, this.cfg.latentSize));
    const decOut = await this.decSess.run({
      z: new T(z, [1, this.cfg.latentSize, 1]),
      noise: new T(noise.slice(), [
        1,
        this.cfg.fullLatentSize - this.cfg.latentSize,
        1,
      ]),
      ...cacheFeeds(this.dec, this.cfg.decCacheShapes, T),
    });
    this.updateCaches(this.dec, decOut);
    return new Float32Array(decOut.y.data);
  }

  private updateCaches(
    g: StreamingGraph,
    out: Record<string, OrtTensor>,
  ): void {
    g.caches.forEach((c, i) => {
      const t = out[`cache_${i}_out`];
      if (t) c.set(t.data.subarray(0, c.length));
    });
  }
}

function cacheFeeds(
  g: StreamingGraph,
  shapes: number[][],
  T: new (data: Float32Array, dims: number[]) => OrtTensor,
): Record<string, OrtTensor> {
  const feeds: Record<string, OrtTensor> = {};
  g.caches.forEach((c, i) => {
    // Exported graph keeps a baked batch dim: [1, C, L] (manifest strips it).
    feeds[`cache_${i}`] = new T(c.slice(), [1, ...shapes[i]]);
  });
  return feeds;
}