import { makeRings, UNDERRUN, LIVE } from "./ring";
import { RaveManifest } from "./manifest";

export interface ModelUrls {
  encoder: string;
  decoder: string;
  manifest: string;
  warmup?: string;
}

export function crossOriginIsolated(): boolean {
  return typeof SharedArrayBuffer !== "undefined" &&
    "crossOriginIsolated" in self && self.crossOriginIsolated === true;
}

/** Latency estimate (ms): stability read-ahead (slack blocks) + worker
 * turnaround + device output latency. The stability slider trades latency for
 * spike headroom, so both numbers move together. */
export function estimateLatency(
  manifest: RaveManifest,
  blockMs: number,
  avgTurnaroundMs: number,
  extraMs: number,
  readAheadBlocks: number,
): number {
  const slackMs = readAheadBlocks * blockMs;
  return Math.round(slackMs + avgTurnaroundMs + extraMs);
}

/** Audio-engine controls settable from the UI while a session lives. */
export interface LiveControls {
  /** 0..1 blend of processed vs dry input. */
  setMix: (wet: number) => void;
  /** Input pre-gain (applied before EQ + model). */
  setInputGain: (g: number) => void;
  /** 3-band EQ: [lowShelf, peak, highShelf] gains in dB. */
  setEq: (db: [number, number, number]) => void;
  /** Per-latent-dim bias/scale + residual noise amount; pushed to the worker. */
  setLatent: (bias: number[], scale: number[], noiseGain: number) => void;
}

export interface Realtime extends LiveControls {
  start: () => Promise<void>;
  stop: () => void;
  running: () => boolean;
  underruns: () => number;
  blocksProcessed: () => number;
  avgTurnaroundMs: () => number;
  maxTurnaroundMs: () => number;
  latencyMs: () => number;
}

export interface CreateRealtimeOptions {
  log: (line: string, cls?: string) => void;
  tick: () => void;
  urls: ModelUrls;
  buffers?: { encoder: ArrayBuffer; decoder: ArrayBuffer };
  slackBlocks: number;
  onScope?: (z: Float32Array) => void;
  onError?: (message: string) => void;
}

export async function createRealtime(
  opts: CreateRealtimeOptions,
): Promise<Realtime> {
  const { log, tick, onScope, onError } = opts;
  const manifest: RaveManifest = await fetch(opts.urls.manifest).then((r) => r.json());
  // Ring cap = slack + 4 blocks of write-side guard so the worker never has to
  // stall while it processes.
  const ringBlocks = opts.slackBlocks + 4;
  const rings = makeRings(manifest.block_size, ringBlocks);
  const ctx = new AudioContext({
    sampleRate: manifest.sampling_rate,
    latencyHint: "interactive",
  });

  await ctx.audioWorklet.addModule("worklet/live-processor.js");
  const node = new AudioWorkletNode(ctx, "live-rave", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  node.port.postMessage(
    { control: rings.sabControl, rings: rings.sabData, cap: rings.cap },
  );

  // Input chain: mic -> gain -> EQ -> (dry path | node -> wet path).
  const gain = ctx.createGain();
  gain.gain.value = 1;
  const eqLow = ctx.createBiquadFilter();
  eqLow.type = "lowshelf"; eqLow.frequency.value = 200; eqLow.gain.value = 0;
  const eqMid = ctx.createBiquadFilter();
  eqMid.type = "peaking"; eqMid.frequency.value = 1000; eqMid.Q.value = 0.8; eqMid.gain.value = 0;
  const eqHigh = ctx.createBiquadFilter();
  eqHigh.type = "highshelf"; eqHigh.frequency.value = 4200; eqHigh.gain.value = 0;
  const dry = ctx.createGain();
  dry.gain.value = 0;
  const wet = ctx.createGain();
  wet.gain.value = 1;

  let source: MediaStreamAudioSourceNode | null = null;
  let stream: MediaStream | null = null;

  const rtWorker = new Worker(
    new URL("./realtime-worker.ts", import.meta.url),
    { type: "module" },
  );
  let _running = false;
  let connected = false;
  let blocks = 0;
  let avgMs = 0;
  let maxMs = 0;

  const connectOut = async () => {
    if (connected) return;
    connected = true;
    if (source) {
      source.connect(gain);
      gain.connect(eqLow);
      eqLow.connect(eqMid);
      eqMid.connect(eqHigh);
      eqHigh.connect(node); // wet path into the worklet
      eqHigh.connect(dry); // dry path straight out
      node.connect(wet);
      wet.connect(ctx.destination);
      dry.connect(ctx.destination);
    }
    await ctx.resume();
    Atomics.store(rings.control, LIVE, 1);
    _running = true;
    log("realtime running: mic -> eq -> model -> rings -> speakers");
    tick();
  };

  rtWorker.onmessage = (e) => {
    const d = e.data;
    if (d.type === "ready") {
      void connectOut();
    } else if (d.type === "metrics") {
      blocks = d.blocks;
      avgMs = d.avgMs;
      maxMs = d.maxMs;
      tick();
    } else if (d.type === "scope") {
      onScope?.(d.z);
    } else if (d.type === "rt-error") {
      onError?.(d.message);
      log(`realtime worker: ${d.message}`, "err");
    } else if (d.type === "rt-log") {
      log(`realtime worker: ${d.message}`);
    }
  };

  const latency = (): number => {
    const blockMs = (manifest.block_size / manifest.sampling_rate) * 1000;
    const extra =
      ctx.baseLatency * 1000 +
      (ctx.outputLatency ? ctx.outputLatency * 1000 : 10);
    return estimateLatency(manifest, blockMs, avgMs, extra, opts.slackBlocks);
  };

  const start = async (): Promise<void> => {
    if (_running) return;
    if (!crossOriginIsolated()) {
      throw new Error(
        "not cross-origin isolated (SharedArrayBuffer unavailable) — " +
          "serve with COOP/COEP headers",
      );
    }
    if (!source) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
      source = ctx.createMediaStreamSource(stream);
    }
    const fetchModel = async (p: string): Promise<ArrayBuffer> =>
      (await fetch(p, { cache: "no-store" })).arrayBuffer();
    // Preload on the main thread (worker-context fetches of the 60MB onnx hit
    // ERR_CACHE_WRITE_FAILURE under headless COI).
    const encFile = opts.urls.encoder;
    const decFile = opts.urls.decoder;
    const buffers = opts.buffers ?? {
      encoder: await fetchModel(encFile),
      decoder: await fetchModel(decFile),
    };
    rtWorker.postMessage({
      type: "start",
      control: rings.sabControl,
      rings: rings.sabData,
      cap: rings.cap,
      blockSize: manifest.block_size,
      slackBlocks: opts.slackBlocks,
      urls: opts.urls,
      buffers,
    }, [buffers.encoder, buffers.decoder]);
    log("realtime: loading + priming (output connects on ready)");
  };

  const stop = (): void => {
    if (!_running) return;
    rtWorker.postMessage({ type: "stop" });
    Atomics.store(rings.control, LIVE, 0);
    node.disconnect();
    source?.disconnect();
    gain.disconnect();
    eqLow.disconnect();
    eqMid.disconnect();
    eqHigh.disconnect();
    dry.disconnect();
    wet.disconnect();
    void ctx.suspend();
    connected = false;
    _running = false;
    log("realtime stopped");
  };

  return {
    running: () => _running,
    start,
    stop,
    underruns: () => Atomics.load(rings.control, UNDERRUN),
    blocksProcessed: () => blocks,
    avgTurnaroundMs: () => avgMs,
    maxTurnaroundMs: () => maxMs,
    latencyMs: latency,
    setMix: (wetValue: number) => {
      wet.gain.value = wetValue;
      dry.gain.value = 1 - wetValue;
    },
    setInputGain: (g: number) => {
      gain.gain.value = g;
    },
    setEq: ([low, mid, high]: [number, number, number]) => {
      eqLow.gain.value = low;
      eqMid.gain.value = mid;
      eqHigh.gain.value = high;
    },
    setLatent: (bias: number[], scale: number[], noiseGain: number) => {
      rtWorker.postMessage({ type: "params", bias, scale, noiseGain });
    },
  };
}
