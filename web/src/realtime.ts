import { makeRings, UNDERRUN, LIVE } from "./ring";
import { RaveManifest } from "./manifest";

interface ModelUrls {
  encoder: string;
  decoder: string;
  manifest: string;
  warmup?: string;
}

const MODEL_URLS: ModelUrls = {
  encoder: "models/encoder.onnx",
  decoder: "models/decoder.onnx",
  manifest: "models/manifest.json",
};
const ENCODER_FILE = "encoder.onnx";
const DECODER_FILE = "decoder.onnx";

export function crossOriginIsolated(): boolean {
  return typeof SharedArrayBuffer !== "undefined" &&
    "crossOriginIsolated" in self && self.crossOriginIsolated === true;
}

/** Latency estimate (ms): ring slack + worker turnaround + device output
 * latency. Rings are RING_BLOCKS blocks each; block = manifest.block_size. */
export function estimateLatency(
  manifest: RaveManifest,
  blockMs: number,
  avgTurnaroundMs: number,
  extraMs: number,
): number {
  const blocks = 2; // output ring read-ahead held during streaming
  const slackMs = blocks * blockMs;
  return Math.round(slackMs + avgTurnaroundMs + extraMs);
}

export interface Realtime {
  start: () => Promise<void>;
  stop: () => void;
  running: () => boolean;
  underruns: () => number;
  blocksProcessed: () => number;
  avgTurnaroundMs: () => number;
  maxTurnaroundMs: () => number;
  latencyMs: () => number;
}

export async function createRealtime(
  log: (line: string, cls?: string) => void,
  tick: () => void,
): Promise<Realtime> {
  const rings = makeRings();
  const ctx = new AudioContext({
    sampleRate: 44100,
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

  const gain = ctx.createGain();
  gain.gain.value = 1;
  let source: MediaStreamAudioSourceNode | null = null;
  let stream: MediaStream | null = null;

  const rtWorker = new Worker(
    new URL("./realtime-worker.ts", import.meta.url),
    { type: "module" },
  );
  let manifest: RaveManifest | null = null;
  const loadManifest = async () =>
    manifest ??= await fetch("models/manifest.json").then((r) => r.json());

  let _running = false;
  let connected = false;
  let blocks = 0;
  let avgMs = 0;
  let maxMs = 0;

  const connectOut = async () => {
    if (connected) return;
    connected = true;
    if (source && stream) {
      source.connect(gain);
      gain.connect(node);
    }
    node.connect(ctx.destination);
    await ctx.resume();
    Atomics.store(rings.control, LIVE, 1);
    _running = true;
    log("realtime running: mic -> rings -> worker -> rings -> speakers");
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
    } else if (d.type === "rt-error") {
      log(`realtime worker: ${d.message}`, "err");
    } else if (d.type === "rt-log") {
      log(`realtime worker: ${d.message}`);
    }
  };

  const latency = (): number => {
    void rings;
    if (!manifest) return 0;
    const blockMs = (manifest.block_size / manifest.sampling_rate) * 1000;
    // device output buffer + processing quantum contribution
    const extra =
      ctx.baseLatency * 1000 +
      (ctx.outputLatency ? ctx.outputLatency * 1000 : 10);
    return estimateLatency(manifest, blockMs, avgMs, extra);
  };

  return {
    running: () => _running,
    async start() {
      if (_running) return;
      if (!crossOriginIsolated()) {
        throw new Error(
          "not cross-origin isolated (SharedArrayBuffer unavailable) — " +
            "serve with COOP/COEP headers",
        );
      }
      manifest = await loadManifest();
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
      // NOTE: the input graph (source→gain→node→destination) is NOT wired
      // until ready — an input edge alone would activate the worklet and
      // underrun while models load.
      const fetchModel = async (p: string): Promise<ArrayBuffer> =>
        (await fetch("models/" + p, { cache: "no-store" })).arrayBuffer();
      // Preload on the main thread (proven reliable under headless COI; the
      // worker's own fetch of the 60MB models hits ERR_CACHE_WRITE_FAILURE).
      const buffers = {
        encoder: await fetchModel(ENCODER_FILE),
        decoder: await fetchModel(DECODER_FILE),
      };
      rtWorker.postMessage({
        type: "start",
        control: rings.sabControl,
        rings: rings.sabData,
        cap: rings.cap,
        urls: MODEL_URLS,
        buffers,
      }, [buffers.encoder, buffers.decoder]);
      log("realtime: loading + priming (output connects on ready)");
    },
    stop() {
      if (!_running) return;
      rtWorker.postMessage({ type: "stop" });
      Atomics.store(rings.control, LIVE, 0);
      node.disconnect();
      source?.disconnect();
      gain.disconnect();
      void ctx.suspend();
      connected = false;
      _running = false;
      log("realtime stopped");
    },
    underruns: () => Atomics.load(rings.control, UNDERRUN),
    blocksProcessed: () => blocks,
    avgTurnaroundMs: () => avgMs,
    maxTurnaroundMs: () => maxMs,
    latencyMs: latency,
  };
}