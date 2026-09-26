/** Lock-free SPSC rings shared between the AudioWorklet and the realtime
 * worker. Indices are monotonic; the SPSC discipline is single-writer /
 * single-reader per direction:
 *
 *   input ring : worklet writes (IW), worker reads (IR)
 *   output ring: worker writes (OW), worklet reads (OR)
 *
 * All counters live in a SharedArrayBuffer-backed Int32Array view; audio data
 * lives in a second SAB as a Float32Array split into two equal rings.
 */

export const QUANTUM = 128; // AudioWorklet quantum
export const RING_BLOCKS = 8; // output slack: ~371 ms headroom over spikes

export const IW = 0; // input write index
export const IR = 1; // input read index
export const OW = 2; // output write index
export const OR = 3; // output read index
export const UNDERRUN = 4; // cumulative output underruns (worklet owns)
export const WORK_EV = 5; // worklet -> worker signal counter
export const LIVE = 6; // set by main when output connects (streaming gate)

export const CONTROL_LEN = 7;

export function ringCap(blockSize: number, blocks = RING_BLOCKS): number {
  return blockSize * blocks;
}

export interface Rings {
  sabControl: SharedArrayBuffer;
  sabData: SharedArrayBuffer;
  control: Int32Array;
  data: Float32Array;
  cap: number;
  blocks: number;
}

export function makeRings(blockSize: number, blocks = RING_BLOCKS): Rings {
  const cap = ringCap(blockSize, blocks);
  const sabControl = new SharedArrayBuffer(CONTROL_LEN * 4);
  const sabData = new SharedArrayBuffer(cap * 2 * 4);
  return {
    sabControl,
    sabData,
    control: new Int32Array(sabControl),
    data: new Float32Array(sabData),
    cap,
    blocks,
  };
}

/** Copy `n` interleaved samples out of a ring at monotonic position `pos`. */
export function readRing(
  ring: Float32Array,
  cap: number,
  pos: number,
  n: number,
  out: Float32Array,
  outOff = 0,
): void {
  for (let i = 0; i < n; i++) out[outOff + i] = ring[(pos + i) % cap];
}

/** Copy `n` samples into a ring at monotonic position `pos`. */
export function writeRing(
  ring: Float32Array,
  cap: number,
  pos: number,
  src: Float32Array,
  srcOff = 0,
  n = src.length,
): void {
  for (let i = 0; i < n; i++) ring[(pos + i) % cap] = src[srcOff + i];
}
