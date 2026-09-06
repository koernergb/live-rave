/* AudioWorklet processor (M3). Does NO inference — copies mic input into the
 * shared input ring, pops decoded audio from the output ring for every
 * quantum, crossfades to silence on underrun and counts it. Classic script
 * (no imports) so it can be addModule()'d; shared constants mirror ring.ts.
 */
const IW = 0;
const IR = 1;
const OW = 2;
const OR = 3;
const UNDERRUN = 4;
const WORK_EV = 5;
const LIVE = 6;
const QUANTUM = 128;

class LiveRaveProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ctrl = null;
    this.inRing = null;
    this.outRing = null;
    this.cap = 0;
    this.prev = new Float32Array(QUANTUM); // last good output (crossfade base)
    this.port.onmessage = (e) => {
      this.ctrl = new Int32Array(e.data.control);
      const data = new Float32Array(e.data.rings);
      this.cap = e.data.cap;
      this.inRing = data.subarray(0, this.cap);
      this.outRing = data.subarray(this.cap, 2 * this.cap);
      this.prev.fill(0);
    };
  }

  process(inputs, outputs) {
    if (!this.ctrl || !this.inRing || !this.outRing) {
      const out = outputs[0];
      if (out) for (let c = 0; c < out.length; c++) out[c].fill(0);
      return true;
    }

    const out = outputs[0][0];
    const inp = inputs[0] && inputs[0][0];

    // input ring: mic -> worker
    if (inp) {
      const iw = Atomics.load(this.ctrl, IW);
      const ir = Atomics.load(this.ctrl, IR);
      if (iw - ir < this.cap) {
        for (let i = 0; i < QUANTUM; i++) {
          this.inRing[(iw + i) % this.cap] = inp[i];
        }
        Atomics.store(this.ctrl, IW, iw + QUANTUM);
      }
    }

    // output ring: worker -> speakers
    const ow = Atomics.load(this.ctrl, OW);
    const orr = Atomics.load(this.ctrl, OR);
    if (ow - orr >= QUANTUM) {
      for (let i = 0; i < QUANTUM; i++) {
        const s = this.outRing[(orr + i) % this.cap];
        out[i] = s;
        this.prev[i] = s;
      }
      Atomics.store(this.ctrl, OR, orr + QUANTUM);
    } else {
      // underrun: short crossfade to silence from the last good output
      for (let i = 0; i < QUANTUM; i++) {
        const g = 1 - i / QUANTUM;
        out[i] = this.prev[i] * g;
        this.prev[i] = out[i];
      }
      // Count only once the main thread has declared the output LIVE, so the
      // metric covers sustained streaming rather than model-load init.
      if (Atomics.load(this.ctrl, LIVE)) {
        Atomics.add(this.ctrl, UNDERRUN, 1);
      }
    }

    Atomics.add(this.ctrl, WORK_EV, 1);
    Atomics.notify(this.ctrl, WORK_EV);
    return true;
  }
}

registerProcessor("live-rave", LiveRaveProcessor);