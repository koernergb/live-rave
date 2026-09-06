/** Minimal WAV decode/encode for 16-bit PCM and float32. */

export interface WavData {
  sampleRate: number;
  numChannels: number;
  /** Interleaved float32 samples in [-1, 1]. */
  samples: Float32Array;
}

const readString = (dv: DataView, off: number, len: number): string => {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(dv.getUint8(off + i));
  return s;
};

export function decodeWav(buf: ArrayBuffer): WavData {
  const dv = new DataView(buf);
  if (readString(dv, 0, 4) !== "RIFF" || readString(dv, 8, 4) !== "WAVE") {
    throw new Error("not a RIFF/WAVE file");
  }
  let off = 12;
  let fmtOff = -1;
  let dataOff = -1;
  let dataLen = 0;
  while (off + 8 <= buf.byteLength) {
    const id = readString(dv, off, 4);
    const size = dv.getUint32(off + 4, true);
    if (id === "fmt ") fmtOff = off + 8;
    else if (id === "data") {
      dataOff = off + 8;
      dataLen = size;
    }
    off += 8 + size + (size % 2);
  }
  if (fmtOff < 0 || dataOff < 0) throw new Error("missing fmt/data chunks");
  const audioFormat = dv.getUint16(fmtOff, true);
  const numChannels = dv.getUint16(fmtOff + 2, true);
  const sampleRate = dv.getUint32(fmtOff + 4, true);
  const bitsPerSample = dv.getUint16(fmtOff + 14, true);
  const n = dataLen / (bitsPerSample / 8);
  const samples = new Float32Array(n / numChannels);
  if (audioFormat === 1 && bitsPerSample === 16) {
    for (let i = 0; i < samples.length; i++) {
      samples[i] =
        dv.getInt16(dataOff + i * numChannels * 2, true) / 32768;
    }
  } else if (audioFormat === 1 && bitsPerSample === 32) {
    for (let i = 0; i < samples.length; i++) {
      samples[i] = dv.getFloat32(dataOff + i * 4, true);
    }
  } else if (audioFormat === 3) {
    for (let i = 0; i < samples.length; i++) {
      samples[i] = dv.getFloat32(dataOff + i * 4, true);
    }
  } else {
    throw new Error(`unsupported WAV format ${audioFormat}/${bitsPerSample}`);
  }
  return { sampleRate, numChannels, samples };
}

export function encodeWavPcm16(
  mono: Float32Array,
  sampleRate: number,
): ArrayBuffer {
  const n = mono.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const writeAscii = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
  };
  writeAscii(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  writeAscii(36, "data");
  dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, mono[i]));
    dv.setInt16(44 + i * 2, Math.round(v * 32767), true);
  }
  return buf;
}