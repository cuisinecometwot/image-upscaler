// Lanczos3 resampling on float planes, done in plain JS so large images never need a
// full-size canvas (phones crash or silently fail on those).

const A = 3;

function sinc(x) {
  if (x === 0) return 1;
  const p = Math.PI * x;
  return Math.sin(p) / p;
}

const lanczos = (x) => (x > -A && x < A ? sinc(x) * sinc(x / A) : 0);

// Per output index: first source index and normalised weights (edge indices clamped).
function axisWeights(srcSize, dstSize) {
  const ratio = srcSize / dstSize;
  const support = A * Math.max(ratio, 1);
  const taps = Math.ceil(support) * 2 + 1;
  const first = new Int32Array(dstSize);
  const weights = new Float32Array(dstSize * taps);
  const index = new Int32Array(dstSize * taps);
  const stretch = Math.max(ratio, 1);
  for (let i = 0; i < dstSize; i++) {
    const center = (i + 0.5) * ratio - 0.5;
    const start = Math.floor(center - support) + 1;
    first[i] = start;
    let sum = 0;
    for (let k = 0; k < taps; k++) {
      const w = lanczos((start + k - center) / stretch);
      weights[i * taps + k] = w;
      index[i * taps + k] = Math.min(Math.max(start + k, 0), srcSize - 1);
      sum += w;
    }
    for (let k = 0; k < taps; k++) weights[i * taps + k] /= sum;
  }
  return { taps, weights, index };
}

// Resizes a single-channel float plane. The result goes to `out[offset + i * stride]`,
// so it can be written straight into one channel of an RGBA buffer.
export function resizePlane(src, sw, sh, dw, dh, out, { stride = 1, offset = 0 } = {}) {
  const h = axisWeights(sw, dw);
  const v = axisWeights(sh, dh);
  const mid = new Float32Array(dw * sh);
  for (let y = 0; y < sh; y++) {
    const row = y * sw;
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      const base = x * h.taps;
      for (let k = 0; k < h.taps; k++) acc += src[row + h.index[base + k]] * h.weights[base + k];
      mid[y * dw + x] = acc;
    }
  }
  const result = out ?? new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const base = y * v.taps;
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      for (let k = 0; k < v.taps; k++) acc += mid[v.index[base + k] * dw + x] * v.weights[base + k];
      result[offset + (y * dw + x) * stride] = acc;
    }
  }
  return result;
}

// Small preview of an RGBA buffer by averaging factor x factor blocks.
export function boxDownscale(rgba, w, h, factor) {
  const pw = Math.ceil(w / factor);
  const ph = Math.ceil(h / factor);
  const out = new Uint8ClampedArray(pw * ph * 4);
  for (let py = 0; py < ph; py++) {
    const y1 = Math.min((py + 1) * factor, h);
    for (let px = 0; px < pw; px++) {
      const x1 = Math.min((px + 1) * factor, w);
      let r = 0,
        g = 0,
        b = 0,
        a = 0,
        n = 0;
      for (let y = py * factor; y < y1; y++) {
        for (let x = px * factor; x < x1; x++) {
          const i = (y * w + x) * 4;
          r += rgba[i];
          g += rgba[i + 1];
          b += rgba[i + 2];
          a += rgba[i + 3];
          n++;
        }
      }
      const o = (py * pw + px) * 4;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
      out[o + 3] = a / n;
    }
  }
  return { data: out, width: pw, height: ph };
}
