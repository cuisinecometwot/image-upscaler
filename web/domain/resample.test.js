import assert from "node:assert/strict";
import { test } from "node:test";

import { boxDownscale, resizePlane } from "./resample.js";

test("resizePlane keeps a flat plane flat (weights are normalised)", () => {
  const src = new Float32Array(9 * 7).fill(0.6);
  for (const [dw, dh] of [
    [18, 14],
    [4, 3],
    [36, 28],
  ]) {
    const out = resizePlane(src, 9, 7, dw, dh);
    for (const v of out) assert.ok(Math.abs(v - 0.6) < 1e-5);
  }
});

test("resizePlane x2 downscale is shift-consistent, so tiles can be resized separately", () => {
  // A crop that starts on an even pixel resizes to the matching crop of the full result.
  const sw = 64,
    sh = 40;
  const src = Float32Array.from({ length: sw * sh }, (_, i) => Math.sin(i * 0.37) * 0.5 + 0.5);
  const full = resizePlane(src, sw, sh, sw / 2, sh / 2);
  const cx = 16,
    cy = 8,
    cw = 40,
    ch = 28; // crop in source pixels
  const crop = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) crop[y * cw + x] = src[(cy + y) * sw + cx + x];
  const part = resizePlane(crop, cw, ch, cw / 2, ch / 2);
  // Away from the crop border (Lanczos3 at x2 reaches 6 source = 3 output pixels).
  for (let y = 3; y < ch / 2 - 3; y++) {
    for (let x = 3; x < cw / 2 - 3; x++) {
      const a = part[y * (cw / 2) + x];
      const b = full[(cy / 2 + y) * (sw / 2) + cx / 2 + x];
      assert.ok(Math.abs(a - b) < 1e-5, `${x},${y}: ${a} vs ${b}`);
    }
  }
});

test("resizePlane writes into one channel of an interleaved buffer", () => {
  const out = new Uint8ClampedArray(4 * 4 * 4).fill(9);
  resizePlane(new Float32Array(4).fill(200), 2, 2, 4, 4, out, { stride: 4, offset: 3 });
  for (let i = 0; i < out.length; i++) assert.equal(out[i], i % 4 === 3 ? 200 : 9);
});

test("boxDownscale averages blocks and keeps partial edge blocks", () => {
  const rgba = new Uint8ClampedArray(3 * 1 * 4);
  rgba.set([0, 0, 0, 255, 100, 100, 100, 255, 50, 60, 70, 80]);
  const p = boxDownscale(rgba, 3, 1, 2);
  assert.deepEqual([p.width, p.height], [2, 1]);
  assert.deepEqual([...p.data], [50, 50, 50, 255, 50, 60, 70, 80]);
});
