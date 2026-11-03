import assert from "node:assert/strict";
import { test } from "node:test";
import { inflateSync } from "node:zlib";

import { crc32, encodePng } from "./png.js";

// Minimal decoder for what encodePng writes (8-bit RGB/RGBA, any filter).
function decodePng(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  let off = 8;
  let width, height, channels;
  const idat = [];
  while (off < bytes.length) {
    const len = view.getUint32(off);
    const type = String.fromCharCode(...bytes.subarray(off + 4, off + 8));
    const data = bytes.subarray(off + 8, off + 8 + len);
    const crc = view.getUint32(off + 8 + len);
    assert.equal(crc, (crc32(bytes.subarray(off + 4, off + 8 + len)) ^ 0xffffffff) >>> 0, type);
    if (type === "IHDR") {
      width = view.getUint32(off + 8);
      height = view.getUint32(off + 12);
      channels = data[9] === 6 ? 4 : 3;
    } else if (type === "IDAT") idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    for (let i = 0; i < stride; i++) {
      const x = raw[y * (stride + 1) + 1 + i];
      const a = i >= channels ? out[y * stride + i - channels] : 0;
      const b = y > 0 ? out[(y - 1) * stride + i] : 0;
      const c = i >= channels && y > 0 ? out[(y - 1) * stride + i - channels] : 0;
      const p = a + b - c;
      const pa = Math.abs(p - a),
        pb = Math.abs(p - b),
        pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f];
      out[y * stride + i] = (x + pred) & 0xff;
    }
  }
  return { width, height, channels, pixels: out };
}

function randomRgba(w, h, seed = 1) {
  const px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < px.length; i++) px[i] = (seed = (seed * 1103515245 + 12345) >>> 0) >>> 24;
  return px;
}

test("crc32 matches the PNG spec check value", () => {
  const bytes = new TextEncoder().encode("123456789");
  assert.equal((crc32(bytes) ^ 0xffffffff) >>> 0, 0xcbf43926);
});

test("RGBA round-trips through encodePng", async () => {
  const w = 37,
    h = 70; // more rows than one write batch
  const px = randomRgba(w, h);
  const blob = await encodePng(px, w, h, { alpha: true });
  assert.equal(blob.type, "image/png");
  const png = decodePng(new Uint8Array(await blob.arrayBuffer()));
  assert.deepEqual([png.width, png.height, png.channels], [w, h, 4]);
  assert.deepEqual(png.pixels, new Uint8Array(px));
});

test("alpha=false writes RGB and drops the alpha byte", async () => {
  const w = 5,
    h = 3;
  const px = randomRgba(w, h, 7);
  const png = decodePng(
    new Uint8Array(await (await encodePng(px, w, h, { alpha: false })).arrayBuffer()),
  );
  assert.equal(png.channels, 3);
  const rgb = px.filter((_, i) => i % 4 !== 3);
  assert.deepEqual(png.pixels, new Uint8Array(rgb));
});
