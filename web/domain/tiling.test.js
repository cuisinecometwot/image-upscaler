import assert from "node:assert/strict";
import { test } from "node:test";

import { CONTEXT, mirrorIndex, planTiles, targetSize } from "./tiling.js";

test("mirrorIndex reflects without repeating the edge pixel", () => {
  assert.deepEqual(
    [-3, -2, -1, 0, 4, 5, 6, 7].map((i) => mirrorIndex(i, 5)),
    [3, 2, 1, 0, 4, 3, 2, 1],
  );
});

test("mirrorIndex handles overshoot beyond one image width", () => {
  for (let i = -20; i < 25; i++) {
    const m = mirrorIndex(i, 4);
    assert.ok(m >= 0 && m < 4, `${i} -> ${m}`);
  }
  assert.equal(mirrorIndex(7, 1), 0);
});

test("planTiles covers every pixel with same-shaped tiles", () => {
  const W = 1216,
    H = 832,
    T = 256;
  const tiles = planTiles(W, H, T);
  const covered = new Uint8Array(W * H);
  for (const t of tiles) {
    assert.equal(t.w, 244);
    assert.equal(t.h, 208);
    assert.ok(t.x >= 0 && t.x + t.w <= W && t.y >= 0 && t.y + t.h <= H);
    assert.equal(t.inX, t.x - CONTEXT);
    assert.equal(t.inW, t.w + 2 * CONTEXT);
    for (let y = t.y; y < t.y + t.h; y++) covered.fill(1, y * W + t.x, y * W + t.x + t.w);
  }
  assert.ok(covered.every((v) => v === 1));
  assert.equal(tiles.length, 5 * 4);
});

test("planTiles never exceeds the requested tile size", () => {
  for (const [W, H, T] of [
    [257, 513, 256],
    [1000, 1, 128],
    [4000, 3000, 384],
  ]) {
    for (const t of planTiles(W, H, T)) assert.ok(t.w <= T && t.h <= T, `${W}x${H}@${T}`);
  }
});

test("planTiles shrinks tiles to a small image", () => {
  assert.deepEqual(planTiles(100, 60, 256), [
    { x: 0, y: 0, w: 100, h: 60, inX: -10, inY: -10, inW: 120, inH: 80 },
  ]);
});

test("planTiles rejects bad input", () => {
  assert.throws(() => planTiles(0, 10, 256), RangeError);
  assert.throws(() => planTiles(10, 10, 0), RangeError);
});

test("targetSize supports x2 and x4 only", () => {
  assert.deepEqual(targetSize(1216, 832, 2), { width: 2432, height: 1664 });
  assert.throws(() => targetSize(10, 10, 3), RangeError);
});
