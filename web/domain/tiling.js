// Tile planning for running the x4 model on images larger than the GPU can take at once.
// Pure JS, no browser APIs, so it is unit tested with `node --test`.

// Context pixels around each tile so seams match a whole-image run (ncnn uses 10 as well).
export const CONTEXT = 10;

// Mirror an out-of-range index back into [0, n), matching numpy/ncnn "reflect" padding.
export function mirrorIndex(i, n) {
  if (n === 1) return 0;
  const period = 2 * n - 2;
  const m = ((i % period) + period) % period;
  return m < n ? m : period - m;
}

// Every tile has the same input shape, so the GPU compiles its shaders only once.
// The tile edge is balanced (e.g. 1216px at max 256 -> 5 tiles of 244, not 256)
// so the last tile, shifted inward to stay in the image, barely overlaps its neighbour.
export function planTiles(width, height, tileSize, context = CONTEXT) {
  if (width < 1 || height < 1) throw new RangeError("image is empty");
  if (tileSize < 1) throw new RangeError("tile size must be positive");
  const balanced = (size) => Math.ceil(size / Math.ceil(size / tileSize));
  const tw = balanced(width);
  const th = balanced(height);
  const starts = (size, t) => {
    const out = [];
    for (let s = 0; s < size; s += t) out.push(Math.min(s, size - t));
    return out;
  };
  const tiles = [];
  for (const y of starts(height, th)) {
    for (const x of starts(width, tw)) {
      tiles.push({
        x,
        y,
        w: tw,
        h: th,
        inX: x - context,
        inY: y - context,
        inW: tw + 2 * context,
        inH: th + 2 * context,
      });
    }
  }
  return tiles;
}

// x2/x4 targets: one x4 AI pass, then Lanczos down to the exact size when needed.
export const MODEL_SCALE = 4;

export function targetSize(width, height, scale) {
  if (scale !== 2 && scale !== 4) throw new RangeError("scale must be 2 or 4");
  return { width: width * scale, height: height * scale };
}
