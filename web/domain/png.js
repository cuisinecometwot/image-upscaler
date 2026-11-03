// Streaming PNG encoder. canvas.toBlob needs a full-size canvas plus its own copy of the
// pixels, which is what crashed phones on big outputs; this compresses row batches as it
// goes and returns a Blob made of small chunks. Uses only web-standard globals
// (CompressionStream, Blob), which Node has too, so it is unit tested.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes, crc = 0xffffffff) {
  for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return crc;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, (crc32(out.subarray(4, 8 + data.length)) ^ 0xffffffff) >>> 0);
  return out;
}

const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const ROWS_PER_WRITE = 32;

// Paeth filter on every row: a good size/speed trade-off for illustrations and photos.
function filterRow(rgba, width, y, channels, out, o) {
  const bpp = channels;
  out[o] = 4;
  const row = y * width * 4;
  const up = row - width * 4;
  for (let x = 0; x < width; x++) {
    for (let c = 0; c < channels; c++) {
      const cur = rgba[row + x * 4 + c];
      const a = x > 0 ? rgba[row + (x - 1) * 4 + c] : 0;
      const b = y > 0 ? rgba[up + x * 4 + c] : 0;
      const d = x > 0 && y > 0 ? rgba[up + (x - 1) * 4 + c] : 0;
      const p = a + b - d;
      const pa = Math.abs(p - a);
      const pb = Math.abs(p - b);
      const pc = Math.abs(p - d);
      const pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : d;
      out[o + 1 + x * bpp + c] = cur - pred;
    }
  }
}

// rgba: Uint8(Clamped)Array, 4 bytes per pixel. alpha=false writes RGB only (smaller file).
export async function encodePng(rgba, width, height, { alpha = true } = {}) {
  const channels = alpha ? 4 : 3;
  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = alpha ? 6 : 2; // colour type RGBA / RGB
  const parts = [SIGNATURE, chunk("IHDR", ihdr)];

  const stream = new CompressionStream("deflate"); // zlib format, as IDAT expects
  const reader = stream.readable.getReader();
  const reading = (async () => {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      parts.push(chunk("IDAT", value));
    }
  })();

  const writer = stream.writable.getWriter();
  const stride = 1 + width * channels;
  for (let y0 = 0; y0 < height; y0 += ROWS_PER_WRITE) {
    const rows = Math.min(ROWS_PER_WRITE, height - y0);
    const buf = new Uint8Array(rows * stride);
    for (let r = 0; r < rows; r++) filterRow(rgba, width, y0 + r, channels, buf, r * stride);
    await writer.ready;
    writer.write(buf);
  }
  await writer.close();
  await reading;

  parts.push(chunk("IEND", new Uint8Array(0)));
  return new Blob(parts, { type: "image/png" });
}
