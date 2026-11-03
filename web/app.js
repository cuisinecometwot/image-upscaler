import * as ort from "./vendor/ort.webgpu.min.mjs";
import { encodePng } from "./domain/png.js";
import { boxDownscale, resizePlane } from "./domain/resample.js";
import { MODEL_SCALE, mirrorIndex, planTiles, targetSize } from "./domain/tiling.js";

const MODEL_URLS = {
  fp32: new URL("./models/realesr-animevideov3-x4.onnx", import.meta.url).href,
  // Same weights stored/computed in half precision; float32 in/out so the pipeline is unchanged.
  fp16: new URL("./models/realesr-animevideov3-x4-fp16.onnx", import.meta.url).href,
};
// The full-res RGBA output buffer (4 bytes/px) has to fit in memory.
const MAX_OUTPUT_PIXELS = 80e6;
const WEBP_MAX_EDGE = 16383;
const PREVIEW_MAX_EDGE = 1600;

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(navigator.hardwareConcurrency || 4, 8)
  : 1;
ort.env.webgpu.powerPreference = "high-performance";

const $ = (id) => document.getElementById(id);
const ui = {
  env: $("env-list"),
  drop: $("drop"),
  dropText: $("drop-text"),
  file: $("file"),
  scale: $("scale"),
  backend: $("backend"),
  precision: $("precision"),
  tile: $("tile"),
  format: $("format"),
  run: $("run"),
  cancel: $("cancel"),
  progress: $("progress"),
  status: $("status"),
  reportCard: $("report-card"),
  report: $("report"),
  copy: $("copy"),
  resultCard: $("result-card"),
  download: $("download"),
  result: $("result"),
};

const state = {
  file: null,
  sessions: new Map(),
  env: {},
  cancelled: false,
  resultUrl: null,
  previewUrl: null,
};

// ---------- environment ----------

async function detectEnv() {
  const env = { webgpu: false, gpu: "-", f16: false };
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter) {
        const info = adapter.info || {};
        env.webgpu = true;
        env.gpu =
          [info.vendor, info.architecture, info.device, info.description]
            .filter(Boolean)
            .join(" / ") || "unknown";
        env.f16 = adapter.features.has("shader-f16");
      }
    } catch (e) {
      env.gpu = `adapter error: ${e.message}`;
    }
  }
  env.isolated = self.crossOriginIsolated;
  env.threads = ort.env.wasm.numThreads;
  env.ua = navigator.userAgent;
  state.env = env;

  const row = (k, v, cls = "") => `<dt>${k}</dt><dd class="${cls}">${v}</dd>`;
  ui.env.innerHTML =
    row(
      "WebGPU",
      env.webgpu ? "available" : "not available — CPU only (slow)",
      env.webgpu ? "good" : "bad",
    ) +
    row("GPU", env.gpu) +
    row("shader-f16", env.f16 ? "yes" : "no") +
    row("CPU threads", `${env.threads}${env.isolated ? "" : " (not cross-origin isolated)"}`) +
    row("Browser", env.ua);
  if (!env.webgpu) ui.backend.value = "wasm";
  if (!env.f16) ui.precision.value = "fp32";
}

// ---------- file input ----------

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|avif)$/i;

function isImage(file) {
  // Some OS/browser combos give dropped files an empty type.
  return file.type ? file.type.startsWith("image/") : IMAGE_EXT.test(file.name);
}

function setFile(file) {
  if (!file) return;
  if (!isImage(file)) {
    ui.status.textContent = `Not an image: ${file.name || file.type}`;
    return;
  }
  state.file = file;
  ui.dropText.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
  ui.status.textContent = "";
  ui.run.disabled = false;
}

// An image dragged from another tab usually arrives as HTML/URL, not as a file.
// Must run synchronously in the drop handler: dataTransfer is emptied afterwards.
function droppedUrl(dt) {
  const uriList = dt
    .getData("text/uri-list")
    .split(/\r?\n/)
    .find((l) => l && !l.startsWith("#"));
  // Prefer the <img> itself: for a linked image the uri-list is the link target.
  const html = dt.getData("text/html");
  const src = html && new DOMParser().parseFromString(html, "text/html").querySelector("img[src]");
  if (src) return new URL(src.getAttribute("src"), uriList || location.href).href;
  if (uriList) return uriList;
  const text = dt.getData("text/plain").trim();
  return /^(https?:|data:image\/)/.test(text) ? text : null;
}

async function fileFromUrl(url) {
  let blob;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    blob = await res.blob();
  } catch {
    // Almost always CORS: the other site doesn't let pages read its images.
    throw new Error(
      "That site doesn't allow reading the image from here. Copy the image and paste it (Ctrl+V), or save it and drop the file.",
    );
  }
  const name = url.startsWith("data:")
    ? "image"
    : decodeURIComponent(new URL(url).pathname.split("/").pop() || "image");
  return new File([blob], name, { type: blob.type });
}

ui.file.addEventListener("change", () => setFile(ui.file.files[0]));
for (const ev of ["dragenter", "dragover"]) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    ui.drop.classList.add("over");
  });
}
for (const ev of ["dragleave", "drop"]) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    ui.drop.classList.remove("over");
  });
}
document.addEventListener("drop", async (e) => {
  const file = e.dataTransfer.files[0];
  if (file) return setFile(file);
  const url = droppedUrl(e.dataTransfer);
  if (!url) return;
  ui.status.textContent = "Fetching dropped image…";
  try {
    setFile(await fileFromUrl(url));
  } catch (err) {
    ui.status.textContent = err.message;
  }
});
document.addEventListener("paste", (e) => {
  const file = [...e.clipboardData.files].find(isImage);
  if (file) setFile(file);
});

// ---------- pipeline ----------

async function getSession(backend, precision) {
  const key = `${backend}/${precision}`;
  if (!state.sessions.has(key)) {
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(MODEL_URLS[precision], {
      executionProviders: [backend],
      graphOptimizationLevel: "all",
    });
    state.sessions.set(key, { session, loadMs: performance.now() - t0 });
  }
  return state.sessions.get(key);
}

function readPixels(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
}

function hasAlpha(px) {
  for (let i = 3; i < px.length; i += 4) if (px[i] < 255) return true;
  return false;
}

// Builds the CHW float input for one tile; out-of-image context is mirrored like ncnn does.
function tileInput(px, W, H, t) {
  const plane = t.inW * t.inH;
  const data = new Float32Array(3 * plane);
  const cols = Int32Array.from({ length: t.inW }, (_, c) => mirrorIndex(t.inX + c, W));
  for (let r = 0; r < t.inH; r++) {
    const rowBase = mirrorIndex(t.inY + r, H) * W;
    for (let c = 0; c < t.inW; c++) {
      const s = (rowBase + cols[c]) * 4;
      const o = r * t.inW + c;
      data[o] = px[s] / 255;
      data[plane + o] = px[s + 1] / 255;
      data[2 * plane + o] = px[s + 2] / 255;
    }
  }
  return new ort.Tensor("float32", data, [1, 3, t.inH, t.inW]);
}

// Copies the tile's centre (context cropped off) into the full-size RGBA buffer. For x2 the
// tile is Lanczos-resized first; the 40px of x4 context covers the filter's reach, so
// tiles join without seams and no full x4 image is ever held in memory.
function writeTile(out, OW, y, t, scale) {
  const mw = t.inW * MODEL_SCALE;
  const mh = t.inH * MODEL_SCALE;
  let planes = [0, 1, 2].map((c) => y.subarray(c * mw * mh, (c + 1) * mw * mh));
  let pw = mw;
  if (scale !== MODEL_SCALE) {
    pw = t.inW * scale;
    planes = planes.map((p) => resizePlane(p, mw, mh, pw, t.inH * scale));
  }
  const [R, G, B] = planes;
  const offX = (t.x - t.inX) * scale;
  const offY = (t.y - t.inY) * scale;
  for (let r = 0; r < t.h * scale; r++) {
    const srcRow = (r + offY) * pw + offX;
    let d = ((t.y * scale + r) * OW + t.x * scale) * 4;
    for (let c = 0; c < t.w * scale; c++, d += 4) {
      const s = srcRow + c;
      // Uint8ClampedArray clamps and rounds for us.
      out[d] = R[s] * 255;
      out[d + 1] = G[s] * 255;
      out[d + 2] = B[s] * 255;
      out[d + 3] = 255;
    }
  }
}

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((res) => canvas.toBlob(res, type, quality));
}

// JPG/WebP have no streaming encoder here, so they need one full-size canvas.
async function encodeWithCanvas(out, w, h, format) {
  const canvas = makeCanvas(w, h);
  canvas.getContext("2d").putImageData(new ImageData(out, w, h), 0, 0);
  const blob = await canvasToBlob(canvas, format, 0.95);
  // Give the backing store back now; Safari otherwise keeps it until GC.
  canvas.width = canvas.height = 0;
  if (!blob || blob.type !== format) {
    throw new Error(
      `This browser can't encode a ${w}x${h} ${format.slice(6).toUpperCase()}. Choose PNG.`,
    );
  }
  return blob;
}

// Showing the full result in an <img> means decoding it again at full size, which is
// enough to kill a phone tab; show a small preview instead (Download has the full file).
async function makePreview(out, w, h) {
  const factor = Math.ceil(Math.max(w, h) / PREVIEW_MAX_EDGE);
  const p = factor > 1 ? boxDownscale(out, w, h, factor) : { data: out, width: w, height: h };
  const canvas = makeCanvas(p.width, p.height);
  canvas.getContext("2d").putImageData(new ImageData(p.data, p.width, p.height), 0, 0);
  return canvasToBlob(canvas, "image/png");
}

async function upscale(file, { scale, backend, precision, tileSize, format }, onProgress) {
  const times = {};
  let t0 = performance.now();
  const bitmap = await createImageBitmap(file);
  const W = bitmap.width,
    H = bitmap.height;
  const px = readPixels(bitmap);
  bitmap.close();
  const alpha = hasAlpha(px);
  const target = targetSize(W, H, scale);
  const OW = target.width,
    OH = target.height;
  if (OW * OH > MAX_OUTPUT_PIXELS) {
    throw new Error(`Image too large for this test page (${W}x${H} -> ${OW}x${OH}).`);
  }
  if (format === "image/webp" && Math.max(OW, OH) > WEBP_MAX_EDGE) {
    throw new Error(`WebP is limited to ${WEBP_MAX_EDGE}px per side (${OW}x${OH}). Choose PNG.`);
  }
  times.decode = performance.now() - t0;

  onProgress(0, "Loading model…");
  const { session, loadMs } = await getSession(backend, precision);
  times.modelLoad = loadMs;

  const tiles = planTiles(W, H, tileSize);
  const out = new Uint8ClampedArray(OW * OH * 4);
  const tileMs = [];
  t0 = performance.now();
  for (let i = 0; i < tiles.length; i++) {
    if (state.cancelled) throw new Error("Cancelled.");
    const t = tiles[i];
    const ts = performance.now();
    const input = tileInput(px, W, H, t);
    const { output } = await session.run({ input });
    writeTile(out, OW, output.data, t, scale);
    input.dispose();
    output.dispose();
    tileMs.push(performance.now() - ts);
    const avg =
      tileMs.length > 1
        ? tileMs.slice(1).reduce((a, b) => a + b, 0) / (tileMs.length - 1)
        : tileMs[0];
    const eta = Math.round((avg * (tiles.length - i - 1)) / 1000);
    onProgress((i + 1) / tiles.length, `AI pass: tile ${i + 1}/${tiles.length} · ~${eta}s left`);
  }
  times.inference = performance.now() - t0;

  t0 = performance.now();
  if (alpha) {
    // Alpha skips the AI: Lanczos-resize it on its own straight into the alpha channel.
    onProgress(1, "Resizing alpha…");
    const a = new Float32Array(W * H);
    for (let i = 0; i < a.length; i++) a[i] = px[i * 4 + 3];
    resizePlane(a, W, H, OW, OH, out, { stride: 4, offset: 3 });
  }
  times.resize = performance.now() - t0;

  onProgress(1, "Encoding…");
  t0 = performance.now();
  const blob =
    format === "image/png"
      ? await encodePng(out, OW, OH, { alpha })
      : await encodeWithCanvas(out, OW, OH, format);
  times.encode = performance.now() - t0;
  const preview = await makePreview(out, OW, OH);

  return { blob, preview, W, H, target, tiles, tileMs, times, alpha };
}

// ---------- report ----------

function report(r, opts) {
  const s = (ms) => `${(ms / 1000).toFixed(2)}s`;
  const rest = r.tileMs.slice(1);
  const avg = rest.length ? rest.reduce((a, b) => a + b, 0) / rest.length : r.tileMs[0];
  const total = r.times.decode + r.times.inference + r.times.resize + r.times.encode;
  const e = state.env;
  return [
    `Input        ${r.W}x${r.H}${r.alpha ? " (alpha)" : ""} -> ${r.target.width}x${r.target.height} (x${opts.scale})`,
    `Backend      ${opts.backend} ${opts.precision}${opts.backend === "wasm" ? ` (${e.threads} threads)` : ""}`,
    `GPU          ${e.gpu}`,
    `Tiles        ${r.tiles.length} x ${r.tiles[0].inW}x${r.tiles[0].inH} input`,
    `Model load   ${s(r.times.modelLoad)}  (first use per backend only)`,
    `First tile   ${s(r.tileMs[0])}  (includes shader compile)`,
    `Avg tile     ${s(avg)}`,
    `AI pass      ${s(r.times.inference)}`,
    `Resize       ${s(r.times.resize)}`,
    `Encode       ${s(r.times.encode)}  (${opts.format}, ${(r.blob.size / 1e6).toFixed(1)} MB)`,
    `Total        ${s(total)}  (excluding model load)`,
    `Browser      ${e.ua}`,
  ].join("\n");
}

// ---------- run ----------

ui.run.addEventListener("click", async () => {
  const opts = {
    scale: Number(ui.scale.value),
    backend: ui.backend.value,
    // fp16 math needs the GPU shader-f16 feature; the CPU backend would just emulate it slowly.
    precision: ui.backend.value === "webgpu" && state.env.f16 ? ui.precision.value : "fp32",
    tileSize: Number(ui.tile.value),
    format: ui.format.value,
  };
  state.cancelled = false;
  ui.run.disabled = true;
  ui.cancel.hidden = false;
  ui.progress.hidden = false;
  ui.progress.value = 0;
  try {
    const r = await upscale(state.file, opts, (p, msg) => {
      ui.progress.value = p;
      ui.status.textContent = msg;
    });
    ui.report.textContent = report(r, opts);
    ui.reportCard.hidden = false;
    for (const url of [state.resultUrl, state.previewUrl]) if (url) URL.revokeObjectURL(url);
    state.resultUrl = URL.createObjectURL(r.blob);
    state.previewUrl = URL.createObjectURL(r.preview);
    const ext = { "image/png": "png", "image/webp": "webp", "image/jpeg": "jpg" }[opts.format];
    ui.download.href = state.resultUrl;
    ui.download.textContent = `Download (${(r.blob.size / 1e6).toFixed(1)} MB)`;
    ui.result.src = state.previewUrl;
    ui.download.download = `${state.file.name.replace(/\.[^.]+$/, "")}_x${opts.scale}.${ext}`;
    ui.resultCard.hidden = false;
    ui.status.textContent = `Done in ${(r.times.inference / 1000).toFixed(1)}s AI time.`;
  } catch (e) {
    console.error(e);
    const hint =
      opts.backend === "webgpu" && !state.cancelled
        ? " Try the CPU backend or a smaller tile size."
        : "";
    ui.status.textContent = `Error: ${e.message}${hint}`;
  } finally {
    ui.run.disabled = false;
    ui.cancel.hidden = true;
    ui.progress.hidden = true;
  }
});

ui.cancel.addEventListener("click", () => {
  state.cancelled = true;
});
ui.copy.addEventListener("click", () => navigator.clipboard.writeText(ui.report.textContent));

detectEnv();
