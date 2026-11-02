import * as ort from "./vendor/ort.webgpu.min.mjs";
import { MODEL_SCALE, mirrorIndex, planTiles, targetSize } from "./domain/tiling.js";

const MODEL_URLS = {
  fp32: new URL("./models/realesr-animevideov3-x4.onnx", import.meta.url).href,
  // Same weights stored/computed in half precision; float32 in/out so the pipeline is unchanged.
  fp16: new URL("./models/realesr-animevideov3-x4-fp16.onnx", import.meta.url).href,
};
// Browsers refuse canvases much beyond this; the full-res RGBA buffer also has to fit in memory.
const MAX_AI_PIXELS = 80e6;

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

const state = { file: null, sessions: new Map(), env: {}, cancelled: false, resultUrl: null };

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

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.file = file;
  ui.dropText.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
  ui.run.disabled = false;
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
document.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

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

// Copies the tile's centre (context cropped off) into the full-size RGBA buffer.
function writeTile(out, OW, y, t) {
  const S = MODEL_SCALE;
  const oW = t.inW * S;
  const plane = oW * t.inH * S;
  const offX = (t.x - t.inX) * S;
  const offY = (t.y - t.inY) * S;
  for (let r = 0; r < t.h * S; r++) {
    const srcRow = (r + offY) * oW + offX;
    let d = ((t.y * S + r) * OW + t.x * S) * 4;
    for (let c = 0; c < t.w * S; c++, d += 4) {
      const s = srcRow + c;
      // Uint8ClampedArray clamps and rounds for us.
      out[d] = y[s] * 255;
      out[d + 1] = y[plane + s] * 255;
      out[d + 2] = y[2 * plane + s] * 255;
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

async function lanczos(from, w, h) {
  const to = makeCanvas(w, h);
  await pica().resize(from, to, { filter: "lanczos3" });
  return to;
}

async function upscale(file, { scale, backend, precision, tileSize, format }, onProgress) {
  const times = {};
  let t0 = performance.now();
  const bitmap = await createImageBitmap(file);
  const W = bitmap.width,
    H = bitmap.height;
  const px = readPixels(bitmap);
  const alpha = hasAlpha(px);
  const OW = W * MODEL_SCALE,
    OH = H * MODEL_SCALE;
  if (OW * OH > MAX_AI_PIXELS) {
    throw new Error(
      `Image too large for this test page (${W}x${H}; x4 pass would be ${OW}x${OH}).`,
    );
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
    writeTile(out, OW, output.data, t);
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

  onProgress(1, "Resizing & encoding…");
  t0 = performance.now();
  const aiCanvas = makeCanvas(OW, OH);
  aiCanvas.getContext("2d").putImageData(new ImageData(out, OW, OH), 0, 0);
  const target = targetSize(W, H, scale);
  let final = aiCanvas;
  if (target.width !== OW) final = await lanczos(aiCanvas, target.width, target.height);
  if (alpha) {
    // Alpha skips the AI: Lanczos-resize it as a grayscale image and put it back.
    const a = makeCanvas(W, H);
    const aData = new ImageData(W, H);
    for (let i = 0; i < px.length; i += 4) {
      aData.data[i] = aData.data[i + 1] = aData.data[i + 2] = px[i + 3];
      aData.data[i + 3] = 255;
    }
    a.getContext("2d").putImageData(aData, 0, 0);
    const big = (await lanczos(a, target.width, target.height))
      .getContext("2d")
      .getImageData(0, 0, target.width, target.height).data;
    const ctx = final.getContext("2d");
    const img = ctx.getImageData(0, 0, target.width, target.height);
    for (let i = 0; i < big.length; i += 4) img.data[i + 3] = big[i];
    ctx.putImageData(img, 0, 0);
  }
  times.resize = performance.now() - t0;

  t0 = performance.now();
  const blob = await new Promise((res, rej) =>
    final.toBlob((b) => (b ? res(b) : rej(new Error("Encoding failed"))), format, 0.95),
  );
  times.encode = performance.now() - t0;

  return { blob, W, H, target, tiles, tileMs, times, alpha };
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
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
    state.resultUrl = URL.createObjectURL(r.blob);
    const ext = { "image/png": "png", "image/webp": "webp", "image/jpeg": "jpg" }[opts.format];
    ui.download.href = ui.result.src = state.resultUrl;
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
