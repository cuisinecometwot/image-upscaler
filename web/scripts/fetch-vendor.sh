#!/usr/bin/env bash
# Download pinned third-party browser files into web/vendor (gitignored).
# Served from our own origin so they work under cross-origin isolation.
set -euo pipefail

ORT_VERSION=1.31.1
COI_VERSION=0.1.7

WEB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$WEB_DIR/vendor"
CDN=https://cdn.jsdelivr.net/npm
mkdir -p "$VENDOR"

fetch() { # url dest
  [ -s "$2" ] && return
  echo "fetch $(basename "$2")"
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

# The WebGPU build drives both the webgpu and wasm (CPU) backends.
for f in ort.webgpu.min.mjs ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm; do
  fetch "$CDN/onnxruntime-web@$ORT_VERSION/dist/$f" "$VENDOR/$f"
done
# Must sit next to index.html: a service worker only controls pages within its own path.
fetch "$CDN/coi-serviceworker@$COI_VERSION/coi-serviceworker.min.js" "$WEB_DIR/coi-serviceworker.min.js"
