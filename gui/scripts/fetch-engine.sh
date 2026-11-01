#!/usr/bin/env bash
# Downloads Real-ESRGAN ncnn-vulkan v0.2.5.0 and copies only what the GUI needs into gui/engine/,
# which core/build.rs embeds into the executable.
# Usage: scripts/fetch-engine.sh [windows|ubuntu]   (default: windows)
set -euo pipefail

platform="${1:-windows}"
case "$platform" in
  windows) binary="realesrgan-ncnn-vulkan.exe"; extra=(vcomp140.dll vcomp140d.dll) ;;
  ubuntu) binary="realesrgan-ncnn-vulkan"; extra=() ;;
  *) echo "unknown platform: $platform" >&2; exit 2 ;;
esac
url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-${platform}.zip"

gui_dir="$(cd "$(dirname "$0")/.." && pwd)"
engine="$gui_dir/engine"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $url"
curl -fsSL -o "$tmp/engine.zip" "$url"
py="$(command -v python3 || command -v python)"
"$py" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$tmp/engine.zip" "$tmp/x"

rm -rf "$engine"
mkdir -p "$engine/models"
for f in "$binary" "${extra[@]}"; do
  cp "$tmp/x/$f" "$engine/"
done
for m in realesr-animevideov3-x2 realesr-animevideov3-x4 realesrgan-x4plus; do
  cp "$tmp/x/models/$m.param" "$tmp/x/models/$m.bin" "$engine/models/"
done
chmod +x "$engine/$binary"
echo "Engine ready in $engine"
