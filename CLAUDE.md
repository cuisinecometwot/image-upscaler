# image-upscaler

Local CLI (`upscale`) that enlarges images to x2/x4/2K/4K and makes them sharper using Real-ESRGAN AI models, plus a Windows GUI (`gui/`, portable exe). Goal: a reliable personal tool for upscaling anime/illustration art.

## Product
- User: the owner only, on a Linux laptop (Intel HD 620 iGPU, 8 GB RAM, no NVIDIA). One-off utility per batch; no accounts, UI or deployment.
- Problem: small/soft images look blurry when enlarged.
- Solution: AI upscale (Real-ESRGAN via ncnn-vulkan on the iGPU) overshooting to a native scale, then exact Lanczos resize to target, optional unsharp mask.
- Targets: `xN` (N > 1) or `2k`/`4k` = long edge 2560/3840, aspect ratio kept.
- Default content: anime/illustration (`-m anime`); `anime-hq` (sharper, ~6x slower) and `photo` available.
- Non-commercial, English CLI.
- Windows GUI: pick files/folder (or drag & drop) + output folder; only x2/x4 and Normal (`realesrgan-x4plus`) / Anime (`realesr-animevideov3`). Ships as one portable exe for testing on other machines; never overwrites outputs (`name_x2 (2).png`).

## Stack & architecture
- Python + uv, Pillow only; argparse CLI; ruff + pytest.
- Engine: `realesrgan-ncnn-vulkan` binary (Real-ESRGAN v0.2.5.0) installed by `upscale setup` into `~/.local/share/upscale` (override with `UPSCALE_HOME`). Not in git.
- `upscaler/domain/sizing.py`: pure target parsing + pass planning (unit tested). Everything else (`engine`, `imaging`, `installer`, `cli`) depends on it.
- Alpha is split off and Lanczos-resized; only RGB goes through the AI.
- GUI: Tauri v2 + React/Vite/TS in `gui/` (cargo workspace). `gui/core` = framework-free Rust crate (pipeline via `image` crate; `core/src/domain` pure std, unit tested); `gui/src-tauri` = commands/events only.
- GUI engine files live in `gui/engine/` (gitignored, `scripts/fetch-engine.sh`) and are embedded by `core/build.rs`, extracted to `%LOCALAPPDATA%\Upscaler\engine-v0.2.5.0` on first run.
- The exe is built only by GitHub Actions (`.github/workflows/windows-gui.yml`, windows-latest, `tauri build --no-bundle`); this laptop can't build it (no WebKitGTK; Rust is user-local in `~/.cargo`).

## Progress
- Done: v1 CLI — single/folder input, x2/x4/2k/4k, 3 models, alpha, png/jpg/webp, sharpen, skip-existing.
- Measured on HD 620 (1216x832 → 4K): anime ~23s, anime-hq ~136s.
- Done: Windows GUI (Tauri) — pipeline verified on Linux for all 4 mode/scale combos; Windows exe from CI.
- In progress: owner testing the exe on Windows.
- Next: (ideas) parallel batch, face-restoration for photos, progress for long passes.
