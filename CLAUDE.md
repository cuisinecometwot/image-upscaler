# image-upscaler

Local CLI (`upscale`) that enlarges images to x2/x4/2K/4K and makes them sharper using Real-ESRGAN AI models. Goal: a reliable personal tool for upscaling anime/illustration art on this laptop.

## Product
- User: the owner only, on a Linux laptop (Intel HD 620 iGPU, 8 GB RAM, no NVIDIA). One-off utility per batch; no accounts, UI or deployment.
- Problem: small/soft images look blurry when enlarged.
- Solution: AI upscale (Real-ESRGAN via ncnn-vulkan on the iGPU) overshooting to a native scale, then exact Lanczos resize to target, optional unsharp mask.
- Targets: `xN` (N > 1) or `2k`/`4k` = long edge 2560/3840, aspect ratio kept.
- Default content: anime/illustration (`-m anime`); `anime-hq` (sharper, ~6x slower) and `photo` available.
- Non-commercial, English CLI.

## Stack & architecture
- Python + uv, Pillow only; argparse CLI; ruff + pytest.
- Engine: `realesrgan-ncnn-vulkan` binary (Real-ESRGAN v0.2.5.0) installed by `upscale setup` into `~/.local/share/upscale` (override with `UPSCALE_HOME`). Not in git.
- `upscaler/domain/sizing.py`: pure target parsing + pass planning (unit tested). Everything else (`engine`, `imaging`, `installer`, `cli`) depends on it.
- Alpha is split off and Lanczos-resized; only RGB goes through the AI.

## Progress
- Done: v1 CLI — single/folder input, x2/x4/2k/4k, 3 models, alpha, png/jpg/webp, sharpen, skip-existing.
- Measured on HD 620 (1216x832 → 4K): anime ~23s, anime-hq ~136s.
- Next: (ideas) parallel batch, face-restoration for photos, progress for long passes.
