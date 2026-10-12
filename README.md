# upscale

Local AI image upscaler (Real-ESRGAN, runs on any Vulkan GPU incl. Intel iGPUs).

```bash
uv sync
uv run upscale setup                       # one-time: downloads binary + models (~45 MB)
uv run upscale photo.png -t 4k             # long edge 3840px
uv run upscale input/ -t x2 -m anime-hq    # whole folder, best quality for art
```

| Option | Meaning |
|---|---|
| `-t` | `x2`, `x4`, any `xN`, `2k` (2560px long edge), `4k` (3840px). Default `x2` |
| `-m` | `anime` (default, fast), `anime-hq` (sharper, slower), `photo` |
| `-s` | extra sharpening `0`–`2` (default off) |
| `-f` | output format `png`/`jpg`/`webp` (default: same as input) |
| `-o` | output file or folder (default `./output`) |
| `--tile` | GPU tile size; lower (e.g. 128) if you get GPU memory errors |
| `--overwrite` | replace existing outputs (otherwise skipped) |

Install globally: `uv tool install .` then just `upscale ...`.
