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

## Windows GUI (portable exe)

`gui/` is a Tauri app: add files or a folder (or drag & drop), pick an output folder,
choose **x2/x4** and **Normal/Anime**, press Start. Outputs are named `<name>_x2.<ext>`
and never overwrite existing files.

Get the exe: GitHub → Actions → **Windows GUI** → latest run → artifact `Upscaler-portable`
(`Upscaler.exe`, ~45 MB, engine and models built in). Needs Windows 10/11 (WebView2) and a GPU with
Vulkan drivers. First launch unpacks the engine into `%LOCALAPPDATA%\Upscaler`.

Develop / build locally:

```bash
cd gui
scripts/fetch-engine.sh windows   # or `ubuntu`; fills gui/engine/ (embedded at build time)
npm ci
cargo test -p upscaler-core       # pipeline rules
npm run tauri build -- --no-bundle   # on Windows → target/release/upscaler-gui.exe
# Pipeline smoke test without the GUI (Linux engine):
cargo run --release -p upscaler-core --example upscale -- <file|folder> <out_dir> anime x2
```
