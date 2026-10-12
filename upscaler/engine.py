"""Subprocess wrapper around realesrgan-ncnn-vulkan."""

import subprocess
from pathlib import Path

from upscaler import installer


class EngineError(RuntimeError):
    pass


def run_pass(src: Path, dst: Path, model_name: str, scale: int, tile: int = 0) -> None:
    if not installer.is_installed():
        raise EngineError("Real-ESRGAN binary not found. Run `upscale setup` first.")

    cmd = [
        str(installer.binary_path()),
        "-i", str(src),
        "-o", str(dst),
        "-n", model_name,
        "-s", str(scale),
        "-t", str(tile),
        "-m", str(installer.models_dir()),
        "-f", "png",
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True)
    # The binary can exit 0 without writing output (e.g. Vulkan init failure), so check the file.
    if result.returncode != 0 or not dst.is_file():
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise EngineError(
            f"Real-ESRGAN failed (exit {result.returncode}).\n{tail}\n"
            "Hint: try a smaller tile (--tile 128) or check Vulkan drivers (vulkaninfo)."
        )
