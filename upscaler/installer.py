"""Downloads the Real-ESRGAN ncnn-vulkan binary and its models."""

import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
)
BINARY_NAME = "realesrgan-ncnn-vulkan"


def install_dir() -> Path:
    if custom := os.environ.get("UPSCALE_HOME"):
        return Path(custom).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data_home) / "upscale"


def binary_path() -> Path:
    return install_dir() / BINARY_NAME


def models_dir() -> Path:
    return install_dir() / "models"


def is_installed() -> bool:
    return binary_path().is_file() and models_dir().is_dir()


def _report(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    pct = done * 100 // total
    # Redraw only on whole-percent changes so piped/captured output stays readable.
    if pct != (done - block_size) * 100 // total or done == total:
        sys.stderr.write(f"\r  {pct:3d}%  {done / 2**20:5.1f} / {total / 2**20:.1f} MiB")
        sys.stderr.flush()


def install(force: bool = False) -> Path:
    target = install_dir()
    if is_installed() and not force:
        print(f"Already installed at {target}")
        return target

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "realesrgan.zip"
        print(f"Downloading {RELEASE_URL}")
        urllib.request.urlretrieve(RELEASE_URL, archive, _report)
        sys.stderr.write("\n")

        extracted = Path(tmp) / "x"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)

        # The archive layout has changed between releases, so locate the binary instead of assuming.
        found = next(extracted.rglob(BINARY_NAME), None)
        if found is None:
            raise RuntimeError("binary not found in downloaded archive")
        src_root = found.parent

        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, binary_path())
        if models_dir().exists():
            shutil.rmtree(models_dir())
        shutil.copytree(src_root / "models", models_dir())

    mode = binary_path().stat().st_mode
    binary_path().chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed to {target}")
    return target
