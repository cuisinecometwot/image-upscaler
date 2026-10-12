"""Pillow side of the pipeline: alpha handling, AI passes, exact resize, sharpening, save."""

import tempfile
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from upscaler import engine
from upscaler.domain.sizing import Model, Plan

SAVE_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def load(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    return img


def _split_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        # A fully opaque alpha would only cost time to carry through.
        if alpha.getextrema() == (255, 255):
            return rgba.convert("RGB"), None
        return rgba.convert("RGB"), alpha
    return img.convert("RGB"), None


def _ai_upscale(rgb: Image.Image, plan: Plan, model: Model, tile: int) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="upscale-") as tmp:
        current = Path(tmp) / "in.png"
        rgb.save(current, compress_level=1)
        for i, scale in enumerate(plan.passes):
            out = Path(tmp) / f"pass{i}.png"
            engine.run_pass(current, out, model.engine_name, scale, tile)
            current = out
        with Image.open(current) as result:
            return result.convert("RGB")


def sharpen(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return img
    return img.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=round(80 * strength), threshold=2)
    )


def upscale(img: Image.Image, plan: Plan, model: Model, tile: int = 0, sharpness: float = 0):
    rgb, alpha = _split_alpha(img)
    if plan.passes:
        rgb = _ai_upscale(rgb, plan, model, tile)
    if rgb.size != plan.output_size:
        rgb = rgb.resize(plan.output_size, Image.Resampling.LANCZOS)
    rgb = sharpen(rgb, sharpness)
    if alpha is None:
        return rgb
    rgb.putalpha(alpha.resize(plan.output_size, Image.Resampling.LANCZOS))
    return rgb


def save(img: Image.Image, path: Path) -> None:
    fmt = SAVE_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"unsupported output format: {path.suffix}")
    if fmt == "JPEG" and img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"JPEG": {"quality": 95, "subsampling": 0}, "WEBP": {"quality": 95}}
    img.save(path, fmt, **options.get(fmt, {}))
