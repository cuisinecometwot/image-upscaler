import argparse
import sys
import time
from pathlib import Path

from upscaler import imaging, installer
from upscaler.domain.sizing import MODELS, make_plan, parse_target
from upscaler.engine import EngineError

INPUT_EXTS = set(imaging.SAVE_FORMATS)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upscale",
        description="AI image upscaler (Real-ESRGAN). Run `upscale setup` once first.",
        epilog="examples:\n  upscale photo.png -t 4k\n  upscale input/ -t x2 -m anime-hq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="image file or folder of images")
    p.add_argument(
        "-t", "--target", default="x2", help="x2, x4 (any xN > 1), 2k or 4k (long edge 2560/3840)"
    )
    p.add_argument(
        "-m",
        "--model",
        choices=sorted(MODELS),
        default="anime",
        help="anime: fast, illustrations (default); anime-hq: slower, sharper lines; "
        "photo: real-world photos",
    )
    p.add_argument("-o", "--output", type=Path, help="output file or folder (default: ./output)")
    p.add_argument(
        "-f", "--format", choices=["png", "jpg", "webp"], help="output format (default: same)"
    )
    p.add_argument(
        "-s", "--sharpen", type=float, default=0, help="extra sharpening 0-2 (default 0 = off)"
    )
    p.add_argument("--tile", type=int, default=0, help="GPU tile size, 0 = auto; lower if OOM")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    return p


def _collect_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in INPUT_EXTS)
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"input not found: {path}")


def _output_path(src: Path, args, label: str, single: bool) -> Path:
    suffix = f".{args.format}" if args.format else src.suffix.lower()
    name = f"{src.stem}_{label}{suffix}"
    if args.output is None:
        return Path("output") / name
    if single and args.output.suffix:
        return args.output
    return args.output / name


def _run(args) -> int:
    target = parse_target(args.target)
    model = MODELS[args.model]
    inputs = _collect_inputs(args.input)
    if not installer.is_installed():
        print("error: Real-ESRGAN binary not found. Run `upscale setup` first.", file=sys.stderr)
        return 2
    if not inputs:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 1

    failures = 0
    for idx, src in enumerate(inputs, 1):
        dst = _output_path(src, args, target.label, single=len(inputs) == 1)
        prefix = f"[{idx}/{len(inputs)}] {src.name}"
        if dst.exists() and not args.overwrite:
            print(f"{prefix}: skip, {dst} exists (use --overwrite)")
            continue
        try:
            img = imaging.load(src)
            plan = make_plan(*img.size, target, model)
            w, h = img.size
            ow, oh = plan.output_size
            steps = " -> ".join(f"x{s}" for s in plan.passes) or "resize only"
            print(f"{prefix}: {w}x{h} -> {ow}x{oh} ({model.key}, {steps})", flush=True)
            if not plan.passes:
                print("  note: image already meets target size, no AI pass needed")
            started = time.monotonic()
            result = imaging.upscale(img, plan, model, args.tile, args.sharpen)
            imaging.save(result, dst)
            print(f"  saved {dst} in {time.monotonic() - started:.1f}s")
        except (OSError, ValueError, EngineError) as exc:
            failures += 1
            print(f"  error: {exc}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "setup":
        installer.install(force="--force" in argv)
        return 0

    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
