"""Target parsing and upscale planning. Pure standard library."""

from dataclasses import dataclass

# Float factors like 3840/1216 must not tip into the next native scale through rounding noise.
_EPS = 1e-6

LONG_EDGE_PRESETS = {"2k": 2560, "4k": 3840}


@dataclass(frozen=True)
class Model:
    key: str
    engine_name: str
    native_scales: tuple[int, ...]


MODELS = {
    "anime": Model("anime", "realesr-animevideov3", (2, 3, 4)),
    "anime-hq": Model("anime-hq", "realesrgan-x4plus-anime", (4,)),
    "photo": Model("photo", "realesrgan-x4plus", (4,)),
}


@dataclass(frozen=True)
class Target:
    label: str
    scale: float | None = None
    long_edge: int | None = None


@dataclass(frozen=True)
class Plan:
    output_size: tuple[int, int]
    passes: tuple[int, ...]

    @property
    def ai_size_factor(self) -> int:
        factor = 1
        for s in self.passes:
            factor *= s
        return factor


def parse_target(text: str) -> Target:
    t = text.strip().lower()
    if t in LONG_EDGE_PRESETS:
        return Target(label=t, long_edge=LONG_EDGE_PRESETS[t])
    if t.startswith("x"):
        try:
            scale = float(t[1:])
        except ValueError:
            scale = 0
        if scale > 1:
            return Target(label=t, scale=scale)
    raise ValueError(f"invalid target {text!r}: use x2, x4 (any xN > 1), 2k or 4k")


def target_size(width: int, height: int, target: Target) -> tuple[int, int]:
    if target.scale is not None:
        factor = target.scale
    else:
        factor = target.long_edge / max(width, height)
    return max(1, round(width * factor)), max(1, round(height * factor))


def plan_passes(factor: float, model: Model) -> tuple[int, ...]:
    """Pick native AI scales whose product is >= factor, overshooting as little as possible.

    Overshoot is intended: the result is Lanczos-downscaled to the exact size, which stays crisp.
    """
    passes: list[int] = []
    remaining = factor
    while remaining > 1 + _EPS:
        largest = max(model.native_scales)
        if remaining > largest + _EPS:
            passes.append(largest)
            remaining /= largest
            continue
        passes.append(min(s for s in model.native_scales if s >= remaining - _EPS))
        break
    return tuple(passes)


def make_plan(width: int, height: int, target: Target, model: Model) -> Plan:
    out_w, out_h = target_size(width, height, target)
    factor = max(out_w / width, out_h / height)
    return Plan(output_size=(out_w, out_h), passes=plan_passes(factor, model))
