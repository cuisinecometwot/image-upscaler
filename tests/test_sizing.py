import pytest

from upscaler.domain.sizing import MODELS, make_plan, parse_target, plan_passes, target_size

ANIME = MODELS["anime"]
PHOTO = MODELS["photo"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("x2", (2432, 1664)), ("x4", (4864, 3328)), ("2k", (2560, 1752)), ("4K", (3840, 2627))],
)
def test_target_size_for_sample_image(text, expected):
    assert target_size(1216, 832, parse_target(text)) == expected


def test_long_edge_uses_height_for_portrait():
    assert target_size(832, 1216, parse_target("4k")) == (2627, 3840)


@pytest.mark.parametrize("bad", ["x1", "x0.5", "8k", "", "xabc", "2"])
def test_parse_target_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_target(bad)


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(2, (2,)), (2.1, (3,)), (3, (3,)), (3.16, (4,)), (4, (4,)), (6, (4, 2)), (16, (4, 4))],
)
def test_anime_picks_smallest_sufficient_scale(factor, expected):
    assert plan_passes(factor, ANIME) == expected


@pytest.mark.parametrize(("factor", "expected"), [(2, (4,)), (4, (4,)), (5, (4, 4))])
def test_photo_only_has_x4(factor, expected):
    assert plan_passes(factor, PHOTO) == expected


def test_already_large_enough_needs_no_ai_pass():
    plan = make_plan(5000, 3000, parse_target("4k"), ANIME)
    assert plan.passes == ()
    assert plan.output_size == (3840, 2304)


def test_tiny_image_to_4k_chains_passes():
    plan = make_plan(300, 200, parse_target("4k"), ANIME)
    assert plan.passes == (4, 4)
    assert plan.ai_size_factor == 16
