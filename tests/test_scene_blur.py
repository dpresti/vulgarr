from app.audio.mute import MuteInterval
from app.mux.scene_blur import build_blur_filter, sibling_edit_path
from pathlib import Path


def test_empty_intervals_is_passthrough():
    expr = build_blur_filter([], "0:v", "blurred")
    assert expr == "[0:v]null[blurred]"


def test_builds_between_expression():
    intervals = [MuteInterval(start=1.0, end=2.0)]
    expr = build_blur_filter(intervals, "0:v", "blurred")
    assert expr == "[0:v]boxblur=luma_radius=25:luma_power=3:chroma_radius=25:chroma_power=3:enable='between(t,1.000,2.000)'[blurred]"


def test_chains_stages_for_many_intervals():
    # Same batching-limit reasoning as build_volume_filter's own test -- ffmpeg's
    # expression parser breaks past ~80-90 between() terms in one enable=
    # expression, batched here in groups of 20 (_MAX_TERMS_PER_STAGE).
    intervals = [MuteInterval(start=float(i * 10), end=float(i * 10 + 1)) for i in range(45)]
    expr = build_blur_filter(intervals, "0:v", "blurred")
    stages = expr.split(";")
    assert len(stages) == 3  # 45 intervals / 20 per stage -> 3 stages
    for stage in stages:
        assert stage.count("between(") <= 20

    assert stages[0].startswith("[0:v]boxblur=")
    assert stages[-1].endswith("[blurred]")
    assert "[blurred_stage0]" in stages[0] and stages[1].startswith("[blurred_stage0]")
    assert "[blurred_stage1]" in stages[1] and stages[2].startswith("[blurred_stage1]")


def test_sibling_edit_path_naming():
    # Plex "Versions" naming convention (flat, same folder) -- confirmed during
    # planning as what actually groups a second file under one library item.
    result = sibling_edit_path(Path("/plex/Movies/Deadpool (2016)/Deadpool (2016) Bluray-1080p.mkv"))
    assert result == Path("/plex/Movies/Deadpool (2016)/Deadpool (2016) Bluray-1080p - Vulgarr Edit.mkv")


def test_sibling_edit_path_preserves_suffix():
    result = sibling_edit_path(Path("/plex/Movies/Foo/Foo (2020) WEBRip.mp4"))
    assert result.suffix == ".mp4"
    assert result.name == "Foo (2020) WEBRip - Vulgarr Edit.mp4"
