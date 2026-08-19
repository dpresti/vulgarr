from app.domain import Severity
from app.processing import check_subtitle_video_duration_match, effective_severity_levels
from app.subtitles.parser import SubtitleCue


def cue(start, end):
    return SubtitleCue(index=1, start_seconds=start, end_seconds=end, text="fuck")


def test_matched_duration_is_fine():
    cues = [cue(10, 15), cue(100, 105)]
    assert check_subtitle_video_duration_match(cues, video_duration=120) is None


def test_subtitle_ending_well_before_video_is_fine():
    # Normal: trailing credits with no dialogue after the last subtitle cue.
    cues = [cue(10, 15), cue(100, 105)]
    assert check_subtitle_video_duration_match(cues, video_duration=7000) is None


def test_subtitle_extending_past_video_is_flagged():
    # A subtitle timed for a longer (e.g. extended/director's) cut than this video file.
    cues = [cue(10, 15), cue(8000, 8010)]
    error = check_subtitle_video_duration_match(cues, video_duration=7160)
    assert error is not None
    assert "mismatched" in error
    assert "7160" in error


def test_small_overrun_within_tolerance_is_fine():
    # A few seconds over is within container-duration rounding slop, not a real mismatch.
    cues = [cue(10, 15), cue(100, 105)]
    assert check_subtitle_video_duration_match(cues, video_duration=100) is None


def test_no_cues_is_fine():
    assert check_subtitle_video_duration_match([], video_duration=7160) is None


def test_zero_video_duration_is_skipped():
    # Can't meaningfully compare against an unknown/zero duration -- don't false-flag.
    cues = [cue(8000, 8010)]
    assert check_subtitle_video_duration_match(cues, video_duration=0) is None


def test_single_severity_mp4_uses_that_level():
    levels = effective_severity_levels("/plex/Movie.mp4", [Severity.teen])
    assert levels == [Severity.teen]


def test_multi_severity_mkv_keeps_all_levels():
    levels = effective_severity_levels("/plex/Movie.mkv", [Severity.child, Severity.teen])
    assert levels == [Severity.child, Severity.teen]


def test_multi_severity_mp4_falls_back_to_most_inclusive():
    # child is more inclusive than teen (catches everything vs only teen-tagged words),
    # and comes first in canonical order -- that's the one that should survive.
    levels = effective_severity_levels("/plex/Movie.mp4", [Severity.teen, Severity.child])
    assert levels == [Severity.child]


def test_no_severity_levels_defaults_to_child():
    assert effective_severity_levels("/plex/Movie.mp4", None) == [Severity.child]
