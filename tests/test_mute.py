import asyncio
from pathlib import Path

import pytest

from app.audio.forced_align import WordWindow
from app.audio.mute import build_mute_intervals, build_mute_intervals_whisper, build_volume_filter
from app.domain import Severity
from app.subtitles.matcher import CueMatch, ProfanityMatcher, WordListTerm
from app.subtitles.parser import SubtitleCue


def make_match(start, end, i=1):
    cue = SubtitleCue(index=i, start_seconds=start, end_seconds=end, text="x")
    return CueMatch(cue=cue, matched_terms=("x",), severity=Severity.teen)


def real_match(text, start, end, term="fuck", i=1):
    """Build a CueMatch with real spans/normalized_length via the actual matcher,
    for precise-mode tests that depend on those fields being populated correctly."""
    cue = SubtitleCue(index=i, start_seconds=start, end_seconds=end, text=text)
    matcher = ProfanityMatcher([WordListTerm(term=term, severity=Severity.teen)])
    return matcher.match_cue(cue)


def test_pads_and_clamps_at_zero():
    intervals = build_mute_intervals([make_match(0.05, 1.0)], pad_seconds=0.15)
    assert intervals[0].start == 0.0  # clamped, not negative
    assert intervals[0].end == 1.15


def test_merges_close_intervals():
    intervals = build_mute_intervals(
        [make_match(1.0, 2.0, i=1), make_match(2.2, 3.0, i=2)],
        pad_seconds=0.0,
        merge_gap_seconds=0.25,
    )
    assert len(intervals) == 1
    assert intervals[0].start == 1.0
    assert intervals[0].end == 3.0


def test_keeps_far_apart_intervals_separate():
    intervals = build_mute_intervals(
        [make_match(1.0, 2.0, i=1), make_match(10.0, 11.0, i=2)],
        pad_seconds=0.0,
        merge_gap_seconds=0.25,
    )
    assert len(intervals) == 2


def test_empty_matches_yields_no_intervals():
    assert build_mute_intervals([]) == []


def test_volume_filter_empty_intervals_is_passthrough():
    expr = build_volume_filter([], "0:a:0", "clean")
    assert expr == "[0:a:0]anull[clean]"


def test_volume_filter_builds_between_expression():
    intervals = build_mute_intervals([make_match(1.0, 2.0)], pad_seconds=0.0)
    expr = build_volume_filter(intervals, "0:a:0", "clean")
    assert expr == "[0:a:0]volume=0:enable='between(t,1.000,2.000)'[clean]"


def test_volume_filter_chains_stages_for_many_intervals():
    # A high-profanity movie can have 80-90+ cues, which broke ffmpeg's expression
    # parser ("Cannot allocate memory") when crammed into one `enable=` expression.
    # Verify large interval counts get split into multiple chained volume stages
    # instead of one giant expression.
    matches = [make_match(float(i * 10), float(i * 10 + 1), i=i) for i in range(45)]
    intervals = build_mute_intervals(matches, pad_seconds=0.0, merge_gap_seconds=0.0)
    assert len(intervals) == 45

    expr = build_volume_filter(intervals, "0:a:0", "clean")
    stages = expr.split(";")
    assert len(stages) == 3  # 45 intervals / 20 per stage -> 3 stages
    for stage in stages:
        assert stage.count("between(") <= 20

    # First stage reads from the real source label, last stage writes the final output label.
    assert stages[0].startswith("[0:a:0]volume=")
    assert stages[-1].endswith("[clean]")
    # Intermediate stages chain together via matching labels.
    assert "[clean_stage0]" in stages[0] and stages[1].startswith("[clean_stage0]")
    assert "[clean_stage1]" in stages[1] and stages[2].startswith("[clean_stage1]")


def test_precise_mode_narrows_to_word_position():
    # "fuck" is the last word of a 10-second cue -- precise mode should mute near
    # the end, not the whole 10 seconds.
    match = real_match("This is a really really long sentence that ends in the word fuck", 100.0, 110.0)
    whole = build_mute_intervals([match], precise=False)
    precise = build_mute_intervals([match], precise=True)

    assert whole[0].start < 100.5 and whole[0].end > 109.5  # covers essentially the whole cue
    whole_span = whole[0].end - whole[0].start
    precise_span = precise[0].end - precise[0].start
    assert precise_span < whole_span / 2  # meaningfully narrower
    assert precise[0].start > 105.0  # positioned near the end of the cue, not the start


def test_precise_mode_falls_back_to_whole_cue_without_spans():
    # A CueMatch built without spans/normalized_length (e.g. hand-constructed, or a
    # future caller that doesn't populate them) should degrade to whole-cue muting
    # rather than producing a nonsensical zero-width interval.
    match = make_match(10.0, 12.0)
    intervals = build_mute_intervals([match], precise=True)
    assert intervals[0].start <= 10.0
    assert intervals[0].end >= 12.0


def test_precise_mode_handles_repeated_term_in_one_cue():
    # "shit" appears twice -- both occurrences should get their own narrow window,
    # not just the first.
    match = real_match("shit, that is some real shit right there", 50.0, 55.0, term="shit")
    assert len(match.spans) == 2
    intervals = build_mute_intervals([match], precise=True, merge_gap_seconds=0.0)
    assert len(intervals) == 2
    assert intervals[0].start < intervals[1].start


def test_whisper_mode_uses_aligned_word_window(monkeypatch):
    match = real_match("this has some shit in it", 10.0, 12.0, term="shit")

    async def fake_align(video_path, ffmpeg_bin, m):
        return [WordWindow(start=10.8, end=11.0)]

    monkeypatch.setattr("app.audio.forced_align.align_matches_for_cue", fake_align)
    intervals = asyncio.run(
        build_mute_intervals_whisper([match], Path("/fake.mkv"), "ffmpeg", pad_seconds=0.1)
    )
    assert len(intervals) == 1
    assert intervals[0].start == pytest.approx(10.7)
    assert intervals[0].end == pytest.approx(11.1)


def test_whisper_mode_falls_back_to_estimate_per_cue_on_alignment_failure(monkeypatch):
    # Alignment failing for one cue shouldn't fail the whole job -- it should fall
    # back to the proportional-estimate window for just that cue.
    match = real_match("this is a really really long sentence that ends in the word fuck", 100.0, 110.0)

    async def fake_align(video_path, ffmpeg_bin, m):
        return None

    monkeypatch.setattr("app.audio.forced_align.align_matches_for_cue", fake_align)
    whisper_intervals = asyncio.run(build_mute_intervals_whisper([match], Path("/fake.mkv"), "ffmpeg"))
    estimate_intervals = build_mute_intervals([match], precise=True, pad_seconds=0.4, merge_gap_seconds=0.0)

    assert whisper_intervals == estimate_intervals


def test_whisper_mode_empty_matches_yields_no_intervals():
    assert asyncio.run(build_mute_intervals_whisper([], Path("/fake.mkv"), "ffmpeg")) == []


def test_whisper_mode_reports_progress_per_cue(monkeypatch):
    matches = [
        real_match("this has some shit in it", 0.0, 2.0, term="shit", i=1),
        real_match("god damn it works", 3.0, 5.0, term="damn", i=2),
    ]

    async def fake_align(video_path, ffmpeg_bin, m):
        return None  # forces the fallback path -- progress should still fire

    monkeypatch.setattr("app.audio.forced_align.align_matches_for_cue", fake_align)

    calls = []

    async def on_progress(done, total):
        calls.append((done, total))

    asyncio.run(build_mute_intervals_whisper(matches, Path("/fake.mkv"), "ffmpeg", on_progress=on_progress))
    assert calls == [(1, 2), (2, 2)]
