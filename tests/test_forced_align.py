import asyncio
from pathlib import Path

import pytest

from app.audio.forced_align import (
    WordWindow,
    _spans_to_word_windows,
    _tokenize_with_offsets,
    align_matches_for_cue,
)
from app.domain import Severity
from app.subtitles.matcher import ProfanityMatcher, WordListTerm
from app.subtitles.parser import SubtitleCue


def real_match(text, start, end, term="fuck", i=1):
    cue = SubtitleCue(index=i, start_seconds=start, end_seconds=end, text=text)
    matcher = ProfanityMatcher([WordListTerm(term=term, severity=Severity.teen)])
    return matcher.match_cue(cue)


def test_tokenize_with_offsets_basic():
    words = _tokenize_with_offsets("this is a test")
    assert words == [("this", 0, 4), ("is", 5, 7), ("a", 8, 9), ("test", 10, 14)]


def test_tokenize_with_offsets_handles_double_spaces():
    words = _tokenize_with_offsets("shit  right there")
    assert [w[0] for w in words] == ["shit", "right", "there"]
    # Offsets point at the real character positions, gaps and all -- not assuming
    # single-space-separated words.
    assert words[0] == ("shit", 0, 4)
    assert words[1] == ("right", 6, 11)


def test_spans_to_word_windows_single_word():
    match = real_match("this is a really long sentence ending in fuck", 100.0, 110.0)
    words = _tokenize_with_offsets("this is a really long sentence ending in fuck")
    # One fake aligned time per word, strictly increasing.
    word_times = [(float(i), float(i) + 0.5) for i in range(len(words))]

    windows = _spans_to_word_windows(words, word_times, match.spans, window_start=100.0)
    assert len(windows) == 1
    # "fuck" is the last word (index 8) -> word_times[8] = (8.0, 8.5), offset by window_start.
    assert windows[0] == WordWindow(start=108.0, end=108.5)


def test_spans_to_word_windows_multi_word_phrase_unions():
    match = real_match("get out of here", 0.0, 5.0, term="get out")
    assert len(match.spans) == 1
    words = _tokenize_with_offsets("get out of here")
    word_times = [(0.0, 0.4), (0.5, 0.9), (1.0, 1.2), (1.3, 1.6)]

    windows = _spans_to_word_windows(words, word_times, match.spans, window_start=0.0)
    assert len(windows) == 1
    # Union of "get" (0.0-0.4) and "out" (0.5-0.9).
    assert windows[0] == WordWindow(start=0.0, end=0.9)


def test_spans_to_word_windows_raises_when_no_word_covers_span():
    match = real_match("shit happens", 0.0, 2.0, term="shit")
    words: list[tuple[str, int, int]] = []  # no words at all
    with pytest.raises(ValueError):
        _spans_to_word_windows(words, [], match.spans, window_start=0.0)


def test_align_matches_for_cue_returns_none_without_spans():
    cue = SubtitleCue(index=1, start_seconds=1.0, end_seconds=2.0, text="x")
    from app.subtitles.matcher import CueMatch

    match = CueMatch(cue=cue, matched_terms=("x",), severity=Severity.teen)  # no spans
    result = asyncio.run(align_matches_for_cue(Path("/fake.mkv"), "ffmpeg", match))
    assert result is None


def test_align_matches_for_cue_falls_back_to_none_on_error(monkeypatch):
    match = real_match("this has some shit in it", 10.0, 12.0, term="shit")

    def _boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("app.audio.forced_align._align_blocking", _boom)
    result = asyncio.run(align_matches_for_cue(Path("/fake.mkv"), "ffmpeg", match))
    assert result is None


def test_align_matches_for_cue_success_passes_through(monkeypatch):
    match = real_match("this has some shit in it", 10.0, 12.0, term="shit")
    expected = [WordWindow(start=10.5, end=10.8)]

    captured = {}

    def _fake_align_blocking(ffmpeg_bin, video_path, window_start, window_end, words, spans):
        captured["window_start"] = window_start
        captured["window_end"] = window_end
        return expected

    monkeypatch.setattr("app.audio.forced_align._align_blocking", _fake_align_blocking)
    result = asyncio.run(align_matches_for_cue(Path("/fake.mkv"), "ffmpeg", match))

    assert result == expected
    # Window padding applied and clamped at 0 on the low side.
    assert captured["window_start"] == pytest.approx(max(0.0, 10.0 - 0.75))
    assert captured["window_end"] == pytest.approx(12.0 + 0.75)
