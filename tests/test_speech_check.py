import pytest

from app.audio.forced_align import _spans_to_word_windows
from app.audio.source import AudioSource
from app.audio.speech_check import HeardSegment, HeardWord, find_heard_matches, is_duplicate_of_subtitle
from app.domain import Severity
from app.subtitles.matcher import CueMatch, MatchSpan, ProfanityMatcher, WordListTerm
from app.subtitles.parser import SubtitleCue


def _matcher(*terms):
    return ProfanityMatcher([WordListTerm(term=t, severity=Severity.teen, match_whole_word=True) for t in terms])


def _seg(*words):
    ws = tuple(HeardWord(t, s, s + 0.3, p) for t, s, p in words)
    return HeardSegment(ws[0].start, ws[-1].end, " ".join(w.text for w in ws), ws)


def test_heard_word_mapped_to_its_own_timestamp():
    seg = _seg(("We", 1248.0, 0.9), ("don't", 1248.2, 0.9), ("know", 1248.3, 0.9), ("what", 1248.4, 0.9),
               ("the", 1248.5, 0.9), ("hell", 1248.6, 0.99), ("it", 1248.9, 0.9), ("is.", 1249.0, 0.9))
    [hit] = find_heard_matches([seg], _matcher("hell"))
    assert hit.windows == (pytest.approx((1248.6, 1248.9)),)
    assert hit.match.cue.text == "We don't know what the hell it is."


def test_punctuation_and_case_dont_block_a_match_and_phrases_span_words():
    seg = _seg(("Jesus", 1243.1, 0.97), ("Christ.", 1243.5, 0.99))
    [hit] = find_heard_matches([seg], _matcher("jesus christ"))
    assert hit.windows == (pytest.approx((1243.1, 1243.8)),)


def test_low_probability_word_ignored():
    seg = _seg(("what", 10.0, 0.9), ("the", 10.2, 0.9), ("hell", 10.4, 0.1))
    assert find_heard_matches([seg], _matcher("hell")) == []


def test_whole_word_only():
    seg = _seg(("La", 992.0, 0.9), ("Rochelle", 992.3, 0.5))
    assert find_heard_matches([seg], _matcher("hell")) == []


def test_duplicate_detection_needs_same_term_and_time():
    seg = _seg(("what", 1082.0, 0.9), ("the", 1082.2, 0.9), ("hell's", 1082.5, 0.96))
    [hit] = find_heard_matches([seg], _matcher("hell"))
    sub = CueMatch(cue=SubtitleCue(1, 1081.2, 1084.3, "Artie, what the hell's wrong?"), matched_terms=("hell",), severity=Severity.teen)
    other_word = CueMatch(cue=sub.cue, matched_terms=("damn",), severity=Severity.teen)
    far_away = CueMatch(cue=SubtitleCue(2, 1200.0, 1202.0, "hell"), matched_terms=("hell",), severity=Severity.teen)
    assert is_duplicate_of_subtitle(hit, [sub])
    assert not is_duplicate_of_subtitle(hit, [other_word, far_away])


def test_low_confidence_alignment_widens_to_neighbors():
    words = [("artie", 0, 5), ("what", 6, 10), ("the", 11, 14), ("hell's", 15, 21), ("wrong", 22, 27)]
    times = [(0.3, 0.8), (0.8, 0.9), (1.1, 1.2), (1.2, 1.4), (1.45, 1.9)]
    span = (MatchSpan(term="hell", severity=Severity.teen, start=15, end=19),)
    [confident] = _spans_to_word_windows(words, times, span, 1080.0, [0.5, 0.5, 0.5, 0.8, 0.5])
    assert (confident.start, confident.end) == pytest.approx((1081.2, 1081.4))
    [guessing] = _spans_to_word_windows(words, times, span, 1080.0, [0.2, 0.0, 0.03, 0.10, 0.26])
    assert (guessing.start, guessing.end) == pytest.approx((1081.1, 1081.9))


def test_audio_source_uses_center_channel_on_surround_only():
    assert AudioSource(2, 6).ffmpeg_args() == ["-map", "0:2", "-af", "pan=mono|c0=FC"]
    assert AudioSource(1, 2).ffmpeg_args() == ["-map", "0:1", "-ac", "1"]
