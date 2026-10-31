from app.processing import is_kept_match
from app.subtitles.matcher import CueMatch, normalize_cue_text
from app.subtitles.parser import SubtitleCue
from app.domain import Severity


def _match(start, text):
    return CueMatch(cue=SubtitleCue(index=1, start_seconds=start, end_seconds=start + 2, text=text), matched_terms=("shoot",), severity=Severity.child)


def test_kept_when_text_and_time_line_up():
    kept = [(412.0, normalize_cue_text("Shoot it! Shoot it now!"))]
    assert is_kept_match(_match(412.0, "Shoot it! Shoot it now!"), kept)
    # A re-synced subtitle drifting by a second or so still lines up.
    assert is_kept_match(_match(413.4, "Shoot it! Shoot it now!"), kept)


def test_not_kept_for_same_word_elsewhere():
    kept = [(412.0, normalize_cue_text("Shoot it! Shoot it now!"))]
    assert not is_kept_match(_match(1903.0, "Oh, shoot. I left the keys in the car."), kept)
    assert not is_kept_match(_match(500.0, "Shoot it! Shoot it now!"), kept)


def test_nothing_kept_by_default():
    assert not is_kept_match(_match(412.0, "Shoot it!"), [])
