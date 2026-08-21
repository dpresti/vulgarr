from app.vision.claude_verify import VerifyResult, frame_timestamps, parse_verdict, top_score_timestamps
from app.vision.scene_cluster import FrameScore


def test_frame_timestamps_default_max_frames_is_two():
    assert frame_timestamps(0.0, 10.0) == [0.0, 10.0]


def test_frame_timestamps_spans_the_full_window():
    assert frame_timestamps(0.0, 9.0, max_frames=3) == [0.0, 4.5, 9.0]


def test_frame_timestamps_single_frame_uses_midpoint():
    assert frame_timestamps(0.0, 9.0, max_frames=1) == [4.5]


def test_frame_timestamps_zero_width_window_returns_one_point():
    assert frame_timestamps(5.0, 5.0) == [5.0]


def test_frame_timestamps_inverted_window_returns_midpoint():
    # Shouldn't happen in practice, but must not crash or produce a negative step.
    assert frame_timestamps(9.0, 0.0) == [4.5]


def test_parse_verdict_empty_returns_none():
    assert parse_verdict("") is None
    assert parse_verdict(None) is None


def test_parse_verdict_two_line_response_has_no_mute_signal():
    # Backward-compatible with a response that skips the mute-audio line
    # entirely -- the second line just becomes the reason instead.
    assert parse_verdict("YES\nClearly visible nudity.") == VerifyResult(
        confirmed=True, mute_audio=False, reason="Clearly visible nudity."
    )


def test_parse_verdict_no_with_reason():
    assert parse_verdict("NO\nJust swimwear, not exposed.") == VerifyResult(
        confirmed=False, mute_audio=False, reason="Just swimwear, not exposed."
    )


def test_parse_verdict_is_case_insensitive_and_tolerates_extra_text_on_first_line():
    assert parse_verdict("yes, this shows nudity\nreason here") == VerifyResult(
        confirmed=True, mute_audio=False, reason="reason here"
    )


def test_parse_verdict_missing_reason_line_defaults_to_empty():
    assert parse_verdict("YES") == VerifyResult(confirmed=True, mute_audio=False, reason="")


def test_parse_verdict_unparseable_response_returns_none():
    assert parse_verdict("I'm not sure how to answer that.") is None


def test_parse_verdict_ignores_blank_lines():
    assert parse_verdict("\n\nNO\n\nNothing explicit here.\n") == VerifyResult(
        confirmed=False, mute_audio=False, reason="Nothing explicit here."
    )


def test_parse_verdict_three_line_sex_scene_mutes_audio():
    assert parse_verdict("YES\nYES\nExplicit sexual activity shown.") == VerifyResult(
        confirmed=True, mute_audio=True, reason="Explicit sexual activity shown."
    )


def test_parse_verdict_three_line_nudity_without_sex_scene_does_not_mute():
    assert parse_verdict("YES\nNO\nNudity while undressing, not sexual activity.") == VerifyResult(
        confirmed=True, mute_audio=False, reason="Nudity while undressing, not sexual activity."
    )


def test_parse_verdict_mute_audio_forced_false_when_not_confirmed():
    # A malformed/inconsistent response (NO blur verdict but YES sex-scene
    # line) must never end up muting audio for something that isn't even
    # being blurred.
    assert parse_verdict("NO\nYES\ninconsistent response").mute_audio is False


def test_top_score_timestamps_empty_returns_empty():
    assert top_score_timestamps([]) == []


def test_top_score_timestamps_picks_highest_confidence_samples():
    scores = [
        FrameScore(timestamp=0.0, confidence=0.1),
        FrameScore(timestamp=1.0, confidence=0.9),
        FrameScore(timestamp=2.0, confidence=0.2),
        FrameScore(timestamp=3.0, confidence=0.8),
    ]
    assert top_score_timestamps(scores, max_frames=2) == [1.0, 3.0]


def test_top_score_timestamps_returns_chronological_order_not_confidence_order():
    scores = [
        FrameScore(timestamp=5.0, confidence=0.9),
        FrameScore(timestamp=1.0, confidence=0.8),
    ]
    # The higher-confidence sample comes later in time -- output must still be
    # sorted by timestamp, not by the order picked.
    assert top_score_timestamps(scores, max_frames=2) == [1.0, 5.0]


def test_top_score_timestamps_respects_max_frames_cap():
    scores = [FrameScore(timestamp=float(i), confidence=float(i)) for i in range(10)]
    assert top_score_timestamps(scores, max_frames=3) == [7.0, 8.0, 9.0]


def test_top_score_timestamps_fewer_scores_than_max_frames_returns_all():
    scores = [FrameScore(timestamp=1.0, confidence=0.5)]
    assert top_score_timestamps(scores, max_frames=5) == [1.0]
