from app.vision.classifier import _load_resume_scores, _resumable_prefix, frame_confidence
from app.vision.scene_cluster import FrameScore


def test_no_detections_yields_zero():
    assert frame_confidence([]) == 0.0


def test_ignores_non_explicit_classes():
    detections = [{"class": "FACE_FEMALE", "score": 0.95}, {"class": "FEET_EXPOSED", "score": 0.9}]
    assert frame_confidence(detections) == 0.0


def test_returns_max_score_among_explicit_classes():
    detections = [
        {"class": "FACE_FEMALE", "score": 0.99},
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.6},
        {"class": "BUTTOCKS_EXPOSED", "score": 0.85},
    ]
    assert frame_confidence(detections) == 0.85


def scores(pairs):
    return [FrameScore(timestamp=t, confidence=c) for t, c in pairs]


def test_resumable_prefix_empty_scores():
    assert _resumable_prefix([], start_offset=0.0, frame_interval_seconds=0.5) == (0.0, [])


def test_resumable_prefix_fully_contiguous_run():
    s = scores([(0.0, 0.1), (0.5, 0.2), (1.0, 0.3), (1.5, 0.4)])
    covered, trusted = _resumable_prefix(s, start_offset=0.0, frame_interval_seconds=0.5)
    assert covered == 2.0  # last timestamp (1.5) + one more interval
    assert trusted == s


def test_resumable_prefix_stops_at_first_gap():
    # A real gap (not just interval rounding) at t=1.0 -- classic
    # out-of-order-completion scenario: 1.5 finished and got checkpointed
    # before 1.0 did, and the process was killed before 1.0 ever completed.
    s = scores([(0.0, 0.1), (0.5, 0.2), (1.5, 0.4), (2.0, 0.5)])
    covered, trusted = _resumable_prefix(s, start_offset=0.0, frame_interval_seconds=0.5)
    assert covered == 1.0  # only trusts through 0.5 + one interval
    assert trusted == scores([(0.0, 0.1), (0.5, 0.2)])


def test_resumable_prefix_out_of_order_input_still_works():
    # Same scores as the fully-contiguous case, but appended out of order --
    # exactly what concurrent classification produces in the real checkpoint
    # file. Must sort internally, not assume file order is chronological.
    s = scores([(1.0, 0.3), (0.0, 0.1), (1.5, 0.4), (0.5, 0.2)])
    covered, trusted = _resumable_prefix(s, start_offset=0.0, frame_interval_seconds=0.5)
    assert covered == 2.0
    assert len(trusted) == 4


def test_resumable_prefix_ignores_entries_before_start_offset():
    s = scores([(5.0, 0.1), (5.5, 0.2), (6.0, 0.3)])
    covered, trusted = _resumable_prefix(s, start_offset=5.0, frame_interval_seconds=0.5)
    assert covered == 1.5
    assert len(trusted) == 3


def test_resumable_prefix_nothing_at_or_after_start_offset_is_untrusted():
    s = scores([(0.0, 0.1), (0.5, 0.2)])
    covered, trusted = _resumable_prefix(s, start_offset=5.0, frame_interval_seconds=0.5)
    assert covered == 0.0
    assert trusted == []


def test_resumable_prefix_single_entry():
    s = scores([(0.0, 0.5)])
    covered, trusted = _resumable_prefix(s, start_offset=0.0, frame_interval_seconds=0.5)
    assert covered == 0.5
    assert trusted == s


def test_load_resume_scores_missing_file_returns_empty(tmp_path):
    assert _load_resume_scores(tmp_path / "does_not_exist.jsonl") == []


def test_load_resume_scores_round_trips(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"t": 0.0, "c": 0.1}\n{"t": 0.5, "c": 0.9}\n')
    result = _load_resume_scores(path)
    assert result == scores([(0.0, 0.1), (0.5, 0.9)])


def test_load_resume_scores_tolerates_corrupt_trailing_line(tmp_path):
    # A process killed mid-write can leave a truncated final line -- must not
    # lose the good lines before it.
    path = tmp_path / "scores.jsonl"
    path.write_text('{"t": 0.0, "c": 0.1}\n{"t": 0.5, "c": 0.9}\n{"t": 1.0, "c":')
    result = _load_resume_scores(path)
    assert result == scores([(0.0, 0.1), (0.5, 0.9)])


def test_load_resume_scores_skips_blank_lines(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"t": 0.0, "c": 0.1}\n\n{"t": 0.5, "c": 0.9}\n')
    result = _load_resume_scores(path)
    assert result == scores([(0.0, 0.1), (0.5, 0.9)])
