from app.vision.scene_cluster import FrameScore, cluster_scenes, refine_scene_boundary, verified_fraction


def make_scores(pairs):
    """pairs: list of (timestamp, confidence)."""
    return [FrameScore(timestamp=t, confidence=c) for t, c in pairs]


def cluster(scores, **kwargs):
    kwargs.setdefault("confidence_threshold", 0.5)
    kwargs.setdefault("frame_interval_seconds", 2.0)
    kwargs.setdefault("merge_gap_seconds", 1.0)
    kwargs.setdefault("min_duration_seconds", 0.0)
    kwargs.setdefault("min_consecutive_frames", 2)
    return cluster_scenes(scores, **kwargs)


def test_no_scores_yields_no_candidates():
    assert cluster([]) == []


def test_all_below_threshold_yields_nothing():
    scores = make_scores([(0.0, 0.1), (2.0, 0.2), (4.0, 0.3)])
    assert cluster(scores) == []


def test_single_isolated_hit_is_dropped_by_persistence_requirement():
    # A lone spiking frame surrounded by low-confidence frames is more likely a
    # classifier false-positive than a real scene -- shouldn't alone form a candidate.
    scores = make_scores([(0.0, 0.1), (2.0, 0.9), (4.0, 0.1)])
    assert cluster(scores) == []


def test_consecutive_hits_form_one_candidate():
    scores = make_scores([(0.0, 0.9), (2.0, 0.8), (4.0, 0.1)])
    result = cluster(scores)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 2.0
    assert result[0].peak_confidence == 0.9


def test_a_dropped_sample_does_not_split_an_otherwise_continuous_run():
    # scan_video_frames skips a frame it couldn't extract/classify rather than
    # zero-filling it -- one missing sample (at what would've been t=2, with a 2s
    # frame interval) shouldn't split an otherwise-continuous run in two.
    scores = make_scores([(0.0, 0.9), (4.0, 0.9), (6.0, 0.9)])
    result = cluster(scores, frame_interval_seconds=2.0)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 6.0


def test_a_large_sample_gap_does_split_a_run_even_without_a_low_confidence_frame():
    # Two consecutive above-threshold *samples* that are implausibly far apart in
    # time for the configured frame interval shouldn't be treated as one continuous
    # run just because nothing lower-confidence happened to separate them in the list.
    scores = make_scores([(0.0, 0.9), (2.0, 0.9), (100.0, 0.9), (102.0, 0.9)])
    result = cluster(scores, frame_interval_seconds=2.0, merge_gap_seconds=1.0)
    assert len(result) == 2


def test_merges_runs_within_merge_gap():
    # Two separate above-threshold runs, only briefly interrupted by a low-confidence
    # frame between them, should merge into one candidate when the real gap between
    # them is within merge_gap_seconds.
    scores = make_scores([(0.0, 0.9), (2.0, 0.9), (4.0, 0.1), (6.0, 0.9), (8.0, 0.9)])
    result = cluster(scores, frame_interval_seconds=2.0, merge_gap_seconds=5.0)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 8.0


def test_keeps_far_apart_runs_separate():
    scores = make_scores([(0.0, 0.9), (2.0, 0.9), (100.0, 0.9), (102.0, 0.9)])
    result = cluster(scores, frame_interval_seconds=2.0, merge_gap_seconds=1.0)
    assert len(result) == 2


def test_drops_candidates_under_minimum_duration():
    scores = make_scores([(0.0, 0.9), (2.0, 0.9)])
    result = cluster(scores, min_duration_seconds=10.0)
    assert result == []


def test_peak_confidence_is_max_across_merged_run():
    scores = make_scores([(0.0, 0.6), (2.0, 0.95), (4.0, 0.7)])
    result = cluster(scores)
    assert len(result) == 1
    assert result[0].peak_confidence == 0.95


def test_unsorted_input_is_handled():
    scores = make_scores([(4.0, 0.1), (0.0, 0.9), (2.0, 0.9)])
    result = cluster(scores)
    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 2.0


def test_verified_fraction_empty_is_zero():
    assert verified_fraction([], 0.5) == 0.0


def test_verified_fraction_all_above_threshold():
    scores = make_scores([(0.0, 0.9), (1.0, 0.8), (2.0, 0.7)])
    assert verified_fraction(scores, 0.5) == 1.0


def test_verified_fraction_none_above_threshold():
    scores = make_scores([(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)])
    assert verified_fraction(scores, 0.5) == 0.0


def test_verified_fraction_partial():
    # A brief flash within a longer padded window should score low here even
    # though it was real -- exactly the case meant to stay in manual review
    # rather than qualify for bulk-approve.
    scores = make_scores([(0.0, 0.9), (1.0, 0.1), (2.0, 0.1), (3.0, 0.1)])
    assert verified_fraction(scores, 0.5) == 0.25


def test_verified_fraction_boundary_is_inclusive():
    scores = make_scores([(0.0, 0.5)])
    assert verified_fraction(scores, 0.5) == 1.0


def test_refine_scene_boundary_empty_returns_none():
    assert refine_scene_boundary([], 0.5) is None


def test_refine_scene_boundary_nothing_above_threshold_returns_none():
    scores = make_scores([(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)])
    assert refine_scene_boundary(scores, 0.5) is None


def test_refine_scene_boundary_tightens_to_actual_hits():
    # A dense re-scan of a padded window -- real signal only in the middle.
    scores = make_scores([(0.0, 0.1), (1.0, 0.6), (2.0, 0.7), (3.0, 0.65), (4.0, 0.2)])
    assert refine_scene_boundary(scores, 0.5) == (1.0, 3.0)


def test_refine_scene_boundary_can_widen_into_padding():
    # Coarse candidate was (say) 2.0-3.0, but the denser re-scan (which pads
    # beyond that) finds real signal continuing past it -- the whole point of
    # this function is that the refined boundary isn't capped at the coarse
    # candidate's own start/end.
    scores = make_scores([(0.0, 0.1), (1.0, 0.2), (2.0, 0.6), (3.0, 0.6), (4.0, 0.55), (5.0, 0.1)])
    assert refine_scene_boundary(scores, 0.5) == (2.0, 4.0)


def test_refine_scene_boundary_single_hit_collapses_to_a_point():
    scores = make_scores([(0.0, 0.1), (2.5, 0.9), (5.0, 0.1)])
    assert refine_scene_boundary(scores, 0.5) == (2.5, 2.5)
