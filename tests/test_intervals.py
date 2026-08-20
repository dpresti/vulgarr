from app.common.intervals import merge_intervals


def test_empty_input_yields_no_intervals():
    assert merge_intervals([], merge_gap_seconds=0.25) == []


def test_single_interval_passes_through():
    assert merge_intervals([(1.0, 2.0)], merge_gap_seconds=0.25) == [(1.0, 2.0)]


def test_merges_close_intervals():
    result = merge_intervals([(1.0, 2.0), (2.2, 3.0)], merge_gap_seconds=0.25)
    assert result == [(1.0, 3.0)]


def test_keeps_far_apart_intervals_separate():
    result = merge_intervals([(1.0, 2.0), (10.0, 11.0)], merge_gap_seconds=0.25)
    assert result == [(1.0, 2.0), (10.0, 11.0)]


def test_sorts_unordered_input():
    result = merge_intervals([(10.0, 11.0), (1.0, 2.0)], merge_gap_seconds=0.25)
    assert result == [(1.0, 2.0), (10.0, 11.0)]


def test_overlapping_intervals_merge_and_keep_the_later_end():
    result = merge_intervals([(1.0, 5.0), (2.0, 3.0)], merge_gap_seconds=0.0)
    assert result == [(1.0, 5.0)]


def test_chain_of_three_merges_into_one():
    result = merge_intervals([(1.0, 2.0), (2.1, 3.0), (3.1, 4.0)], merge_gap_seconds=0.25)
    assert result == [(1.0, 4.0)]
