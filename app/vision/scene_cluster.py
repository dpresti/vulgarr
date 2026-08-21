"""Turn per-frame classifier scores into candidate scene windows.

Pure Python, no model dependency -- kept separate from app.vision.classifier the
same way app.audio.mute's build_mute_intervals (pure) is kept separate from
align_matches_for_cue (model-calling), so this logic is directly unit-testable
against synthetic scores without loading a real classifier.
"""

from dataclasses import dataclass

from app.common.intervals import merge_intervals

# A single frame spiking above threshold, surrounded by low-confidence frames, is
# far more likely to be a classifier false-positive (motion blur, skin-toned
# lighting, a fast cut) than a real several-second scene -- require at least this
# many consecutive above-threshold samples before a run counts as a real hit.
DEFAULT_MIN_CONSECUTIVE_FRAMES = 2


@dataclass(frozen=True)
class FrameScore:
    timestamp: float
    confidence: float


@dataclass(frozen=True)
class SceneCandidate:
    start: float
    end: float
    peak_confidence: float


# A missed/failed sample (see app.vision.classifier.scan_video_frames, which skips
# a frame it couldn't extract/classify rather than zero-filling it) shouldn't alone
# break an otherwise-continuous run. Losing exactly one sample leaves a real gap of
# 2x the frame interval between the surviving neighbors (e.g. samples at 0/2/4/6 with
# the one at 2 dropped leaves a 4-second gap between 0 and 4) -- the tolerance factor
# needs to be > 2.0 to actually cover that case, with a little slack beyond it.
_SAMPLE_GAP_TOLERANCE = 2.5


def cluster_scenes(
    scores: list[FrameScore],
    confidence_threshold: float,
    frame_interval_seconds: float,
    merge_gap_seconds: float,
    min_duration_seconds: float,
    min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
) -> list[SceneCandidate]:
    """Threshold-crossing + persistence-run requirement, then merge nearby runs
    (same gap-merge algorithm the audio pipeline uses for cue intervals), then drop
    anything that still doesn't meet the minimum duration after merging.

    frame_interval_seconds and merge_gap_seconds are deliberately separate concerns:
    the former bounds how far apart two samples can be and still count as the *same*
    continuous run (real scans can have gaps from skipped/failed frames, not just
    genuinely low-confidence ones); the latter bounds how big a gap between two
    already-separate runs is still treated as one candidate scene.
    """
    if not scores:
        return []

    ordered = sorted(scores, key=lambda s: s.timestamp)
    max_run_gap = frame_interval_seconds * _SAMPLE_GAP_TOLERANCE

    runs: list[list[FrameScore]] = []
    current: list[FrameScore] = []
    for score in ordered:
        above = score.confidence >= confidence_threshold
        if above and current and score.timestamp - current[-1].timestamp > max_run_gap:
            runs.append(current)
            current = []
        if above:
            current.append(score)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    raw: list[tuple[float, float]] = [
        (run[0].timestamp, run[-1].timestamp) for run in runs if len(run) >= min_consecutive_frames
    ]
    if not raw:
        return []

    candidates: list[SceneCandidate] = []
    for start, end in merge_intervals(raw, merge_gap_seconds):
        if end - start < min_duration_seconds:
            continue
        peak = max(s.confidence for s in ordered if start <= s.timestamp <= end)
        candidates.append(SceneCandidate(start=start, end=end, peak_confidence=peak))
    return candidates


def verified_fraction(scores: list[FrameScore], confidence_threshold: float) -> float:
    """Fraction of scores at/above confidence_threshold -- meant to be computed
    from a second, much denser re-scan of just one candidate's padded window
    (see app.vision.classifier.scan_window_frames), not the sparser samples that
    found the candidate in the first place.

    A single peak_confidence value can't distinguish "flickered above threshold
    once" from "consistently above threshold throughout the window" -- both can
    produce the same peak. This can, and is a meaningfully more robust signal
    for deciding which candidates are safe to bulk-approve without a human
    looking at every one (see scene_review_list.html's "Approve high-confidence"
    action) versus which stay borderline enough to need a real look."""
    if not scores:
        return 0.0
    hits = sum(1 for s in scores if s.confidence >= confidence_threshold)
    return hits / len(scores)


def refine_scene_boundary(scores: list[FrameScore], confidence_threshold: float) -> tuple[float, float] | None:
    """Tightest [start, end] spanning every sample at/above confidence_threshold
    in a dense re-scan of one candidate's padded window -- the video-side
    analog of what Whisper forced-alignment does for a subtitle cue: the main
    scan (or a human's manual "Save" adjustment) only ever localizes a scene
    roughly, at whatever its own sample interval allows; this takes a much
    denser second look at just that small window and reports the real
    boundary the classifier actually saw, which the caller then uses as the
    candidate's stored start/end.

    Can extend the boundary in *either* direction relative to the coarse
    candidate: narrower if the coarse hit was a brief spike inside a longer
    quiet window, wider if real signal continues into the padding the coarse
    scan never sampled densely enough to catch (this is exactly what a real
    ground-truth comparison against GoT S01E10 found by hand this session --
    a scene's real tail extended past where the coarse scan had stopped).

    Returns None if nothing in the dense re-scan cleared the threshold at all
    (the coarse candidate was likely a borderline/noisy hit) -- callers should
    fall back to the original coarse boundary in that case rather than
    collapsing the scene to nothing."""
    hits = [s.timestamp for s in scores if s.confidence >= confidence_threshold]
    if not hits:
        return None
    return min(hits), max(hits)
