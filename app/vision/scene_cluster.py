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
#
# This alone throws away genuine brief flashes too, though -- confirmed directly
# against GoT S06E07 (2151.0-2151.5): a single 0.25s-interval sample at 2151.25
# scored FEMALE_BREAST_EXPOSED=0.73 (a real, high-confidence hit) with both its
# immediate neighbors at hard 0.0, because the actual exposure was a quick cut
# narrower than even a 0.25s sampling stride -- denser sampling doesn't fix this,
# since the flash is shorter than the grid no matter how fine the grid gets. See
# high_confidence_override below for the fix.
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
    min_consecutive_frames: int = DEFAULT_MIN_CONSECUTIVE_FRAMES,
    high_confidence_override: float | None = None,
) -> list[SceneCandidate]:
    """Threshold-crossing + persistence-run requirement, then merge nearby runs
    (same gap-merge algorithm the audio pipeline uses for cue intervals).

    frame_interval_seconds and merge_gap_seconds are deliberately separate concerns:
    the former bounds how far apart two samples can be and still count as the *same*
    continuous run (real scans can have gaps from skipped/failed frames, not just
    genuinely low-confidence ones); the latter bounds how big a gap between two
    already-separate runs is still treated as one candidate scene.

    high_confidence_override, when given, lets a run shorter than
    min_consecutive_frames still count if its peak confidence clears this (higher)
    bar -- a real quick-cut flash can be both genuinely high-confidence and
    genuinely one sample wide (see DEFAULT_MIN_CONSECUTIVE_FRAMES's comment); this
    recovers exactly that case without lowering min_consecutive_frames itself,
    which would also let through the ordinary low-confidence single-frame noise
    (motion blur, skin-toned lighting) that requirement exists to filter.

    Deliberately does NOT drop runs under scene_min_duration_seconds here --
    confirmed directly against GoT S03E03 that doing so throws away real,
    already-persistence-validated detections with nothing nearby to merge into
    (a genuine 2-consecutive-frame hit at 2265.0-2265.5, and a high-confidence
    override hit collapsing to a single point at 2361.0, both isolated by more
    than merge_gap_seconds from anything else). Every caller already re-scans
    each returned candidate at a denser interval and pads it out to
    min_duration_seconds around its refined center if still too short after
    that (see app.scenes.pipeline.scan_for_scenes and
    scripts/bench_scene_detection.py's scan_title) -- rejecting short-but-real
    candidates here only pre-empts that existing safety net for no benefit,
    trading a real miss for saving one cheap dense re-scan.
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

    def is_real_run(run: list[FrameScore]) -> bool:
        if len(run) >= min_consecutive_frames:
            return True
        return high_confidence_override is not None and max(s.confidence for s in run) >= high_confidence_override

    raw: list[tuple[float, float]] = [(run[0].timestamp, run[-1].timestamp) for run in runs if is_real_run(run)]
    if not raw:
        return []

    candidates: list[SceneCandidate] = []
    for start, end in merge_intervals(raw, merge_gap_seconds):
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


def boundary_touches_window_edge(
    boundary: tuple[float, float], window_start: float, window_end: float, tolerance: float
) -> tuple[bool, bool]:
    """Whether refine_scene_boundary's result sits close enough to either edge of
    the dense re-scan window it was computed from that real content is likely
    continuing past what was actually searched -- refine_scene_boundary can only
    ever report a hit *within* the window it was given (app.vision.classifier.
    scan_window_frames' window_start-pad_seconds .. window_end+pad_seconds), so a
    boundary landing right at that wall is indistinguishable from "the scene
    genuinely ends here" only by coincidence; real content stopping exactly on the
    edge of an arbitrarily-placed search window is the less likely explanation.
    Pure function, no I/O -- callers (app.scenes.pipeline.scan_for_scenes) use this
    to decide whether to re-scan with a wider window rather than trusting the wall,
    the same way this whole padded-re-scan mechanism exists so a boundary isn't
    trusted at the original coarse candidate's own edges either.

    tolerance should be at least the dense scan's own frame_interval_seconds --
    otherwise a hit one sample width short of the true edge (an ordinary sampling
    quantization, not a sign of more content beyond it) would trigger an
    unnecessary expansion."""
    start, end = boundary
    return (start - window_start) <= tolerance, (window_end - end) <= tolerance
