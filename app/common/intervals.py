"""Generic time-interval merging, shared by the audio-mute pipeline
(app.audio.mute) and the scene-detection pipeline (app.vision.scene_cluster).
Kept dataclass-free -- callers wrap the merged (start, end) pairs in whatever
domain-specific type they need (MuteInterval, SceneCandidate, ...)."""


def merge_intervals(raw: list[tuple[float, float]], merge_gap_seconds: float) -> list[tuple[float, float]]:
    """Sort by start, then greedily merge any interval that starts within
    merge_gap_seconds of the current merged interval's end."""
    if not raw:
        return []
    ordered = sorted(raw, key=lambda pair: pair[0])
    merged: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + merge_gap_seconds:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]
