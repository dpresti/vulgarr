"""Turn matched subtitle cues into ffmpeg mute intervals and filter expressions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from app.common.intervals import merge_intervals
from app.subtitles.matcher import CueMatch

# Called after each cue's alignment attempt (success or per-cue fallback) with
# (cues_done_so_far, total_cues) -- lets the caller surface real progress during
# what can be a slow, per-cue loop (a real audio extract + model inference each).
AlignProgressCallback = Callable[[int, int], Awaitable[None]]

# Small padding around each muted cue smooths the mute boundary (avoids an
# audible "click" right at the edge of the subtitle timestamp, since subtitle
# timing is rarely frame-accurate to the audio).
DEFAULT_PAD_SECONDS = 0.15
# Wider padding for word-level precision mode: the mute window is now a proportional
# *estimate* of where a word falls within a cue (based on its position in the line's
# text, assuming roughly steady speech pace), not the cue's real, known boundaries --
# extra margin on each side reduces the risk of the estimate clipping part of the word.
PRECISE_PAD_SECONDS = 0.4
# Whisper forced-alignment mode locates the word in the real audio rather than
# estimating from text position, so it needs much less safety margin than the
# estimate above -- alignment boundaries land close enough to the real word that
# only a touch of edge softening is needed, not real slop coverage.
WHISPER_PAD_SECONDS = 0.2
# Cues whose (padded) intervals are within this gap of each other are merged
# into one interval, both to avoid a click-unmute-click and to keep the
# generated ffmpeg filter expression shorter.
DEFAULT_MERGE_GAP_SECONDS = 0.25


@dataclass(frozen=True)
class MuteInterval:
    start: float
    end: float


def _merge_raw_intervals(raw: list[tuple[float, float]], merge_gap_seconds: float) -> list[MuteInterval]:
    return [MuteInterval(start=s, end=e) for s, e in merge_intervals(raw, merge_gap_seconds)]


def build_mute_intervals(
    matches: list[CueMatch],
    pad_seconds: float | None = None,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    precise: bool = False,
) -> list[MuteInterval]:
    """Convert matched cues into a sorted, non-overlapping, padded/merged list of mute intervals.

    By default mutes each whole cue's on-screen duration (the only timing a plain
    subtitle file gives us). With precise=True, mutes a narrower window per matched
    word instead, estimated by that word's proportional position within the cue's
    text -- meaningfully tighter, but an estimate, not exact, since subtitle files
    have no real word-level timestamps.
    """
    if not matches:
        return []

    if pad_seconds is None:
        pad_seconds = PRECISE_PAD_SECONDS if precise else DEFAULT_PAD_SECONDS

    raw: list[tuple[float, float]] = []
    for m in matches:
        if precise and m.spans and m.normalized_length:
            cue_duration = m.cue.end_seconds - m.cue.start_seconds
            for span in m.spans:
                word_start = m.cue.start_seconds + (span.start / m.normalized_length) * cue_duration
                word_end = m.cue.start_seconds + (span.end / m.normalized_length) * cue_duration
                raw.append((max(0.0, word_start - pad_seconds), word_end + pad_seconds))
        else:
            raw.append((max(0.0, m.cue.start_seconds - pad_seconds), m.cue.end_seconds + pad_seconds))

    return _merge_raw_intervals(raw, merge_gap_seconds)


async def build_mute_intervals_whisper(
    matches: list[CueMatch],
    video_path: Path,
    ffmpeg_bin: str,
    pad_seconds: float = WHISPER_PAD_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    on_progress: AlignProgressCallback | None = None,
) -> list[MuteInterval]:
    """Like build_mute_intervals(precise=True), but locates each matched word's real
    position in the audio via forced alignment (app.audio.forced_align) instead of
    estimating it from the word's position within the cue's text. Falls back to the
    proportional estimate on a per-cue basis if alignment fails for that cue (e.g. a
    cue whose audio doesn't actually contain clean, alignable speech).

    Each cue involves a real ffmpeg audio extract plus a model inference, so this can
    be slow (seconds per cue) -- on_progress(done, total), relative to this call's own
    `matches`, lets the caller surface that rather than looking frozen."""
    from app.audio.forced_align import align_matches_for_cue

    if not matches:
        return []

    raw: list[tuple[float, float]] = []
    for i, m in enumerate(matches):
        windows = await align_matches_for_cue(video_path, ffmpeg_bin, m)
        if windows is None:
            raw.extend(
                (iv.start, iv.end)
                for iv in build_mute_intervals(
                    [m], precise=True, pad_seconds=PRECISE_PAD_SECONDS, merge_gap_seconds=0.0
                )
            )
        else:
            for w in windows:
                raw.append((max(0.0, w.start - pad_seconds), w.end + pad_seconds))
        if on_progress is not None:
            await on_progress(i + 1, len(matches))

    return _merge_raw_intervals(raw, merge_gap_seconds)


# Chaining too many between(t,a,b) terms into one `volume` filter's `enable=` expression
# eventually breaks ffmpeg's expression parser -- observed failing around ~80-90 terms
# with "Cannot allocate memory" (a misleading error for what's actually a parser
# complexity limit, not real memory exhaustion; a high-profanity movie easily exceeds
# this). Batching intervals across multiple chained `volume` filter stages keeps each
# individual expression small regardless of how many total intervals there are.
_MAX_TERMS_PER_STAGE = 20


def build_volume_filter(intervals: list[MuteInterval], input_label: str, output_label: str) -> str:
    """Build a chain of ffmpeg `volume` filters that silence audio during the given intervals.

    Uses the standard `between(t,a,b)` summation trick within each stage: `between()`
    evaluates to 1 or 0, so summing several of them and comparing to the `volume`
    filter's boolean `enable` expression mutes whenever *any* interval in that batch
    is active. Stages are chained (each stage's output feeds the next) rather than
    combined into a single filter, to keep each `enable=` expression short.
    """
    if not intervals:
        return f"[{input_label}]anull[{output_label}]"

    batches = [intervals[i : i + _MAX_TERMS_PER_STAGE] for i in range(0, len(intervals), _MAX_TERMS_PER_STAGE)]

    stages = []
    current_label = input_label
    for i, batch in enumerate(batches):
        is_last = i == len(batches) - 1
        next_label = output_label if is_last else f"{output_label}_stage{i}"
        conditions = "+".join(f"between(t,{iv.start:.3f},{iv.end:.3f})" for iv in batch)
        stages.append(f"[{current_label}]volume=0:enable='{conditions}'[{next_label}]")
        current_label = next_label

    return ";".join(stages)
