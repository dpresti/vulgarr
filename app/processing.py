"""Ties subtitle parsing -> matching -> muting -> remux together for one title."""

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audio.mute import MuteInterval, build_mute_intervals
from app.config import settings as app_settings
from app.db.models import WordListEntry
from app.db.session import get_setting
from app.domain import SEVERITY_CANONICAL_ORDER, SEVERITY_RANK, Severity, is_mkv_path
from app.mux.remux import ProgressCallback, RemuxError, StageCallback, probe, remux_with_clean_track
from app.subtitles.matcher import CueMatch, ProfanityMatcher, WordListTerm
from app.subtitles.parser import parse_srt_file

logger = logging.getLogger(__name__)

# A subtitle cue timestamped after the video's own duration can only happen if the
# subtitle was made for a longer cut/edition of the film than this actual video file
# (e.g. an extended cut's subtitles used against the theatrical release) -- there's no
# legitimate reason for a correctly-matched subtitle to have this happen. Small
# tolerance for container duration rounding/slop, not for genuine drift.
SUBTITLE_OVERRUN_TOLERANCE_SECONDS = 10.0


class ProcessingError(RuntimeError):
    pass


def check_subtitle_video_duration_match(cues: list, video_duration: float) -> str | None:
    """Returns an error message if the subtitle looks mismatched with the video
    (e.g. timed for a longer cut/edition), or None if it looks fine."""
    if not cues or video_duration <= 0:
        return None
    last_cue_end = max(c.end_seconds for c in cues)
    overrun = last_cue_end - video_duration
    if overrun > SUBTITLE_OVERRUN_TOLERANCE_SECONDS:
        return (
            f"Subtitle looks mismatched with this video (likely a different cut/edition): "
            f"last subtitle cue ends at {last_cue_end:.0f}s but the video is only "
            f"{video_duration:.0f}s long, {overrun:.0f}s over. Try a different subtitle "
            f"for this exact release."
        )
    return None


def effective_severity_levels(video_path: str, severity_levels: list[Severity] | None) -> list[Severity]:
    """Resolve which severity level(s) to actually generate tracks for. Multiple tracks
    require .mkv; on other file types the saved multi-level selection is preserved (not
    lost) but only the single most inclusive level is generated until an .mkv
    replacement is available."""
    levels = [s for s in SEVERITY_CANONICAL_ORDER if s in (severity_levels or [Severity.child])] or [Severity.child]
    if len(levels) > 1 and not is_mkv_path(video_path):
        return [levels[0]]
    return levels


@dataclass(frozen=True)
class ProcessingOutcome:
    matched_cue_count: int  # for the most inclusive (lowest-rank) selected level
    backup_path: Path
    clean_track_indices: list[int]


async def load_active_terms(session: AsyncSession, min_severity: Severity = Severity.child) -> list[WordListTerm]:
    result = await session.execute(select(WordListEntry).where(WordListEntry.enabled.is_(True)))
    min_rank = SEVERITY_RANK[min_severity]
    return [
        WordListTerm(term=row.term, severity=row.severity, match_whole_word=row.match_whole_word)
        for row in result.scalars().all()
        if SEVERITY_RANK[row.severity] >= min_rank
    ]


async def process_video(
    session: AsyncSession,
    *,
    video_path: Path,
    subtitle_path: Path,
    severity_levels: list[Severity] | None = None,
    known_clean_indices: list[int] | None = None,
    precise_mute: bool = False,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> ProcessingOutcome:
    """Run the full pipeline for one file, generating one clean track per severity level.
    Raises ProcessingError/RemuxError on failure."""
    if not subtitle_path.exists():
        raise ProcessingError(f"Subtitle file not found: {subtitle_path}")

    levels = effective_severity_levels(str(video_path), severity_levels)

    cues = parse_srt_file(str(subtitle_path))

    src_probe = await probe(app_settings.ffprobe_bin, video_path)
    video_duration = float(src_probe["format"].get("duration", 0))
    duration_error = check_subtitle_video_duration_match(cues, video_duration)
    if duration_error:
        raise ProcessingError(duration_error)

    intervals_by_level: list[tuple[Severity, list[MuteInterval]]] = []
    matches_by_level: dict[Severity, list[CueMatch]] = {}
    for level in levels:
        terms = await load_active_terms(session, min_severity=level)
        if not terms:
            raise ProcessingError(f"Word list is empty at the '{level.value}' severity level -- nothing to filter")
        matcher = ProfanityMatcher(terms)
        matches = matcher.match_all(cues)
        matches_by_level[level] = matches
        intervals_by_level.append((level, build_mute_intervals(matches, precise=precise_mute)))

    # Report the count for whichever selected level catches the most (the lowest-rank
    # one selected), since that's the most complete picture of what's being filtered.
    most_inclusive_level = levels[0]
    representative_count = len(matches_by_level[most_inclusive_level])

    backup_root = app_settings.data_dir / "backups"
    result = await remux_with_clean_track(
        video_path=video_path,
        matches=matches_by_level[most_inclusive_level],
        intervals_by_level=intervals_by_level,
        media_root=app_settings.media_root,
        backup_root=backup_root,
        ffmpeg_bin=app_settings.ffmpeg_bin,
        ffprobe_bin=app_settings.ffprobe_bin,
        clean_track_title=app_settings.clean_track_title,
        clean_track_language=app_settings.clean_track_language,
        known_clean_indices=known_clean_indices,
        on_progress=on_progress,
        on_stage=on_stage,
    )

    return ProcessingOutcome(
        matched_cue_count=representative_count,
        backup_path=result.backup_path,
        clean_track_indices=result.clean_track_indices,
    )
