"""Ties video frame classification -> clustering -> DetectedScene persistence
together for one title's scene-detection scan. Mirrors app.processing's role for
the subtitle/mute pipeline, but is a wholly independent feature -- see the
scene-detection plan for why this doesn't share Title.status/ProcessingJob."""

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audio.mute import MuteInterval
from app.common.intervals import merge_intervals
from app.config import settings as app_settings
from app.db.models import DetectedScene, Title
from app.db.session import get_setting
from app.domain import SceneReviewStatus
from app.mux.remux import ProgressCallback, StageCallback, probe
from app.mux.scene_blur import apply_blur
from app.vision.classifier import scan_video_frames
from app.vision.scene_cluster import cluster_scenes

logger = logging.getLogger(__name__)

ScanProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True)
class ScanOutcome:
    candidate_count: int


@dataclass(frozen=True)
class BlurOutcome:
    output_path: Path
    scene_count: int


async def scan_for_scenes(
    session: AsyncSession,
    title: Title,
    on_progress: ScanProgressCallback | None = None,
) -> ScanOutcome:
    """Scan title's video for candidate explicit-content scenes.

    Replaces any prior pending/rejected candidates from an earlier scan --
    approved or already-applied ones represent a human decision (or a real edit
    already baked into the sibling file) and must never be silently discarded by
    a re-scan. A candidate that happens to overlap an already-approved scene on
    re-scan is not deduplicated here -- left for the review UI (Phase 2) to
    surface sensibly, not a data-integrity concern at this layer.
    """
    confidence_threshold = float(await get_setting(session, "scene_confidence_threshold"))
    frame_interval = float(await get_setting(session, "scene_frame_interval_seconds"))
    min_duration = float(await get_setting(session, "scene_min_duration_seconds"))
    merge_gap = float(await get_setting(session, "scene_merge_gap_seconds"))

    video_path = Path(title.video_path)
    src_probe = await probe(app_settings.ffprobe_bin, video_path)
    duration = float(src_probe["format"].get("duration", 0))

    scores = await scan_video_frames(
        video_path,
        app_settings.ffmpeg_bin,
        duration_seconds=duration,
        frame_interval_seconds=frame_interval,
        on_progress=on_progress,
    )

    candidates = cluster_scenes(
        scores,
        confidence_threshold=confidence_threshold,
        frame_interval_seconds=frame_interval,
        merge_gap_seconds=merge_gap,
        min_duration_seconds=min_duration,
    )

    stale = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.status.in_([SceneReviewStatus.pending, SceneReviewStatus.rejected]),
        )
    )
    for row in stale.scalars().all():
        await session.delete(row)

    for candidate in candidates:
        session.add(
            DetectedScene(
                title_id=title.id,
                start_seconds=candidate.start,
                end_seconds=candidate.end,
                peak_confidence=candidate.peak_confidence,
                status=SceneReviewStatus.pending,
            )
        )
    await session.commit()

    return ScanOutcome(candidate_count=len(candidates))


async def apply_scene_blur(
    session: AsyncSession,
    title: Title,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> BlurOutcome:
    """Snapshot title's currently approved-and-unapplied scenes, blur+mute them
    into a new sibling "Vulgarr Edit" file, and mark that exact snapshot applied.
    The source file is never opened for writing -- a failed blur just means the
    temp file gets deleted, nothing about the original changes.

    Raises ValueError if there's nothing approved-and-unapplied to do (the caller
    is expected to have already checked this before enqueuing a job at all, but
    it's re-checked here too since the set can change between "Apply" being
    clicked and the job actually running)."""
    result = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.status == SceneReviewStatus.approved,
            DetectedScene.applied_at.is_(None),
        )
    )
    scenes = result.scalars().all()
    if not scenes:
        raise ValueError("No approved, unapplied scenes to blur")

    # Overlapping/adjacent approved windows (real after enough independent
    # "adjust" edits) are merged before building the filter graph -- not required
    # for correctness (ffmpeg's between() OR-summation handles overlaps fine
    # regardless), just keeps the term count down against the batching limit.
    merged = merge_intervals([(s.start_seconds, s.end_seconds) for s in scenes], merge_gap_seconds=0.0)
    intervals = [MuteInterval(start=s, end=e) for s, e in merged]

    video_crf = int(await get_setting(session, "blur_video_crf"))
    video_preset = await get_setting(session, "blur_video_preset")

    final_path = await apply_blur(
        video_path=Path(title.video_path),
        intervals=intervals,
        ffmpeg_bin=app_settings.ffmpeg_bin,
        ffprobe_bin=app_settings.ffprobe_bin,
        video_crf=video_crf,
        video_preset=video_preset,
        on_progress=on_progress,
        on_stage=on_stage,
    )

    now = datetime.datetime.utcnow()
    title.vulgarr_edit_path = str(final_path)
    title.vulgarr_edit_generated_at = now
    for scene in scenes:
        scene.applied_at = now
    await session.commit()

    return BlurOutcome(output_path=final_path, scene_count=len(scenes))
