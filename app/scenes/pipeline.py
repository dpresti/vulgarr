"""Ties video frame classification -> clustering -> DetectedScene persistence
together for one title's scene-detection scan. Mirrors app.processing's role for
the subtitle/mute pipeline, but is a wholly independent feature -- see the
scene-detection plan for why this doesn't share Title.status/ProcessingJob."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_settings
from app.db.models import DetectedScene, Title
from app.db.session import get_setting
from app.domain import SceneReviewStatus
from app.mux.remux import probe
from app.vision.classifier import scan_video_frames
from app.vision.scene_cluster import cluster_scenes

logger = logging.getLogger(__name__)

ScanProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True)
class ScanOutcome:
    candidate_count: int


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
