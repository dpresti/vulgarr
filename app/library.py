"""Upserts Title rows from Sonarr/Radarr library data or webhook payloads,
resolving each video's subtitle file by convention.
"""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_settings
from app.db.models import JobState, MediaType, ProcessingJob, Title, TriggerSource
from app.db.session import get_session, get_setting
from app.integrations.radarr import RadarrClient
from app.integrations.sonarr import SonarrClient
from app.integrations.subtitle_lookup import find_subtitle_for_video

logger = logging.getLogger(__name__)


def is_outdated(title: Title, current_wordlist_version: int) -> bool:
    return (
        title.last_processed_wordlist_version is not None
        and title.last_processed_wordlist_version < current_wordlist_version
    )


async def upsert_title(
    session: AsyncSession,
    *,
    media_type: MediaType,
    display_name: str,
    video_path: str,
    sonarr_series_id: int | None = None,
    sonarr_episode_id: int | None = None,
    radarr_movie_id: int | None = None,
    series_title: str | None = None,
    season_number: int | None = None,
    episode_number: int | None = None,
    default_precise_mute: bool | None = None,
) -> Title:
    result = await session.execute(select(Title).where(Title.video_path == video_path))
    title = result.scalar_one_or_none()

    subtitle = find_subtitle_for_video(Path(video_path), app_settings.default_subtitle_language)

    if title is None:
        if default_precise_mute is None:
            # Caller didn't pre-fetch it (e.g. a single webhook-triggered upsert, not a
            # bulk sync) -- fine to look it up here for a one-off call.
            default_precise_mute = await get_setting(session, "default_precise_mute")
        title = Title(
            media_type=media_type,
            display_name=display_name,
            video_path=video_path,
            sonarr_series_id=sonarr_series_id,
            sonarr_episode_id=sonarr_episode_id,
            radarr_movie_id=radarr_movie_id,
            precise_mute=default_precise_mute,
        )
        session.add(title)
    else:
        title.display_name = display_name

    title.series_title = series_title
    title.season_number = season_number
    title.episode_number = episode_number
    title.subtitle_path = str(subtitle) if subtitle else None
    title.subtitle_language = app_settings.default_subtitle_language if subtitle else None

    await session.commit()
    await session.refresh(title)
    return title


async def sync_sonarr_library(session: AsyncSession, client: SonarrClient) -> int:
    default_precise_mute = await get_setting(session, "default_precise_mute")
    count = 0
    for ep_file in await client.list_episode_files():
        await upsert_title(
            session,
            media_type=MediaType.episode,
            display_name=ep_file.display_name,
            video_path=ep_file.path,
            sonarr_series_id=ep_file.series_id,
            sonarr_episode_id=ep_file.episode_id,
            series_title=ep_file.series_title,
            season_number=ep_file.season_number,
            episode_number=ep_file.episode_number,
            default_precise_mute=default_precise_mute,
        )
        count += 1
    return count


async def sync_radarr_library(session: AsyncSession, client: RadarrClient) -> int:
    default_precise_mute = await get_setting(session, "default_precise_mute")
    count = 0
    for movie_file in await client.list_movie_files():
        await upsert_title(
            session,
            media_type=MediaType.movie,
            display_name=movie_file.display_name,
            video_path=movie_file.path,
            radarr_movie_id=movie_file.movie_id,
            default_precise_mute=default_precise_mute,
        )
        count += 1
    return count


async def enqueue_if_not_already_active(title_id: int, trigger_source: TriggerSource) -> None:
    # Deferred import: app.queue.worker imports from this module too (to reuse
    # poll_for_subtitle_then_enqueue below), so this has to stay a function-local
    # import to avoid a circular import at module load time.
    from app.queue.worker import job_queue

    async with get_session() as session:
        result = await session.execute(
            select(ProcessingJob).where(
                ProcessingJob.title_id == title_id,
                ProcessingJob.state.in_([JobState.queued, JobState.processing]),
            )
        )
        if result.scalar_one_or_none() is not None:
            logger.info("Title %s already has an active job, skipping duplicate enqueue from %s", title_id, trigger_source)
            return
    await job_queue.enqueue(title_id, trigger_source)


async def poll_for_subtitle_then_enqueue(
    *,
    video_path: str,
    media_type: MediaType,
    display_name: str,
    sonarr_series_id: int | None,
    sonarr_episode_id: int | None,
    radarr_movie_id: int | None,
    trigger_source: TriggerSource,
    series_title: str | None = None,
    season_number: int | None = None,
    episode_number: int | None = None,
) -> None:
    """Used when Sonarr/Radarr import fires (or an .mkv replacement lands) before
    Bazarr has fetched a subtitle. Polls for the subtitle to show up (bounded by
    settings), then enqueues.
    """
    async with get_session() as session:
        timeout = float(await get_setting(session, "sonarr_radarr_subtitle_poll_timeout_seconds"))
        interval = float(await get_setting(session, "sonarr_radarr_subtitle_poll_interval_seconds"))

    elapsed = 0.0
    while elapsed < timeout:
        subtitle = find_subtitle_for_video(Path(video_path), app_settings.default_subtitle_language)
        if subtitle is not None:
            async with get_session() as session:
                title = await upsert_title(
                    session,
                    media_type=media_type,
                    display_name=display_name,
                    video_path=video_path,
                    sonarr_series_id=sonarr_series_id,
                    sonarr_episode_id=sonarr_episode_id,
                    radarr_movie_id=radarr_movie_id,
                    series_title=series_title,
                    season_number=season_number,
                    episode_number=episode_number,
                )
            await enqueue_if_not_already_active(title.id, trigger_source)
            return
        await asyncio.sleep(interval)
        elapsed += interval

    logger.warning(
        "No subtitle found for %s within %.0fs poll window; will not auto-process. "
        "It can still be processed manually once a subtitle exists.",
        video_path,
        timeout,
    )
    async with get_session() as session:
        result = await session.execute(select(Title).where(Title.video_path == video_path))
        title = result.scalar_one_or_none()
        if title is not None and title.status == "awaiting_subtitle":
            title.status = "not_processed"
            await session.commit()
