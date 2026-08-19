import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.db.models import MediaType, TriggerSource
from app.db.session import get_session, get_setting
from app.integrations import bazarr as bazarr_integration
from app.integrations import radarr as radarr_integration
from app.integrations import sonarr as sonarr_integration
from app.integrations.subtitle_lookup import find_subtitle_for_video
from app.library import enqueue_if_not_already_active, poll_for_subtitle_then_enqueue, upsert_title

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)

_EPISODE_PATH_HINT_RE = re.compile(r"season\s*\d+|s\d{2}e\d{2}", re.IGNORECASE)


def _guess_media_type(video_path: Path) -> MediaType:
    """Best-effort guess used only when Bazarr's webhook is the first time we've
    seen this file (i.e. no Title row from Sonarr/Radarr exists yet to tell us).
    Once Sonarr/Radarr populates the library normally, upsert_title preserves
    whatever media_type the row already has -- this guess is never overwritten.
    """
    return MediaType.episode if _EPISODE_PATH_HINT_RE.search(str(video_path)) else MediaType.movie


async def _check_token(request: Request) -> None:
    async with get_session() as session:
        token = await get_setting(session, "webhook_token")
    if not token:
        return
    if request.query_params.get("token") != token:
        raise HTTPException(status_code=401, detail="Invalid or missing webhook token")


@router.post("/sonarr")
async def sonarr_webhook(request: Request):
    await _check_token(request)
    async with get_session() as session:
        if not await get_setting(session, "trigger_sonarr_radarr_enabled"):
            return {"status": "ignored", "reason": "sonarr/radarr trigger disabled in settings"}
        default_subtitle_language = await get_setting(session, "default_subtitle_language")

    payload = await request.json()
    parsed = sonarr_integration.parse_import_webhook(payload)
    if parsed is None:
        return {"status": "ignored", "reason": "not an import/upgrade event"}

    video_path = parsed["video_path"]
    subtitle = find_subtitle_for_video(Path(video_path), default_subtitle_language)

    if subtitle is not None:
        async with get_session() as session:
            title = await upsert_title(
                session,
                media_type=MediaType.episode,
                display_name=parsed["display_name"],
                video_path=video_path,
                sonarr_series_id=parsed["series_id"],
                sonarr_episode_id=parsed["episode_id"],
                series_title=parsed["series_title"],
                season_number=parsed["season_number"],
                episode_number=parsed["episode_number"],
            )
        await enqueue_if_not_already_active(title.id, TriggerSource.sonarr)
        return {"status": "queued"}

    asyncio.create_task(
        poll_for_subtitle_then_enqueue(
            video_path=video_path,
            media_type=MediaType.episode,
            display_name=parsed["display_name"],
            sonarr_series_id=parsed["series_id"],
            sonarr_episode_id=parsed["episode_id"],
            radarr_movie_id=None,
            trigger_source=TriggerSource.sonarr,
            series_title=parsed["series_title"],
            season_number=parsed["season_number"],
            episode_number=parsed["episode_number"],
        )
    )
    return {"status": "polling_for_subtitle"}


@router.post("/radarr")
async def radarr_webhook(request: Request):
    await _check_token(request)
    async with get_session() as session:
        if not await get_setting(session, "trigger_sonarr_radarr_enabled"):
            return {"status": "ignored", "reason": "sonarr/radarr trigger disabled in settings"}
        default_subtitle_language = await get_setting(session, "default_subtitle_language")

    payload = await request.json()
    parsed = radarr_integration.parse_import_webhook(payload)
    if parsed is None:
        return {"status": "ignored", "reason": "not an import/upgrade event"}

    video_path = parsed["video_path"]
    subtitle = find_subtitle_for_video(Path(video_path), default_subtitle_language)

    if subtitle is not None:
        async with get_session() as session:
            title = await upsert_title(
                session,
                media_type=MediaType.movie,
                display_name=parsed["display_name"],
                video_path=video_path,
                radarr_movie_id=parsed["movie_id"],
            )
        await enqueue_if_not_already_active(title.id, TriggerSource.radarr)
        return {"status": "queued"}

    asyncio.create_task(
        poll_for_subtitle_then_enqueue(
            video_path=video_path,
            media_type=MediaType.movie,
            display_name=parsed["display_name"],
            sonarr_series_id=None,
            sonarr_episode_id=None,
            radarr_movie_id=parsed["movie_id"],
            trigger_source=TriggerSource.radarr,
        )
    )
    return {"status": "polling_for_subtitle"}


@router.post("/bazarr")
async def bazarr_webhook(request: Request):
    await _check_token(request)
    async with get_session() as session:
        if not await get_setting(session, "trigger_bazarr_enabled"):
            return {"status": "ignored", "reason": "bazarr trigger disabled in settings"}

    parsed = bazarr_integration.parse_subtitle_downloaded_event(dict(request.query_params))
    if parsed is None:
        raise HTTPException(status_code=400, detail="Missing video_path/subtitle_path query params")

    video_path = Path(parsed["video_path"])
    async with get_session() as session:
        title = await upsert_title(
            session,
            media_type=_guess_media_type(video_path),
            display_name=video_path.stem,
            video_path=str(video_path),
        )
    await enqueue_if_not_already_active(title.id, TriggerSource.bazarr)
    return {"status": "queued"}
