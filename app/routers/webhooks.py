import asyncio
import hmac
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.db.models import MediaType, TriggerSource
from app.db.session import get_session, get_setting
from app.domain import tags_request_audio, tags_request_video
from app.integrations import bazarr as bazarr_integration
from app.integrations import radarr as radarr_integration
from app.integrations import sonarr as sonarr_integration
from app.integrations.subtitle_lookup import find_subtitle_for_video
from app.library import enqueue_if_not_already_active, poll_for_subtitle_then_enqueue, spawn_background, upsert_title
from app.queue.scene_worker import scene_job_queue

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
    # Fail closed, not open -- an empty stored token used to skip the check
    # entirely (auth silently disabled) rather than reject every request. In
    # practice webhook_token is always non-empty (DEFAULT_SETTINGS seeds a
    # random one, the settings form refuses to save a blank value), but
    # nothing structurally guarantees that stays true, so treat "empty" as
    # "misconfigured" rather than "no auth required".
    if not token or not hmac.compare_digest(request.query_params.get("token", ""), token):
        raise HTTPException(status_code=401, detail="Invalid or missing webhook token")


async def _fetch_tags_safe(fetch_coro) -> set[str]:
    """Any failure (Radarr/Sonarr unreachable, bad API key, 404, unexpected payload
    shape, etc.) degrades to "no tags found" -- exactly today's behavior for an
    untagged title. Never blocks or fails the webhook; a tag-fetch problem can only
    ever under-trigger relative to what the user intended, never over-trigger."""
    try:
        return await fetch_coro
    except Exception:
        logger.exception("Failed to fetch Radarr/Sonarr tags; treating as no tags")
        return set()


@router.post("/sonarr")
async def sonarr_webhook(request: Request):
    await _check_token(request)
    async with get_session() as session:
        trigger_enabled = bool(await get_setting(session, "trigger_sonarr_radarr_enabled"))
        default_subtitle_language = await get_setting(session, "default_subtitle_language")
        sonarr_url = await get_setting(session, "sonarr_url")
        sonarr_api_key = await get_setting(session, "sonarr_api_key")

    payload = await request.json()
    parsed = sonarr_integration.parse_import_webhook(payload)
    if parsed is None:
        return {"status": "ignored", "reason": "not an import/upgrade event"}

    # Title sync always happens now, regardless of trigger_sonarr_radarr_enabled or
    # tags -- a vulgarr-video-tagged title needs to be synced+scanned even when the
    # audio auto-trigger setting is off, and the video trigger below has no
    # subtitle dependency to branch on the way the audio path does.
    async with get_session() as session:
        title = await upsert_title(
            session,
            media_type=MediaType.episode,
            display_name=parsed["display_name"],
            video_path=parsed["video_path"],
            sonarr_series_id=parsed["series_id"],
            sonarr_episode_id=parsed["episode_id"],
            series_title=parsed["series_title"],
            season_number=parsed["season_number"],
            episode_number=parsed["episode_number"],
        )

    tags: set[str] = set()
    if sonarr_url and sonarr_api_key and parsed["series_id"]:
        client = sonarr_integration.SonarrClient(sonarr_url, sonarr_api_key)
        tags = await _fetch_tags_safe(client.get_series_tag_labels(parsed["series_id"]))

    should_run_audio = trigger_enabled or tags_request_audio(tags)
    should_run_video = tags_request_video(tags)

    if should_run_video:
        await scene_job_queue.enqueue_scan_if_not_already_active(title.id)

    if should_run_audio:
        video_path = parsed["video_path"]
        subtitle = find_subtitle_for_video(Path(video_path), default_subtitle_language)
        if subtitle is not None:
            await enqueue_if_not_already_active(title.id, TriggerSource.sonarr)
            return {"status": "queued"}
        spawn_background(
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

    if should_run_video:
        return {"status": "queued_video_only"}
    return {"status": "ignored", "reason": "no auto-trigger for this title"}


@router.post("/radarr")
async def radarr_webhook(request: Request):
    await _check_token(request)
    async with get_session() as session:
        trigger_enabled = bool(await get_setting(session, "trigger_sonarr_radarr_enabled"))
        default_subtitle_language = await get_setting(session, "default_subtitle_language")
        radarr_url = await get_setting(session, "radarr_url")
        radarr_api_key = await get_setting(session, "radarr_api_key")

    payload = await request.json()
    parsed = radarr_integration.parse_import_webhook(payload)
    if parsed is None:
        return {"status": "ignored", "reason": "not an import/upgrade event"}

    async with get_session() as session:
        title = await upsert_title(
            session,
            media_type=MediaType.movie,
            display_name=parsed["display_name"],
            video_path=parsed["video_path"],
            radarr_movie_id=parsed["movie_id"],
        )

    tags: set[str] = set()
    if radarr_url and radarr_api_key and parsed["movie_id"]:
        client = radarr_integration.RadarrClient(radarr_url, radarr_api_key)
        tags = await _fetch_tags_safe(client.get_movie_tag_labels(parsed["movie_id"]))

    should_run_audio = trigger_enabled or tags_request_audio(tags)
    should_run_video = tags_request_video(tags)

    if should_run_video:
        await scene_job_queue.enqueue_scan_if_not_already_active(title.id)

    if should_run_audio:
        video_path = parsed["video_path"]
        subtitle = find_subtitle_for_video(Path(video_path), default_subtitle_language)
        if subtitle is not None:
            await enqueue_if_not_already_active(title.id, TriggerSource.radarr)
            return {"status": "queued"}
        spawn_background(
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

    if should_run_video:
        return {"status": "queued_video_only"}
    return {"status": "ignored", "reason": "no auto-trigger for this title"}


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
