import asyncio
import datetime
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from app.config import settings as app_settings
from app.db.models import DetectedScene, SceneJob, Title
from app.db.session import get_session, get_setting
from app.domain import SceneJobKind, SceneReviewStatus
from app.queue.scene_worker import scene_job_queue
from app.vision.classifier import extract_frame

router = APIRouter(tags=["scenes"])
templates = Jinja2Templates(directory="app/templates")


async def _scene_review_context(session, title_id: int) -> dict:
    title = await session.get(Title, title_id)
    scenes_result = await session.execute(
        select(DetectedScene).where(DetectedScene.title_id == title_id).order_by(DetectedScene.start_seconds)
    )
    scenes = scenes_result.scalars().all()
    high_confidence_fraction = float(await get_setting(session, "scene_high_confidence_fraction"))

    # The most recent job for this title, whatever its state -- while it's still
    # queued/processing this is what drives the real progress%/message shown
    # instead of a bare "Scanning…" spinner; once done it's still useful context
    # (e.g. a failed job's error message) even though scene_scan_status alone
    # already reflects the outcome.
    scan_job = (
        await session.execute(
            select(SceneJob)
            .where(SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.scan)
            .order_by(SceneJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    # Same reasoning as scan_job above, for the blur/Apply side.
    blur_job = (
        await session.execute(
            select(SceneJob)
            .where(SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.blur)
            .order_by(SceneJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    pending_count = sum(1 for s in scenes if s.status == SceneReviewStatus.pending)
    high_confidence_pending_count = sum(
        1
        for s in scenes
        if s.status == SceneReviewStatus.pending
        and s.verified_fraction is not None
        and s.verified_fraction >= high_confidence_fraction
    )

    return {
        "title_id": title_id,
        "scene_scan_status": title.scene_scan_status if title else "not_scanned",
        "scenes": scenes,
        "scan_job": scan_job,
        "blur_job": blur_job,
        "vulgarr_edit_path": title.vulgarr_edit_path if title else None,
        "high_confidence_fraction": high_confidence_fraction,
        "pending_count": pending_count,
        "high_confidence_pending_count": high_confidence_pending_count,
    }


async def _render_scene_review(request: Request, title_id: int) -> HTMLResponse:
    async with get_session() as session:
        context = await _scene_review_context(session, title_id)
    return templates.TemplateResponse(
        "partials/scene_review_list.html", {"request": request, **context}
    )


@router.get("/library/title/{title_id}/scenes", response_class=HTMLResponse)
async def scene_review_refresh(request: Request, title_id: int):
    """Self-polling target for scene_review_list.html while a scan is in progress
    (see its own hx-get) -- same idea as title_row.html's self-poll on
    queued/processing status."""
    return await _render_scene_review(request, title_id)


@router.get("/library/title/{title_id}/scene-review", response_class=HTMLResponse)
async def scene_review_page(request: Request, title_id: int):
    """Standalone scene-review page. Every title (movie or episode) also gets
    this same list embedded inline on its own detail page
    (title_detail_card.html, reachable via title_href() for both media
    types) -- this route exists as a quicker jump straight to just this
    section, e.g. from title_row.html's "Review" link, without opening the
    full detail page."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
        if title is None:
            raise HTTPException(status_code=404)
        context = await _scene_review_context(session, title_id)
    return templates.TemplateResponse(
        "scene_review_page.html", {"request": request, "title": title, **context}
    )


@router.post("/library/title/{title_id}/scan-scenes", response_class=HTMLResponse)
async def scan_scenes(request: Request, title_id: int):
    await scene_job_queue.enqueue_scan_if_not_already_active(title_id)
    return await _render_scene_review(request, title_id)


@router.post("/library/title/{title_id}/apply-scenes", response_class=HTMLResponse)
async def apply_scenes(request: Request, title_id: int):
    """Rebuilds the "Vulgarr Edit" sibling file from every currently-approved
    scene on this title. Real work happens in the queue worker
    (apply_scene_blur via SceneJobKind.blur) -- this just enqueues, same shape
    as scan-scenes above.

    Gated on "any approved scene exists" rather than "approved and unapplied"
    -- this is also the manual trigger for the "reject a scene that was
    already baked into an earlier Vulgarr Edit, then re-process" flow (see
    apply_scene_blur), where every approved scene may already carry an
    applied_at from the prior run but the file still needs regenerating to
    drop the one that's no longer approved."""
    async with get_session() as session:
        approved = await session.execute(
            select(DetectedScene).where(
                DetectedScene.title_id == title_id,
                DetectedScene.status == SceneReviewStatus.approved,
            )
        )
        has_approved = approved.first() is not None
    if has_approved:
        await scene_job_queue.enqueue_blur_if_not_already_active(title_id)
    return await _render_scene_review(request, title_id)


@router.post("/library/shows/{series_id}/scan-scenes", response_class=HTMLResponse)
async def scan_scenes_show(request: Request, series_id: int):
    from app.db.models import MediaType
    from app.routers.library import SEVERITY_OPTIONS, _load_seasons

    async with get_session() as session:
        result = await session.execute(
            select(Title.id).where(Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
        )
        title_ids = [row[0] for row in result.all()]
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, seasons = await _load_seasons(session, series_id, current_version)

    queued = sum([await scene_job_queue.enqueue_scan_if_not_already_active(title_id) for title_id in title_ids])
    scan_message = f"Queued {queued} episode(s) for scene scanning."

    return templates.TemplateResponse(
        "show_detail.html",
        {
            "request": request,
            "series_id": series_id,
            "series_title": series_title,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
            "scan_message": scan_message,
        },
    )


@router.post("/library/shows/{series_id}/season/{season_number}/scan-scenes", response_class=HTMLResponse)
async def scan_scenes_season(request: Request, series_id: int, season_number: int):
    from app.db.models import MediaType
    from app.routers.library import _load_season

    async with get_session() as session:
        result = await session.execute(
            select(Title.id).where(
                Title.sonarr_series_id == series_id,
                Title.season_number == season_number,
                Title.media_type == MediaType.episode,
            )
        )
        title_ids = [row[0] for row in result.all()]
        current_version = int(await get_setting(session, "wordlist_version"))
        season = await _load_season(session, series_id, season_number, current_version)

    queued = sum([await scene_job_queue.enqueue_scan_if_not_already_active(title_id) for title_id in title_ids])
    scan_message = f"Queued {queued} episode(s) for scene scanning."

    return templates.TemplateResponse(
        "partials/season_row.html",
        {"request": request, "series_id": series_id, "season": season, "scan_message": scan_message},
    )


@router.post("/scene-jobs/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_scene_job(request: Request, job_id: int):
    """Standalone cancel action (queue page context, not tied to one title's review
    section) -- unlike the other scene endpoints, this re-renders the main queue
    partial, not scene_review_list.html."""
    from app.routers.queue import _render_queue_partial

    success, error = await scene_job_queue.cancel_job(job_id)
    return await _render_queue_partial(request, cancel_error=None if success else error)


@router.post("/scenes/{scene_id}/approve", response_class=HTMLResponse)
async def approve_scene(request: Request, scene_id: int):
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        title_id = scene.title_id if scene else None
        if scene is not None:
            scene.status = SceneReviewStatus.approved
            scene.reviewed_at = datetime.datetime.utcnow()
            await session.commit()
    if title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, title_id)


@router.post("/scenes/{scene_id}/reject", response_class=HTMLResponse)
async def reject_scene(request: Request, scene_id: int):
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        title_id = scene.title_id if scene else None
        if scene is not None:
            scene.status = SceneReviewStatus.rejected
            scene.reviewed_at = datetime.datetime.utcnow()
            await session.commit()
    if title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, title_id)


@router.post("/scenes/{scene_id}/toggle-mute", response_class=HTMLResponse)
async def toggle_scene_mute(request: Request, scene_id: int):
    """Flips whether Apply also mutes audio for this scene, on top of always
    blurring the video -- default off (see DetectedScene.mute_audio). Doesn't
    touch status/reviewed_at -- this is independent of approve/reject, and can
    be changed either before or after approving."""
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        title_id = scene.title_id if scene else None
        if scene is not None:
            scene.mute_audio = not scene.mute_audio
            await session.commit()
    if title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, title_id)


@router.post("/library/title/{title_id}/approve-high-confidence", response_class=HTMLResponse)
async def approve_high_confidence_scenes(request: Request, title_id: int):
    """Bulk-approves every still-pending candidate whose dense verification pass
    (see app.scenes.pipeline.scan_for_scenes) cleared scene_high_confidence_fraction
    -- the review-friction reducer for the clear-cut cases, while anything short/
    borderline enough to score low on that fraction still needs an actual look."""
    async with get_session() as session:
        high_confidence_fraction = float(await get_setting(session, "scene_high_confidence_fraction"))
        result = await session.execute(
            select(DetectedScene).where(
                DetectedScene.title_id == title_id,
                DetectedScene.status == SceneReviewStatus.pending,
                DetectedScene.verified_fraction.is_not(None),
                DetectedScene.verified_fraction >= high_confidence_fraction,
            )
        )
        now = datetime.datetime.utcnow()
        for scene in result.scalars().all():
            scene.status = SceneReviewStatus.approved
            scene.reviewed_at = now
        await session.commit()
    return await _render_scene_review(request, title_id)


@router.post("/library/title/{title_id}/approve-all-scenes", response_class=HTMLResponse)
async def approve_all_scenes(request: Request, title_id: int):
    """Bulk-approves every still-pending candidate regardless of verified_fraction
    -- the blunt version of approve-high-confidence above, for anyone who'd rather
    just approve everything a scan found than review one at a time."""
    async with get_session() as session:
        result = await session.execute(
            select(DetectedScene).where(
                DetectedScene.title_id == title_id,
                DetectedScene.status == SceneReviewStatus.pending,
            )
        )
        now = datetime.datetime.utcnow()
        for scene in result.scalars().all():
            scene.status = SceneReviewStatus.approved
            scene.reviewed_at = now
        await session.commit()
    return await _render_scene_review(request, title_id)


# Padding on each side of the requested [start, end], for the same reason the
# preview player pads it -- lets the reviewer judge the boundaries, not just
# confirm the flagged moment.
_CLIP_PAD_SECONDS = 3.0
_CLIP_MAX_DURATION_SECONDS = 60.0  # sanity cap regardless of what a caller requests


def _extract_clip_blocking(ffmpeg_bin: str, video_path: Path, start: float, duration: float, out_path: Path) -> None:
    proc = subprocess.run(
        [
            ffmpeg_bin, "-nostdin", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}",
            # Re-encode to H.264/AAC regardless of the source codec -- real-world
            # rips are often HEVC video and/or AC3/DTS audio, neither of which any
            # mainstream browser decodes natively, MP4 container or not. A clip
            # this short (seconds, capped well below a minute) re-encodes in
            # roughly real time or faster even on modest hardware, so the cost is
            # negligible against the guarantee of it actually playing.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            str(out_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg clip extraction failed at {start:.2f}s+{duration:.2f}s in {video_path}")


@router.get("/library/title/{title_id}/scene-clip")
async def scene_clip(title_id: int, start: float, end: float):
    """A short, re-encoded (guaranteed browser-playable) MP4 clip covering
    [start, end] plus padding -- replaces an earlier whole-file Range-streaming
    approach that turned out not to work in practice: most real media here is
    .mkv (inconsistent <video> container support across browsers even when the
    codecs would otherwise play) and/or HEVC/AC3 (which browsers can't decode
    natively at all, container aside). A short clip sidesteps both problems at
    once instead of trying to solve either."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
    if title is None:
        raise HTTPException(status_code=404)

    video_path = Path(title.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404)

    clip_start = max(0.0, start - _CLIP_PAD_SECONDS)
    clip_duration = min(_CLIP_MAX_DURATION_SECONDS, (end - start) + 2 * _CLIP_PAD_SECONDS)

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_path = Path(tmpdir) / "clip.mp4"
        try:
            await asyncio.to_thread(
                _extract_clip_blocking, app_settings.ffmpeg_bin, video_path, clip_start, clip_duration, clip_path
            )
        except RuntimeError:
            raise HTTPException(status_code=404)
        data = clip_path.read_bytes()

    return Response(content=data, media_type="video/mp4")


@router.get("/library/title/{title_id}/scene-thumbnail")
async def scene_thumbnail(title_id: int, t: float):
    """One representative frame at timestamp t, for the review list's thumbnail
    and the preview player's poster image -- so approving isn't just trusting the
    classifier's confidence number blind.

    Cached aggressively (title_id+t is a deterministic key -- the same source
    frame every time, since the original file is never modified in place) so
    the review list's 3-second self-poll while a scan/blur job is running
    doesn't re-run ffmpeg and re-fetch every thumbnail over the network on each
    tick -- that was visibly flashing every image on the page each poll. The
    one edge case this doesn't cover is Sonarr/Radarr silently replacing the
    underlying file with a different cut -- same staleness caveat that already
    applies to the scene timestamps themselves, not new here."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
    if title is None:
        raise HTTPException(status_code=404)

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = Path(tmpdir) / "thumb.jpg"
        try:
            await asyncio.to_thread(
                extract_frame, app_settings.ffmpeg_bin, Path(title.video_path), max(0.0, t), frame_path
            )
        except RuntimeError:
            raise HTTPException(status_code=404)
        data = frame_path.read_bytes()

    return Response(
        content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"}
    )


@router.post("/scenes/{scene_id}/adjust", response_class=HTMLResponse)
async def adjust_scene(
    request: Request,
    scene_id: int,
    start_seconds: float = Form(...),
    end_seconds: float = Form(...),
):
    async with get_session() as session:
        scene = await session.get(DetectedScene, scene_id)
        title_id = scene.title_id if scene else None
        if scene is not None and end_seconds > start_seconds:
            scene.start_seconds = max(0.0, start_seconds)
            scene.end_seconds = end_seconds
            await session.commit()
    if title_id is None:
        return HTMLResponse("")
    return await _render_scene_review(request, title_id)
