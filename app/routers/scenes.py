import asyncio
import datetime
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from app.config import settings as app_settings
from app.db.models import DetectedScene, Title
from app.db.session import get_session, get_setting
from app.domain import SceneReviewStatus
from app.queue.scene_worker import scene_job_queue
from app.vision.classifier import extract_frame

router = APIRouter(tags=["scenes"])
templates = Jinja2Templates(directory="app/templates")

# Only the containers this app's own media actually shows up in (movies/episodes
# synced from Sonarr/Radarr) -- unrecognized suffixes fall back to a generic type
# below, which most browsers still handle fine via content-sniffing.
_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_STREAM_CHUNK_SIZE = 1024 * 1024


async def _scene_review_context(session, title_id: int) -> dict:
    title = await session.get(Title, title_id)
    scenes_result = await session.execute(
        select(DetectedScene).where(DetectedScene.title_id == title_id).order_by(DetectedScene.start_seconds)
    )
    scenes = scenes_result.scalars().all()
    scene_detection_enabled = bool(await get_setting(session, "scene_detection_enabled"))
    return {
        "title_id": title_id,
        "scene_scan_status": title.scene_scan_status if title else "not_scanned",
        "scenes": scenes,
        "scene_detection_enabled": scene_detection_enabled,
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
    """Standalone scene-review page. Movies already get this inline on their own
    detail page (title_detail_card.html), but episodes have no standalone detail
    page at all -- title_href() routes them straight to their season's table
    (app/domain.py) -- so this is the only entry point for episodes."""
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
    async with get_session() as session:
        scene_detection_enabled = bool(await get_setting(session, "scene_detection_enabled"))
    if scene_detection_enabled:
        await scene_job_queue.enqueue_scan(title_id)
    return await _render_scene_review(request, title_id)


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


async def _file_range_iterator(path: Path, start: int, end: int, chunk_size: int = _STREAM_CHUNK_SIZE):
    """Yields [start, end] (inclusive) of path in chunk_size pieces. Every blocking
    file op (open+seek, each read, close) is pushed to a worker thread via
    asyncio.to_thread -- media lives on an NFS mount in this deployment (see
    app/mux/remux.py's backup-move comment for the same concern), and a synchronous
    read directly on the event loop would stall the whole app for as long as a slow
    network read takes, not just this one response."""

    def _open_and_seek():
        f = open(path, "rb")
        f.seek(start)
        return f

    handle = await asyncio.to_thread(_open_and_seek)
    try:
        remaining = end - start + 1
        while remaining > 0:
            chunk = await asyncio.to_thread(handle.read, min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)


@router.get("/library/title/{title_id}/video")
async def stream_video(title_id: int, request: Request):
    """Range-request-capable video streaming, for the in-browser scene-preview
    player (see toggleScenePreview() in base.html) -- lets a <video> element seek
    directly to a candidate scene's timestamp instead of downloading the whole
    file. Starlette's own FileResponse doesn't implement Range support (confirmed
    against the installed version), hence this hand-rolled version."""
    async with get_session() as session:
        title = await session.get(Title, title_id)
    if title is None:
        raise HTTPException(status_code=404)

    path = Path(title.video_path)
    if not path.exists():
        raise HTTPException(status_code=404)

    file_size = path.stat().st_size
    media_type = _VIDEO_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")

    range_header = request.headers.get("range")
    if range_header:
        match = _RANGE_RE.match(range_header)
        if not match:
            raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
        start_str, end_str = match.groups()
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
        return StreamingResponse(
            _file_range_iterator(path, start, end),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
            },
        )

    return StreamingResponse(
        _file_range_iterator(path, 0, file_size - 1),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@router.get("/library/title/{title_id}/scene-thumbnail")
async def scene_thumbnail(title_id: int, t: float):
    """One representative frame at timestamp t, for the review list's thumbnail
    and the preview player's poster image -- so approving isn't just trusting the
    classifier's confidence number blind."""
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

    return Response(content=data, media_type="image/jpeg")


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
