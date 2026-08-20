import datetime

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from app.db.models import DetectedScene, Title
from app.db.session import get_session, get_setting
from app.domain import SceneReviewStatus
from app.queue.scene_worker import scene_job_queue

router = APIRouter(tags=["scenes"])
templates = Jinja2Templates(directory="app/templates")


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
