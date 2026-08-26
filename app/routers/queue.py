from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import JobState, ProcessingJob, SceneJob, Title
from app.domain import format_duration, title_href
from app.queue.scene_worker import scene_job_queue
from app.queue.worker import job_queue

router = APIRouter(prefix="/queue", tags=["queue"])
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["title_href"] = title_href
templates.env.globals["format_duration"] = format_duration

# Titles waiting on an external system (Sonarr/Radarr grabbing an .mkv replacement,
# or Bazarr fetching a subtitle) -- not ProcessingJob rows, so they never occupy a
# concurrency-cap slot, but still worth surfacing so it's visible they're not stuck.
_WAITING_STATUSES = ["awaiting_mkv", "awaiting_subtitle"]


async def _load_jobs():
    from app.db.session import get_session

    async with get_session() as session:
        result = await session.execute(
            select(ProcessingJob)
            .options(selectinload(ProcessingJob.title))
            .where(ProcessingJob.state.in_([JobState.queued, JobState.processing]))
            .order_by(ProcessingJob.created_at)
        )
        active = result.scalars().all()

        result = await session.execute(
            select(ProcessingJob)
            .options(selectinload(ProcessingJob.title))
            .where(ProcessingJob.state.in_([JobState.done, JobState.failed, JobState.cancelled]))
            .order_by(ProcessingJob.finished_at.desc())
            .limit(25)
        )
        recent = result.scalars().all()

        result = await session.execute(
            select(Title).where(Title.status.in_(_WAITING_STATUSES)).order_by(Title.replacement_requested_at)
        )
        waiting = result.scalars().all()

    running = sum(1 for j in active if j.state == JobState.processing)
    return active, recent, running, waiting


async def _load_scene_jobs():
    """Mute-pipeline counterpart above lives against ProcessingJob; this is the
    same shape against SceneJob, kept separate (own function, own section on the
    queue page) rather than merged into one table -- the two job types don't share
    enough fields (no trigger_source/wordlist_version on SceneJob, no kind on
    ProcessingJob) for a single template to render both cleanly."""
    from app.db.session import get_session

    async with get_session() as session:
        result = await session.execute(
            select(SceneJob)
            .options(selectinload(SceneJob.title))
            .where(SceneJob.state.in_([JobState.queued, JobState.processing]))
            .order_by(SceneJob.created_at)
        )
        active = result.scalars().all()

        result = await session.execute(
            select(SceneJob)
            .options(selectinload(SceneJob.title))
            .where(SceneJob.state.in_([JobState.done, JobState.failed, JobState.cancelled]))
            .order_by(SceneJob.finished_at.desc())
            .limit(25)
        )
        recent = result.scalars().all()

    running = sum(1 for j in active if j.state == JobState.processing)
    return active, recent, running


def _audio_row(job: ProcessingJob) -> dict:
    return {
        "title": job.title,
        "kind_label": "Audio",
        "detail": job.trigger_source.value,
        "state": job.state,
        "progress_percent": job.progress_percent,
        "progress_message": job.progress_message,
        "error": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "sort_key": job.created_at,
        "cancel_url": f"/queue/{job.id}/cancel",
        "waiting_status": None,
    }


def _scene_row(job: SceneJob) -> dict:
    return {
        "title": job.title,
        "kind_label": "Scene",
        "detail": job.kind.value,
        "state": job.state,
        "progress_percent": job.progress_percent,
        "progress_message": job.progress_message,
        "error": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "sort_key": job.created_at,
        "cancel_url": f"/scene-jobs/{job.id}/cancel",
        "waiting_status": None,
    }


def _waiting_row(title: Title) -> dict:
    return {
        "title": title,
        "kind_label": "Audio",
        "detail": "waiting",
        "state": None,
        "progress_percent": None,
        "progress_message": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "sort_key": title.replacement_requested_at or datetime.min,
        "cancel_url": None,
        "waiting_status": title.status,
    }


async def _queue_context(cancel_error: str | None = None) -> dict:
    active, recent, running, waiting = await _load_jobs()
    scene_active, scene_recent, scene_running = await _load_scene_jobs()

    in_progress = sorted(
        [_audio_row(j) for j in active] + [_scene_row(j) for j in scene_active] + [_waiting_row(t) for t in waiting],
        key=lambda r: r["sort_key"],
    )
    completed = sorted(
        [_audio_row(j) for j in recent] + [_scene_row(j) for j in scene_recent],
        key=lambda r: r["finished_at"] or datetime.min,
        reverse=True,
    )[:30]

    return {
        "in_progress": in_progress,
        "completed": completed,
        "running": running + scene_running,
        "total_active": len(in_progress),
        "cancel_error": cancel_error,
    }


async def _render_queue_partial(request: Request, cancel_error: str | None = None) -> HTMLResponse:
    context = await _queue_context(cancel_error)
    return templates.TemplateResponse("partials/queue_table.html", {"request": request, **context})


@router.get("", response_class=HTMLResponse)
async def queue_page(request: Request):
    context = await _queue_context()
    return templates.TemplateResponse("queue.html", {"request": request, **context})


@router.get("/topbar", response_class=HTMLResponse)
async def queue_topbar(request: Request):
    from app.db.session import get_session

    async with get_session() as session:
        result = await session.execute(
            select(ProcessingJob)
            .options(selectinload(ProcessingJob.title))
            .where(ProcessingJob.state == JobState.processing)
            .order_by(ProcessingJob.started_at)
        )
        processing = result.scalars().all()

        result = await session.execute(
            select(SceneJob)
            .options(selectinload(SceneJob.title))
            .where(SceneJob.state == JobState.processing)
            .order_by(SceneJob.started_at)
        )
        scene_processing = result.scalars().all()

    return templates.TemplateResponse(
        "partials/queue_topbar.html",
        {"request": request, "active": processing, "scene_active": scene_processing},
    )


@router.get("/partial", response_class=HTMLResponse)
async def queue_partial(request: Request):
    return await _render_queue_partial(request)


@router.post("/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_job(request: Request, job_id: int):
    success, error = await job_queue.cancel_job(job_id)
    return await _render_queue_partial(request, cancel_error=None if success else error)
