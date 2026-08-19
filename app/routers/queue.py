from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import JobState, ProcessingJob, Title
from app.queue.worker import job_queue

router = APIRouter(prefix="/queue", tags=["queue"])
templates = Jinja2Templates(directory="app/templates")

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


@router.get("", response_class=HTMLResponse)
async def queue_page(request: Request):
    active, recent, running, waiting = await _load_jobs()
    return templates.TemplateResponse(
        "queue.html",
        {
            "request": request,
            "active": active,
            "recent": recent,
            "running": running,
            "total_active": len(active),
            "waiting": waiting,
        },
    )


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
    return templates.TemplateResponse(
        "partials/queue_topbar.html",
        {"request": request, "active": processing},
    )


@router.get("/partial", response_class=HTMLResponse)
async def queue_partial(request: Request):
    active, recent, running, waiting = await _load_jobs()
    return templates.TemplateResponse(
        "partials/queue_table.html",
        {
            "request": request,
            "active": active,
            "recent": recent,
            "running": running,
            "total_active": len(active),
            "waiting": waiting,
        },
    )


@router.post("/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_job(request: Request, job_id: int):
    success, error = await job_queue.cancel_job(job_id)
    active, recent, running, waiting = await _load_jobs()
    return templates.TemplateResponse(
        "partials/queue_table.html",
        {
            "request": request,
            "active": active,
            "recent": recent,
            "running": running,
            "total_active": len(active),
            "waiting": waiting,
            "cancel_error": None if success else error,
        },
    )
