"""Second, small job dispatcher for scene-detection work (scan now, blur in a
later phase). Deliberately not merged into app.queue.worker's JobQueue -- that
class already drives all of the proven audio-pipeline's behavior/tests, and
generalizing it to be job-kind-polymorphic is a real refactor of working code
for a mostly-aesthetic benefit. A little duplication (this file) is accepted
instead; see the scene-detection plan for the reasoning."""

import asyncio
import datetime
import logging
import time

from sqlalchemy import select

from app.db.models import JobState, SceneJob, SceneJobKind, Title
from app.db.session import get_session, get_setting
from app.queue.worker import is_within_off_hours_window
from app.scenes.pipeline import scan_for_scenes

logger = logging.getLogger(__name__)

_PROGRESS_UPDATE_MIN_INTERVAL_SECONDS = 1.5


def _parse_hhmm(value: str) -> datetime.time:
    hour, minute = value.split(":")
    return datetime.time(hour=int(hour), minute=int(minute))


class SceneJobQueue:
    def __init__(self) -> None:
        self._pending: asyncio.Queue[int] = asyncio.Queue()
        self._running: dict[int, asyncio.Task] = {}
        self._dispatcher_task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        # Re-queue anything left mid-flight from a prior process crash/restart.
        async with get_session() as session:
            result = await session.execute(
                select(SceneJob).where(SceneJob.state.in_([JobState.queued, JobState.processing]))
            )
            for job in result.scalars().all():
                job.state = JobState.queued
                job.started_at = None
                title = await session.get(Title, job.title_id)
                if title is not None:
                    title.scene_scan_status = "queued"
            await session.commit()

            result = await session.execute(select(SceneJob.id).where(SceneJob.state == JobState.queued))
            for (job_id,) in result.all():
                await self._pending.put(job_id)

        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
        for task in list(self._running.values()):
            task.cancel()

    async def enqueue_scan(self, title_id: int) -> int:
        async with get_session() as session:
            job = SceneJob(title_id=title_id, kind=SceneJobKind.scan, state=JobState.queued)
            session.add(job)
            title = await session.get(Title, title_id)
            if title is not None:
                title.scene_scan_status = "queued"
            await session.commit()
            await session.refresh(job)
        await self._pending.put(job.id)
        return job.id

    async def _dispatch_loop(self) -> None:
        while not self._stopping:
            try:
                async with get_session() as session:
                    cap = int(await get_setting(session, "scene_scan_concurrency_cap"))
                    off_hours_enabled = bool(await get_setting(session, "off_hours_enabled"))
                    off_start = _parse_hhmm(await get_setting(session, "off_hours_start"))
                    off_end = _parse_hhmm(await get_setting(session, "off_hours_end"))

                allowed_to_run = True
                if off_hours_enabled:
                    now = datetime.datetime.now().time()
                    allowed_to_run = is_within_off_hours_window(now, off_start, off_end)

                if allowed_to_run:
                    while len(self._running) < cap and not self._pending.empty():
                        job_id = self._pending.get_nowait()
                        task = asyncio.create_task(self._run_job(job_id))
                        self._running[job_id] = task
                        task.add_done_callback(lambda t, jid=job_id: self._running.pop(jid, None))

                await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scene dispatcher loop error")
                await asyncio.sleep(5)

    async def _run_job(self, job_id: int) -> None:
        async with get_session() as session:
            job = await session.get(SceneJob, job_id)
            if job is None:
                return
            if job.state == JobState.cancelled:
                return
            title = await session.get(Title, job.title_id)
            if title is None:
                job.state = JobState.failed
                job.error = "Title no longer exists"
                job.finished_at = datetime.datetime.utcnow()
                await session.commit()
                return

            job.state = JobState.processing
            job.started_at = datetime.datetime.utcnow()
            job.progress_message = "Scanning for scenes"
            job.progress_percent = 0.0
            title.scene_scan_status = "scanning"
            await session.commit()

            last_update_monotonic = 0.0

            async def on_progress(done: int, total: int) -> None:
                nonlocal last_update_monotonic
                if total <= 0:
                    return
                now = time.monotonic()
                is_final = done >= total
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                job.progress_percent = round(100 * done / total, 1)
                job.progress_message = f"Scanning for scenes -- frame {done}/{total}"
                await session.commit()

            try:
                if job.kind != SceneJobKind.scan:
                    raise NotImplementedError(f"SceneJobKind.{job.kind.value} not yet implemented")

                outcome = await scan_for_scenes(session, title, on_progress=on_progress)

                job.state = JobState.done
                job.progress_message = f"Done -- found {outcome.candidate_count} candidate scene(s)"
                job.progress_percent = 100.0
                job.finished_at = datetime.datetime.utcnow()
                title.scene_scan_status = "scanned" if outcome.candidate_count else "no_scenes_found"

            except asyncio.CancelledError:
                job.state = JobState.cancelled
                job.error = "Cancelled by user"
                job.finished_at = datetime.datetime.utcnow()
                title.scene_scan_status = "not_scanned"
                await session.commit()
                raise  # required so the Task itself is properly marked cancelled

            except Exception as exc:  # noqa: BLE001 -- surface all failures to the UI
                logger.exception("Scene scan failed for title %s", title.id)
                job.state = JobState.failed
                job.error = str(exc)
                job.finished_at = datetime.datetime.utcnow()
                title.scene_scan_status = "scan_failed"

            await session.commit()


scene_job_queue = SceneJobQueue()
