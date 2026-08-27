"""Second, small job dispatcher for scene-detection work (scan now, blur in a
later phase). Deliberately not merged into app.queue.worker's JobQueue -- that
class already drives all of the proven audio-pipeline's behavior/tests, and
generalizing it to be job-kind-polymorphic is a real refactor of working code
for a mostly-aesthetic benefit. A little duplication (this file) is accepted
instead; see the scene-detection plan for the reasoning."""

import asyncio
import datetime
import logging
import shutil
import time

from sqlalchemy import select, update

from app.config import settings as app_settings
from app.db.models import JobState, SceneJob, SceneJobKind, Title
from app.db.session import get_session, get_setting
from app.queue.worker import is_within_off_hours_window
from app.scenes.pipeline import apply_scene_blur, reverify_scenes_with_claude, scan_for_scenes

logger = logging.getLogger(__name__)

_PROGRESS_UPDATE_MIN_INTERVAL_SECONDS = 1.5

# A scan's per-frame ETA is far more stable than the audio pipeline's ffmpeg-progress
# ETA (uniform-cost work, not variable-length cues), so no warm-up-fraction gating is
# needed here -- shown as soon as there's more than one frame's worth of data.


def _parse_hhmm(value: str) -> datetime.time:
    hour, minute = value.split(":")
    return datetime.time(hour=int(hour), minute=int(minute))


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m remaining"
    if minutes:
        return f"{minutes}m {seconds}s remaining"
    return f"{seconds}s remaining"


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
            active_scan_title_ids: set[int] = set()
            for job in result.scalars().all():
                job.state = JobState.queued
                job.started_at = None
                title = await session.get(Title, job.title_id)
                if title is not None:
                    title.scene_scan_status = "queued"
                if job.kind == SceneJobKind.scan:
                    active_scan_title_ids.add(job.title_id)

            # Defensive reconciliation: a Title stuck showing "queued"/"scanning"
            # with no actual SceneJob behind it (the requeue loop above only
            # re-queues real orphaned *jobs*; this catches an orphaned *status*
            # with no job at all, e.g. from a job being deleted/edited outside
            # the normal cancel_job()/_run_job() paths, which are the only two
            # places that otherwise keep this field in sync). Not expected in
            # normal operation, but cheap to check and prevents a permanently
            # stuck-looking scan indicator either way.
            result = await session.execute(select(Title).where(Title.scene_scan_status.in_(["queued", "scanning"])))
            for title in result.scalars().all():
                if title.id not in active_scan_title_ids:
                    title.scene_scan_status = "not_scanned"

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

    async def cancel_job(self, job_id: int) -> tuple[bool, str | None]:
        """Mirrors JobQueue.cancel_job -- same >=97% "too far along" refusal, more
        important here than for a scan (harmless to abandon mid-way) once blur jobs
        (a real file-swap tail step) start using this same queue/dispatcher."""
        task = self._running.get(job_id)
        if task is not None:
            async with get_session() as session:
                job = await session.get(SceneJob, job_id)
                # A task is registered here as soon as it's created, before its
                # own body has necessarily run far enough to flip state to
                # "processing" -- treat that brief "queued" window the same as
                # "processing" rather than rejecting the cancel (see
                # JobQueue.cancel_job for the same fix/reasoning).
                if job is None or job.state not in (JobState.queued, JobState.processing):
                    return False, "Job is no longer running"
                if job.state == JobState.processing and (job.progress_percent or 0) >= 97:
                    return False, "Too far along to cancel -- finishing up"
            task.cancel()
            return True, None

        async with get_session() as session:
            job = await session.get(SceneJob, job_id)
            if job is None or job.state != JobState.queued:
                return False, "Job is not queued or running"
            job.state = JobState.cancelled
            job.error = "Cancelled by user"
            job.finished_at = datetime.datetime.utcnow()
            title = await session.get(Title, job.title_id)
            if title is not None and job.kind == SceneJobKind.scan:
                title.scene_scan_status = "not_scanned"
            await session.commit()
        return True, None

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

    async def enqueue_scan_if_not_already_active(self, title_id: int) -> bool:
        """Mirrors app.library.enqueue_if_not_already_active -- used by both the
        single-title scan button and the bulk season/show scan actions, so a
        double-click or an overlapping bulk trigger doesn't queue a second scan for
        a title that's already queued/processing. Returns whether it actually
        enqueued (vs. skipped as a duplicate)."""
        async with get_session() as session:
            existing = await session.execute(
                select(SceneJob).where(
                    SceneJob.title_id == title_id,
                    SceneJob.kind == SceneJobKind.scan,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                )
            )
            if existing.scalar_one_or_none() is not None:
                return False
        await self.enqueue_scan(title_id)
        return True

    async def enqueue_scan_for_titles_if_not_already_active(self, title_ids: list[int]) -> int:
        """Batched form of enqueue_scan_if_not_already_active for the bulk
        "process both" season/show actions -- one session/transaction for the
        whole list instead of 1-2 DB round trips per title, so a large show
        doesn't turn one click into dozens of sequential queries before the
        request can return. Returns how many were actually enqueued (vs.
        skipped as already active)."""
        if not title_ids:
            return 0
        async with get_session() as session:
            existing = await session.execute(
                select(SceneJob.title_id).where(
                    SceneJob.title_id.in_(title_ids),
                    SceneJob.kind == SceneJobKind.scan,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                )
            )
            already_active = {row[0] for row in existing.all()}
            to_enqueue = [tid for tid in title_ids if tid not in already_active]
            if not to_enqueue:
                return 0
            jobs = [SceneJob(title_id=tid, kind=SceneJobKind.scan, state=JobState.queued) for tid in to_enqueue]
            session.add_all(jobs)
            await session.execute(update(Title).where(Title.id.in_(to_enqueue)).values(scene_scan_status="queued"))
            # Flush (not commit) to assign each job's id while the objects are still
            # attached and readable -- commit would expire them, and this async
            # session can't do the implicit lazy-reload that a sync session would
            # use to re-fetch an expired attribute on next access.
            await session.flush()
            job_ids = [job.id for job in jobs]
            await session.commit()
        for job_id in job_ids:
            await self._pending.put(job_id)
        return len(job_ids)

    async def enqueue_blur(self, title_id: int) -> int:
        async with get_session() as session:
            job = SceneJob(title_id=title_id, kind=SceneJobKind.blur, state=JobState.queued)
            session.add(job)
            await session.commit()
            await session.refresh(job)
        await self._pending.put(job.id)
        return job.id

    async def enqueue_blur_if_not_already_active(self, title_id: int) -> bool:
        """Same dedup reasoning as enqueue_scan_if_not_already_active -- a
        double-click on "Apply" shouldn't queue a second multi-hour re-encode for
        the same title."""
        async with get_session() as session:
            existing = await session.execute(
                select(SceneJob).where(
                    SceneJob.title_id == title_id,
                    SceneJob.kind == SceneJobKind.blur,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                )
            )
            if existing.scalar_one_or_none() is not None:
                return False
        await self.enqueue_blur(title_id)
        return True

    async def enqueue_claude_verify(self, title_id: int) -> int:
        async with get_session() as session:
            job = SceneJob(title_id=title_id, kind=SceneJobKind.claude_verify, state=JobState.queued)
            session.add(job)
            await session.commit()
            await session.refresh(job)
        await self._pending.put(job.id)
        return job.id

    async def enqueue_claude_verify_if_not_already_active(self, title_id: int) -> bool:
        """Same dedup reasoning as enqueue_scan_if_not_already_active -- a double-click
        on "Reprocess with Claude Vision" shouldn't queue a second pass over the same
        title's candidates while one is already running."""
        async with get_session() as session:
            existing = await session.execute(
                select(SceneJob).where(
                    SceneJob.title_id == title_id,
                    SceneJob.kind == SceneJobKind.claude_verify,
                    SceneJob.state.in_([JobState.queued, JobState.processing]),
                )
            )
            if existing.scalar_one_or_none() is not None:
                return False
        await self.enqueue_claude_verify(title_id)
        return True

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

            is_scan = job.kind == SceneJobKind.scan
            is_claude_verify = job.kind == SceneJobKind.claude_verify
            job.state = JobState.processing
            job.started_at = datetime.datetime.utcnow()
            job.progress_percent = 0.0
            if is_scan:
                job.progress_message = "Scanning for scenes"
                title.scene_scan_status = "scanning"
            elif is_claude_verify:
                job.progress_message = "Verifying scenes with Claude Vision"
            else:
                job.progress_message = "Blurring video"
            await session.commit()

            job_start_monotonic = time.monotonic()
            last_update_monotonic = 0.0

            async def on_scan_progress(done: int, total: int) -> None:
                nonlocal last_update_monotonic
                if total <= 0:
                    return
                now = time.monotonic()
                is_final = done >= total
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                fraction = done / total
                job.progress_percent = round(100 * fraction, 1)
                message = f"Scanning for scenes -- frame {done}/{total}"
                if fraction > 0 and not is_final:
                    elapsed = now - job_start_monotonic
                    eta_seconds = elapsed * (1 - fraction) / fraction
                    message += f", {_format_eta(eta_seconds)}"
                job.progress_message = message
                await session.commit()

            # Unlike app.queue.worker's mute pipeline (one whole-file ffmpeg pass,
            # where a 97% floor reserves the last few points for the post-encode
            # verify/publish stage below), app.mux.scene_blur's split/re-encode/
            # concat pipeline calls on_stage throughout the whole job (once per
            # scene segment, not just at the tail), so on_stage below no longer
            # forces a progress floor -- fraction naturally reaches ~97% once all
            # scene segments are done re-encoding, before verify/publish ever run.
            # (When any approved scene has mute_audio set, apply_scene_blur scales
            # this fraction down to leave room for the audio-mute phase that
            # follows -- see its docstring -- so this doesn't hit 97% until that
            # phase, which on_stage alone doesn't move progress_percent for, has
            # also actually finished.)
            async def on_blur_progress(fraction: float) -> None:
                nonlocal last_update_monotonic
                now = time.monotonic()
                is_final = fraction >= 0.999
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                job.progress_percent = round(fraction * 97.0, 1)
                message = f"Blurring video (ffmpeg) -- {round(fraction * 100):.0f}%"
                if fraction > 0.02 and not is_final:
                    elapsed = now - job_start_monotonic
                    eta_seconds = elapsed * (1 - fraction) / fraction
                    message += f", {_format_eta(eta_seconds)}"
                job.progress_message = message
                await session.commit()

            async def on_blur_stage(message: str) -> None:
                job.progress_message = message
                await session.commit()

            async def on_scan_stage(message: str) -> None:
                # No progress-percent bump like on_blur_stage's 97% floor -- the
                # frame-count progress above already reaches 100% once the main
                # scan finishes, so the per-candidate verification pass that
                # follows is a (typically short) tail step, not something that
                # needs its own slice of the bar.
                job.progress_message = message
                await session.commit()

            async def on_claude_verify_progress(done: int, total: int) -> None:
                nonlocal last_update_monotonic
                if total <= 0:
                    return
                now = time.monotonic()
                is_final = done >= total
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now
                job.progress_percent = round(100 * done / total, 1)
                job.progress_message = f"Verifying with Claude Vision -- scene {done}/{total}"
                await session.commit()

            try:
                if is_scan:
                    scan_outcome = await scan_for_scenes(
                        session, title, job.id, on_progress=on_scan_progress, on_stage=on_scan_stage
                    )
                    job.progress_message = f"Done -- found {scan_outcome.candidate_count} candidate scene(s)"
                    title.scene_scan_status = "scanned" if scan_outcome.candidate_count else "no_scenes_found"

                    # Chain straight into a blur/Apply job when auto-process is on
                    # and this scan actually found something -- scan_for_scenes
                    # already auto-approved the candidates above (same setting
                    # gates both halves), this just closes the loop so the user
                    # doesn't have to separately click Apply. Committed before
                    # enqueuing so the blur job's own session sees the scenes as
                    # approved. A concurrent manual scan-scenes/apply-scenes click
                    # can't double-enqueue here -- enqueue_blur_if_not_already_active
                    # checks for an existing queued/processing blur job first.
                    if scan_outcome.candidate_count and bool(await get_setting(session, "scene_auto_process")):
                        await session.commit()
                        await self.enqueue_blur_if_not_already_active(title.id)
                elif is_claude_verify:
                    reverify_outcome = await reverify_scenes_with_claude(
                        session, title, on_progress=on_claude_verify_progress
                    )
                    job.progress_message = (
                        f"Done -- checked {reverify_outcome.checked_count} scene(s), "
                        f"{reverify_outcome.approved_count} confirmed, "
                        f"{reverify_outcome.rejected_count} rejected"
                        + (f", {reverify_outcome.failed_count} failed" if reverify_outcome.failed_count else "")
                    )
                else:
                    blur_outcome = await apply_scene_blur(
                        session, title, job.id, on_progress=on_blur_progress, on_stage=on_blur_stage
                    )
                    job.progress_message = f"Done -- blurred {blur_outcome.scene_count} scene(s)"

                job.state = JobState.done
                job.progress_percent = 100.0
                job.finished_at = datetime.datetime.utcnow()

            except asyncio.CancelledError:
                job.state = JobState.cancelled
                job.error = "Cancelled by user"
                job.finished_at = datetime.datetime.utcnow()
                if is_scan:
                    title.scene_scan_status = "not_scanned"
                await session.commit()
                raise  # required so the Task itself is properly marked cancelled

            except Exception as exc:  # noqa: BLE001 -- surface all failures to the UI
                logger.exception("Scene %s failed for title %s", job.kind.value, title.id)
                job.state = JobState.failed
                job.error = str(exc)
                job.finished_at = datetime.datetime.utcnow()
                if is_scan:
                    title.scene_scan_status = "scan_failed"
                if job.kind == SceneJobKind.blur:
                    # A failed blur job is terminal -- apply_scene_blur's docstring
                    # only resumes a work_dir when the *same* job_id gets
                    # redispatched after an interrupted (CancelledError) run; a
                    # retry after a genuine failure is always a fresh Apply click
                    # with a new job_id, so this one's work_dir is never coming
                    # back for. Left behind otherwise, this is a multi-GB ffmpeg
                    # scratch dir that sits on disk forever.
                    work_dir = app_settings.data_dir / "blur_work" / str(job.id)
                    await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)

            await session.commit()


scene_job_queue = SceneJobQueue()
