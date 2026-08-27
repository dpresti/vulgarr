"""Lightweight in-process asyncio job queue.

Deliberately not Celery/Redis: a single asyncio.Queue plus a bounded set of
worker tasks, backed by the ProcessingJob table for persistence/visibility.
"""

import asyncio
import datetime
import logging
import os
import time
from pathlib import Path

from sqlalchemy import select

from app.config import settings as app_settings
from app.db.models import JobState, MediaType, ProcessingJob, Title, TriggerSource
from app.db.session import get_session, get_setting
from app.domain import is_mkv_path, parse_index_list, parse_severity_levels, serialize_index_list
from app.integrations.bazarr import BazarrClient
from app.integrations.radarr import RadarrClient
from app.integrations.sonarr import SonarrClient
from app.integrations.subtitle_lookup import find_subtitle_for_video
from app.library import poll_for_subtitle_then_enqueue
from app.mux.remux import RemuxError
from app.processing import ProcessingError, process_video

logger = logging.getLogger(__name__)

# How often the ffmpeg progress callback is allowed to actually write to the DB.
# ffmpeg's -progress stream emits updates far more often than that; without this,
# a single job would generate a commit per frame.
_PROGRESS_UPDATE_MIN_INTERVAL_SECONDS = 1.5

# Whisper forced alignment (a real ffmpeg extract + model inference per matched cue)
# runs before the ffmpeg mux step, as "Step 1/2" -- muting/muxing is "Step 2/2". Split
# evenly down the middle rather than weighting toward alignment: unlike a straight
# fraction-of-total-time split, an even split keeps the boundary predictable regardless
# of how a given title's cue count/length happens to balance against mux time.
_WHISPER_ALIGN_MAX_PERCENT = 50.0

# Alignment's own per-cue ETA was volatile enough early in a job to look broken (see
# the ffmpeg-ETA-instability note this mirrors below) and cues aren't uniform-length
# work anyway, so Step 1 shows raw progress only, no ETA. Step 2's ETA instead waits
# until this many percentage points into its own phase before showing anything, then
# bases the estimate on the elapsed time for those first points -- avoids the same
# nonsense-number-when-fraction-is-near-zero problem ffmpeg's own progress stream has.
_MUX_ETA_WARMUP_PERCENT = 2.0

# How often to check Sonarr/Radarr for whether a requested mkv replacement has landed
# yet. This is a real search+grab+import cycle on their end, which can take anywhere
# from seconds to a long time depending on availability -- no need to check often.
_REPLACEMENT_CHECK_INTERVAL_SECONDS = 60.0

# How often to sweep /data/backups for files past the configured retention window.
# Retention is opt-in (0 = disabled/keep forever, see Settings > Backups), and even
# when enabled a stale backup sitting an extra hour costs nothing -- no need to poll
# more often than this.
_BACKUP_PRUNE_INTERVAL_SECONDS = 3600.0


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m remaining"
    if minutes:
        return f"{minutes}m {seconds}s remaining"
    return f"{seconds}s remaining"


def _parse_hhmm(value: str) -> datetime.time:
    hour, minute = value.split(":")
    return datetime.time(hour=int(hour), minute=int(minute))


def is_within_off_hours_window(now: datetime.time, start: datetime.time, end: datetime.time) -> bool:
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # window wraps past midnight


def _delete_backups_older_than(backup_root: Path, retention_days: int) -> int:
    """Synchronous (run off the event loop via asyncio.to_thread -- backup_root can be
    a slow network mount) sweep of every file under backup_root older than the cutoff,
    by mtime. Returns how many files were deleted."""
    if not backup_root.exists():
        return 0
    cutoff = time.time() - retention_days * 86400
    deleted = 0
    for path in backup_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            logger.warning("Could not delete backup %s", path, exc_info=True)
    # Clean up any directories left empty behind deleted backups.
    for dirpath, dirnames, filenames in os.walk(backup_root, topdown=False):
        if dirpath != str(backup_root) and not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass
    return deleted


class JobQueue:
    def __init__(self) -> None:
        self._pending: asyncio.Queue[int] = asyncio.Queue()
        self._running: dict[int, asyncio.Task] = {}
        self._dispatcher_task: asyncio.Task | None = None
        self._stopping = False
        self._last_replacement_check = 0.0
        self._last_backup_prune = 0.0

    async def start(self) -> None:
        # Re-queue anything left mid-flight from a prior process crash/restart.
        async with get_session() as session:
            result = await session.execute(
                select(ProcessingJob).where(ProcessingJob.state.in_([JobState.queued, JobState.processing]))
            )
            active_title_ids: set[int] = set()
            for job in result.scalars().all():
                job.state = JobState.queued
                job.started_at = None
                title = await session.get(Title, job.title_id)
                if title is not None:
                    title.status = "queued"
                active_title_ids.add(job.title_id)

            # Defensive reconciliation: a Title stuck showing "queued" with no
            # actual ProcessingJob behind it (the requeue loop above only
            # re-queues real orphaned *jobs*; this catches an orphaned *status*
            # with no job at all -- see the identical fix in scene_worker.py's
            # start() for the fuller reasoning, same category of drift). Not
            # expected in normal operation, but cheap to check.
            result = await session.execute(select(Title).where(Title.status == "queued"))
            for title in result.scalars().all():
                if title.id not in active_title_ids:
                    title.status = "not_processed"

            await session.commit()

            result = await session.execute(
                select(ProcessingJob.id).where(ProcessingJob.state == JobState.queued)
            )
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
        """Cancel a queued or in-progress job. Returns (success, error_message).

        A job that's already past the backup/swap step (progress nudged to >=97%
        by on_stage) is refused rather than interrupted -- that step moves the
        original file aside and swaps the verified replacement in; cutting it off
        mid-way risks leaving the file in a half-swapped state, and it's normally
        brief anyway, so it's simpler and safer to just let it finish.
        """
        task = self._running.get(job_id)
        if task is not None:
            async with get_session() as session:
                job = await session.get(ProcessingJob, job_id)
                # A task is registered here as soon as it's created (see
                # _dispatch_loop), before _run_job's own body has necessarily
                # run far enough to flip state to "processing" -- treat that
                # brief "queued" window the same as "processing" rather than
                # rejecting the cancel, since task.cancel() is what actually
                # matters here (cancelling before the coroutine body has even
                # started prevents it from ever running at all).
                if job is None or job.state not in (JobState.queued, JobState.processing):
                    return False, "Job is no longer running"
                if job.state == JobState.processing and (job.progress_percent or 0) >= 97:
                    return False, "Too far along to cancel -- finishing up"
            # Let _run_job's own CancelledError handler own the state transition,
            # rather than writing to the job row from this separate session too --
            # avoids two sessions racing to commit conflicting updates to the same row.
            task.cancel()
            return True, None

        async with get_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.state != JobState.queued:
                return False, "Job is not queued or running"
            job.state = JobState.cancelled
            job.error = "Cancelled by user"
            job.finished_at = datetime.datetime.utcnow()
            title = await session.get(Title, job.title_id)
            if title is not None:
                title.status = "cancelled"
            await session.commit()
        return True, None

    async def enqueue(self, title_id: int, trigger_source: TriggerSource) -> int:
        async with get_session() as session:
            wordlist_version = int(await get_setting(session, "wordlist_version"))
            job = ProcessingJob(
                title_id=title_id,
                state=JobState.queued,
                trigger_source=trigger_source,
                wordlist_version=wordlist_version,
            )
            session.add(job)
            title = await session.get(Title, title_id)
            if title is not None:
                title.status = "queued"
            await session.commit()
            await session.refresh(job)
        await self._pending.put(job.id)
        return job.id

    async def _dispatch_loop(self) -> None:
        while not self._stopping:
            try:
                async with get_session() as session:
                    cap = int(await get_setting(session, "concurrency_cap"))
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

                now_monotonic = time.monotonic()
                if now_monotonic - self._last_replacement_check >= _REPLACEMENT_CHECK_INTERVAL_SECONDS:
                    self._last_replacement_check = now_monotonic
                    await self._check_awaiting_replacements()

                if now_monotonic - self._last_backup_prune >= _BACKUP_PRUNE_INTERVAL_SECONDS:
                    self._last_backup_prune = now_monotonic
                    await self._prune_old_backups()

                await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Dispatcher loop error")
                await asyncio.sleep(5)

    async def _check_awaiting_replacements(self) -> None:
        async with get_session() as session:
            result = await session.execute(select(Title.id).where(Title.status == "awaiting_mkv"))
            title_ids = [row[0] for row in result.all()]
        for title_id in title_ids:
            try:
                await self._check_one_replacement(title_id)
            except Exception:
                logger.exception("Replacement check failed for title %s", title_id)

    async def _prune_old_backups(self) -> None:
        async with get_session() as session:
            retention_days = int(await get_setting(session, "backup_retention_days") or 0)
        if retention_days <= 0:
            return  # 0 = keep every backup forever (defaults to 7 -- see DEFAULT_SETTINGS)
        backup_root = app_settings.data_dir / "backups"
        try:
            deleted = await asyncio.to_thread(_delete_backups_older_than, backup_root, retention_days)
        except Exception:
            logger.exception("Backup retention sweep failed")
            return
        if deleted:
            logger.info("Backup retention: deleted %d file(s) older than %d day(s)", deleted, retention_days)

    async def _check_one_replacement(self, title_id: int) -> None:
        async with get_session() as session:
            title = await session.get(Title, title_id)
            if title is None or title.status != "awaiting_mkv":
                return

            radarr_url = await get_setting(session, "radarr_url")
            radarr_api_key = await get_setting(session, "radarr_api_key")
            sonarr_url = await get_setting(session, "sonarr_url")
            sonarr_api_key = await get_setting(session, "sonarr_api_key")
            bazarr_url = await get_setting(session, "bazarr_url")
            bazarr_api_key = await get_setting(session, "bazarr_api_key")
            default_subtitle_language = await get_setting(session, "default_subtitle_language")

            new_path: str | None = None
            if title.media_type == MediaType.movie and title.radarr_movie_id:
                if radarr_url and radarr_api_key:
                    client = RadarrClient(radarr_url, radarr_api_key)
                    new_path = await client.get_current_movie_file_path(title.radarr_movie_id)
            elif title.media_type == MediaType.episode and title.sonarr_episode_id:
                if sonarr_url and sonarr_api_key:
                    client = SonarrClient(sonarr_url, sonarr_api_key)
                    new_path = await client.get_current_episode_file_path(title.sonarr_episode_id)

            if not new_path or new_path == title.video_path or not is_mkv_path(new_path):
                return  # nothing changed yet, or the grabbed replacement still isn't mkv

            logger.info("mkv replacement landed for title %s: %s", title_id, new_path)
            title.video_path = new_path
            title.has_clean_track = False
            title.clean_track_audio_indices = None
            title.replacement_requested_at = None
            title.last_error = None

            subtitle = find_subtitle_for_video(Path(new_path), default_subtitle_language)
            if subtitle:
                title.subtitle_path = str(subtitle)
                title.subtitle_language = default_subtitle_language
                title.status = "queued"
                await session.commit()
                await self.enqueue(title_id, TriggerSource.manual)
                return

            # No subtitle yet for the brand-new file. Trigger a Bazarr search proactively
            # (Bazarr may not have scanned/searched this just-grabbed file on its own
            # yet), then fall back to the same bounded polling the Sonarr/Radarr import
            # webhook uses -- a single immediate check here was the bug: Bazarr's search
            # can take a few seconds to actually place the file, and checking only once
            # right away missed it, permanently giving up instead of catching it shortly
            # after.
            title.subtitle_path = None
            title.status = "awaiting_subtitle"
            media_type = title.media_type
            display_name = title.display_name
            sonarr_series_id = title.sonarr_series_id
            sonarr_episode_id = title.sonarr_episode_id
            radarr_movie_id = title.radarr_movie_id
            series_title = title.series_title
            season_number = title.season_number
            episode_number = title.episode_number
            await session.commit()

        if bazarr_url and bazarr_api_key:
            bazarr = BazarrClient(bazarr_url, bazarr_api_key)
            try:
                if media_type == MediaType.episode:
                    await bazarr.search_episode_subtitle(
                        series_id=sonarr_series_id,
                        episode_id=sonarr_episode_id,
                        language=default_subtitle_language,
                    )
                else:
                    await bazarr.search_movie_subtitle(
                        radarr_id=radarr_movie_id, language=default_subtitle_language
                    )
            except Exception:
                logger.exception("Bazarr search failed for replacement title %s", title_id)

        asyncio.create_task(
            poll_for_subtitle_then_enqueue(
                video_path=new_path,
                media_type=media_type,
                display_name=display_name,
                sonarr_series_id=sonarr_series_id,
                sonarr_episode_id=sonarr_episode_id,
                radarr_movie_id=radarr_movie_id,
                trigger_source=TriggerSource.manual,
                series_title=series_title,
                season_number=season_number,
                episode_number=episode_number,
            )
        )

    async def _run_job(self, job_id: int) -> None:
        async with get_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            if job.state == JobState.cancelled:
                # Cancelled while still sitting in self._pending, before this
                # task started -- nothing to do.
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
            job.progress_message = "Parsing subtitles and matching word list"
            job.progress_percent = None
            title.status = "processing"
            await session.commit()

            last_update_monotonic = 0.0
            mux_phase_start_monotonic: float | None = None
            # Whisper mode already spends Step 1 (0-50%) on alignment before ffmpeg ever
            # starts -- Step 2 continues the bar from there instead of restarting its own
            # percent/ETA from 0, which previously made the bar (and the ETA, computed
            # off the whole job's elapsed time including however long alignment took)
            # jump backwards the moment muting began.
            is_whisper = title.precise_mode == "whisper"
            mux_base_percent = _WHISPER_ALIGN_MAX_PERCENT if is_whisper else 0.0
            mux_percent_span = 97.0 - mux_base_percent
            mux_step_prefix = "Step 2/2: " if is_whisper else ""
            # Phase-local fraction at which Step 2's ETA turns on -- e.g. whisper mode's
            # 47-point span (50-97%) means 2 points in is ~4.3% fraction; non-whisper's
            # full 97-point span means the same 2 points is ~2%, matching the plain
            # ffmpeg-only threshold this replaced.
            mux_eta_warmup_fraction = _MUX_ETA_WARMUP_PERCENT / mux_percent_span

            async def on_progress(fraction: float) -> None:
                nonlocal last_update_monotonic, mux_phase_start_monotonic
                now = time.monotonic()
                if mux_phase_start_monotonic is None:
                    mux_phase_start_monotonic = now
                is_final = fraction >= 0.999
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now

                elapsed = now - mux_phase_start_monotonic
                pct = round(mux_base_percent + fraction * mux_percent_span, 1)
                message = f"{mux_step_prefix}Muting audio (ffmpeg) -- {round(fraction * 100):.0f}%"
                if fraction > mux_eta_warmup_fraction and not is_final:
                    eta_seconds = elapsed * (1 - fraction) / fraction
                    message += f", {_format_eta(eta_seconds)}"

                job.progress_percent = pct
                job.progress_message = message
                await session.commit()

            async def on_align_progress(done: int, total: int) -> None:
                nonlocal last_update_monotonic
                if total <= 0:
                    return
                now = time.monotonic()
                is_final = done >= total
                if not is_final and now - last_update_monotonic < _PROGRESS_UPDATE_MIN_INTERVAL_SECONDS:
                    return
                last_update_monotonic = now

                fraction = done / total
                # No ETA here -- cues vary too much in length/difficulty for an early
                # done/total ratio to predict remaining time, and Step 2 already covers
                # ETA once there's ffmpeg progress to base one on.
                job.progress_percent = round(fraction * _WHISPER_ALIGN_MAX_PERCENT, 1)
                job.progress_message = f"Step 1/2: Aligning word timing (Whisper) -- {done}/{total} cues"
                await session.commit()

            async def on_stage(message: str) -> None:
                job.progress_message = message
                # Nudge the bar forward past wherever ffmpeg's own progress topped
                # out (throttling means it rarely reports exactly 100%), so these
                # short post-mux steps still read as forward progress, not a freeze.
                job.progress_percent = max(job.progress_percent or 0, 97.0)
                await session.commit()

            try:
                if not title.subtitle_path:
                    raise ProcessingError("No subtitle file associated with this title")

                outcome = await process_video(
                    session,
                    video_path=Path(title.video_path),
                    subtitle_path=Path(title.subtitle_path),
                    severity_levels=parse_severity_levels(title.severity_levels),
                    known_clean_indices=parse_index_list(title.clean_track_audio_indices),
                    precise_mode=title.precise_mode,
                    on_progress=on_progress,
                    on_stage=on_stage,
                    on_align_progress=on_align_progress,
                )

                job.state = JobState.done
                job.progress_message = f"Done -- muted {outcome.matched_cue_count} cue(s)"
                job.progress_percent = 100.0
                job.finished_at = datetime.datetime.utcnow()

                title.last_processed_at = datetime.datetime.utcnow()
                title.last_processed_wordlist_version = job.wordlist_version
                title.matched_cue_count = outcome.matched_cue_count
                title.has_clean_track = True
                title.clean_track_audio_indices = serialize_index_list(outcome.clean_track_indices)
                title.last_error = None
                title.status = "done"

            except asyncio.CancelledError:
                job.state = JobState.cancelled
                job.error = "Cancelled by user"
                job.finished_at = datetime.datetime.utcnow()
                title.last_error = None
                title.status = "cancelled"
                await session.commit()
                raise  # required so the Task itself is properly marked cancelled

            except (ProcessingError, RemuxError, Exception) as exc:  # noqa: BLE001 -- surface all failures to the UI
                logger.exception("Processing failed for title %s", title.id)
                job.state = JobState.failed
                job.error = str(exc)
                job.finished_at = datetime.datetime.utcnow()
                title.last_error = str(exc)
                title.status = "failed"

            await session.commit()


job_queue = JobQueue()
