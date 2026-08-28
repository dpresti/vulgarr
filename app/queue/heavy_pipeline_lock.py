"""Cross-queue gate ensuring at most one heavy, ffmpeg-invoking pipeline job
(audio-mute via app.queue.worker.JobQueue, or scan/Claude-verify/blur via
app.queue.scene_worker.SceneJobQueue) runs at a time, process-wide.

Each queue already has its own intra-queue concurrency cap (concurrency_cap,
scene_scan_concurrency_cap), but those are independent of each other -- two
jobs from DIFFERENT queues (or two audio jobs, since that cap defaults to 2)
could still run their real ffmpeg work concurrently, competing for I/O/cache
against the same NFS-mounted media library. Real, documented flakiness from
exactly this class of contention already exists elsewhere in this codebase
(see _KEYFRAME_PROBE_TIMEOUT_SECONDS's docstring in app/mux/scene_blur.py --
a command that succeeded live timed out on a clean rerun, attributed to an
NFS/OS cache difference). This semaphore adds a cross-queue ceiling of 1 on
top of each queue's own cap, without changing either queue's own scheduling
or settings."""

import asyncio

HEAVY_PIPELINE_LOCK = asyncio.Semaphore(1)
