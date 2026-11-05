"""Cleanup for this app's own transient files left next to a video.

build_clean_track / build_blurred_video write their output to a hidden
".<stem>.spf-tmp.<id><ext>" / ".<stem>.spf-blur-tmp.<id><ext>" sibling and
remove it on every in-process failure path -- but a run that dies with the
process (container restart/recreate mid-ffmpeg, OOM kill, a disk-full crash)
never gets that chance, and the multi-GB file is left in the library forever.
Nothing else ever looks at it again: it's hidden, so neither Plex nor the user
notices, and is_own_temp_path() makes webhooks ignore it on purpose.
"""

import asyncio
import logging
import time
from pathlib import Path

from app.domain import _OWN_TEMP_FILE_MARKERS

logger = logging.getLogger(__name__)

# A live run's ffmpeg rewrites its temp file continuously, so one untouched for
# this long belongs to a dead run, not one still in progress. Only matters for
# the per-run sweep, which can overlap a concurrent job on the same title.
STALE_AFTER_SECONDS = 3600


def remove_orphaned_temp_files(video_path: Path, *, min_age_seconds: float = STALE_AFTER_SECONDS) -> list[Path]:
    """Delete video_path's leftover temp files from dead runs; returns what was removed.

    Matches only this exact title's temp names (same stem, same extension, one
    of our markers), so it can never touch the real video, a Vulgarr Edit, or
    another title's files in the same folder. Synchronous (directory listing on
    what is usually a network share) -- call it via asyncio.to_thread.
    """
    prefixes = tuple(f".{video_path.stem}{marker}" for marker in _OWN_TEMP_FILE_MARKERS)
    cutoff = time.time() - min_age_seconds
    removed: list[Path] = []
    try:
        entries = list(video_path.parent.iterdir())
    except OSError:
        return removed
    for path in entries:
        if not path.name.startswith(prefixes) or path.suffix != video_path.suffix:
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
            path.unlink()
        except OSError:
            continue
        logger.info("Removed orphaned temp file from an interrupted run: %s", path)
        removed.append(path)
    return removed


async def sweep_interrupted_temp_files(video_paths: list[Path]) -> None:
    """Startup sweep for jobs a queue found still marked "processing" -- i.e.
    the previous process died mid-run. No job is running yet, so no age guard.
    Never raises: a slow or unmounted share must not block the app starting."""
    for video_path in video_paths:
        try:
            await asyncio.to_thread(remove_orphaned_temp_files, video_path, min_age_seconds=0)
        except Exception:
            logger.warning("Temp-file sweep failed for %s", video_path, exc_info=True)
