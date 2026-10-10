"""Low-res, browser-playable stand-in for a title's full video, used only by the
manual-scene timeline on the scene review page.

Real media here is mostly HEVC/AC3 in .mkv, which browsers can't play (see the
same note on scenes.scene_clip), and scene_clip's short re-encoded clips can't
support scrubbing across a whole movie. So the first time a title's timeline is
opened we transcode one small H.264/AAC MP4 (same timeline as the source -- no
trimming -- so every timestamp maps 1:1) plus a thumbnail sprite, cache both
under the data dir, and serve those instead."""

import asyncio
import logging
from pathlib import Path

from app.config import settings as app_settings
from app.db.models import Title
from app.mux.remux import probe

logger = logging.getLogger(__name__)

SPRITE_FRAMES = 60
SPRITE_THUMB_WIDTH = 160

# title_id -> {"state": "building"|"error", "percent": float, "error": str|None}.
# In-process only: a restart mid-build just leaves a stray .tmp file, and the
# next request starts over. "ready" is never stored here -- it's whatever the
# cache directory says.
_builds: dict[int, dict] = {}


def _cache_dir() -> Path:
    return app_settings.data_dir / "scrub_proxy"


def _cache_key(title: Title) -> str:
    # Keyed on the source file's identity so a replaced/re-muxed file gets a
    # fresh proxy instead of serving a stale one.
    stat = Path(title.video_path).stat()
    return f"{title.id}-{stat.st_mtime_ns}-{stat.st_size}"


def video_path_for(title: Title) -> Path:
    return _cache_dir() / f"{_cache_key(title)}.mp4"


def sprite_path_for(title: Title) -> Path:
    return _cache_dir() / f"{_cache_key(title)}.jpg"


def is_ready(title: Title) -> bool:
    try:
        return video_path_for(title).exists() and sprite_path_for(title).exists()
    except OSError:
        return False


def status_for(title: Title) -> dict:
    if is_ready(title):
        return {"state": "ready"}
    build = _builds.get(title.id)
    if build is not None:
        return build
    return {"state": "missing"}


async def _run_ffmpeg(args: list[str], duration: float | None, title_id: int) -> None:
    proc = await asyncio.create_subprocess_exec(
        app_settings.ffmpeg_bin, "-nostdin", "-y", "-loglevel", "error", "-progress", "pipe:1", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").strip()
        if duration and line.startswith("out_time_us="):
            try:
                seconds = int(line.split("=", 1)[1]) / 1_000_000
            except ValueError:
                continue
            # Leave the last few percent for the sprite pass.
            _builds[title_id]["percent"] = min(95.0, seconds / duration * 95.0)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace").strip()[-500:] or f"ffmpeg exited {proc.returncode}")


async def build(title: Title) -> None:
    """Idempotent: no-ops if already built or a build for this title is running."""
    title_id = title.id
    if is_ready(title) or _builds.get(title_id, {}).get("state") == "building":
        return
    _builds[title_id] = {"state": "building", "percent": 0.0, "error": None}

    video_out, sprite_out = video_path_for(title), sprite_path_for(title)
    video_tmp, sprite_tmp = video_out.with_suffix(".mp4.tmp"), sprite_out.with_suffix(".jpg.tmp")
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        data = await probe(app_settings.ffprobe_bin, Path(title.video_path))
        duration = float(data["format"].get("duration", 0)) or None

        await _run_ffmpeg(
            ["-i", title.video_path, "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
             "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "33",
             "-maxrate", "600k", "-bufsize", "1200k", "-g", "24",
             "-c:a", "aac", "-ac", "2", "-b:a", "64k", "-movflags", "+faststart", "-f", "mp4", str(video_tmp)],
            duration, title_id,
        )
        # One row of evenly spaced thumbnails, read from the (tiny) proxy rather than the source.
        fps = f"{SPRITE_FRAMES}/{duration:.3f}" if duration else "1/60"
        await _run_ffmpeg(
            ["-i", str(video_tmp), "-vf", f"fps={fps},scale={SPRITE_THUMB_WIDTH}:-2,tile={SPRITE_FRAMES}x1",
             "-frames:v", "1", "-q:v", "5", "-f", "image2", str(sprite_tmp)],
            None, title_id,
        )
        video_tmp.replace(video_out)
        sprite_tmp.replace(sprite_out)
        for stale in _cache_dir().glob(f"{title_id}-*"):
            if stale not in (video_out, sprite_out):
                stale.unlink(missing_ok=True)
        _builds.pop(title_id, None)
    except Exception as exc:
        logger.exception("Scrub proxy build failed for title %s", title_id)
        video_tmp.unlink(missing_ok=True)
        sprite_tmp.unlink(missing_ok=True)
        _builds[title_id] = {"state": "error", "percent": 0.0, "error": str(exc)[:300]}
