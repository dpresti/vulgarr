"""ffprobe/ffmpeg orchestration: append a muted "Clean" audio track to a video file
in place, without ever rewriting the original streams' bytes.

Strategy ("in-place remux with safety net", per project decision):
  1. ffmpeg stream-copies every existing stream (video/audio/subtitles/chapters)
     untouched and adds one new, filtered ("muted") audio track, writing to a
     temp file next to the original.
  2. The temp file is verified (stream count, duration, and that the new track
     actually decodes) before anything touches the original.
  3. The original file is moved to a backup location (never deleted).
  4. The verified temp file is atomically renamed into the original's path.

If any step fails, the original file is left completely untouched.
"""

import asyncio
import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from app.audio.mute import MuteInterval, build_volume_filter
from app.domain import Severity
from app.subtitles.matcher import CueMatch

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], Awaitable[None]]
StageCallback = Callable[[str], Awaitable[None]]


class RemuxError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemuxResult:
    output_path: Path
    backup_path: Path
    matched_cue_count: int
    clean_track_indices: list[int]  # 0-based audio-stream indices of the new tracks, in canonical order


async def _run(cmd: list[str], timeout: float | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RemuxError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except asyncio.CancelledError:
        # Awaiting proc.communicate() being cancelled does not kill the child
        # process on its own -- without this it would keep running as an orphan.
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _run_with_progress(
    cmd: list[str],
    *,
    total_duration_seconds: float,
    on_progress: ProgressCallback | None,
    timeout: float | None = None,
) -> tuple[int, str]:
    """Like _run, but parses ffmpeg's `-progress pipe:1` key=value stream to report
    fractional (0..1) completion via on_progress as the command runs, instead of
    only finding out the result once the whole thing finishes.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stderr_chunks: list[bytes] = []
    current_out_time = 0.0

    async def read_stderr() -> None:
        assert proc.stderr is not None
        async for chunk in proc.stderr:
            stderr_chunks.append(chunk)

    async def read_progress() -> None:
        nonlocal current_out_time
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    current_out_time = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
            elif line.startswith("progress=") and on_progress is not None and total_duration_seconds > 0:
                fraction = max(0.0, min(1.0, current_out_time / total_duration_seconds))
                await on_progress(fraction)

    try:
        await asyncio.wait_for(
            asyncio.gather(read_stderr(), read_progress(), proc.wait()), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RemuxError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except asyncio.CancelledError:
        # Same reasoning as _run: cancellation must kill the actual ffmpeg
        # process, not just stop awaiting it.
        proc.kill()
        await proc.wait()
        raise

    return proc.returncode, b"".join(stderr_chunks).decode(errors="replace")


async def probe(ffprobe_bin: str, path: Path) -> dict:
    code, out, err = await _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=60,
    )
    if code != 0:
        raise RemuxError(f"ffprobe failed on {path}: {err.strip()}")
    return json.loads(out)


def _audio_streams(probe_data: dict) -> list[dict]:
    return [s for s in probe_data["streams"] if s["codec_type"] == "audio"]


@dataclass(frozen=True)
class AudioStreamPlan:
    source_index: int  # audio-relative index to derive the clean tracks from
    source_stream: dict
    existing_clean_indices: list[int]  # audio-relative indices of prior clean track(s) to drop
    new_track_output_indices: list[int]  # audio-relative indices the new tracks will land at, in order


def plan_audio_streams(
    probe_data: dict,
    clean_track_title: str,
    num_new_tracks: int,
    known_clean_indices: list[int] | None = None,
) -> AudioStreamPlan:
    """Decide which audio stream to mute from, and which streams to drop/replace.

    Reprocessing an already-processed file is a normal operation (word list edited,
    severity levels changed, manual "Reprocess"), so this treats existing clean
    track(s) as stale copies to be replaced, not as another "original" track to
    copy through untouched -- otherwise every reprocess would stack more clean
    tracks onto the file. All previously-generated tracks are dropped and however
    many are currently wanted are regenerated fresh, in fixed canonical order.

    known_clean_indices, when given, are the audio-relative indices
    Title.clean_track_audio_indices recorded after our own last successful run on
    this file -- authoritative, and preferred over reading them back from file
    metadata. Some real-world MP4s (particularly ones muxed by non-ffmpeg tools)
    silently drop the title tag ffmpeg is told to write, which makes metadata-based
    detection unreliable; the DB-tracked indices don't have that problem.
    """
    audio_streams = _audio_streams(probe_data)
    if not audio_streams:
        raise RemuxError("No audio streams found in source file")

    valid_known = [i for i in (known_clean_indices or []) if 0 <= i < len(audio_streams)]
    if valid_known:
        existing_clean_indices = sorted(set(valid_known))
    else:
        normalized_title = clean_track_title.strip().lower()
        # Exact match against every title this app itself could actually have
        # written (see build_clean_track's track_title above: the plain base
        # title for a single track, or "<base> (Child)"/"<base> (Teen)" for a
        # multi-track run) -- not startswith(), which could misidentify a real
        # original track whose own embedded title happens to start with the
        # same text (e.g. a generic clean_track_title of "Clean" matching a
        # legitimate "Clean Surround 5.1" source track) and permanently drop it.
        expected_titles = {normalized_title} | {
            f"{normalized_title} ({sev.value.capitalize()})".lower() for sev in Severity
        }
        existing_clean_indices = [
            i
            for i, s in enumerate(audio_streams)
            if (s.get("tags", {}).get("title") or "").strip().lower() in expected_titles
        ]

    candidates = [(i, s) for i, s in enumerate(audio_streams) if i not in existing_clean_indices]
    if not candidates:
        # Every audio stream looks like one of our own clean tracks (shouldn't happen
        # in practice) -- fall back to treating them all as real rather than erroring.
        candidates = list(enumerate(audio_streams))

    source_index, source_stream = candidates[0]
    for i, s in candidates:
        if s.get("disposition", {}).get("default") == 1:
            source_index, source_stream = i, s
            break

    kept_count = len(audio_streams) - len(existing_clean_indices)
    new_track_output_indices = list(range(kept_count, kept_count + num_new_tracks))
    return AudioStreamPlan(source_index, source_stream, existing_clean_indices, new_track_output_indices)


async def build_clean_track(
    *,
    video_path: Path,
    intervals_by_level: list[tuple[Severity, list[MuteInterval]]],  # canonical order, matches plan.new_track_output_indices
    ffmpeg_bin: str,
    plan: AudioStreamPlan,
    total_duration: float,
    clean_track_title: str,
    clean_track_language: str,
    audio_bitrate: str = "192k",
    timeout: float = 3600,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Run ffmpeg to produce a temp file with all original streams copied plus one new
    muted track per (severity, intervals) pair, all generated in a single pass."""
    # Mirror the source track's language tag only when it's a specific, real value
    # (e.g. "eng") -- that's accurate and worth preserving. But when the source is
    # untagged/"und" (common for scene-release rips), mirroring "und" onto the clean
    # track too makes both streams look identical ("Unknown (AAC 5.1)") in Plex with
    # no way to tell them apart. ffmpeg's `title`/`name` metadata -- the mechanism
    # meant to label the clean track distinctly -- turns out to silently fail to
    # write on some real-world MP4s regardless of technique (confirmed via direct
    # testing: ffmpeg's own printed output plan omits it even when explicitly
    # requested, for files muxed by certain non-ffmpeg tools). Language is the only
    # field that reliably persists, so it's the only reliable way left to keep clean
    # tracks distinguishable from the original when it has no specific language of
    # its own -- track order (fixed: child, teen) is what distinguishes
    # multiple clean tracks from each other when title tags don't stick.
    source_language = plan.source_stream.get("tags", {}).get("language")
    if source_language and source_language.lower() not in ("und", ""):
        clean_track_language = source_language

    if len(intervals_by_level) != len(plan.new_track_output_indices):
        raise RemuxError("intervals_by_level and plan.new_track_output_indices must be the same length")

    multi_track = len(intervals_by_level) > 1
    filter_parts = []
    codec_args: list[str] = []
    map_args = ["-map", "0"]
    for stale_index in plan.existing_clean_indices:
        map_args += ["-map", f"-0:a:{stale_index}"]  # drop prior clean track(s) before re-adding fresh ones

    for (severity, intervals), out_index in zip(intervals_by_level, plan.new_track_output_indices):
        label = f"clean_{severity.value}"
        filter_parts.append(
            build_volume_filter(intervals, input_label=f"0:a:{plan.source_index}", output_label=label)
        )
        map_args += ["-map", f"[{label}]"]
        track_title = f"{clean_track_title} ({severity.value.capitalize()})" if multi_track else clean_track_title
        codec_args += [
            f"-c:a:{out_index}",
            "aac",
            f"-b:a:{out_index}",
            audio_bitrate,
            f"-metadata:s:a:{out_index}",
            f"title={track_title}",
            f"-metadata:s:a:{out_index}",
            f"language={clean_track_language}",
            f"-disposition:a:{out_index}",
            "0",
        ]

    filter_str = ";".join(filter_parts)
    # Unique per call, not just per video_path -- two concurrent jobs on the
    # same title (a double-click, or a webhook firing while a manual job is
    # already running) would otherwise share this exact path and race on it.
    tmp_path = video_path.with_name(f".{video_path.stem}.spf-tmp.{uuid.uuid4().hex[:8]}{video_path.suffix}")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(video_path),
        "-filter_complex",
        filter_str,
        *map_args,
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c",
        "copy",
        *codec_args,
        str(tmp_path),
    ]

    logger.info("Running ffmpeg remux for %s (%d clean track(s))", video_path, len(intervals_by_level))
    try:
        code, err = await _run_with_progress(
            cmd, total_duration_seconds=total_duration, on_progress=on_progress, timeout=timeout
        )
    except asyncio.CancelledError:
        tmp_path.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp_path.unlink(missing_ok=True)
        raise RemuxError(f"ffmpeg remux failed for {video_path}: {err.strip()[-4000:]}")

    return tmp_path


async def verify_output(
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    original_probe: dict,
    tmp_path: Path,
    new_track_output_indices: list[int],
    stale_clean_tracks_dropped: int = 0,
) -> None:
    """Sanity-check the produced file before it ever replaces the original."""
    new_probe = await probe(ffprobe_bin, tmp_path)

    # +1 per freshly-generated clean track, minus any stale clean track(s) from a
    # prior run that got replaced rather than kept.
    expected_streams = len(original_probe["streams"]) + len(new_track_output_indices) - stale_clean_tracks_dropped
    actual_streams = len(new_probe["streams"])
    if actual_streams != expected_streams:
        raise RemuxError(
            f"Stream count mismatch for {tmp_path}: expected {expected_streams}, got {actual_streams}"
        )

    orig_duration = float(original_probe["format"].get("duration", 0))
    new_duration = float(new_probe["format"].get("duration", 0))
    if orig_duration and abs(orig_duration - new_duration) > 2.0:
        raise RemuxError(
            f"Duration mismatch for {tmp_path}: original={orig_duration:.1f}s new={new_duration:.1f}s"
        )

    # Decode each new track in full (audio-only, cheap even for a full movie's
    # length) to confirm none of them are corrupt -- a fixed short window here
    # previously only caught corruption in the first few seconds, leaving any
    # mid-file or end-of-file glitch in a 2+ hour track undetected before the
    # file gets swapped into place as "verified".
    for idx in new_track_output_indices:
        code, _out, err = await _run(
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-i",
                str(tmp_path),
                "-map",
                f"0:a:{idx}",
                "-f",
                "null",
                "-",
            ],
            timeout=600,
        )
        if code != 0 or err.strip():
            raise RemuxError(f"Clean track a:{idx} failed decode check for {tmp_path}: {err.strip()[-2000:]}")


def backup_path_for(video_path: Path, media_root: Path, backup_root: Path) -> Path:
    try:
        rel = video_path.relative_to(media_root)
    except ValueError:
        rel = Path(video_path.name)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    return backup_root / rel.parent / f"{rel.stem}.{timestamp}{rel.suffix}.orig"


async def remux_with_clean_track(
    *,
    video_path: Path,
    matches: list[CueMatch],
    intervals_by_level: list[tuple[Severity, list[MuteInterval]]],
    media_root: Path,
    backup_root: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    clean_track_title: str,
    clean_track_language: str,
    known_clean_indices: list[int] | None = None,
    keep_backup: bool = True,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> RemuxResult:
    if not video_path.exists():
        raise RemuxError(f"Video file does not exist: {video_path}")

    src_probe = await probe(ffprobe_bin, video_path)
    total_duration = float(src_probe["format"].get("duration", 0))
    plan = plan_audio_streams(
        src_probe, clean_track_title, num_new_tracks=len(intervals_by_level), known_clean_indices=known_clean_indices
    )

    tmp_path = await build_clean_track(
        video_path=video_path,
        intervals_by_level=intervals_by_level,
        ffmpeg_bin=ffmpeg_bin,
        plan=plan,
        total_duration=total_duration,
        clean_track_title=clean_track_title,
        clean_track_language=clean_track_language,
        on_progress=on_progress,
    )

    if on_stage is not None:
        await on_stage("Verifying output")
    try:
        await verify_output(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            original_probe=src_probe,
            tmp_path=tmp_path,
            new_track_output_indices=plan.new_track_output_indices,
            stale_clean_tracks_dropped=len(plan.existing_clean_indices),
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if on_stage is not None:
        await on_stage("Backing up original and swapping in the clean version")

    backup_path = backup_path_for(video_path, media_root, backup_root)
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # Move (not copy) the original aside first -- if this fails, the original
    # is untouched and the temp file is simply discarded. Only once the
    # original is safely backed up do we swap the verified temp file in.
    #
    # backup_root and the media live on different filesystems in this deployment
    # (network share vs local disk), so this "move" is really a full copy+delete
    # under the hood and can take a while for a multi-GB file. shutil.move is
    # synchronous, so it's pushed onto a worker thread -- otherwise it would
    # freeze the entire app (including progress polling for every other job)
    # for as long as the copy takes.
    #
    # The whole backup-then-swap sequence is shielded from cancellation --
    # asyncio.to_thread can't actually stop the underlying OS-thread move once
    # started, so an unshielded cancellation here (e.g. JobQueue.stop() on
    # container shutdown, which cancels unconditionally with none of
    # cancel_job()'s 97%-progress protection) could let the background move
    # finish *after* the coroutine has already unwound as cancelled, leaving
    # video_path missing with the replacement swap never applied. Shielding
    # means a cancellation here still reports promptly, but the file ends up
    # correctly swapped in (or restored on failure) either way -- the job row
    # may end up marked "cancelled" even though the file was actually
    # finished, which is the better of the two outcomes.
    async def _finalize_swap() -> None:
        await asyncio.to_thread(shutil.move, str(video_path), str(backup_path))
        try:
            tmp_path.replace(video_path)
        except Exception:
            # Best-effort restore of the original if the final swap somehow fails.
            await asyncio.to_thread(shutil.move, str(backup_path), str(video_path))
            tmp_path.unlink(missing_ok=True)
            raise

    await asyncio.shield(asyncio.create_task(_finalize_swap()))

    if not keep_backup:
        # Backups disabled -- the original still had to be moved aside above for
        # crash-safety during the swap (so a failed swap can restore it), but with
        # backups off there's no reason to keep that copy around now that the swap
        # succeeded. Best-effort: a failure to delete here shouldn't fail the job.
        try:
            await asyncio.to_thread(backup_path.unlink)
            await asyncio.to_thread(backup_path.parent.rmdir)
        except OSError:
            pass

    return RemuxResult(
        output_path=video_path,
        backup_path=backup_path,
        matched_cue_count=len(matches),
        clean_track_indices=plan.new_track_output_indices,
    )
