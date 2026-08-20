"""Blurs the video and mutes every audio track during a title's approved scene
windows, producing a brand-new sibling "Vulgarr Edit" file next to the original --
the source file is never opened for writing.

Unlike app.mux.remux (which stream-copies everything and only adds one new,
filtered audio track), this is the one place in the app that has to re-encode
video: `enable=` only toggles whether a filter is *active* per-frame, so the
whole stream still has to be decoded/encoded regardless of how much of it is
actually blurred.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from app.audio.mute import MuteInterval, _MAX_TERMS_PER_STAGE, build_volume_filter
from app.mux.remux import ProgressCallback, RemuxError, StageCallback, _audio_streams, _run, _run_with_progress, probe

logger = logging.getLogger(__name__)


def build_blur_filter(intervals: list[MuteInterval], input_label: str, output_label: str) -> str:
    """Same batching/between()-summation technique as build_volume_filter
    (app/audio/mute.py), applied to a video boxblur instead of an audio volume
    filter -- ffmpeg's expression *parser* is what breaks past ~80-90 between()
    terms in one enable= expression, not the specific filter consuming it, so the
    same _MAX_TERMS_PER_STAGE limit applies here too.
    """
    if not intervals:
        return f"[{input_label}]null[{output_label}]"

    batches = [intervals[i : i + _MAX_TERMS_PER_STAGE] for i in range(0, len(intervals), _MAX_TERMS_PER_STAGE)]

    stages = []
    current_label = input_label
    for i, batch in enumerate(batches):
        is_last = i == len(batches) - 1
        next_label = output_label if is_last else f"{output_label}_stage{i}"
        conditions = "+".join(f"between(t,{iv.start:.3f},{iv.end:.3f})" for iv in batch)
        stages.append(
            f"[{current_label}]boxblur=luma_radius=25:luma_power=3:chroma_radius=25:chroma_power=3:"
            f"enable='{conditions}'[{next_label}]"
        )
        current_label = next_label

    return ";".join(stages)


def sibling_edit_path(video_path: Path) -> Path:
    """Plex "Versions" naming (flat, same folder, no braces/tags) -- confirmed
    against support.plex.tv during planning as the mechanism that groups a
    second file under the same library item with a version picker, unlike
    Plex's separate "Editions" convention (which creates a distinct library item
    and has no per-episode support at all)."""
    return video_path.with_name(f"{video_path.stem} - Vulgarr Edit{video_path.suffix}")


async def build_blurred_video(
    *,
    video_path: Path,
    intervals: list[MuteInterval],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_crf: int,
    video_preset: str,
    audio_bitrate: str = "192k",
    timeout: float = 14400,  # a real encode can genuinely run an hour+ on this hardware
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Runs ffmpeg to produce a temp file (same directory as video_path, so the
    final publish step is a same-filesystem atomic rename, not a cross-filesystem
    copy) with the video blurred and every audio stream muted during intervals,
    everything else (subtitles/chapters/attachments) passed through untouched."""
    src_probe = await probe(ffprobe_bin, video_path)
    total_duration = float(src_probe["format"].get("duration", 0))
    audio_streams = _audio_streams(src_probe)
    if not audio_streams:
        raise RemuxError("No audio streams found in source file")

    filter_parts = [build_blur_filter(intervals, input_label="0:v", output_label="vblurred")]
    map_args = ["-map", "[vblurred]"]
    codec_args = ["-c:v", "libx264", "-crf", str(video_crf), "-preset", video_preset]

    for i in range(len(audio_streams)):
        label = f"a{i}muted"
        filter_parts.append(build_volume_filter(intervals, input_label=f"0:a:{i}", output_label=label))
        map_args += ["-map", f"[{label}]"]
        codec_args += [f"-c:a:{i}", "aac", f"-b:a:{i}", audio_bitrate]

    # Explicit maps (not "-map 0" + negative-exclude) keep stream order predictable:
    # video first, audio streams next in original order, subtitles/other after --
    # "-map 0" followed by excluding video/audio would instead push the new
    # filtered streams to the end, behind subtitles, which is atypical and worth
    # avoiding even though most players don't strictly require video at index 0.
    map_args += ["-map", "0:s?", "-map", "0:d?", "-map", "0:t?"]

    filter_str = ";".join(filter_parts)
    tmp_path = video_path.with_name(f".{video_path.stem}.spf-blur-tmp{video_path.suffix}")

    cmd = [
        ffmpeg_bin, "-y", "-progress", "pipe:1", "-nostats",
        "-i", str(video_path),
        "-filter_complex", filter_str,
        *map_args,
        "-map_metadata", "0", "-map_chapters", "0",
        *codec_args,
        "-c:s", "copy",
        str(tmp_path),
    ]

    logger.info(
        "Running ffmpeg scene-blur for %s (%d window(s), %d audio stream(s))",
        video_path, len(intervals), len(audio_streams),
    )
    try:
        code, err = await _run_with_progress(
            cmd, total_duration_seconds=total_duration, on_progress=on_progress, timeout=timeout
        )
    except asyncio.CancelledError:
        tmp_path.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp_path.unlink(missing_ok=True)
        raise RemuxError(f"ffmpeg scene-blur failed for {video_path}: {err.strip()[-4000:]}")

    return tmp_path


async def verify_blurred_output(
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    original_probe: dict,
    tmp_path: Path,
    blurred_windows: list[MuteInterval],
) -> None:
    """Simpler than remux.py's verify_output -- there's no "streams added"
    arithmetic to reconcile here, since this is a fresh file with the same
    stream count as the source, not a modified one."""
    new_probe = await probe(ffprobe_bin, tmp_path)

    expected_streams = len(original_probe["streams"])
    actual_streams = len(new_probe["streams"])
    if actual_streams != expected_streams:
        raise RemuxError(f"Stream count mismatch for {tmp_path}: expected {expected_streams}, got {actual_streams}")

    orig_duration = float(original_probe["format"].get("duration", 0))
    new_duration = float(new_probe["format"].get("duration", 0))
    if orig_duration and abs(orig_duration - new_duration) > 2.0:
        raise RemuxError(f"Duration mismatch for {tmp_path}: original={orig_duration:.1f}s new={new_duration:.1f}s")

    # Decode a window spanning the first blurred scene (not just the first few
    # seconds, unlike remux.py's check -- the whole point here is confirming the
    # blur/mute filter graph itself didn't corrupt the output).
    first = blurred_windows[0]
    check_start = max(0.0, first.start - 2.0)
    check_duration = min(15.0, (first.end - first.start) + 4.0)
    code, _out, err = await _run(
        [
            ffmpeg_bin, "-v", "error",
            "-ss", f"{check_start:.3f}", "-i", str(tmp_path),
            "-t", f"{check_duration:.3f}", "-f", "null", "-",
        ],
        timeout=120,
    )
    if code != 0 or err.strip():
        raise RemuxError(f"Blurred output failed decode check for {tmp_path}: {err.strip()[-2000:]}")


async def apply_blur(
    *,
    video_path: Path,
    intervals: list[MuteInterval],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_crf: int,
    video_preset: str,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> Path:
    if not video_path.exists():
        raise RemuxError(f"Video file does not exist: {video_path}")
    if not intervals:
        raise RemuxError("No approved scene windows to apply")

    src_probe = await probe(ffprobe_bin, video_path)

    tmp_path = await build_blurred_video(
        video_path=video_path,
        intervals=intervals,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        video_crf=video_crf,
        video_preset=video_preset,
        on_progress=on_progress,
    )

    if on_stage is not None:
        await on_stage("Verifying output")
    try:
        await verify_blurred_output(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            original_probe=src_probe,
            tmp_path=tmp_path,
            blurred_windows=intervals,
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if on_stage is not None:
        await on_stage("Publishing Vulgarr Edit")

    final_path = sibling_edit_path(video_path)
    tmp_path.replace(final_path)  # same-directory rename: atomic, no partial-file window
    return final_path
