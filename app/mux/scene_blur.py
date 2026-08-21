"""Blurs the video during a title's approved scene windows, and mutes every
audio track during the subset of those windows opted into muting too, producing
a brand-new sibling "Vulgarr Edit" file next to the original -- the source file
is never opened for writing.

Only the scene windows (plus a small keyframe-alignment buffer) are actually
re-encoded -- everything else is stream-copied byte-for-byte from the source,
via a split/re-encode/concat pipeline (see plan_video_segments):
  1. Probe the source's keyframe timestamps (fast -- ffprobe with -skip_frame
     nokey never decodes a non-keyframe).
  2. Split [0, duration] into an alternating sequence of stream-copy segments
     (the untouched majority) and re-encode segments (each approved scene,
     expanded to the nearest keyframes so the copy segments can cut cleanly).
  3. Extract each segment to a Matroska (.mkv) file (copy is near-instant;
     re-encode only pays for the scene's own duration, not the whole runtime).
     Matroska, not MPEG-TS: concatenating an independently re-encoded segment
     before a stream-copied one via MPEG-TS produced genuine "Could not find
     ref with POC ..." decode corruption at the splice point, regardless of
     which HEVC keyframe type the copy segment started at -- confirmed via
     rigorous, frame-exact pixel comparison against two different real 4K
     HEVC files. MPEG-TS apparently gives decoders no signal to treat an
     internal splice as a hard discontinuity requiring a full reference-
     buffer reset; MP4/MKV don't have that gap (same choice smartcut,
     github.com/skeskinen/smartcut, makes for this exact problem).
  4. Concatenate the segments (ffmpeg's concat demuxer, -c copy -- lossless,
     no second video re-encode) into one continuous video stream.
  5. Mux that against the original's audio/subtitles/chapters/metadata (audio
     muted the same way as before, cheap either way since audio is a tiny
     fraction of total bitrate) into the final output.

This replaces an earlier single-pass whole-file re-encode (kept working but
dramatically slower -- ~30min for a 52min episode on this host's non-AVX CPU
at the default "medium" x264 preset -- and, more importantly, subjected the
entire file to generational quality loss even in the ~95%+ that was never
actually blurred, for no reason: `enable=` only toggles a filter's activity
per-frame, so the OLD approach decoded/encoded every frame regardless of how
much of it was ever touched).

Resumable across a process restart: every intermediate segment/concat file is
written into a *persistent*, caller-owned work_dir (not an auto-deleted temp
dir), each one atomically (written to a `.tmp` sibling, renamed into place
only on success -- see _atomic_run). build_blurred_video treats a segment
file's existence at its real name as a reliable "already done" signal and
skips redoing it, so re-running the same job after an interruption (e.g. this
app's own container being rebuilt mid-job -- a real, repeated annoyance
during this feature's own development) picks up close to where it left off
instead of starting a multi-hour re-encode over from zero. Guarded by
_blur_job_fingerprint against resuming into a work_dir left by a *different*
plan (approved scenes changed between attempts, settings changed, etc.) --
that's treated as fully stale and wiped, never partially trusted.
"""

import asyncio
import bisect
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.audio.mute import MuteInterval, _MAX_TERMS_PER_STAGE
from app.common.intervals import merge_intervals
from app.mux.remux import ProgressCallback, RemuxError, StageCallback, _audio_streams, _run, probe

logger = logging.getLogger(__name__)

# Generous per-segment ceiling -- even a single scene's re-encode should finish
# in well under this on any hardware this app has actually run on, but a
# pathologically long approved "scene" (a user could adjust one to span many
# minutes) shouldn't hang forever either.
_SEGMENT_TIMEOUT_SECONDS = 1800


async def _run_atomic(cmd_for_output: Callable[[Path], list[str]], out_path: Path, timeout: float, error_ctx: str) -> None:
    """Runs an ffmpeg command that writes a single output file, via a `.tmp`
    sibling renamed into place only once ffmpeg actually succeeds -- makes
    "out_path exists" a reliable "fully written" signal for build_blurred_video's
    resume-skip checks, not something a process killed mid-write could fake.
    cmd_for_output takes the actual (temp) output path and returns the full
    argv, so callers build their command around whatever path they're really
    writing to without needing to know about the temp-then-rename dance."""
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    code, _out, err = await _run(cmd_for_output(tmp_out), timeout=timeout)
    if code != 0:
        tmp_out.unlink(missing_ok=True)
        raise RemuxError(f"{error_ctx}: {err.strip()[-2000:]}")
    tmp_out.replace(out_path)

# A single 1-5 "intensity" knob (settings_form.html's Blur intensity slider) is a
# friendlier UI than exposing boxblur's two independent radius/power parameters
# directly -- most people don't have an intuition for what either one means on
# its own. Levels 1-3 mirror this feature's actual tuning history this session
# (25/3 was the original, too-light default; 45/5 and 60/6 were the first two
# rounds of "heavier" before settling); 4 (90/8) is the current default,
# confirmed via a direct visual comparison to leave nothing recognizable; 5
# goes further still for anyone who wants extra headroom. Keyed 1-5 (not
# 0-indexed) to match the slider's own min=1.
BLUR_LEVEL_PRESETS: dict[int, tuple[str, int, int]] = {
    1: ("Light", 25, 3),
    2: ("Medium", 45, 5),
    3: ("Heavy", 60, 6),
    4: ("Very Heavy", 90, 8),
    5: ("Maximum", 120, 10),
}
DEFAULT_BLUR_LEVEL = 4


def blur_level_to_radius_power(level: int) -> tuple[int, int]:
    _, radius, power = BLUR_LEVEL_PRESETS.get(level, BLUR_LEVEL_PRESETS[DEFAULT_BLUR_LEVEL])
    return radius, power


def radius_power_to_blur_level(radius: int, power: int) -> int:
    """Inverse mapping, for pre-selecting the slider from whatever's currently
    stored -- exact match if the stored values came from this preset table (the
    only way to set them, via the settings form), nearest by simple distance
    otherwise (e.g. values set some other way, or a future preset table change)."""
    for level, (_, r, p) in BLUR_LEVEL_PRESETS.items():
        if r == radius and p == power:
            return level
    return min(BLUR_LEVEL_PRESETS, key=lambda lvl: abs(BLUR_LEVEL_PRESETS[lvl][1] - radius) + abs(BLUR_LEVEL_PRESETS[lvl][2] - power))


def build_blur_filter(
    intervals: list[MuteInterval], input_label: str, output_label: str, radius: int = 90, power: int = 8
) -> str:
    """Same batching/between()-summation technique as build_volume_filter
    (app/audio/mute.py), applied to a video boxblur instead of an audio volume
    filter -- ffmpeg's expression *parser* is what breaks past ~80-90 between()
    terms in one enable= expression, not the specific filter consuming it, so the
    same _MAX_TERMS_PER_STAGE limit applies here too.

    radius/power default to 90/8 -- confirmed via a direct visual comparison
    (raw frame vs. 45/5 vs. 60/6 vs. 90/8) that this is heavy enough to leave
    nothing recognizable, just a smooth tone/color gradient. Both are exposed
    as settings (scene_blur_radius/scene_blur_power) rather than fixed, since
    boxblur's cost is roughly independent of radius (sliding-window algorithm),
    so there's no real reason to cap how heavy someone can go.
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
            f"[{current_label}]boxblur=luma_radius={radius}:luma_power={power}:"
            f"chroma_radius={radius}:chroma_power={power}:enable='{conditions}'[{next_label}]"
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


@dataclass(frozen=True)
class VideoSegmentPlan:
    start: float
    end: float
    reencode: bool
    # Approved-scene windows inside this segment, converted to segment-local
    # time (offset by -start) -- only meaningful when reencode=True. A tuple
    # (not list) so the whole dataclass stays hashable/frozen for easy testing.
    local_blur_intervals: tuple[MuteInterval, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_video_segments(
    blur_intervals: list[MuteInterval],
    keyframe_timestamps: list[float],
    total_duration: float,
) -> list[VideoSegmentPlan]:
    """Splits [0, total_duration] into an alternating sequence of stream-copy
    segments (byte-identical to the source -- zero quality loss, effectively
    free to produce) and re-encode segments (containing one or more approved
    scenes, boxblur applied only during their exact windows).

    A stream-copy segment can only start/end at a real keyframe -- ffmpeg
    can't cut a compressed GOP mid-stream without decoding it -- so each
    scene's re-encode segment is expanded to [nearest keyframe at/before its
    start, nearest keyframe at/after its end]. This means slightly more than
    the scene itself gets re-encoded, bounded by the source's GOP size
    (typically a few seconds for real-world rips), never the reverse. Scenes
    whose expanded ranges touch or overlap (close together scenes, or a large
    GOP) are merged into one re-encode segment via the same interval-merge
    helper used elsewhere, so there's never a degenerate zero-length copy
    segment squeezed between two of them.

    Degrades gracefully to the old "re-encode the whole file" behavior if
    keyframe_timestamps is empty (e.g. a probe failure) or otherwise doesn't
    bracket every scene -- the missing-keyframe fallbacks below (0.0 / total
    duration) just make each affected expanded range as wide as the file
    itself, which the merge step then collapses into one big segment. Pure
    function, no ffmpeg/filesystem access -- the actual segment *files* are
    produced by _extract_video_segment.
    """
    if not blur_intervals:
        return [VideoSegmentPlan(start=0.0, end=total_duration, reencode=False)]

    keyframes = sorted(keyframe_timestamps)

    def at_or_before(t: float) -> float:
        idx = bisect.bisect_right(keyframes, t) - 1
        return keyframes[idx] if idx >= 0 else 0.0

    def at_or_after(t: float) -> float:
        idx = bisect.bisect_left(keyframes, t)
        return keyframes[idx] if idx < len(keyframes) else total_duration

    expanded = [(at_or_before(iv.start), at_or_after(iv.end)) for iv in blur_intervals]
    reencode_ranges = merge_intervals(expanded, merge_gap_seconds=0.0)

    segments: list[VideoSegmentPlan] = []
    cursor = 0.0
    for seg_start, seg_end in reencode_ranges:
        if seg_start > cursor:
            segments.append(VideoSegmentPlan(start=cursor, end=seg_start, reencode=False))
        local_scenes = tuple(
            MuteInterval(start=iv.start - seg_start, end=iv.end - seg_start)
            for iv in blur_intervals
            if iv.start >= seg_start - 1e-6 and iv.end <= seg_end + 1e-6
        )
        segments.append(VideoSegmentPlan(start=seg_start, end=seg_end, reencode=True, local_blur_intervals=local_scenes))
        cursor = seg_end
    if cursor < total_duration:
        segments.append(VideoSegmentPlan(start=cursor, end=total_duration, reencode=False))

    return segments


@dataclass(frozen=True)
class AudioSegmentPlan:
    start: float
    end: float
    mute: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_audio_segments(mute_intervals: list[MuteInterval], total_duration: float) -> list[AudioSegmentPlan]:
    """The audio-side analog of plan_video_segments -- splits [0, total_duration]
    into alternating stream-copy segments (the untouched majority) and mute
    segments (decoded, volume=0 applied for the whole segment, re-encoded).

    Simpler than the video version: audio doesn't need video's keyframe/GOP
    expansion. A stream-copy cut still has to land on a real packet boundary
    (an AAC frame is ~21-23ms at typical rates), but that's negligible next to
    the fixed scene_blur_pad_start/end_seconds safety margin already baked
    into mute_intervals before this ever runs (app.scenes.pipeline.
    apply_scene_blur) -- so mute_intervals are used directly here, no
    per-source keyframe probe or expansion needed.

    Exists because a single whole-file `volume=0:enable=between(...)` pass
    (the previous approach, still used when there's nothing to mute) forces
    ffmpeg to decode+re-encode the *entire* audio track through the filter
    graph even though the filter only actually changes anything during a
    couple of short windows -- fine for a short episode with one relevant
    track, but a real, measured bottleneck on a long movie with multiple
    audio tracks (confirmed directly: ~27 minutes on a 3-hour, 2-track file
    to mute ~60 seconds of it). Pure function, no ffmpeg/filesystem access."""
    if not mute_intervals:
        return [AudioSegmentPlan(start=0.0, end=total_duration, mute=False)]

    merged = merge_intervals([(iv.start, iv.end) for iv in mute_intervals], merge_gap_seconds=0.0)

    segments: list[AudioSegmentPlan] = []
    cursor = 0.0
    for raw_start, raw_end in merged:
        seg_start = max(0.0, raw_start)
        seg_end = min(total_duration, raw_end)
        if seg_end <= cursor:
            continue
        if seg_start > cursor:
            segments.append(AudioSegmentPlan(start=cursor, end=seg_start, mute=False))
        segments.append(AudioSegmentPlan(start=max(cursor, seg_start), end=seg_end, mute=True))
        cursor = seg_end
    if cursor < total_duration:
        segments.append(AudioSegmentPlan(start=cursor, end=total_duration, mute=False))

    return segments


def _parse_keyframe_csv(output: str) -> list[float]:
    """`ffprobe ... -show_entries frame=pts_time -of csv=p=0` output, one
    timestamp per line -- pulled out as its own pure function after a real bug:
    `csv=p=0` with a single requested field still emits a trailing comma per
    line (e.g. "0.000000,"), which silently made every float(line) call fail
    and produce an empty keyframe list (degrading to a full whole-file
    re-encode every time, unnoticed until a live test caught it). Takes the
    first comma-separated field per line instead of parsing the whole line."""
    timestamps = []
    for line in output.splitlines():
        field = line.split(",", 1)[0].strip()
        if not field:
            continue
        try:
            timestamps.append(float(field))
        except ValueError:
            continue
    return timestamps


# 180s was the original ceiling here and proved unreliable in practice: this
# exact command timed out standalone against a real 4K/HEVC file (The
# Housemaid) even though the file's own real job later succeeded, apparently
# because that run benefited from a warm OS/NFS cache the standalone rerun
# didn't have. 900s was confirmed sufficient for that same file in a direct
# diagnostic rerun (completed with 1801 keyframes found).
_KEYFRAME_PROBE_TIMEOUT_SECONDS = 900

async def _probe_keyframe_timestamps(ffprobe_bin: str, video_path: Path) -> list[float]:
    """-skip_frame nokey makes ffprobe skip decoding every non-keyframe, so this
    is fast even on a multi-hour file -- it's reading frame headers, not doing
    real video decode work for the vast majority of frames.

    Used uniformly for every codec, HEVC included. An earlier version of this
    module restricted HEVC sources to a NAL-level "true IDR/BLA only" probe
    (a whole streaming Annex-B walker, since removed), on the theory that
    ffprobe's generic keyframe flag conflating true IDR/BLA with CRA made
    stream-copy-cutting at a CRA unsafe. Real testing against actual 4K HEVC
    files (splicing a re-encoded segment before a stream-copied one) showed
    the "Could not find ref with POC ..." corruption that motivated that
    restriction persisted even at a genuine, confirmed true-IDR boundary with
    matched codecs and byte-correct parameter sets -- it was never about
    which HEVC keyframe type the cut landed on. The real causes were the
    MPEG-TS intermediate container (see _concat_video_segments) and a
    timestamp-metadata bug in the copy-segment split (see
    _normalize_segment_timestamps); with both fixed, the generic keyframe
    probe is exactly as safe for HEVC as it always was for H.264."""
    code, out, err = await _run(
        [
            ffprobe_bin, "-v", "error",
            "-select_streams", "v:0", "-skip_frame", "nokey",
            "-show_entries", "frame=pts_time", "-of", "csv=p=0",
            str(video_path),
        ],
        timeout=_KEYFRAME_PROBE_TIMEOUT_SECONDS,
    )
    if code != 0:
        logger.warning("Keyframe probe failed for %s, falling back to whole-file re-encode: %s", video_path, err.strip())
        return []
    return _parse_keyframe_csv(out)


def _annexb_buf_has_rasl(buf: bytes, max_nals_checked: int = 20) -> bool:
    """Pure NAL-type scan of a raw Annex-B byte buffer: True if any of the
    first max_nals_checked NAL units is RASL_N(8) or RASL_R(9). Split out
    from _cut_point_has_rasl so the actual scanning logic is testable without
    a real ffmpeg subprocess, matching this module's existing convention of
    keeping pure parsing logic separate from the I/O around it."""
    i, n, checked = 0, len(buf), 0
    while i < n - 2 and checked < max_nals_checked:
        if buf[i] == 0 and buf[i + 1] == 0 and buf[i + 2] == 1:
            if i + 3 < n:
                nal_type = (buf[i + 3] >> 1) & 0x3F
                if nal_type in (8, 9):
                    return True
                checked += 1
            i += 3
        else:
            i += 1
    return False


async def _segment_file_starts_with_rasl(ffmpeg_bin: str, ts_path: Path) -> bool:
    """Checks whether an ALREADY-PRODUCED copy segment's own true start has a
    RASL picture (NAL type 8 or 9) immediately following its leading CRA.
    Rare -- across dozens of real candidate keyframes checked directly
    against a real 4K HEVC file this session, only a couple actually carried
    one -- but real: a RASL picture references frames before its CRA that
    won't exist after a splice there, producing genuine "Could not find ref
    with POC ..." corruption specifically at that boundary. This persisted
    even after fixing the two other real, more common causes of that same
    error message (the MPEG-TS intermediate container, see
    _concat_video_segments, and a copy-segment timestamp-metadata bug, see
    _normalize_segment_timestamps).

    Deliberately reads an already-cut segment file's own start rather than
    seeking to an arbitrary timestamp in the middle of the full source file:
    an earlier version of this check did the latter (independent `-ss
    <timestamp>` before `-i`) and gave inconsistent verdicts for the exact
    same real timestamp across repeated runs -- traced directly to the same
    "several-second overshoot" stream-copy seek imprecision
    _split_copy_segments's own docstring already documents for this class of
    source file. Reading straight from an already-produced segment's true
    byte 0 has no seek step to be imprecise about."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(ts_path), "-t", "2",
        "-map", "0:v:0", "-an", "-sn", "-c:v", "copy", "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    buf, _ = await proc.communicate()
    return _annexb_buf_has_rasl(buf)


async def _split_copy_segments(
    ffmpeg_bin: str,
    video_path: Path,
    segment_boundaries: list[float],
    out_dir: Path,
) -> None:
    """One continuous stream-copy pass, split into copy_0000.mkv, copy_0001.mkv,
    ... at segment_boundaries via ffmpeg's *segment muxer* -- deliberately not
    implemented as N independent `-ss <t> ... -c:v copy` extractions.

    Confirmed via direct testing against a real file: independent per-segment
    `-ss <t> -t <duration> -c:v copy` calls do NOT reliably land on the exact
    requested cut point for this source -- every copy segment with a non-zero
    -ss showed the same systematic several-second overshoot (a keyframe/cue-
    index granularity mismatch between what ffprobe's frame-level keyframe
    scan reports and what ffmpeg's own fast-seek actually uses), which
    compounded across several segments into a duration inflation well past
    verify_blurred_output()'s 2-second tolerance. The segment muxer sidesteps
    this entirely: it's one continuous decode-free copy pass that splits as it
    goes, so there's no independent re-seek to disagree with itself -- verified
    directly to reproduce the source's real boundaries to within ~0.02s total
    across a 13-segment split.

    Every boundary gets a copy segment this way, including the ones that will
    actually be re-encoded (see build_blurred_video) -- those get discarded
    unused. Wasteful in principle, but this is a stream-copy pass over the
    whole file either way, the cheapest thing this module does.

    -fflags +genpts: a real bug found via direct testing -- some cut points
    land where several initial frames' presentation order genuinely can't be
    resolved from a pure stream copy alone (deeper B-frame reordering right at
    that specific boundary than at others), leaving a couple of packets with
    no timestamp at all. MPEG-TS silently tolerated that; Matroska's stricter
    muxer rejects it outright ("Can't write packet with unknown timestamp").

    Deliberately NOT passing -reset_timestamps 1 (present in an earlier
    version of this call) -- confirmed directly it fights with +genpts across
    a multi-boundary pass: genpts resolved a problem boundary fine in
    isolation, but the same boundary still failed inside the real multi-cut
    split until -reset_timestamps was also removed, apparently because
    resetting the timeline at each internal cut breaks genpts's ability to
    interpolate from the surrounding context at the NEXT cut. Safe to drop:
    the concat demuxer in _concat_video_segments rebases every file's
    timestamps relative to where the previous one ended regardless of each
    segment's own absolute starting timestamp, so copy segments keeping their
    real source-relative timestamps (instead of each restarting near zero)
    doesn't change the final output's timing -- confirmed directly against a
    real multi-scene job (duration and per-boundary decode/pixel checks all
    matched expectations).
    """
    cmd = [
        ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error", "-fflags", "+genpts",
        "-i", str(video_path),
        "-map", "0:v:0", "-c:v", "copy", "-an", "-sn",
        "-f", "segment", "-segment_format", "matroska",
        "-segment_times", ",".join(f"{t:.3f}" for t in segment_boundaries),
        str(out_dir / "copy_%04d.mkv"),
    ]
    code, _out, err = await _run(cmd, timeout=_SEGMENT_TIMEOUT_SECONDS * 4)
    if code != 0:
        raise RemuxError(f"ffmpeg segment-copy split failed for {video_path}: {err.strip()[-2000:]}")


async def _normalize_segment_timestamps(ffmpeg_bin: str, in_path: Path, out_ts_path: Path) -> None:
    """Fixes a real, separate bug the +genpts/no-reset_timestamps change above
    introduces: without -reset_timestamps, each copy segment keeps the
    ORIGINAL source's absolute timestamps internally (needed to avoid the
    genpts interaction bug -- see _split_copy_segments), but that leaves each
    segment file's own container-level duration/start_time metadata wrong --
    ffprobe (and, critically, the concat demuxer _concat_video_segments
    relies on for sequencing) reports each segment's "duration" as its
    absolute end timestamp rather than its own actual span, since the
    container's start_time metadata never got reset to reflect this file's
    own true beginning. Confirmed directly: a real job's concatenated output
    came out at ~1.95x the correct total duration, with wrong content
    appearing at several timestamps, from exactly this.

    A second, separate stream-copy remux with -avoid_negative_ts make_zero,
    treating the already-cut segment as a fresh input, is what actually fixes
    it -- confirmed directly (a real problem segment's reported duration went
    from a bogus ~6348s to the correct ~2207s, and it still remuxed cleanly).
    Doing this as its own pass (not folded into the original split command)
    is deliberate: applying -avoid_negative_ts at the ORIGINAL segment-muxer
    call has nothing to correct there (no timestamp is actually negative in a
    single continuous pass over the whole file), so it's a no-op unless run
    against each segment independently, after the fact."""
    def build_cmd(tmp_out: Path) -> list[str]:
        return [
            ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error",
            "-i", str(in_path), "-map", "0", "-c", "copy",
            "-avoid_negative_ts", "make_zero", "-f", "matroska", str(tmp_out),
        ]

    await _run_atomic(build_cmd, out_ts_path, _SEGMENT_TIMEOUT_SECONDS, f"ffmpeg segment timestamp-normalize failed for {in_path}")


async def _split_copy_audio_segments(
    ffmpeg_bin: str,
    video_path: Path,
    segment_boundaries: list[float],
    num_audio_streams: int,
    out_dir: Path,
) -> None:
    """Audio-side analog of _split_copy_segments -- one continuous stream-copy
    segment-muxer pass over every audio track at once, split at
    segment_boundaries. Same reasoning applies here: an independent per-
    segment `-ss`/`-t` copy can't reliably land on the exact requested cut
    point, this sidesteps that with one continuous decode-free pass.

    +genpts here too for the same reason as the video split -- audio frames
    don't have video's B-frame reordering, so this is more a defensive
    consistency measure than a confirmed-necessary fix on the audio side, but
    it's a no-op when timestamps are already fine, so there's no cost to
    covering the same class of issue here too. No -reset_timestamps either,
    same reasoning as the video split -- see its docstring."""
    map_args = []
    for i in range(num_audio_streams):
        map_args += ["-map", f"0:a:{i}"]
    cmd = [
        ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error", "-fflags", "+genpts",
        "-i", str(video_path),
        *map_args, "-c:a", "copy", "-vn", "-sn",
        "-f", "segment", "-segment_format", "matroska",
        "-segment_times", ",".join(f"{t:.3f}" for t in segment_boundaries),
        str(out_dir / "acopy_%04d.mkv"),
    ]
    code, _out, err = await _run(cmd, timeout=_SEGMENT_TIMEOUT_SECONDS * 4)
    if code != 0:
        raise RemuxError(f"ffmpeg audio segment-copy split failed for {video_path}: {err.strip()[-2000:]}")


async def _extract_mute_audio_segment(
    ffmpeg_bin: str,
    video_path: Path,
    segment: AudioSegmentPlan,
    num_audio_streams: int,
    audio_bitrate: str,
    out_ts_path: Path,
) -> None:
    """Re-encodes one muted segment's audio, flat volume=0 for every track
    across the *entire* segment -- no enable=between() gating needed the way
    the video blur filter needs it, since by construction (see
    plan_audio_segments) a mute segment's whole span is inside a mute
    interval. Independent per-segment `-ss`/`-t` is accurate here for the same
    reason it is for _extract_reencode_segment: any real decode path seeks
    accurately, unlike a stream copy."""
    def build_cmd(tmp_out: Path) -> list[str]:
        cmd = [ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error"]
        if segment.start > 0:
            cmd += ["-ss", f"{segment.start:.3f}"]
        cmd += ["-i", str(video_path), "-t", f"{segment.duration:.3f}"]

        filter_parts = [f"[0:a:{i}]volume=0[a{i}]" for i in range(num_audio_streams)]
        map_args = []
        codec_args = []
        for i in range(num_audio_streams):
            map_args += ["-map", f"[a{i}]"]
            codec_args += [f"-c:a:{i}", "aac", f"-b:a:{i}", audio_bitrate]

        return cmd + [
            "-filter_complex", ";".join(filter_parts), *map_args, *codec_args, "-vn", "-sn", "-f", "matroska", str(tmp_out),
        ]

    await _run_atomic(
        build_cmd, out_ts_path, _SEGMENT_TIMEOUT_SECONDS,
        f"ffmpeg mute-audio segment re-encode failed for {video_path} [{segment.start:.2f}-{segment.end:.2f}]",
    )


def _reencode_video_codec(source_codec: str) -> str:
    """Picks the re-encode segment's own codec to MATCH the source's, so a
    stream-copy segment (still in the source's native codec) can be
    concatenated after it without a codec discontinuity mid-stream.

    Real bug this fixes: re-encode segments used to hardcode libx264
    unconditionally, regardless of source codec. For an HEVC source, that
    meant every splice between a (forced-H.264) re-encode segment and an
    (HEVC) copy segment fed HEVC bytes into what the concat'd stream declares
    as an H.264 track -- confirmed directly against a real file to produce
    outright garbage (H.264 decoder logging "missing picture in access unit",
    "no frame!" etc. -- a different, more basic failure than any reference-
    frame/keyframe-type issue). x264 remains the default for everything else,
    unchanged from the original behavior."""
    if source_codec in ("hevc", "h265"):
        return "libx265"
    return "libx264"


async def _extract_reencode_segment(
    ffmpeg_bin: str,
    video_path: Path,
    segment: VideoSegmentPlan,
    video_crf: int,
    video_preset: str,
    blur_radius: int,
    blur_power: int,
    out_ts_path: Path,
    source_codec: str = "",
) -> None:
    """Re-encodes one scene (plus its keyframe-alignment buffer), boxblur active
    only during the segment's own local scene windows. Unlike the copy case
    above, independent per-segment `-ss`/`-t` here is accurate: any ffmpeg
    output that requires actual decoding (i.e. not a pure stream copy) performs
    real accurate seeking by default -- decode from the nearest keyframe and
    discard extra frames before the target -- which is exactly what a copy
    can't do. Confirmed directly: these segments' durations matched their
    planned values to within a few milliseconds even before the copy-side fix
    above.

    source_codec picks the output codec via _reencode_video_codec -- see its
    docstring for why this must match the source rather than always being
    libx264. video_crf/video_preset are reused as-is for libx265 too (ffmpeg's
    libx265 wrapper accepts the same -crf/-preset flags); libx265 is
    meaningfully slower than libx264 on non-AVX hardware (confirmed directly:
    ~5.8fps for a short 4K segment) but this only ever runs against the
    approved-scene window, not the whole file, so the absolute cost stays small."""
    codec = _reencode_video_codec(source_codec)

    def build_cmd(tmp_out: Path) -> list[str]:
        cmd = [ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error"]
        if segment.start > 0:
            cmd += ["-ss", f"{segment.start:.3f}"]
        cmd += ["-i", str(video_path), "-t", f"{segment.duration:.3f}"]

        filter_str = build_blur_filter(
            list(segment.local_blur_intervals), input_label="0:v", output_label="vb", radius=blur_radius, power=blur_power
        )
        return cmd + [
            "-filter_complex", filter_str, "-map", "[vb]",
            "-c:v", codec, "-crf", str(video_crf), "-preset", video_preset,
            "-an", "-sn", "-f", "matroska", str(tmp_out),
        ]

    await _run_atomic(
        build_cmd, out_ts_path, _SEGMENT_TIMEOUT_SECONDS,
        f"ffmpeg segment re-encode failed for {video_path} [{segment.start:.2f}-{segment.end:.2f}]",
    )


async def _concat_video_segments(ffmpeg_bin: str, segment_ts_paths: list[Path], out_path: Path) -> None:
    """Lossless concat via ffmpeg's concat *demuxer* (a file-list, not the raw
    concat *protocol*'s dumb byte-level join) -- the demuxer explicitly rebases
    each subsequent file's timestamps to continue smoothly from the previous
    one, which matters here since the copy segments (reset_timestamps=1 from
    the segment muxer) and the independently re-encoded segments don't share a
    common internal timestamp base. Confirmed directly against a mix of both
    kinds of segment that this reproduces the correct total duration.

    Intermediate/output container is Matroska (.mkv), NOT MPEG-TS -- a real,
    previously-undiscovered bug: concatenating an independently re-encoded
    segment before a stream-copied one via MPEG-TS produced genuine
    "Could not find ref with POC ..." decode corruption at the splice point,
    reproducibly, REGARDLESS of what HEVC keyframe type (true IDR/BLA or CRA)
    the copy segment started at -- confirmed directly against two different
    real 4K HEVC files, matched codecs, byte-correct parameter sets/IDR all
    verified present and correctly positioned, so it wasn't a bitstream
    correctness issue at all. Switching the container to Matroska (verified
    via a rigorous, seek-free, frame-exact pixel comparison, not just an
    absence of decoder errors) eliminates the corruption completely at both
    kinds of boundary -- MPEG-TS apparently provides decoders no signal to
    treat an internal splice as a hard discontinuity requiring a full
    reference-buffer reset, something MP4/MKV don't have this gap for. This
    matches how smartcut (github.com/skeskinen/smartcut, a purpose-built tool
    for this exact problem) does it too: it never uses MPEG-TS as an
    intermediate, only MP4/MOV/MKV."""
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file '{p}'\n" for p in segment_ts_paths))

    def build_cmd(tmp_out: Path) -> list[str]:
        return [
            ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-map", "0:v:0", "-c:v", "copy", "-f", "matroska", str(tmp_out),
        ]

    await _run_atomic(build_cmd, out_path, _SEGMENT_TIMEOUT_SECONDS, f"ffmpeg segment concat failed for {out_path}")


async def _concat_audio_segments(
    ffmpeg_bin: str, segment_ts_paths: list[Path], num_audio_streams: int, out_path: Path
) -> None:
    """Audio-side analog of _concat_video_segments -- same concat-demuxer
    reasoning (rebases timestamps between segments of different origin,
    unlike the raw concat protocol's dumb byte-level join)."""
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file '{p}'\n" for p in segment_ts_paths))
    map_args = []
    for i in range(num_audio_streams):
        map_args += ["-map", f"0:a:{i}"]

    def build_cmd(tmp_out: Path) -> list[str]:
        return [
            ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            *map_args, "-c:a", "copy", "-f", "matroska", str(tmp_out),
        ]

    await _run_atomic(build_cmd, out_path, _SEGMENT_TIMEOUT_SECONDS, f"ffmpeg audio segment concat failed for {out_path}")


def _blur_job_fingerprint(
    video_path: Path,
    blur_intervals: list[MuteInterval],
    mute_intervals: list[MuteInterval],
    video_crf: int,
    video_preset: str,
    blur_radius: int,
    blur_power: int,
    audio_bitrate: str,
) -> str:
    """A stable fingerprint of every input that determines this job's exact
    output -- used to detect a work_dir left by a *different* plan (approved
    scenes changed between attempts, settings changed, etc.) so a resume
    never silently reuses segment files that don't actually match the plan
    being run now. Not a security hash, just a cheap change-detector."""
    payload = repr((
        str(video_path),
        [(iv.start, iv.end) for iv in blur_intervals],
        [(iv.start, iv.end) for iv in mute_intervals],
        video_crf, video_preset, blur_radius, blur_power, audio_bitrate,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _reset_stale_work_dir(work_dir: Path, fingerprint: str) -> None:
    fingerprint_path = work_dir / "fingerprint.txt"
    if fingerprint_path.exists() and fingerprint_path.read_text().strip() == fingerprint:
        return
    for child in work_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    fingerprint_path.write_text(fingerprint)


async def build_blurred_video(
    *,
    video_path: Path,
    blur_intervals: list[MuteInterval],
    mute_intervals: list[MuteInterval],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_crf: int,
    video_preset: str,
    work_dir: Path,
    blur_radius: int = 90,
    blur_power: int = 8,
    audio_bitrate: str = "192k",
    timeout: float = 14400,  # ceiling for the final mux step only now, see module docstring
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> Path:
    """Produces a temp file (same directory as video_path, so the final publish
    step is a same-filesystem atomic rename, not a cross-filesystem copy) with
    the video blurred during blur_intervals and every audio stream muted during
    mute_intervals -- deliberately separate lists, not the same windows reused
    for both: blurring is what every approved scene gets, muting is an explicit
    per-scene opt-in (default off) since plot-relevant nudity doesn't always
    mean the dialogue over it needs muting too. mute_intervals is always a
    subset of blur_intervals. Everything else (subtitles/chapters/attachments)
    passed through untouched.

    work_dir is a persistent, caller-owned directory (see apply_blur -- it's
    keyed off the SceneJob id, under app_settings.data_dir so it survives a
    container restart, not /tmp which doesn't) -- every intermediate segment
    is written here and reused across calls instead of an auto-deleted temp
    dir, which is what makes this resumable. See the module docstring for the
    split/re-encode/concat pipeline this runs instead of one whole-file
    re-encode, and _blur_job_fingerprint for the staleness guard."""
    src_probe = await probe(ffprobe_bin, video_path)
    total_duration = float(src_probe["format"].get("duration", 0))
    audio_streams = _audio_streams(src_probe)
    if not audio_streams:
        raise RemuxError("No audio streams found in source file")

    tmp_path = video_path.with_name(f".{video_path.stem}.spf-blur-tmp{video_path.suffix}")

    work_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _blur_job_fingerprint(
        video_path, blur_intervals, mute_intervals, video_crf, video_preset, blur_radius, blur_power, audio_bitrate
    )
    _reset_stale_work_dir(work_dir, fingerprint)
    tmp_root = work_dir

    if on_stage is not None:
        await on_stage("Scanning keyframes")
    video_stream = next((s for s in src_probe["streams"] if s["codec_type"] == "video"), None)
    video_codec = (video_stream or {}).get("codec_name", "")
    # Generic keyframe probe works for every codec, HEVC included -- see
    # _probe_keyframe_timestamps for why an earlier HEVC-specific "true
    # IDR/BLA only" restriction here (worked around via a whole streaming
    # Annex-B walker, since removed) turned out to be the wrong fix for most
    # of what it was trying to prevent -- the MPEG-TS container and a
    # timestamp-metadata bug (see _concat_video_segments and
    # _normalize_segment_timestamps) accounted for the vast majority of real
    # corruption regardless of keyframe type. The one thing that restriction
    # WAS incidentally protecting against -- a CRA cut point with a RASL
    # picture following it -- is real but rare, so it's handled narrowly
    # instead, after the real split below (see _segment_file_starts_with_rasl
    # for why it has to be checked against the actually-produced files, not a
    # separate probing pass against the source).
    keyframe_timestamps = list(await _probe_keyframe_timestamps(ffprobe_bin, video_path))
    segments = plan_video_segments(blur_intervals, keyframe_timestamps, total_duration)

    try:
        concatenated_video_path = tmp_root / "concatenated.mkv"
        if concatenated_video_path.exists():
            # A prior (interrupted) attempt already finished this whole stage
            # -- _run_atomic's rename-on-success means this file existing at
            # all means it's complete, not partially written.
            logger.info("Resuming blur for %s: video already combined, skipping segment production", video_path)
            reencode_segments = [s for s in segments if s.reencode]
            total_reencode_seconds = sum(s.duration for s in reencode_segments) or 1.0
            done_reencode_seconds = 0.0
        else:
            segment_ts_paths: list[Path] = []
            # One continuous stream-copy pass covers every boundary at once
            # (see _split_copy_segments for why this replaced N independent
            # per-segment `-ss`/`-t` copy extractions). Always redone in full
            # on a resume rather than individually checkpointed -- it's a
            # stream copy over the whole file, cheap regardless, and not
            # worth the complexity of partial-resume for this specific step.
            if len(segments) > 1:
                if on_stage is not None:
                    await on_stage("Splitting video")
                boundaries = [s.end for s in segments[:-1]]
                await _split_copy_segments(ffmpeg_bin, video_path, boundaries, tmp_root)

            if video_codec in ("hevc", "h265") and len(segments) > 1:
                # Validate against the segments actually just produced, not a
                # separate probe against the source -- an earlier version of
                # this check seeked independently into the source file per
                # candidate and got inconsistent verdicts for the same real
                # timestamp across repeated runs (the same stream-copy seek
                # imprecision _split_copy_segments's own docstring already
                # documents for this class of source). Reading an
                # already-cut segment's own true start has no seek step to
                # be imprecise about.
                for _attempt in range(10):
                    bad_boundary = None
                    for i, seg in enumerate(segments):
                        if not seg.reencode and seg.start > 0.0:
                            if await _segment_file_starts_with_rasl(ffmpeg_bin, tmp_root / f"copy_{i:04d}.mkv"):
                                bad_boundary = seg.start
                                break
                    if bad_boundary is None:
                        break
                    logger.info("Cut point %.3fs for %s has a RASL picture following it -- excluding and re-splitting", bad_boundary, video_path)
                    keyframe_timestamps = [t for t in keyframe_timestamps if abs(t - bad_boundary) > 1e-6]
                    segments = plan_video_segments(blur_intervals, keyframe_timestamps, total_duration)
                    # Old copy_*.mkv indices no longer correspond to the new
                    # plan's boundaries -- wipe and re-split rather than risk
                    # a later stale-file reuse under a shifted index.
                    for stale in tmp_root.glob("copy_*.mkv"):
                        stale.unlink(missing_ok=True)
                    for stale in tmp_root.glob("reencode_*.mkv"):
                        stale.unlink(missing_ok=True)
                    if len(segments) > 1:
                        if on_stage is not None:
                            await on_stage("Splitting video")
                        boundaries = [s.end for s in segments[:-1]]
                        await _split_copy_segments(ffmpeg_bin, video_path, boundaries, tmp_root)
                else:
                    logger.warning("Could not find an all-RASL-safe segment plan for %s after 10 attempts -- proceeding anyway", video_path)

            reencode_segments = [s for s in segments if s.reencode]
            total_reencode_seconds = sum(s.duration for s in reencode_segments) or 1.0
            done_reencode_seconds = 0.0
            logger.info(
                "Scene-blur split for %s: %d segment(s) (%d re-encode, %d copy), "
                "%.1fs of %.1fs actually needs re-encoding (%.1f%%)",
                video_path, len(segments), len(reencode_segments), len(segments) - len(reencode_segments),
                sum(s.duration for s in reencode_segments), total_duration,
                100 * sum(s.duration for s in reencode_segments) / total_duration if total_duration else 0,
            )

            total_scene_count = sum(len(s.local_blur_intervals) for s in reencode_segments)
            scenes_done = 0
            for i, segment in enumerate(segments):
                if segment.reencode:
                    # A segment can carry more than one original scene when two
                    # approved scenes are close enough together to share the
                    # same keyframe-expanded re-encode range (see
                    # plan_video_segments) -- report progress against the
                    # scene count you actually approved, not the (fewer)
                    # merged segments, so this doesn't read as "missing" scenes.
                    n = len(segment.local_blur_intervals)
                    seg_ts_path = tmp_root / f"reencode_{i:04d}.mkv"
                    if seg_ts_path.exists():
                        # Already produced by a prior interrupted attempt.
                        scenes_done += n
                        segment_ts_paths.append(seg_ts_path)
                        done_reencode_seconds += segment.duration
                        if on_progress is not None:
                            await on_progress(min(1.0, done_reencode_seconds / total_reencode_seconds))
                        continue
                    if n > 1:
                        stage_msg = f"Blurring scenes {scenes_done + 1}-{scenes_done + n}/{total_scene_count}"
                    else:
                        stage_msg = f"Blurring scene {scenes_done + 1}/{total_scene_count}"
                    scenes_done += n
                    if on_stage is not None:
                        await on_stage(stage_msg)
                    await _extract_reencode_segment(
                        ffmpeg_bin, video_path, segment, video_crf, video_preset, blur_radius, blur_power, seg_ts_path,
                        source_codec=video_codec,
                    )
                    done_reencode_seconds += segment.duration
                    if on_progress is not None:
                        await on_progress(min(1.0, done_reencode_seconds / total_reencode_seconds))
                else:
                    # Raw copy segment already produced by the whole-file
                    # split pass above, but its own duration/start_time
                    # metadata is wrong (see _normalize_segment_timestamps)
                    # -- normalize it into a separate file so "the normalized
                    # file exists" stays a reliable resume-completion signal
                    # independent of the raw one.
                    raw_ts_path = tmp_root / f"copy_{i:04d}.mkv"
                    seg_ts_path = tmp_root / f"copy_{i:04d}_fixed.mkv"
                    if not seg_ts_path.exists():
                        await _normalize_segment_timestamps(ffmpeg_bin, raw_ts_path, seg_ts_path)
                segment_ts_paths.append(seg_ts_path)

            if on_stage is not None:
                await on_stage("Combining segments")
            await _concat_video_segments(ffmpeg_bin, segment_ts_paths, concatenated_video_path)

        # Same split/copy/reencode/concat approach as video, applied to
        # audio -- only the muted windows get decoded+re-encoded, the rest
        # is stream-copied. Skipped entirely when there's nothing to mute
        # (the common case): _mux_final_output's plain -c:a copy path
        # below is already as cheap as this could ever get. See
        # plan_audio_segments for why this exists at all -- a real,
        # measured bottleneck on a long multi-track file, not a
        # hypothetical one.
        concatenated_audio_path: Path | None = None
        if mute_intervals:
            concatenated_audio_path = tmp_root / "concatenated_audio.mkv"
            if concatenated_audio_path.exists():
                logger.info("Resuming blur for %s: audio already combined, skipping", video_path)
            else:
                if on_stage is not None:
                    await on_stage("Splitting audio")
                audio_segments = plan_audio_segments(mute_intervals, total_duration)
                audio_segment_paths: list[Path] = []
                if len(audio_segments) > 1:
                    audio_boundaries = [s.end for s in audio_segments[:-1]]
                    await _split_copy_audio_segments(
                        ffmpeg_bin, video_path, audio_boundaries, len(audio_streams), tmp_root
                    )
                for i, aseg in enumerate(audio_segments):
                    if aseg.mute:
                        aseg_ts_path = tmp_root / f"amute_{i:04d}.mkv"
                        if not aseg_ts_path.exists():
                            await _extract_mute_audio_segment(
                                ffmpeg_bin, video_path, aseg, len(audio_streams), audio_bitrate, aseg_ts_path
                            )
                    else:
                        # Same duration/start_time metadata problem and fix
                        # as the video-side copy segments -- see
                        # _normalize_segment_timestamps.
                        raw_aseg_ts_path = tmp_root / f"acopy_{i:04d}.mkv"
                        aseg_ts_path = tmp_root / f"acopy_{i:04d}_fixed.mkv"
                        if not aseg_ts_path.exists():
                            await _normalize_segment_timestamps(ffmpeg_bin, raw_aseg_ts_path, aseg_ts_path)
                    audio_segment_paths.append(aseg_ts_path)

                if on_stage is not None:
                    await on_stage("Combining audio")
                await _concat_audio_segments(
                    ffmpeg_bin, audio_segment_paths, len(audio_streams), concatenated_audio_path
                )

        if on_stage is not None:
            await on_stage("Finalizing audio and muxing")
        await _mux_final_output(
            ffmpeg_bin=ffmpeg_bin,
            video_path=video_path,
            concatenated_video_path=concatenated_video_path,
            concatenated_audio_path=concatenated_audio_path,
            audio_streams=audio_streams,
            out_path=tmp_path,
            timeout=timeout,
        )
    except asyncio.CancelledError:
        tmp_path.unlink(missing_ok=True)
        raise  # work_dir deliberately left in place -- see apply_blur, cleaned up only on full success
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return tmp_path


async def _mux_final_output(
    *,
    ffmpeg_bin: str,
    video_path: Path,
    concatenated_video_path: Path,
    concatenated_audio_path: Path | None,
    audio_streams: list[dict],
    out_path: Path,
    timeout: float,
) -> None:
    """Combines the already-fully-processed concatenated video (input 0, pure
    stream-copy here -- all the real encoding work already happened per-segment)
    with the original source's subtitles/chapters/metadata (input 1) and audio.

    Audio comes from wherever it was actually processed: concatenated_audio_path
    (input 2, already fully muted+copied+concatenated by the split pipeline in
    build_blurred_video, see plan_audio_segments) when there was anything to
    mute, or directly from the original (input 1) with a plain -c:a copy when
    there wasn't -- both cases are pure stream-copy by the time this runs,
    this function does no audio encoding itself either way."""
    map_args = ["-map", "0:v:0"]
    codec_args = ["-c:v", "copy"]

    inputs = [concatenated_video_path, video_path]
    if concatenated_audio_path is not None:
        audio_input_index = 2
        inputs.append(concatenated_audio_path)
    else:
        audio_input_index = 1
    for i in range(len(audio_streams)):
        map_args += ["-map", f"{audio_input_index}:a:{i}"]
        codec_args += [f"-c:a:{i}", "copy"]

    # Subtitles/chapters/metadata always come from the original (index 1) --
    # the concatenated .mkv video (index 0) and audio (index 2, if present)
    # carry only their own stream types.
    map_args += ["-map", "1:s?", "-map", "1:d?", "-map", "1:t?"]

    cmd = [ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error"]
    for inp in inputs:
        cmd += ["-i", str(inp)]
    cmd += map_args + ["-map_metadata", "1", "-map_chapters", "1"] + codec_args + [
        "-c:s", "copy",
        # Plex's "Versions" picker shows auto-derived technical info (resolution/
        # codec/container), not a custom label -- since this file matches the
        # source on all of those, the picker alone can't tell them apart. These
        # tags are a cheap attempt at a workaround (some clients surface
        # embedded title metadata elsewhere in the UI), not a confirmed fix --
        # -map_metadata above already copied the source's tags, so these,
        # placed after, override just the "title" key.
        "-metadata", "title=Vulgarr Edit",
        "-metadata:s:v:0", "title=Vulgarr Edit",
        str(out_path),
    ]

    code, _out, err = await _run(cmd, timeout=timeout)
    if code != 0:
        raise RemuxError(f"ffmpeg final mux failed for {video_path}: {err.strip()[-4000:]}")


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

    # Decode a window spanning EVERY blurred scene, not just the first --
    # a real bug shipped past an earlier version of this check that only
    # looked at blurred_windows[0]: a corrupted segment-copy cut (wrong
    # HEVC CRA vs. IDR boundary, see module docstring) produced a technically
    # decodable file where only the second scene's boundary was actually
    # broken, which a first-window-only check can't catch. Corrupted-but-
    # decodable frames don't necessarily error loudly, so this is the whole
    # point of checking every window rather than trusting one.
    for window in blurred_windows:
        check_start = max(0.0, window.start - 2.0)
        check_duration = min(15.0, (window.end - window.start) + 4.0)
        code, _out, err = await _run(
            [
                ffmpeg_bin, "-v", "error",
                "-ss", f"{check_start:.3f}", "-i", str(tmp_path),
                "-t", f"{check_duration:.3f}", "-f", "null", "-",
            ],
            timeout=120,
        )
        if code != 0 or err.strip():
            raise RemuxError(
                f"Blurred output failed decode check for {tmp_path} at window "
                f"[{window.start:.2f}-{window.end:.2f}]: {err.strip()[-2000:]}"
            )


async def apply_blur(
    *,
    video_path: Path,
    blur_intervals: list[MuteInterval],
    mute_intervals: list[MuteInterval],
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_crf: int,
    video_preset: str,
    work_dir: Path,
    blur_radius: int = 90,
    blur_power: int = 8,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> Path:
    """work_dir: a persistent directory (see app.scenes.pipeline.apply_scene_blur
    for how it's derived from the SceneJob id) that build_blurred_video reuses
    across resumed attempts -- deleted here only once the output has been
    verified and published, so a failed or interrupted run always leaves it
    behind for the next attempt to pick up from."""
    if not video_path.exists():
        raise RemuxError(f"Video file does not exist: {video_path}")
    if not blur_intervals:
        raise RemuxError("No approved scene windows to apply")

    src_probe = await probe(ffprobe_bin, video_path)

    tmp_path = await build_blurred_video(
        video_path=video_path,
        blur_intervals=blur_intervals,
        mute_intervals=mute_intervals,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        video_crf=video_crf,
        video_preset=video_preset,
        work_dir=work_dir,
        blur_radius=blur_radius,
        blur_power=blur_power,
        on_progress=on_progress,
        on_stage=on_stage,
    )

    if on_stage is not None:
        await on_stage("Verifying output")
    try:
        await verify_blurred_output(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            original_probe=src_probe,
            tmp_path=tmp_path,
            blurred_windows=blur_intervals,
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if on_stage is not None:
        await on_stage("Publishing Vulgarr Edit")

    final_path = sibling_edit_path(video_path)
    tmp_path.replace(final_path)  # same-directory rename: atomic, no partial-file window

    # Full success -- the work_dir has served its purpose. A future Apply for
    # this same title (even if it needs to redo everything, e.g. a rejected
    # scene) gets a fresh job id and thus a fresh work_dir regardless.
    shutil.rmtree(work_dir, ignore_errors=True)

    return final_path
