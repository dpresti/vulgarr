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
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.audio.mute import MuteInterval, _MAX_TERMS_PER_STAGE
from app.common.intervals import merge_intervals
from app.mux.remux import (
    ProgressCallback,
    RemuxError,
    StageCallback,
    _audio_streams,
    _run,
    _run_with_progress,
    probe,
)

logger = logging.getLogger(__name__)

# Generous per-segment ceiling -- even a single scene's re-encode should finish
# in well under this on any hardware this app has actually run on, but a
# pathologically long approved "scene" (a user could adjust one to span many
# minutes) shouldn't hang forever either.
_SEGMENT_TIMEOUT_SECONDS = 1800

# Ceiling for the whole-file re-encode fallback (see _build_video_whole_file_reencode)
# -- a real, full-length re-encode, not a short segment, so this needs to be far
# more generous than a hang-detection timeout normally would be, not a realistic
# "this is how long it should take" estimate. Confirmed directly against the
# real file this fallback exists for (28 Years Later, ~165,600 frames): even at
# the fallback's own faster preset (see _WHOLE_FILE_FALLBACK_PRESET), real
# throughput on this CPU was ~7fps, i.e. ~6.5 hours end to end -- 24h leaves
# real margin above that without being effectively infinite.
_WHOLE_FILE_REENCODE_TIMEOUT_SECONDS = 24 * 60 * 60

# Deliberately faster than whatever blur_video_preset the user has configured
# for the normal segment-based re-encode (which only ever encodes a short scene,
# where preset speed barely matters) -- this fallback re-encodes the ENTIRE
# file, where it matters enormously. Confirmed directly against the real file:
# medium=3.43fps, veryfast=7.02fps, faster=7.20fps -- faster's extra 3% over
# veryfast isn't worth its slightly worse quality-per-bit, so veryfast wins.
_WHOLE_FILE_FALLBACK_PRESET = "veryfast"


# A confirmed-benign ffmpeg stderr line, specific to verify_blurred_output's
# own null-format decode-check invocation: seeking (-ss) to within ~3s of a
# short re-encoded segment's own GOP start makes ffmpeg's null muxer emit
# this exact warning even though the process exits 0 and the file decodes
# perfectly cleanly end to end with no seek at all -- confirmed directly,
# live, on two separate real re-encodes of the same file/scene (the exact
# byte offsets in the message differ run to run, since x265's own encode
# isn't fully deterministic, but the message text doesn't). Widening the
# lookback before the window of interest was tried first and found NOT
# reliable -- confirmed directly: a second re-encode of the identical scene
# reproduced the same warning at a different seek offset, so this can't be
# dodged with a fixed distance. Confirmed this is genuinely a false positive,
# not a laxer check papering over real corruption: a REAL corrupted file
# (the exact "Could not find ref with POC"/"alignment_bit_equal_to_one"
# splice-corruption bug this module spent most of a session chasing) does
# NOT produce this message under the identical check, even though it fails
# loudly with everything else -- so filtering this one specific message
# doesn't weaken this function's ability to catch that failure class at all.
_BENIGN_DECODE_WARNING_RE = re.compile(
    r"^\[null[^\]]*\] Application provided invalid, non monotonically increasing dts to muxer.*$", re.MULTILINE
)


def _filter_benign_decode_warnings(stderr_text: str) -> str:
    """Strips _BENIGN_DECODE_WARNING_RE lines from a decode-check's captured
    stderr, leaving only content that should actually fail the check. Split
    out as a pure function for testability, same convention as this file's
    other stderr/output parsers."""
    return _BENIGN_DECODE_WARNING_RE.sub("", stderr_text).strip()


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
        # Overlap + clamp to the segment's own (possibly keyframe/duration-
        # clamped) bounds, rather than requiring the scene's original,
        # unclamped interval to fit entirely inside it -- a scene padded past
        # total_duration (e.g. near the file's end) would otherwise fail a
        # strict containment check against the clamped seg_end and silently
        # get dropped from local_blur_intervals, leaving its segment
        # re-encoded but never actually blurred.
        local_scenes = tuple(
            MuteInterval(start=max(iv.start, seg_start) - seg_start, end=min(iv.end, seg_end) - seg_start)
            for iv in blur_intervals
            if iv.start < seg_end - 1e-6 and iv.end > seg_start + 1e-6
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


def _annexb_buf_last_picture_nal_type(buf: bytes) -> int | None:
    """Pure NAL-type scan of a raw Annex-B byte buffer: the type of the LAST
    picture-class NAL unit found (0-21 -- VCL slice types; excludes parameter
    sets/SEI/AUD, types 32+), or None if the buffer has no picture NALs at
    all. Split out for testability, same convention as _annexb_buf_has_rasl.

    Unlike that function, this one has no early-exit NAL-count cap -- a real
    bug caught by testing this live against a real file: capping at 200 (10x
    that function's 20, seemingly generous) still stopped well short of the
    buffer's true end, because a GOP spanning the -sseof window in
    _segment_file_ends_with_cra can contain many thousands of small slice
    NALs between the boundary CRA and the container's real last frame. We
    need the actual last one regardless of how many precede it, so this
    scans the whole buffer every time -- a few hundred KB of Python byte
    scanning, not expensive enough to matter for a check that only runs a
    handful of times per Apply job."""
    i, n = 0, len(buf)
    last_picture_type = None
    while i < n - 2:
        if buf[i] == 0 and buf[i + 1] == 0 and buf[i + 2] == 1:
            if i + 3 < n:
                nal_type = (buf[i + 3] >> 1) & 0x3F
                if nal_type <= 21:
                    last_picture_type = nal_type
            i += 3
        else:
            i += 1
    return last_picture_type


async def _segment_file_ends_with_cra(ffmpeg_bin: str, ts_path: Path) -> bool:
    """Checks whether an ALREADY-PRODUCED copy segment's own true end lands on
    a CRA frame (NAL type 21) -- confirmed directly, live, against a real
    file (28 Years Later): a copy segment ending on a CRA produces genuine
    "Could not find ref with POC ..." decode corruption immediately after the
    splice into the following re-encoded segment, even with no RASL picture
    involved at all (ruled out directly: the frame immediately after this
    CRA, still within the same segment, was a plain TRAIL_R, not RASL/RADL) --
    both this segment and the one after it decode perfectly cleanly in total
    isolation, so this is specific to CRA-as-a-splice-boundary itself, not a
    decodable-on-its-own bitstream defect. Matches prior art (smartcut,
    already cited in _concat_video_segments) treating CRA as categorically
    unsafe to cut on, unlike true IDR/BLA (NAL types 16-20).

    This is the copy-segment-END-side counterpart to
    _segment_file_starts_with_rasl's copy-segment-START-side check -- that
    one guards a copy segment's start against inheriting a bad boundary from
    whatever came before; this one guards a copy segment's end against
    handing a bad boundary to whatever comes after (always a re-encoded
    segment, per plan_video_segments -- a copy segment never directly abuts
    another copy segment).

    Reads from the already-cut segment file's own true end via -sseof, same
    "no independent re-seek to be imprecise about" reasoning as the start-side
    check and _segment_actual_end_time."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-v", "error", "-nostdin", "-sseof", "-2", "-i", str(ts_path),
        "-map", "0:v:0", "-an", "-sn", "-c:v", "copy", "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    buf, _ = await proc.communicate()
    return _annexb_buf_last_picture_nal_type(buf) == 21


@dataclass(frozen=True)
class _SourceHevcParams:
    """The subset of a source HEVC stream's own active SPS/PPS derived-state
    fields that _build_matching_x265_params needs to reproduce in a re-encoded
    segment -- see that function's docstring for why this matters at all."""

    high_tier: bool
    level_x265: str  # e.g. "4.1", derived from general_level_idc / 30
    dpb_minus1: int  # sps_max_dec_pic_buffering_minus1 -- passed to x265 as --ref directly
    wpp: bool  # entropy_coding_sync_enabled_flag
    deblock: tuple[int, int] | None  # (pps_beta_offset_div2, pps_tc_offset_div2), or None if the PPS doesn't signal custom deblocking


# Exact field-name strings as printed by ffmpeg's trace_headers bitstream
# filter -- confirmed directly against real output. VPS-scoped duplicates of
# the profile/tier/level fields use a different prefix (vps_max_... vs
# sps_max_...), so there's no collision between VPS and SPS occurrences of
# these names; general_tier_flag/general_level_idc do appear under both VPS's
# and SPS's own profile_tier_level(), but per spec these always match for
# single-layer HEVC (the only kind this app ever produces or consumes), so
# which occurrence wins doesn't matter in practice.
_TRACE_HEADERS_FIELD_RE = re.compile(r"^\[trace_headers[^\]]*\]\s+\d+\s+(\S+)\s+\S+\s+=\s+(-?\d+)\s*$", re.MULTILINE)


def _parse_hevc_trace_fields(trace_text: str) -> _SourceHevcParams | None:
    """Pure parser for _probe_source_hevc_params's captured ffmpeg output --
    split out for testability with a captured text fixture, same convention
    as _parse_keyframe_csv vs _probe_keyframe_timestamps.

    Returns None if any of the four fields this app can always expect from a
    real HEVC SPS/PPS (tier/level/dpb/wpp) is missing -- a non-HEVC source, an
    unexpected ffmpeg/trace_headers output format (e.g. a future ffmpeg
    version changing field names), or a garbled probe. Callers treat None as
    "skip the matching, re-encode exactly as before" -- graceful degradation
    to this feature's pre-existing behavior, same convention already used for
    a RASL-detection probe failure elsewhere in this file."""
    fields: dict[str, int] = {}
    for m in _TRACE_HEADERS_FIELD_RE.finditer(trace_text):
        fields[m.group(1)] = int(m.group(2))

    try:
        high_tier = bool(fields["general_tier_flag"])
        level_idc = fields["general_level_idc"]
        dpb_minus1 = fields["sps_max_dec_pic_buffering_minus1[0]"]
        wpp = bool(fields["entropy_coding_sync_enabled_flag"])
    except KeyError:
        return None

    deblock = None
    if fields.get("deblocking_filter_control_present_flag") == 1:
        try:
            deblock = (fields["pps_beta_offset_div2"], fields["pps_tc_offset_div2"])
        except KeyError:
            pass  # signaled as present but fields missing -- leave unmatched rather than guess

    return _SourceHevcParams(
        high_tier=high_tier, level_x265=f"{level_idc / 30:.1f}", dpb_minus1=dpb_minus1, wpp=wpp, deblock=deblock
    )


async def _probe_source_hevc_params(ffmpeg_bin: str, video_path: Path, at_seconds: float) -> _SourceHevcParams | None:
    """Probes the source's own active HEVC parameter-set fields at the point a
    re-encoded segment is about to start -- this is the parameter-set state
    the decoder will have just been given right before transitioning into our
    re-encode, i.e. exactly what _build_matching_x265_params needs to
    replicate. No decode needed -- trace_headers runs on the packet stream
    itself (-c:v copy), so this is cheap even though -v trace's own decode
    logging is verbose; confirmed directly this session that this exact
    invocation reliably prints parsed SPS/PPS field values."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-v", "trace", "-ss", f"{at_seconds:.3f}", "-i", str(video_path), "-t", "1",
        "-map", "0:v:0", "-c:v", "copy", "-bsf:v", "trace_headers", "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _out, err = await proc.communicate()
    return _parse_hevc_trace_fields(err.decode(errors="replace"))


def _build_matching_x265_params(p: _SourceHevcParams) -> str:
    """Builds an -x265-params value that makes a re-encoded segment's own
    SPS/PPS match the source's derived-state fields as closely as x265's CLI
    allows.

    Why this exists: a real, reproducible libavcodec HEVC decoder bug,
    confirmed directly against the real segment files that corrupted 3 times
    this session (see _concat_video_segments' docstring for the full
    history). The stream-copied segments around a blurred scene carry the
    SOURCE's own SPS/PPS (parameter-set ID 0); the freshly libx265-encoded
    scene segment also uses ID 0, but by default with DIFFERENT
    derived-state fields -- tier, level, DPB buffering size, reorder/latency
    counts, WPP wavefront sync, and deblocking-parameter presence. Direct
    inspection via trace_headers confirmed the source's own parameter sets
    never vary internally, so this is specifically about the re-encode
    silently redefining an already-active ID with different content, which
    the decoder doesn't handle correctly across the splice.

    Confirmed by direct live testing against the real failing file/scene:
    `open-gop=0` alone (the closed-GOP hypothesis -- x265 defaults to
    open-GOP, a documented "unsafe to splice" HEVC construct) was
    INSUFFICIENT on its own, identical corruption persisted. Matching tier,
    level, DPB size (via --ref), WPP, and deblocking offsets -- everything
    this function builds -- produced a completely clean decode across a wide
    window spanning both splice points.

    This is a reliability-improving heuristic layered on top of the existing
    verify_blurred_output decode-integrity check and whole-file fallback, not
    a hard guarantee: the ref->dpb_minus1 mapping was confirmed empirically
    for one file's specific bframes/b-pyramid settings, not derived from
    x265's own source, so a file whose reorder/latency needs don't line up
    with a plain --ref override may still get an imperfect match. Any
    residual mismatch still safely falls through to the fallback exactly as
    it did before this fix existed."""
    parts = [
        "open-gop=0",
        f"high-tier={1 if p.high_tier else 0}",
        f"level-idc={p.level_x265}",
        f"ref={p.dpb_minus1}",
    ]
    if p.wpp:
        # A real allocated thread pool is required for x265 to actually honor
        # wpp on a short segment -- confirmed directly this session: without
        # an explicit pool size, x265 silently disables WPP for a short clip
        # ("No thread pool allocated, --wpp disabled") regardless of this
        # flag. A fixed size, not pools=+ (auto) -- not the configuration
        # that was actually tested.
        parts += ["pools=4", "wpp=1"]
    else:
        parts.append("wpp=0")
    if p.deblock is not None:
        parts.append(f"deblock={p.deblock[0]},{p.deblock[1]}")
    return ":".join(parts)


# How far an already-produced copy segment's true first-frame timestamp is
# allowed to drift from the boundary plan_video_segments actually requested
# before build_blurred_video's retry loop (comparing this function's return
# value against the planned boundary) treats it as a bad cut -- generous
# relative to normal sub-frame-interval seek jitter (a real frame is ~0.04s
# at 24fps), but tight enough to catch the multi-second snaps a keyframe
# mismatch produces. Real numbers seen directly on a real source: correct
# cuts landed within ~0.02s of their plan; bad ones were consistently several
# seconds to ~10s off.
_BOUNDARY_DRIFT_TOLERANCE_SECONDS = 1.0


async def _segment_actual_start_time(ffprobe_bin: str, ts_path: Path) -> float | None:
    """Reads an already-produced copy segment's own true first video frame
    timestamp -- the ground truth for where ffmpeg's segment muxer actually
    cut, as opposed to where plan_video_segments requested (ts_path's file
    name encodes which boundary that is). Same "read the produced file
    itself, don't re-seek" reasoning as _segment_file_starts_with_rasl above:
    an independent re-seek into the source gave inconsistent verdicts for the
    same real timestamp across repeated runs.

    Codec-agnostic, unlike the RASL check above (which only detects one
    specific HEVC picture-type corruption) -- this instead catches the more
    general case that check can't: the segment muxer's own keyframe-cut
    decision landing on a different frame than plan_video_segments' keyframe
    probe expected, for ANY codec. Confirmed directly against a real H.264
    source (24000/1001fps, no HEVC involved at all): several cuts landed
    multiple seconds away from their planned boundary, none of them anywhere
    near a RASL-class issue, but all past this function's tolerance -- and
    every one exactly matched a copy segment whose overall duration came out
    wrong, which is what first surfaced this as a real (not theoretical) bug:
    a whole-file Apply job failing verify_blurred_output's duration check by
    tens of seconds despite the RASL check finding nothing, because that
    check never runs for a non-HEVC source at all.

    Returns None (never treated as bad -- see the caller's None-safe compare)
    if the probe itself fails or the segment is unexpectedly empty; a boundary
    this function can't evaluate shouldn't block on its own, since
    verify_blurred_output's own whole-file duration check is still the real
    safety net regardless of what this narrower per-boundary check concludes.
    """
    code, out, _err = await _run(
        [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pts_time", "-read_intervals", "%+#1",
            "-of", "csv=p=0", str(ts_path),
        ],
        timeout=30,
    )
    if code != 0:
        return None
    timestamps = _parse_keyframe_csv(out)
    return timestamps[0] if timestamps else None


async def _segment_actual_end_time(ffprobe_bin: str, ts_path: Path, actual_start: float) -> float | None:
    """actual_start plus this segment's own real container duration -- the
    ground truth for the OTHER boundary a copy segment touches (the cut
    AFTER it, where the next re-encode segment is expected to pick up).
    Checking only a copy segment's start (_segment_actual_start_time) misses
    this side entirely, which is the more dangerous direction to get wrong:
    a late real cut here means the copy segment eats into the front of the
    next approved scene, and unlike the start-side case, the re-encode
    segment right after it always covers its own exact planned window
    regardless (accurate `-ss`/`-t` seek from the original source, see
    _extract_reencode_segment) -- so a bad cut here isn't just wasted
    duplicate footage the way a bad start-side cut is, it can leave the real
    start of an approved scene sitting in the *unblurred* copy segment,
    before the (correctly blurred, but now late) re-encode segment even
    begins. Both directions have to be checked -- see build_blurred_video's
    retry loop, which probes every copy segment's start AND end."""
    probe_result = await probe(ffprobe_bin, ts_path)
    duration = probe_result["format"].get("duration")
    if duration is None:
        return None
    return actual_start + float(duration)


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
    frame/keyframe-type issue). Same class of splice mismatch applies to any
    non-H.264 source, not just HEVC -- extended to cover the other codecs
    this ffmpeg build has an encoder for (vp9/av1/mpeg2/mpeg4, all realistic
    for older DVD rips or web-dl remuxes); libx264 remains the default for
    true H.264 sources and anything else unrecognized, unchanged from the
    original behavior."""
    codec_map = {
        "hevc": "libx265",
        "h265": "libx265",
        "vp9": "libvpx-vp9",
        "av1": "libsvtav1",
        "mpeg2video": "mpeg2video",
        "mpeg4": "mpeg4",
    }
    return codec_map.get(source_codec, "libx264")


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
    approved-scene window, not the whole file, so the absolute cost stays small.

    For HEVC sources, also probes the source's own SPS/PPS derived-state
    fields (via _probe_source_hevc_params) and matches them in the re-encode's
    own SPS/PPS (via _build_matching_x265_params) -- see that function's
    docstring for the real splice-corruption bug this avoids. A probe failure
    (non-HEVC source, unexpected ffmpeg output) falls back to today's
    unmatched re-encode with no other change in behavior."""
    codec = _reencode_video_codec(source_codec)
    x265_params = None
    if source_codec == "hevc":
        source_params = await _probe_source_hevc_params(ffmpeg_bin, video_path, segment.start)
        if source_params is not None:
            x265_params = _build_matching_x265_params(source_params)

    def build_cmd(tmp_out: Path) -> list[str]:
        cmd = [ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error"]
        if segment.start > 0:
            cmd += ["-ss", f"{segment.start:.3f}"]
        cmd += ["-i", str(video_path), "-t", f"{segment.duration:.3f}"]

        filter_str = build_blur_filter(
            list(segment.local_blur_intervals), input_label="0:v", output_label="vb", radius=blur_radius, power=blur_power
        )
        cmd += [
            "-filter_complex", filter_str, "-map", "[vb]",
            "-c:v", codec, "-crf", str(video_crf), "-preset", video_preset,
        ]
        if x265_params is not None:
            cmd += ["-x265-params", x265_params]
        return cmd + ["-an", "-sn", "-f", "matroska", str(tmp_out)]

    await _run_atomic(
        build_cmd, out_ts_path, _SEGMENT_TIMEOUT_SECONDS,
        f"ffmpeg segment re-encode failed for {video_path} [{segment.start:.2f}-{segment.end:.2f}]",
    )


async def _build_video_whole_file_reencode(
    ffmpeg_bin: str,
    video_path: Path,
    blur_intervals: list[MuteInterval],
    video_crf: int,
    total_duration: float,
    blur_radius: int,
    blur_power: int,
    source_codec: str,
    out_path: Path,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Fallback for when the split/copy/re-encode/concat pipeline (see
    build_blurred_video) fails -- decodes and re-encodes the ENTIRE file in one
    continuous ffmpeg session instead of stream-copying the untouched parts
    around a handful of re-encoded scenes, at the cost of a full re-encode
    instead of a partial one.

    Real motivation: confirmed directly, live, against a real file (28 Years
    Later) that the segment pipeline can produce two independently
    perfectly-decodable pieces (a stream-copied "before" segment, a
    re-encoded scene segment) whose *concatenation* still corrupts
    (`Could not find ref with POC ...`), despite this module's existing three
    real fixes for that same symptom class (MPEG-TS->MKV container, a
    timestamp-metadata bug, a RASL-picture check) all still being in place.
    This turned out to run deeper than a muxing/timestamp problem: a from-
    scratch PyAV mux with fully explicit, manually-computed, monotonicity-
    enforced packet timestamps (the technique real prior art -- smartcut,
    github.com/skeskinen/smartcut -- actually uses for this exact class of
    problem) reproduced the identical corruption, ruling out container/muxer/
    timestamp handling entirely regardless of implementation. A follow-up
    test re-encoding the scene with a simpler reference structure (bframes=0,
    no B-pyramid) changed the symptom -- the POC-lookup errors disappeared,
    replaced by pervasive `alignment_bit_equal_to_one=0` corruption on nearly
    every frame -- consistent with the decoder using the wrong active
    parameter set (SPS/PPS) across the splice, not a DPB/reference-buffer
    sizing issue either.

    Root cause since confirmed and mitigated upstream: see
    _build_matching_x265_params' docstring -- _extract_reencode_segment now
    probes each HEVC source's own SPS/PPS derived-state fields and matches
    them in the re-encoded segment, which a live test against this exact
    file/scene confirmed eliminates the corruption. This function remains the
    safety net regardless: a single continuous decode+encode session has no
    concat step at all, so it structurally can't hit this failure class (or
    any future one) no matter the cause -- still relevant for a probe
    failure, a file whose needs don't map cleanly onto the matching
    heuristic, or any other corruption class the fast path might hit.

    Deliberately uses _WHOLE_FILE_FALLBACK_PRESET, not whatever blur_video_preset
    the user configured for the normal segment-based re-encode -- see that
    constant's own comment for the real benchmark numbers behind this.

    Reuses build_blur_filter unchanged -- it already handles an arbitrary
    number of absolute-timestamp intervals (with its own >_MAX_TERMS_PER_STAGE
    batching), the only difference from the segment case is these timestamps
    are absolute against the whole file instead of segment-local."""
    if out_path.exists():
        return  # already produced by a prior interrupted attempt
    codec = _reencode_video_codec(source_codec)
    filter_str = build_blur_filter(
        blur_intervals, input_label="0:v", output_label="vb", radius=blur_radius, power=blur_power
    )
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    cmd = [
        ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error", "-progress", "pipe:1",
        "-i", str(video_path),
        "-filter_complex", filter_str, "-map", "[vb]",
        "-c:v", codec, "-crf", str(video_crf), "-preset", _WHOLE_FILE_FALLBACK_PRESET,
        "-an", "-sn", "-f", "matroska", str(tmp_out),
    ]
    code, err = await _run_with_progress(
        cmd, total_duration_seconds=total_duration, on_progress=on_progress,
        timeout=_WHOLE_FILE_REENCODE_TIMEOUT_SECONDS,
    )
    if code != 0:
        tmp_out.unlink(missing_ok=True)
        raise RemuxError(f"ffmpeg whole-file re-encode failed for {video_path}: {err.strip()[-2000:]}")
    tmp_out.replace(out_path)


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
    intermediate, only MP4/MOV/MKV.

    Despite all of the above, this function could still fail on some real
    files with that same "Could not find ref with POC ..." corruption at the
    copy-segment/re-encoded-segment splice -- confirmed directly against a
    real file (28 Years Later) with every fix in this docstring already in
    place. Root-caused as far as: a from-scratch PyAV mux with fully explicit,
    manually-computed, monotonicity-enforced PTS/DTS (bypassing this function
    and ffmpeg's concat demuxer entirely) reproduced the identical corruption,
    which ruled out timestamp/muxer handling -- including this function's own
    approach -- as the cause. A follow-up re-encode with a simplified
    reference structure (bframes=0) changed the symptom to pervasive
    `alignment_bit_equal_to_one=0` errors instead, consistent with the
    decoder picking up the wrong active SPS/PPS across the splice rather than
    a reference-buffer sizing problem.

    Root cause since confirmed: the stream-copied segments this function
    concatenates carry the SOURCE's own SPS/PPS; the re-encoded segment
    between them reuses the same parameter-set ID but with different
    derived-state fields (tier, level, DPB size, WPP, deblocking params) by
    default, which libavcodec's HEVC decoder doesn't handle correctly across
    the splice. _extract_reencode_segment now probes the source's own fields
    and matches them in the re-encode (see _build_matching_x265_params) to
    avoid triggering this at all for HEVC sources -- this function and its
    concat mechanism are unchanged, since the fix is applied upstream, before
    the segments this function receives are ever produced. This is a
    mitigation, not a patch to ffmpeg/libavcodec itself: a probe failure, or a
    file whose reorder/latency needs don't map cleanly onto the matching
    heuristic's plain --ref override, can still hit this corruption, which is
    exactly what verify_blurred_output's decode-integrity check and
    _build_video_whole_file_reencode's fallback remain in place for."""
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
    force_whole_file_video: bool = False,
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
    re-encode, and _blur_job_fingerprint for the staleness guard.

    force_whole_file_video skips that split/re-encode/concat pipeline entirely
    in favor of _build_video_whole_file_reencode -- apply_blur's fallback for
    when the concat-based approach produces output that fails
    verify_blurred_output (see that function's docstring for the real,
    directly-confirmed corruption case that motivated this)."""
    src_probe = await probe(ffprobe_bin, video_path)
    total_duration = float(src_probe["format"].get("duration", 0))
    audio_streams = _audio_streams(src_probe)
    if not audio_streams:
        raise RemuxError("No audio streams found in source file")

    # Scoped by job id (work_dir.name), not just video_path -- two concurrent
    # Apply jobs for the same title (e.g. a double-clicked button, or a scan's
    # auto-chain racing a manual click) would otherwise share this exact path
    # and one job's failure/cancel cleanup could delete it out from under the
    # other mid-mux.
    tmp_path = video_path.with_name(f".{video_path.stem}.spf-blur-tmp.{work_dir.name}{video_path.suffix}")

    work_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _blur_job_fingerprint(
        video_path, blur_intervals, mute_intervals, video_crf, video_preset, blur_radius, blur_power, audio_bitrate
    )
    _reset_stale_work_dir(work_dir, fingerprint)
    tmp_root = work_dir

    video_stream = next((s for s in src_probe["streams"] if s["codec_type"] == "video"), None)
    video_codec = (video_stream or {}).get("codec_name", "")

    try:
        if force_whole_file_video:
            # apply_blur's fallback path -- see _build_video_whole_file_reencode's
            # docstring for why this exists at all. Separate filename from the
            # segment pipeline's concatenated.mkv (not reused) so a stale copy
            # left behind by a failed first attempt in this same work_dir can
            # never be mistaken for this attempt's own output.
            concatenated_video_path = tmp_root / "concatenated_wholefile.mkv"
            if on_stage is not None:
                await on_stage("Blurring video (whole-file re-encode)")
            await _build_video_whole_file_reencode(
                ffmpeg_bin, video_path, blur_intervals, video_crf, total_duration, blur_radius, blur_power,
                video_codec, concatenated_video_path, on_progress=on_progress,
            )
        else:
            if on_stage is not None:
                await on_stage("Scanning keyframes")
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

                if len(segments) > 1:
                    # Validate against the segments actually just produced, not a
                    # separate probe against the source -- an earlier version of
                    # the RASL check below seeked independently into the source
                    # file per candidate and got inconsistent verdicts for the
                    # same real timestamp across repeated runs (the same
                    # stream-copy seek imprecision _split_copy_segments's own
                    # docstring already documents for this class of source).
                    # Reading an already-cut segment's own true start has no seek
                    # step to be imprecise about -- same reasoning
                    # _segment_actual_start_time uses for the codec-agnostic
                    # check below.
                    #
                    # Two independent checks share this one retry loop/bad-
                    # boundary-exclusion mechanism: the RASL check (HEVC-only,
                    # a specific corrupt-reference-frame byte pattern) and the
                    # boundary-drift check (every codec, the segment muxer's own
                    # keyframe-cut decision landing somewhere plan_video_segments
                    # didn't expect -- see _segment_actual_start_time's docstring
                    # for the real H.264 case that surfaced this: no RASL
                    # anywhere, but several cuts several-to-ten seconds off their
                    # plan, silently duplicating/omitting footage across the
                    # copy/re-encode splice until verify_blurred_output's
                    # whole-file duration check caught it after the fact).
                    for _attempt in range(20):
                        bad_boundary = None
                        bad_reason = ""
                        confirmed_keyframe = None
                        for i, seg in enumerate(segments):
                            if seg.reencode or seg.start >= seg.end:
                                continue
                            ts_path = tmp_root / f"copy_{i:04d}.mkv"
                            if seg.start > 0.0 and video_codec in ("hevc", "h265") and await _segment_file_starts_with_rasl(ffmpeg_bin, ts_path):
                                bad_boundary = seg.start
                                bad_reason = "has a RASL picture following it"
                                break
                            actual_start = await _segment_actual_start_time(ffprobe_bin, ts_path)
                            # Both boundaries this copy segment touches get checked, not just
                            # its start -- a bad cut on the *end* side (where the next
                            # re-encoded scene is supposed to pick up) is the more dangerous
                            # direction to miss: see _segment_actual_end_time's docstring for
                            # why that one can leave the real start of an approved scene
                            # sitting unblurred, rather than just duplicated.
                            if seg.start > 0.0 and actual_start is not None and abs(actual_start - seg.start) > _BOUNDARY_DRIFT_TOLERANCE_SECONDS:
                                bad_boundary = seg.start
                                bad_reason = f"actually cut at {actual_start:.3f}s"
                                confirmed_keyframe = actual_start
                                break
                            if seg.end < total_duration and actual_start is not None:
                                actual_end = await _segment_actual_end_time(ffprobe_bin, ts_path, actual_start)
                                if actual_end is not None and abs(actual_end - seg.end) > _BOUNDARY_DRIFT_TOLERANCE_SECONDS:
                                    bad_boundary = seg.end
                                    bad_reason = f"actually cut at {actual_end:.3f}s"
                                    confirmed_keyframe = actual_end
                                    break
                            if (
                                seg.end < total_duration
                                and video_codec in ("hevc", "h265")
                                and await _segment_file_ends_with_cra(ffmpeg_bin, ts_path)
                            ):
                                bad_boundary = seg.end
                                bad_reason = "ends on a CRA frame"
                                break
                        if bad_boundary is None:
                            break
                        logger.info("Cut point %.3fs for %s %s -- excluding and re-splitting", bad_boundary, video_path, bad_reason)
                        keyframe_timestamps = [t for t in keyframe_timestamps if abs(t - bad_boundary) > 1e-6]
                        if confirmed_keyframe is not None:
                            keyframe_timestamps.append(confirmed_keyframe)
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
                        # Unlike the old HEVC-only version of this loop, which proceeded
                        # anyway after exhausting attempts (a RASL miss is rare enough
                        # that this was an acceptable bet, and verify_blurred_output
                        # would still catch a real corruption via the stream-count/
                        # decode checks even if the duration happened to still line up),
                        # failing fast here instead: the boundary-drift case, generalized
                        # to every codec, is a duration mismatch by construction, which
                        # verify_blurred_output WILL catch regardless -- so "proceed
                        # anyway" here only ever bought a guaranteed-wasted re-encode
                        # pass (potentially the whole file's approved-scene set, tens of
                        # minutes) before failing at the exact same check at the very end
                        # instead. Raising immediately, before any of that expensive work
                        # starts, reaches the same safe outcome (nothing ever gets
                        # published) far faster.
                        raise RemuxError(
                            f"Could not find a reliable stream-copy cut point near {bad_boundary:.3f}s for {video_path} "
                            f"after 20 attempts ({bad_reason}) -- this source likely has an unusually sparse or "
                            f"irregular keyframe structure in that region"
                        )

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
    #
    # A slightly wider lookback than the bare minimum needed -- gives
    # ffmpeg's own accurate-seek preroll more room before the window of
    # interest, though see _BENIGN_DECODE_WARNING_RE below for why this
    # alone isn't what actually keeps this check honest.
    _WINDOW_LOOKBACK_SECONDS = 6.0
    for window in blurred_windows:
        check_start = max(0.0, window.start - _WINDOW_LOOKBACK_SECONDS)
        check_duration = min(19.0, (window.end - window.start) + _WINDOW_LOOKBACK_SECONDS + 2.0)
        code, _out, err = await _run(
            [
                ffmpeg_bin, "-v", "error",
                "-ss", f"{check_start:.3f}", "-i", str(tmp_path),
                "-t", f"{check_duration:.3f}", "-f", "null", "-",
            ],
            timeout=120,
        )
        real_errors = _filter_benign_decode_warnings(err)
        if code != 0 or real_errors:
            raise RemuxError(
                f"Blurred output failed decode check for {tmp_path} at window "
                f"[{window.start:.2f}-{window.end:.2f}]: {real_errors[-2000:]}"
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
    behind for the next attempt to pick up from.

    Tries the segment-based split/copy/re-encode/concat pipeline first (fast:
    only the approved scenes actually get re-encoded); if that fails --
    either build_blurred_video itself raising, or its output failing
    verify_blurred_output -- retries once with force_whole_file_video=True
    (see that function and _build_video_whole_file_reencode's docstrings for
    the real, directly-confirmed concat corruption that motivated this). A
    single continuous decode+encode session has no concat step to corrupt,
    so it's slower but structurally can't hit that failure class. If the
    fallback also fails, its error is what propagates -- the segment-based
    attempt's failure is logged but not re-raised, since the fallback
    represents the more useful, more recent information."""
    if not video_path.exists():
        raise RemuxError(f"Video file does not exist: {video_path}")
    if not blur_intervals:
        raise RemuxError("No approved scene windows to apply")

    src_probe = await probe(ffprobe_bin, video_path)

    async def _build_and_verify(*, force_whole_file_video: bool) -> Path:
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
            force_whole_file_video=force_whole_file_video,
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
        return tmp_path

    try:
        tmp_path = await _build_and_verify(force_whole_file_video=False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Segment-based blur failed for %s (%s) -- falling back to a whole-file re-encode", video_path, exc
        )
        tmp_path = await _build_and_verify(force_whole_file_video=True)

    if on_stage is not None:
        await on_stage("Publishing Vulgarr Edit")

    final_path = sibling_edit_path(video_path)
    tmp_path.replace(final_path)  # same-directory rename: atomic, no partial-file window

    # Full success -- the work_dir has served its purpose. A future Apply for
    # this same title (even if it needs to redo everything, e.g. a rejected
    # scene) gets a fresh job id and thus a fresh work_dir regardless.
    shutil.rmtree(work_dir, ignore_errors=True)

    return final_path
