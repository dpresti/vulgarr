"""Per-frame explicit-content classification via NudeNet, for scene detection.

NudeNet (not OpenNSFW2 -- see the scene-detection plan for why) is loaded lazily
and only inside functions, never at module import, mirroring
app.audio.forced_align's discipline exactly: every processing job that never
scans for scenes pays zero import/model-load cost for this being available.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from app.vision.scene_cluster import FrameScore

logger = logging.getLogger(__name__)

# NudeNet's 18-class taxonomy includes body parts that are just anatomically
# present in an ordinary frame (faces, feet, armpits, belly) alongside the
# classes that actually indicate nudity/exposure -- only the latter should drive
# a scene-detection score. Confirmed against a real spike run before committing
# to this set (see the scene-detection plan's Phase 0 findings).
_EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

_bundle: dict = {}


def _get_detector():
    if "detector" not in _bundle:
        from nudenet import NudeDetector

        _bundle["detector"] = NudeDetector()
    return _bundle["detector"]


def extract_frame(
    ffmpeg_bin: str, video_path: Path, timestamp: float, out_path: Path, max_long_edge: int | None = None
) -> None:
    """Blocking -- callers on the event loop must offload via asyncio.to_thread.
    Shared by the scan pipeline (below) and the scene-review thumbnail endpoint
    (app/routers/scenes.py). max_long_edge downscales (never upscales) before
    encoding -- used by app.vision.claude_verify to shrink the image-token cost
    of a Claude Vision call (Claude tokenizes images as ceil(w/28)*ceil(h/28)
    patches, so a 1920x1080 frame costs ~1560 tokens even after Anthropic's own
    auto-downscale; sending it pre-shrunk to ~512px costs closer to ~200).
    Left unset (None) for the thumbnail/review-UI callers, which want full
    quality for a human to actually look at."""
    cmd = [ffmpeg_bin, "-nostdin", "-loglevel", "error", "-ss", f"{timestamp:.3f}", "-i", str(video_path)]
    if max_long_edge is not None:
        cmd += ["-vf", f"scale='min({max_long_edge},iw)':'min({max_long_edge},ih)':force_original_aspect_ratio=decrease"]
    cmd += ["-frames:v", "1", "-q:v", "4", str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg frame extraction failed at {timestamp:.2f}s in {video_path}")


def frame_confidence(detections: list[dict]) -> float:
    """Pure mapping from NudeNet's per-class detection list (each a dict with a
    "class" and "score") to a single confidence score -- the max score among
    classes that actually indicate exposure, 0.0 if none matched or the frame had
    no detections at all. Kept as a standalone function so it's testable without
    a real model."""
    scores = [d["score"] for d in detections if d.get("class") in _EXPLICIT_CLASSES]
    return max(scores, default=0.0)


# How much of a scan's total wall time the extraction phase accounts for, used
# only to blend a smooth combined progress bar across the two phases (not for
# scheduling). Measured directly on real homelab hardware against real
# NFS-mounted media: extracting 40 frames via one continuous decode took ~5.2s,
# classifying those same 40 already-extracted frames at concurrency=8 took
# ~3.0s -- extraction is the larger share of the two.
_EXTRACT_PROGRESS_WEIGHT = 0.6


async def _extract_frames_continuous(
    ffmpeg_bin: str,
    video_path: Path,
    frame_interval_seconds: float,
    out_dir: Path,
    duration_seconds: float,
    on_extract_progress=None,
    start_offset: float = 0.0,
    clip_duration: float | None = None,
) -> None:
    """One continuous sequential decode pass emitting a frame every
    frame_interval_seconds, replacing what used to be one independent
    `ffmpeg -ss <t>` re-seek per sampled timestamp.

    Confirmed via direct A/B benchmark against real homelab hardware and the
    same real (NFS-mounted) media file, at a cold/never-touched offset so
    neither side benefited from page-cache warmth: 40 individually re-seeked
    frames -- even run at concurrency=8 -- took ~26.6s; one continuous decode
    over the same 40s span took ~5.2s. Repeatedly re-seeking a multi-GB
    container is genuinely expensive here; one sequential read is not.
    `format=yuvj420p` works around an mjpeg-encoder failure ffmpeg throws on
    some full-range-YUV source segments otherwise ("Non full-range YUV is
    non-standard... ff_frame_thread_encoder_init failed"), found while
    benchmarking against a real mid-episode offset.

    start_offset/clip_duration scope the decode to a short window instead of
    the whole file -- used by scan_window_frames for the dense per-candidate
    verification pass, where re-decoding the entire movie just to resample a
    single 20-40s scene would defeat the point of it being cheap. A single
    `-ss` before `-i` still costs one seek either way, but that's one seek per
    candidate (a handful per scan), not one per sampled frame.
    """
    cmd = [ffmpeg_bin, "-nostdin", "-loglevel", "error", "-progress", "pipe:1"]
    if start_offset > 0:
        cmd += ["-ss", f"{start_offset:.3f}"]
    cmd += ["-i", str(video_path)]
    if clip_duration is not None:
        cmd += ["-t", f"{clip_duration:.3f}"]
    cmd += [
        "-vf", f"fps=1/{frame_interval_seconds},format=yuvj420p",
        "-q:v", "4",
        str(out_dir / "frame_%08d.jpg"),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stderr_chunks: list[bytes] = []

    async def read_stderr() -> None:
        assert proc.stderr is not None
        async for chunk in proc.stderr:
            stderr_chunks.append(chunk)

    async def read_progress() -> None:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    current = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
                if on_extract_progress is not None and duration_seconds > 0:
                    await on_extract_progress(max(0.0, min(1.0, current / duration_seconds)))

    try:
        await asyncio.gather(read_stderr(), read_progress(), proc.wait())
    except asyncio.CancelledError:
        # Same reasoning as remux.py's _run_with_progress -- cancellation must
        # kill the actual ffmpeg process, not just stop awaiting it.
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg continuous frame extraction failed for {video_path}: "
            f"{b''.join(stderr_chunks).decode(errors='replace')[:500]}"
        )


def _load_resume_scores(path: Path) -> list[FrameScore]:
    """One JSON object per line ({"t": timestamp, "c": confidence}), tolerating
    a corrupt/truncated trailing line -- a process killed mid-write (e.g. this
    app's own container being rebuilt mid-scan) can leave one partial line at
    the end of an append-only file; skip it rather than fail the whole load."""
    if not path.exists():
        return []
    scores: list[FrameScore] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            scores.append(FrameScore(timestamp=float(obj["t"]), confidence=float(obj["c"])))
        except Exception:
            continue
    return scores


def _resumable_prefix(
    scores: list[FrameScore], start_offset: float, frame_interval_seconds: float
) -> tuple[float, list[FrameScore]]:
    """How many seconds of *contiguous* coverage, starting at start_offset,
    the given (possibly out-of-order, possibly gappy) scores actually
    represent, and the subset that falls within that trusted run -- pure
    function, testable without any real scan.

    Frame classification runs at concurrency > 1 (see _scan_frames), so
    completions -- and therefore what gets checkpointed to a resume file --
    can land out of order. If a prior attempt was interrupted right after a
    later-timestamped frame happened to finish first, trusting a raw
    max(timestamp) would silently skip re-checking earlier frames that were
    still in flight and never got written at all -- a real, silent detection
    gap, not just wasted work (exactly the kind of bug this whole feature has
    spent a lot of effort avoiding elsewhere: a missed frame near a scene
    boundary is indistinguishable from the classifier just not seeing
    anything there). This walks the sorted timestamps from start_offset and
    stops at the first real gap (more than one sample interval, with a little
    float-rounding slack), so a resume only ever trusts a genuinely unbroken
    run from the very start and re-does everything from the first hole
    onward -- always safe (worst case, a little redundant reclassification
    right at the resume boundary), never silently lossy."""
    if not scores:
        return 0.0, []
    ordered = sorted((s for s in scores if s.timestamp >= start_offset - 1e-6), key=lambda s: s.timestamp)
    if not ordered:
        return 0.0, []
    tolerance = frame_interval_seconds * 1.5
    trusted: list[FrameScore] = []
    expected = start_offset
    for s in ordered:
        if s.timestamp > expected + tolerance:
            break
        trusted.append(s)
        expected = max(expected, s.timestamp)
    if not trusted:
        return 0.0, []
    return (expected - start_offset) + frame_interval_seconds, trusted


# Bounds how much extraction work an interruption can ever lose on the
# resumable whole-file path (see _scan_frames): extraction runs as one
# continuous ffmpeg decode pass with no progress checkpointed until it
# finishes and classification of its frames begins, so a scan done as a
# single pass over a multi-hour file has no real resumability at all -- a
# restart mid-extraction (confirmed live on a real 2h11m 4K/HEVC title: still
# zero checkpointed frames after several minutes, all of it inside one
# extraction pass covering the *entire remaining* duration) would still lose
# everything back to the last successfully *classified* frame, which could be
# the whole file. Chunking bounds the loss to at most one chunk's worth
# regardless of total file length. 5 minutes is a lot more decode work than
# one seek's cost is worth worrying about -- this codebase's own existing
# per-candidate windowed scan (scan_window_frames) already accepts one `-ss`
# seek as cheap relative to a handful of seconds of decode, and 5 minutes is
# far more decode than that per seek paid for here.
_SCAN_CHUNK_SECONDS = 300.0


async def _scan_frames(
    video_path: Path,
    ffmpeg_bin: str,
    frame_interval_seconds: float,
    scan_duration_seconds: float,
    on_progress=None,
    concurrency: int = 1,
    start_offset: float = 0.0,
    clip_duration: float | None = None,
    resume_scores_path: Path | None = None,
) -> list[FrameScore]:
    """Shared implementation behind scan_video_frames (whole file) and
    scan_window_frames (one short padded clip) -- extract via one continuous
    decode pass per chunk, classify each sampled frame, and return one
    FrameScore per successfully classified frame, timestamped relative to the
    real file (not the clip) via start_offset. A frame that fails to classify
    is skipped (logged, not fatal) -- same fall-back-gracefully philosophy as
    align_matches_for_cue's per-cue try/except. If extraction of a chunk fails
    outright, returns whatever was already classified (and, for the resumable
    path, already checkpointed) rather than raising -- callers already treat
    a short/empty result as "found less," and a truly broken/corrupt source
    shouldn't crash the whole job.

    concurrency > 1 runs multiple already-extracted frames' classification in
    flight at once via a bounded semaphore -- each is an independent
    onnxruntime call with no shared mutable state (the detector's a single
    loaded session, safe for concurrent .run() calls), so this is pure
    parallelism with no effect on which candidates come out the other end, only
    how long it takes to get there.

    resume_scores_path, when given (only scan_video_frames' whole-file caller
    does -- scan_window_frames' per-candidate passes are already cheap enough
    not to need this), makes the scan resumable across a process restart: the
    remaining span is processed in _SCAN_CHUNK_SECONDS chunks (extract, then
    classify+checkpoint that chunk, before starting the next), each
    classified frame appended to the file as an independent JSON line as soon
    as it completes, and a prior run's *trusted contiguous* checkpoint (see
    _resumable_prefix -- concurrent completion means the raw file can have
    out-of-order or gappy entries, which this does not naively trust) is
    loaded and skipped past on the next call. Without chunking, a scan
    interrupted mid-extraction would have nothing checkpointed at all yet for
    however long that one pass over the *entire* remaining duration takes --
    confirmed live on a real multi-hour file, see _SCAN_CHUNK_SECONDS."""
    expected_total = max(1, round(scan_duration_seconds / frame_interval_seconds))

    resumed_scores: list[FrameScore] = []
    resume_seconds = 0.0
    if resume_scores_path is not None:
        raw_scores = _load_resume_scores(resume_scores_path)
        resume_seconds, resumed_scores = _resumable_prefix(raw_scores, start_offset, frame_interval_seconds)
        if resumed_scores:
            logger.info(
                "Resuming scene scan for %s: %d frame(s) already checkpointed (%.1fs covered)",
                video_path, len(resumed_scores), resume_seconds,
            )
        # Rewrite the checkpoint file to hold exactly the trusted prefix --
        # drops any untrusted/gappy tail from an interrupted attempt (see
        # _resumable_prefix) so newly (re)classified frames append cleanly
        # after it, instead of piling up alongside stale/ambiguous entries.
        if raw_scores:
            resume_scores_path.write_text(
                "".join(json.dumps({"t": s.timestamp, "c": s.confidence}) + "\n" for s in resumed_scores)
            )

    effective_start_offset = start_offset + resume_seconds
    full_span = clip_duration if clip_duration is not None else scan_duration_seconds
    remaining_duration = full_span - resume_seconds
    if remaining_duration <= 1e-6:
        return resumed_scores

    already_done_units = round(resume_seconds / frame_interval_seconds)
    remaining_units = expected_total - already_done_units
    extract_units = round(remaining_units * _EXTRACT_PROGRESS_WEIGHT)
    classify_units = max(1, remaining_units - extract_units)
    total_frames_remaining = max(1, round(remaining_duration / frame_interval_seconds))

    # Only the resumable whole-file path chunks -- scan_window_frames (no
    # resume_scores_path) keeps its original single-pass behavior exactly,
    # since its clips are already short enough not to need this.
    chunk_size = _SCAN_CHUNK_SECONDS if resume_scores_path is not None else remaining_duration

    detector = _get_detector()
    scores: list[FrameScore] = list(resumed_scores)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    # Callers' on_progress (see app.queue.scene_worker.on_scan_progress) commits
    # against a single AsyncSession shared across the whole job -- SQLAlchemy's
    # async sessions reject concurrent operations outright ("This session is
    # provisioning a new connection..."). classify_one runs concurrently now
    # (that's the whole point), so multiple completions can land in the same
    # event-loop tick; this lock serializes the on_progress call and the
    # resume-checkpoint file append together, not the actual extract+
    # classify work, so it doesn't undo the parallelism.
    progress_lock = asyncio.Lock()
    completed_overall = 0
    # Seconds of remaining_duration whose extraction has *fully* completed,
    # across all prior chunks -- updated only after a chunk's extraction
    # finishes, so classify_one's progress contribution (below) reflects how
    # much extraction is actually done so far, not the whole scan's total
    # extraction budget the moment any classification starts at all (a real
    # bug caught live: reusing the single-pass formula's `+ extract_units`
    # unconditionally made progress jump straight to ~60% as soon as the
    # first chunk's classification began, since extract_units there meant
    # "all extraction, ever," not "this chunk's share").
    extraction_seconds_done = 0.0

    chunk_start = 0.0
    while chunk_start < remaining_duration - 1e-6:
        this_chunk_duration = min(chunk_size, remaining_duration - chunk_start)
        chunk_abs_start = effective_start_offset + chunk_start
        chunk_prior_extraction_seconds = extraction_seconds_done  # snapshot before this chunk's own extraction

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            async def on_extract_progress(fraction: float) -> None:
                if on_progress is not None:
                    seconds_done = chunk_prior_extraction_seconds + fraction * this_chunk_duration
                    frac_of_remaining = seconds_done / remaining_duration if remaining_duration else 0.0
                    done = already_done_units + round(frac_of_remaining * extract_units)
                    await on_progress(min(done, expected_total), expected_total)

            try:
                await _extract_frames_continuous(
                    ffmpeg_bin, video_path, frame_interval_seconds, out_dir, this_chunk_duration,
                    on_extract_progress=on_extract_progress, start_offset=chunk_abs_start,
                    clip_duration=this_chunk_duration,
                )
            except Exception:
                logger.warning(
                    "Continuous frame extraction failed for %s at +%.1fs -- scan yields what's already checkpointed",
                    video_path, chunk_abs_start, exc_info=True,
                )
                return scores

            extraction_seconds_done += this_chunk_duration
            extract_frac_so_far = extraction_seconds_done / remaining_duration if remaining_duration else 0.0
            extract_done_units_so_far = round(extract_frac_so_far * extract_units)

            frame_files = sorted(out_dir.glob("frame_*.jpg"))

            async def classify_one(index: int, frame_path: Path) -> None:
                nonlocal completed_overall
                # fps=1/interval numbers frames from 1 -- frame N was sampled at
                # roughly (N-1)*interval seconds into this chunk, plus
                # chunk_abs_start to map back onto the real file's timeline.
                ts = chunk_abs_start + index * frame_interval_seconds
                score: FrameScore | None = None
                async with semaphore:
                    try:
                        detections = await asyncio.to_thread(detector.detect, str(frame_path))
                        score = FrameScore(timestamp=ts, confidence=frame_confidence(detections))
                        scores.append(score)
                    except Exception:
                        logger.warning(
                            "Scene-scan frame classification failed at %.2fs in %s", ts, video_path, exc_info=True
                        )
                    completed_overall += 1
                    async with progress_lock:
                        if resume_scores_path is not None and score is not None:
                            with resume_scores_path.open("a") as fh:
                                fh.write(json.dumps({"t": score.timestamp, "c": score.confidence}) + "\n")
                        if on_progress is not None:
                            done = already_done_units + extract_done_units_so_far + round(
                                completed_overall / total_frames_remaining * classify_units
                            )
                            await on_progress(min(done, expected_total), expected_total)

            if frame_files:
                await asyncio.gather(*(classify_one(i, f) for i, f in enumerate(frame_files)))

        chunk_start += this_chunk_duration

    return scores


async def scan_video_frames(
    video_path: Path,
    ffmpeg_bin: str,
    duration_seconds: float,
    frame_interval_seconds: float,
    on_progress=None,
    concurrency: int = 1,
    resume_scores_path: Path | None = None,
) -> list[FrameScore]:
    """Sample the whole file every frame_interval_seconds. See _scan_frames for
    the shared implementation. resume_scores_path, when given, makes this
    resumable across a process restart -- see _scan_frames' own docstring."""
    return await _scan_frames(
        video_path, ffmpeg_bin, frame_interval_seconds, duration_seconds,
        on_progress=on_progress, concurrency=concurrency, resume_scores_path=resume_scores_path,
    )


async def scan_window_frames(
    video_path: Path,
    ffmpeg_bin: str,
    window_start: float,
    window_end: float,
    pad_seconds: float,
    frame_interval_seconds: float,
    concurrency: int = 1,
) -> list[FrameScore]:
    """Densely re-sample just one candidate scene's padded window -- a stronger
    per-scene confidence signal than the initial full-file scan's coarser
    sampling gives (see verified_fraction in scene_cluster.py), and cheap
    because it's a single short clip instead of the whole movie. Reuses the
    same continuous-decode approach as scan_video_frames, just scoped via
    start_offset/clip_duration to a few seconds of source."""
    clip_start = max(0.0, window_start - pad_seconds)
    clip_duration = (window_end - window_start) + 2 * pad_seconds
    return await _scan_frames(
        video_path, ffmpeg_bin, frame_interval_seconds, clip_duration,
        concurrency=concurrency, start_offset=clip_start, clip_duration=clip_duration,
    )
