"""Per-frame explicit-content classification via NudeNet, for scene detection.

NudeNet (not OpenNSFW2 -- see the scene-detection plan for why) is loaded lazily
and only inside functions, never at module import, mirroring
app.audio.forced_align's discipline exactly: every processing job that never
scans for scenes pays zero import/model-load cost for this being available.
"""

import asyncio
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


def extract_frame(ffmpeg_bin: str, video_path: Path, timestamp: float, out_path: Path) -> None:
    """Blocking -- callers on the event loop must offload via asyncio.to_thread.
    Shared by the scan pipeline (below) and the scene-review thumbnail endpoint
    (app/routers/scenes.py)."""
    proc = subprocess.run(
        [
            ffmpeg_bin, "-nostdin", "-loglevel", "error",
            "-ss", f"{timestamp:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "4", str(out_path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
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


def _classify_frame_blocking(ffmpeg_bin: str, video_path: Path, timestamp: float) -> float:
    detector = _get_detector()
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = Path(tmpdir) / "frame.jpg"
        extract_frame(ffmpeg_bin, video_path, timestamp, frame_path)
        detections = detector.detect(str(frame_path))
    return frame_confidence(detections)


async def scan_video_frames(
    video_path: Path,
    ffmpeg_bin: str,
    duration_seconds: float,
    frame_interval_seconds: float,
    on_progress=None,
) -> list[FrameScore]:
    """Sample video_path every frame_interval_seconds across its full duration,
    classify each sampled frame, and return one FrameScore per successfully
    classified frame. A frame that fails to extract/classify is skipped (logged,
    not fatal to the whole scan) -- same fall-back-gracefully philosophy as
    align_matches_for_cue's per-cue try/except."""
    timestamps = []
    t = 0.0
    while t < duration_seconds:
        timestamps.append(t)
        t += frame_interval_seconds

    scores: list[FrameScore] = []
    for i, ts in enumerate(timestamps):
        try:
            confidence = await asyncio.to_thread(_classify_frame_blocking, ffmpeg_bin, video_path, ts)
            scores.append(FrameScore(timestamp=ts, confidence=confidence))
        except Exception:
            logger.warning("Scene-scan frame classification failed at %.2fs in %s", ts, video_path, exc_info=True)
        if on_progress is not None:
            await on_progress(i + 1, len(timestamps))

    return scores
