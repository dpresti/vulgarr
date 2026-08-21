"""Optional precision-filter step for scene-detection auto-approval: ask a
Claude-vision-capable endpoint (any OpenAI-compatible chat completions API --
e.g. a self-hosted LiteLLM proxy in front of Anthropic's API, see
/docker/stack/litellm) whether a NudeNet candidate is real adult content,
to catch classifier false positives (ordinary, non-sexualized content like
beach/pool swimwear) before app.scenes.pipeline.scan_for_scenes auto-approves
it. Also catches the inverse case: sexualized/provocative content NudeNet's
narrow exposed-body-part classes can miss entirely (thongs, lingerie, and
similar, even without literal nudity) -- see _PROMPT below for the exact
criteria, which came directly from real false-positive/false-negative
examples found while using this feature, not written in the abstract.

Entirely optional -- see claude_vision_verify_enabled in DEFAULT_SETTINGS,
off by default so self-hosters never need an API key or an extra service
unless they opt in. Every failure mode here (unreachable endpoint, bad
response, extraction failure) must resolve to "not confirmed", never
"confirmed" -- an unreachable/misconfigured endpoint should degrade to the
existing manual-review path, not cause a silent auto-approval. An explicit
"confirmed=False" verdict is different from that, though -- see
app.scenes.pipeline.scan_for_scenes, which treats a real Claude "no" as
grounds to reject the candidate outright, not just leave it unconfirmed.
"""

import asyncio
import base64
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.vision.classifier import extract_frame
from app.vision.scene_cluster import FrameScore

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are one automated step in a personal, self-hosted content filter that "
    "blurs nudity and sexual content out of a user's own legally-owned video "
    "files. You will see one or more frames sampled across a short clip that a "
    "local classifier already flagged as a candidate. Reply with exactly three "
    "lines:\n"
    "Line 1: YES or NO for whether this clip should be blurred. Answer YES for "
    "real nudity or sexually explicit content, and also for clearly sexualized "
    "or provocative content even without full nudity -- including thongs, "
    "lingerie, and similarly minimal or revealing clothing shown in a sexual "
    "context (e.g. posing on a bed, seductive dancing). Answer NO only for "
    "ordinary, non-sexualized content with no such intent, like normal "
    "swimwear in a beach or pool setting, or otherwise fully covered clothing.\n"
    "Line 2: YES or NO for whether this specifically depicts a sex scene or "
    "sexual activity (as opposed to nudity alone, e.g. undressing, bathing, or "
    "posing) -- this also controls whether the audio gets muted, not just the "
    "video blurred, so only answer YES here for actual sexual activity.\n"
    "Line 3: a short one-sentence reason."
)

# Capped low deliberately -- this is a yes/no classification call, not a task
# that benefits from more samples, and every extra frame is extra image-token
# cost multiplied across every candidate scene in every scan. 2 rather than
# more is workable specifically because callers with real classifier scores
# (see top_score_timestamps below) target the frames actually worth looking
# at, rather than blindly spending frames on a window's quiet stretches.
_MAX_FRAMES = 2

# Claude tokenizes images as ceil(w/28) * ceil(h/28) 28px patches -- a
# 1920x1080 frame costs ~1560 tokens even after Anthropic's own auto-
# downscale to their standard tier's 1568px/1568-token ceiling. This task is
# a coarse yes/no exposure classification, not fine detail or text reading,
# so there's no accuracy reason to pay for anywhere near that resolution.
# 512px long edge costs ~200 tokens (roughly 8x cheaper per image) while
# staying comfortably above Anthropic's own documented "under 200px starts
# to see accuracy loss" floor.
_MAX_LONG_EDGE = 512


@dataclass(frozen=True)
class VerifyResult:
    confirmed: bool
    # True only when Claude specifically identified sexual activity (not just
    # nudity/exposure) -- see app.scenes.pipeline.scan_for_scenes, which wires
    # this straight into DetectedScene.mute_audio. Always False when confirmed
    # is False (never mute audio for something that isn't even being blurred).
    mute_audio: bool
    reason: str


def frame_timestamps(start: float, end: float, max_frames: int = _MAX_FRAMES) -> list[float]:
    """Evenly-spaced sample points across [start, end], capped at max_frames --
    pure function, testable without a real classifier/network call."""
    if max_frames <= 1 or end <= start:
        return [(start + end) / 2]
    step = (end - start) / (max_frames - 1)
    return [start + i * step for i in range(max_frames)]


def top_score_timestamps(scores: list[FrameScore], max_frames: int = _MAX_FRAMES) -> list[float]:
    """The max_frames highest-confidence sample timestamps from a candidate's
    dense re-scan (see app.vision.classifier.scan_window_frames), sorted back
    into chronological order. Pure function, testable without a real
    classifier/network call.

    Targets the frames the local classifier itself found most convincing,
    rather than frame_timestamps' blind even-spacing across the whole window
    -- both cheaper (fewer frames needed for the same confidence) and more
    accurate (a scene's real exposure is rarely spread evenly across its
    whole padded duration; this points Claude straight at the moment NudeNet
    actually flagged, not wherever an even split happened to land)."""
    if not scores:
        return []
    ranked = sorted(scores, key=lambda s: s.confidence, reverse=True)[:max_frames]
    return sorted(s.timestamp for s in ranked)


def _leading_yes_no(line: str) -> bool | None:
    """True/False for a line whose first *word* is exactly YES/NO (trailing
    punctuation tolerated), None if it doesn't start that way. A plain
    line.upper().startswith("NO") check would false-positive on a reason
    line like "Nothing explicit here." -- this checks the first whitespace-
    delimited token, not just a string prefix."""
    words = line.split()
    if not words:
        return None
    word = words[0].upper().rstrip(".,!:;")
    if word == "YES":
        return True
    if word == "NO":
        return False
    return None


def parse_verdict(text: str | None) -> VerifyResult | None:
    """Pure parser for the model's response -- expects YES/NO on the first
    non-blank line (the blur verdict), then an optional second YES/NO line
    (the sex-scene/mute-audio signal), then an optional one-sentence reason.
    Returns None if the first line doesn't parse as either, letting the caller
    fail safe (treat as unconfirmed) rather than guess at intent.

    The second line is read leniently -- if it's not itself a YES/NO (a
    2-line response, or the model drifting from the requested format),
    mute_audio just defaults to False and that line is treated as the reason
    instead. A malformed mute signal should never block the blur verdict
    itself from being usable."""
    if not text:
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None

    leading = _leading_yes_no(lines[0])
    if leading is None:
        return None
    confirmed = leading

    remaining = lines[1:]
    mute_audio = False
    if remaining:
        leading_second = _leading_yes_no(remaining[0])
        if leading_second is not None:
            mute_audio = confirmed and leading_second
            remaining = remaining[1:]
    reason = remaining[0] if remaining else ""
    return VerifyResult(confirmed=confirmed, mute_audio=mute_audio, reason=reason)


async def verify_candidate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    ffmpeg_bin: str,
    video_path: Path,
    start: float,
    end: float,
    sample_timestamps: list[float] | None = None,
    timeout: float = 60.0,
) -> VerifyResult | None:
    """Extracts a handful of frames across the candidate's window and asks the
    configured vision endpoint for a yes/no verdict. Returns None on *any*
    failure -- callers must treat None the same as "not confirmed".

    sample_timestamps, when given (see top_score_timestamps), overrides the
    default even-spacing with specific timestamps a caller already knows are
    worth looking at -- callers that have the candidate's real dense-re-scan
    scores on hand should pass those instead of leaving this as None."""
    if not base_url:
        return None

    timestamps = sample_timestamps if sample_timestamps else frame_timestamps(start, end)
    images: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        for i, ts in enumerate(timestamps):
            frame_path = out_dir / f"frame_{i}.jpg"
            try:
                await asyncio.to_thread(
                    extract_frame, ffmpeg_bin, video_path, max(0.0, ts), frame_path, max_long_edge=_MAX_LONG_EDGE
                )
                images.append(base64.b64encode(frame_path.read_bytes()).decode("ascii"))
            except Exception:
                logger.warning(
                    "Claude-vision-verify frame extraction failed at %.2fs in %s", ts, video_path, exc_info=True
                )

    if not images:
        return None

    content: list[dict] = [{"type": "text", "text": _PROMPT}]
    for b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # temperature=0 for determinism -- confirmed via a real A/B test that the
    # default sampling temperature genuinely produces different verdicts on
    # the *exact same* frames run twice (once wrongly calling clear, explicit
    # nudity "underwear/sleepwear"). This is a classification task, not a
    # creative one; there's no reason to accept that variance.
    payload = {
        "model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 100, "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except Exception:
        logger.warning(
            "Claude-vision-verify request failed for %s (%.2f-%.2fs)", video_path, start, end, exc_info=True
        )
        return None

    return parse_verdict(text)
