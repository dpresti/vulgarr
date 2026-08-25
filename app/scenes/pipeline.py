"""Ties video frame classification -> clustering -> DetectedScene persistence
together for one title's scene-detection scan. Mirrors app.processing's role for
the subtitle/mute pipeline, but is a wholly independent feature -- see the
scene-detection plan for why this doesn't share Title.status/ProcessingJob."""

import datetime
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audio.mute import MuteInterval
from app.common.intervals import merge_intervals
from app.config import settings as app_settings
from app.db.models import DetectedScene, Title
from app.db.session import get_setting
from app.domain import SceneReviewStatus
from app.mux.remux import ProgressCallback, StageCallback, probe
from app.mux.scene_blur import apply_blur
from app.vision.claude_verify import top_score_timestamps, verify_candidate
from app.vision.classifier import scan_video_frames, scan_window_frames
from app.vision.scene_cluster import (
    boundary_touches_window_edge,
    cluster_scenes,
    refine_scene_boundary,
    verified_fraction,
)

logger = logging.getLogger(__name__)

ScanProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True)
class ScanOutcome:
    candidate_count: int


@dataclass(frozen=True)
class BlurOutcome:
    output_path: Path
    scene_count: int


def _scan_fingerprint(video_path: Path, frame_interval_seconds: float, duration: float) -> str:
    """Guards a scan's resume checkpoint the same way _blur_job_fingerprint
    guards the blur pipeline's work_dir: frame timestamps in the checkpoint
    are computed as index*frame_interval_seconds, so if that setting (or the
    source file itself) changed between an interrupted attempt and its
    resume, blindly trusting the old checkpoint's timestamps would silently
    corrupt them rather than just costing a little redundant work -- treated
    as fully stale and wiped instead."""
    payload = repr((str(video_path), frame_interval_seconds, round(duration, 1)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def _refine_candidate_boundary(
    video_path: Path,
    candidate_start: float,
    candidate_end: float,
    verify_pad: float,
    verify_max_pad: float,
    verify_interval: float,
    confidence_threshold: float,
    frame_concurrency: int,
) -> tuple[float, float, float | None, list]:
    """Dense re-scan of a candidate's padded window (app.vision.classifier.
    scan_window_frames), doubling the pad outward -- up to verify_max_pad -- and
    re-scanning whenever the refined boundary (app.vision.scene_cluster.
    refine_scene_boundary) lands right at the edge of what was actually searched.

    Exists because a boundary at the search window's own wall is indistinguishable
    from "the scene genuinely ends here" only by coincidence (see
    boundary_touches_window_edge's docstring) -- without this, a scene whose real
    extent runs past the fixed verify_pad is silently clipped there every time,
    which is exactly the "blur starts after nudity is already on screen / stops
    before the scene ends" failure real usage reported (confirmed independently by
    this app's own benchmark harness: mean boundary error of ~31s across a real
    25-episode GoT sample, far more than a single fixed 5s pad could ever explain).

    Returns (start, end, verified_fraction, last successful pass's window_scores).
    verified_fraction and window_scores come from the *widest* pass that actually
    ran, not necessarily the one whose boundary was kept -- there isn't a case
    where a later pass finds a narrower real boundary than an earlier one already
    did, since each pass only ever searches a superset of the previous pass's
    window.

    Falls back to the original coarse candidate boundary, unchanged, if the very
    first pass fails outright or finds nothing above threshold -- same
    fail-gracefully contract callers had before this function existed. A failure
    on a *later* (expanded) pass instead keeps whatever the last successful pass
    already found, rather than discarding it -- an expansion attempt failing
    shouldn't cost the candidate a perfectly good narrower result it already had.
    """
    start, end, frac = candidate_start, candidate_end, None
    window_scores: list = []
    pad = verify_pad
    while True:
        try:
            new_window_scores = await scan_window_frames(
                video_path,
                app_settings.ffmpeg_bin,
                window_start=candidate_start,
                window_end=candidate_end,
                pad_seconds=pad,
                frame_interval_seconds=verify_interval,
                concurrency=frame_concurrency,
            )
        except Exception:
            logger.warning(
                "Scene verification pass failed for %s at %.1f-%.1fs (pad=%.1fs)",
                video_path, candidate_start, candidate_end, pad, exc_info=True,
            )
            break
        window_scores = new_window_scores
        frac = verified_fraction(window_scores, confidence_threshold)
        boundary = refine_scene_boundary(window_scores, confidence_threshold)
        if boundary is None:
            break
        start, end = boundary
        window_start = max(0.0, candidate_start - pad)
        window_end = candidate_end + pad
        touches_start, touches_end = boundary_touches_window_edge(
            boundary, window_start, window_end, tolerance=verify_interval * 1.5
        )
        if (not touches_start and not touches_end) or pad >= verify_max_pad:
            break
        pad = min(pad * 2, verify_max_pad)
    return start, end, frac, window_scores


async def scan_for_scenes(
    session: AsyncSession,
    title: Title,
    job_id: int,
    on_progress: ScanProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> ScanOutcome:
    """Scan title's video for candidate explicit-content scenes.

    Replaces any prior pending/rejected candidates from an earlier scan --
    approved or already-applied ones represent a human decision (or a real edit
    already baked into the sibling file) and must never be silently discarded by
    a re-scan. A candidate that happens to overlap an already-approved scene on
    re-scan is not deduplicated here -- left for the review UI (Phase 2) to
    surface sensibly, not a data-integrity concern at this layer.

    job_id keys the resumable checkpoint file (app_settings.data_dir /
    "scan_work" / str(job_id) / "scores.jsonl") the main frame scan uses --
    see app.vision.classifier._scan_frames. Same reasoning as
    apply_scene_blur's work_dir: scene_worker's startup reconciliation
    redispatches an interrupted job under this same id, so a resumed attempt
    naturally finds and continues from its own prior checkpoint; a genuinely
    new scan (new job_id) always starts clean.

    Each candidate the main scan finds then gets a second, much denser re-scan
    of just its own padded window (scan_window_frames) -- cheap, since it's a
    handful of short clips rather than the whole movie, and produces a
    verified_fraction (see scene_cluster.verified_fraction) that's a more
    robust per-scene signal than the main scan's single peak_confidence value.
    Powers the review list's one-click "Approve high-confidence" bulk action.
    """
    auto_process = bool(await get_setting(session, "scene_auto_process"))
    confidence_threshold = float(await get_setting(session, "scene_confidence_threshold"))
    frame_interval = float(await get_setting(session, "scene_frame_interval_seconds"))
    min_duration = float(await get_setting(session, "scene_min_duration_seconds"))
    merge_gap = float(await get_setting(session, "scene_merge_gap_seconds"))
    frame_concurrency = int(await get_setting(session, "scene_frame_classify_concurrency"))
    verify_pad = float(await get_setting(session, "scene_verify_pad_seconds"))
    verify_interval = float(await get_setting(session, "scene_verify_frame_interval_seconds"))
    verify_max_pad = float(await get_setting(session, "scene_verify_max_pad_seconds"))
    high_confidence_override = float(await get_setting(session, "scene_high_confidence_single_frame_threshold"))

    video_path = Path(title.video_path)
    src_probe = await probe(app_settings.ffprobe_bin, video_path)
    duration = float(src_probe["format"].get("duration", 0))

    scan_work_dir = app_settings.data_dir / "scan_work" / str(job_id)
    scan_work_dir.mkdir(parents=True, exist_ok=True)
    resume_scores_path = scan_work_dir / "scores.jsonl"
    fingerprint_path = scan_work_dir / "fingerprint.txt"
    fingerprint = _scan_fingerprint(video_path, frame_interval, duration)
    if not fingerprint_path.exists() or fingerprint_path.read_text().strip() != fingerprint:
        resume_scores_path.unlink(missing_ok=True)
        fingerprint_path.write_text(fingerprint)

    scores = await scan_video_frames(
        video_path,
        app_settings.ffmpeg_bin,
        duration_seconds=duration,
        frame_interval_seconds=frame_interval,
        on_progress=on_progress,
        concurrency=frame_concurrency,
        resume_scores_path=resume_scores_path,
    )

    candidates = cluster_scenes(
        scores,
        confidence_threshold=confidence_threshold,
        frame_interval_seconds=frame_interval,
        merge_gap_seconds=merge_gap,
        high_confidence_override=high_confidence_override,
    )

    # Each candidate's dense re-scan drives two things from the same data: the
    # verified_fraction confidence signal (unchanged), and a refined start/end
    # -- the video-side analog of what Whisper forced-alignment does for a
    # subtitle cue, narrowing (or, just as often, widening into the padding)
    # the coarse main-scan boundary to whatever the denser look actually
    # found. Falls back to the original coarse boundary if the dense re-scan
    # itself fails, or somehow finds nothing above threshold at all.
    refined: list[tuple[float, float, float | None]] = []
    # Retained alongside `refined` so the optional Claude-verify step below can
    # target the exact frames this dense re-scan itself found most convincing
    # (see app.vision.claude_verify.top_score_timestamps), instead of blindly
    # guessing at even-spaced timestamps across the window. Empty for a
    # candidate whose verification pass failed.
    all_window_scores: list[list] = []
    if candidates and on_stage is not None:
        await on_stage(f"Verifying {len(candidates)} candidate scene(s)")
    for candidate in candidates:
        start, end, frac, window_scores = await _refine_candidate_boundary(
            video_path,
            candidate.start,
            candidate.end,
            verify_pad,
            verify_max_pad,
            verify_interval,
            confidence_threshold,
            frame_concurrency,
        )
        if end - start < min_duration:
            # A single (or tightly clustered) dense hit can refine to a
            # near-zero-width span -- expand symmetrically around its midpoint
            # rather than storing a degenerate scene that's effectively
            # invisible to both the blur filter and the review UI's Start/End
            # fields.
            mid = (start + end) / 2
            start, end = max(0.0, mid - min_duration / 2), mid + min_duration / 2
        refined.append((start, end, frac))
        all_window_scores.append(window_scores)

    # Optional precision filter, off by default (claude_vision_verify_enabled)
    # -- asks a Claude-vision-capable endpoint for a yes/no verdict on each
    # candidate's *refined* window, using it as an extra gate before
    # auto-approval decides anything below. claude_confirmed stays None for
    # every candidate when the setting is off, which the status-decision loop
    # below treats identically to "not confirmed" -- this step only ever adds
    # a stricter bar on top of auto_process, never approves anything by
    # itself. See app.vision.claude_verify for the fail-safe-toward-manual-
    # review contract on any request failure.
    claude_verify_enabled = bool(await get_setting(session, "claude_vision_verify_enabled"))
    claude_reasons: list[str | None] = [None] * len(candidates)
    claude_confirmed: list[bool | None] = [None] * len(candidates)
    # True only for a candidate Claude specifically flagged as sexual activity
    # (not just nudity/exposure) -- wired straight into DetectedScene.mute_audio
    # below. Stays False for the skip-high-confidence path (no Claude call
    # means no signal either way) and for anything Claude never confirmed.
    claude_mute_audio: list[bool] = [False] * len(candidates)
    if claude_verify_enabled and candidates:
        claude_base_url = await get_setting(session, "claude_vision_base_url")
        claude_api_key = await get_setting(session, "claude_vision_api_key")
        claude_model = await get_setting(session, "claude_vision_model")
        claude_skip_above = float(await get_setting(session, "claude_vision_skip_above_fraction"))
        if on_stage is not None:
            await on_stage(f"Verifying {len(candidates)} candidate scene(s) with Claude Vision")
        for i, (start, end, frac) in enumerate(refined):
            # Cost optimization: a candidate this confident from NudeNet's own
            # dense re-scan isn't the kind of borderline case this filter
            # exists to catch (see claude_vision_skip_above_fraction) -- skip
            # the paid call and treat it the same as a real "YES" verdict.
            if frac is not None and frac >= claude_skip_above:
                claude_confirmed[i] = True
                claude_reasons[i] = f"Skipped -- NudeNet dense re-scan already {frac:.0%} confident"
                continue
            result = await verify_candidate(
                base_url=claude_base_url,
                api_key=claude_api_key,
                model=claude_model,
                ffmpeg_bin=app_settings.ffmpeg_bin,
                video_path=video_path,
                start=start,
                end=end,
                sample_timestamps=top_score_timestamps(all_window_scores[i]) or None,
            )
            if result is not None:
                claude_confirmed[i] = result.confirmed
                claude_mute_audio[i] = result.mute_audio
                reason = f"{'YES' if result.confirmed else 'NO'}: {result.reason}".strip(": ")
                if result.mute_audio:
                    reason += " (sex scene -- audio muted)"
                claude_reasons[i] = reason
            else:
                claude_reasons[i] = "Verification failed -- left for manual review"

    stale = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.status.in_([SceneReviewStatus.pending, SceneReviewStatus.rejected]),
        )
    )
    for row in stale.scalars().all():
        await session.delete(row)

    # scene_auto_process controls whether a fresh candidate lands as
    # already-approved (the default -- see the setting's own comment in
    # DEFAULT_SETTINGS for why manual-review-first was dropped as this
    # feature's launch default) or pending, same as before. When the Claude
    # Vision precision filter is also on, it adds a stricter bar on top: a
    # candidate only auto-approves if *both* auto_process is on and Claude
    # confirmed it (claude_confirmed[i] is True). Either way this only decides
    # the *scene's* status -- whether a blur job actually runs off the back of
    # it is the scene_worker's call (it needs candidate_count from
    # ScanOutcome, which isn't known until after this function returns).
    #
    # An explicit Claude "NO" is a considered verdict, not an unknown -- it
    # rejects the candidate outright, the same as a human looking at it and
    # clicking Reject, rather than leaving it sitting in the pending queue
    # (which nobody routinely checks once auto_process is relied on). Only
    # genuine "nothing actually looked at this" cases -- the filter disabled,
    # or the request itself failing/timing out (confirmed is None either way)
    # -- fall back to pending, since those are real unknowns, not verdicts.
    now = datetime.datetime.utcnow()
    for candidate, (start, end, frac), confirmed, reason, mute_audio in zip(
        candidates, refined, claude_confirmed, claude_reasons, claude_mute_audio
    ):
        auto_approve = auto_process and (not claude_verify_enabled or confirmed is True)
        claude_rejected = claude_verify_enabled and confirmed is False
        if auto_approve:
            status = SceneReviewStatus.approved
        elif claude_rejected:
            status = SceneReviewStatus.rejected
        else:
            status = SceneReviewStatus.pending
        session.add(
            DetectedScene(
                title_id=title.id,
                start_seconds=start,
                end_seconds=end,
                peak_confidence=candidate.peak_confidence,
                verified_fraction=frac,
                claude_verify_reason=reason,
                mute_audio=mute_audio,
                status=status,
                reviewed_at=now if status != SceneReviewStatus.pending else None,
            )
        )
    await session.commit()

    # Full success -- the checkpoint has served its purpose. A future scan
    # (even a manual rescan of this same title) gets a new job_id and thus a
    # fresh scan_work_dir regardless, so there's nothing to gain by keeping it.
    shutil.rmtree(scan_work_dir, ignore_errors=True)

    return ScanOutcome(candidate_count=len(candidates))


async def apply_scene_blur(
    session: AsyncSession,
    title: Title,
    job_id: int,
    on_progress: ProgressCallback | None = None,
    on_stage: StageCallback | None = None,
) -> BlurOutcome:
    """Rebuild the "Vulgarr Edit" sibling file from scratch, blurring+muting
    every currently-approved scene on this title -- not just newly-approved
    ones. The source file is never opened for writing -- a failed blur just
    means the temp file gets deleted, nothing about the original changes, and
    any previously-generated "Vulgarr Edit" file is left in place untouched.

    job_id keys apply_blur's persistent work_dir (app_settings.data_dir /
    "blur_work" / str(job_id)) -- on a container restart, scene_worker's own
    startup reconciliation resets an interrupted job back to "queued" and
    redispatches it with this *same* job_id, so the resumed attempt's
    work_dir naturally lines up with whatever the interrupted attempt already
    produced. A brand-new manual Apply click (a genuinely new job_id) always
    gets a fresh work_dir, never accidentally inherits a stale one.

    Deliberately *not* scoped to approved-and-unapplied: this always operates
    on the full current approved set so that rejecting a scene that was
    already baked into an earlier "Vulgarr Edit" -- the whole point of the
    "auto-process now, correct it after the fact" workflow (see
    scene_auto_process) -- and then re-running Apply actually regenerates the
    file *without* that scene, rather than silently leaving its old blur/mute
    in place forever. Each regeneration re-encodes fresh from the untouched
    original, so there's no generational quality loss from doing this
    repeatedly.

    Raises ValueError if there's nothing currently approved at all (the caller
    is expected to have already checked this before enqueuing a job, but it's
    re-checked here too since the set can change between "Apply" being
    clicked and the job actually running)."""
    result = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.status == SceneReviewStatus.approved,
        )
    )
    scenes = result.scalars().all()
    if not scenes:
        raise ValueError("No approved scenes to blur")

    # Padded before merging (not after) -- a fixed safety margin against
    # boundary imprecision in an already-approved scene (a detected/adjusted
    # start or end that's a little tight against the real content, showing up
    # as a brief flash of unblurred/unmuted content right at the edge).
    # Padding two close-together approved scenes can make their windows
    # overlap; merging afterward (as already happened here regardless, for
    # independently-adjusted scenes) collapses that into one window rather
    # than erroring.
    #
    # Asymmetric on purpose (previously one symmetric value) -- direct
    # frame-by-frame ground truth found the classifier's boundary imprecision
    # is one-directional: it reliably catches a scene's real start, but
    # repeatedly stops firing several seconds before the shot actually cuts
    # away at the end (see scene_blur_pad_end_seconds's DEFAULT_SETTINGS
    # comment for the two real cases this was found on). No reason to pay for
    # a wide start pad the evidence doesn't call for.
    blur_pad_start = float(await get_setting(session, "scene_blur_pad_start_seconds"))
    blur_pad_end = float(await get_setting(session, "scene_blur_pad_end_seconds"))

    # Overlapping/adjacent approved windows (real after enough independent
    # "adjust" edits, or now from padding) are merged before building the
    # filter graph -- not required for correctness (ffmpeg's between()
    # OR-summation handles overlaps fine regardless), just keeps the term
    # count down against the batching limit. Every approved scene gets
    # blurred; only the subset with mute_audio set also gets its audio muted
    # -- a separate, explicit per-scene opt-in (default off) since
    # plot-relevant nudity doesn't always mean the dialogue over it needs
    # muting too. mute_scenes is always a subset of scenes.
    blur_merged = merge_intervals(
        [(max(0.0, s.start_seconds - blur_pad_start), s.end_seconds + blur_pad_end) for s in scenes],
        merge_gap_seconds=0.0,
    )
    blur_intervals = [MuteInterval(start=s, end=e) for s, e in blur_merged]

    mute_scenes = [s for s in scenes if s.mute_audio]
    mute_intervals: list[MuteInterval] = []
    if mute_scenes:
        mute_merged = merge_intervals(
            [(max(0.0, s.start_seconds - blur_pad_start), s.end_seconds + blur_pad_end) for s in mute_scenes],
            merge_gap_seconds=0.0,
        )
        mute_intervals = [MuteInterval(start=s, end=e) for s, e in mute_merged]

    video_crf = int(await get_setting(session, "blur_video_crf"))
    video_preset = await get_setting(session, "blur_video_preset")
    blur_radius = int(await get_setting(session, "scene_blur_radius"))
    blur_power = int(await get_setting(session, "scene_blur_power"))

    work_dir = app_settings.data_dir / "blur_work" / str(job_id)
    final_path = await apply_blur(
        video_path=Path(title.video_path),
        blur_intervals=blur_intervals,
        mute_intervals=mute_intervals,
        ffmpeg_bin=app_settings.ffmpeg_bin,
        ffprobe_bin=app_settings.ffprobe_bin,
        video_crf=video_crf,
        video_preset=video_preset,
        work_dir=work_dir,
        blur_radius=blur_radius,
        blur_power=blur_power,
        on_progress=on_progress,
        on_stage=on_stage,
    )

    now = datetime.datetime.utcnow()
    title.vulgarr_edit_path = str(final_path)
    title.vulgarr_edit_generated_at = now
    for scene in scenes:
        scene.applied_at = now

    # A scene that was applied by an earlier run but isn't in this run's
    # approved set (rejected sometime after being baked in) is no longer
    # reflected in the file that was just written above -- clear its
    # applied_at so it doesn't keep reading as "currently applied" now that
    # it isn't.
    stale_applied = await session.execute(
        select(DetectedScene).where(
            DetectedScene.title_id == title.id,
            DetectedScene.status != SceneReviewStatus.approved,
            DetectedScene.applied_at.is_not(None),
        )
    )
    for scene in stale_applied.scalars().all():
        scene.applied_at = None

    await session.commit()

    return BlurOutcome(output_path=final_path, scene_count=len(scenes))
