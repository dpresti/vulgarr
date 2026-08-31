import hashlib
import hmac
import json
import re
import secrets
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.db.models import AppSetting, Base, MediaType, Title, WordListEntry
from app.wordlist_defaults import DEFAULT_WORDLIST

# (table, column, DDL type/default) added after the initial release. SQLite's
# create_all() only creates missing *tables*, never adds columns to a table
# that already exists, so already-deployed databases need an explicit
# ALTER TABLE for each of these, run once at startup.
_COLUMNS_ADDED_LATER = [
    ("titles", "series_title", "VARCHAR(512)"),
    ("titles", "season_number", "INTEGER"),
    ("titles", "episode_number", "INTEGER"),
    ("titles", "status", "VARCHAR(20) NOT NULL DEFAULT 'not_processed'"),
    ("titles", "severity_threshold", "VARCHAR(20) NOT NULL DEFAULT 'child'"),
    ("titles", "clean_track_audio_index", "INTEGER"),
    ("titles", "severity_levels", "VARCHAR(40) NOT NULL DEFAULT 'child'"),
    ("titles", "clean_track_audio_indices", "VARCHAR(40)"),
    ("titles", "precise_mute", "BOOLEAN NOT NULL DEFAULT 0"),
    ("titles", "precise_mode", "VARCHAR(20) NOT NULL DEFAULT 'whole_line'"),
    ("titles", "replacement_requested_at", "DATETIME"),
    ("titles", "poster_url", "VARCHAR(1024)"),
    ("processing_jobs", "progress_percent", "FLOAT"),
    ("titles", "scene_scan_status", "VARCHAR(20) NOT NULL DEFAULT 'not_scanned'"),
    ("titles", "vulgarr_edit_path", "VARCHAR(1024)"),
    ("titles", "vulgarr_edit_generated_at", "DATETIME"),
    ("detected_scenes", "verified_fraction", "FLOAT"),
    ("detected_scenes", "mute_audio", "BOOLEAN NOT NULL DEFAULT 0"),
    ("detected_scenes", "claude_verify_reason", "VARCHAR(500)"),
    ("titles", "imdb_id", "VARCHAR(32)"),
    ("titles", "content_advisory_summary", "VARCHAR(255)"),
    ("titles", "content_advisory_checked_at", "DATETIME"),
    ("titles", "content_advisory_item_id", "INTEGER"),
]

_DISPLAY_NAME_EPISODE_RE = re.compile(r"^(?P<series>.+) - S(?P<season>\d+)E(?P<episode>\d+)(?: - .*)?$")


async def _run_column_migrations(conn: AsyncConnection) -> None:
    table_columns: dict[str, set[str]] = {}
    for table, column, ddl_type in _COLUMNS_ADDED_LATER:
        if table not in table_columns:
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            table_columns[table] = {row[1] for row in result.fetchall()}
        if column not in table_columns[table]:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
            table_columns[table].add(column)


async def _run_severity_vocabulary_migration(conn: AsyncConnection) -> None:
    """Collapse the old 3-level mild/moderate/strong severity vocabulary into the new
    2-level child/teen vocabulary (child = mutes everything, teen = mutes only what
    used to be tagged moderate or strong). Runs as raw SQL, before any ORM query
    touches wordlist_entries -- that column is a real Enum(Severity) type, and any
    row still holding an old string value would fail to deserialize the instant the
    Severity enum stops knowing those members. Each UPDATE's WHERE clause only
    matches pre-migration values, so this is safe to run on every startup."""
    await conn.execute(text("UPDATE wordlist_entries SET severity = 'child' WHERE severity = 'mild'"))
    await conn.execute(text("UPDATE wordlist_entries SET severity = 'teen' WHERE severity IN ('moderate', 'strong')"))


_SEVERITY_LEVEL_VOCAB_MAP = {"mild": "child", "moderate": "teen", "strong": "teen"}


def _migrate_severity_levels_string(value: str) -> str:
    mapped: list[str] = []
    for token in (t.strip() for t in value.split(",")):
        new_token = _SEVERITY_LEVEL_VOCAB_MAP.get(token, token)
        if new_token and new_token not in mapped:
            mapped.append(new_token)
    order = {"child": 0, "teen": 1}
    mapped.sort(key=lambda t: order.get(t, 99))
    return ",".join(mapped) or "child"


async def _migrate_title_severity_levels_vocabulary(session: AsyncSession) -> None:
    """titles.severity_levels is a plain string column (no Enum type), so this is
    safe to do via the ORM -- but still only touches rows that still hold an old
    mild/moderate/strong token, so it's a no-op once migrated."""
    result = await session.execute(
        select(Title).where(
            Title.severity_levels.like("%mild%")
            | Title.severity_levels.like("%moderate%")
            | Title.severity_levels.like("%strong%")
        )
    )
    changed = False
    for title in result.scalars().all():
        title.severity_levels = _migrate_severity_levels_string(title.severity_levels)
        changed = True
    if changed:
        await session.commit()


async def _backfill_multi_severity_fields(session: AsyncSession) -> None:
    """One-time migration from the old single-value severity_threshold/clean_track_audio_index
    columns into the new multi-value severity_levels/clean_track_audio_indices columns.

    severity_levels defaults to 'mild' for every row via the ALTER TABLE DEFAULT, so it can't
    be used to detect "never migrated" after the fact -- gated by a settings flag instead, so
    this only runs once and never clobbers a user's later multi-select choices.
    """
    already_migrated = await get_setting(session, "migrated_severity_levels")
    if not already_migrated:
        result = await session.execute(select(Title))
        for title in result.scalars().all():
            title.severity_levels = title.severity_threshold
        await set_setting(session, "migrated_severity_levels", True)

    result = await session.execute(
        select(Title).where(Title.clean_track_audio_index.is_not(None), Title.clean_track_audio_indices.is_(None))
    )
    changed = False
    for title in result.scalars().all():
        title.clean_track_audio_indices = str(title.clean_track_audio_index)
        changed = True
    if changed:
        await session.commit()


async def _backfill_precise_mode(session: AsyncSession) -> None:
    """One-time migration from the old boolean precise_mute column into the new
    3-way precise_mode column ("whole_line"/"estimate"/"whisper"). Gated by a
    settings flag rather than an unmigrated-value check, since precise_mode's ALTER
    TABLE DEFAULT already sets every row to 'whole_line', which is indistinguishable
    from a row that was genuinely never precise."""
    already_migrated = await get_setting(session, "migrated_precise_mode")
    if already_migrated:
        return
    result = await session.execute(select(Title).where(Title.precise_mute.is_(True)))
    for title in result.scalars().all():
        title.precise_mode = "estimate"
    await set_setting(session, "migrated_precise_mode", True)
    await session.commit()


async def _backfill_episode_grouping_fields(session: AsyncSession) -> None:
    """One-time backfill for rows synced before series_title/season/episode existed
    as columns -- parses them back out of display_name ("Show - S01E02")."""
    result = await session.execute(
        select(Title).where(Title.season_number.is_(None), Title.media_type == MediaType.episode)
    )
    changed = False
    for title in result.scalars().all():
        match = _DISPLAY_NAME_EPISODE_RE.match(title.display_name)
        if match:
            title.series_title = match.group("series")
            title.season_number = int(match.group("season"))
            title.episode_number = int(match.group("episode"))
            changed = True
    if changed:
        await session.commit()


engine = create_async_engine(settings.database_url, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

# Defaults for settings that must exist for the app to function. Integration
# URLs/keys and other values that used to be env-only (see app/config.py) are
# seeded here from the environment on first run, then fully owned by the
# Settings page from then on -- editing them there no longer requires a
# redeploy/restart.
DEFAULT_SETTINGS: dict[str, Any] = {
    "wordlist_version": 1,
    "concurrency_cap": settings.default_concurrency,
    "off_hours_enabled": False,
    "off_hours_start": "01:00",
    "off_hours_end": "07:00",
    # Ordered list of enabled trigger sources; earlier entries take priority when
    # de-duplicating near-simultaneous webhook events for the same title.
    "trigger_priority": ["sonarr_radarr", "bazarr"],
    "trigger_bazarr_enabled": False,
    "trigger_sonarr_radarr_enabled": True,
    "sonarr_radarr_subtitle_poll_timeout_seconds": 900,
    "sonarr_radarr_subtitle_poll_interval_seconds": 30,
    "wordlist_seeded": False,
    "migrated_severity_levels": False,
    "migrated_precise_mode": False,
    # Which mute-precision mode newly-discovered titles default to: "whole_line",
    # "estimate" (word-level, proportional guess), or "whisper" (word-level, forced
    # alignment against the real audio). Existing titles are unaffected by changing
    # this -- it's only applied at creation.
    "default_precise_mode": "whole_line",
    # Integrations -- seeded from SPF_* env vars once, editable in Settings after that.
    "sonarr_url": settings.sonarr_url,
    "sonarr_api_key": settings.sonarr_api_key,
    "radarr_url": settings.radarr_url,
    "radarr_api_key": settings.radarr_api_key,
    "bazarr_url": settings.bazarr_url,
    "bazarr_api_key": settings.bazarr_api_key,
    "doesthedogdie_api_key": settings.doesthedogdie_api_key,
    # Never seeded blank -- an unset webhook token used to mean "no auth check at
    # all" on inbound webhooks. A fresh install with nothing in .env now gets a
    # random token instead of silently being wide open.
    "webhook_token": settings.webhook_token or secrets.token_urlsafe(32),
    "clean_track_title": settings.clean_track_title,
    "clean_track_language": settings.clean_track_language,
    "default_subtitle_language": settings.default_subtitle_language,
    # 0 = keep every backup forever. Defaults to 7 so backups don't accumulate
    # unbounded out of the box; editable in Settings > Backups.
    "backup_retention_days": 7,
    # False disables backups entirely -- the original is still moved aside
    # during the swap for crash-safety (see remux_with_clean_track), but the
    # backup copy is deleted immediately afterward instead of being kept.
    "backups_enabled": True,
    # Scene detection/blur has no master on/off switch -- it never runs
    # automatically on import (unlike the word-list mute pipeline), only ever
    # starting from an explicit "Scan for scenes" click, so there's nothing an
    # enable toggle would actually be guarding against.
    # Minimum per-frame classifier confidence to count as a hit at all -- kept on
    # the permissive side deliberately, since every candidate still goes through
    # manual review; a stricter default would risk silently missing real scenes
    # rather than just producing an extra reviewable candidate. Lowered from an
    # original 0.5 after real-world testing against GoT S01E10 against a
    # community-compiled ground-truth timestamp list: within genuine nudity
    # windows, NudeNet's per-frame confidence on real (non-static, often
    # motion-blurred or partially occluded) footage routinely dips into the
    # 0.3-0.5 range even while a spike a second or two away clears 0.6-0.7 --
    # 0.5 was cutting out most of the real signal, not just noise. Outside real
    # windows, scores in that same test sat at ~0, so 0.35 stays well clear of
    # the false-positive floor while catching far more of the true windows.
    # Lowered again from 0.35 to 0.3 after diagnosing a specific miss on GoT
    # S01E10's 50:40-51:05 window: the tail of that scene (~3046.5-3051.75)
    # showed real but weaker/noisier signal (0.29-0.66) that a 0.35 floor was
    # clipping off, even though the scene's core (3042.5-3046.25) was a clean
    # 0.66-0.72. 0.3 catches more of that noisy tail without meaningfully
    # opening the false-positive floor (still well above the ~0 baseline
    # measured outside real windows in the original 0.5->0.35 tuning pass).
    "scene_confidence_threshold": 0.3,
    # How often to sample a frame during a scan. Lowered from an original 2.0s
    # after the same real-world test above: per-frame confidence during a real
    # scene fluctuates second-to-second (motion, angle, occlusion), so a 2s
    # sample interval routinely landed *between* the brief above-threshold
    # spikes and only caught one out of a ~10-25s window -- confirmed by
    # sampling every second across all six ground-truth windows and finding
    # real signal that a 2s grid would have mostly missed.
    #
    # Lowered again from 1.0s to 0.5s after a real full-episode scan (GoT
    # S01E10, post continuous-decode + concurrency work) still missed one of
    # six ground-truth windows (51:15-51:25): a direct 0.25s-dense re-scan of
    # that exact window confirmed real signal was present but compressed into
    # a ~2.5s spike train swinging from 0.0 to 0.57 confidence within
    # quarter-second steps -- at a 1.0s grid, only 2 consecutive samples ever
    # landed on a peak, right at the min_consecutive_frames minimum, and
    # whether the grid phase lands on a peak or one of the dips is essentially
    # chance for content this brief/volatile. A 0.5s grid halves that risk.
    # Thanks to the continuous-decode-pass + parallel-classification work,
    # this no longer means "hours" -- a real full-episode scan at 1.0s (52min
    # episode, 3157 samples) measured ~4m10s end-to-end; halving the interval
    # roughly doubles the sample count, so expect ~8min for an episode this
    # length, or a very rough ~15-25min for a 2-hour movie depending on scene
    # count (the per-candidate verification pass below adds a little more).
    "scene_frame_interval_seconds": 0.5,
    # Not a candidate-rejection filter -- app.vision.scene_cluster.cluster_scenes
    # deliberately never drops a real, persistence-validated coarse candidate for
    # being short (confirmed against GoT S03E03 that doing so throws away
    # genuine short detections with nothing nearby to merge into). This only
    # pads a candidate's *refined* boundary (after its own denser per-candidate
    # re-scan) out to this length around its center, if the real signal found
    # there is even shorter -- avoids storing a near-zero-width scene that's
    # effectively invisible to both the blur filter and the review UI.
    "scene_min_duration_seconds": 1.0,
    # Wider than the audio pipeline's cue-merge gap (0.25s) -- a visual scene
    # tolerates a brief cutaway shot without splitting into two separate candidates
    # the way two adjacent profane words wouldn't. Widened from an original 2.0s
    # in the same real-world tuning pass: real scenes showed brief (3-4s) dips
    # below threshold mid-scene (an actor's angle changing, a face-only shot)
    # that would otherwise split one real scene into two adjacent candidates --
    # 4.0s bridges those without also merging genuinely separate scenes, which
    # in this test were separated by 15s+ gaps at minimum.
    #
    # Widened again to 6.0s after the same 50:40-51:05 diagnosis above found a
    # real (if intermittent) signal run continuing to ~3051.75 -- a couple of
    # below-threshold samples right around 3051-3052 were enough to cut the
    # merged run short at the old 4.0s gap tolerance.
    #
    # Widened again, substantially, to 25.0s after direct per-second ground
    # truth (frame-by-frame confidence dump, not just cluster output) on GoT
    # S01E10 disproved the "15s+ separation = genuinely distinct scene"
    # assumption above: a single continuous nude scene (~35:10-35:41) had the
    # classifier hard-zero (no detection at all, not just below threshold) for
    # 14 consecutive seconds in the middle of it, and another
    # (~50:43-51:16) had a 21-second dead zone -- both flanked by frames
    # scoring 0.4-0.7 on either side. This isn't a threshold problem (those
    # frames score exactly 0.0, not borderline) -- NudeNet's per-frame
    # confidence just genuinely blinks off for stretches this long even
    # mid-scene (camera angle, framing, motion blur), so bridging has to
    # tolerate real gaps this size or the interior of an otherwise-detected
    # scene is left completely unblurred. The cost of merging two actually-
    # distinct scenes that happen to fall within 25s of each other is a false
    # positive (harmless: a few extra seconds blurred) -- much cheaper than
    # the false negative this was previously causing.
    "scene_merge_gap_seconds": 25.0,
    # A run shorter than the min-consecutive-frames persistence check (see
    # app.vision.scene_cluster.DEFAULT_MIN_CONSECUTIVE_FRAMES) still counts as a
    # real candidate if its peak confidence clears this higher bar -- added after
    # directly confirming, against GoT S06E07's 2151-2153 ground-truth window,
    # that a genuine quick-cut flash (FEMALE_BREAST_EXPOSED=0.73 at a single
    # sampled frame, both neighbors hard 0.0) was being discarded purely for
    # being one sample wide, at every sampling density tried down to 0.25s --
    # the flash itself is narrower than the grid, so denser sampling alone can't
    # fix it. 0.6 sits comfortably above scene_confidence_threshold's 0.3 (and
    # above the ~0-0.3 noise band single-frame spikes from motion blur/skin-toned
    # lighting were observed at) while still catching confirmed real hits like
    # the 0.73 case above.
    "scene_high_confidence_single_frame_threshold": 0.6,
    "scene_scan_concurrency_cap": 1,
    # How many frames' extract+classify to run in flight at once *within* a
    # single scan -- distinct from scene_scan_concurrency_cap above, which caps
    # how many titles can scan simultaneously. Each sample is an independent
    # ffmpeg subprocess + onnxruntime inference call with no shared mutable
    # state, so this is free parallelism, not a tuning tradeoff. Benchmarked
    # directly against a real file on real homelab hardware (20 cores
    # available): concurrency=1 measured 1.77s/frame effective, concurrency=8
    # dropped that to 0.32s/frame (~5.6x), concurrency=16 only added another
    # ~1.3x on top of that for double the in-flight work -- 8 is the point
    # where this stops paying for itself. Lower this if the box is under
    # memory/CPU pressure from other services sharing it.
    "scene_frame_classify_concurrency": 8,
    # After the main scan finds candidates, each one gets a second, much denser
    # re-scan of just its own padded window (see
    # app.vision.classifier.scan_window_frames) -- cheap, since it's a single
    # short clip rather than the whole movie. This drives two things now: the
    # per-scene "verified_fraction" confidence signal (scene_cluster.
    # verified_fraction), and the actual stored start/end boundary
    # (scene_cluster.refine_scene_boundary) -- the video-side analog of what
    # Whisper forced-alignment does for a subtitle cue, narrowing or widening
    # the coarse candidate to whatever this denser look actually finds.
    "scene_verify_pad_seconds": 5.0,
    # Ceiling on how far scene_verify_pad_seconds can grow when app.scenes.
    # pipeline._refine_candidate_boundary re-scans with a wider window because
    # the refined boundary landed right at the edge of what was already searched
    # (app.vision.scene_cluster.boundary_touches_window_edge) -- a real scene's
    # true extent past the coarse candidate's own edges is exactly what a fixed
    # 5s pad can silently clip, which is the "blur starts after nudity is
    # already visible / stops before the scene ends" failure real usage
    # reported. Not unbounded -- a false-positive edge-touch on a genuinely
    # short/noisy candidate shouldn't be able to balloon into scanning tens of
    # minutes of surrounding footage. 30s (6x the base pad, reached via at most
    # 3 doublings: 5->10->20->30) is an initial estimate, not yet validated
    # against the benchmark harness (docs/scene-detection.md) -- the harness's
    # own S01E10 run measured boundary errors well past what the old fixed 5s
    # pad could ever explain, so this exists to close that gap, but the exact
    # ceiling value should be tuned against real recall/boundary-error numbers
    # rather than trusted as-is.
    "scene_verify_max_pad_seconds": 30.0,
    # Denser than the main scan's scene_frame_interval_seconds (0.5s default)
    # deliberately -- the whole point of this second pass is a finer-grained
    # look at a window we already know is interesting, not just a repeat of the
    # same sampling density. Lowered further, from 0.25 to 0.1, once this pass
    # started also driving the stored scene boundary rather than just a
    # confidence stat -- boundary precision directly benefits from a denser
    # look the same way the confidence fraction always did.
    "scene_verify_frame_interval_seconds": 0.1,
    # Fraction of the dense verification pass's samples that must clear
    # scene_confidence_threshold for a candidate to be eligible for the review
    # list's one-click "Approve high-confidence" bulk action (see
    # app/routers/scenes.py:approve_high_confidence_scenes). Deliberately still
    # opt-in per-title, not automatic -- a real, sustained scene should clear
    # this comfortably, while a brief flash or a borderline/ambiguous frame run
    # (exactly the cases most worth a human actually looking) naturally scores
    # low on this fraction and stays in the manual queue.
    "scene_high_confidence_fraction": 0.5,
    # x264 quality (lower = better quality/bigger file) for the "Vulgarr Edit"
    # sibling file's re-encode -- unlike the mute pipeline, this is the one place
    # in the app that can't stream-copy video, so this is a genuine, exposed
    # speed/quality tradeoff rather than an internal implementation detail.
    "blur_video_crf": 23,
    "blur_video_preset": "medium",
    # boxblur strength for the video during blurred windows. Defaults tuned via
    # a direct visual comparison (raw frame vs. increasing radius/power) --
    # 90/8 was the point where nothing recognizable remained, just a smooth
    # tone/color gradient. boxblur uses a sliding-window algorithm, so its cost
    # is roughly independent of radius -- there's no real reason to cap how
    # heavy this can go, hence exposing both rather than hardcoding.
    "scene_blur_radius": 90,
    "scene_blur_power": 8,
    # Extra margin added to each approved scene's start/end before it's
    # actually blurred (and muted, for scenes with mute_audio on) at Apply
    # time -- distinct from the detection-side tuning above, which is about
    # finding candidates in the first place. This is a safety margin against
    # boundary imprecision in an already-approved scene: real content directly
    # reported as visible right at the edge of a blurred window ("flashes" at
    # the start/end of a scene). Applied to blur_intervals/mute_intervals in
    # app.scenes.pipeline.apply_scene_blur, then merged the same way adjacent
    # approved scenes already are, so padding two close-together scenes into
    # overlapping ranges just merges them into one window rather than erroring.
    #
    # Split into separate start/end values (previously one symmetric
    # scene_blur_pad_seconds) after direct frame-by-frame ground truth on GoT
    # S01E10 found a consistent, one-directional pattern: the classifier
    # reliably detects the *start* of a scene right where it actually begins,
    # but stops firing several seconds before the shot actually cuts away at
    # the *end* -- confirmed on two independent scenes the same session (the
    # 50:40-51:05 scene tuned earlier, and a second one at 35:41-35:45.7,
    # where direct frame extraction showed real nudity continuing to ~35:44.7,
    # a real shot cut, while classifier confidence was already flat 0.0 from
    # ~35:41.5 onward). Keeping the start pad small avoids wastefully blurring
    # extra seconds that were never a problem, while the end pad stays wide
    # to cover this asymmetric, recurring miss -- extra seconds of blur past a
    # scene's real end are harmless, an unblurred tail is not.
    "scene_blur_pad_start_seconds": 2.0,
    "scene_blur_pad_end_seconds": 5.0,
    # When on, a scan's candidate scenes are approved automatically (no manual
    # "Approve" click) and immediately chained into an Apply/blur job once the
    # scan finishes -- see app.scenes.pipeline.scan_for_scenes and
    # app.queue.scene_worker._run_job. Reverses this feature's original
    # "100% manual review, no auto-approve" launch decision, per an explicit
    # later call: manual review before every blur wasn't worth the friction
    # once detection accuracy was dialed in, and the safety net moved to
    # *after* the fact instead -- reject a scene post-hoc (even one already
    # applied) and re-run Apply to regenerate the "Vulgarr Edit" file without
    # it (apply_scene_blur always rebuilds from every *currently* approved
    # scene, not just newly-approved ones, specifically so this works). Kept
    # as a real toggle, not hardcoded, so manual review can be switched back
    # on without a code change if detection quality ever regresses.
    "scene_auto_process": True,
    # Optional precision filter on top of scene_auto_process: before
    # auto-approving a NudeNet candidate, ask a Claude-vision-capable endpoint
    # (any OpenAI-compatible chat completions API -- e.g. a self-hosted
    # LiteLLM proxy in front of Anthropic's API, see /docker/stack/litellm)
    # for a yes/no verdict, to catch NudeNet false positives (swimwear,
    # underwear, suggestive-but-covered shots) that would otherwise get
    # auto-blurred with no human ever looking at them. See
    # app.vision.claude_verify. Off by default -- this is the one piece of
    # this app that costs real, external, per-request money (a few cents per
    # scan; see the scene-detection plan's cost research), so nobody self-
    # hosting vulgarr should need an API key or an extra service unless they
    # deliberately opt in. A disabled/unreachable/misconfigured endpoint
    # always fails toward "leave pending for manual review", never toward
    # auto-approval -- see claude_verify.verify_candidate.
    "claude_vision_verify_enabled": False,
    # Base URL of an OpenAI-compatible chat completions API, e.g.
    # "http://litellm:4000/v1" for the LiteLLM stack referenced above.
    "claude_vision_base_url": "",
    # Bearer token sent to claude_vision_base_url -- LiteLLM's own
    # LITELLM_MASTER_KEY (or a virtual key), not your raw Anthropic key.
    "claude_vision_api_key": "",
    # Must match a model_name alias defined in the proxy's own config, not a
    # raw Anthropic model ID. Sonnet by default, not Haiku -- switched after a
    # real A/B test on unambiguous full-nudity frames from a real title found
    # Haiku genuinely inconsistent (wrong on one run, right on an identical
    # rerun of the same frames -- sampling variance, not a frame-selection
    # problem, see temperature=0 in claude_verify.verify_candidate). Missing
    # real content is the worse failure mode this whole feature exists to
    # prevent, and the cost difference between the two models is negligible
    # at this call volume (a low-cents-per-month difference at most) --
    # nowhere near enough to justify the accuracy risk.
    "claude_vision_model": "claude-sonnet-5",
    # Skips the Claude Vision call entirely for a candidate whose own dense
    # re-scan (verified_fraction) is already at/above this fraction -- the
    # precision filter exists to catch NudeNet false positives, which show up
    # as *low* verified_fraction (a brief/borderline hit), not high, so a
    # candidate this confident is spending real per-request money to
    # double-check something NudeNet was barely uncertain about at all.
    # Kept as its own, stricter setting rather than reusing
    # scene_high_confidence_fraction (0.5, used for the manual "Approve
    # high-confidence" bulk button) -- skipping a paid verification step
    # outright is a more consequential call than just surfacing a bulk-
    # approve button to a human, so it defaults to a much higher bar.
    "claude_vision_skip_above_fraction": 0.9,
    # Optional auth in front of the whole UI (webhooks are exempt -- they already
    # require their own token). "none" (default) leaves existing deployments
    # behind a trusted network/reverse proxy unaffected; "password" is the
    # original HTTP Basic Auth; "sso" adds Authentik OIDC login while keeping
    # Basic Auth alive underneath as a break-glass fallback (see app/auth.py).
    # auth_enabled/auth_username/auth_password_hash are kept (not renamed) --
    # "password" mode still reads them directly, and _migrate_auth_mode below
    # seeds auth_mode from auth_enabled once, for upgrades from before auth_mode
    # existed.
    "auth_mode": "none",
    "auth_enabled": False,
    "auth_username": "",
    "auth_password_hash": "",
    "sso_issuer_url": "",
    "sso_client_id": "",
    "sso_client_secret": "",
    "migrated_auth_mode": False,
    # First-run setup wizard (see app/routers/setup.py). Left False here (the
    # fresh-install default) -- init_db flips it True immediately for any
    # database that already had settings before this key existed, so upgrading
    # an existing deployment never forces it through the wizard.
    "setup_wizard_completed": False,
    "checked_setup_wizard_upgrade": False,
}


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash or "$" not in stored_hash:
        return False
    salt, _, expected = stored_hash.partition("$")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return hmac.compare_digest(digest.hex(), expected)


async def _seed_default_wordlist_if_empty(session: AsyncSession) -> None:
    # Tracked via a settings flag (not "table is currently empty") so this only ever
    # seeds once -- if the user deliberately clears the word list out later, a
    # restart shouldn't silently bring it back.
    already_seeded = await get_setting(session, "wordlist_seeded")
    if already_seeded:
        return
    for term, severity in DEFAULT_WORDLIST:
        session.add(WordListEntry(term=term, severity=severity))
    await set_setting(session, "wordlist_seeded", True)


async def _migrate_auth_mode(session: AsyncSession) -> None:
    """One-time migration for upgrades from before auth_mode existed. auth_mode is
    already seeded to "none" by the DEFAULT_SETTINGS loop in init_db -- this only
    needs to override that once, to "password", for a deployment that already had
    the old auth_enabled flag on."""
    if await get_setting(session, "migrated_auth_mode"):
        return
    if await get_setting(session, "auth_enabled"):
        await set_setting(session, "auth_mode", "password")
    await set_setting(session, "migrated_auth_mode", True)


async def _mark_setup_wizard_completed_for_upgrade(session: AsyncSession, is_upgrade: bool) -> None:
    """Called once ever (guarded the same way _migrate_auth_mode is) -- an existing
    deployment upgrading to the version that introduced the first-run wizard must
    never be forced through it. Guarded so a user who deliberately resets
    setup_wizard_completed later (to revisit onboarding) doesn't get it silently
    flipped back on the next restart."""
    if await get_setting(session, "checked_setup_wizard_upgrade"):
        return
    if is_upgrade:
        await set_setting(session, "setup_wizard_completed", True)
    await set_setting(session, "checked_setup_wizard_upgrade", True)


async def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_column_migrations(conn)
        await _run_severity_vocabulary_migration(conn)
    async with async_session_maker() as session:
        # Checked before the DEFAULT_SETTINGS loop below adds every key (including
        # setup_wizard_completed itself) with its default value -- a fresh install
        # has no AppSetting rows at all yet, while an upgrade already has this one
        # from every prior version of this app. Must be read before the loop, since
        # afterward "does this key exist" can no longer tell fresh installs apart
        # from upgrades (the loop just created it either way).
        is_upgrade = await session.get(AppSetting, "wordlist_version") is not None
        for key, value in DEFAULT_SETTINGS.items():
            existing = await session.get(AppSetting, key)
            if existing is None:
                session.add(AppSetting(key=key, value=json.dumps(value)))
        await session.commit()
        await _backfill_episode_grouping_fields(session)
        await _backfill_multi_severity_fields(session)
        await _backfill_precise_mode(session)
        await _migrate_title_severity_levels_vocabulary(session)
        await _seed_default_wordlist_if_empty(session)
        await _migrate_auth_mode(session)
        await _mark_setup_wizard_completed_for_upgrade(session, is_upgrade)


@asynccontextmanager
async def get_session():
    async with async_session_maker() as session:
        yield session


async def get_setting(session: AsyncSession, key: str) -> Any:
    row = await session.get(AppSetting, key)
    if row is None:
        return DEFAULT_SETTINGS.get(key)
    return json.loads(row.value)


async def set_setting(session: AsyncSession, key: str, value: Any) -> None:
    row = await session.get(AppSetting, key)
    if row is None:
        session.add(AppSetting(key=key, value=json.dumps(value)))
    else:
        row.value = json.dumps(value)
    await session.commit()


async def bump_wordlist_version(session: AsyncSession) -> int:
    current = await get_setting(session, "wordlist_version")
    new_version = int(current) + 1
    await set_setting(session, "wordlist_version", new_version)
    return new_version
