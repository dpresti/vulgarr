"""Plain enums shared across layers (DB models, matcher, queue) with no heavy deps."""

import enum


class MediaType(str, enum.Enum):
    movie = "movie"
    episode = "episode"


class Severity(str, enum.Enum):
    child = "child"  # kid-safe: mutes everything (was "mild")
    teen = "teen"    # teen-safe: mutes only what used to be tagged moderate/strong


# Ordering for severity-level filtering: a track generated at a given level mutes
# every severity at or above that rank (e.g. a "teen" track mutes only teen-tagged
# words, skipping child-tagged ones).
SEVERITY_RANK: dict[Severity, int] = {Severity.child: 0, Severity.teen: 1}

# Fixed canonical order multi-track output always uses, regardless of selection order --
# lets users learn "1st clean track is always Child" without depending on file-embedded
# labels, which aren't reliable on every container/muxer.
SEVERITY_CANONICAL_ORDER: list[Severity] = [Severity.child, Severity.teen]


def parse_severity_levels(value: str) -> list[Severity]:
    """Parse a comma-separated severity_levels string into canonical-order Severity list."""
    if not value:
        return [Severity.child]
    selected = {v.strip() for v in value.split(",") if v.strip()}
    levels = [s for s in SEVERITY_CANONICAL_ORDER if s.value in selected]
    return levels or [Severity.child]


def serialize_severity_levels(levels: list[Severity]) -> str:
    ordered = [s for s in SEVERITY_CANONICAL_ORDER if s in levels]
    return ",".join(s.value for s in ordered) or Severity.child.value


def parse_index_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(v) for v in value.split(",") if v.strip()]


def serialize_index_list(indices: list[int]) -> str:
    return ",".join(str(i) for i in indices)


# Mute-precision modes, selectable per title/season. "whole_line" mutes the cue's
# whole on-screen duration (safest, no assumption about word position). "estimate"
# narrows to a proportional guess of the word's position within the cue's text.
# "whisper" narrows to the word's real position in the audio via forced alignment
# (app.audio.forced_align) -- more accurate than the estimate, but slower to process.
PRECISE_MODES: list[str] = ["whole_line", "estimate", "whisper"]


class JobState(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class SceneReviewStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class SceneJobKind(str, enum.Enum):
    scan = "scan"
    blur = "blur"
    claude_verify = "claude_verify"


def is_mkv_path(video_path: str) -> bool:
    return video_path.lower().endswith(".mkv")


def title_href(title) -> str:
    """Library URL for a Title row -- every title (movie or episode) has its own
    detail page at /library/title/{id}; title_detail_card.html has no
    movie-specific assumptions, so this is genuinely shared, not just movies."""
    return f"/library/title/{title.id}"


def format_duration(seconds: float) -> str:
    """Human-readable elapsed duration, e.g. "1h 2m 3s" / "4m 5s" / "6s"."""
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


class TriggerSource(str, enum.Enum):
    manual = "manual"
    sonarr = "sonarr"
    radarr = "radarr"
    bazarr = "bazarr"


# Radarr/Sonarr tag labels a user can apply (via Overseerr's Advanced Request tag
# picker, which pushes selected tags straight into the Radarr/Sonarr "add" payload,
# or manually in Radarr/Sonarr's own UI) to opt a specific title into a pipeline --
# even when that pipeline's global auto-trigger setting is off (audio) or when no
# global auto-trigger exists at all (video/scene-detection, which is otherwise
# 100% manual). Additive only -- see tags_request_audio/tags_request_video below;
# neither ever suppresses what would already run without a tag.
TAG_VULGARR_AUDIO = "vulgarr-audio"
TAG_VULGARR_VIDEO = "vulgarr-video"
TAG_VULGARR_BOTH = "vulgarr-both"


def tags_request_audio(tags: set[str]) -> bool:
    """True if this title's Radarr/Sonarr tags opt it into the audio-mute pipeline,
    regardless of the trigger_sonarr_radarr_enabled setting. Lowercases defensively
    -- Radarr/Sonarr already lowercase tag labels on creation, but a caller
    shouldn't have to rely on that alone."""
    tags = {t.lower() for t in tags}
    return TAG_VULGARR_AUDIO in tags or TAG_VULGARR_BOTH in tags


def tags_request_video(tags: set[str]) -> bool:
    """True if this title's Radarr/Sonarr tags opt it into scene-detection
    scanning -- today scene scanning has no global auto-trigger setting at all
    (100% manual), so this is the only way scanning can run automatically."""
    tags = {t.lower() for t in tags}
    return TAG_VULGARR_VIDEO in tags or TAG_VULGARR_BOTH in tags
