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
