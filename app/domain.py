"""Plain enums shared across layers (DB models, matcher, queue) with no heavy deps."""

import enum


class MediaType(str, enum.Enum):
    movie = "movie"
    episode = "episode"


class Severity(str, enum.Enum):
    mild = "mild"          # e.g. "butt", "poop"
    moderate = "moderate"  # e.g. "damn", "hell"
    strong = "strong"      # hard profanity


# Ordering for severity-level filtering: a track generated at a given level mutes
# every severity at or above that rank (e.g. a "moderate" track mutes moderate+strong,
# skips mild).
SEVERITY_RANK: dict[Severity, int] = {Severity.mild: 0, Severity.moderate: 1, Severity.strong: 2}

# Fixed canonical order multi-track output always uses, regardless of selection order --
# lets users learn "1st clean track is always Mild" without depending on file-embedded
# labels, which aren't reliable on every container/muxer.
SEVERITY_CANONICAL_ORDER: list[Severity] = [Severity.mild, Severity.moderate, Severity.strong]


def parse_severity_levels(value: str) -> list[Severity]:
    """Parse a comma-separated severity_levels string into canonical-order Severity list."""
    if not value:
        return [Severity.mild]
    selected = {v.strip() for v in value.split(",") if v.strip()}
    levels = [s for s in SEVERITY_CANONICAL_ORDER if s.value in selected]
    return levels or [Severity.mild]


def serialize_severity_levels(levels: list[Severity]) -> str:
    ordered = [s for s in SEVERITY_CANONICAL_ORDER if s in levels]
    return ",".join(s.value for s in ordered) or Severity.mild.value


def parse_index_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(v) for v in value.split(",") if v.strip()]


def serialize_index_list(indices: list[int]) -> str:
    return ",".join(str(i) for i in indices)


class JobState(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


def is_mkv_path(video_path: str) -> bool:
    return video_path.lower().endswith(".mkv")


class TriggerSource(str, enum.Enum):
    manual = "manual"
    sonarr = "sonarr"
    radarr = "radarr"
    bazarr = "bazarr"
