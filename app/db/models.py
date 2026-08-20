from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain import JobState, MediaType, SceneJobKind, SceneReviewStatus, Severity, TriggerSource

__all__ = [
    "Base",
    "MediaType",
    "Severity",
    "JobState",
    "TriggerSource",
    "SceneReviewStatus",
    "SceneJobKind",
    "Title",
    "ProcessingJob",
    "DetectedScene",
    "SceneJob",
    "WordListEntry",
    "AppSetting",
]


class Base(DeclarativeBase):
    pass


class Title(Base):
    """A single processable video file (a movie, or one TV episode)."""

    __tablename__ = "titles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)

    # External library identifiers, so we can re-look-up paths/metadata via Sonarr/Radarr API
    sonarr_series_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sonarr_episode_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    radarr_movie_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Episode-only grouping fields, used to drill down show -> season -> episode
    # in the library UI without a per-row jobs join (this library has 18k+ episodes).
    series_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    season_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    episode_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    video_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    subtitle_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    subtitle_language: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Poster art URL from Radarr/Sonarr (their `remoteUrl`, i.e. the original TMDB/TVDB
    # CDN link -- directly loadable by a browser, unlike their `url` which only resolves
    # inside the Radarr/Sonarr container). Refreshed on every /library/sync; null until
    # the first sync after upgrading. For episodes this is the show's poster, denormalized
    # onto every episode row the same way series_title already is.
    poster_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Cached display status ("not_processed"/"queued"/"processing"/"done"/"failed"), kept in
    # sync by the queue worker on every job transition. Avoids joining ProcessingJob for every
    # row just to render a status badge, which doesn't scale to a library this size.
    status: Mapped[str] = mapped_column(String(20), default="not_processed", nullable=False)

    # Deprecated single-value threshold, superseded by severity_levels below. Left in
    # place (unused) rather than dropped -- SQLite can't cheaply drop columns, and old
    # rows are migrated into severity_levels on first startup after the upgrade.
    severity_threshold: Mapped[str] = mapped_column(String(20), default="mild", nullable=False)

    # Comma-separated subset of {"child","teen"} -- which threshold tracks to generate
    # for this title, e.g. "child" (one track) or "child,teen" (two tracks). Each
    # selected level gets its own clean audio track ("child" mutes everything, "teen"
    # mutes only teen-tagged words). Set per-movie, or in bulk for every episode in a show.
    severity_levels: Mapped[str] = mapped_column(String(40), default="child", nullable=False)

    # Deprecated boolean, superseded by precise_mode below. Left in place (unused)
    # rather than dropped -- SQLite can't cheaply drop columns.
    precise_mute: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Mute precision mode: "whole_line" (mute the cue's whole on-screen duration),
    # "estimate" (narrower window, proportional guess from the word's position in the
    # cue's text), or "whisper" (narrower window from real audio via forced alignment
    # -- app.audio.forced_align). New titles inherit the "default_precise_mode" app
    # setting at creation time; overridable per title/season afterward.
    precise_mode: Mapped[str] = mapped_column(String(20), default="whole_line", nullable=False)

    # Deprecated single-index field, superseded by clean_track_audio_indices below.
    clean_track_audio_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Comma-separated audio-relative indices of our own generated clean track(s), in the
    # same fixed order as severity_levels (child, teen), recorded after each
    # successful processing run. Used to detect/replace stale clean tracks on reprocess
    # instead of appending more -- file-embedded metadata (title tag) isn't a reliable way
    # to detect this, since some real-world MP4s silently drop it on remux.
    clean_track_audio_indices: Mapped[str | None] = mapped_column(String(40), nullable=True)

    # Processing state
    last_processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_processed_wordlist_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_clean_track: Mapped[bool] = mapped_column(default=False)
    matched_cue_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Set when an mkv-replacement search has been triggered via Sonarr/Radarr (status
    # becomes "awaiting_mkv" while this is set) -- the queue worker periodically checks
    # for a new file landing and clears this once it does (or the search is abandoned).
    replacement_requested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Scene detection/blur -- a wholly separate feature from the profanity-mute
    # pipeline above, deliberately not sharing its `status` field (see DetectedScene/
    # SceneJob below for why). Free-text like `status`, for the same reason: an
    # organically-growing small set of display states, not a fixed vocabulary
    # anything else needs to switch on.
    scene_scan_status: Mapped[str] = mapped_column(String(20), default="not_scanned", nullable=False)

    # Set once a blur "Apply" run successfully produces the sibling "Vulgarr Edit"
    # file (see app/scenes/pipeline.py) -- re-running Apply after more scenes are
    # approved regenerates both fields. Never points at the original video_path.
    vulgarr_edit_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    vulgarr_edit_generated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    jobs: Mapped[list["ProcessingJob"]] = relationship(back_populates="title", cascade="all, delete-orphan")
    detected_scenes: Mapped[list["DetectedScene"]] = relationship(back_populates="title", cascade="all, delete-orphan")
    scene_jobs: Mapped[list["SceneJob"]] = relationship(back_populates="title", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("video_path", name="uq_titles_video_path"),)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title_id: Mapped[int] = mapped_column(ForeignKey("titles.id"), nullable=False)
    state: Mapped[JobState] = mapped_column(Enum(JobState), default=JobState.queued, nullable=False)
    trigger_source: Mapped[TriggerSource] = mapped_column(Enum(TriggerSource), nullable=False)
    wordlist_version: Mapped[int] = mapped_column(Integer, nullable=False)

    progress_message: Mapped[str | None] = mapped_column(String(256), nullable=True)
    progress_percent: Mapped[float | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    title: Mapped["Title"] = relationship(back_populates="jobs")


class DetectedScene(Base):
    """A candidate scene window found by a vision-classifier scan (app/scenes/pipeline.py),
    pending human review. Deliberately a separate table from ProcessingJob/Title.status --
    those are already fully owned by the subtitle/mute pipeline's semantics, and scene
    detection is a wholly independent feature with its own review lifecycle."""

    __tablename__ = "detected_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title_id: Mapped[int] = mapped_column(ForeignKey("titles.id"), nullable=False)

    start_seconds: Mapped[float] = mapped_column(nullable=False)
    end_seconds: Mapped[float] = mapped_column(nullable=False)
    peak_confidence: Mapped[float] = mapped_column(nullable=False)

    status: Mapped[SceneReviewStatus] = mapped_column(
        Enum(SceneReviewStatus), default=SceneReviewStatus.pending, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Set once this scene's window has actually been baked into a "Vulgarr Edit"
    # sibling file by an Apply run (see app/scenes/pipeline.py:apply_scene_blur) --
    # distinct from `reviewed_at`/`status`, since approving and applying are two
    # separate, explicit actions.
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    title: Mapped["Title"] = relationship(back_populates="detected_scenes")


class SceneJob(Base):
    """Mirrors ProcessingJob's shape almost exactly, but as its own table (rather than
    adding a `kind` column to ProcessingJob) so none of the audio pipeline's worker
    logic or tests are touched by this feature."""

    __tablename__ = "scene_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title_id: Mapped[int] = mapped_column(ForeignKey("titles.id"), nullable=False)
    kind: Mapped[SceneJobKind] = mapped_column(Enum(SceneJobKind), nullable=False)
    state: Mapped[JobState] = mapped_column(Enum(JobState), default=JobState.queued, nullable=False)

    progress_message: Mapped[str | None] = mapped_column(String(256), nullable=True)
    progress_percent: Mapped[float | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    title: Mapped["Title"] = relationship(back_populates="scene_jobs")


class WordListEntry(Base):
    __tablename__ = "wordlist_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    term: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    severity: Mapped[Severity] = mapped_column(Enum(Severity), default=Severity.teen, nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True)
    match_whole_word: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppSetting(Base):
    """Simple key/value store for runtime-configurable settings."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
