from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.domain import JobState, MediaType, Severity, TriggerSource

__all__ = ["Base", "MediaType", "Severity", "JobState", "TriggerSource", "Title", "ProcessingJob", "WordListEntry", "AppSetting"]


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

    # Opt-in: mute a narrower window estimated around each matched word's position within
    # its cue, instead of the whole cue's on-screen duration. New titles inherit the
    # "default_precise_mute" app setting at creation time; overridable per title afterward.
    precise_mute: Mapped[bool] = mapped_column(default=False, nullable=False)

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

    jobs: Mapped[list["ProcessingJob"]] = relationship(back_populates="title", cascade="all, delete-orphan")

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
