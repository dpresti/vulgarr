import json
import re
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
    ("titles", "replacement_requested_at", "DATETIME"),
    ("processing_jobs", "progress_percent", "FLOAT"),
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

# Defaults for settings that must exist for the app to function.
DEFAULT_SETTINGS: dict[str, Any] = {
    "wordlist_version": 1,
    "concurrency_cap": settings.default_concurrency,
    "off_hours_enabled": False,
    "off_hours_start": "01:00",
    "off_hours_end": "07:00",
    # Ordered list of enabled trigger sources; earlier entries take priority when
    # de-duplicating near-simultaneous webhook events for the same title.
    # Bazarr is included but the user's Bazarr instance is currently down, so
    # sonarr/radarr (import + poll-for-subtitle) is first until that's fixed.
    "trigger_priority": ["sonarr_radarr", "bazarr"],
    "trigger_bazarr_enabled": False,
    "trigger_sonarr_radarr_enabled": True,
    "sonarr_radarr_subtitle_poll_timeout_seconds": 900,
    "sonarr_radarr_subtitle_poll_interval_seconds": 30,
    "wordlist_seeded": False,
    "migrated_severity_levels": False,
    # Whether newly-discovered titles default to word-level mute precision (narrower,
    # estimated windows) or whole-cue muting (safer, guaranteed to cover the word).
    # Existing titles are unaffected by changing this -- it's only applied at creation.
    "default_precise_mute": False,
}


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


async def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_column_migrations(conn)
        await _run_severity_vocabulary_migration(conn)
    async with async_session_maker() as session:
        for key, value in DEFAULT_SETTINGS.items():
            existing = await session.get(AppSetting, key)
            if existing is None:
                session.add(AppSetting(key=key, value=json.dumps(value)))
        await session.commit()
        await _backfill_episode_grouping_fields(session)
        await _backfill_multi_severity_fields(session)
        await _migrate_title_severity_levels_vocabulary(session)
        await _seed_default_wordlist_if_empty(session)


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
