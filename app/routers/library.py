import datetime
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Select, case, func, select, update

from app.config import settings as app_settings
from app.db.models import MediaType, Title, TriggerSource
from app.db.session import get_session, get_setting
from app.domain import Severity, is_mkv_path, parse_severity_levels, serialize_severity_levels
from app.integrations.bazarr import BazarrClient
from app.integrations.radarr import RadarrClient
from app.integrations.sonarr import SonarrClient
from app.integrations.subtitle_lookup import find_subtitle_for_video
from app.library import enqueue_if_not_already_active, is_outdated, sync_radarr_library, sync_sonarr_library
from app.queue.worker import job_queue

router = APIRouter(prefix="/library", tags=["library"])
templates = Jinja2Templates(directory="app/templates")

PAGE_SIZE = 100
SEVERITY_OPTIONS = list(Severity)


def _severity_levels_from_checkboxes(sev_child: bool, sev_teen: bool) -> str:
    levels = [s for s, checked in [(Severity.child, sev_child), (Severity.teen, sev_teen)] if checked]
    return serialize_severity_levels(levels)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _qs(params: dict, **overrides) -> str:
    """Build a query string from the current params with some keys overridden, dropping empties."""
    merged = {**params, **overrides}
    merged = {k: v for k, v in merged.items() if v not in (None, "")}
    return "?" + urlencode(merged) if merged else ""


def _outdated_case(current_version: int):
    return case(
        (
            (Title.last_processed_wordlist_version.is_not(None))
            & (Title.last_processed_wordlist_version < current_version),
            1,
        ),
        else_=0,
    )


def _short_episode_label(title: Title) -> str:
    """Episode display_name is stored as "{series} - S{ss}E{ee} - {episode title}"
    (see app/library.py upsert_title). Within a season page the series+season are
    already shown in the breadcrumb/heading, so strip that prefix back off rather
    than repeating it on every row."""
    if title.season_number is None or title.episode_number is None:
        return title.display_name
    prefix = f"{title.series_title} - S{title.season_number:02d}E{title.episode_number:02d} - "
    suffix = title.display_name[len(prefix):] if title.display_name.startswith(prefix) else title.display_name
    return f"E{title.episode_number:02d} - {suffix}"


def _row_dict(
    title: Title, current_version: int, bazarr_message: str | None = None, short_label: bool = False
) -> dict:
    # subtitle_path can go stale if the .srt is deleted/moved outside the app (e.g.
    # manually removed from Bazarr) -- check the file is actually still there rather
    # than trusting the DB column, so the UI doesn't keep showing "yes" for a ghost path.
    has_subtitle = bool(title.subtitle_path) and Path(title.subtitle_path).exists()
    is_mkv = is_mkv_path(title.video_path)

    # Derived from current saved state (not a one-off message from whichever endpoint
    # happened to trigger this render) -- otherwise it would only show right after the
    # /severity POST and vanish the moment any other action (e.g. toggling precision)
    # re-rendered the row, even though the underlying mismatch is still true.
    severity_error = None
    if len(title.severity_levels.split(",")) > 1 and not is_mkv:
        severity_error = (
            "Multiple severity tracks aren't supported for this file type (only .mkv) -- "
            "processing now will use a single track for the most inclusive level selected. "
            "Search for an .mkv replacement below to get all selected tracks."
        )

    display_name = (
        _short_episode_label(title) if short_label and title.media_type == MediaType.episode else title.display_name
    )

    return {
        "title": title,
        "display_name": display_name,
        "status": title.status,
        "outdated": is_outdated(title, current_version),
        "bazarr_message": bazarr_message,
        "is_mkv": is_mkv,
        "severity_error": severity_error,
        "has_subtitle": has_subtitle,
    }


MOVIE_SORT_COLUMNS = {
    "title": Title.display_name.collate("NOCASE"),
    "status": Title.status,
    "subtitle": Title.subtitle_path,
    "muted_cues": Title.matched_cue_count,
    "severity": Title.severity_levels,
}


async def _load_movies(
    session, current_version: int, page: int, q: str | None, sort: str | None, sort_dir: str
) -> tuple[list[dict], int]:
    base: Select = select(Title).where(Title.media_type == MediaType.movie)
    if q:
        base = base.where(Title.display_name.like(f"%{_escape_like(q)}%", escape="\\"))

    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()

    sort_col = MOVIE_SORT_COLUMNS.get(sort, Title.display_name)
    order = sort_col.desc() if sort_dir == "desc" else sort_col.asc()

    result = await session.execute(
        base.order_by(order, Title.display_name.collate("NOCASE")).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
    )
    rows = [_row_dict(t, current_version) for t in result.scalars().all()]
    return rows, total


async def _load_processed(
    session, current_version: int, page: int, q: str | None, sort: str | None, sort_dir: str
) -> tuple[list[dict], int]:
    """Every fully-processed title (movie or episode) -- the combined view shown at
    the main /library route, per the sidebar's "Library" item being distinct from
    the "Movies"/"TV Shows" sub-items (which list everything, not just done ones)."""
    base: Select = select(Title).where(Title.status == "done")
    if q:
        base = base.where(Title.display_name.like(f"%{_escape_like(q)}%", escape="\\"))

    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()

    sort_col = MOVIE_SORT_COLUMNS.get(sort, Title.display_name)
    order = sort_col.desc() if sort_dir == "desc" else sort_col.asc()

    result = await session.execute(
        base.order_by(order, Title.display_name.collate("NOCASE")).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
    )
    rows = [_row_dict(t, current_version) for t in result.scalars().all()]
    return rows, total


SHOW_SORT_KEYS = {"name", "total", "done_count", "active_count", "failed_count", "outdated_count"}


async def _load_shows(
    session, current_version: int, page: int, q: str | None, sort: str | None, sort_dir: str
) -> tuple[list[dict], int]:
    outdated_expr = _outdated_case(current_version)

    total_col = func.count().label("total")
    done_col = func.sum(case((Title.status == "done", 1), else_=0)).label("done_count")
    failed_col = func.sum(case((Title.status == "failed", 1), else_=0)).label("failed_count")
    active_col = func.sum(case((Title.status.in_(["queued", "processing"]), 1), else_=0)).label("active_count")
    outdated_col = func.sum(outdated_expr).label("outdated_count")
    # Representative (not a true aggregate) values -- a show's episodes could have
    # mixed settings once per-episode editing exists, same caveat as severity_levels
    # already had below.
    severity_col = func.min(Title.severity_levels).label("severity_levels")
    precise_col = func.min(Title.precise_mute).label("precise_mute")

    query = select(
        Title.sonarr_series_id,
        Title.series_title,
        total_col,
        done_col,
        failed_col,
        active_col,
        outdated_col,
        severity_col,
        precise_col,
    ).where(Title.media_type == MediaType.episode)

    if q:
        query = query.where(Title.series_title.like(f"%{_escape_like(q)}%", escape="\\"))

    query = query.group_by(Title.sonarr_series_id, Title.series_title)

    count_subquery = query.with_only_columns(Title.sonarr_series_id).subquery()
    total = (await session.execute(select(func.count()).select_from(count_subquery))).scalar_one()

    sort_columns = {
        "name": Title.series_title.collate("NOCASE"),
        "total": total_col,
        "done_count": done_col,
        "active_count": active_col,
        "failed_count": failed_col,
        "outdated_count": outdated_col,
    }
    sort_col = sort_columns.get(sort, Title.series_title.collate("NOCASE"))
    order = sort_col.desc() if sort_dir == "desc" else sort_col.asc()

    result = await session.execute(
        query.order_by(order, Title.series_title.collate("NOCASE")).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
    )
    shows = [dict(row._mapping) for row in result.all()]
    return shows, total


async def _load_seasons(session, series_id: int, current_version: int) -> tuple[str | None, list[dict]]:
    outdated_expr = _outdated_case(current_version)
    query = (
        select(
            Title.season_number,
            func.count().label("total"),
            func.sum(case((Title.status == "done", 1), else_=0)).label("done_count"),
            func.sum(case((Title.status == "failed", 1), else_=0)).label("failed_count"),
            func.sum(case((Title.status.in_(["queued", "processing"]), 1), else_=0)).label("active_count"),
            func.sum(outdated_expr).label("outdated_count"),
            # Representative (not a true aggregate) values -- a season's episodes could
            # have mixed settings once per-episode editing exists.
            func.min(Title.severity_levels).label("severity_levels"),
            func.min(Title.precise_mute).label("precise_mute"),
        )
        .where(Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
        .group_by(Title.season_number)
        .order_by(Title.season_number)
    )
    result = await session.execute(query)
    seasons = [dict(row._mapping) for row in result.all()]
    for season in seasons:
        season["gating_message"] = await _non_mkv_gating_message(
            session,
            Title.sonarr_series_id == series_id,
            Title.season_number == season["season_number"],
            levels=season["severity_levels"] or "",
        )

    name_result = await session.execute(
        select(Title.series_title).where(Title.sonarr_series_id == series_id).limit(1)
    )
    series_title = name_result.scalar_one_or_none()
    return series_title, seasons


@router.get("", response_class=HTMLResponse)
async def library_page(
    request: Request,
    q: str | None = None,
    page: int = 1,
    sort: str | None = None,
    dir: str = "asc",
):
    """The main sidebar "Library" item -- every fully-processed title, movies and
    episodes combined. Distinct from the "Movies"/"TV Shows" sub-items, which list
    everything regardless of processing status."""
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        rows, total = await _load_processed(session, current_version, page, q, sort, dir)

    current_params = {"q": q, "page": page, "sort": sort, "dir": dir}

    def sort_link(key: str) -> str:
        new_dir = "desc" if sort == key and dir == "asc" else "asc"
        return _qs(current_params, sort=key, dir=new_dir, page=1)

    return templates.TemplateResponse(
        "library_processed.html",
        {
            "request": request,
            "q": q or "",
            "movie_rows": rows,
            "movie_total": total,
            "movie_page": page,
            "movie_page_size": PAGE_SIZE,
            "movie_sort": sort,
            "movie_dir": dir,
            "movie_sort_link": sort_link,
            "movie_pager_qs": lambda p: _qs(current_params, page=p),
            "wordlist_version": current_version,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.get("/movies", response_class=HTMLResponse)
async def movies_page(
    request: Request,
    q: str | None = None,
    movie_page: int = 1,
    movie_sort: str | None = None,
    movie_dir: str = "asc",
):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        movie_rows, movie_total = await _load_movies(session, current_version, movie_page, q, movie_sort, movie_dir)

    current_params = {"q": q, "movie_page": movie_page, "movie_sort": movie_sort, "movie_dir": movie_dir}

    def movie_sort_link(key: str) -> str:
        new_dir = "desc" if movie_sort == key and movie_dir == "asc" else "asc"
        return _qs(current_params, movie_sort=key, movie_dir=new_dir, movie_page=1)

    return templates.TemplateResponse(
        "library_movies.html",
        {
            "request": request,
            "q": q or "",
            "movie_rows": movie_rows,
            "movie_page": movie_page,
            "movie_total": movie_total,
            "movie_page_size": PAGE_SIZE,
            "movie_sort": movie_sort,
            "movie_dir": movie_dir,
            "movie_sort_link": movie_sort_link,
            "movie_pager_qs": lambda p: _qs(current_params, movie_page=p),
            "wordlist_version": current_version,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.get("/shows", response_class=HTMLResponse)
async def shows_page(
    request: Request,
    q: str | None = None,
    show_page: int = 1,
    show_sort: str | None = None,
    show_dir: str = "asc",
):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        shows, show_total = await _load_shows(session, current_version, show_page, q, show_sort, show_dir)

    current_params = {"q": q, "show_page": show_page, "show_sort": show_sort, "show_dir": show_dir}

    def show_sort_link(key: str) -> str:
        new_dir = "desc" if show_sort == key and show_dir == "asc" else "asc"
        return _qs(current_params, show_sort=key, show_dir=new_dir, show_page=1)

    return templates.TemplateResponse(
        "library_shows.html",
        {
            "request": request,
            "q": q or "",
            "shows": shows,
            "show_page": show_page,
            "show_total": show_total,
            "show_page_size": PAGE_SIZE,
            "show_sort": show_sort,
            "show_dir": show_dir,
            "show_sort_link": show_sort_link,
            "show_pager_qs": lambda p: _qs(current_params, show_page=p),
            "wordlist_version": current_version,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


async def _current_season_precise_mute(session, series_id: int, season_number: int) -> bool:
    result = await session.execute(
        select(Title.precise_mute)
        .where(Title.sonarr_series_id == series_id, Title.season_number == season_number)
        .limit(1)
    )
    return bool(result.scalar_one_or_none())


async def _non_mkv_gating_message(session, *conditions, levels: str) -> str | None:
    """Same gating concept as the per-movie one in _row_dict, but for a bulk
    show/season action: how many of the affected episodes aren't .mkv and will
    therefore only get a single track (the most inclusive level selected) instead
    of the full set, until each gets replaced."""
    if len(levels.split(",")) <= 1:
        return None
    total = (await session.execute(select(func.count()).where(*conditions))).scalar_one()
    non_mkv = (
        await session.execute(
            select(func.count()).where(*conditions, func.lower(Title.video_path).notlike("%.mkv"))
        )
    ).scalar_one()
    if not non_mkv:
        return None
    return (
        f"{non_mkv} of {total} episode(s) aren't .mkv and will only get a single track "
        f"(the most inclusive level selected) until replaced."
    )


async def _load_season(session, series_id: int, season_number: int, current_version: int) -> dict | None:
    """Single-season counterpart to _load_seasons, used to re-render just one season
    row after a bulk severity/precision/process action on it."""
    outdated_expr = _outdated_case(current_version)
    query = (
        select(
            Title.season_number,
            func.count().label("total"),
            func.sum(case((Title.status == "done", 1), else_=0)).label("done_count"),
            func.sum(case((Title.status == "failed", 1), else_=0)).label("failed_count"),
            func.sum(case((Title.status.in_(["queued", "processing"]), 1), else_=0)).label("active_count"),
            func.sum(outdated_expr).label("outdated_count"),
            func.min(Title.severity_levels).label("severity_levels"),
            func.min(Title.precise_mute).label("precise_mute"),
        )
        .where(
            Title.sonarr_series_id == series_id,
            Title.season_number == season_number,
            Title.media_type == MediaType.episode,
        )
        .group_by(Title.season_number)
    )
    row = (await session.execute(query)).first()
    if row is None:
        return None
    season = dict(row._mapping)
    season["gating_message"] = await _non_mkv_gating_message(
        session,
        Title.sonarr_series_id == series_id,
        Title.season_number == season_number,
        levels=season["severity_levels"] or "",
    )
    return season


@router.get("/shows/{series_id}", response_class=HTMLResponse)
async def show_detail(request: Request, series_id: int):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, seasons = await _load_seasons(session, series_id, current_version)

    return templates.TemplateResponse(
        "show_detail.html",
        {
            "request": request,
            "series_id": series_id,
            "series_title": series_title,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.get("/shows/{series_id}/season/{season_number}", response_class=HTMLResponse)
async def season_detail(request: Request, series_id: int, season_number: int):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        name_result = await session.execute(
            select(Title.series_title).where(Title.sonarr_series_id == series_id).limit(1)
        )
        series_title = name_result.scalar_one_or_none()

        result = await session.execute(
            select(Title)
            .where(Title.sonarr_series_id == series_id, Title.season_number == season_number)
            .order_by(Title.episode_number)
        )
        rows = [_row_dict(t, current_version, short_label=True) for t in result.scalars().all()]

    return templates.TemplateResponse(
        "season_detail.html",
        {
            "request": request,
            "series_id": series_id,
            "series_title": series_title,
            "season_number": season_number,
            "rows": rows,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.post("/sync")
async def sync_library(request: Request):
    async with get_session() as session:
        if app_settings.sonarr_url and app_settings.sonarr_api_key:
            client = SonarrClient(app_settings.sonarr_url, app_settings.sonarr_api_key)
            await sync_sonarr_library(session, client)
        if app_settings.radarr_url and app_settings.radarr_api_key:
            client = RadarrClient(app_settings.radarr_url, app_settings.radarr_api_key)
            await sync_radarr_library(session, client)

    return RedirectResponse(url="/settings", status_code=303)


@router.post("/{title_id}/process", response_class=HTMLResponse)
async def process_title(request: Request, title_id: int, short_label: bool = Form(False)):
    await job_queue.enqueue(title_id, TriggerSource.manual)

    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        row = _row_dict(title, current_version, short_label=short_label)

    return templates.TemplateResponse(
        "partials/title_row.html", {"request": request, "row": row, "severity_options": SEVERITY_OPTIONS}
    )


async def _bulk_enqueue_titles(session, *conditions) -> tuple[int, int]:
    """Enqueue every title matching conditions that has a real subtitle file on disk
    and isn't already queued/processing. Returns (queued_count, skipped_count) --
    skipped titles have no subtitle yet, so use Search Bazarr & Process on them
    individually rather than silently failing them here."""
    result = await session.execute(select(Title).where(*conditions))
    titles = result.scalars().all()
    queued = 0
    skipped = 0
    for title in titles:
        if not (title.subtitle_path and Path(title.subtitle_path).exists()):
            skipped += 1
            continue
        await enqueue_if_not_already_active(title.id, TriggerSource.manual)
        queued += 1
    return queued, skipped


def _process_message(queued: int, skipped: int) -> str:
    message = f"Queued {queued} episode(s)."
    if skipped:
        message += f" Skipped {skipped} with no subtitle -- use Search Bazarr & Process on those individually."
    return message


@router.post("/shows/{series_id}/process", response_class=HTMLResponse)
async def process_show(request: Request, series_id: int):
    conditions = (Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
    async with get_session() as session:
        queued, skipped = await _bulk_enqueue_titles(session, *conditions)
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, seasons = await _load_seasons(session, series_id, current_version)

    return templates.TemplateResponse(
        "show_detail.html",
        {
            "request": request,
            "series_id": series_id,
            "series_title": series_title,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
            "process_message": _process_message(queued, skipped),
        },
    )


@router.post("/shows/{series_id}/season/{season_number}/process", response_class=HTMLResponse)
async def process_season(request: Request, series_id: int, season_number: int):
    conditions = (Title.sonarr_series_id == series_id, Title.season_number == season_number)
    async with get_session() as session:
        queued, skipped = await _bulk_enqueue_titles(session, *conditions)
        current_version = int(await get_setting(session, "wordlist_version"))
        season = await _load_season(session, series_id, season_number, current_version)

    return templates.TemplateResponse(
        "partials/season_row.html",
        {
            "request": request,
            "series_id": series_id,
            "season": season,
            "process_message": _process_message(queued, skipped),
        },
    )


@router.post("/{title_id}/severity", response_class=HTMLResponse)
async def set_title_severity(
    request: Request,
    title_id: int,
    sev_child: bool = Form(False),
    sev_teen: bool = Form(False),
    short_label: bool = Form(False),
):
    levels = _severity_levels_from_checkboxes(sev_child, sev_teen)
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)

        # Always save the requested selection, even if it can't be fully honored yet --
        # otherwise picking multiple severities on a non-mkv file would be silently
        # discarded rather than remembered for once an .mkv replacement lands. The
        # resulting gating message/button are derived fresh in _row_dict from this
        # saved state, not computed here, so they persist across any later re-render
        # (e.g. toggling precision) rather than only showing right after this POST.
        title.severity_levels = levels
        await session.commit()
        await session.refresh(title)

        row = _row_dict(title, current_version, short_label=short_label)

    return templates.TemplateResponse(
        "partials/title_row.html", {"request": request, "row": row, "severity_options": SEVERITY_OPTIONS}
    )


@router.post("/shows/{series_id}/season/{season_number}/severity", response_class=HTMLResponse)
async def set_season_severity(
    request: Request,
    series_id: int,
    season_number: int,
    sev_child: bool = Form(False),
    sev_teen: bool = Form(False),
):
    levels = _severity_levels_from_checkboxes(sev_child, sev_teen)
    conditions = (Title.sonarr_series_id == series_id, Title.season_number == season_number)
    async with get_session() as session:
        await session.execute(update(Title).where(*conditions).values(severity_levels=levels))
        await session.commit()
        current_version = int(await get_setting(session, "wordlist_version"))
        season = await _load_season(session, series_id, season_number, current_version)

    return templates.TemplateResponse(
        "partials/season_row.html",
        {"request": request, "series_id": series_id, "season": season},
    )


@router.post("/{title_id}/toggle-precise", response_class=HTMLResponse)
async def toggle_title_precise_mute(request: Request, title_id: int, short_label: bool = Form(False)):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        title.precise_mute = not title.precise_mute
        await session.commit()
        await session.refresh(title)
        row = _row_dict(title, current_version, short_label=short_label)

    return templates.TemplateResponse(
        "partials/title_row.html", {"request": request, "row": row, "severity_options": SEVERITY_OPTIONS}
    )


@router.post("/shows/{series_id}/season/{season_number}/toggle-precise", response_class=HTMLResponse)
async def set_season_precise_mute(request: Request, series_id: int, season_number: int):
    async with get_session() as session:
        current = await _current_season_precise_mute(session, series_id, season_number)
        await session.execute(
            update(Title)
            .where(Title.sonarr_series_id == series_id, Title.season_number == season_number)
            .values(precise_mute=not current)
        )
        await session.commit()
        current_version = int(await get_setting(session, "wordlist_version"))
        season = await _load_season(session, series_id, season_number, current_version)

    return templates.TemplateResponse(
        "partials/season_row.html",
        {"request": request, "series_id": series_id, "season": season},
    )


@router.post("/{title_id}/search-subtitle", response_class=HTMLResponse)
async def search_subtitle_via_bazarr(request: Request, title_id: int, short_label: bool = Form(False)):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)

        if not app_settings.bazarr_url or not app_settings.bazarr_api_key:
            message = "Bazarr isn't configured (missing URL/API key)."
        elif title.media_type == MediaType.episode and not (title.sonarr_series_id and title.sonarr_episode_id):
            message = "Missing Sonarr series/episode ID -- try syncing the library first."
        elif title.media_type == MediaType.movie and not title.radarr_movie_id:
            message = "Missing Radarr movie ID -- try syncing the library first."
        else:
            client = BazarrClient(app_settings.bazarr_url, app_settings.bazarr_api_key)
            try:
                if title.media_type == MediaType.episode:
                    ok = await client.search_episode_subtitle(
                        series_id=title.sonarr_series_id,
                        episode_id=title.sonarr_episode_id,
                        language=app_settings.default_subtitle_language,
                    )
                else:
                    ok = await client.search_movie_subtitle(
                        radarr_id=title.radarr_movie_id, language=app_settings.default_subtitle_language
                    )
            except Exception as exc:  # noqa: BLE001 -- surface any connection/API error to the UI
                ok = False
                message = f"Bazarr request failed: {exc}"
            else:
                if not ok:
                    message = "Bazarr rejected the search request."
                else:
                    subtitle = find_subtitle_for_video(Path(title.video_path), app_settings.default_subtitle_language)
                    if subtitle:
                        title.subtitle_path = str(subtitle)
                        title.subtitle_language = app_settings.default_subtitle_language
                        await session.commit()
                        await job_queue.enqueue(title.id, TriggerSource.manual)
                        message = "Found it! Subtitle downloaded, queued for processing."
                    else:
                        message = "Searched, but no subtitle was found by any provider."

        await session.refresh(title)
        row = _row_dict(title, current_version, bazarr_message=message, short_label=short_label)

    return templates.TemplateResponse(
        "partials/title_row.html", {"request": request, "row": row, "severity_options": SEVERITY_OPTIONS}
    )


@router.post("/{title_id}/search-mkv-replacement", response_class=HTMLResponse)
async def search_mkv_replacement(request: Request, title_id: int, short_label: bool = Form(False)):
    """Trigger a Sonarr/Radarr search for an .mkv release of this title, so multiple
    severity tracks become available. The queue worker polls for the new file to land
    and, once it does, searches Bazarr for a subtitle if needed and auto-queues processing --
    see JobQueue._check_one_replacement."""
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)

        if is_mkv_path(title.video_path):
            message = "This file is already .mkv."
        elif title.media_type == MediaType.episode and not (title.sonarr_series_id and title.sonarr_episode_id):
            message = "Missing Sonarr series/episode ID -- try syncing the library first."
        elif title.media_type == MediaType.movie and not title.radarr_movie_id:
            message = "Missing Radarr movie ID -- try syncing the library first."
        elif title.media_type == MediaType.episode and not (app_settings.sonarr_url and app_settings.sonarr_api_key):
            message = "Sonarr isn't configured (missing URL/API key)."
        elif title.media_type == MediaType.movie and not (app_settings.radarr_url and app_settings.radarr_api_key):
            message = "Radarr isn't configured (missing URL/API key)."
        else:
            try:
                if title.media_type == MediaType.episode:
                    client = SonarrClient(app_settings.sonarr_url, app_settings.sonarr_api_key)
                    await client.trigger_episode_search(title.sonarr_episode_id)
                else:
                    client = RadarrClient(app_settings.radarr_url, app_settings.radarr_api_key)
                    await client.trigger_movie_search(title.radarr_movie_id)
            except Exception as exc:  # noqa: BLE001 -- surface any connection/API error to the UI
                message = f"Search request failed: {exc}"
            else:
                title.status = "awaiting_mkv"
                title.replacement_requested_at = datetime.datetime.utcnow()
                await session.commit()
                message = "Search triggered -- will auto-process once an .mkv release is grabbed."

        await session.refresh(title)
        row = _row_dict(title, current_version, bazarr_message=message, short_label=short_label)

    return templates.TemplateResponse(
        "partials/title_row.html", {"request": request, "row": row, "severity_options": SEVERITY_OPTIONS}
    )
