import asyncio
import datetime
import logging
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Select, case, func, select, update

from app.db.models import DetectedScene, MatchedCue, MediaType, ProcessingJob, SceneJob, Title, TriggerSource
from app.db.session import get_session, get_setting
from app.domain import (
    PRECISE_MODES,
    SceneJobKind,
    SceneReviewStatus,
    Severity,
    format_duration,
    is_mkv_path,
    parse_severity_levels,
    serialize_severity_levels,
    title_href,
)
from app.integrations.bazarr import BazarrClient
from app.integrations.radarr import RadarrClient
from app.integrations.sonarr import SonarrClient
from app.integrations.subtitle_lookup import find_subtitle_for_video
from app.library import (
    enqueue_if_not_already_active,
    is_outdated,
    poll_for_subtitle_then_enqueue,
    spawn_background,
    sync_radarr_library,
    sync_sonarr_library,
)
from app.queue.scene_worker import scene_job_queue
from app.queue.worker import job_queue

router = APIRouter(prefix="/library", tags=["library"])
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["title_href"] = title_href
logger = logging.getLogger(__name__)
templates.env.globals["format_duration"] = format_duration

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


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _is_load_more(request: Request) -> bool:
    """True only for the infinite-scroll sentinel's hx-get (it has no named form
    field, so htmx sends no HX-Trigger-Name). Search/filter inputs do have a name,
    so they fall through to a full-page render instead -- hx-select then pulls just
    the refreshed results region back out of it, keeping the title count in sync.

    A real bug this depends on: the pip-legend status-filter buttons
    (partials/pip_legend.html) were plain unnamed <button>s, so htmx sent no
    HX-Trigger-Name for them either -- indistinguishable from the sentinel,
    so clicking one wrongly hit this branch and got back a bare items partial
    with no #movie-results/#show-results wrapper for hx-select to find,
    swapping in nothing until a full page reload. Fixed by giving those
    buttons name="pip", matching every other filter control here."""
    return _is_htmx(request) and not request.headers.get("HX-Trigger-Name")


async def _pending_scene_title_ids(session, title_ids: list[int]) -> set[int]:
    """Which of title_ids have at least one DetectedScene still awaiting review.
    Batched (one query for however many ids are passed) rather than per-row, so
    list pages don't pay an N+1 cost -- callers rendering many rows should collect
    ids first and check membership in the returned set."""
    if not title_ids:
        return set()
    result = await session.execute(
        select(DetectedScene.title_id)
        .where(DetectedScene.title_id.in_(title_ids), DetectedScene.status == SceneReviewStatus.pending)
        .distinct()
    )
    return {row[0] for row in result.all()}


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


_PIP_LABELS = {
    "done": "done",
    "processing": "processing",
    "queued": "queued",
    "failed": "failed",
    "not_processed": "not processed",
    "outdated": "outdated word list",
    "no_subtitle": "no subtitle",
    "scenes_pending_review": "scenes pending review",
}


def _pip_state(status: str, outdated: bool, has_subtitle: bool, has_pending_scenes: bool = False) -> str:
    """Single-glance poster status, in priority order: an active/failed job always wins
    (most actionable), then pending scene-detection review (also actionable, and
    independent of the mute pipeline's own state), then an outdated or
    missing-subtitle title (both blockers on an otherwise-fine title), then plain
    done/not-processed."""
    if status == "failed":
        return "failed"
    if status == "processing":
        return "processing"
    if status in ("queued", "awaiting_mkv", "awaiting_subtitle"):
        return "queued"
    if has_pending_scenes:
        return "scenes_pending_review"
    if outdated:
        return "outdated"
    if not has_subtitle:
        return "no_subtitle"
    if status == "done":
        return "done"
    return "not_processed"


def _aggregate_show_pip(seasons: list[dict]) -> tuple[str, str]:
    """Collapse a show_detail.html season list down to the single representative
    pip/label its detail-header badge shows, same aggregation _load_shows already
    does per-card on the Shows grid, just summed across seasons instead of read
    straight off the grouped query."""
    pip = _show_pip_state(
        total=sum(s["total"] for s in seasons),
        done_count=sum(s["done_count"] for s in seasons),
        failed_count=sum(s["failed_count"] for s in seasons),
        active_count=sum(s["active_count"] for s in seasons),
        outdated_count=sum(s["outdated_count"] for s in seasons),
    )
    return pip, _PIP_LABELS[pip]


def _show_pip_state(total: int, done_count: int, failed_count: int, active_count: int, outdated_count: int) -> str:
    """Aggregate counterpart to _pip_state for a show card, which represents many
    episodes at once rather than one title -- same priority order (an active/failed
    episode always wins), collapsed to a single representative dot. No "queued" vs
    "processing" distinction (the underlying query counts them together) or
    "no_subtitle" state (not meaningful in aggregate) -- both fold into the closest
    single-title equivalent below."""
    if failed_count:
        return "failed"
    if active_count:
        return "processing"
    if outdated_count:
        return "outdated"
    if total and done_count == total:
        return "done"
    return "not_processed"


def _row_dict(
    title: Title,
    current_version: int,
    bazarr_message: str | None = None,
    short_label: bool = False,
    has_pending_scenes: bool = False,
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
    outdated = is_outdated(title, current_version)
    pip = _pip_state(title.status, outdated, has_subtitle, has_pending_scenes)

    return {
        "title": title,
        "display_name": display_name,
        "status": title.status,
        "outdated": outdated,
        "bazarr_message": bazarr_message,
        "is_mkv": is_mkv,
        "severity_error": severity_error,
        "has_subtitle": has_subtitle,
        "has_pending_scenes": has_pending_scenes,
        "pip": pip,
        "pip_label": _PIP_LABELS[pip],
    }


async def _load_last_job(title_id: int) -> tuple[ProcessingJob | None, str | None]:
    async with get_session() as session:
        last_job = (
            await session.execute(
                select(ProcessingJob)
                .where(ProcessingJob.title_id == title_id)
                .order_by(ProcessingJob.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    last_job_duration = None
    if last_job is not None and last_job.started_at and last_job.finished_at:
        last_job_duration = format_duration((last_job.finished_at - last_job.started_at).total_seconds())
    return last_job, last_job_duration


async def _load_last_scan_job(title_id: int) -> tuple[SceneJob | None, str | None]:
    """Video-side counterpart to _load_last_job -- the scan (not blur/Apply) is
    the direct analog of audio's ProcessingJob for "Last processed"/"Time to
    finish" purposes, since it's the step that produces the Scenes found count,
    the same way ProcessingJob produces the Muted cues count. Unlike audio,
    there's no dedicated Title column for this -- SceneJob's own timestamps are
    the only record, so this queries them directly rather than a cached field."""
    async with get_session() as session:
        last_job = (
            await session.execute(
                select(SceneJob)
                .where(SceneJob.title_id == title_id, SceneJob.kind == SceneJobKind.scan)
                .order_by(SceneJob.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    last_job_duration = None
    if last_job is not None and last_job.started_at and last_job.finished_at:
        last_job_duration = format_duration((last_job.finished_at - last_job.started_at).total_seconds())
    return last_job, last_job_duration


async def _load_matched_cues(title_id: int) -> list[MatchedCue]:
    async with get_session() as session:
        return list(
            (
                await session.execute(
                    select(MatchedCue).where(MatchedCue.title_id == title_id).order_by(MatchedCue.start_seconds)
                )
            )
            .scalars()
            .all()
        )


async def _load_scene_review_context(title_id: int) -> dict:
    """Thin wrapper around scenes.py's _scene_review_context -- that's the complete
    version (also fetches scan_job/blur_job/claude_verify_job, needed to render an
    in-progress scan/apply/Claude-verify job's progress bar correctly), this one used
    to be a stale, incomplete duplicate that left every caller here (the title detail
    page's own GET route, plus every detail_view re-render after severity/process/etc.)
    unable to show that progress bar until the embedded scene-review section's own
    3-second self-poll caught up -- single source of truth now instead of two drifting
    copies of the same query."""
    from app.routers.scenes import _scene_review_context

    async with get_session() as session:
        return await _scene_review_context(session, title_id)


async def _render_title(request: Request, row: dict, detail_view: bool, **extra):
    """Single-title action routes are called both from a table row (title_row.html)
    and from the title detail page (title_detail_card.html) -- render whichever one
    the caller came from, identified by the hidden detail_view form field each of
    that page's forms carries."""
    if detail_view:
        last_job, last_job_duration = await _load_last_job(row["title"].id)
        last_scan_job, last_scan_job_duration = await _load_last_scan_job(row["title"].id)
        matched_cues = await _load_matched_cues(row["title"].id)
        scene_context = await _load_scene_review_context(row["title"].id)
        return templates.TemplateResponse(
            "partials/title_detail_card.html",
            {
                "request": request,
                "row": row,
                "last_job": last_job,
                "last_job_duration": last_job_duration,
                "last_scan_job": last_scan_job,
                "last_scan_job_duration": last_scan_job_duration,
                "matched_cues": matched_cues,
                **scene_context,
                **extra,
            },
        )
    return templates.TemplateResponse(
        "partials/title_row.html", {"request": request, "row": row, "severity_options": SEVERITY_OPTIONS, **extra}
    )


MOVIE_SORT_COLUMNS = {
    "title": Title.display_name.collate("NOCASE"),
    "status": Title.status,
    "subtitle": Title.subtitle_path,
    "muted_cues": Title.matched_cue_count,
    "severity": Title.severity_levels,
}


async def _used_pips_titles(session, base: Select, current_version: int) -> set[str]:
    """Which of pip_legend.html's status keys actually occur among titles matching
    `base` (a page's own scope query, minus any pip filter itself) -- lets the
    legend hide filter buttons that would return nothing.

    Uses `subtitle_path is not None` as a cheap stand-in for has_subtitle,
    deliberately skipping _row_dict's real Path.exists() filesystem check --
    running that over every title in scope, on every page load, just to
    populate a legend would mean an NFS stat() per title regardless of
    whether a pip filter is even in use (today it only pays that cost when
    one is). A title whose recorded subtitle has since been deleted or moved
    might not register as "no_subtitle" here; an acceptable, rare inaccuracy
    for a legend's own visibility, not a real per-row correctness issue."""
    result = await session.execute(
        base.with_only_columns(Title.id, Title.status, Title.subtitle_path, Title.last_processed_wordlist_version)
    )
    rows = result.all()
    pending_ids = await _pending_scene_title_ids(session, [r.id for r in rows])
    used: set[str] = set()
    for r in rows:
        outdated = r.last_processed_wordlist_version is not None and r.last_processed_wordlist_version < current_version
        used.add(_pip_state(r.status, outdated, has_subtitle=bool(r.subtitle_path), has_pending_scenes=r.id in pending_ids))
    return used


async def _used_pips_shows(session, query: Select) -> set[str]:
    """Show-aggregate counterpart to _used_pips_titles -- `query` is the same
    grouped total/done/failed/active/outdated column query _load_shows/
    _load_processed_shows already build for their own pagination (any HAVING
    already applied), just run unpaginated here instead of offset/limited."""
    result = await session.execute(query)
    used: set[str] = set()
    for row in result.all():
        used.add(_show_pip_state(row.total, row.done_count, row.failed_count, row.active_count, row.outdated_count))
    return used


async def _load_movies(
    session, current_version: int, page: int, q: str | None, sort: str | None, sort_dir: str, pip: str | None = None
) -> tuple[list[dict], int, set[str]]:
    base: Select = select(Title).where(Title.media_type == MediaType.movie)
    if q:
        base = base.where(Title.display_name.like(f"%{_escape_like(q)}%", escape="\\"))

    if sort == "recent":
        order = Title.last_processed_at.desc()
    else:
        sort_col = MOVIE_SORT_COLUMNS.get(sort, Title.display_name)
        order = sort_col.desc() if sort_dir == "desc" else sort_col.asc()

    used_pips = await _used_pips_titles(session, base, current_version)

    if pip:
        # pip is derived per-row (has_subtitle alone needs a real filesystem check --
        # see _row_dict), not a plain column, so there's no SQL WHERE for it: fetch
        # every matching title, compute pip in Python, filter, then paginate the
        # filtered list ourselves instead of pushing offset/limit to the query.
        result = await session.execute(base.order_by(order, Title.display_name.collate("NOCASE")))
        titles = result.scalars().all()
        pending_ids = await _pending_scene_title_ids(session, [t.id for t in titles])
        rows = [_row_dict(t, current_version, has_pending_scenes=t.id in pending_ids) for t in titles]
        rows = [r for r in rows if r["pip"] == pip]
        total = len(rows)
        start = (page - 1) * PAGE_SIZE
        return rows[start : start + PAGE_SIZE], total, used_pips

    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    result = await session.execute(
        base.order_by(order, Title.display_name.collate("NOCASE")).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
    )
    titles = result.scalars().all()
    pending_ids = await _pending_scene_title_ids(session, [t.id for t in titles])
    rows = [_row_dict(t, current_version, has_pending_scenes=t.id in pending_ids) for t in titles]
    return rows, total, used_pips


async def _load_processed_movies(
    session, current_version: int, offset: int, limit: int, q: str | None, sort: str = "name", pip: str | None = None
) -> tuple[list[dict], int, set[str]]:
    """Done movies, one card each -- half of the main /library route's combined
    feed (the other half is _load_processed_shows below). Takes an explicit
    offset/limit rather than a page number since the two feeds are interleaved
    into one virtual list by the caller (see library_page).

    While actively searching (q set), the done-only filter is dropped so the
    search reaches every movie regardless of processing status -- otherwise
    a title that hasn't been processed yet is invisible from this page even
    though the user typed its exact name. Same reasoning applies to a pip
    filter: e.g. "not processed" or "failed" would otherwise never match
    anything here."""
    unrestricted_base: Select = select(Title).where(Title.media_type == MediaType.movie)
    if q:
        unrestricted_base = unrestricted_base.where(Title.display_name.like(f"%{_escape_like(q)}%", escape="\\"))
    base = unrestricted_base
    if not q and not pip:
        base = base.where(Title.status == "done")

    order = Title.last_processed_at.desc() if sort == "recent" else Title.display_name.collate("NOCASE")

    # Always computed against unrestricted_base (the scope a pip click would
    # actually search, per the done-only-filter-drops-once-clicked reasoning
    # above), not the possibly done-only `base` -- otherwise a legend entry
    # for e.g. "not processed" would never appear even though clicking it
    # would find results.
    used_pips = await _used_pips_titles(session, unrestricted_base, current_version)

    if pip:
        result = await session.execute(base.order_by(order))
        titles = result.scalars().all()
        pending_ids = await _pending_scene_title_ids(session, [t.id for t in titles])
        rows = [_row_dict(t, current_version, has_pending_scenes=t.id in pending_ids) for t in titles]
        rows = [r for r in rows if r["pip"] == pip]
        total = len(rows)
        return rows[offset : offset + limit], total, used_pips

    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    result = await session.execute(base.order_by(order).offset(offset).limit(limit))
    titles = result.scalars().all()
    pending_ids = await _pending_scene_title_ids(session, [t.id for t in titles])
    rows = [_row_dict(t, current_version, has_pending_scenes=t.id in pending_ids) for t in titles]
    return rows, total, used_pips


async def _load_processed_shows(
    session, current_version: int, offset: int, limit: int, q: str | None, sort: str = "name", pip: str | None = None
) -> tuple[list[dict], int, set[str]]:
    """Shows with at least one done episode, one card each (not one per episode) --
    other half of the main /library route's combined feed. Same aggregation as
    _load_shows (TV Shows page) but restricted to done_count > 0.

    While actively searching (q set), that done_count > 0 restriction is dropped
    so the search reaches every show regardless of processing status -- see
    _load_processed_movies above for the same reasoning on the movie side."""
    total_col = func.count().label("total")
    done_col = func.sum(case((Title.status == "done", 1), else_=0)).label("done_count")
    failed_col = func.sum(case((Title.status == "failed", 1), else_=0)).label("failed_count")
    active_col = func.sum(case((Title.status.in_(["queued", "processing"]), 1), else_=0)).label("active_count")
    outdated_col = func.sum(_outdated_case(current_version)).label("outdated_count")
    poster_col = func.max(Title.poster_url).label("poster_url")
    last_processed_col = func.max(Title.last_processed_at).label("last_processed_at")

    unrestricted_query = select(
        Title.sonarr_series_id,
        Title.series_title,
        total_col,
        done_col,
        failed_col,
        active_col,
        outdated_col,
        poster_col,
        last_processed_col,
    ).where(Title.media_type == MediaType.episode)
    if q:
        unrestricted_query = unrestricted_query.where(Title.series_title.like(f"%{_escape_like(q)}%", escape="\\"))
    unrestricted_query = unrestricted_query.group_by(Title.sonarr_series_id, Title.series_title)
    query = unrestricted_query
    if not q and not pip:
        query = query.having(done_col > 0)

    order = last_processed_col.desc() if sort == "recent" else Title.series_title.collate("NOCASE")

    # Same reasoning as _load_processed_movies' used_pips: always computed
    # against the unrestricted query (what a pip click would actually
    # search), not the possibly done_count>0-restricted `query`.
    used_pips = await _used_pips_shows(session, unrestricted_query)

    if pip:
        result = await session.execute(query.order_by(order))
        shows = [dict(row._mapping) for row in result.all()]
        for show in shows:
            show["pip"] = _show_pip_state(
                show["total"], show["done_count"], show["failed_count"], show["active_count"], show["outdated_count"]
            )
            show["pip_label"] = _PIP_LABELS[show["pip"]]
        shows = [s for s in shows if s["pip"] == pip]
        total = len(shows)
        return shows[offset : offset + limit], total, used_pips

    count_subquery = query.with_only_columns(Title.sonarr_series_id).subquery()
    total = (await session.execute(select(func.count()).select_from(count_subquery))).scalar_one()
    result = await session.execute(query.order_by(order).offset(offset).limit(limit))
    shows = [dict(row._mapping) for row in result.all()]
    for show in shows:
        show["pip"] = _show_pip_state(
            show["total"], show["done_count"], show["failed_count"], show["active_count"], show["outdated_count"]
        )
        show["pip_label"] = _PIP_LABELS[show["pip"]]
    return shows, total, used_pips


SHOW_SORT_KEYS = {"name", "total", "done_count", "active_count", "failed_count", "outdated_count"}


async def _load_shows(
    session, current_version: int, page: int, q: str | None, sort: str | None, sort_dir: str, pip: str | None = None
) -> tuple[list[dict], int, set[str]]:
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
    precise_col = func.min(Title.precise_mode).label("precise_mode")
    poster_col = func.max(Title.poster_url).label("poster_url")
    last_processed_col = func.max(Title.last_processed_at).label("last_processed_at")

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
        poster_col,
        last_processed_col,
    ).where(Title.media_type == MediaType.episode)

    if q:
        query = query.where(Title.series_title.like(f"%{_escape_like(q)}%", escape="\\"))

    query = query.group_by(Title.sonarr_series_id, Title.series_title)

    if sort == "recent":
        order = last_processed_col.desc()
    else:
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

    used_pips = await _used_pips_shows(session, query)

    if pip:
        # Every value _show_pip_state needs is already an aggregate column, so no
        # filesystem check is required here (unlike the movie side) -- still can't
        # push the filter into SQL though, since it's derived (priority-ordered
        # across 5 aggregate columns), not a plain one. Fetch every group, compute,
        # filter, then paginate the filtered list ourselves.
        result = await session.execute(query.order_by(order, Title.series_title.collate("NOCASE")))
        shows = [dict(row._mapping) for row in result.all()]
        for show in shows:
            show["pip"] = _show_pip_state(
                show["total"], show["done_count"], show["failed_count"], show["active_count"], show["outdated_count"]
            )
            show["pip_label"] = _PIP_LABELS[show["pip"]]
        shows = [s for s in shows if s["pip"] == pip]
        total = len(shows)
        start = (page - 1) * PAGE_SIZE
        return shows[start : start + PAGE_SIZE], total, used_pips

    count_subquery = query.with_only_columns(Title.sonarr_series_id).subquery()
    total = (await session.execute(select(func.count()).select_from(count_subquery))).scalar_one()

    result = await session.execute(
        query.order_by(order, Title.series_title.collate("NOCASE")).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
    )
    shows = [dict(row._mapping) for row in result.all()]
    for show in shows:
        show["pip"] = _show_pip_state(
            show["total"], show["done_count"], show["failed_count"], show["active_count"], show["outdated_count"]
        )
        show["pip_label"] = _PIP_LABELS[show["pip"]]
    return shows, total, used_pips


async def _load_seasons(
    session, series_id: int, current_version: int, came_from: str | None = None
) -> tuple[str | None, str | None, str | None, str | None, str | None, list[dict]]:
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
            func.min(Title.precise_mode).label("precise_mode"),
        )
        .where(Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
        .group_by(Title.season_number)
        .order_by(Title.season_number)
    )
    result = await session.execute(query)
    seasons = [dict(row._mapping) for row in result.all()]
    if came_from == "library":
        # Reached from the Library page's "done only" view -- seasons with nothing
        # processed yet aren't relevant there, same idea as filtering episodes below
        # in _load_season_detail_context.
        seasons = [s for s in seasons if s["done_count"]]
    for season in seasons:
        season["gating_message"] = await _non_mkv_gating_message(
            session,
            Title.sonarr_series_id == series_id,
            Title.season_number == season["season_number"],
            levels=season["severity_levels"] or "",
        )

    meta_result = await session.execute(
        select(
            func.max(Title.series_title),
            func.max(Title.poster_url),
            # Same "representative, not a true aggregate" caveat as the per-season
            # values above, one level up -- a show-wide bulk edit (see set_show_severity/
            # set_show_precise_mode) overwrites every episode in every season with
            # whatever's picked here, same as the per-season bulk edit already does.
            func.min(Title.severity_levels),
            func.min(Title.precise_mode),
        ).where(Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
    )
    series_title, poster_url, severity_levels, precise_mode = meta_result.one()
    gating_message = await _non_mkv_gating_message(
        session, Title.sonarr_series_id == series_id, levels=severity_levels or ""
    )
    return series_title, poster_url, severity_levels, precise_mode, gating_message, seasons


@router.get("", response_class=HTMLResponse)
async def library_page(
    request: Request, q: str | None = None, page: int = 1, sort: str = "name", pip: str | None = None
):
    """The main sidebar "Library" item -- every done movie (one card each) plus
    every show with at least one done episode (one card per show, not per
    episode). Distinct from the "Movies"/"TV Shows" sub-items, which list
    everything regardless of processing status. The two queries are paginated
    independently, then interleaved into one virtual list (movies first, then
    shows) via the offset math below, so the single infinite-scroll feed here
    doesn't need a real SQL UNION across two very different row shapes.

    When a search query is active, the done-only restriction is dropped (see
    _load_processed_movies/_load_processed_shows) so searching from this page
    reaches every movie and show regardless of processing status, not just
    ones that have already finished."""
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        _, movie_total, movie_used_pips = await _load_processed_movies(session, current_version, 0, 0, q, sort, pip)
        _, show_total, show_used_pips = await _load_processed_shows(session, current_version, 0, 0, q, sort, pip)
        used_pips = movie_used_pips | show_used_pips

        offset = (page - 1) * PAGE_SIZE
        movie_rows: list[dict] = []
        show_rows: list[dict] = []
        if offset < movie_total:
            movie_rows, _, _ = await _load_processed_movies(session, current_version, offset, PAGE_SIZE, q, sort, pip)
            remaining = PAGE_SIZE - len(movie_rows)
            if remaining > 0:
                show_rows, _, _ = await _load_processed_shows(session, current_version, 0, remaining, q, sort, pip)
        else:
            show_rows, _, _ = await _load_processed_shows(
                session, current_version, offset - movie_total, PAGE_SIZE, q, sort, pip
            )

    combined_total = movie_total + show_total
    current_params = {"q": q, "sort": sort, "pip": pip}
    next_url = _qs(current_params, page=page + 1) if offset + PAGE_SIZE < combined_total else None

    if _is_load_more(request):
        # A revealed load-more sentinel asking for the next batch -- return just
        # the new cards (+ a fresh sentinel if there's still more), not the whole page.
        return templates.TemplateResponse(
            "partials/poster_grid_library_items.html",
            {"request": request, "movie_rows": movie_rows, "show_rows": show_rows, "next_url": next_url},
        )

    def pip_link(key: str) -> str:
        return _qs(current_params, pip=None if pip == key else key, page=1)

    return templates.TemplateResponse(
        "library_processed.html",
        {
            "request": request,
            "q": q or "",
            "sort": sort,
            "pip": pip,
            "pip_link": pip_link,
            "used_pips": used_pips,
            "movie_rows": movie_rows,
            "show_rows": show_rows,
            "movie_total": combined_total,
            "next_url": next_url,
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
    movie_dir: str | None = None,
    pip: str | None = None,
):
    movie_dir = movie_dir or "asc"
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        movie_rows, movie_total, used_pips = await _load_movies(
            session, current_version, movie_page, q, movie_sort, movie_dir, pip
        )

    current_params = {"q": q, "movie_sort": movie_sort, "movie_dir": movie_dir, "pip": pip}
    next_url = _qs(current_params, movie_page=movie_page + 1) if movie_page * PAGE_SIZE < movie_total else None

    if _is_load_more(request):
        return templates.TemplateResponse(
            "partials/poster_grid_items.html",
            {"request": request, "rows": movie_rows, "next_url": next_url},
        )

    def movie_sort_link(key: str) -> str:
        new_dir = "desc" if movie_sort == key and movie_dir == "asc" else "asc"
        return _qs(current_params, movie_sort=key, movie_dir=new_dir, movie_page=1)

    def pip_link(key: str) -> str:
        return _qs(current_params, pip=None if pip == key else key, movie_page=1)

    return templates.TemplateResponse(
        "library_movies.html",
        {
            "request": request,
            "q": q or "",
            "pip": pip,
            "pip_link": pip_link,
            "used_pips": used_pips,
            "movie_rows": movie_rows,
            "movie_total": movie_total,
            "movie_sort": movie_sort,
            "movie_dir": movie_dir,
            "movie_sort_link": movie_sort_link,
            "next_url": next_url,
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
    show_dir: str | None = None,
    pip: str | None = None,
):
    show_dir = show_dir or "asc"
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        shows, show_total, used_pips = await _load_shows(session, current_version, show_page, q, show_sort, show_dir, pip)

    current_params = {"q": q, "show_sort": show_sort, "show_dir": show_dir, "pip": pip}
    next_url = _qs(current_params, show_page=show_page + 1) if show_page * PAGE_SIZE < show_total else None

    if _is_load_more(request):
        return templates.TemplateResponse(
            "partials/poster_grid_shows_items.html",
            {"request": request, "shows": shows, "next_url": next_url},
        )

    def show_sort_link(key: str) -> str:
        new_dir = "desc" if show_sort == key and show_dir == "asc" else "asc"
        return _qs(current_params, show_sort=key, show_dir=new_dir, show_page=1)

    def pip_link(key: str) -> str:
        return _qs(current_params, pip=None if pip == key else key, show_page=1)

    return templates.TemplateResponse(
        "library_shows.html",
        {
            "request": request,
            "q": q or "",
            "pip": pip,
            "pip_link": pip_link,
            "used_pips": used_pips,
            "shows": shows,
            "show_total": show_total,
            "show_sort": show_sort,
            "show_dir": show_dir,
            "show_sort_link": show_sort_link,
            "next_url": next_url,
            "wordlist_version": current_version,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


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
            func.min(Title.precise_mode).label("precise_mode"),
            func.max(Title.poster_url).label("poster_url"),
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


async def _load_season_detail_context(
    session, series_id: int, season_number: int, current_version: int, came_from: str | None = None
) -> dict:
    """Everything season_detail.html needs -- shared by its own GET route and by
    the season-level bulk actions below when called from that page's own header
    (detail_view=true) rather than from a season row inside show_detail.html."""
    name_result = await session.execute(
        select(Title.series_title).where(Title.sonarr_series_id == series_id).limit(1)
    )
    series_title = name_result.scalar_one_or_none()

    result = await session.execute(
        select(Title)
        .where(Title.sonarr_series_id == series_id, Title.season_number == season_number)
        .order_by(Title.episode_number)
    )
    titles = result.scalars().all()
    pending_ids = await _pending_scene_title_ids(session, [t.id for t in titles])
    rows = [_row_dict(t, current_version, short_label=True, has_pending_scenes=t.id in pending_ids) for t in titles]
    if came_from == "library":
        # Reached from the Library page's "done only" view -- episodes that haven't
        # been processed yet aren't relevant there.
        rows = [r for r in rows if r["status"] != "not_processed"]

    season = await _load_season(session, series_id, season_number, current_version)
    pip, pip_label = _aggregate_show_pip([season] if season else [])

    return {
        "series_title": series_title,
        "rows": rows,
        "poster_url": season["poster_url"] if season else None,
        "pip": pip,
        "pip_label": pip_label,
        "severity_levels": season["severity_levels"] if season else "",
        "precise_mode": season["precise_mode"] if season else "whole_line",
        "gating_message": season["gating_message"] if season else None,
        "came_from": came_from,
    }


@router.get("/title/{title_id}", response_class=HTMLResponse)
async def title_detail(request: Request, title_id: int, from_: str | None = Query(default=None, alias="from")):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            raise HTTPException(status_code=404, detail="Title not found")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, has_pending_scenes=title_id in pending_ids)
    last_job, last_job_duration = await _load_last_job(title_id)
    last_scan_job, last_scan_job_duration = await _load_last_scan_job(title_id)
    matched_cues = await _load_matched_cues(title_id)
    scene_context = await _load_scene_review_context(title_id)

    return templates.TemplateResponse(
        "title_detail.html",
        {
            "request": request,
            "row": row,
            "came_from": from_,
            "last_job": last_job,
            "last_job_duration": last_job_duration,
            "last_scan_job": last_scan_job,
            "last_scan_job_duration": last_scan_job_duration,
            "matched_cues": matched_cues,
            **scene_context,
        },
    )


@router.get("/{title_id}/row", response_class=HTMLResponse)
async def title_row_refresh(request: Request, title_id: int, season_context: bool = False):
    """Self-polling target for a single title_row.html <tr> (see its hx-get) -- lets a
    row still in the movies/season table pick up a status change (e.g. queued ->
    processing) without the user reloading the whole page."""
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, short_label=season_context, has_pending_scenes=title_id in pending_ids)

    return templates.TemplateResponse(
        "partials/title_row.html",
        {
            "request": request,
            "row": row,
            "severity_options": SEVERITY_OPTIONS,
            "season_context": season_context,
        },
    )


@router.get("/{title_id}/card", response_class=HTMLResponse)
async def title_card_refresh(request: Request, title_id: int, from_page: str | None = None):
    """Self-polling target for a single poster_card.html (see its hx-get) -- same idea
    as title_row_refresh above, for the poster-grid views (Movies, Library)."""
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, has_pending_scenes=title_id in pending_ids)

    return templates.TemplateResponse(
        "partials/poster_card.html", {"request": request, "row": row, "from_page": from_page}
    )


@router.get("/shows/{series_id}", response_class=HTMLResponse)
async def show_detail(request: Request, series_id: int, from_: str | None = Query(default=None, alias="from")):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, poster_url, severity_levels, precise_mode, gating_message, seasons = await _load_seasons(
            session, series_id, current_version, came_from=from_
        )
    pip, pip_label = _aggregate_show_pip(seasons)

    return templates.TemplateResponse(
        "show_detail.html",
        {
            "request": request,
            "series_id": series_id,
            "came_from": from_,
            "series_title": series_title,
            "poster_url": poster_url,
            "severity_levels": severity_levels,
            "precise_mode": precise_mode,
            "gating_message": gating_message,
            "pip": pip,
            "pip_label": pip_label,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.get("/shows/{series_id}/season/{season_number}", response_class=HTMLResponse)
async def season_detail(
    request: Request,
    series_id: int,
    season_number: int,
    from_: str | None = Query(default=None, alias="from"),
):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        context = await _load_season_detail_context(session, series_id, season_number, current_version, came_from=from_)

    return templates.TemplateResponse(
        "season_detail.html",
        {
            "request": request,
            "series_id": series_id,
            "season_number": season_number,
            "severity_options": SEVERITY_OPTIONS,
            **context,
        },
    )


@router.post("/sync")
async def sync_library(request: Request):
    # Sonarr and Radarr are synced independently -- one failing (unreachable,
    # unexpected response shape) previously raised unhandled from inside the
    # session and aborted the whole route, so Radarr's sync never even ran
    # since Sonarr's happens first. Each is now caught and logged on its own,
    # so a Sonarr outage doesn't also silently skip Radarr.
    sync_errors: list[str] = []
    async with get_session() as session:
        sonarr_url = await get_setting(session, "sonarr_url")
        sonarr_api_key = await get_setting(session, "sonarr_api_key")
        radarr_url = await get_setting(session, "radarr_url")
        radarr_api_key = await get_setting(session, "radarr_api_key")
        if sonarr_url and sonarr_api_key:
            try:
                client = SonarrClient(sonarr_url, sonarr_api_key)
                await sync_sonarr_library(session, client)
            except Exception:  # noqa: BLE001 -- log and continue to Radarr regardless
                logger.exception("Sonarr sync failed")
                sync_errors.append("sonarr")
                # A caught-but-not-rolled-back exception can leave the session's
                # transaction in a state SQLAlchemy refuses to reuse (e.g. after
                # a flush failure) -- without this, Radarr's sync below (same
                # session) could itself immediately fail with a confusing
                # PendingRollbackError misattributed to Radarr.
                await session.rollback()
        if radarr_url and radarr_api_key:
            try:
                client = RadarrClient(radarr_url, radarr_api_key)
                await sync_radarr_library(session, client)
            except Exception:  # noqa: BLE001 -- log, still return normally
                logger.exception("Radarr sync failed")
                sync_errors.append("radarr")
                await session.rollback()

    redirect_url = "/settings"
    if sync_errors:
        redirect_url += "?" + urlencode({"sync_error": ",".join(sync_errors)})
    return RedirectResponse(url=redirect_url, status_code=303)


@router.post("/{title_id}/process", response_class=HTMLResponse)
async def process_title(
    request: Request, title_id: int, short_label: bool = Form(False), detail_view: bool = Form(False)
):
    # Verify the title still exists before enqueueing -- enqueue() unconditionally
    # inserts a ProcessingJob row even for a nonexistent title_id (e.g. deleted
    # between page load and click), leaving a dangling FK the worker later has to
    # notice and mark failed, with this route meanwhile returning an empty
    # response as if the click did nothing.
    async with get_session() as session:
        if await session.get(Title, title_id) is None:
            return HTMLResponse("")

    await job_queue.enqueue(title_id, TriggerSource.manual)

    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, short_label=short_label, has_pending_scenes=title_id in pending_ids)

    return await _render_title(request, row, detail_view)


@router.post("/{title_id}/process-video", response_class=HTMLResponse)
async def process_video_title(
    request: Request, title_id: int, short_label: bool = Form(False), detail_view: bool = Form(False)
):
    """Video-side counterpart to /process (audio-only, word-list mute) -- scans
    for candidate scenes without touching the audio pipeline."""
    async with get_session() as session:
        if await session.get(Title, title_id) is None:
            return HTMLResponse("")

    await scene_job_queue.enqueue_scan_if_not_already_active(title_id)

    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, short_label=short_label, has_pending_scenes=title_id in pending_ids)

    return await _render_title(request, row, detail_view)


@router.post("/{title_id}/process-both", response_class=HTMLResponse)
async def process_both_title(
    request: Request, title_id: int, short_label: bool = Form(False), detail_view: bool = Form(False)
):
    """Audio + Video together, one click -- both jobs always enqueue."""
    async with get_session() as session:
        if await session.get(Title, title_id) is None:
            return HTMLResponse("")

    await job_queue.enqueue(title_id, TriggerSource.manual)
    await scene_job_queue.enqueue_scan_if_not_already_active(title_id)

    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, short_label=short_label, has_pending_scenes=title_id in pending_ids)

    return await _render_title(request, row, detail_view)


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
        message += f" Skipped {skipped} with no subtitle -- use the Audio/Both button on those individually."
    return message


@router.post("/shows/{series_id}/process", response_class=HTMLResponse)
async def process_show(request: Request, series_id: int, came_from: str | None = Form(default=None)):
    conditions = (Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
    async with get_session() as session:
        queued, skipped = await _bulk_enqueue_titles(session, *conditions)
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, poster_url, severity_levels, precise_mode, gating_message, seasons = await _load_seasons(
            session, series_id, current_version, came_from=came_from
        )
    pip, pip_label = _aggregate_show_pip(seasons)

    return templates.TemplateResponse(
        "partials/show_detail_card.html",
        {
            "request": request,
            "series_id": series_id,
            "came_from": came_from,
            "series_title": series_title,
            "poster_url": poster_url,
            "severity_levels": severity_levels,
            "precise_mode": precise_mode,
            "gating_message": gating_message,
            "pip": pip,
            "pip_label": pip_label,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
            "process_message": _process_message(queued, skipped),
        },
    )


@router.post("/shows/{series_id}/process-both", response_class=HTMLResponse)
async def process_both_show(request: Request, series_id: int, came_from: str | None = Form(default=None)):
    """Audio + Video together for every episode in the show, one click --
    mirrors process_show (audio) and app.routers.scenes.scan_scenes_show
    (video) combined."""
    conditions = (Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
    async with get_session() as session:
        queued, skipped = await _bulk_enqueue_titles(session, *conditions)
        result = await session.execute(select(Title.id).where(*conditions))
        title_ids = [row[0] for row in result.all()]
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, poster_url, severity_levels, precise_mode, gating_message, seasons = await _load_seasons(
            session, series_id, current_version, came_from=came_from
        )
    pip, pip_label = _aggregate_show_pip(seasons)

    scan_queued = await scene_job_queue.enqueue_scan_for_titles_if_not_already_active(title_ids)

    return templates.TemplateResponse(
        "partials/show_detail_card.html",
        {
            "request": request,
            "series_id": series_id,
            "came_from": came_from,
            "series_title": series_title,
            "poster_url": poster_url,
            "severity_levels": severity_levels,
            "precise_mode": precise_mode,
            "gating_message": gating_message,
            "pip": pip,
            "pip_label": pip_label,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
            "process_message": _process_message(queued, skipped),
            "scan_message": f"Queued {scan_queued} episode(s) for scene scanning.",
        },
    )


@router.post("/shows/{series_id}/severity", response_class=HTMLResponse)
async def set_show_severity(
    request: Request,
    series_id: int,
    sev_child: bool = Form(False),
    sev_teen: bool = Form(False),
    came_from: str | None = Form(default=None),
):
    levels = _severity_levels_from_checkboxes(sev_child, sev_teen)
    conditions = (Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
    async with get_session() as session:
        await session.execute(update(Title).where(*conditions).values(severity_levels=levels))
        await session.commit()
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, poster_url, severity_levels, precise_mode, gating_message, seasons = await _load_seasons(
            session, series_id, current_version, came_from=came_from
        )
    pip, pip_label = _aggregate_show_pip(seasons)

    return templates.TemplateResponse(
        "partials/show_detail_card.html",
        {
            "request": request,
            "series_id": series_id,
            "came_from": came_from,
            "series_title": series_title,
            "poster_url": poster_url,
            "severity_levels": severity_levels,
            "precise_mode": precise_mode,
            "gating_message": gating_message,
            "pip": pip,
            "pip_label": pip_label,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.post("/shows/{series_id}/set-precise-mode", response_class=HTMLResponse)
async def set_show_precise_mode(
    request: Request, series_id: int, precise_mode: str = Form(...), came_from: str | None = Form(default=None)
):
    async with get_session() as session:
        if precise_mode in PRECISE_MODES:
            await session.execute(
                update(Title)
                .where(Title.sonarr_series_id == series_id, Title.media_type == MediaType.episode)
                .values(precise_mode=precise_mode)
            )
            await session.commit()
        current_version = int(await get_setting(session, "wordlist_version"))
        series_title, poster_url, severity_levels, precise_mode, gating_message, seasons = await _load_seasons(
            session, series_id, current_version, came_from=came_from
        )
    pip, pip_label = _aggregate_show_pip(seasons)

    return templates.TemplateResponse(
        "partials/show_detail_card.html",
        {
            "request": request,
            "series_id": series_id,
            "came_from": came_from,
            "series_title": series_title,
            "poster_url": poster_url,
            "severity_levels": severity_levels,
            "precise_mode": precise_mode,
            "gating_message": gating_message,
            "pip": pip,
            "pip_label": pip_label,
            "seasons": seasons,
            "severity_options": SEVERITY_OPTIONS,
        },
    )


@router.post("/shows/{series_id}/season/{season_number}/process", response_class=HTMLResponse)
async def process_season(
    request: Request,
    series_id: int,
    season_number: int,
    detail_view: bool = Form(False),
    came_from: str | None = Form(default=None),
):
    conditions = (Title.sonarr_series_id == series_id, Title.season_number == season_number)
    async with get_session() as session:
        queued, skipped = await _bulk_enqueue_titles(session, *conditions)
        current_version = int(await get_setting(session, "wordlist_version"))
        if detail_view:
            context = await _load_season_detail_context(
                session, series_id, season_number, current_version, came_from=came_from
            )
        else:
            season = await _load_season(session, series_id, season_number, current_version)

    if detail_view:
        return templates.TemplateResponse(
            "partials/season_detail_card.html",
            {
                "request": request,
                "series_id": series_id,
                "season_number": season_number,
                "severity_options": SEVERITY_OPTIONS,
                "process_message": _process_message(queued, skipped),
                **context,
            },
        )
    return templates.TemplateResponse(
        "partials/season_row.html",
        {
            "request": request,
            "series_id": series_id,
            "season": season,
            "process_message": _process_message(queued, skipped),
        },
    )


@router.post("/shows/{series_id}/season/{season_number}/process-both", response_class=HTMLResponse)
async def process_both_season(
    request: Request,
    series_id: int,
    season_number: int,
    detail_view: bool = Form(False),
    came_from: str | None = Form(default=None),
):
    """Audio + Video together for every episode in the season, one click --
    mirrors process_season (audio) and app.routers.scenes.scan_scenes_season
    (video) combined."""
    conditions = (Title.sonarr_series_id == series_id, Title.season_number == season_number)
    async with get_session() as session:
        queued, skipped = await _bulk_enqueue_titles(session, *conditions)
        result = await session.execute(select(Title.id).where(*conditions))
        title_ids = [row[0] for row in result.all()]
        current_version = int(await get_setting(session, "wordlist_version"))
        if detail_view:
            context = await _load_season_detail_context(
                session, series_id, season_number, current_version, came_from=came_from
            )
        else:
            season = await _load_season(session, series_id, season_number, current_version)

    scan_queued = await scene_job_queue.enqueue_scan_for_titles_if_not_already_active(title_ids)

    if detail_view:
        return templates.TemplateResponse(
            "partials/season_detail_card.html",
            {
                "request": request,
                "series_id": series_id,
                "season_number": season_number,
                "severity_options": SEVERITY_OPTIONS,
                "process_message": _process_message(queued, skipped),
                "scan_message": f"Queued {scan_queued} episode(s) for scene scanning.",
                **context,
            },
        )
    return templates.TemplateResponse(
        "partials/season_row.html",
        {
            "request": request,
            "series_id": series_id,
            "season": season,
            "process_message": _process_message(queued, skipped),
            "scan_message": f"Queued {scan_queued} episode(s) for scene scanning.",
        },
    )


@router.post("/{title_id}/severity", response_class=HTMLResponse)
async def set_title_severity(
    request: Request,
    title_id: int,
    sev_child: bool = Form(False),
    sev_teen: bool = Form(False),
    short_label: bool = Form(False),
    detail_view: bool = Form(False),
):
    levels = _severity_levels_from_checkboxes(sev_child, sev_teen)
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")

        # Always save the requested selection, even if it can't be fully honored yet --
        # otherwise picking multiple severities on a non-mkv file would be silently
        # discarded rather than remembered for once an .mkv replacement lands. The
        # resulting gating message/button are derived fresh in _row_dict from this
        # saved state, not computed here, so they persist across any later re-render
        # (e.g. toggling precision) rather than only showing right after this POST.
        title.severity_levels = levels
        await session.commit()
        await session.refresh(title)

        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, short_label=short_label, has_pending_scenes=title_id in pending_ids)

    return await _render_title(request, row, detail_view, severity_saved=detail_view)


@router.post("/shows/{series_id}/season/{season_number}/severity", response_class=HTMLResponse)
async def set_season_severity(
    request: Request,
    series_id: int,
    season_number: int,
    sev_child: bool = Form(False),
    sev_teen: bool = Form(False),
    detail_view: bool = Form(False),
    came_from: str | None = Form(default=None),
):
    levels = _severity_levels_from_checkboxes(sev_child, sev_teen)
    conditions = (Title.sonarr_series_id == series_id, Title.season_number == season_number)
    async with get_session() as session:
        await session.execute(update(Title).where(*conditions).values(severity_levels=levels))
        await session.commit()
        current_version = int(await get_setting(session, "wordlist_version"))
        if detail_view:
            context = await _load_season_detail_context(
                session, series_id, season_number, current_version, came_from=came_from
            )
        else:
            season = await _load_season(session, series_id, season_number, current_version)

    if detail_view:
        return templates.TemplateResponse(
            "partials/season_detail_card.html",
            {"request": request, "series_id": series_id, "season_number": season_number, **context},
        )
    return templates.TemplateResponse(
        "partials/season_row.html",
        {"request": request, "series_id": series_id, "season": season},
    )


@router.post("/{title_id}/set-precise-mode", response_class=HTMLResponse)
async def set_title_precise_mode(
    request: Request,
    title_id: int,
    precise_mode: str = Form(...),
    short_label: bool = Form(False),
    detail_view: bool = Form(False),
):
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        if precise_mode in PRECISE_MODES:
            title.precise_mode = precise_mode
        await session.commit()
        await session.refresh(title)
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(title, current_version, short_label=short_label, has_pending_scenes=title_id in pending_ids)

    return await _render_title(request, row, detail_view, precise_saved=detail_view)


@router.post("/shows/{series_id}/season/{season_number}/set-precise-mode", response_class=HTMLResponse)
async def set_season_precise_mode(
    request: Request,
    series_id: int,
    season_number: int,
    precise_mode: str = Form(...),
    detail_view: bool = Form(False),
    came_from: str | None = Form(default=None),
):
    async with get_session() as session:
        if precise_mode in PRECISE_MODES:
            await session.execute(
                update(Title)
                .where(Title.sonarr_series_id == series_id, Title.season_number == season_number)
                .values(precise_mode=precise_mode)
            )
            await session.commit()
        current_version = int(await get_setting(session, "wordlist_version"))
        if detail_view:
            context = await _load_season_detail_context(
                session, series_id, season_number, current_version, came_from=came_from
            )
        else:
            season = await _load_season(session, series_id, season_number, current_version)

    if detail_view:
        return templates.TemplateResponse(
            "partials/season_detail_card.html",
            {"request": request, "series_id": series_id, "season_number": season_number, **context},
        )
    return templates.TemplateResponse(
        "partials/season_row.html",
        {"request": request, "series_id": series_id, "season": season},
    )


_SUBTITLE_POLL_ATTEMPTS = 6
_SUBTITLE_POLL_INTERVAL_SECONDS = 3.0


async def _search_and_queue_subtitle(
    title_id: int,
    media_type: MediaType,
    video_path: str,
    sonarr_series_id: int | None,
    sonarr_episode_id: int | None,
    radarr_movie_id: int | None,
    default_subtitle_language: str,
    bazarr_url: str,
    bazarr_api_key: str,
) -> str:
    """Search Bazarr for a subtitle and, if found, enqueue an audio-mute job.
    Returns a human-readable status message. Shared by search_subtitle_via_bazarr
    and search_subtitle_and_process_both below -- the only difference between
    those two is whether a scene-scan job also gets enqueued alongside this.

    Takes plain values rather than an open session/Title object on purpose --
    this runs a Bazarr HTTP call plus up to ~18s of polling (see below), and
    previously did all of that with a DB session/pooled connection held open
    the whole time. With the app's own self-polling UI (title/queue/scene-review
    refresh every few seconds) opening additional concurrent sessions, a
    couple of in-flight searches could exhaust a small connection pool and
    stall unrelated requests. Only the final DB write, at the very end, opens
    its own short-lived session.

    Bazarr's search API replies success as soon as the search is *triggered*,
    not once a subtitle is actually downloaded -- the real provider search +
    download happens asynchronously afterward on Bazarr's side. Checking the
    filesystem exactly once immediately after that reply is a real race:
    confirmed live against a real title where the subtitle didn't actually
    land on disk until ~1m45s after Bazarr's API had already returned
    success, producing a flatly wrong "no subtitle was found by any
    provider" message when Bazarr was in fact still working on it. Polls a
    few times over ~15-18s to catch the common case where it finishes
    quickly; a real subtitle that takes longer than that (like the one that
    caught this bug) still won't show up in this one request/response cycle
    -- see the honest, non-final message below rather than pretending
    otherwise."""
    if not bazarr_url or not bazarr_api_key:
        return "Bazarr isn't configured (missing URL/API key) -- set it in Settings."
    if media_type == MediaType.episode and not (sonarr_series_id and sonarr_episode_id):
        return "Missing Sonarr series/episode ID -- try syncing the library first."
    if media_type == MediaType.movie and not radarr_movie_id:
        return "Missing Radarr movie ID -- try syncing the library first."

    client = BazarrClient(bazarr_url, bazarr_api_key)
    try:
        if media_type == MediaType.episode:
            ok = await client.search_episode_subtitle(
                series_id=sonarr_series_id, episode_id=sonarr_episode_id,
                language=default_subtitle_language,
            )
        else:
            ok = await client.search_movie_subtitle(
                radarr_id=radarr_movie_id, language=default_subtitle_language
            )
    except Exception as exc:  # noqa: BLE001 -- surface any connection/API error to the UI
        return f"Bazarr request failed: {exc}"

    if not ok:
        return "Bazarr rejected the search request."

    subtitle = None
    for attempt in range(_SUBTITLE_POLL_ATTEMPTS):
        subtitle = await asyncio.to_thread(find_subtitle_for_video, Path(video_path), default_subtitle_language)
        if subtitle:
            break
        if attempt < _SUBTITLE_POLL_ATTEMPTS - 1:
            await asyncio.sleep(_SUBTITLE_POLL_INTERVAL_SECONDS)

    if not subtitle:
        # Not "no subtitle was found" -- that's a claim about Bazarr's own
        # search outcome, which this request has no way to know yet. Bazarr
        # may still be searching/downloading well past this request's own
        # lifetime (see the docstring above).
        #
        # Real bug this fixes: this used to just return the message below and
        # stop -- unlike the Sonarr/Radarr/Bazarr webhook flows (and the mkv-
        # replacement flow), a manual "Process Audio" click had no fallback
        # once its own short inline poll gave up, so a subtitle landing any
        # time after ~18s (confirmed to take up to ~1m45s in a real case, see
        # above) was never picked up by anything -- the title just sat there
        # looking stuck until someone happened to notice and retry by hand.
        # Falls back to the same bounded background poller the automated
        # paths use, and flips status to "awaiting_subtitle" so the row
        # starts self-polling and picks up the change once the poller finds it.
        async with get_session() as session:
            title = await session.get(Title, title_id)
            if title is not None:
                title.status = "awaiting_subtitle"
                await session.commit()
                spawn_background(
                    poll_for_subtitle_then_enqueue(
                        video_path=video_path,
                        media_type=media_type,
                        display_name=title.display_name,
                        sonarr_series_id=sonarr_series_id,
                        sonarr_episode_id=sonarr_episode_id,
                        radarr_movie_id=radarr_movie_id,
                        trigger_source=TriggerSource.manual,
                        series_title=title.series_title,
                        season_number=title.season_number,
                        episode_number=title.episode_number,
                    )
                )
        return (
            "Bazarr search triggered, but no subtitle has landed yet after "
            f"~{_SUBTITLE_POLL_ATTEMPTS * _SUBTITLE_POLL_INTERVAL_SECONDS:.0f}s -- "
            "still watching for it in the background, this page will pick it up "
            "automatically once it appears."
        )

    async with get_session() as session:
        title = await session.get(Title, title_id)
        if title is not None:
            title.subtitle_path = str(subtitle)
            title.subtitle_language = default_subtitle_language
            await session.commit()
    await job_queue.enqueue(title_id, TriggerSource.manual)
    return "Found it! Subtitle downloaded, queued for processing."


@router.post("/{title_id}/search-subtitle", response_class=HTMLResponse)
async def search_subtitle_via_bazarr(
    request: Request, title_id: int, short_label: bool = Form(False), detail_view: bool = Form(False)
):
    async with get_session() as session:
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        default_subtitle_language = await get_setting(session, "default_subtitle_language")
        bazarr_url = await get_setting(session, "bazarr_url")
        bazarr_api_key = await get_setting(session, "bazarr_api_key")
        search_args = (
            title.id, title.media_type, title.video_path,
            title.sonarr_series_id, title.sonarr_episode_id, title.radarr_movie_id,
        )

    # No session held open across this -- see _search_and_queue_subtitle's docstring.
    message = await _search_and_queue_subtitle(
        *search_args, default_subtitle_language, bazarr_url, bazarr_api_key
    )

    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(
            title, current_version, bazarr_message=message, short_label=short_label,
            has_pending_scenes=title_id in pending_ids,
        )

    return await _render_title(request, row, detail_view)


@router.post("/{title_id}/search-subtitle-and-process-both", response_class=HTMLResponse)
async def search_subtitle_and_process_both(
    request: Request, title_id: int, short_label: bool = Form(False), detail_view: bool = Form(False)
):
    """Same subtitle search as search_subtitle_via_bazarr, but also always
    queues a scene scan alongside it -- scene detection doesn't need a
    subtitle at all (see process_video's own tooltip), so there's no reason
    to gate it on whether Bazarr finds one. Mirrors process_both_title's
    "both jobs always enqueue" contract, just with a Bazarr search
    interleaved on the audio side instead of enqueueing it unconditionally.

    Scene scan is enqueued *before* the subtitle search, not after -- the
    search now polls for up to ~18s (see _search_and_queue_subtitle), and the
    scan has nothing to do with Bazarr at all, so there's no reason to make
    it wait on an unrelated subsystem's latency."""
    await scene_job_queue.enqueue_scan_if_not_already_active(title_id)

    async with get_session() as session:
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        default_subtitle_language = await get_setting(session, "default_subtitle_language")
        bazarr_url = await get_setting(session, "bazarr_url")
        bazarr_api_key = await get_setting(session, "bazarr_api_key")
        search_args = (
            title.id, title.media_type, title.video_path,
            title.sonarr_series_id, title.sonarr_episode_id, title.radarr_movie_id,
        )

    # No session held open across this -- see _search_and_queue_subtitle's docstring.
    subtitle_message = await _search_and_queue_subtitle(
        *search_args, default_subtitle_language, bazarr_url, bazarr_api_key
    )

    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(
            title, current_version, bazarr_message=subtitle_message, short_label=short_label,
            has_pending_scenes=title_id in pending_ids,
        )

    return await _render_title(request, row, detail_view)


@router.post("/{title_id}/search-mkv-replacement", response_class=HTMLResponse)
async def search_mkv_replacement(
    request: Request, title_id: int, short_label: bool = Form(False), detail_view: bool = Form(False)
):
    """Trigger a Sonarr/Radarr search for an .mkv release of this title, so multiple
    severity tracks become available. The queue worker polls for the new file to land
    and, once it does, searches Bazarr for a subtitle if needed and auto-queues processing --
    see JobQueue._check_one_replacement."""
    async with get_session() as session:
        current_version = int(await get_setting(session, "wordlist_version"))
        title = await session.get(Title, title_id)
        if title is None:
            return HTMLResponse("")
        sonarr_url = await get_setting(session, "sonarr_url")
        sonarr_api_key = await get_setting(session, "sonarr_api_key")
        radarr_url = await get_setting(session, "radarr_url")
        radarr_api_key = await get_setting(session, "radarr_api_key")

        if is_mkv_path(title.video_path):
            message = "This file is already .mkv."
        elif title.media_type == MediaType.episode and not (title.sonarr_series_id and title.sonarr_episode_id):
            message = "Missing Sonarr series/episode ID -- try syncing the library first."
        elif title.media_type == MediaType.movie and not title.radarr_movie_id:
            message = "Missing Radarr movie ID -- try syncing the library first."
        elif title.media_type == MediaType.episode and not (sonarr_url and sonarr_api_key):
            message = "Sonarr isn't configured (missing URL/API key) -- set it in Settings."
        elif title.media_type == MediaType.movie and not (radarr_url and radarr_api_key):
            message = "Radarr isn't configured (missing URL/API key) -- set it in Settings."
        else:
            try:
                if title.media_type == MediaType.episode:
                    client = SonarrClient(sonarr_url, sonarr_api_key)
                    await client.trigger_episode_search(title.sonarr_episode_id)
                else:
                    client = RadarrClient(radarr_url, radarr_api_key)
                    await client.trigger_movie_search(title.radarr_movie_id)
            except Exception as exc:  # noqa: BLE001 -- surface any connection/API error to the UI
                message = f"Search request failed: {exc}"
            else:
                title.status = "awaiting_mkv"
                title.replacement_requested_at = datetime.datetime.utcnow()
                await session.commit()
                message = "Search triggered -- will auto-process once an .mkv release is grabbed."

        await session.refresh(title)
        pending_ids = await _pending_scene_title_ids(session, [title_id])
        row = _row_dict(
            title, current_version, bazarr_message=message, short_label=short_label,
            has_pending_scenes=title_id in pending_ids,
        )

    return await _render_title(request, row, detail_view)
