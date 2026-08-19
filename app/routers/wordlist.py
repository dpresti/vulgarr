from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Select, select

from app.db.models import Severity, WordListEntry
from app.db.session import bump_wordlist_version, get_session

router = APIRouter(prefix="/wordlist", tags=["wordlist"])
templates = Jinja2Templates(directory="app/templates")

_SEVERITY_RANK = {"mild": 0, "moderate": 1, "strong": 2}


def _severity_from_checkboxes(sev_mild: bool, sev_moderate: bool, sev_strong: bool) -> Severity:
    """A term's severity is the highest checked box -- strong implies moderate implies
    mild, so checking "strong" is what makes a word get muted at every threshold."""
    if sev_strong:
        return Severity.strong
    if sev_moderate:
        return Severity.moderate
    return Severity.mild


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _qs(params: dict, **overrides) -> str:
    merged = {**params, **overrides}
    merged = {k: v for k, v in merged.items() if v not in (None, "", "all")}
    return "?" + urlencode(merged) if merged else ""


async def _load_entries(
    q: str | None, severity: str | None, enabled: str | None, sort: str | None, sort_dir: str
) -> list[WordListEntry]:
    query: Select = select(WordListEntry)
    if q:
        query = query.where(WordListEntry.term.like(f"%{_escape_like(q)}%", escape="\\"))
    if severity in ("mild", "moderate", "strong"):
        query = query.where(WordListEntry.severity == severity)
    if enabled == "enabled":
        query = query.where(WordListEntry.enabled.is_(True))
    elif enabled == "disabled":
        query = query.where(WordListEntry.enabled.is_(False))

    if sort != "severity":
        # Severity has no natural SQL ordering (mild/moderate/strong isn't alphabetical),
        # so it's sorted in Python below instead; every other column sorts in SQL.
        sort_columns = {
            "term": WordListEntry.term.collate("NOCASE"),
            "enabled": WordListEntry.enabled,
        }
        sort_col = sort_columns.get(sort, WordListEntry.term.collate("NOCASE"))
        order = sort_col.desc() if sort_dir == "desc" else sort_col.asc()
        query = query.order_by(order, WordListEntry.term.collate("NOCASE"))

    async with get_session() as session:
        result = await session.execute(query)
        entries = list(result.scalars().all())

    if sort == "severity":
        entries.sort(key=lambda e: _SEVERITY_RANK[e.severity.value], reverse=(sort_dir == "desc"))

    return entries


def _view_context(request: Request, q, severity, enabled, sort, sort_dir) -> dict:
    current_params = {"q": q, "severity": severity, "enabled": enabled, "sort": sort, "dir": sort_dir}

    def sort_link(key: str) -> str:
        new_dir = "desc" if sort == key and sort_dir == "asc" else "asc"
        return _qs(current_params, sort=key, dir=new_dir)

    return {
        "request": request,
        "q": q or "",
        "severity": severity or "all",
        "enabled": enabled or "all",
        "sort": sort,
        "sort_dir": sort_dir,
        "sort_link": sort_link,
        "current_params": current_params,
    }


@router.get("", response_class=HTMLResponse)
async def wordlist_page(
    request: Request,
    q: str | None = None,
    severity: str | None = None,
    enabled: str | None = None,
    sort: str | None = None,
    dir: str = "asc",
):
    entries = await _load_entries(q, severity, enabled, sort, dir)
    context = _view_context(request, q, severity, enabled, sort, dir)
    context["entries"] = entries
    return templates.TemplateResponse("wordlist.html", context)


async def _rerender_table(request: Request, q, severity, enabled, sort, sort_dir):
    entries = await _load_entries(q, severity, enabled, sort, sort_dir)
    context = _view_context(request, q, severity, enabled, sort, sort_dir)
    context["entries"] = entries
    return templates.TemplateResponse("partials/wordlist_table.html", context)


@router.post("", response_class=HTMLResponse)
async def add_term(
    request: Request,
    term: str = Form(...),
    sev_mild: bool = Form(False),
    sev_moderate: bool = Form(False),
    sev_strong: bool = Form(False),
    match_whole_word: bool = Form(True),
    q: str | None = Form(None),
    severity: str | None = Form(None),
    enabled: str | None = Form(None),
    sort: str | None = Form(None),
    sort_dir: str = Form("asc"),
):
    term = term.strip().lower()
    new_severity = _severity_from_checkboxes(sev_mild, sev_moderate, sev_strong)
    async with get_session() as session:
        if term:
            existing = await session.execute(select(WordListEntry).where(WordListEntry.term == term))
            if existing.scalar_one_or_none() is None:
                session.add(WordListEntry(term=term, severity=new_severity, match_whole_word=match_whole_word))
                await session.commit()
                await bump_wordlist_version(session)

    return await _rerender_table(request, q, severity, enabled, sort, sort_dir)


@router.post("/{entry_id}/toggle", response_class=HTMLResponse)
async def toggle_term(
    request: Request,
    entry_id: int,
    q: str | None = Form(None),
    severity: str | None = Form(None),
    enabled: str | None = Form(None),
    sort: str | None = Form(None),
    sort_dir: str = Form("asc"),
):
    async with get_session() as session:
        entry = await session.get(WordListEntry, entry_id)
        if entry is not None:
            entry.enabled = not entry.enabled
            await session.commit()
            await bump_wordlist_version(session)

    return await _rerender_table(request, q, severity, enabled, sort, sort_dir)


@router.post("/{entry_id}/severity", response_class=HTMLResponse)
async def update_severity(
    request: Request,
    entry_id: int,
    sev_mild: bool = Form(False),
    sev_moderate: bool = Form(False),
    sev_strong: bool = Form(False),
    q: str | None = Form(None),
    severity: str | None = Form(None),
    enabled: str | None = Form(None),
    sort: str | None = Form(None),
    sort_dir: str = Form("asc"),
):
    new_severity = _severity_from_checkboxes(sev_mild, sev_moderate, sev_strong)
    async with get_session() as session:
        entry = await session.get(WordListEntry, entry_id)
        if entry is not None:
            entry.severity = new_severity
            await session.commit()
            await bump_wordlist_version(session)

    return await _rerender_table(request, q, severity, enabled, sort, sort_dir)


@router.post("/{entry_id}/delete", response_class=HTMLResponse)
async def delete_term(
    request: Request,
    entry_id: int,
    q: str | None = Form(None),
    severity: str | None = Form(None),
    enabled: str | None = Form(None),
    sort: str | None = Form(None),
    sort_dir: str = Form("asc"),
):
    async with get_session() as session:
        entry = await session.get(WordListEntry, entry_id)
        if entry is not None:
            await session.delete(entry)
            await session.commit()
            await bump_wordlist_version(session)

    return await _rerender_table(request, q, severity, enabled, sort, sort_dir)
