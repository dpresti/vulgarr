from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.db.session import get_session, get_setting, set_setting

router = APIRouter(prefix="/settings", tags=["settings"])
templates = Jinja2Templates(directory="app/templates")

SETTING_KEYS = [
    "concurrency_cap",
    "off_hours_enabled",
    "off_hours_start",
    "off_hours_end",
    "trigger_bazarr_enabled",
    "trigger_sonarr_radarr_enabled",
    "trigger_priority",
    "default_precise_mute",
]


async def _load_all() -> dict:
    async with get_session() as session:
        return {key: await get_setting(session, key) for key in SETTING_KEYS}


@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request):
    values = await _load_all()
    return templates.TemplateResponse("settings.html", {"request": request, "values": values})


@router.post("", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    concurrency_cap: int = Form(...),
    off_hours_enabled: bool = Form(False),
    off_hours_start: str = Form(...),
    off_hours_end: str = Form(...),
    trigger_bazarr_enabled: bool = Form(False),
    trigger_sonarr_radarr_enabled: bool = Form(False),
    trigger_priority_first: str = Form("sonarr_radarr"),
    default_precise_mute: bool = Form(False),
):
    trigger_priority_second = "bazarr" if trigger_priority_first == "sonarr_radarr" else "sonarr_radarr"

    async with get_session() as session:
        await set_setting(session, "concurrency_cap", max(1, concurrency_cap))
        await set_setting(session, "off_hours_enabled", off_hours_enabled)
        await set_setting(session, "off_hours_start", off_hours_start)
        await set_setting(session, "off_hours_end", off_hours_end)
        await set_setting(session, "trigger_bazarr_enabled", trigger_bazarr_enabled)
        await set_setting(session, "trigger_sonarr_radarr_enabled", trigger_sonarr_radarr_enabled)
        await set_setting(session, "trigger_priority", [trigger_priority_first, trigger_priority_second])
        await set_setting(session, "default_precise_mute", default_precise_mute)

    values = await _load_all()
    return templates.TemplateResponse(
        "partials/settings_form.html", {"request": request, "values": values, "saved": True}
    )
