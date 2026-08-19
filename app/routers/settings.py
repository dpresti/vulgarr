from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.db.session import get_session, get_setting, hash_password, set_setting

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
    "sonarr_url",
    "sonarr_api_key",
    "radarr_url",
    "radarr_api_key",
    "bazarr_url",
    "bazarr_api_key",
    "webhook_token",
    "clean_track_title",
    "clean_track_language",
    "default_subtitle_language",
    "backup_retention_days",
    "auth_enabled",
    "auth_username",
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
    sonarr_url: str = Form(""),
    sonarr_api_key: str = Form(""),
    radarr_url: str = Form(""),
    radarr_api_key: str = Form(""),
    bazarr_url: str = Form(""),
    bazarr_api_key: str = Form(""),
    webhook_token: str = Form(""),
    clean_track_title: str = Form("Clean"),
    clean_track_language: str = Form("eng"),
    default_subtitle_language: str = Form("en"),
    backup_retention_days: int = Form(0),
    auth_enabled: bool = Form(False),
    auth_username: str = Form(""),
    auth_password: str = Form(""),
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
        await set_setting(session, "sonarr_url", sonarr_url.strip())
        await set_setting(session, "sonarr_api_key", sonarr_api_key.strip())
        await set_setting(session, "radarr_url", radarr_url.strip())
        await set_setting(session, "radarr_api_key", radarr_api_key.strip())
        await set_setting(session, "bazarr_url", bazarr_url.strip())
        await set_setting(session, "bazarr_api_key", bazarr_api_key.strip())
        # Blank means "keep the current token" -- never silently clear it back to
        # open/unauthenticated, since that's exactly the footgun this replaced.
        if webhook_token.strip():
            await set_setting(session, "webhook_token", webhook_token.strip())
        await set_setting(session, "clean_track_title", clean_track_title.strip() or "Clean")
        await set_setting(session, "clean_track_language", clean_track_language.strip() or "eng")
        await set_setting(session, "default_subtitle_language", default_subtitle_language.strip() or "en")
        await set_setting(session, "backup_retention_days", max(0, backup_retention_days))

        # auth_enabled is only allowed on once a username and (new or existing)
        # password are actually in place -- otherwise saving the form with the
        # toggle checked but the password field left blank would lock the UI
        # behind a login nobody can complete.
        existing_hash = await get_setting(session, "auth_password_hash")
        auth_username = auth_username.strip()
        if auth_password:
            await set_setting(session, "auth_password_hash", hash_password(auth_password))
            existing_hash = "set"
        await set_setting(session, "auth_username", auth_username)
        await set_setting(session, "auth_enabled", bool(auth_enabled and auth_username and existing_hash))

    values = await _load_all()
    return templates.TemplateResponse(
        "partials/settings_form.html", {"request": request, "values": values, "saved": True}
    )
