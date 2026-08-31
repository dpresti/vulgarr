"""First-run setup wizard -- reuses the existing Settings page's fields/save route
(app/routers/settings.py) rather than duplicating every integration/toggle field.
See app/auth.py's SetupWizardGateMiddleware for how a fresh install ends up here.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.db.session import get_session, set_setting
from app.routers.settings import SETTING_KEYS, _load_all

router = APIRouter(prefix="/setup", tags=["setup"])
templates = Jinja2Templates(directory="app/templates")


def _filter_importable_settings(data) -> dict:
    """Whitelist-filters an uploaded config dict down to recognized setting keys --
    reuses SETTING_KEYS, the same whitelist app.routers.settings.update_settings
    itself trusts, so an imported file can't inject an arbitrary/unknown key into
    app_settings. No per-field type validation beyond this -- the only realistic
    input is a file this same app exported (see settings.py's GET /settings/export)."""
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in SETTING_KEYS}


@router.get("", response_class=HTMLResponse)
async def setup_wizard(request: Request):
    values = await _load_all()
    return templates.TemplateResponse(
        "setup.html", {"request": request, "values": values, "wizard_mode": True}
    )


@router.post("/complete")
async def complete_setup():
    async with get_session() as session:
        await set_setting(session, "setup_wizard_completed", True)
    return RedirectResponse(url="/library", status_code=303)


@router.post("/import")
async def import_config(request: Request):
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 -- malformed upload, not a server error
        return JSONResponse({"ok": False, "error": "Not valid JSON"}, status_code=400)

    filtered = _filter_importable_settings(data)
    if not filtered:
        return JSONResponse(
            {"ok": False, "error": "No recognized settings found in that file"}, status_code=400
        )

    async with get_session() as session:
        for key, value in filtered.items():
            await set_setting(session, key, value)
    return JSONResponse({"ok": True, "imported": len(filtered)})
