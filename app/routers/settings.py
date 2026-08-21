from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.db.session import get_session, get_setting, hash_password, set_setting
from app.domain import PRECISE_MODES
from app.mux.scene_blur import BLUR_LEVEL_PRESETS, blur_level_to_radius_power, radius_power_to_blur_level

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
    "default_precise_mode",
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
    "scene_confidence_threshold",
    "scene_frame_interval_seconds",
    "scene_min_duration_seconds",
    "scene_merge_gap_seconds",
    "scene_scan_concurrency_cap",
    "scene_frame_classify_concurrency",
    "scene_verify_pad_seconds",
    "scene_verify_frame_interval_seconds",
    "scene_high_confidence_fraction",
    "blur_video_crf",
    "blur_video_preset",
    "scene_blur_radius",
    "scene_blur_power",
    "scene_blur_pad_start_seconds",
    "scene_blur_pad_end_seconds",
    "scene_auto_process",
    "claude_vision_verify_enabled",
    "claude_vision_base_url",
    "claude_vision_api_key",
    "claude_vision_model",
    "claude_vision_skip_above_fraction",
]


async def _load_all() -> dict:
    async with get_session() as session:
        values = {key: await get_setting(session, key) for key in SETTING_KEYS}
    # scene_blur_level/scene_blur_level_label are display-only, derived from the
    # stored radius/power pair -- the single slider in settings_form.html edits
    # this instead of the two underlying values directly (see
    # app.mux.scene_blur.BLUR_LEVEL_PRESETS for why: friendlier than exposing
    # boxblur's two independent parameters).
    level = radius_power_to_blur_level(int(values["scene_blur_radius"]), int(values["scene_blur_power"]))
    values["scene_blur_level"] = level
    values["scene_blur_level_label"] = BLUR_LEVEL_PRESETS[level][0]
    return values


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
    default_precise_mode: str = Form("whole_line"),
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
    scene_confidence_threshold: float = Form(0.3),
    scene_frame_interval_seconds: float = Form(0.5),
    scene_min_duration_seconds: float = Form(1.0),
    scene_merge_gap_seconds: float = Form(6.0),
    scene_scan_concurrency_cap: int = Form(1),
    scene_frame_classify_concurrency: int = Form(8),
    scene_verify_pad_seconds: float = Form(5.0),
    scene_verify_frame_interval_seconds: float = Form(0.1),
    scene_high_confidence_fraction: float = Form(0.5),
    blur_video_crf: int = Form(23),
    blur_video_preset: str = Form("medium"),
    scene_blur_level: int = Form(4),
    scene_blur_pad_start_seconds: float = Form(2.0),
    scene_blur_pad_end_seconds: float = Form(5.0),
    scene_auto_process: bool = Form(False),
    claude_vision_verify_enabled: bool = Form(False),
    claude_vision_base_url: str = Form(""),
    claude_vision_api_key: str = Form(""),
    claude_vision_model: str = Form("claude-sonnet-5"),
    claude_vision_skip_above_fraction: float = Form(0.9),
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
        if default_precise_mode not in PRECISE_MODES:
            default_precise_mode = "whole_line"
        await set_setting(session, "default_precise_mode", default_precise_mode)
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

        await set_setting(session, "scene_confidence_threshold", max(0.0, min(1.0, scene_confidence_threshold)))
        await set_setting(session, "scene_frame_interval_seconds", max(0.5, scene_frame_interval_seconds))
        await set_setting(session, "scene_min_duration_seconds", max(0.0, scene_min_duration_seconds))
        await set_setting(session, "scene_merge_gap_seconds", max(0.0, scene_merge_gap_seconds))
        await set_setting(session, "scene_scan_concurrency_cap", max(1, scene_scan_concurrency_cap))
        await set_setting(session, "scene_frame_classify_concurrency", max(1, scene_frame_classify_concurrency))
        await set_setting(session, "scene_verify_pad_seconds", max(0.0, scene_verify_pad_seconds))
        await set_setting(session, "scene_verify_frame_interval_seconds", max(0.02, scene_verify_frame_interval_seconds))
        await set_setting(session, "scene_high_confidence_fraction", max(0.0, min(1.0, scene_high_confidence_fraction)))
        await set_setting(session, "blur_video_crf", max(0, min(51, blur_video_crf)))
        await set_setting(session, "blur_video_preset", blur_video_preset.strip() or "medium")
        radius, power = blur_level_to_radius_power(max(1, min(5, scene_blur_level)))
        await set_setting(session, "scene_blur_radius", radius)
        await set_setting(session, "scene_blur_power", power)
        await set_setting(session, "scene_blur_pad_start_seconds", max(0.0, scene_blur_pad_start_seconds))
        await set_setting(session, "scene_blur_pad_end_seconds", max(0.0, scene_blur_pad_end_seconds))
        await set_setting(session, "scene_auto_process", scene_auto_process)
        claude_vision_base_url = claude_vision_base_url.strip()
        claude_vision_api_key = claude_vision_api_key.strip()
        # Same defensive pattern as auth_enabled above -- the checkbox can't
        # actually take effect without both a base URL and a key, no matter
        # what the form submitted, so the stored value never lands in an
        # inconsistent "on but unconfigured" state. claude_vision_blocked
        # (below) is what tells the template *why*, for the one save where
        # this actually mattered.
        claude_vision_ready = bool(claude_vision_base_url and claude_vision_api_key)
        claude_vision_blocked = claude_vision_verify_enabled and not claude_vision_ready
        await set_setting(session, "claude_vision_verify_enabled", claude_vision_verify_enabled and claude_vision_ready)
        await set_setting(session, "claude_vision_base_url", claude_vision_base_url)
        await set_setting(session, "claude_vision_api_key", claude_vision_api_key)
        await set_setting(session, "claude_vision_model", claude_vision_model.strip() or "claude-sonnet-5")
        await set_setting(
            session, "claude_vision_skip_above_fraction", max(0.0, min(1.0, claude_vision_skip_above_fraction))
        )

    values = await _load_all()
    return templates.TemplateResponse(
        "partials/settings_form.html",
        {"request": request, "values": values, "saved": True, "claude_vision_blocked": claude_vision_blocked},
    )
