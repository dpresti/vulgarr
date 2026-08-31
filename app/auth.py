"""Auth gate in front of the whole UI, branching on the `auth_mode` setting:
"none" (default -- most deployments sit behind a private network or their own
reverse-proxy auth), "password" (the original HTTP Basic Auth, unchanged), or
"sso" (Authentik OIDC login via app/routers/auth.py, with Basic Auth kept alive
underneath as a break-glass fallback in case Authentik itself is unreachable or
misconfigured).

Webhook endpoints are exempt: Sonarr/Radarr/Bazarr can't do an interactive
login, and they already carry their own `?token=` check. /auth/* is exempt too --
an unauthenticated user must be able to reach the login routes themselves.
"""

import base64
import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from app.db.session import get_session, get_setting, verify_password

_EXEMPT_PREFIXES = ("/webhooks", "/static", "/auth")


class AuthGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)

        async with get_session() as session:
            mode = await get_setting(session, "auth_mode")
            if mode == "none":
                return await call_next(request)
            username = await get_setting(session, "auth_username")
            password_hash = await get_setting(session, "auth_password_hash")

        basic_ok = _credentials_valid(request.headers.get("authorization"), username, password_hash)
        has_session_user = bool(request.session.get("user")) if mode == "sso" else False

        if _is_authenticated(mode, basic_ok, has_session_user):
            return await call_next(request)

        if mode == "password":
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Vulgarr"'})

        if request.headers.get("hx-request") == "true":
            # An htmx fragment request can't follow a normal redirect into a full
            # login page -- HX-Redirect tells htmx to navigate the whole browser
            # instead of trying to swap this response into a small hx-target.
            return Response(status_code=200, headers={"HX-Redirect": "/auth/login"})
        return RedirectResponse(url="/auth/login")


def _is_authenticated(mode: str, basic_ok: bool, has_session_user: bool) -> bool:
    """Pure decision extracted out of AuthGateMiddleware.dispatch so it's testable
    without a real Request/DB (see tests/test_auth.py) -- mirrors the branching
    there exactly: "none" always passes, a valid Basic Auth header always passes
    (both "password" mode's normal login and "sso" mode's break-glass fallback),
    and "sso" mode additionally passes with a valid session (the OIDC callback)."""
    if mode == "none":
        return True
    if basic_ok:
        return True
    return mode == "sso" and has_session_user


def _credentials_valid(auth_header: str | None, username: str, password_hash: str) -> bool:
    if not auth_header or not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
    except Exception:
        return False
    supplied_user, _, supplied_pass = decoded.partition(":")
    return hmac.compare_digest(supplied_user, username) and verify_password(supplied_pass, password_hash)


_WIZARD_EXEMPT_PREFIXES = ("/setup", "/static", "/webhooks")


class SetupWizardGateMiddleware(BaseHTTPMiddleware):
    """Runs before AuthGateMiddleware (see main.py's registration order) -- a fresh
    install has no auth configured yet, so ordering rarely matters in practice, but
    the wizard must always be reachable regardless of whatever auth state exists."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(_WIZARD_EXEMPT_PREFIXES):
            return await call_next(request)

        async with get_session() as session:
            completed = await get_setting(session, "setup_wizard_completed")
        if completed:
            return await call_next(request)

        if request.headers.get("hx-request") == "true":
            return Response(status_code=200, headers={"HX-Redirect": "/setup"})
        return RedirectResponse(url="/setup")
