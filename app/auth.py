"""Optional HTTP Basic Auth in front of the whole UI. Off by default (see
DEFAULT_SETTINGS["auth_enabled"]) -- most deployments sit behind a private
network or their own reverse-proxy auth, but this covers the case where this
app is reachable by more than just its owner.

Webhook endpoints are exempt: Sonarr/Radarr/Bazarr can't do an interactive
Basic Auth prompt, and they already carry their own `?token=` check.
"""

import base64
import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.db.session import get_session, get_setting, verify_password

_EXEMPT_PREFIXES = ("/webhooks", "/static")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)

        async with get_session() as session:
            enabled = await get_setting(session, "auth_enabled")
            if not enabled:
                return await call_next(request)
            username = await get_setting(session, "auth_username")
            password_hash = await get_setting(session, "auth_password_hash")

        if _credentials_valid(request.headers.get("authorization"), username, password_hash):
            return await call_next(request)

        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Vulgarr"'})


def _credentials_valid(auth_header: str | None, username: str, password_hash: str) -> bool:
    if not auth_header or not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
    except Exception:
        return False
    supplied_user, _, supplied_pass = decoded.partition(":")
    return hmac.compare_digest(supplied_user, username) and verify_password(supplied_pass, password_hash)
