"""Builds an authlib OAuth client for Authentik SSO from live settings
(sso_issuer_url/sso_client_id/sso_client_secret). Read fresh on every call rather
than cached at import time, matching this codebase's existing pattern for other
optional integrations (RadarrClient/SonarrClient/BazarrClient are all constructed
fresh per call too) -- these settings can change at runtime via the Settings page
with no restart, so nothing about the client should be cached.
"""

from authlib.integrations.starlette_client import OAuth
from authlib.integrations.starlette_client import StarletteOAuth2App

from app.db.session import get_session, get_setting

_REGISTRATION_NAME = "authentik"


async def build_oauth_client() -> StarletteOAuth2App | None:
    """None if SSO isn't configured yet (any of the three settings blank) --
    callers should treat that as "SSO unavailable", not attempt a request that
    would just fail against an empty issuer URL."""
    async with get_session() as session:
        issuer_url = (await get_setting(session, "sso_issuer_url") or "").strip()
        client_id = await get_setting(session, "sso_client_id")
        client_secret = await get_setting(session, "sso_client_secret")

    if not (issuer_url and client_id and client_secret):
        return None

    oauth = OAuth()
    oauth.register(
        name=_REGISTRATION_NAME,
        server_metadata_url=f"{issuer_url.rstrip('/')}/.well-known/openid-configuration",
        client_id=client_id,
        client_secret=client_secret,
        client_kwargs={"scope": "openid profile email"},
    )
    return oauth.create_client(_REGISTRATION_NAME)
