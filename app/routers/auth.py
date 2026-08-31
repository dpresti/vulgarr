"""SSO login/callback/logout routes (Authentik via OIDC), plus the Basic Auth
break-glass path. See app/auth.py's AuthGateMiddleware for how these fit
together, and the SSO plan for the overall design.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.auth import _credentials_valid
from app.db.session import get_session, get_setting
from app.sso import build_oauth_client

router = APIRouter(prefix="/auth", tags=["auth"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/login", response_class=HTMLResponse)
async def login_landing(request: Request):
    async with get_session() as session:
        mode = await get_setting(session, "auth_mode")
    if mode != "sso":
        # Nothing to choose between if SSO isn't the active mode -- "password"
        # mode challenges for Basic Auth itself the moment this redirects back
        # into any real page.
        return RedirectResponse(url="/")
    if request.session.get("user"):
        return RedirectResponse(url="/")
    return templates.TemplateResponse("auth_login.html", {"request": request})


@router.get("/login/sso")
async def login_sso(request: Request):
    oauth_client = await build_oauth_client()
    if oauth_client is None:
        return HTMLResponse(
            "SSO isn't configured yet -- set an Issuer URL, Client ID, and Client secret in Settings.",
            status_code=503,
        )
    # request.url_for needs the route below's `name=` to match exactly. Built from
    # the incoming request's own scheme/host -- if this app sits behind a
    # TLS-terminating reverse proxy that doesn't forward X-Forwarded-Proto, this
    # can resolve to http:// instead of https://, which Authentik will reject as
    # a redirect_uri mismatch; not handled here, flagged for whoever deploys behind
    # such a proxy to add uvicorn's --proxy-headers / forwarded-allow-ips.
    redirect_uri = request.url_for("auth_callback")
    try:
        return await oauth_client.authorize_redirect(request, redirect_uri)
    except Exception:  # noqa: BLE001 -- Authentik unreachable/misconfigured issuer
        # URL both surface here (e.g. DNS failure fetching the discovery document)
        # -- this is exactly the scenario the Basic Auth break-glass path exists
        # for, so point at it instead of a bare 500.
        return HTMLResponse(
            'Could not reach the configured SSO provider. '
            '<a href="/auth/login/basic">Use local account instead</a>.',
            status_code=503,
        )


@router.get("/callback", name="auth_callback")
async def callback(request: Request):
    oauth_client = await build_oauth_client()
    if oauth_client is None:
        return HTMLResponse("SSO isn't configured.", status_code=503)
    try:
        token = await oauth_client.authorize_access_token(request)
    except Exception:  # noqa: BLE001 -- e.g. Authentik rejected the code, or became unreachable mid-flow
        return HTMLResponse(
            'SSO sign-in failed. <a href="/auth/login">Try again</a> or '
            '<a href="/auth/login/basic">use local account instead</a>.',
            status_code=502,
        )
    userinfo = token.get("userinfo") or {}
    request.session["user"] = {
        "sub": userinfo.get("sub"),
        "email": userinfo.get("email"),
        "name": userinfo.get("name") or userinfo.get("preferred_username"),
    }
    return RedirectResponse(url="/")


@router.get("/login/basic")
async def login_basic(request: Request):
    """Always challenges for Basic Auth -- the break-glass path when Authentik is
    unreachable or misconfigured. Once the browser has valid credentials cached
    for this realm, AuthGateMiddleware's own Basic check (checked before the
    session check) passes on every later request without needing this route
    again."""
    async with get_session() as session:
        username = await get_setting(session, "auth_username")
        password_hash = await get_setting(session, "auth_password_hash")
    if _credentials_valid(request.headers.get("authorization"), username, password_hash):
        return RedirectResponse(url="/")
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Vulgarr"'})


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")
