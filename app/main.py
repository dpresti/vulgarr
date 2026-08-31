import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.auth import AuthGateMiddleware, SetupWizardGateMiddleware
from app.config import settings as app_settings
from app.db.session import init_db
from app.library import reconcile_stale_awaiting_subtitle_titles
from app.queue.scene_worker import scene_job_queue
from app.queue.worker import job_queue
from app.routers import auth as auth_router
from app.routers import library, queue, scenes, settings as settings_router, setup as setup_router, webhooks, wordlist
from app.session_secret import get_or_create_session_secret_key

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await job_queue.start()
    await scene_job_queue.start()
    await reconcile_stale_awaiting_subtitle_titles()
    yield
    await scene_job_queue.stop()
    await job_queue.stop()


app = FastAPI(title="Vulgarr", lifespan=lifespan)
# Registration order matters, and is counter-intuitive: Starlette's own
# add_middleware() does `self.user_middleware.insert(0, ...)`, and
# build_middleware_stack() then wraps that list in `reversed()` order -- the two
# reversals cancel out such that the LAST middleware added ends up OUTERMOST
# (runs first on every request), not the first. Confirmed the hard way: adding
# these in the seemingly-obvious order raised "SessionMiddleware must be
# installed to access request.session" from inside AuthGateMiddleware, because
# it was actually running before SessionMiddleware had populated the session.
# Real desired order (outermost/first to innermost/last): Session -> SetupWizard
# -> AuthGate -> routes -- so they're added here in the exact reverse.
app.add_middleware(AuthGateMiddleware)
app.add_middleware(SetupWizardGateMiddleware)
app.add_middleware(SessionMiddleware, secret_key=get_or_create_session_secret_key(app_settings.data_dir), same_site="lax")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(library.router)
app.include_router(scenes.router)
app.include_router(wordlist.router)
app.include_router(queue.router)
app.include_router(settings_router.router)
app.include_router(webhooks.router)
app.include_router(auth_router.router)
app.include_router(setup_router.router)


@app.get("/")
async def root():
    return RedirectResponse(url="/library")
