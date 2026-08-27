import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.auth import BasicAuthMiddleware
from app.db.session import init_db
from app.library import reconcile_stale_awaiting_subtitle_titles
from app.queue.scene_worker import scene_job_queue
from app.queue.worker import job_queue
from app.routers import library, queue, scenes, settings as settings_router, webhooks, wordlist

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
app.add_middleware(BasicAuthMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(library.router)
app.include_router(scenes.router)
app.include_router(wordlist.router)
app.include_router(queue.router)
app.include_router(settings_router.router)
app.include_router(webhooks.router)


@app.get("/")
async def root():
    return RedirectResponse(url="/library")
