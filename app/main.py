import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.auth import BasicAuthMiddleware
from app.db.session import init_db
from app.queue.worker import job_queue
from app.routers import library, queue, settings as settings_router, webhooks, wordlist

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await job_queue.start()
    yield
    await job_queue.stop()


app = FastAPI(title="Subtitle Profanity Filter", lifespan=lifespan)
app.add_middleware(BasicAuthMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(library.router)
app.include_router(wordlist.router)
app.include_router(queue.router)
app.include_router(settings_router.router)
app.include_router(webhooks.router)


@app.get("/")
async def root():
    return RedirectResponse(url="/library")
