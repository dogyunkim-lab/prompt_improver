import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from database import init_db
from routers import tasks, runs, phases
from config import HOST, PORT, GPT_API_BASE, GPT_MODEL, DIFY_BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    await init_db()
    logger.info("=" * 55)
    logger.info(f"  GPT_API_BASE : {GPT_API_BASE}")
    logger.info(f"  GPT_MODEL    : {GPT_MODEL}")
    logger.info(f"  DIFY_BASE_URL: {DIFY_BASE_URL}")
    logger.info("=" * 55)
    yield


app = FastAPI(title="Prompt Improver", lifespan=lifespan)

app.include_router(tasks.router)
app.include_router(runs.router)
app.include_router(phases.router)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
