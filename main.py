import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from database import init_db
from routers import tasks, runs, phases
from config import HOST, PORT


@asynccontextmanager
async def lifespan(app):
    await init_db()
    yield


app = FastAPI(title="Prompt Improver", lifespan=lifespan)

app.include_router(tasks.router)
app.include_router(runs.router)
app.include_router(phases.router)


@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
