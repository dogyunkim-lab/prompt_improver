import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from database import init_db
from routers import tasks, runs, phases
from config import HOST, PORT

app = FastAPI(title="Prompt Improver")

app.include_router(tasks.router)
app.include_router(runs.router)
app.include_router(phases.router)


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
