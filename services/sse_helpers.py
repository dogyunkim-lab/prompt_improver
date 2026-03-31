import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def log_event(level: str, message: str) -> str:
    return sse_event({"type": "log", "level": level, "message": message, "ts": ts()})


def progress_event(current: int, total: int) -> str:
    return sse_event({"type": "progress", "current": current, "total": total})


def result_event(data: dict) -> str:
    return sse_event({"type": "result", "data": data})


def done_event(status: str) -> str:
    return sse_event({"type": "done", "status": status})


def case_event(data: dict) -> str:
    return sse_event({"type": "case", "data": data})
