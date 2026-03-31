import os
import sys
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# prompt-improver/ 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import database
from database import init_db
from main import app
from routers import phases as phases_mod


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    """테스트용 격리된 SQLite DB + FastAPI 테스트 클라이언트"""
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr(database, "DB_PATH", test_db)
    await init_db()
    phases_mod._stream_queues.clear()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
