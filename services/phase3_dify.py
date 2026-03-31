import asyncio
import json
import time
from datetime import datetime
from typing import AsyncGenerator
import httpx
from database import get_db
from services.sse_helpers import log_event, progress_event, result_event, done_event

DIFY_CONCURRENCY = 5


async def verify_dify_connection(api_url: str, api_key: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{api_url}/workflows/run", headers=headers)
            # 400 이상도 연결은 된 것 (endpoint 존재)
            return resp.status_code < 500
    except Exception:
        return False


async def call_dify_workflow(api_url: str, api_key: str, stt: str, keywords: str, generation_task: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": {"stt": stt, "keywords": keywords, "generation_task": generation_task},
        "response_mode": "blocking",
        "user": "improver-system"
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{api_url}/workflows/run", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["data"]["outputs"]["generated"]


async def run_phase3(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        # 연결 정보 조회
        async with db.execute(
            "SELECT * FROM dify_connections WHERE run_id=? AND status='verified'",
            (run_id,)
        ) as cursor:
            connections = [dict(row) for row in await cursor.fetchall()]

        if not connections:
            yield log_event("error", "검증된 Dify 연결이 없습니다.")
            yield done_event("failed")
            return

        # 케이스 목록
        async with db.execute(
            "SELECT * FROM case_results WHERE run_id=?",
            (run_id,)
        ) as cursor:
            cases = [dict(row) for row in await cursor.fetchall()]

        if not cases:
            yield log_event("error", "케이스 데이터가 없습니다. Phase 1을 먼저 실행하세요.")
            yield done_event("failed")
            return

        await db.execute(
            """INSERT OR REPLACE INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,3,'running',?)""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()
        await db.execute("UPDATE runs SET status='phase3_running', current_phase=3 WHERE id=?", (run_id,))
        await db.commit()

        total = len(cases)
        yield log_event("info", f"총 {total}개 케이스 실행 시작")

        semaphore = asyncio.Semaphore(DIFY_CONCURRENCY)
        completed = 0
        errors = 0

        conn = connections[0]  # 첫 번째 연결 사용 (탐색 모드면 복수이나 순차 처리)

        async def process_case(case: dict):
            nonlocal completed, errors
            async with semaphore:
                case_id = case["case_id"]
                for attempt in range(3):
                    try:
                        start_t = time.time()
                        generated = await call_dify_workflow(
                            conn["dify_api_url"], conn["dify_api_key"],
                            case.get("stt", ""), case.get("keywords", ""),
                            case.get("generation_task", "")
                        )
                        elapsed = round(time.time() - start_t, 1)
                        await db.execute(
                            "UPDATE case_results SET generated=? WHERE run_id=? AND case_id=?",
                            (generated, run_id, case_id)
                        )
                        await db.commit()
                        completed += 1
                        return f"ok:{elapsed}"
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep(2)
                        else:
                            errors += 1
                            return f"error:{e}"

        tasks_list = [process_case(c) for c in cases]

        done_count = 0
        for coro in asyncio.as_completed(tasks_list):
            result_str = await coro
            done_count += 1
            if result_str.startswith("ok:"):
                elapsed = result_str.split(":")[1]
                yield log_event("ok", f"완료 ({elapsed}s)")
            else:
                yield log_event("warn", f"실패: {result_str.split(':', 1)[1]}")
            yield progress_event(done_count, total)

        msg = f"{total}개 케이스 완료 (오류: {errors}건)"
        yield log_event("ok" if errors == 0 else "warn", msg)

        output = {"total": total, "completed": completed, "errors": errors}
        await db.execute(
            "UPDATE phase_results SET status='completed', output_data=?, completed_at=? WHERE run_id=? AND phase=3",
            (json.dumps(output), datetime.utcnow().isoformat(), run_id)
        )
        await db.execute("UPDATE runs SET status='phase3_done' WHERE id=?", (run_id,))
        await db.commit()

        yield result_event(output)
        yield done_event("completed")

    except Exception as e:
        yield log_event("error", f"Phase 3 오류: {e}")
        yield done_event("failed")
        db2 = await get_db()
        try:
            await db2.execute("UPDATE phase_results SET status='failed' WHERE run_id=? AND phase=3", (run_id,))
            await db2.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))
            await db2.commit()
        finally:
            await db2.close()
    finally:
        await db.close()
