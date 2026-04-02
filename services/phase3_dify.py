import asyncio
import json
import time
from datetime import datetime
from typing import AsyncGenerator
import httpx
from config import DIFY_BASE_URL
from database import get_db
from services.dify_auth import get_dify_token
from services.sse_helpers import log_event, progress_event, result_event, done_event

DIFY_CONCURRENCY = 5


async def verify_dify_connection(object_id: str) -> tuple[bool, str]:
    """object_id로 토큰을 발급받아 Dify 연결을 검증한다. (성공여부, 메시지) 반환."""
    try:
        token = await get_dify_token(object_id)
    except NotImplementedError:
        return False, "토큰 발급 함수(_issue_token)가 구현되지 않았습니다. services/dify_auth.py를 확인하세요."
    except Exception as e:
        return False, f"토큰 발급 실패: {e}"

    try:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"inputs": {}, "response_mode": "blocking", "user": "verify-test"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{DIFY_BASE_URL}/workflows/run", json=payload, headers=headers)
            if resp.status_code < 400:
                return True, f"연결 성공 (HTTP {resp.status_code})"
            else:
                body = resp.text[:200]
                return False, f"Dify 응답 오류 (HTTP {resp.status_code}): {body}"
    except Exception as e:
        return False, f"Dify 연결 실패: {e}"


async def call_dify_workflow(object_id: str, stt: str) -> dict:
    """Dify 워크플로우 실행 후 전체 outputs dict 반환 (중간 노드 출력 포함)."""
    token = await get_dify_token(object_id)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": {"stt": stt},
        "response_mode": "blocking",
        "user": "improver-system"
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{DIFY_BASE_URL}/workflows/run", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["data"]["outputs"]


async def run_phase3(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute(
            "SELECT * FROM dify_connections WHERE run_id=? AND status='verified'",
            (run_id,)
        ) as cursor:
            connections = [dict(row) for row in await cursor.fetchall()]

        if not connections:
            yield log_event("error", "검증된 Dify 연결이 없습니다.")
            yield done_event("failed")
            return

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
            """INSERT INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,3,'running',?)
               ON CONFLICT(run_id, phase) DO UPDATE SET status='running', started_at=excluded.started_at""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()
        await db.execute("UPDATE runs SET status='phase3_running', current_phase=3 WHERE id=?", (run_id,))
        await db.commit()

        total = len(cases)
        yield log_event("info", f"총 {total}개 케이스 실행 시작")

        # 선택된 후보의 output_var → node 매핑 조회
        output_var_to_node = {}
        async with db.execute(
            "SELECT selected_candidate_id FROM runs WHERE id=?", (run_id,)
        ) as cursor:
            run_row = await cursor.fetchone()
        selected_cand_id = run_row["selected_candidate_id"] if run_row else None

        if selected_cand_id:
            async with db.execute(
                """SELECT node_a_output_var, node_b_output_var, node_c_output_var,
                          node_a_prompt, node_b_prompt, node_c_prompt
                   FROM prompt_candidates WHERE id=?""",
                (selected_cand_id,)
            ) as cursor:
                cand_row = await cursor.fetchone()
            if cand_row:
                cand_row = dict(cand_row)
                for label in ('a', 'b', 'c'):
                    ov = cand_row.get(f"node_{label}_output_var")
                    has_node = cand_row.get(f"node_{label}_prompt")
                    if ov and has_node:
                        output_var_to_node[ov] = label.upper()
                if output_var_to_node:
                    yield log_event("info",
                        f"노드-변수 매핑: {', '.join(f'Node {v}→{k}' for k, v in output_var_to_node.items())}")

        semaphore = asyncio.Semaphore(DIFY_CONCURRENCY)
        completed = 0
        errors = 0

        conn = connections[0]  # 첫 번째 연결 사용

        async def process_case(case: dict):
            nonlocal completed, errors
            async with semaphore:
                case_id = case["case_id"]
                for attempt in range(3):
                    try:
                        start_t = time.time()
                        outputs = await call_dify_workflow(
                            conn["object_id"],
                            case.get("stt", "")
                        )
                        elapsed = round(time.time() - start_t, 1)
                        generated = outputs.get("generated", "")
                        # generated 외 나머지 = 중간 노드 출력 (노드 라벨 매핑 포함)
                        intermediate = {}
                        for k, v in outputs.items():
                            if k == "generated":
                                continue
                            node_label = output_var_to_node.get(k)
                            if node_label:
                                intermediate[k] = {"node": node_label, "content": str(v) if v else ""}
                            else:
                                intermediate[k] = {"node": "?", "content": str(v) if v else ""}
                        await db.execute(
                            "UPDATE case_results SET generated=?, intermediate_outputs=? WHERE run_id=? AND case_id=?",
                            (generated,
                             json.dumps(intermediate, ensure_ascii=False) if intermediate else None,
                             run_id, case_id)
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

        pending_tasks = [asyncio.create_task(process_case(c)) for c in cases]

        try:
            done_count = 0
            for coro in asyncio.as_completed(pending_tasks):
                result_str = await coro
                done_count += 1
                if result_str.startswith("ok:"):
                    elapsed = result_str.split(":")[1]
                    yield log_event("ok", f"완료 ({elapsed}s)")
                else:
                    yield log_event("warn", f"실패: {result_str.split(':', 1)[1]}")
                yield progress_event(done_count, total)
        finally:
            for t in pending_tasks:
                if not t.done():
                    t.cancel()

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
