import asyncio
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from database import get_db
from services.phase1_analysis import run_phase1
from services.phase2_design import run_phase2
from services.phase3_dify import run_phase3, verify_dify_connection  # noqa
from services.phase4_judge import run_phase4
from services.phase6_strategy import run_phase6
from services.delta import aggregate_scores

logger = logging.getLogger(__name__)

router = APIRouter(tags=["phases"])

# 각 run_id별 스트림 큐 저장
_stream_queues: dict[int, dict[int, asyncio.Queue]] = {}
# 실행 중인 백그라운드 태스크 (run_id, phase) → Task
_running_tasks: dict[tuple, asyncio.Task] = {}


def get_queue(run_id: int, phase: int) -> asyncio.Queue:
    if run_id not in _stream_queues:
        _stream_queues[run_id] = {}
    if phase not in _stream_queues[run_id]:
        _stream_queues[run_id][phase] = asyncio.Queue()
    return _stream_queues[run_id][phase]


async def _run_and_queue(generator, run_id: int, phase: int):
    from services.sse_helpers import log_event as _le, done_event as _de
    q = get_queue(run_id, phase)
    try:
        async for event in generator:
            await q.put(event)
    except asyncio.CancelledError:
        await q.put(_le("warn", "사용자가 Phase를 중단했습니다."))
        await q.put(_de("cancelled"))
        try:
            db = await get_db()
            await db.execute(
                "UPDATE phase_results SET status='cancelled', completed_at=? WHERE run_id=? AND phase=?",
                (datetime.utcnow().isoformat(), run_id, phase)
            )
            await db.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))
            await db.commit()
            await db.close()
        except Exception:
            pass
    except Exception as e:
        logger.exception(f"[Phase {phase} / run {run_id}] 백그라운드 태스크 예외: {e}")
    finally:
        await q.put(None)
        _running_tasks.pop((run_id, phase), None)


def _create_phase_task(generator, run_id: int, phase: int) -> asyncio.Task:
    task = asyncio.create_task(_run_and_queue(generator, run_id, phase))
    _running_tasks[(run_id, phase)] = task
    return task


# ── 중단 엔드포인트 ────────────────────────────────────────────────────────────

@router.post("/api/runs/{run_id}/phase/{phase}/cancel")
async def cancel_phase(run_id: int, phase: int):
    key = (run_id, phase)
    task = _running_tasks.get(key)
    if task and not task.done():
        task.cancel()
        return {"ok": True}
    return {"ok": False, "detail": "실행 중인 Phase가 없습니다"}


# ── Phase 1 ──────────────────────────────────────────────────────────────────

@router.post("/api/runs/{run_id}/phase/1/run")
async def trigger_phase1(run_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM runs WHERE id=?", (run_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Run not found")
    finally:
        await db.close()
    get_queue(run_id, 1)
    _create_phase_task(run_phase1(run_id), run_id, 1)
    return {"ok": True}


@router.get("/api/runs/{run_id}/phase/1/stream")
async def stream_phase1(run_id: int):
    q = get_queue(run_id, 1)

    async def generator():
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Phase 2 ──────────────────────────────────────────────────────────────────

@router.post("/api/runs/{run_id}/phase/2/run")
async def trigger_phase2(run_id: int):
    db = await get_db()
    try:
        async with db.execute(
            "SELECT status FROM phase_results WHERE run_id=? AND phase=1",
            (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row or row["status"] != "completed":
            raise HTTPException(status_code=400, detail="Phase 1이 완료되지 않았습니다.")
    finally:
        await db.close()
    _create_phase_task(run_phase2(run_id), run_id, 2)
    return {"ok": True}


@router.get("/api/runs/{run_id}/phase/2/stream")
async def stream_phase2(run_id: int):
    q = get_queue(run_id, 2)

    async def generator():
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── 후보 선택 (Phase 2→3) ────────────────────────────────────────────────────

class SelectCandidateBody(BaseModel):
    candidate_id: int


@router.post("/api/runs/{run_id}/select-candidate")
async def select_candidate(run_id: int, body: SelectCandidateBody):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM runs WHERE id=?", (run_id,)) as cur:
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="Run not found")
        async with db.execute(
            "SELECT id FROM prompt_candidates WHERE id=? AND run_id=?",
            (body.candidate_id, run_id)
        ) as cur:
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="Candidate not found")
        await db.execute(
            "UPDATE runs SET selected_candidate_id=? WHERE id=?",
            (body.candidate_id, run_id)
        )
        await db.execute(
            "DELETE FROM dify_connections WHERE run_id=?", (run_id,)
        )
        await db.commit()
        return {"ok": True, "selected_candidate_id": body.candidate_id}
    finally:
        await db.close()


# ── Phase 3 ──────────────────────────────────────────────────────────────────

class DifyConnectBody(BaseModel):
    candidate_id: Optional[int] = None
    object_id: str          # Phase 2 설계 기반으로 생성한 Dify 워크플로우 고유 ID
    label: Optional[str] = None


@router.post("/api/runs/{run_id}/phase/3/connect")
async def connect_dify(run_id: int, body: DifyConnectBody):
    db = await get_db()
    try:
        verified, message = await verify_dify_connection(body.object_id)
        status = "verified" if verified else "failed"
        now = datetime.utcnow().isoformat() if verified else None

        # 같은 run의 기존 연결 삭제 후 새로 삽입 (재시도 지원)
        await db.execute("DELETE FROM dify_connections WHERE run_id=?", (run_id,))
        async with db.execute(
            """INSERT INTO dify_connections (run_id, candidate_id, object_id, label, status, verified_at)
               VALUES (?,?,?,?,?,?)""",
            (run_id, body.candidate_id, body.object_id, body.label, status, now)
        ) as cursor:
            conn_id = cursor.lastrowid
        await db.commit()
        return {"id": conn_id, "status": status, "verified": verified, "message": message}
    finally:
        await db.close()


@router.post("/api/runs/{run_id}/phase/3/execute")
async def execute_phase3(run_id: int):
    db = await get_db()
    try:
        async with db.execute(
            "SELECT status FROM phase_results WHERE run_id=? AND phase=2",
            (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row or row["status"] != "completed":
            raise HTTPException(status_code=400, detail="Phase 2가 완료되지 않았습니다.")

        async with db.execute(
            "SELECT id FROM dify_connections WHERE run_id=? AND status='verified'",
            (run_id,)
        ) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=400, detail="검증된 Dify 연결이 없습니다.")
    finally:
        await db.close()
    _create_phase_task(run_phase3(run_id), run_id, 3)
    return {"ok": True}


@router.get("/api/runs/{run_id}/phase/3/stream")
async def stream_phase3(run_id: int):
    q = get_queue(run_id, 3)

    async def generator():
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Phase 4 ──────────────────────────────────────────────────────────────────

@router.post("/api/runs/{run_id}/phase/4/run")
async def trigger_phase4(run_id: int):
    db = await get_db()
    try:
        async with db.execute(
            "SELECT status FROM phase_results WHERE run_id=? AND phase=3",
            (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row or row["status"] != "completed":
            raise HTTPException(status_code=400, detail="Phase 3이 완료되지 않았습니다.")
    finally:
        await db.close()
    _create_phase_task(run_phase4(run_id), run_id, 4)
    return {"ok": True}


@router.get("/api/runs/{run_id}/phase/4/stream")
async def stream_phase4(run_id: int):
    q = get_queue(run_id, 4)

    async def generator():
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Phase 5 ──────────────────────────────────────────────────────────────────

@router.get("/api/runs/{run_id}/phase/5")
async def get_phase5(run_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = await cursor.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        run = dict(run)

        scores = await aggregate_scores(run_id)

        async with db.execute(
            "SELECT * FROM runs WHERE task_id=? ORDER BY run_number",
            (run["task_id"],)
        ) as cursor:
            task_history = [
                {"run_id": r["id"], "run_number": r["run_number"],
                 "score_total": round((r["score_total"] or 0) * 100, 1),
                 "start_mode": r["start_mode"]}
                for r in await cursor.fetchall()
            ]

        async with db.execute(
            "SELECT delta_type, COUNT(*) as cnt FROM case_deltas WHERE to_run_id=? GROUP BY delta_type",
            (run_id,)
        ) as cursor:
            delta_rows = {r["delta_type"]: r["cnt"] for r in await cursor.fetchall()}

        async with db.execute(
            """SELECT d.case_id, d.prev_evaluation, d.curr_evaluation, c.reason
               FROM case_deltas d LEFT JOIN case_results c ON c.run_id=? AND c.case_id=d.case_id
               WHERE d.to_run_id=? AND d.delta_type='regressed'""",
            (run_id, run_id)
        ) as cursor:
            regressed_cases = [
                {"case_id": r["case_id"], "prev": r["prev_evaluation"],
                 "curr": r["curr_evaluation"], "reason": r["reason"] or ""}
                for r in await cursor.fetchall()
            ]

        goal_achieved = scores["score_total"] >= 95.0

        # BUG-3: 프론트 기대 구조로 전면 수정
        output = {
            "scores": {
                "correct_plus_over": scores["score_total"],
                "correct": scores["score_correct"],
                "over": scores["score_over"],
                "wrong": round(100 - scores["score_correct"] - scores["score_over"], 1),
            },
            "delta": {
                "improve": delta_rows.get("improved", 0),
                "regress": delta_rows.get("regressed", 0),
                "same": delta_rows.get("unchanged", 0),
            },
            "trend": {
                "labels": [f"Run {r['run_number']}" for r in task_history],
                "values": [round((r["score_total"] or 0) * 100, 1) for r in task_history],
            },
            "regressed_cases": [
                {
                    "id": r["case_id"],
                    "prev_judge": r["prev"],
                    "curr_judge": r["curr"],
                    "reason": r["reason"],
                }
                for r in regressed_cases
            ],
            "goal_achieved": goal_achieved,
            "gap_to_goal": round(max(0, 95.0 - scores["score_total"]), 1),
        }

        await db.execute(
            """INSERT INTO phase_results (run_id, phase, status, output_data, started_at, completed_at)
               VALUES (?,5,'completed',?,?,?)
               ON CONFLICT(run_id, phase) DO UPDATE SET status='completed', output_data=excluded.output_data, completed_at=excluded.completed_at""",
            (run_id, json.dumps(output), datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
        )
        await db.execute("UPDATE runs SET status='phase5_done' WHERE id=?", (run_id,))
        await db.commit()

        return output
    finally:
        await db.close()


# ── Phase 6 ──────────────────────────────────────────────────────────────────

@router.post("/api/runs/{run_id}/phase/6/run")
async def trigger_phase6(run_id: int):
    db = await get_db()
    try:
        async with db.execute(
            "SELECT status FROM phase_results WHERE run_id=? AND phase=4",
            (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row or row["status"] != "completed":
            raise HTTPException(status_code=400, detail="Phase 4가 완료되지 않았습니다.")
    finally:
        await db.close()
    _create_phase_task(run_phase6(run_id), run_id, 6)
    return {"ok": True}


@router.get("/api/runs/{run_id}/phase/6/stream")
async def stream_phase6(run_id: int):
    q = get_queue(run_id, 6)

    async def generator():
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
