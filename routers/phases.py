import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from database import get_db
from services.phase1_analysis import run_phase1
from services.phase2_design import run_phase2
from services.phase3_dify import run_phase3, verify_dify_connection
from services.phase4_judge import run_phase4
from services.phase6_strategy import run_phase6
from services.delta import aggregate_scores

router = APIRouter(tags=["phases"])

# 각 run_id별 스트림 큐 저장
_stream_queues: dict[int, dict[int, asyncio.Queue]] = {}


def get_queue(run_id: int, phase: int) -> asyncio.Queue:
    if run_id not in _stream_queues:
        _stream_queues[run_id] = {}
    if phase not in _stream_queues[run_id]:
        _stream_queues[run_id][phase] = asyncio.Queue()
    return _stream_queues[run_id][phase]


async def _run_and_queue(generator, run_id: int, phase: int):
    q = get_queue(run_id, phase)
    try:
        async for event in generator:
            await q.put(event)
    finally:
        await q.put(None)  # sentinel


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
    q = get_queue(run_id, 1)
    asyncio.create_task(_run_and_queue(run_phase1(run_id), run_id, 1))
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
    asyncio.create_task(_run_and_queue(run_phase2(run_id), run_id, 2))
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


# ── Phase 3 ──────────────────────────────────────────────────────────────────

class DifyConnectBody(BaseModel):
    candidate_id: Optional[int] = None
    dify_api_url: str
    dify_api_key: str
    label: Optional[str] = None


@router.post("/api/runs/{run_id}/phase/3/connect")
async def connect_dify(run_id: int, body: DifyConnectBody):
    db = await get_db()
    try:
        verified = await verify_dify_connection(body.dify_api_url, body.dify_api_key)
        status = "verified" if verified else "failed"
        now = datetime.utcnow().isoformat() if verified else None

        async with db.execute(
            """INSERT INTO dify_connections (run_id, candidate_id, dify_api_url, dify_api_key, label, status, verified_at)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, body.candidate_id, body.dify_api_url, body.dify_api_key,
             body.label, status, now)
        ) as cursor:
            conn_id = cursor.lastrowid
        await db.commit()
        return {"id": conn_id, "status": status, "verified": verified}
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
    asyncio.create_task(_run_and_queue(run_phase3(run_id), run_id, 3))
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
    asyncio.create_task(_run_and_queue(run_phase4(run_id), run_id, 4))
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

        # Phase 5 결과 저장
        output = {
            "current_run": {
                "score_total": scores["score_total"],
                "score_correct": scores["score_correct"],
                "score_over": scores["score_over"],
                "score_wrong": round(100 - scores["score_correct"] - scores["score_over"], 1)
            },
            "task_history": task_history,
            "delta_summary": {
                "improved": delta_rows.get("improved", 0),
                "regressed": delta_rows.get("regressed", 0),
                "unchanged": delta_rows.get("unchanged", 0)
            },
            "regressed_cases": regressed_cases,
            "goal_achieved": goal_achieved,
            "gap_to_goal": round(max(0, 95.0 - scores["score_total"]), 1)
        }

        await db.execute(
            """INSERT OR REPLACE INTO phase_results
               (run_id, phase, status, output_data, started_at, completed_at)
               VALUES (?,5,'completed',?,?,?)""",
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
    asyncio.create_task(_run_and_queue(run_phase6(run_id), run_id, 6))
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
