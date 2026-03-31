import json
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from database import get_db

router = APIRouter(tags=["runs"])


class RunCreate(BaseModel):
    start_mode: str  # 'zero' | 'continue'
    base_run_id: Optional[int] = None


@router.get("/api/tasks/{task_id}/runs")
async def list_runs(task_id: int):
    db = await get_db()
    try:
        async with db.execute(
            "SELECT * FROM runs WHERE task_id=? ORDER BY run_number",
            (task_id,)
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
    finally:
        await db.close()


@router.post("/api/tasks/{task_id}/runs")
async def create_run(task_id: int, body: RunCreate):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM tasks WHERE id=?", (task_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Task not found")

        async with db.execute(
            "SELECT COALESCE(MAX(run_number), 0) + 1 as next_num FROM runs WHERE task_id=?",
            (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
            run_number = row["next_num"]

        async with db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode, base_run_id, status) VALUES (?,?,?,?,?)",
            (task_id, run_number, body.start_mode, body.base_run_id, "created")
        ) as cursor:
            run_id = cursor.lastrowid
        await db.commit()

        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            return dict(await cursor.fetchone())
    finally:
        await db.close()


@router.get("/api/runs/{run_id}")
async def get_run(run_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = await cursor.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        run = dict(run)

        # BUG-4: phase_results → phases dict 변환 (output_data JSON 파싱 포함)
        async with db.execute(
            "SELECT * FROM phase_results WHERE run_id=? ORDER BY phase",
            (run_id,)
        ) as cursor:
            phase_rows = [dict(row) for row in await cursor.fetchall()]

        phases = {}
        for pr in phase_rows:
            phase_num = pr["phase"]
            output_data = {}
            if pr.get("output_data"):
                try:
                    output_data = json.loads(pr["output_data"])
                except Exception:
                    pass
            phases[phase_num] = {"status": pr["status"], **output_data}

        # BUG-7: prompt_candidates flat columns → node_prompts 배열 변환
        async with db.execute(
            "SELECT * FROM prompt_candidates WHERE run_id=? ORDER BY candidate_label",
            (run_id,)
        ) as cursor:
            candidates = [dict(row) for row in await cursor.fetchall()]

        if candidates:
            candidates_with_nodes = []
            for cand in candidates:
                node_prompts = []
                for label, content_key, reason_key in [
                    ("A", "node_a_prompt", "node_a_reasoning"),
                    ("B", "node_b_prompt", "node_b_reasoning"),
                    ("C", "node_c_prompt", "node_c_reasoning"),
                ]:
                    if cand.get(content_key):
                        node_prompts.append({
                            "label": label,
                            "content": cand[content_key],
                            "reasoning": bool(cand.get(reason_key)),
                        })
                candidates_with_nodes.append({
                    "id": cand["id"],
                    "label": cand["candidate_label"],
                    "node_count": cand.get("node_count", len(node_prompts)),
                    "rationale": cand.get("design_rationale", ""),
                    "node_prompts": node_prompts,
                })
            # Phase 2 데이터에 candidates 주입 (output_data에 없을 경우 대비)
            if 2 not in phases:
                phases[2] = {"status": "completed"}
            if not phases[2].get("candidates"):
                phases[2]["candidates"] = candidates_with_nodes

        run["phases"] = phases
        return run
    finally:
        await db.close()


@router.get("/api/runs/{run_id}/summary")
async def get_run_summary(run_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = await cursor.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        run = dict(run)

        async with db.execute(
            "SELECT * FROM runs WHERE task_id=? ORDER BY run_number",
            (run["task_id"],)
        ) as cursor:
            history = [dict(row) for row in await cursor.fetchall()]

        return {
            "current_run": run,
            "task_history": history
        }
    finally:
        await db.close()


@router.post("/api/runs/{run_id}/upload-judge")
async def upload_judge(run_id: int, file: UploadFile = File(...)):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM runs WHERE id=?", (run_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Run not found")

        os.makedirs("data/uploads", exist_ok=True)
        file_path = f"data/uploads/run_{run_id}_judge.json"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        await db.execute(
            "UPDATE runs SET judge_file_path=? WHERE id=?",
            (file_path, run_id)
        )
        await db.commit()
        return {"ok": True, "file_path": file_path}
    finally:
        await db.close()
