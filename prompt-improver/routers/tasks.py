from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from database import get_db

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskCreate(BaseModel):
    name: str
    description: Optional[str] = None
    generation_task: Optional[str] = None


@router.get("")
async def list_tasks():
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM tasks ORDER BY created_at DESC") as cursor:
            tasks = [dict(row) for row in await cursor.fetchall()]
        for task in tasks:
            async with db.execute(
                "SELECT id, run_number, status, score_total, start_mode, base_run_id, created_at FROM runs WHERE task_id=? ORDER BY run_number",
                (task["id"],)
            ) as cursor:
                task["runs"] = [dict(row) for row in await cursor.fetchall()]
        return tasks
    finally:
        await db.close()


@router.post("")
async def create_task(body: TaskCreate):
    db = await get_db()
    try:
        async with db.execute(
            "INSERT INTO tasks (name, description, generation_task) VALUES (?,?,?)",
            (body.name, body.description, body.generation_task)
        ) as cursor:
            task_id = cursor.lastrowid
        await db.commit()
        async with db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)) as cursor:
            return dict(await cursor.fetchone())
    finally:
        await db.close()


@router.get("/{task_id}")
async def get_task(task_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)) as cursor:
            task = await cursor.fetchone()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task = dict(task)
        async with db.execute(
            "SELECT * FROM runs WHERE task_id=? ORDER BY run_number",
            (task_id,)
        ) as cursor:
            task["runs"] = [dict(row) for row in await cursor.fetchall()]
        return task
    finally:
        await db.close()


class TaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    generation_task: Optional[str] = None


@router.patch("/{task_id}")
async def update_task(task_id: int, body: TaskUpdate):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM tasks WHERE id=?", (task_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Task not found")

        updates = []
        values = []
        if body.name is not None:
            updates.append("name=?")
            values.append(body.name)
        if body.description is not None:
            updates.append("description=?")
            values.append(body.description)
        if body.generation_task is not None:
            updates.append("generation_task=?")
            values.append(body.generation_task)

        if updates:
            values.append(task_id)
            await db.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", values)
            await db.commit()

        async with db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)) as cursor:
            return dict(await cursor.fetchone())
    finally:
        await db.close()


@router.delete("/{task_id}")
async def delete_task(task_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM tasks WHERE id=?", (task_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Task not found")

        # 연관 데이터 cascade 삭제 (run_id 목록 먼저 수집)
        async with db.execute("SELECT id FROM runs WHERE task_id=?", (task_id,)) as cursor:
            run_ids = [row["id"] for row in await cursor.fetchall()]

        for run_id in run_ids:
            await db.execute("DELETE FROM case_results WHERE run_id=?", (run_id,))
            await db.execute("DELETE FROM phase_results WHERE run_id=?", (run_id,))
            await db.execute("DELETE FROM prompt_candidates WHERE run_id=?", (run_id,))
            await db.execute("DELETE FROM dify_connections WHERE run_id=?", (run_id,))

        await db.execute("DELETE FROM case_deltas WHERE task_id=?", (task_id,))
        await db.execute("DELETE FROM runs WHERE task_id=?", (task_id,))
        await db.execute("DELETE FROM tasks WHERE id=?", (task_id,))

        # SQLite autoincrement 카운터 리셋 (다음 id = 현재 최대+1)
        for tbl in ("tasks", "runs", "case_results", "phase_results", "prompt_candidates", "dify_connections", "case_deltas"):
            await db.execute(
                f"UPDATE sqlite_sequence SET seq = (SELECT COALESCE(MAX(id), 0) FROM {tbl}) WHERE name=?",
                (tbl,)
            )

        await db.commit()
        return {"ok": True}
    finally:
        await db.close()
