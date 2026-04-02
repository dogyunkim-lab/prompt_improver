import json
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from database import get_db
from services.phase2_design import _build_candidates_with_nodes

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

        # continue 모드: 이전 Run의 Phase 4 결과를 Judge JSON 형식으로 생성 + 선택된 프롬프트 자동 이관
        judge_file_path = None
        prompt_file_path = None
        if body.start_mode == 'continue' and body.base_run_id:
            async with db.execute(
                """SELECT case_id, generation_task, stt, reference, keywords,
                          generated, evaluation, reason
                   FROM case_results WHERE run_id=? ORDER BY rowid""",
                (body.base_run_id,)
            ) as cursor:
                base_cases = [dict(row) for row in await cursor.fetchall()]

            if base_cases:
                # 실제 Judge process_single_case 출력 형식으로 변환
                judge_json = [
                    {
                        "id": c["case_id"],
                        "generation_task": c["generation_task"] or "",
                        "stt": c["stt"] or "",
                        "reference": c["reference"] or "",
                        "keywords": c["keywords"] or "",
                        "generated": c["generated"] or "",
                        "reasoning_effort": "",
                        "reasoning_effort_result": "",
                        "answer_evaluation": c["evaluation"] or "",
                        "answer_evaluation_reason": c["reason"] or "",
                    }
                    for c in base_cases
                ]
                os.makedirs("data/uploads", exist_ok=True)
                judge_file_path = f"data/uploads/run_{run_number}_from_run_{body.base_run_id}_judge.json"
                with open(judge_file_path, "w", encoding="utf-8") as f:
                    json.dump(judge_json, f, ensure_ascii=False, indent=2)

            # 이전 Run에서 선택된 후보 프롬프트 자동 이관
            async with db.execute(
                "SELECT selected_candidate_id FROM runs WHERE id=?",
                (body.base_run_id,)
            ) as cursor:
                base_run_row = await cursor.fetchone()

            if base_run_row and base_run_row["selected_candidate_id"]:
                async with db.execute(
                    "SELECT * FROM prompt_candidates WHERE id=?",
                    (base_run_row["selected_candidate_id"],)
                ) as cursor:
                    cand = await cursor.fetchone()

                if cand:
                    cand = dict(cand)
                    # 워크플로우 구조 + 노드별 프롬프트를 텍스트 파일로 구성
                    lines = []
                    lines.append(f"# 이전 Run #{body.base_run_id} 선택 프롬프트 (후보 {cand['candidate_label']})")
                    lines.append(f"# 설계 근거: {cand.get('design_rationale', '')}")
                    lines.append(f"# 워크플로우 노드 수: {cand.get('node_count', 1)}")
                    lines.append("")

                    node_idx = 0
                    for label, prompt_key, model_key, reasoning_key in [
                        ("A", "node_a_prompt", "node_a_model", "node_a_reasoning"),
                        ("B", "node_b_prompt", "node_b_model", "node_b_reasoning"),
                        ("C", "node_c_prompt", "node_c_model", "node_c_reasoning"),
                    ]:
                        prompt_text = cand.get(prompt_key)
                        if prompt_text:
                            node_idx += 1
                            model = cand.get(model_key, "qwen3-30b")
                            reasoning = "ON" if cand.get(reasoning_key) else "OFF"
                            lines.append(f"=== Node {label} (모델: {model}, 추론: {reasoning}) ===")
                            # system/user 분리 표시
                            sys_p = cand.get(f"node_{label.lower()}_system_prompt")
                            usr_p = cand.get(f"node_{label.lower()}_user_prompt")
                            if sys_p:
                                lines.append("[SYSTEM PROMPT]")
                                lines.append(sys_p.strip())
                            if usr_p:
                                lines.append("[USER PROMPT]")
                                lines.append(usr_p.strip())
                            if not sys_p and not usr_p:
                                lines.append(prompt_text.strip())  # 하위 호환
                            lines.append("")

                    if node_idx > 0:
                        prompt_content = "\n".join(lines)
                        os.makedirs("data/uploads", exist_ok=True)
                        prompt_file_path = f"data/uploads/run_{run_number}_from_run_{body.base_run_id}_prompt.txt"
                        with open(prompt_file_path, "w", encoding="utf-8") as f:
                            f.write(prompt_content)

        # 자동 생성 파일의 표시 이름
        judge_display = f"Phase 4 판정결과 (Run #{body.base_run_id})" if judge_file_path and body.base_run_id else None
        prompt_display = f"Run #{body.base_run_id} 선택 프롬프트 이관" if prompt_file_path and body.base_run_id else None

        async with db.execute(
            """INSERT INTO runs (task_id, run_number, start_mode, base_run_id, status,
               judge_file_path, prompt_file_path, judge_original_name, prompt_original_name)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (task_id, run_number, body.start_mode, body.base_run_id, "created",
             judge_file_path, prompt_file_path, judge_display, prompt_display)
        ) as cursor:
            run_id = cursor.lastrowid
        await db.commit()

        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            return dict(await cursor.fetchone())
    finally:
        await db.close()


@router.delete("/api/runs/{run_id}")
async def delete_run(run_id: int):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM runs WHERE id=?", (run_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Run not found")

        # 삭제 전 task_id 조회 (run_number 재정렬용)
        async with db.execute("SELECT task_id FROM runs WHERE id=?", (run_id,)) as cur:
            task_id = (await cur.fetchone())["task_id"]

        await db.execute("DELETE FROM case_results WHERE run_id=?", (run_id,))
        await db.execute("DELETE FROM phase_results WHERE run_id=?", (run_id,))
        await db.execute("DELETE FROM prompt_candidates WHERE run_id=?", (run_id,))
        await db.execute("DELETE FROM dify_connections WHERE run_id=?", (run_id,))
        await db.execute("DELETE FROM case_deltas WHERE from_run_id=? OR to_run_id=?", (run_id, run_id))
        await db.execute("DELETE FROM runs WHERE id=?", (run_id,))

        # 남은 Run들의 run_number를 1부터 순차 재정렬
        async with db.execute(
            "SELECT id FROM runs WHERE task_id=? ORDER BY run_number", (task_id,)
        ) as cur:
            remaining = [row["id"] for row in await cur.fetchall()]
        for idx, rid in enumerate(remaining, 1):
            await db.execute("UPDATE runs SET run_number=? WHERE id=?", (idx, rid))

        # SQLite autoincrement 카운터 리셋 (다음 id = 현재 최대+1)
        for tbl in ("runs", "case_results", "phase_results", "prompt_candidates", "dify_connections", "case_deltas"):
            await db.execute(
                f"UPDATE sqlite_sequence SET seq = (SELECT COALESCE(MAX(id), 0) FROM {tbl}) WHERE name=?",
                (tbl,)
            )

        await db.commit()
        return {"ok": True}
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
            candidates_with_nodes = _build_candidates_with_nodes(candidates)
            # Phase 2 데이터에 candidates 주입 (output_data에 없을 경우 대비)
            if 2 not in phases:
                phases[2] = {"status": "completed"}
            if not phases[2].get("candidates"):
                phases[2]["candidates"] = candidates_with_nodes

        # Phase 1 케이스 결과 포함 (페이지 새로고침 시 테이블 복원)
        if 1 in phases:
            try:
                async with db.execute(
                    """SELECT case_id, stt, reference, generated, evaluation, reason,
                              bucket, analysis_summary, stt_uncertain,
                              hallucination_detected, judge_agreement, judge_disagreement
                       FROM case_results WHERE run_id=? ORDER BY rowid""",
                    (run_id,)
                ) as cursor:
                    case_rows = [dict(row) for row in await cursor.fetchall()]
                if case_rows:
                    phases[1]["cases"] = [{
                        "id": r["case_id"],
                        "judge": r["evaluation"] or "",
                        "bucket": r["bucket"] or "",
                        "analysis_summary": r["analysis_summary"] or "",
                        "stt_uncertain": r["stt_uncertain"] or "",
                        "stt": r["stt"] or "",
                        "reference": r["reference"] or "",
                        "generated": r["generated"] or "",
                        "judge_disagreement": (r.get("judge_disagreement") or r["reason"] or ""),
                        "hallucination": bool(r.get("hallucination_detected", 0)),
                    } for r in case_rows]
                    # eval_chart fallback: DB에 저장 안 된 구 데이터 호환
                    if not phases[1].get("eval_chart") and case_rows:
                        correct_n = sum(1 for r in case_rows if r["evaluation"] == "정답")
                        over_n    = sum(1 for r in case_rows if r["evaluation"] == "과답")
                        wrong_n   = sum(1 for r in case_rows if r["evaluation"] == "오답")
                        phases[1]["eval_chart"] = {
                            "labels": ["정답", "과답", "오답"],
                            "values": [correct_n, over_n, wrong_n],
                        }
                    # bucket_chart fallback
                    if not phases[1].get("bucket_chart") and case_rows:
                        bc = {"stt_error": 0, "prompt_missing": 0, "model_behavior": 0, "judge_dispute": 0}
                        for r in case_rows:
                            b = r["bucket"] or ""
                            if b in bc:
                                bc[b] += 1
                        phases[1]["bucket_chart"] = {
                            "labels": ["STT 오류", "프롬프트 누락", "모델 동작", "Judge 이견"],
                            "values": [bc["stt_error"], bc["prompt_missing"], bc["model_behavior"], bc["judge_dispute"]],
                        }
            except Exception:
                pass  # 마이그레이션 전 구 DB 호환

        # Phase 4 케이스 결과 포함 (Judge 판정 테이블 복원)
        if 4 in phases:
            try:
                async with db.execute(
                    """SELECT case_id, stt, reference, generated, evaluation, reason, intermediate_outputs
                       FROM case_results WHERE run_id=? ORDER BY rowid""",
                    (run_id,)
                ) as cursor:
                    p4_rows = [dict(row) for row in await cursor.fetchall()]
                if p4_rows:
                    phases[4]["cases"] = [{
                        "id": r["case_id"],
                        "judge": r["evaluation"] or "",
                        "reason": r["reason"] or "",
                        "stt": r["stt"] or "",
                        "reference": r["reference"] or "",
                        "generated": r["generated"] or "",
                        "intermediate_outputs": json.loads(r["intermediate_outputs"]) if r.get("intermediate_outputs") else {},
                    } for r in p4_rows]
            except Exception:
                pass

        # Dify connections 포함 (Phase 3 복원용)
        async with db.execute(
            "SELECT * FROM dify_connections WHERE run_id=? ORDER BY id",
            (run_id,)
        ) as cursor:
            run["dify_connections"] = [dict(row) for row in await cursor.fetchall()]

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


class UserGuideBody(BaseModel):
    user_guide: str = ""


@router.post("/api/runs/{run_id}/user-guide")
async def save_user_guide(run_id: int, body: UserGuideBody):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM runs WHERE id=?", (run_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Run not found")
        await db.execute("UPDATE runs SET user_guide=? WHERE id=?", (body.user_guide, run_id))
        await db.commit()
        return {"ok": True}
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
            "UPDATE runs SET judge_file_path=?, judge_original_name=? WHERE id=?",
            (file_path, file.filename, run_id)
        )
        await db.commit()
        return {"ok": True, "file_path": file_path, "original_name": file.filename}
    finally:
        await db.close()


@router.post("/api/runs/{run_id}/upload-prompt")
async def upload_prompt(run_id: int, file: UploadFile = File(...)):
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM runs WHERE id=?", (run_id,)) as cursor:
            if not await cursor.fetchone():
                raise HTTPException(status_code=404, detail="Run not found")

        os.makedirs("data/uploads", exist_ok=True)
        file_path = f"data/uploads/run_{run_id}_prompt.txt"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        await db.execute(
            "UPDATE runs SET prompt_file_path=?, prompt_original_name=? WHERE id=?",
            (file_path, file.filename, run_id)
        )
        await db.commit()
        return {"ok": True, "file_path": file_path, "original_name": file.filename}
    finally:
        await db.close()
