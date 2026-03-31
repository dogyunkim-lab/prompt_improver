import json
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.delta import compute_learning_rate, count_completed_runs, get_run_scores
from services.sse_helpers import log_event, result_event, done_event

PROMPT_PATH = "prompts/phase2_design.txt"


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _build_candidates_with_nodes(saved_candidates: list) -> list:
    """DB rows → node_prompts 배열 변환 (BUG-7 공유 로직)"""
    result = []
    for cand in saved_candidates:
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
        result.append({
            "id": cand["id"],
            "label": cand["candidate_label"],
            "node_count": cand.get("node_count", len(node_prompts)),
            "rationale": cand.get("design_rationale", ""),
            "node_prompts": node_prompts,
        })
    return result


def _extract_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


async def run_phase2(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = dict(await cursor.fetchone())

        async with db.execute("SELECT * FROM tasks WHERE id=?", (run["task_id"],)) as cursor:
            task = dict(await cursor.fetchone())

        # Phase 1 결과 조회
        async with db.execute(
            "SELECT output_data FROM phase_results WHERE run_id=? AND phase=1",
            (run_id,)
        ) as cursor:
            p1_row = await cursor.fetchone()

        if not p1_row:
            yield log_event("error", "Phase 1이 완료되지 않았습니다.")
            yield done_event("failed")
            return

        phase1_summary = json.loads(p1_row["output_data"] or "{}")

        await db.execute(
            """INSERT OR REPLACE INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,2,'running',?)""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()
        await db.execute("UPDATE runs SET status='phase2_running', current_phase=2 WHERE id=?", (run_id,))
        await db.commit()

        yield log_event("info", "Design mode 결정 중...")

        prev_completed = await count_completed_runs(run["task_id"], run_id)
        design_mode = "explore" if prev_completed == 0 else "converge"
        learning_rate = await compute_learning_rate(run["task_id"], run_id)

        yield log_event("info", f"Design mode: {design_mode}, Learning rate: {learning_rate}")

        # 이전 실험 이력
        history_runs = await get_run_scores(run["task_id"], run_id)
        if history_runs:
            history_lines = []
            for hr in history_runs:
                history_lines.append(f"Run {hr['run_number']}: score_total={hr['score_total']}%")
            experiment_history = "\n".join(history_lines)
        else:
            experiment_history = "첫 번째 실험"

        # 개선 가능 케이스 수
        improvable_cases = phase1_summary.get("prompt_improvable_cases", [])
        improvable_count = len(improvable_cases)

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM case_results WHERE run_id=?",
            (run_id,)
        ) as cursor:
            total_row = await cursor.fetchone()
            total_count = total_row["cnt"]

        yield log_event("info", f"프롬프트 설계 시작 (개선 가능: {improvable_count}/{total_count}건)")

        prompt_template = load_prompt()
        prompt = prompt_template.format(
            generation_task=task.get("generation_task", "불편사항 요약"),
            phase1_summary=json.dumps(phase1_summary, ensure_ascii=False, indent=2),
            improvable_count=improvable_count,
            total=total_count,
            experiment_history=experiment_history,
            learning_rate=learning_rate
        )

        messages = [{"role": "user", "content": prompt}]
        yield log_event("info", "gpt-oss-120B에게 프롬프트 설계 요청 중...")

        try:
            raw = await call_gpt(messages, reasoning="high")
            result = _extract_json(raw)
        except Exception as e:
            yield log_event("error", f"GPT 호출 실패: {e}")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 2)
            return

        candidates = result.get("candidates", [])
        if not candidates:
            yield log_event("error", "프롬프트 후보 생성 실패")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 2)
            return

        # 후보 저장
        for cand in candidates:
            await db.execute(
                """INSERT INTO prompt_candidates
                   (run_id, candidate_label, mode, workflow_spec, node_count,
                    node_a_prompt, node_b_prompt, node_c_prompt,
                    node_a_reasoning, node_b_reasoning, node_c_reasoning, design_rationale)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, cand.get("label", "A"), design_mode,
                    cand.get("workflow_spec", ""), cand.get("node_count", 1),
                    cand.get("node_a_prompt"), cand.get("node_b_prompt"), cand.get("node_c_prompt"),
                    1 if cand.get("node_a_reasoning") else 0,
                    1 if cand.get("node_b_reasoning") else 0,
                    1 if cand.get("node_c_reasoning") else 0,
                    cand.get("rationale", "")
                )
            )
        await db.commit()

        yield log_event("ok", f"{len(candidates)}개 프롬프트 후보 설계 완료")

        # BUG-5: DB에서 저장된 candidates 조회 후 node_prompts 배열로 변환
        async with db.execute(
            "SELECT * FROM prompt_candidates WHERE run_id=? ORDER BY candidate_label",
            (run_id,)
        ) as cursor:
            saved_candidates = [dict(row) for row in await cursor.fetchall()]

        candidates_with_nodes = _build_candidates_with_nodes(saved_candidates)

        output = {
            "mode": design_mode,
            "learning_rate": learning_rate,
            "candidate_count": len(candidates),
            "design_summary": result.get("design_summary", ""),
            "spec_summary": result.get("spec_summary", ""),
            "candidates": candidates_with_nodes,
        }

        async with db.execute(
            "UPDATE phase_results SET status='completed', output_data=?, completed_at=? WHERE run_id=? AND phase=2",
            (json.dumps(output, ensure_ascii=False), datetime.utcnow().isoformat(), run_id)
        ):
            pass
        await db.execute("UPDATE runs SET status='phase2_done' WHERE id=?", (run_id,))
        await db.commit()

        yield result_event(output)
        yield done_event("completed")

    except Exception as e:
        yield log_event("error", f"Phase 2 오류: {e}")
        yield done_event("failed")
        await _mark_phase_failed(run_id, 2)
    finally:
        await db.close()


async def _mark_phase_failed(run_id: int, phase: int):
    db = await get_db()
    try:
        await db.execute(
            "UPDATE phase_results SET status='failed', completed_at=? WHERE run_id=? AND phase=?",
            (datetime.utcnow().isoformat(), run_id, phase)
        )
        await db.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))
        await db.commit()
    finally:
        await db.close()
