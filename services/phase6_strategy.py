import json
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.delta import compute_learning_rate, get_run_scores
from services.sse_helpers import log_event, result_event, done_event

PROMPT_PATH = "prompts/phase6_strategy.txt"


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


async def run_phase6(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = dict(await cursor.fetchone())

        await db.execute(
            """INSERT INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,6,'running',?)
               ON CONFLICT(run_id, phase) DO UPDATE SET status='running', started_at=excluded.started_at""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()
        await db.execute("UPDATE runs SET status='phase6_running', current_phase=6 WHERE id=?", (run_id,))
        await db.commit()

        current_score = run.get("score_total", 0) or 0
        learning_rate = await compute_learning_rate(run["task_id"], run_id)

        yield log_event("info", f"현재 점수: {round(current_score * 100, 1)}%, Learning rate: {learning_rate}")

        # 전체 실험 이력
        history_runs = await get_run_scores(run["task_id"], run_id)
        history_lines = []
        for hr in history_runs:
            history_lines.append(
                f"Run {hr['run_number']} (id={hr['id']}): score_total={hr['score_total']}%"
            )
        experiment_history = "\n".join(history_lines) if history_lines else "이전 실험 없음"

        # Delta 분석
        async with db.execute(
            """SELECT d.case_id, d.prev_evaluation, d.curr_evaluation, d.delta_type
               FROM case_deltas d
               WHERE d.to_run_id=?""",
            (run_id,)
        ) as cursor:
            deltas = [dict(row) for row in await cursor.fetchall()]

        improved = [d for d in deltas if d["delta_type"] == "improved"]
        regressed = [d for d in deltas if d["delta_type"] == "regressed"]

        delta_summary = f"개선: {len(improved)}건, 회귀: {len(regressed)}건"
        delta_analysis = json.dumps(deltas[:50], ensure_ascii=False)  # 최대 50건

        # 회귀 케이스 상세
        regression_details = []
        for r in regressed[:10]:
            async with db.execute(
                "SELECT reason FROM case_results WHERE run_id=? AND case_id=?",
                (run_id, r["case_id"])
            ) as cursor:
                reason_row = await cursor.fetchone()
            regression_details.append({
                "case_id": r["case_id"],
                "prev": r["prev_evaluation"],
                "curr": r["curr_evaluation"],
                "reason": reason_row["reason"] if reason_row else ""
            })
        regression_analysis = json.dumps(regression_details, ensure_ascii=False)

        yield log_event("info", f"Delta 분석: {delta_summary}")
        yield log_event("info", "gpt-oss-120B에게 전략 수립 요청 중...")

        prompt_template = load_prompt()
        prompt = prompt_template.format(
            current_score=round(current_score * 100, 1),
            learning_rate=learning_rate,
            experiment_history=experiment_history,
            delta_analysis=delta_analysis,
            regression_analysis=regression_analysis
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            raw = await call_gpt(messages, reasoning="high")
            result = _extract_json(raw)
        except Exception as e:
            yield log_event("error", f"GPT 호출 실패: {e}")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 6)
            return

        yield log_event("ok", f"전략 수립 완료: {result.get('strategy_type', 'unknown')}")

        # BUG-6: 프론트 기대 필드명으로 매핑
        frontend_result = {
            "learning_rate": learning_rate,
            "backprop": result.get("backprop_analysis", ""),
            "effective": result.get("effective_elements", []),
            "harmful": result.get("harmful_elements", []),
            "next_direction": result.get("next_direction", ""),
        }

        await db.execute(
            "UPDATE phase_results SET status='completed', output_data=?, completed_at=? WHERE run_id=? AND phase=6",
            (json.dumps(frontend_result, ensure_ascii=False), datetime.utcnow().isoformat(), run_id)
        )
        await db.execute(
            "UPDATE runs SET status='completed', completed_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), run_id)
        )
        await db.commit()

        yield result_event(frontend_result)
        yield done_event("completed")

    except Exception as e:
        yield log_event("error", f"Phase 6 오류: {e}")
        yield done_event("failed")
        await _mark_phase_failed(run_id, 6)
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
