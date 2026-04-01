import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.delta import compute_and_save_deltas, aggregate_scores
from services.sse_helpers import log_event, progress_event, result_event, done_event

SYSTEM_PROMPT_PATH = "prompts/phase4_judge.txt"
USER_PROMPT_PATH = "prompts/phase4_judge_user.txt"
JUDGE_CONCURRENCY = 5


def _load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _extract_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


async def run_phase4(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = dict(await cursor.fetchone())

        async with db.execute(
            "SELECT * FROM case_results WHERE run_id=? AND generated IS NOT NULL AND generated != ''",
            (run_id,)
        ) as cursor:
            cases = [dict(row) for row in await cursor.fetchall()]

        if not cases:
            yield log_event("error", "생성된 요약 데이터가 없습니다. Phase 3을 먼저 실행하세요.")
            yield done_event("failed")
            return

        await db.execute(
            """INSERT INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,4,'running',?)
               ON CONFLICT(run_id, phase) DO UPDATE SET status='running', started_at=excluded.started_at""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()
        await db.execute("UPDATE runs SET status='phase4_running', current_phase=4 WHERE id=?", (run_id,))
        await db.commit()

        judge_system = _load_prompt(SYSTEM_PROMPT_PATH)
        user_template = _load_prompt(USER_PROMPT_PATH)
        if not user_template:
            yield log_event("error", f"User 프롬프트 템플릿이 없습니다: {USER_PROMPT_PATH}")
            yield done_event("failed")
            return

        total = len(cases)
        yield log_event("info", f"Judge 실행 시작: {total}건")

        semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
        done_count = 0

        async def judge_case(case: dict):
            nonlocal done_count
            async with semaphore:
                user_content = user_template.format(
                    stt=case.get("stt", ""),
                    generation_task=case.get("generation_task", ""),
                    reference=case.get("reference", ""),
                    keywords=case.get("keywords", ""),
                    generated=case.get("generated", ""),
                )
                messages = []
                if judge_system:
                    messages.append({"role": "system", "content": judge_system})
                messages.append({"role": "user", "content": user_content})
                try:
                    raw = await call_gpt(messages, reasoning="low")
                    result = _extract_json(raw)
                    evaluation = result.get("rating", result.get("evaluation", "오답"))
                    reason = result.get("reason", "")
                    await db.execute(
                        "UPDATE case_results SET evaluation=?, reason=? WHERE run_id=? AND case_id=?",
                        (evaluation, reason, run_id, case["case_id"])
                    )
                    await db.commit()
                    done_count += 1
                    return evaluation
                except Exception as e:
                    done_count += 1
                    return f"error:{e}"

        tasks_list = [judge_case(c) for c in cases]
        processed = 0
        for coro in asyncio.as_completed(tasks_list):
            eval_result = await coro
            processed += 1
            if not str(eval_result).startswith("error:"):
                yield log_event("ok", f"판정: {eval_result}")
            else:
                yield log_event("warn", f"Judge 실패: {eval_result}")
            yield progress_event(processed, total)

        # 점수 집계
        scores = await aggregate_scores(run_id)
        yield log_event("ok", f"점수 집계 완료 — 정답+과답: {scores['score_total']}% (정답:{scores['score_correct']}% 과답:{scores['score_over']}%)")

        # runs 점수 업데이트
        await db.execute(
            "UPDATE runs SET score_correct=?, score_over=?, score_total=?, status='phase4_done' WHERE id=?",
            (scores["score_correct"] / 100, scores["score_over"] / 100, scores["score_total"] / 100, run_id)
        )
        await db.commit()

        # Delta 계산 (이전 완료 Run이 있으면)
        async with db.execute(
            """SELECT id FROM runs WHERE task_id=? AND id != ? AND status IN ('completed','phase4_done','phase5_done','phase6_done')
               ORDER BY run_number DESC LIMIT 1""",
            (run["task_id"], run_id)
        ) as cursor:
            prev_row = await cursor.fetchone()

        if prev_row:
            yield log_event("info", "케이스별 Delta 계산 중...")
            await compute_and_save_deltas(run["task_id"], prev_row["id"], run_id)
            yield log_event("ok", "Delta 계산 완료")

        # BUG-2: 프론트 기대 필드명으로 변환 (correct_plus_over, correct, over, wrong, total)
        frontend_scores = {
            "correct_plus_over": scores["score_total"],
            "correct": scores["score_correct"],
            "over": scores["score_over"],
            "wrong": round(100 - scores["score_correct"] - scores["score_over"], 1),
            "total": scores["total"],
        }
        output = {"scores": frontend_scores}
        await db.execute(
            "UPDATE phase_results SET status='completed', output_data=?, completed_at=? WHERE run_id=? AND phase=4",
            (json.dumps(output), datetime.utcnow().isoformat(), run_id)
        )
        await db.commit()

        yield result_event(output)
        yield done_event("completed")

    except Exception as e:
        yield log_event("error", f"Phase 4 오류: {e}")
        yield done_event("failed")
        db2 = await get_db()
        try:
            await db2.execute("UPDATE phase_results SET status='failed' WHERE run_id=? AND phase=4", (run_id,))
            await db2.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))
            await db2.commit()
        finally:
            await db2.close()
    finally:
        await db.close()
