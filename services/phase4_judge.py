import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.delta import compute_and_save_deltas, aggregate_scores
from services.sse_helpers import log_event, progress_event, result_event, done_event

SYSTEM_PROMPT_PATH = "prompts/phase4_judge_system.txt"
USER_PROMPT_PATH = "prompts/phase4_judge_user.txt"
JUDGE_CONCURRENCY = 5


def _load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _extract_json(text: str) -> dict:
    """실제 Judge와 동일: re.search(r'\\{[^{}]*\\}') 로 단일 레벨 JSON 추출."""
    import re
    try:
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return {}


def _classify_from_text(raw_text: str) -> tuple:
    """텍스트에서 정답/과답/오답 분류 — 실제 Judge evaluate_answer 폴백 로직 그대로 재현."""
    result = raw_text if isinstance(raw_text, str) else ""
    result_lower = result.lower()

    # 1순위: 결론 상반 감지 (가장 중요!)
    has_opposite = False
    if "불기" in result and "가능" in result:
        if "reference" in result_lower and "generated" in result_lower:
            has_opposite = True
        elif ("가능" in result and "•" in result) or \
             ("opposite" in result_lower) or ("contradict" in result_lower):
            has_opposite = True

    if has_opposite or "opposite conclusion" in result_lower or "결론 상반" in result:
        return "오답", "결론 상반"

    # 2순위: 명시적 평가 키워드 (한국어)
    if "오답" in result and "정답" not in result:
        return "오답", "정보 불일치"
    if "과답" in result and "정답" not in result:
        return "과답", "추가 정보 포함"
    if "누락" in result and "정답" not in result:
        return "오답", "정보 누락"
    if "정답" in result and "오답" not in result and "누락" not in result:
        return "정답", "동일 정보"

    # 3순위: 영어 키워드 - 부정적 키워드 먼저 체크
    if "wrong" in result_lower or "incorrect" in result_lower:
        return "오답", "정보 불일치"
    if "missing" in result_lower or "lacks" in result_lower or "does not mention" in result_lower:
        return "오답", "정보 누락"
    if "extra" in result_lower or "additional" in result_lower or "unnecessary" in result_lower:
        return "과답", "추가 정보 포함"

    # 4순위: 긍정적 키워드 (가장 마지막)
    if ("same" in result_lower or "matches" in result_lower or "identical" in result_lower) and \
            "not" not in result_lower[:50]:
        return "정답", "동일 정보"

    # 폴백: 평가실패 유지 (실제 Judge와 동일)
    return "평가실패", "평가 불기"


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
                    # 실제 Judge와 동일: JSON에서 "rating" 키로 추출
                    evaluation = result.get("rating", "평가실패")
                    reason = result.get("reason", "")

                    # JSON 파싱 실패 시 텍스트에서 추출 (실제 Judge 동일 로직)
                    if evaluation == "평가실패":
                        fb_eval, fb_reason = _classify_from_text(raw)
                        evaluation = fb_eval
                        reason = fb_reason

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

        pending_tasks = [asyncio.create_task(judge_case(c)) for c in cases]
        try:
            processed = 0
            for coro in asyncio.as_completed(pending_tasks):
                eval_result = await coro
                processed += 1
                if not str(eval_result).startswith("error:"):
                    yield log_event("ok", f"판정: {eval_result}")
                else:
                    yield log_event("warn", f"Judge 실패: {eval_result}")
                yield progress_event(processed, total)
        finally:
            for t in pending_tasks:
                if not t.done():
                    t.cancel()

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
        # 1순위: base_run_id (continue 모드), 2순위: 가장 최근 완료 Run
        prev_run_id = None
        if run.get("base_run_id"):
            prev_run_id = run["base_run_id"]
        else:
            async with db.execute(
                """SELECT id FROM runs WHERE task_id=? AND id != ? AND status IN ('completed','phase4_done','phase5_done','phase6_done')
                   ORDER BY run_number DESC LIMIT 1""",
                (run["task_id"], run_id)
            ) as cursor:
                prev_row = await cursor.fetchone()
            if prev_row:
                prev_run_id = prev_row["id"]

        if prev_run_id:
            yield log_event("info", f"케이스별 Delta 계산 중... (비교 대상: Run #{prev_run_id})")
            await compute_and_save_deltas(run["task_id"], prev_run_id, run_id)
            yield log_event("ok", "Delta 계산 완료")

        # BUG-2: 프론트 기대 필드명으로 변환 (correct_plus_over, correct, over, wrong, total)
        frontend_scores = {
            "correct_plus_over": scores["score_total"],
            "correct": scores["score_correct"],
            "over": scores["score_over"],
            "wrong": round(100 - scores["score_correct"] - scores["score_over"], 1),
            "total": scores["total"],
        }
        # 케이스 목록 조회 (프론트 테이블용)
        async with db.execute(
            """SELECT case_id, stt, reference, generated, evaluation, reason, intermediate_outputs
               FROM case_results WHERE run_id=? ORDER BY rowid""",
            (run_id,)
        ) as cursor:
            result_cases = [dict(row) for row in await cursor.fetchall()]

        cases_list = [{
            "id": r["case_id"], "judge": r["evaluation"] or "",
            "reason": r["reason"] or "", "stt": r["stt"] or "",
            "reference": r["reference"] or "", "generated": r["generated"] or "",
            "intermediate_outputs": json.loads(r["intermediate_outputs"]) if r.get("intermediate_outputs") else {},
        } for r in result_cases]

        output = {"scores": frontend_scores, "cases": cases_list}
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
