import json
import os
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.sse_helpers import log_event, progress_event, result_event, done_event

PROMPT_PATH = "prompts/phase1_analysis.txt"
BATCH_SIZE = 5


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


async def run_phase1(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = dict(await cursor.fetchone())

        # phase_result 초기화
        await db.execute(
            """INSERT OR REPLACE INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,1,'running',?)""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()

        await db.execute("UPDATE runs SET status='phase1_running', current_phase=1 WHERE id=?", (run_id,))
        await db.commit()

        judge_file = run.get("judge_file_path")
        if not judge_file or not os.path.exists(judge_file):
            yield log_event("error", "Judge JSON 파일이 없습니다. 먼저 업로드하세요.")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 1)
            return

        yield log_event("info", "Judge JSON 파일 파싱 중...")
        with open(judge_file, "r", encoding="utf-8") as f:
            judge_data = json.load(f)

        # 오답/과답 케이스 추출
        error_cases = [c for c in judge_data if c.get("evaluation") in ("오답", "과답")]
        total_cases = len(judge_data)
        error_count = len(error_cases)

        yield log_event("info", f"전체 {total_cases}건 중 오답/과답 {error_count}건 분석 시작")

        # 전체 케이스 case_results에 저장 (generated 없으므로 judge 결과만)
        for case in judge_data:
            await db.execute(
                """INSERT OR IGNORE INTO case_results
                   (run_id, case_id, generation_task, stt, reference, keywords, generated, evaluation, reason)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, str(case.get("id", "")), case.get("generation_task", ""),
                 case.get("stt", ""), case.get("reference", ""), case.get("keywords", ""),
                 case.get("generated", ""), case.get("evaluation", ""), case.get("reason", ""))
            )
        await db.commit()

        await db.execute("UPDATE runs SET total_cases=? WHERE id=?", (total_cases, run_id))
        await db.commit()

        if error_count == 0:
            yield log_event("ok", "오답/과답 케이스가 없습니다. Phase 1 완료.")
            yield done_event("completed")
            await _mark_phase_completed(run_id, 1, {"bucket_counts": {}, "top_issues": [], "prompt_improvable_cases": [], "judge_dispute_cases": [], "recommended_focus": "오류 케이스 없음"})
            return

        prompt_template = load_prompt()
        case_analyses = []

        # 배치 분석
        for i in range(0, error_count, BATCH_SIZE):
            batch = error_cases[i:i + BATCH_SIZE]
            batch_results = []

            for case in batch:
                case_prompt = prompt_template.format(
                    stt=case.get("stt", ""),
                    reference=case.get("reference", ""),
                    keywords=case.get("keywords", ""),
                    generated=case.get("generated", ""),
                    judge_evaluation=case.get("evaluation", ""),
                    judge_reason=case.get("reason", ""),
                    case_id=case.get("id", "")
                )
                yield log_event("info", f"케이스 {case.get('id', '?')} 분석 중...")
                try:
                    messages = [{"role": "user", "content": case_prompt}]
                    raw = await call_gpt(messages, reasoning="high")
                    # JSON 추출
                    result = _extract_json(raw)
                    batch_results.append(result)
                    # bucket 저장
                    await db.execute(
                        "UPDATE case_results SET bucket=? WHERE run_id=? AND case_id=?",
                        (result.get("bucket", ""), run_id, str(case.get("id", "")))
                    )
                    await db.commit()
                    yield log_event("ok", f"케이스 {case.get('id', '?')} → {result.get('bucket', 'unknown')}")
                except Exception as e:
                    yield log_event("warn", f"케이스 {case.get('id', '?')} 분석 실패: {e}")
                    batch_results.append({"case_id": str(case.get("id", "")), "bucket": "prompt_missing"})

            # BUG-9 fix: extend는 배치 루프 종료 후 한 번만 호출
            case_analyses.extend(batch_results)
            yield progress_event(min(i + len(batch), error_count), error_count)

        # 전체 패턴 요약
        yield log_event("info", "전체 패턴 요약 생성 중...")
        summary = await _summarize_all(case_analyses, error_count)
        yield log_event("ok", f"분석 완료 — 주요 이슈: {', '.join(summary.get('top_issues', []))}")

        # BUG-10: baseline scores, cases, bucket_chart를 output_data에 포함
        correct_count = sum(1 for c in judge_data if c.get("evaluation") == "정답")
        over_count = sum(1 for c in judge_data if c.get("evaluation") == "과답")
        wrong_count = sum(1 for c in judge_data if c.get("evaluation") == "오답")
        summary["scores"] = {
            "correct_plus_over": round((correct_count + over_count) / total_cases * 100, 1) if total_cases else 0,
            "correct": round(correct_count / total_cases * 100, 1) if total_cases else 0,
            "over": round(over_count / total_cases * 100, 1) if total_cases else 0,
            "wrong": round(wrong_count / total_cases * 100, 1) if total_cases else 0,
            "total": total_cases,
        }

        # bucket_id → bucket 레이블 조회용 맵
        bucket_map = {str(a.get("case_id", "")): a.get("bucket", "") for a in case_analyses}
        summary["cases"] = [
            {
                "id": str(c.get("id", "")),
                "judge": c.get("evaluation", ""),
                "bucket": bucket_map.get(str(c.get("id", "")), ""),
                "stt_uncertain": "",
                "summary": c.get("generated", ""),
                "judge_disagreement": c.get("reason", ""),
            }
            for c in judge_data[:100]
        ]

        bucket_counts = summary.get("bucket_counts", {})
        summary["bucket_chart"] = {
            "labels": ["STT 오류", "프롬프트 누락", "모델 동작", "Judge 이견"],
            "values": [
                bucket_counts.get("stt_error", 0),
                bucket_counts.get("prompt_missing", 0),
                bucket_counts.get("model_behavior", 0),
                bucket_counts.get("judge_dispute", 0),
            ],
        }

        await _mark_phase_completed(run_id, 1, summary)
        await db.execute("UPDATE runs SET status='phase1_done' WHERE id=?", (run_id,))
        await db.commit()

        yield result_event(summary)
        yield done_event("completed")

    except Exception as e:
        yield log_event("error", f"Phase 1 오류: {e}")
        yield done_event("failed")
        await _mark_phase_failed(run_id, 1)
    finally:
        await db.close()


def _extract_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


async def _summarize_all(case_analyses: list, n: int) -> dict:
    analyses_text = json.dumps(case_analyses, ensure_ascii=False, indent=2)
    prompt = f"""위 {n}개 케이스 분석 결과를 종합하여 아래 항목을 JSON으로 출력하라:
{{
  "bucket_counts": {{
    "stt_error": n,
    "prompt_missing": n,
    "model_behavior": n,
    "judge_dispute": n
  }},
  "top_issues": ["주요 문제 1", "주요 문제 2", "주요 문제 3"],
  "prompt_improvable_cases": [],
  "judge_dispute_cases": [],
  "recommended_focus": "Phase 2에서 집중해야 할 개선 방향 서술"
}}

분석 데이터:
{analyses_text}"""
    try:
        messages = [{"role": "user", "content": prompt}]
        raw = await call_gpt(messages, reasoning="high")
        return _extract_json(raw)
    except Exception:
        # fallback: 수동 집계
        bucket_counts = {"stt_error": 0, "prompt_missing": 0, "model_behavior": 0, "judge_dispute": 0}
        for a in case_analyses:
            b = a.get("bucket", "")
            if b in bucket_counts:
                bucket_counts[b] += 1
        return {
            "bucket_counts": bucket_counts,
            "top_issues": ["분석 요약 생성 실패"],
            "prompt_improvable_cases": [],
            "judge_dispute_cases": [],
            "recommended_focus": "수동 확인 필요"
        }


async def _mark_phase_completed(run_id: int, phase: int, output_data: dict):
    db = await get_db()
    try:
        await db.execute(
            """UPDATE phase_results SET status='completed', output_data=?, completed_at=?
               WHERE run_id=? AND phase=?""",
            (json.dumps(output_data, ensure_ascii=False), datetime.utcnow().isoformat(), run_id, phase)
        )
        await db.commit()
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
