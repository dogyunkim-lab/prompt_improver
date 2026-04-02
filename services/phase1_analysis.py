import asyncio
import json
import os
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.sse_helpers import log_event, progress_event, result_event, done_event, case_event

PROMPT_PATH = "prompts/phase1_analysis.txt"
CONCURRENT = 5   # 동시 GPT 호출 수


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _detect_eval_field(cases: list) -> str:
    """JSON 케이스에서 오답/정답/과답 값이 들어있는 필드명 자동 감지."""
    eval_values = {"오답", "과답", "정답", "평가실패"}
    for field in ("evaluation", "answer_evaluation", "judge_result", "judge", "result"):
        if any(str(c.get(field, "")) in eval_values for c in cases[:20]):
            return field
    # fallback: 첫 케이스의 모든 필드 스캔
    if cases:
        for key, val in cases[0].items():
            if str(val) in eval_values:
                return key
    return "evaluation"


def _detect_reason_field(cases: list) -> str:
    """Judge reason/explanation 필드명 자동 감지."""
    for field in ("reason", "answer_evaluation_reason", "judge_reason", "evaluation_reason", "rationale", "explanation", "comment"):
        if any(isinstance(c.get(field), str) and c.get(field, "").strip() for c in cases[:10]):
            return field
    return "reason"


async def _call_gpt_case(case: dict, prompt_template: str, eval_field: str, reason_field: str,
                         current_prompt: str = "", generation_task: str = "",
                         intermediate_outputs: str = "") -> dict:
    """단일 케이스 GPT 분석. DB 조작 없음 — asyncio.gather에서 병렬 실행 가능."""
    case_prompt = prompt_template.format(
        stt=case.get("stt", ""),
        reference=case.get("reference", ""),
        keywords=case.get("keywords", ""),
        generated=case.get("generated", ""),
        judge_evaluation=case.get(eval_field, ""),
        judge_reason=case.get(reason_field, ""),
        case_id=case.get("id", ""),
        current_prompt=current_prompt,
        generation_task=generation_task,
        intermediate_outputs=intermediate_outputs or "(중간 출력 없음)",
    )
    raw = await call_gpt([{"role": "user", "content": case_prompt}], reasoning="high")
    result = _extract_json(raw)
    result["case_id"] = str(case.get("id", ""))  # case_id 항상 보장
    return result


async def run_phase1(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = dict(await cursor.fetchone())

        # 요약 Task 조회 (모든 분석에서 반복 참조)
        async with db.execute("SELECT * FROM tasks WHERE id=?", (run["task_id"],)) as cursor:
            task = dict(await cursor.fetchone())
        generation_task = task.get("generation_task", "")

        # phase_result 초기화 (기존 output_data 보존)
        await db.execute(
            """INSERT INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,1,'running',?)
               ON CONFLICT(run_id, phase) DO UPDATE SET status='running', started_at=excluded.started_at""",
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

        # 현재 요약 프롬프트 읽기 (선택사항)
        current_prompt = ""
        prompt_file = run.get("prompt_file_path")
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                current_prompt = f.read().strip()
            yield log_event("info", f"현재 요약 프롬프트 로드 완료 ({len(current_prompt)}자)")
        else:
            yield log_event("warn", "현재 요약 프롬프트가 제공되지 않았습니다. 프롬프트 없이 분석합니다.")

        yield log_event("info", "Judge JSON 파일 파싱 중...")
        with open(judge_file, "r", encoding="utf-8") as f:
            judge_data = json.load(f)

        # JSON 구조 정규화: 최상위가 dict이면 내부 list를 자동 추출
        if isinstance(judge_data, dict):
            list_values = [(k, v) for k, v in judge_data.items() if isinstance(v, list)]
            if list_values:
                key, judge_data = max(list_values, key=lambda x: len(x[1]))
                yield log_event("info", f"JSON 키 '{key}'에서 {len(judge_data)}개 케이스 추출")
            else:
                yield log_event("error", f"JSON 형식 오류: dict에 list가 없습니다. 키 목록: {list(judge_data.keys())}")
                yield done_event("failed")
                await _mark_phase_failed(run_id, 1)
                return

        if not isinstance(judge_data, list):
            yield log_event("error", f"JSON 형식 오류: list 또는 dict가 필요합니다. 실제 타입: {type(judge_data).__name__}")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 1)
            return

        judge_data = [c for c in judge_data if isinstance(c, dict)]
        if not judge_data:
            yield log_event("error", "JSON 파일에서 유효한 케이스(dict 형식)를 찾을 수 없습니다.")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 1)
            return

        yield log_event("info", f"감지된 필드: {list(judge_data[0].keys())}")

        # 판정/사유 필드명 자동 감지
        eval_field = _detect_eval_field(judge_data)
        reason_field = _detect_reason_field(judge_data)
        yield log_event("info", f"판정 필드: '{eval_field}' | 사유 필드: '{reason_field}'")

        # 오답/과답 케이스 추출
        error_cases = [c for c in judge_data if c.get(eval_field) in ("오답", "과답")]
        total_cases = len(judge_data)
        error_count = len(error_cases)

        yield log_event("info", f"전체 {total_cases}건 중 오답/과답 {error_count}건 분석 시작")

        # 재실행 시 기존 케이스 데이터 삭제 (이전 파일 데이터가 남지 않도록)
        await db.execute("DELETE FROM case_results WHERE run_id=?", (run_id,))
        await db.commit()

        # 전체 케이스 case_results에 저장
        for case in judge_data:
            await db.execute(
                """INSERT OR IGNORE INTO case_results
                   (run_id, case_id, generation_task, stt, reference, keywords, generated, evaluation, reason)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, str(case.get("id", "")), case.get("generation_task", ""),
                 case.get("stt", ""), case.get("reference", ""), case.get("keywords", ""),
                 case.get("generated", ""), case.get(eval_field, ""), case.get(reason_field, ""))
            )
        await db.commit()

        await db.execute("UPDATE runs SET total_cases=? WHERE id=?", (total_cases, run_id))
        await db.commit()

        # 정답 케이스 즉시 스트리밍 (GPT 분석 불필요)
        for case in judge_data:
            if case.get(eval_field) not in ("오답", "과답"):
                yield case_event({
                    "id": str(case.get("id", "")),
                    "judge": case.get(eval_field, ""),
                    "bucket": "",
                    "analysis_summary": "",
                    "stt_uncertain": "",
                    "stt": case.get("stt", ""),
                    "reference": case.get("reference", ""),
                    "generated": case.get("generated", ""),
                    "judge_disagreement": case.get(reason_field, ""),
                    "hallucination": False,
                })

        if error_count == 0:
            yield log_event("ok", "오답/과답 케이스가 없습니다. Phase 1 완료.")
            correct_count = sum(1 for c in judge_data if c.get(eval_field) == "정답")
            over_count    = sum(1 for c in judge_data if c.get(eval_field) == "과답")
            wrong_count   = sum(1 for c in judge_data if c.get(eval_field) == "오답")
            zero_summary = {
                "bucket_counts": {}, "top_issues": [], "prompt_improvable_cases": [],
                "judge_dispute_cases": [], "recommended_focus": "오류 케이스 없음",
                "scores": {
                    "correct_plus_over": round((correct_count + over_count) / total_cases * 100, 1) if total_cases else 0,
                    "correct": round(correct_count / total_cases * 100, 1) if total_cases else 0,
                    "over":    round(over_count    / total_cases * 100, 1) if total_cases else 0,
                    "wrong":   round(wrong_count   / total_cases * 100, 1) if total_cases else 0,
                    "total": total_cases,
                    "correct_count": correct_count,
                    "over_count": over_count,
                    "wrong_count": wrong_count,
                },
                "eval_chart": {
                    "labels": ["정답", "과답", "오답"],
                    "values": [correct_count, over_count, wrong_count],
                },
                "bucket_chart": {
                    "labels": ["STT 오류", "프롬프트 누락", "모델 동작", "Judge 이견"],
                    "values": [0, 0, 0, 0],
                },
            }
            await _mark_phase_completed(run_id, 1, zero_summary)
            yield result_event(zero_summary)
            yield done_event("completed")
            return

        prompt_template = load_prompt()
        case_analyses = []
        sem = asyncio.Semaphore(CONCURRENT)

        # Continue 모드: 이전 Run의 intermediate_outputs 조회
        prev_intermediate_map = {}
        if run.get("base_run_id"):
            try:
                async with db.execute(
                    "SELECT case_id, intermediate_outputs FROM case_results WHERE run_id=? AND intermediate_outputs IS NOT NULL",
                    (run["base_run_id"],)
                ) as cursor:
                    for row in await cursor.fetchall():
                        prev_intermediate_map[row["case_id"]] = row["intermediate_outputs"]
                if prev_intermediate_map:
                    yield log_event("info", f"이전 Run의 중간 출력 {len(prev_intermediate_map)}건 로드됨")
            except Exception:
                pass  # 마이그레이션 전 DB 호환

        async def _analyze_with_sem(c):
            async with sem:
                # 이전 Run의 중간 출력 첨부
                case_intermediate = prev_intermediate_map.get(str(c.get("id", "")), "")
                return await _call_gpt_case(c, prompt_template, eval_field, reason_field,
                                            current_prompt=current_prompt,
                                            generation_task=generation_task,
                                            intermediate_outputs=case_intermediate)

        # ── 배치 병렬 분석 ────────────────────────────────────────────────────
        done_count = 0
        for i in range(0, error_count, CONCURRENT):
            batch = error_cases[i:i + CONCURRENT]
            yield log_event("info", f"케이스 {i+1}~{min(i+len(batch), error_count)} 병렬 분석 중...")

            outcomes = await asyncio.gather(
                *[_analyze_with_sem(c) for c in batch],
                return_exceptions=True
            )

            for case, outcome in zip(batch, outcomes):
                done_count += 1
                if isinstance(outcome, Exception):
                    yield log_event("warn", f"케이스 {case.get('id', '?')} 분석 실패: {outcome}")
                    result = {"case_id": str(case.get("id", "")), "bucket": "prompt_missing",
                              "analysis_summary": "", "stt_uncertain_expressions": [],
                              "hallucination_detected": False, "judge_agreement": True,
                              "judge_dispute_reason": ""}
                else:
                    result = outcome

                # 필드 정규화
                stt_uncertain_list = result.get("stt_uncertain_expressions", [])
                stt_uncertain_str = (", ".join(stt_uncertain_list)
                                     if isinstance(stt_uncertain_list, list)
                                     else str(stt_uncertain_list))
                judge_disagree = ""
                if not result.get("judge_agreement", True):
                    judge_disagree = result.get("judge_dispute_reason", "")

                # 새 분석 필드
                missing_instruction = result.get("missing_instruction", "")
                violated_instruction = result.get("violated_instruction", "")
                error_pattern = result.get("error_pattern", "")
                improvement_suggestion = result.get("improvement_suggestion", "")

                # Reference 요약 기준 분석 필드
                reference_criteria = result.get("reference_criteria", "")
                content_gap = result.get("content_gap", "")

                # ── DB 저장 (확장 필드 포함) ──
                await db.execute(
                    """UPDATE case_results
                       SET bucket=?, analysis_summary=?, stt_uncertain=?,
                           hallucination_detected=?, judge_agreement=?, judge_disagreement=?,
                           missing_instruction=?, violated_instruction=?,
                           error_pattern=?, improvement_suggestion=?,
                           reference_criteria=?, content_gap=?
                       WHERE run_id=? AND case_id=?""",
                    (result.get("bucket", ""),
                     result.get("analysis_summary", ""),
                     stt_uncertain_str,
                     1 if result.get("hallucination_detected") else 0,
                     1 if result.get("judge_agreement", True) else 0,
                     judge_disagree,
                     missing_instruction,
                     violated_instruction,
                     error_pattern,
                     improvement_suggestion,
                     reference_criteria,
                     content_gap,
                     run_id, str(case.get("id", "")))
                )

                # 원문 텍스트도 result에 첨부 (summarize_all에서 활용)
                result["_stt"] = case.get("stt", "")
                result["_reference"] = case.get("reference", "")
                result["_generated"] = case.get("generated", "")
                result["_judge_evaluation"] = case.get(eval_field, "")
                case_analyses.append(result)

                bucket_label = result.get("bucket", "unknown")
                secondary = result.get("secondary_bucket")
                if secondary:
                    bucket_label += f"+{secondary}"
                yield log_event("ok", f"케이스 {case.get('id', '?')} → {bucket_label}")
                yield case_event({
                    "id": str(case.get("id", "")),
                    "judge": case.get(eval_field, ""),
                    "bucket": result.get("bucket", ""),
                    "secondary_bucket": result.get("secondary_bucket", ""),
                    "analysis_summary": result.get("analysis_summary", ""),
                    "stt_uncertain": stt_uncertain_str,
                    "stt": case.get("stt", ""),
                    "reference": case.get("reference", ""),
                    "generated": case.get("generated", ""),
                    "judge_disagreement": judge_disagree or case.get(reason_field, ""),
                    "hallucination": bool(result.get("hallucination_detected", False)),
                    "hallucination_detail": result.get("hallucination_detail", ""),
                    "missing_instruction": missing_instruction,
                    "violated_instruction": violated_instruction,
                    "error_pattern": error_pattern,
                    "improvement_suggestion": improvement_suggestion,
                })

            await db.commit()
            yield progress_event(done_count, error_count)

        # ── 전체 패턴 요약 ────────────────────────────────────────────────────
        yield log_event("info", "전체 패턴 요약 생성 중...")
        summary = await _summarize_all(case_analyses, error_count)
        yield log_event("ok", f"분석 완료 — 주요 이슈: {', '.join(summary.get('top_issues', []))}")

        # baseline 점수 추가
        correct_count = sum(1 for c in judge_data if c.get(eval_field) == "정답")
        over_count    = sum(1 for c in judge_data if c.get(eval_field) == "과답")
        wrong_count   = sum(1 for c in judge_data if c.get(eval_field) == "오답")
        summary["scores"] = {
            "correct_plus_over": round((correct_count + over_count) / total_cases * 100, 1) if total_cases else 0,
            "correct": round(correct_count / total_cases * 100, 1) if total_cases else 0,
            "over":    round(over_count    / total_cases * 100, 1) if total_cases else 0,
            "wrong":   round(wrong_count   / total_cases * 100, 1) if total_cases else 0,
            "total": total_cases,
            "correct_count": correct_count,
            "over_count": over_count,
            "wrong_count": wrong_count,
        }

        # 판정 분포 차트 (정답/과답/오답 raw count)
        summary["eval_chart"] = {
            "labels": ["정답", "과답", "오답"],
            "values": [correct_count, over_count, wrong_count],
        }

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
    """GPT 응답에서 JSON 블록 추출 (```json ... ``` 코드펜스 포함 처리)."""
    if not text:
        return {}
    # ```json ... ``` 코드펜스 제거
    import re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # 일반 JSON 추출
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    return {}


async def _summarize_all(case_analyses: list, n: int) -> dict:
    """
    버킷별 집계는 직접 계산 (GPT 의존 없음).
    top_issues / recommended_focus만 GPT에게 요청.
    prompt_improvable_cases에 원문 텍스트 + 구체적 분석 필드를 담아 Phase 2에서 활용.
    """
    # ── 직접 집계 ──────────────────────────────────────────────────────────────
    bucket_counts = {"stt_error": 0, "prompt_missing": 0, "model_behavior": 0, "judge_dispute": 0}
    prompt_improvable = []
    judge_dispute_list = []
    error_pattern_groups = {}  # 오류 패턴별 그룹핑

    reference_criteria_samples = []
    content_gap_samples = []

    for a in case_analyses:
        b = a.get("bucket", "")
        if b in bucket_counts:
            bucket_counts[b] += 1
        # secondary_bucket도 집계
        sb = a.get("secondary_bucket")
        if sb and sb in bucket_counts:
            bucket_counts[sb] += 0.5  # 보조 원인은 0.5로 카운트

        # 오류 패턴 그룹핑
        ep = a.get("error_pattern", "")
        if ep:
            error_pattern_groups.setdefault(ep, []).append(a.get("case_id", ""))

        # Reference 요약 기준 분석 수집
        rc = a.get("reference_criteria", "")
        if rc:
            reference_criteria_samples.append(rc)
        cg = a.get("content_gap", "")
        if cg:
            content_gap_samples.append(cg)

        # 프롬프트로 개선 가능한 케이스 (prompt_missing / model_behavior)
        if b in ("prompt_missing", "model_behavior"):
            prompt_improvable.append({
                "case_id": a.get("case_id", ""),
                "bucket": b,
                "secondary_bucket": a.get("secondary_bucket"),
                "analysis_summary": a.get("analysis_summary", ""),
                "hallucination": bool(a.get("hallucination_detected", False)),
                "hallucination_detail": a.get("hallucination_detail", ""),
                "missing_instruction": a.get("missing_instruction", ""),
                "violated_instruction": a.get("violated_instruction", ""),
                "error_pattern": ep,
                "improvement_suggestion": a.get("improvement_suggestion", ""),
                "reference_criteria": a.get("reference_criteria", ""),
                "content_gap": a.get("content_gap", ""),
                # Phase 2에서 구체적 예시로 활용할 원문 텍스트
                "stt": a.get("_stt", ""),
                "reference": a.get("_reference", ""),
                "generated": a.get("_generated", ""),
            })
        # Judge 이견 케이스
        if b == "judge_dispute" or not a.get("judge_agreement", True):
            judge_dispute_list.append({
                "case_id": a.get("case_id", ""),
                "analysis_summary": a.get("analysis_summary", ""),
                "dispute_reason": a.get("judge_dispute_reason", ""),
            })

    # 오류 패턴 빈도순 정렬
    error_pattern_ranking = sorted(
        [{"pattern": k, "count": len(v), "case_ids": v} for k, v in error_pattern_groups.items()],
        key=lambda x: x["count"], reverse=True
    )

    # bucket_counts 정수화 (secondary로 인한 0.5 → 반올림)
    bucket_counts = {k: round(v) for k, v in bucket_counts.items()}

    # ── GPT: 정성적 요약만 요청 ────────────────────────────────────────────────
    # 내부 참조용 원문 필드는 제거하고 GPT에 전달
    analyses_for_gpt = []
    for a in case_analyses:
        filtered = {k: v for k, v in a.items() if not k.startswith("_")}
        analyses_for_gpt.append(filtered)

    analyses_text = json.dumps(analyses_for_gpt, ensure_ascii=False, indent=2)

    # Reference 요약 기준 샘플 텍스트 (GPT에 전달용, 최대 15개)
    ref_criteria_text = "\n".join(f"- {s}" for s in reference_criteria_samples[:15]) if reference_criteria_samples else "(없음)"
    content_gap_text = "\n".join(f"- {s}" for s in content_gap_samples[:15]) if content_gap_samples else "(없음)"

    prompt = f"""아래는 오답/과답 케이스 {n}개의 분석 데이터다.
이를 바탕으로 다음 JSON만 출력하라 (설명 없이 JSON만):
{{
  "top_issues": ["가장 자주 발생하는 오류 유형 설명 1", "설명 2", "설명 3"],
  "recommended_focus": "Phase 2 프롬프트 개선 시 가장 중요하게 다뤄야 할 방향. 구체적인 프롬프트 수정 방향을 포함하여 4~6문장으로 작성하라.",
  "reference_summary_criteria": "아래 [케이스별 Reference 요약 기준]을 종합하여, 상담사들이 공통으로 적용하는 요약 기준을 5~8문장으로 정리하라. 요약 Task의 목적에 따라 어떤 정보를 핵심으로 포함하고 어떤 정보를 생략하는지, LLM이 동일한 핵심 내용을 요약하려면 프롬프트에 어떤 지시가 필요한지를 포함하라. 문체나 종결어미 같은 표면적 형식이 아니라 내용 선별 기준에 집중하라.",
  "common_content_gaps": "아래 [요약 내용 차이 목록]을 종합하여, Generated가 Reference 대비 가장 자주 빠뜨리거나 불필요하게 추가하는 내용 패턴 3~5가지를 나열하라."
}}

[케이스별 Reference 요약 기준]
{ref_criteria_text}

[요약 내용 차이 목록]
{content_gap_text}

[분석 데이터]
{analyses_text}"""

    top_issues = []
    recommended_focus = ""
    reference_summary_criteria = ""
    common_content_gaps = ""
    try:
        raw = await call_gpt([{"role": "user", "content": prompt}], reasoning="high")
        gpt = _extract_json(raw)
        top_issues = gpt.get("top_issues", [])
        recommended_focus = gpt.get("recommended_focus", "")
        reference_summary_criteria = gpt.get("reference_summary_criteria", "")
        common_content_gaps = gpt.get("common_content_gaps", "")
    except Exception:
        # fallback: analysis_summary에서 대표 텍스트 추출
        top_issues = list({
            a.get("analysis_summary", "")[:60]
            for a in case_analyses[:5]
            if a.get("analysis_summary")
        })
        recommended_focus = "수동 확인 필요"

    return {
        "bucket_counts": bucket_counts,
        "top_issues": top_issues,
        "error_pattern_ranking": error_pattern_ranking,
        "prompt_improvable_cases": prompt_improvable,
        "judge_dispute_cases": judge_dispute_list,
        "recommended_focus": recommended_focus,
        "reference_summary_criteria": reference_summary_criteria,
        "common_content_gaps": common_content_gaps,
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
