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


def _format_candidate_prompt(cand: dict) -> str:
    """후보의 프롬프트를 system/user 구분 + 입출력 변수 포함하여 포맷."""
    parts = []
    for label in ["a", "b", "c"]:
        prompt = cand.get(f"node_{label}_prompt")
        if not prompt:
            continue
        sys_p = cand.get(f"node_{label}_system_prompt") or ""
        usr_p = cand.get(f"node_{label}_user_prompt") or ""
        reasoning = "ON" if cand.get(f"node_{label}_reasoning") else "OFF"
        output_var = cand.get(f"node_{label}_output_var") or ""
        input_vars = cand.get(f"node_{label}_input_vars") or "[]"

        header = f"[노드 {label.upper()} (reasoning: {reasoning}"
        if output_var:
            header += f", output: {output_var}"
        if input_vars and input_vars != "[]":
            header += f", input: {input_vars}"
        header += ")]"

        if sys_p:
            header += f"\n  [SYSTEM]\n{sys_p}"
        if usr_p:
            header += f"\n  [USER]\n{usr_p}"
        if not sys_p and not usr_p:
            header += f"\n{prompt}"
        parts.append(header)
    return "\n\n".join(parts) if parts else "(프롬프트 없음)"


def _build_prompt_diff(prev_cand: dict | None, curr_cand: dict | None) -> str:
    """두 후보 간 구조적 차이를 요약."""
    if not prev_cand:
        return "(이전 프롬프트 없음 — 첫 실험)"
    if not curr_cand:
        return "(현재 프롬프트 없음)"

    diff = []
    prev_count = prev_cand.get("node_count", 1)
    curr_count = curr_cand.get("node_count", 1)

    if prev_count != curr_count:
        diff.append(f"* 노드 수: {prev_count}개 → {curr_count}개")
    else:
        diff.append(f"* 노드 수: {curr_count}개 (동일)")

    for label in ["a", "b", "c"]:
        prev_p = prev_cand.get(f"node_{label}_prompt")
        curr_p = curr_cand.get(f"node_{label}_prompt")
        L = label.upper()

        if not prev_p and not curr_p:
            continue
        elif prev_p and not curr_p:
            diff.append(f"* 노드 {L}: 제거됨")
        elif not prev_p and curr_p:
            diff.append(f"* 노드 {L}: 새로 추가됨")
        else:
            changes = []
            prev_r = bool(prev_cand.get(f"node_{label}_reasoning"))
            curr_r = bool(curr_cand.get(f"node_{label}_reasoning"))
            if prev_r != curr_r:
                changes.append(f"reasoning {'ON→OFF' if prev_r else 'OFF→ON'}")
            prev_ov = prev_cand.get(f"node_{label}_output_var") or ""
            curr_ov = curr_cand.get(f"node_{label}_output_var") or ""
            if prev_ov != curr_ov:
                changes.append(f"output_var '{prev_ov}'→'{curr_ov}'")
            if prev_p.strip() != curr_p.strip():
                changes.append("프롬프트 텍스트 수정됨")
            if changes:
                diff.append(f"* 노드 {L}: {', '.join(changes)}")
            else:
                diff.append(f"* 노드 {L}: 변경 없음")

    return "\n".join(diff)


def _format_intermediate(io_raw: str | None) -> str:
    """intermediate_outputs JSON 문자열을 노드별 읽기 좋은 텍스트로 변환."""
    if not io_raw:
        return "(없음)"
    try:
        io = json.loads(io_raw)
    except Exception:
        return io_raw[:300]
    parts = []
    for var_name, info in io.items():
        if isinstance(info, dict) and "node" in info:
            content = (info.get("content") or "")[:200]
            parts.append(f"  [Node {info['node']}] {var_name}: {content}")
        else:
            parts.append(f"  {var_name}: {str(info)[:200]}")
    return "\n".join(parts) if parts else "(없음)"


async def run_phase6(run_id: int) -> AsyncGenerator[str, None]:
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            run = dict(await cursor.fetchone())

        # 요약 Task 조회 (전략 수립에서 반복 참조)
        async with db.execute("SELECT * FROM tasks WHERE id=?", (run["task_id"],)) as cursor:
            task = dict(await cursor.fetchone())
        generation_task = task.get("generation_task", "")

        # 현재 Run에서 사용된 워크플로우 프롬프트 조회
        current_prompt_text = "(프롬프트 정보 없음)"
        curr_cand_dict = None
        selected_cand_id = run.get("selected_candidate_id")
        if selected_cand_id:
            async with db.execute(
                "SELECT * FROM prompt_candidates WHERE id=?", (selected_cand_id,)
            ) as cursor:
                cand_row = await cursor.fetchone()
            if cand_row:
                curr_cand_dict = dict(cand_row)
                current_prompt_text = _format_candidate_prompt(curr_cand_dict)

        # 이전 Run 프롬프트 조회 (실험 연속성을 위한 비교 대상)
        prev_run_id = None
        prev_cand_dict = None
        previous_prompt_text = "(이전 프롬프트 없음 — 첫 실험)"
        # 1순위: base_run_id (continue 모드), 2순위: 가장 최근 완료 Run
        if run.get("base_run_id"):
            prev_run_id = run["base_run_id"]
        else:
            async with db.execute(
                """SELECT id FROM runs WHERE task_id=? AND id != ?
                   AND status IN ('completed','phase4_done','phase5_done','phase6_done')
                   ORDER BY run_number DESC LIMIT 1""",
                (run["task_id"], run_id)
            ) as cursor:
                prev_row = await cursor.fetchone()
            if prev_row:
                prev_run_id = prev_row["id"]

        if prev_run_id:
            async with db.execute(
                "SELECT selected_candidate_id, run_number, score_total FROM runs WHERE id=?",
                (prev_run_id,)
            ) as cursor:
                prev_run_row = await cursor.fetchone()
            if prev_run_row and prev_run_row["selected_candidate_id"]:
                async with db.execute(
                    "SELECT * FROM prompt_candidates WHERE id=?",
                    (prev_run_row["selected_candidate_id"],)
                ) as cursor:
                    prev_cand_row = await cursor.fetchone()
                if prev_cand_row:
                    prev_cand_dict = dict(prev_cand_row)
                    prev_score = round((prev_run_row["score_total"] or 0) * 100, 1)
                    previous_prompt_text = (
                        f"[이전 Run #{prev_run_row['run_number']} — score: {prev_score}%]\n"
                        + _format_candidate_prompt(prev_cand_dict)
                    )
                    yield log_event("info", f"이전 Run #{prev_run_row['run_number']} 프롬프트 로드됨 (score: {prev_score}%)")

        # 이전↔현재 프롬프트 구조적 diff
        prompt_diff = _build_prompt_diff(prev_cand_dict, curr_cand_dict)

        # Phase 1 분석에서 추출된 Reference 요약 기준
        reference_summary_criteria = ""
        async with db.execute(
            "SELECT output_data FROM phase_results WHERE run_id=? AND phase=1 AND status='completed'",
            (run_id,)
        ) as cursor:
            p1_row = await cursor.fetchone()
        if p1_row and p1_row["output_data"]:
            try:
                p1_data = json.loads(p1_row["output_data"])
                reference_summary_criteria = p1_data.get("reference_summary_criteria", "")
            except Exception:
                pass

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

        # 회귀 케이스 상세 (이전/현재 중간출력 비교 포함)
        regression_details = []
        for r in regressed[:10]:
            async with db.execute(
                "SELECT reason, intermediate_outputs FROM case_results WHERE run_id=? AND case_id=?",
                (run_id, r["case_id"])
            ) as cursor:
                curr_row = await cursor.fetchone()

            # 이전 Run의 중간출력 조회
            prev_io_text = "(이전 중간출력 없음)"
            if prev_run_id:
                try:
                    async with db.execute(
                        "SELECT intermediate_outputs FROM case_results WHERE run_id=? AND case_id=?",
                        (prev_run_id, r["case_id"])
                    ) as cursor:
                        prev_io_row = await cursor.fetchone()
                    if prev_io_row and prev_io_row["intermediate_outputs"]:
                        prev_io_text = _format_intermediate(prev_io_row["intermediate_outputs"])
                except Exception:
                    pass

            curr_io_text = _format_intermediate(
                curr_row["intermediate_outputs"] if curr_row else None
            )

            regression_details.append({
                "case_id": r["case_id"],
                "prev_eval": r["prev_evaluation"],
                "curr_eval": r["curr_evaluation"],
                "reason": curr_row["reason"] if curr_row else "",
                "prev_intermediate": prev_io_text,
                "curr_intermediate": curr_io_text,
            })
        regression_analysis = json.dumps(regression_details, ensure_ascii=False)

        # 중간 출력 패턴 분석 (오답/과답 케이스에서 샘플 최대 5건)
        intermediate_output_analysis = "(중간 출력 없음)"
        try:
            async with db.execute(
                """SELECT case_id, intermediate_outputs, evaluation
                   FROM case_results
                   WHERE run_id=? AND evaluation IN ('오답','과답') AND intermediate_outputs IS NOT NULL
                   LIMIT 5""",
                (run_id,)
            ) as cursor:
                io_rows = [dict(row) for row in await cursor.fetchall()]
            if io_rows:
                io_parts = []
                for r in io_rows:
                    try:
                        io_data = json.loads(r['intermediate_outputs'])
                    except Exception:
                        io_data = {}
                    lines = [f"케이스 {r['case_id']} ({r['evaluation']}):"]
                    for var_name, info in io_data.items():
                        if isinstance(info, dict) and "node" in info:
                            content = (info.get("content") or "")[:300]
                            lines.append(f"  [Node {info['node']}] {var_name}: {content}")
                        else:
                            # 하위 호환: 구조화 이전 데이터
                            lines.append(f"  {var_name}: {str(info)[:300]}")
                    io_parts.append("\n".join(lines))
                intermediate_output_analysis = "\n\n".join(io_parts)
                yield log_event("info", f"중간 출력 샘플 {len(io_rows)}건 수집됨")
        except Exception:
            pass  # 마이그레이션 전 DB 호환

        yield log_event("info", f"Delta 분석: {delta_summary}")
        yield log_event("info", "gpt-oss-120B에게 전략 수립 요청 중...")

        prompt_template = load_prompt()
        prompt = prompt_template.format(
            generation_task=generation_task,
            current_prompt=current_prompt_text,
            previous_prompt=previous_prompt_text,
            prompt_diff=prompt_diff,
            reference_summary_criteria=reference_summary_criteria or "(첫 Run이거나 분석 데이터 없음)",
            current_score=round(current_score * 100, 1),
            learning_rate=learning_rate,
            experiment_history=experiment_history,
            delta_analysis=delta_analysis,
            regression_analysis=regression_analysis,
            intermediate_output_analysis=intermediate_output_analysis,
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
            "constraints": result.get("constraints", ""),
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
