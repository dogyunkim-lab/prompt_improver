import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator
from database import get_db
from services.gpt_client import call_gpt
from services.delta import compute_learning_rate, count_completed_runs, get_run_scores
from services.sse_helpers import log_event, result_event, done_event

logger = logging.getLogger(__name__)

STRATEGY_PROMPT_PATH = "prompts/phase2_strategy.txt"
CANDIDATE_PROMPT_PATH = "prompts/phase2_candidate.txt"
REPAIR_PROMPT_PATH = "prompts/phase2_repair.txt"

NODE_LABELS = ["A", "B", "C"]


# ─── 프롬프트 로더 ──────────────────────────────────────────────────────────

def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ─── 공유 유틸 (기존 유지) ──────────────────────────────────────────────────

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
    if not text:
        return {}
    import re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


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


# ─── Step 0: converge 모드 이전 최적 프롬프트 조회 ──────────────────────────

async def _get_best_previous_candidates(task_id: int, current_run_id: int) -> str:
    """이전 최고 score_total run의 prompt_candidates를 텍스트로 반환."""
    db = await get_db()
    try:
        async with db.execute(
            """SELECT id, run_number, score_total FROM runs
               WHERE task_id=? AND id != ? AND score_total IS NOT NULL
               AND status IN ('completed','phase4_done','phase5_done','phase6_done')
               ORDER BY score_total DESC LIMIT 1""",
            (task_id, current_run_id)
        ) as cursor:
            best_run = await cursor.fetchone()

        if not best_run:
            return ""

        best_run = dict(best_run)
        async with db.execute(
            "SELECT * FROM prompt_candidates WHERE run_id=? ORDER BY candidate_label",
            (best_run["id"],)
        ) as cursor:
            candidates = [dict(row) for row in await cursor.fetchall()]

        if not candidates:
            return ""

        lines = [f"[이전 최고 성적 Run #{best_run['run_number']} — score_total={best_run['score_total']}%]"]
        for c in candidates:
            lines.append(f"\n후보 {c['candidate_label']} (node_count={c['node_count']}):")
            for label in NODE_LABELS:
                key = f"node_{label.lower()}_prompt"
                if c.get(key):
                    lines.append(f"  노드 {label}: {c[key][:200]}...")
        return "\n".join(lines)
    finally:
        await db.close()


# ─── Step 0-b: 이전 Run Phase 6 피드백 조회 ──────────────────────────────────

async def _get_prev_run_feedback(base_run_id: int) -> str:
    """이전 Run의 Phase 6 output_data에서 next_direction, effective, harmful 텍스트를 조합."""
    db = await get_db()
    try:
        async with db.execute(
            "SELECT output_data FROM phase_results WHERE run_id=? AND phase=6 AND status='completed'",
            (base_run_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row or not row["output_data"]:
            return ""

        p6 = json.loads(row["output_data"])
        parts = []

        next_dir = p6.get("next_direction", "")
        if next_dir:
            parts.append(f"[이전 Run Phase 6 — 다음 방향]\n{next_dir}")

        effective = p6.get("effective", [])
        if effective:
            parts.append(f"[효과적 요소]\n" + "\n".join(f"- {e}" for e in effective))

        harmful = p6.get("harmful", [])
        if harmful:
            parts.append(f"[해로운 요소]\n" + "\n".join(f"- {h}" for h in harmful))

        return "\n\n".join(parts)
    finally:
        await db.close()


# ─── Step 1: 전략 수립 ──────────────────────────────────────────────────────

def _build_strategy_input(
    task: dict, phase1_summary: dict, experiment_history: str,
    learning_rate: str, converge_text: str,
    improvable_count: int, total_count: int,
    prev_run_feedback: str = "",
    user_guide: str = ""
) -> str:
    template = _load_prompt(STRATEGY_PROMPT_PATH)

    scores = phase1_summary.get("scores", {})
    bucket_counts = phase1_summary.get("bucket_counts", {})
    error_pattern_ranking = phase1_summary.get("error_pattern_ranking", [])
    top_issues = phase1_summary.get("top_issues", [])
    recommended_focus = phase1_summary.get("recommended_focus", "")

    converge_context = ""
    if converge_text:
        converge_context = f"[converge 모드 — 이전 최고 성적 참조]\n{converge_text}"

    user_guide_section = ""
    if user_guide:
        user_guide_section = f"[사용자 전략 가이드 — 반드시 준수]\n{user_guide}"

    # Reference 요약 기준 (Phase 1에서 분석된 상담사의 핵심 내용 선별 기준)
    ref_criteria = phase1_summary.get("reference_summary_criteria", "")
    common_gaps = phase1_summary.get("common_content_gaps", "")
    reference_criteria_section = ""
    if ref_criteria:
        reference_criteria_section = f"[Reference 요약 기준 — 상담사가 어떤 내용을 포함/생략하는지]\n{ref_criteria}"
        if common_gaps:
            reference_criteria_section += f"\n\n[Generated가 자주 빠뜨리거나 불필요하게 추가하는 내용 패턴]\n{common_gaps}"

    return template.format(
        generation_task=task.get("generation_task", "불편사항 요약"),
        bucket_counts=json.dumps(bucket_counts, ensure_ascii=False),
        error_pattern_ranking=json.dumps(error_pattern_ranking[:10], ensure_ascii=False),
        top_issues=json.dumps(top_issues, ensure_ascii=False),
        recommended_focus=recommended_focus,
        scores=json.dumps(scores, ensure_ascii=False),
        improvable_count=improvable_count,
        total=total_count,
        experiment_history=experiment_history,
        learning_rate=learning_rate,
        converge_context=converge_context,
        prev_run_feedback=prev_run_feedback,
        user_guide=user_guide_section,
        reference_criteria=reference_criteria_section,
    )


async def _call_strategy_step(prompt: str, max_retries: int = 1) -> dict | None:
    """Step 1: GPT에 전략 수립 요청. 실패 시 max_retries만큼 재시도."""
    for attempt in range(max_retries + 1):
        try:
            raw = await call_gpt([{"role": "user", "content": prompt}], reasoning="high")
            result = _extract_json(raw)
            candidates = result.get("candidates", [])
            if not candidates:
                logger.warning(f"[Step1] attempt {attempt+1}: candidates 비어있음")
                continue
            # 기본 검증: 각 후보에 필수 필드 있는지
            valid = True
            for c in candidates:
                if not c.get("label") or not c.get("node_count") or not c.get("node_roles"):
                    valid = False
                    break
                if len(c.get("node_roles", [])) != c["node_count"]:
                    valid = False
                    break
            if valid:
                return result
            logger.warning(f"[Step1] attempt {attempt+1}: 구조 검증 실패")
        except Exception as e:
            logger.warning(f"[Step1] attempt {attempt+1} 오류: {e}")
    return None


# ─── Step 2: 후보별 프롬프트 생성 ────────────────────────────────────────────

def _select_cases_for_candidate(
    focus_patterns: list, improvable_cases: list, max_cases: int = 5
) -> list:
    """focus_patterns에 매칭되는 케이스 우선 선별, 부족하면 빈도순 보충."""
    selected = []
    selected_ids = set()

    # focus_patterns 매칭 우선
    for case in improvable_cases:
        if len(selected) >= max_cases:
            break
        case_pattern = case.get("error_pattern", "")
        if case_pattern and any(fp in case_pattern for fp in focus_patterns):
            if case["case_id"] not in selected_ids:
                selected.append(case)
                selected_ids.add(case["case_id"])

    # 부족하면 빈도순(리스트 순서) 보충
    for case in improvable_cases:
        if len(selected) >= max_cases:
            break
        if case["case_id"] not in selected_ids:
            selected.append(case)
            selected_ids.add(case["case_id"])

    return selected


def _format_cases_text(cases: list) -> str:
    """대표 케이스를 프롬프트에 삽입 가능한 텍스트로 변환."""
    lines = []
    for i, c in enumerate(cases, 1):
        stt = (c.get("stt") or "")[:500]
        ref = (c.get("reference") or "")[:300]
        gen = (c.get("generated") or "")[:300]
        lines.append(f"--- 케이스 {i} (ID: {c['case_id']}) ---")
        lines.append(f"Reference 요약 기준: {c.get('reference_criteria', 'N/A')}")
        lines.append(f"요약 내용 차이: {c.get('content_gap', 'N/A')}")
        lines.append(f"오류 패턴: {c.get('error_pattern', 'N/A')}")
        lines.append(f"분석: {c.get('analysis_summary', 'N/A')}")
        lines.append(f"누락 지시: {c.get('missing_instruction', 'N/A')}")
        lines.append(f"위반 지시: {c.get('violated_instruction', 'N/A')}")
        lines.append(f"개선 제안: {c.get('improvement_suggestion', 'N/A')}")
        lines.append(f"STT: {stt}")
        lines.append(f"정답(Reference): {ref}")
        lines.append(f"생성(Generated): {gen}")
        lines.append("")
    return "\n".join(lines) if lines else "(대표 케이스 없음)"


async def _generate_single_candidate(
    strategy_candidate: dict, design_summary: str,
    task: dict, improvable_cases: list,
    max_retries: int = 1,
    user_guide: str = "",
    reference_style_profile: str = ""
) -> dict | None:
    """Step 2: 개별 후보의 노드 프롬프트 생성. 실패 시 재시도."""
    label = strategy_candidate["label"]
    node_count = strategy_candidate["node_count"]
    focus_patterns = strategy_candidate.get("focus_patterns", [])

    # 대표 케이스 선별
    rep_cases = _select_cases_for_candidate(focus_patterns, improvable_cases)
    cases_text = _format_cases_text(rep_cases)

    user_guide_section = ""
    if user_guide:
        user_guide_section = f"\n[사용자 전략 가이드 — 반드시 준수]\n{user_guide}\n"

    reference_criteria_section = ""
    if reference_style_profile:
        reference_criteria_section = f"\n[Reference 요약 기준 — LLM이 이 기준에 맞게 요약하도록 프롬프트를 작성하라]\n{reference_style_profile}\n"

    template = _load_prompt(CANDIDATE_PROMPT_PATH)
    prompt = template.format(
        generation_task=task.get("generation_task", "불편사항 요약"),
        candidate_label=label,
        node_count=node_count,
        node_roles=json.dumps(strategy_candidate.get("node_roles", []), ensure_ascii=False),
        node_reasoning_config=json.dumps(strategy_candidate.get("node_reasoning_config", []), ensure_ascii=False),
        rationale=strategy_candidate.get("rationale", ""),
        focus_patterns=json.dumps(focus_patterns, ensure_ascii=False),
        design_summary=design_summary,
        representative_cases=cases_text,
        user_guide=user_guide_section,
        reference_criteria=reference_criteria_section,
    )

    for attempt in range(max_retries + 1):
        try:
            raw = await call_gpt([{"role": "user", "content": prompt}], reasoning="high")
            result = _extract_json(raw)
            nodes = result.get("nodes", [])
            if nodes:
                return {
                    "label": label,
                    "node_count": node_count,
                    "rationale": strategy_candidate.get("rationale", ""),
                    "focus_patterns": focus_patterns,
                    "node_roles": strategy_candidate.get("node_roles", []),
                    "node_reasoning_config": strategy_candidate.get("node_reasoning_config", []),
                    "nodes": nodes,
                }
            logger.warning(f"[Step2] 후보 {label} attempt {attempt+1}: nodes 비어있음")
        except Exception as e:
            logger.warning(f"[Step2] 후보 {label} attempt {attempt+1} 오류: {e}")
    return None


# ─── Step 3: 검증 + 보정 ────────────────────────────────────────────────────

def _validate_candidate(candidate: dict) -> list:
    """후보의 노드 검증. 누락된 노드 라벨 리스트 반환."""
    node_count = candidate["node_count"]
    nodes = candidate.get("nodes", [])
    expected_labels = NODE_LABELS[:node_count]

    existing_labels = set()
    for n in nodes:
        lbl = n.get("node_label", "")
        if lbl and n.get("prompt", "").strip():
            existing_labels.add(lbl)

    return [lbl for lbl in expected_labels if lbl not in existing_labels]


async def _repair_candidate(
    candidate: dict, missing_labels: list, task: dict
) -> dict:
    """누락된 노드만 GPT에 보정 요청."""
    # 기존 노드 텍스트 구성
    existing_text_parts = []
    for n in candidate.get("nodes", []):
        if n.get("prompt", "").strip():
            existing_text_parts.append(
                f"노드 {n['node_label']}: {n['prompt'][:300]}..."
            )
    existing_nodes_text = "\n".join(existing_text_parts) if existing_text_parts else "(없음)"

    template = _load_prompt(REPAIR_PROMPT_PATH)
    prompt = template.format(
        generation_task=task.get("generation_task", "불편사항 요약"),
        candidate_label=candidate["label"],
        node_count=candidate["node_count"],
        node_roles=json.dumps(candidate.get("node_roles", []), ensure_ascii=False),
        node_reasoning_config=json.dumps(candidate.get("node_reasoning_config", []), ensure_ascii=False),
        rationale=candidate.get("rationale", ""),
        existing_nodes=existing_nodes_text,
        missing_labels=", ".join(missing_labels),
    )

    try:
        raw = await call_gpt([{"role": "user", "content": prompt}], reasoning="medium")
        result = _extract_json(raw)
        repaired = result.get("repaired_nodes", [])

        # 기존 nodes에 보정된 노드 병합
        existing_map = {n["node_label"]: n for n in candidate["nodes"] if n.get("node_label")}
        for rn in repaired:
            lbl = rn.get("node_label", "")
            if lbl in missing_labels and rn.get("prompt", "").strip():
                existing_map[lbl] = rn

        # 올바른 순서로 재조립
        candidate["nodes"] = [
            existing_map[lbl] for lbl in NODE_LABELS[:candidate["node_count"]]
            if lbl in existing_map
        ]
        return candidate
    except Exception as e:
        logger.warning(f"[Step3] 후보 {candidate['label']} 보정 실패: {e}")
        return candidate


# ─── DB 저장 ──────────────────────────────────────────────────────────────

async def _save_candidate_to_db(
    db, run_id: int, candidate: dict, design_mode: str
) -> None:
    """nodes[] 배열 → 기존 flat 컬럼 변환 후 DB 저장."""
    nodes = candidate.get("nodes", [])

    # flat 컬럼 매핑
    node_prompts = {"a": None, "b": None, "c": None}
    node_reasoning = {"a": 0, "b": 0, "c": 0}
    for n in nodes:
        lbl = n.get("node_label", "").lower()
        if lbl in node_prompts:
            node_prompts[lbl] = n.get("prompt")
            node_reasoning[lbl] = 1 if n.get("reasoning") else 0

    await db.execute(
        """INSERT INTO prompt_candidates
           (run_id, candidate_label, mode, workflow_spec, node_count,
            node_a_prompt, node_b_prompt, node_c_prompt,
            node_a_reasoning, node_b_reasoning, node_c_reasoning, design_rationale)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            candidate.get("label", "A"),
            design_mode,
            "",  # workflow_spec — 전략에서 별도로 안 씀
            candidate.get("node_count", 1),
            node_prompts["a"], node_prompts["b"], node_prompts["c"],
            node_reasoning["a"], node_reasoning["b"], node_reasoning["c"],
            candidate.get("rationale", ""),
        )
    )


# ─── 메인 파이프라인 ────────────────────────────────────────────────────────

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
            """INSERT INTO phase_results (run_id, phase, status, started_at)
               VALUES (?,2,'running',?)
               ON CONFLICT(run_id, phase) DO UPDATE SET status='running', started_at=excluded.started_at""",
            (run_id, datetime.utcnow().isoformat())
        )
        await db.commit()
        await db.execute("UPDATE runs SET status='phase2_running', current_phase=2 WHERE id=?", (run_id,))
        await db.commit()

        # ── 준비: 모드, learning rate, 실험 이력 ──
        prev_completed = await count_completed_runs(run["task_id"], run_id)
        design_mode = "explore" if prev_completed == 0 else "converge"
        learning_rate = await compute_learning_rate(run["task_id"], run_id)

        history_runs = await get_run_scores(run["task_id"], run_id)
        if history_runs:
            experiment_history = "\n".join(
                f"Run {hr['run_number']}: score_total={hr['score_total']}%"
                for hr in history_runs
            )
        else:
            experiment_history = "첫 번째 실험"

        improvable_cases = phase1_summary.get("prompt_improvable_cases", [])
        improvable_count = len(improvable_cases)

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM case_results WHERE run_id=?", (run_id,)
        ) as cursor:
            total_count = (await cursor.fetchone())["cnt"]

        # converge 모드: 이전 최고 프롬프트 조회
        converge_text = ""
        if design_mode == "converge":
            converge_text = await _get_best_previous_candidates(run["task_id"], run_id)

        # 이전 Run 피드백 조회 (continue 모드)
        prev_run_feedback = ""
        if run.get("base_run_id"):
            prev_run_feedback = await _get_prev_run_feedback(run["base_run_id"])
            if prev_run_feedback:
                yield log_event("info", f"이전 Run #{run['base_run_id']} 피드백 주입됨")

        # 사용자 전략 가이드
        user_guide = run.get("user_guide") or ""
        if user_guide:
            yield log_event("info", f"사용자 전략 가이드 반영: {user_guide[:80]}{'...' if len(user_guide) > 80 else ''}")

        # Reference 요약 기준 프로파일
        reference_style_profile = phase1_summary.get("reference_summary_criteria", "")
        if reference_style_profile:
            yield log_event("info", f"Reference 요약 기준 프로파일 로드됨 ({len(reference_style_profile)}자)")

        yield log_event("info", f"Design mode: {design_mode}, Learning rate: {learning_rate}")

        # ════════════════════════════════════════════════════════════════════
        # Step 1/3: 전략 수립
        # ════════════════════════════════════════════════════════════════════
        yield log_event("info", "Step 1/3: 후보 구조 전략 수립 중...")

        strategy_prompt = _build_strategy_input(
            task, phase1_summary, experiment_history,
            learning_rate, converge_text, improvable_count, total_count,
            prev_run_feedback=prev_run_feedback,
            user_guide=user_guide,
        )
        strategy_result = await _call_strategy_step(strategy_prompt, max_retries=1)

        if not strategy_result:
            yield log_event("error", "전략 수립 실패 — Phase 2를 중단합니다.")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 2)
            return

        strategy_candidates = strategy_result.get("candidates", [])
        design_summary = strategy_result.get("design_summary", "")

        node_desc = ", ".join(
            f"후보 {c['label']}: {c['node_count']}노드"
            for c in strategy_candidates
        )
        yield log_event("ok", f"전략 수립 완료 — {node_desc}")

        # ════════════════════════════════════════════════════════════════════
        # Step 2/3: 후보별 프롬프트 생성 (병렬)
        # ════════════════════════════════════════════════════════════════════
        yield log_event("info", "Step 2/3: 후보별 프롬프트 생성 중...")

        generation_tasks = [
            _generate_single_candidate(sc, design_summary, task, improvable_cases,
                                       user_guide=user_guide,
                                       reference_style_profile=reference_style_profile)
            for sc in strategy_candidates
        ]
        generation_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        generated_candidates = []
        for sc, gen_result in zip(strategy_candidates, generation_results):
            if isinstance(gen_result, Exception):
                yield log_event("warn", f"후보 {sc['label']} 생성 실패: {gen_result}")
            elif gen_result is None:
                yield log_event("warn", f"후보 {sc['label']} 생성 실패 — 스킵")
            else:
                node_count = len(gen_result.get("nodes", []))
                yield log_event("ok", f"후보 {sc['label']} 생성 완료 ({node_count} 노드)")
                generated_candidates.append(gen_result)

        if not generated_candidates:
            yield log_event("error", "모든 후보 생성 실패 — Phase 2를 중단합니다.")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 2)
            return

        # ════════════════════════════════════════════════════════════════════
        # Step 3/3: 검증 및 보정
        # ════════════════════════════════════════════════════════════════════
        yield log_event("info", "Step 3/3: 검증 및 보정 중...")

        final_candidates = []
        for cand in generated_candidates:
            missing = _validate_candidate(cand)
            if not missing:
                yield log_event("ok", f"후보 {cand['label']}: 검증 통과")
                final_candidates.append(cand)
            else:
                yield log_event("warn", f"후보 {cand['label']}: 노드 {', '.join(missing)} 누락 — 보정 중...")
                repaired = await _repair_candidate(cand, missing, task)

                # 보정 후 재검증
                still_missing = _validate_candidate(repaired)
                if not still_missing:
                    yield log_event("ok", f"후보 {repaired['label']}: 보정 완료")
                    final_candidates.append(repaired)
                else:
                    # node_count를 실제 수로 하향 조정
                    actual_count = len([
                        n for n in repaired.get("nodes", [])
                        if n.get("prompt", "").strip()
                    ])
                    if actual_count > 0:
                        repaired["node_count"] = actual_count
                        yield log_event("warn",
                            f"후보 {repaired['label']}: 보정 불완전 — node_count를 {actual_count}로 하향 조정")
                        final_candidates.append(repaired)
                    else:
                        yield log_event("warn", f"후보 {repaired['label']}: 보정 실패 — 스킵")

        if not final_candidates:
            yield log_event("error", "모든 후보 검증 실패 — Phase 2를 중단합니다.")
            yield done_event("failed")
            await _mark_phase_failed(run_id, 2)
            return

        # ════════════════════════════════════════════════════════════════════
        # DB 저장
        # ════════════════════════════════════════════════════════════════════
        for cand in final_candidates:
            await _save_candidate_to_db(db, run_id, cand, design_mode)
        await db.commit()

        yield log_event("ok", f"{len(final_candidates)}개 프롬프트 후보 설계 완료")

        # DB에서 저장된 candidates 조회 후 node_prompts 배열로 변환
        async with db.execute(
            "SELECT * FROM prompt_candidates WHERE run_id=? ORDER BY candidate_label",
            (run_id,)
        ) as cursor:
            saved_candidates = [dict(row) for row in await cursor.fetchall()]

        candidates_with_nodes = _build_candidates_with_nodes(saved_candidates)

        output = {
            "mode": design_mode,
            "learning_rate": learning_rate,
            "candidate_count": len(final_candidates),
            "design_summary": design_summary,
            "candidates": candidates_with_nodes,
            "prev_run_feedback": prev_run_feedback if prev_run_feedback else None,
            "user_guide": user_guide if user_guide else None,
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
