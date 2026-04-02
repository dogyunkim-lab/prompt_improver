from database import get_db


async def compute_learning_rate(task_id: int, current_run_id: int) -> str:
    db = await get_db()
    try:
        async with db.execute(
            """SELECT score_total FROM runs
               WHERE task_id=? AND id != ? AND score_total IS NOT NULL
               AND status IN ('completed','phase4_done','phase5_done','phase6_done')
               ORDER BY run_number""",
            (task_id, current_run_id)
        ) as cursor:
            rows = await cursor.fetchall()
        history = [row["score_total"] for row in rows]

        if len(history) == 0:
            return "explore"

        curr_score = history[-1]

        if len(history) >= 3:
            recent = history[-3:]
            if max(recent) - min(recent) < 0.02:
                return "major"

        if curr_score < 0.70:
            return "major"
        if curr_score < 0.85:
            return "medium"
        return "minor"
    finally:
        await db.close()


async def get_run_scores(task_id: int, exclude_run_id: int) -> list:
    db = await get_db()
    try:
        async with db.execute(
            """SELECT id, run_number, score_total, score_correct, score_over
               FROM runs WHERE task_id=? AND id != ? AND score_total IS NOT NULL
               AND status IN ('completed','phase4_done','phase5_done','phase6_done')
               ORDER BY run_number""",
            (task_id, exclude_run_id)
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
    finally:
        await db.close()


async def count_completed_runs(task_id: int, exclude_run_id: int) -> int:
    db = await get_db()
    try:
        async with db.execute(
            """SELECT COUNT(*) as cnt FROM runs
               WHERE task_id=? AND id != ? AND score_total IS NOT NULL
               AND status IN ('completed','phase4_done','phase5_done','phase6_done')""",
            (task_id, exclude_run_id)
        ) as cursor:
            row = await cursor.fetchone()
            return row["cnt"]
    finally:
        await db.close()


async def compute_and_save_deltas(task_id: int, prev_run_id: int, curr_run_id: int):
    db = await get_db()
    try:
        async with db.execute(
            "SELECT case_id, evaluation FROM case_results WHERE run_id=?",
            (prev_run_id,)
        ) as cursor:
            prev_rows = {row["case_id"]: dict(row) for row in await cursor.fetchall()}

        async with db.execute(
            "SELECT case_id, evaluation FROM case_results WHERE run_id=?",
            (curr_run_id,)
        ) as cursor:
            curr_rows = {row["case_id"]: dict(row) for row in await cursor.fetchall()}

        # 기존 delta 삭제 후 재계산 (중복 방지)
        await db.execute(
            "DELETE FROM case_deltas WHERE to_run_id=?", (curr_run_id,)
        )

        POSITIVE = {"정답", "과답"}

        for case_id, curr in curr_rows.items():
            prev_eval = prev_rows.get(case_id, {}).get("evaluation") or "없음"
            curr_eval = curr["evaluation"] or "없음"

            prev_good = prev_eval in POSITIVE
            curr_good = curr_eval in POSITIVE

            if prev_eval == curr_eval:
                delta_type = "unchanged"
            elif not prev_good and curr_good:
                # 오답/평가실패/없음 → 정답/과답: 개선
                delta_type = "improved"
            elif prev_good and not curr_good:
                # 정답/과답 → 오답/평가실패: 회귀
                delta_type = "regressed"
            else:
                # 둘 다 positive 내 이동(정답↔과답) 또는 둘 다 negative 내 이동
                delta_type = "unchanged"

            await db.execute(
                """INSERT INTO case_deltas
                   (task_id, case_id, from_run_id, to_run_id, prev_evaluation, curr_evaluation, delta_type)
                   VALUES (?,?,?,?,?,?,?)""",
                (task_id, case_id, prev_run_id, curr_run_id, prev_eval, curr_eval, delta_type)
            )

        await db.commit()
    finally:
        await db.close()


async def aggregate_scores(run_id: int) -> dict:
    db = await get_db()
    try:
        async with db.execute(
            "SELECT evaluation FROM case_results WHERE run_id=?",
            (run_id,)
        ) as cursor:
            rows = await cursor.fetchall()

        total = len(rows)
        if total == 0:
            return {"total": 0, "correct": 0, "over": 0, "wrong": 0,
                    "score_correct": 0.0, "score_over": 0.0, "score_total": 0.0}

        correct = sum(1 for r in rows if r["evaluation"] == "정답")
        over    = sum(1 for r in rows if r["evaluation"] == "과답")
        wrong   = sum(1 for r in rows if r["evaluation"] == "오답")

        return {
            "total": total,
            "correct": correct,
            "over": over,
            "wrong": wrong,
            "score_correct": round(correct / total * 100, 1),
            "score_over":    round(over    / total * 100, 1),
            "score_total":   round((correct + over) / total * 100, 1)
        }
    finally:
        await db.close()
