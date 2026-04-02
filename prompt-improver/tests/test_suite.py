"""
Prompt Improver — 통합 테스트 스위트
모든 핵심 기능(SSE 포맷, Task/Run CRUD, Phase 전제조건, JSON 정규화, 델타 계산)을 검증한다.
"""

import json
import pytest
import database
from database import init_db
from services.sse_helpers import log_event, progress_event, result_event, done_event
from services.delta import (
    aggregate_scores,
    compute_and_save_deltas,
    compute_learning_rate,
)


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def _parse_sse(event_str: str) -> dict:
    """'data: {...}\n\n' 형식 SSE 문자열 → dict"""
    assert event_str.startswith("data: "), f"SSE prefix missing: {event_str!r}"
    assert event_str.endswith("\n\n"), f"SSE suffix missing: {event_str!r}"
    return json.loads(event_str[6:].strip())


async def _fresh_db(tmp_path, monkeypatch) -> str:
    """테스트용 격리 DB 생성 (client 픽스처 없이 서비스 함수 직접 테스트 시 사용)"""
    test_db = str(tmp_path / "svc_test.db")
    monkeypatch.setattr(database, "DB_PATH", test_db)
    await init_db()
    return test_db


# ── SSE 헬퍼 포맷 테스트 ────────────────────────────────────────────────────────

class TestSSEHelpers:

    def test_log_event_structure(self):
        ev = log_event("info", "시작합니다")
        p = _parse_sse(ev)
        assert p["type"] == "log"
        assert p["level"] == "info"
        assert p["message"] == "시작합니다"
        assert "ts" in p

    def test_log_event_levels(self):
        for level in ("info", "ok", "warn", "error"):
            p = _parse_sse(log_event(level, "msg"))
            assert p["level"] == level

    def test_progress_event(self):
        p = _parse_sse(progress_event(3, 10))
        assert p["type"] == "progress"
        assert p["current"] == 3
        assert p["total"] == 10

    def test_progress_event_boundary(self):
        p = _parse_sse(progress_event(0, 0))
        assert p["current"] == 0

    def test_done_event_completed(self):
        p = _parse_sse(done_event("completed"))
        assert p["type"] == "done"
        assert p["status"] == "completed"

    def test_done_event_failed(self):
        p = _parse_sse(done_event("failed"))
        assert p["status"] == "failed"

    def test_result_event(self):
        data = {"scores": {"correct": 80}, "top_issues": ["issue1"]}
        p = _parse_sse(result_event(data))
        assert p["type"] == "result"
        assert p["data"]["scores"]["correct"] == 80
        assert p["data"]["top_issues"] == ["issue1"]

    def test_korean_unicode_preserved(self):
        p = _parse_sse(log_event("ok", "오답/과답 케이스 분석 완료"))
        assert p["message"] == "오답/과답 케이스 분석 완료"


# ── Task CRUD ──────────────────────────────────────────────────────────────────

class TestTasksAPI:

    async def test_list_tasks_empty(self, client):
        r = await client.get("/api/tasks")
        assert r.status_code == 200
        assert r.json() == []

    async def test_create_task_minimal(self, client):
        r = await client.post("/api/tasks", json={"name": "테스트 태스크"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "테스트 태스크"
        assert isinstance(d["id"], int)
        assert "runs" not in d or d.get("runs") is not None

    async def test_create_task_with_all_fields(self, client):
        r = await client.post("/api/tasks", json={
            "name": "Full Task",
            "description": "desc",
            "generation_task": "요약"
        })
        assert r.status_code == 200
        d = r.json()
        assert d["description"] == "desc"
        assert d["generation_task"] == "요약"

    async def test_list_tasks_includes_runs(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        r = await client.get("/api/tasks")
        task = next(t for t in r.json() if t["id"] == tid)
        assert len(task["runs"]) == 1

    async def test_get_task(self, client):
        cr = await client.post("/api/tasks", json={"name": "Get Me"})
        tid = cr.json()["id"]
        r = await client.get(f"/api/tasks/{tid}")
        assert r.status_code == 200
        assert r.json()["name"] == "Get Me"

    async def test_get_task_not_found(self, client):
        r = await client.get("/api/tasks/9999")
        assert r.status_code == 404

    async def test_update_task_name(self, client):
        cr = await client.post("/api/tasks", json={"name": "Old"})
        tid = cr.json()["id"]
        r = await client.patch(f"/api/tasks/{tid}", json={"name": "New"})
        assert r.status_code == 200
        assert r.json()["name"] == "New"

    async def test_update_task_partial_keeps_other_fields(self, client):
        cr = await client.post("/api/tasks", json={
            "name": "Task", "description": "Original Desc"
        })
        tid = cr.json()["id"]
        r = await client.patch(f"/api/tasks/{tid}", json={"name": "Renamed"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "Renamed"
        assert d["description"] == "Original Desc"

    async def test_update_task_empty_body_no_error(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        r = await client.patch(f"/api/tasks/{tid}", json={})
        assert r.status_code == 200
        assert r.json()["name"] == "Task"

    async def test_update_task_not_found(self, client):
        r = await client.patch("/api/tasks/9999", json={"name": "X"})
        assert r.status_code == 404

    async def test_delete_task(self, client):
        cr = await client.post("/api/tasks", json={"name": "Delete Me"})
        tid = cr.json()["id"]
        r = await client.delete(f"/api/tasks/{tid}")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert (await client.get(f"/api/tasks/{tid}")).status_code == 404

    async def test_delete_task_not_found(self, client):
        r = await client.delete("/api/tasks/9999")
        assert r.status_code == 404

    async def test_delete_task_cascades_runs(self, client):
        cr = await client.post("/api/tasks", json={"name": "Cascade"})
        tid = cr.json()["id"]
        rr = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        run_id = rr.json()["id"]
        await client.delete(f"/api/tasks/{tid}")
        assert (await client.get(f"/api/runs/{run_id}")).status_code == 404

    async def test_multiple_tasks_listed(self, client):
        await client.post("/api/tasks", json={"name": "A"})
        await client.post("/api/tasks", json={"name": "B"})
        r = await client.get("/api/tasks")
        names = [t["name"] for t in r.json()]
        assert "A" in names and "B" in names


# ── Run CRUD ───────────────────────────────────────────────────────────────────

class TestRunsAPI:

    async def test_create_run(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        r = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        assert r.status_code == 200
        d = r.json()
        assert d["task_id"] == tid
        assert d["run_number"] == 1
        assert d["start_mode"] == "zero"
        assert d["status"] == "created"

    async def test_run_number_auto_increments(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        r1 = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        r2 = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        assert r1.json()["run_number"] == 1
        assert r2.json()["run_number"] == 2

    async def test_run_number_independent_per_task(self, client):
        t1 = (await client.post("/api/tasks", json={"name": "T1"})).json()["id"]
        t2 = (await client.post("/api/tasks", json={"name": "T2"})).json()["id"]
        r1 = await client.post(f"/api/tasks/{t1}/runs", json={"start_mode": "zero"})
        r2 = await client.post(f"/api/tasks/{t2}/runs", json={"start_mode": "zero"})
        assert r1.json()["run_number"] == 1
        assert r2.json()["run_number"] == 1

    async def test_create_run_task_not_found(self, client):
        r = await client.post("/api/tasks/9999/runs", json={"start_mode": "zero"})
        assert r.status_code == 404

    async def test_list_runs(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        r = await client.get(f"/api/tasks/{tid}/runs")
        assert r.status_code == 200
        assert len(r.json()) == 2

    async def test_get_run_has_phases(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        rr = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        run_id = rr.json()["id"]
        r = await client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200
        d = r.json()
        assert "phases" in d
        assert isinstance(d["phases"], dict)

    async def test_get_run_not_found(self, client):
        r = await client.get("/api/runs/9999")
        assert r.status_code == 404

    async def test_delete_run(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        rr = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        run_id = rr.json()["id"]
        r = await client.delete(f"/api/runs/{run_id}")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert (await client.get(f"/api/runs/{run_id}")).status_code == 404

    async def test_delete_run_not_found(self, client):
        r = await client.delete("/api/runs/9999")
        assert r.status_code == 404

    async def test_delete_run_cascades_phase_results(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        rr = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        run_id = rr.json()["id"]

        db = await database.get_db()
        await db.execute(
            "INSERT INTO phase_results (run_id, phase, status) VALUES (?,1,'completed')",
            (run_id,)
        )
        await db.commit()
        await db.close()

        await client.delete(f"/api/runs/{run_id}")

        db = await database.get_db()
        async with db.execute(
            "SELECT id FROM phase_results WHERE run_id=?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        await db.close()
        assert row is None

    async def test_get_run_phases_populated(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        rr = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        run_id = rr.json()["id"]

        output = {"scores": {"correct": 70}, "top_issues": ["이슈1"]}
        db = await database.get_db()
        await db.execute(
            "INSERT INTO phase_results (run_id, phase, status, output_data) VALUES (?,1,'completed',?)",
            (run_id, json.dumps(output))
        )
        await db.commit()
        await db.close()

        r = await client.get(f"/api/runs/{run_id}")
        d = r.json()
        # JSON serialization converts integer keys to strings
        phase1 = d["phases"].get("1") or d["phases"].get(1)
        assert phase1 is not None
        assert phase1["status"] == "completed"
        assert phase1["scores"]["correct"] == 70
        assert phase1["top_issues"] == ["이슈1"]


# ── Phase 전제조건 검증 ──────────────────────────────────────────────────────────

class TestPhasePrerequisites:

    async def _make_run(self, client):
        cr = await client.post("/api/tasks", json={"name": "Task"})
        tid = cr.json()["id"]
        rr = await client.post(f"/api/tasks/{tid}/runs", json={"start_mode": "zero"})
        return rr.json()["id"]

    async def _insert_phase(self, run_id: int, phase: int, status: str):
        db = await database.get_db()
        await db.execute(
            "INSERT INTO phase_results (run_id, phase, status) VALUES (?,?,?)",
            (run_id, phase, status)
        )
        await db.commit()
        await db.close()

    async def test_phase2_requires_phase1_completed(self, client):
        run_id = await self._make_run(client)
        r = await client.post(f"/api/runs/{run_id}/phase/2/run")
        assert r.status_code == 400

    async def test_phase2_blocked_if_phase1_running(self, client):
        run_id = await self._make_run(client)
        await self._insert_phase(run_id, 1, "running")
        r = await client.post(f"/api/runs/{run_id}/phase/2/run")
        assert r.status_code == 400

    async def test_phase2_allowed_after_phase1_completed(self, client):
        run_id = await self._make_run(client)
        await self._insert_phase(run_id, 1, "completed")
        r = await client.post(f"/api/runs/{run_id}/phase/2/run")
        assert r.status_code == 200

    async def test_phase3_execute_requires_phase2(self, client):
        run_id = await self._make_run(client)
        r = await client.post(f"/api/runs/{run_id}/phase/3/execute")
        assert r.status_code == 400

    async def test_phase4_requires_phase3_completed(self, client):
        run_id = await self._make_run(client)
        r = await client.post(f"/api/runs/{run_id}/phase/4/run")
        assert r.status_code == 400

    async def test_phase6_requires_phase4_completed(self, client):
        run_id = await self._make_run(client)
        r = await client.post(f"/api/runs/{run_id}/phase/6/run")
        assert r.status_code == 400

    async def test_phase1_trigger_no_run(self, client):
        r = await client.post("/api/runs/9999/phase/1/run")
        assert r.status_code == 404


# ── Phase 1 JSON 정규화 로직 (단위 테스트) ────────────────────────────────────────

class TestPhase1JSONNormalization:
    """
    phase1_analysis.py의 정규화 로직을 순수 파이썬으로 검증한다.
    GPT 호출 없이 정규화 경로만 테스트.
    """

    def _normalize(self, judge_data):
        """phase1_analysis.py의 JSON 정규화 로직 복제"""
        errors = []
        if isinstance(judge_data, dict):
            list_values = [(k, v) for k, v in judge_data.items() if isinstance(v, list)]
            if list_values:
                key, judge_data = max(list_values, key=lambda x: len(x[1]))
            else:
                errors.append(f"dict에 list가 없습니다. 키: {list(judge_data.keys())}")
                return None, errors

        if not isinstance(judge_data, list):
            errors.append(f"list가 필요합니다. 실제 타입: {type(judge_data).__name__}")
            return None, errors

        judge_data = [c for c in judge_data if isinstance(c, dict)]
        if not judge_data:
            errors.append("유효한 케이스(dict)를 찾을 수 없습니다.")
            return None, errors

        return judge_data, errors

    def test_plain_list(self):
        data = [{"id": "1", "evaluation": "오답"}, {"id": "2", "evaluation": "정답"}]
        result, errs = self._normalize(data)
        assert errs == []
        assert len(result) == 2

    def test_dict_wrapped_results_key(self):
        data = {"results": [{"id": "1"}, {"id": "2"}]}
        result, errs = self._normalize(data)
        assert errs == []
        assert len(result) == 2

    def test_dict_wrapped_data_key(self):
        data = {"data": [{"id": str(i)} for i in range(5)]}
        result, errs = self._normalize(data)
        assert errs == []
        assert len(result) == 5

    def test_dict_multiple_lists_picks_largest(self):
        data = {"small": [{"id": "1"}], "large": [{"id": str(i)} for i in range(8)]}
        result, errs = self._normalize(data)
        assert errs == []
        assert len(result) == 8

    def test_dict_no_list_returns_error(self):
        data = {"key": "value", "num": 123}
        result, errs = self._normalize(data)
        assert result is None
        assert len(errs) > 0

    def test_non_list_non_dict_returns_error(self):
        result, errs = self._normalize("invalid string")
        assert result is None
        assert len(errs) > 0

    def test_non_dict_items_filtered_out(self):
        data = [{"id": "1"}, "invalid", None, 42, {"id": "2"}]
        result, errs = self._normalize(data)
        assert errs == []
        assert len(result) == 2
        assert all(isinstance(c, dict) for c in result)

    def test_all_non_dict_items_returns_error(self):
        data = ["string1", "string2", None]
        result, errs = self._normalize(data)
        assert result is None
        assert len(errs) > 0

    def test_error_case_extraction(self):
        data = [
            {"id": "1", "evaluation": "오답"},
            {"id": "2", "evaluation": "정답"},
            {"id": "3", "evaluation": "과답"},
            {"id": "4", "evaluation": "정답"},
        ]
        error_cases = [c for c in data if c.get("evaluation") in ("오답", "과답")]
        assert len(error_cases) == 2
        assert {c["id"] for c in error_cases} == {"1", "3"}

    def test_empty_list_returns_error(self):
        result, errs = self._normalize([])
        assert result is None
        assert len(errs) > 0


# ── 델타 계산 (aggregate_scores, compute_and_save_deltas, compute_learning_rate) ─

class TestDeltaCalculations:

    async def test_aggregate_scores_no_cases(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        result = await aggregate_scores(9999)
        assert result["total"] == 0
        assert result["score_total"] == 0.0

    async def test_aggregate_scores_all_correct(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        for i in range(4):
            await db.execute(
                "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,?,?)",
                (str(i), "정답")
            )
        await db.commit()
        await db.close()

        result = await aggregate_scores(1)
        assert result["total"] == 4
        assert result["correct"] == 4
        assert result["score_total"] == 100.0

    async def test_aggregate_scores_mixed(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        # 5 정답, 2 과답, 3 오답 → total=10, correct+over=70%
        for i, ev in enumerate(["정답"] * 5 + ["과답"] * 2 + ["오답"] * 3):
            await db.execute(
                "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,?,?)",
                (str(i), ev)
            )
        await db.commit()
        await db.close()

        result = await aggregate_scores(1)
        assert result["total"] == 10
        assert result["correct"] == 5
        assert result["over"] == 2
        assert result["wrong"] == 3
        assert result["score_correct"] == 50.0
        assert result["score_over"] == 20.0
        assert result["score_total"] == 70.0

    async def test_compute_deltas_improved_regressed(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        # prev: c1=오답, c2=정답
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,'c1','오답')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,'c2','정답')"
        )
        # curr: c1=정답(개선), c2=오답(퇴보)
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (2,'c1','정답')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (2,'c2','오답')"
        )
        await db.commit()
        await db.close()

        await compute_and_save_deltas(task_id=1, prev_run_id=1, curr_run_id=2)

        db = await database.get_db()
        async with db.execute(
            "SELECT case_id, delta_type FROM case_deltas ORDER BY case_id"
        ) as cur:
            rows = {r["case_id"]: r["delta_type"] for r in await cur.fetchall()}
        await db.close()

        assert rows["c1"] == "improved"
        assert rows["c2"] == "regressed"

    async def test_compute_deltas_unchanged(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,'c1','정답')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (2,'c1','정답')"
        )
        await db.commit()
        await db.close()

        await compute_and_save_deltas(task_id=1, prev_run_id=1, curr_run_id=2)

        db = await database.get_db()
        async with db.execute(
            "SELECT delta_type FROM case_deltas WHERE case_id='c1'"
        ) as cur:
            row = await cur.fetchone()
        await db.close()
        assert row["delta_type"] == "unchanged"

    async def test_compute_deltas_과답_to_오답_is_regressed(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,'c1','과답')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (2,'c1','오답')"
        )
        await db.commit()
        await db.close()

        await compute_and_save_deltas(task_id=1, prev_run_id=1, curr_run_id=2)

        db = await database.get_db()
        async with db.execute(
            "SELECT delta_type FROM case_deltas WHERE case_id='c1'"
        ) as cur:
            row = await cur.fetchone()
        await db.close()
        assert row["delta_type"] == "regressed"

    async def test_compute_deltas_오답_to_과답_is_improved(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (1,'c1','오답')"
        )
        await db.execute(
            "INSERT INTO case_results (run_id, case_id, evaluation) VALUES (2,'c1','과답')"
        )
        await db.commit()
        await db.close()

        await compute_and_save_deltas(task_id=1, prev_run_id=1, curr_run_id=2)

        db = await database.get_db()
        async with db.execute(
            "SELECT delta_type FROM case_deltas WHERE case_id='c1'"
        ) as cur:
            row = await cur.fetchone()
        await db.close()
        assert row["delta_type"] == "improved"

    async def test_learning_rate_explore_no_history(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,1,'zero')"
        )
        await db.commit()
        await db.close()

        rate = await compute_learning_rate(task_id=1, current_run_id=1)
        assert rate == "explore"

    async def test_learning_rate_major_low_score(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode, score_total, status)"
            " VALUES (1,1,'zero',0.60,'completed')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        await db.commit()
        await db.close()

        rate = await compute_learning_rate(task_id=1, current_run_id=2)
        assert rate == "major"

    async def test_learning_rate_medium_mid_score(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode, score_total, status)"
            " VALUES (1,1,'zero',0.78,'completed')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        await db.commit()
        await db.close()

        rate = await compute_learning_rate(task_id=1, current_run_id=2)
        assert rate == "medium"

    async def test_learning_rate_minor_high_score(self, tmp_path, monkeypatch):
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode, score_total, status)"
            " VALUES (1,1,'zero',0.90,'completed')"
        )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,2,'zero')"
        )
        await db.commit()
        await db.close()

        rate = await compute_learning_rate(task_id=1, current_run_id=2)
        assert rate == "minor"

    async def test_learning_rate_plateau_triggers_major(self, tmp_path, monkeypatch):
        """최근 3회 점수 편차 < 0.02 → 정체 → major"""
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        for i, score in enumerate([0.91, 0.91, 0.92], start=1):
            await db.execute(
                "INSERT INTO runs (task_id, run_number, start_mode, score_total, status)"
                " VALUES (1,?,?,?,'completed')",
                (i, "zero", score)
            )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,4,'zero')"
        )
        await db.commit()
        await db.close()

        rate = await compute_learning_rate(task_id=1, current_run_id=4)
        assert rate == "major"

    async def test_learning_rate_plateau_boundary_exact(self, tmp_path, monkeypatch):
        """편차 정확히 0.02 → 정체 아님 (< 0.02 조건)"""
        await _fresh_db(tmp_path, monkeypatch)
        db = await database.get_db()
        await db.execute("INSERT INTO tasks (name) VALUES ('T')")
        for i, score in enumerate([0.90, 0.91, 0.92], start=1):
            await db.execute(
                "INSERT INTO runs (task_id, run_number, start_mode, score_total, status)"
                " VALUES (1,?,?,?,'completed')",
                (i, "zero", score)
            )
        await db.execute(
            "INSERT INTO runs (task_id, run_number, start_mode) VALUES (1,4,'zero')"
        )
        await db.commit()
        await db.close()

        rate = await compute_learning_rate(task_id=1, current_run_id=4)
        # max-min = 0.02, NOT < 0.02, so fall through to score-based: 0.92 ≥ 0.85 → minor
        assert rate == "minor"
