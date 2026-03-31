import aiosqlite
import os
from config import DB_PATH


async def get_db() -> aiosqlite.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                description TEXT,
                generation_task TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL REFERENCES tasks(id),
                run_number      INTEGER NOT NULL,
                start_mode      TEXT NOT NULL,
                base_run_id     INTEGER,
                status          TEXT DEFAULT 'created',
                current_phase   INTEGER DEFAULT 0,
                judge_file_path TEXT,
                total_cases     INTEGER DEFAULT 0,
                score_correct   REAL,
                score_over      REAL,
                score_total     REAL,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at    DATETIME
            );

            CREATE TABLE IF NOT EXISTS phase_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                phase       INTEGER NOT NULL,
                status      TEXT DEFAULT 'pending',
                input_data  TEXT,
                output_data TEXT,
                log_text    TEXT,
                started_at  DATETIME,
                completed_at DATETIME
            );

            CREATE TABLE IF NOT EXISTS prompt_candidates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id),
                candidate_label TEXT NOT NULL,
                mode            TEXT NOT NULL,
                workflow_spec   TEXT,
                node_count      INTEGER DEFAULT 1,
                node_a_prompt   TEXT,
                node_b_prompt   TEXT,
                node_c_prompt   TEXT,
                node_a_model    TEXT DEFAULT 'qwen3-30b',
                node_b_model    TEXT DEFAULT 'qwen3-30b',
                node_c_model    TEXT DEFAULT 'qwen3-30b',
                node_a_reasoning BOOLEAN DEFAULT FALSE,
                node_b_reasoning BOOLEAN DEFAULT FALSE,
                node_c_reasoning BOOLEAN DEFAULT FALSE,
                design_rationale TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS dify_connections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id),
                candidate_id    INTEGER REFERENCES prompt_candidates(id),
                dify_api_url    TEXT NOT NULL,
                dify_api_key    TEXT NOT NULL,
                label           TEXT,
                status          TEXT DEFAULT 'pending',
                verified_at     DATETIME,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS case_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id),
                candidate_id    INTEGER REFERENCES prompt_candidates(id),
                case_id         TEXT NOT NULL,
                generation_task TEXT,
                stt             TEXT,
                reference       TEXT,
                keywords        TEXT,
                generated       TEXT,
                evaluation      TEXT,
                reason          TEXT,
                bucket          TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS case_deltas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL REFERENCES tasks(id),
                case_id         TEXT NOT NULL,
                from_run_id     INTEGER NOT NULL REFERENCES runs(id),
                to_run_id       INTEGER NOT NULL REFERENCES runs(id),
                prev_evaluation TEXT,
                curr_evaluation TEXT,
                delta_type      TEXT,
                attributed_element TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.commit()
