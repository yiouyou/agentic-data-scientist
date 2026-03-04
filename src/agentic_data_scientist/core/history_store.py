"""Persistent minimal trajectory history store for planning/routing learning."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_enabled(raw: str) -> bool:
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _truncate_text(value: Any, *, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 27] + "... [truncated for history]"


class HistoryStore:
    """SQLite-backed store for compact run/stage/decision records."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS run_summary (
                        run_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        agent_type TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        events_count INTEGER NOT NULL,
                        files_count INTEGER NOT NULL,
                        total_input_tokens INTEGER NOT NULL,
                        cached_input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        error_text TEXT,
                        working_dir TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS stage_outcome (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        stage_index INTEGER NOT NULL,
                        stage_id TEXT,
                        stage_title TEXT,
                        attempt INTEGER NOT NULL,
                        approved INTEGER NOT NULL,
                        status TEXT,
                        execution_mode TEXT,
                        workflow_id TEXT,
                        implementation_summary TEXT,
                        review_reason TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS decision_trace (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        decision_key TEXT NOT NULL,
                        decision_value TEXT NOT NULL,
                        reason TEXT,
                        source TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_run_summary_created_at
                      ON run_summary(created_at);

                    CREATE INDEX IF NOT EXISTS idx_stage_outcome_run_id
                      ON stage_outcome(run_id);

                    CREATE INDEX IF NOT EXISTS idx_decision_trace_run_id
                      ON decision_trace(run_id);
                    """
                )
            self._initialized = True

    def record_run(
        self,
        *,
        run_id: str,
        session_id: str,
        started_at: str,
        finished_at: str,
        status: str,
        agent_type: str,
        duration_seconds: float,
        events_count: int,
        files_count: int,
        total_input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        working_dir: str,
        error_text: Optional[str] = None,
    ) -> None:
        self._ensure_schema()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_summary (
                    run_id, session_id, started_at, finished_at, status, agent_type,
                    duration_seconds, events_count, files_count,
                    total_input_tokens, cached_input_tokens, output_tokens,
                    error_text, working_dir
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    started_at,
                    finished_at,
                    status,
                    agent_type,
                    float(duration_seconds),
                    int(events_count),
                    int(files_count),
                    int(total_input_tokens),
                    int(cached_input_tokens),
                    int(output_tokens),
                    _truncate_text(error_text, max_chars=1000) if error_text else None,
                    working_dir,
                ),
            )

    def record_stage_outcomes(
        self,
        *,
        run_id: str,
        stage_attempts: List[Dict[str, Any]],
        stages_by_index: Dict[int, Dict[str, Any]],
    ) -> None:
        if not stage_attempts:
            return
        self._ensure_schema()

        rows = []
        for attempt in stage_attempts:
            stage_index = int(attempt.get("stage_index", -1))
            stage = stages_by_index.get(stage_index, {})
            rows.append(
                (
                    run_id,
                    stage_index,
                    stage.get("stage_id"),
                    stage.get("title") or attempt.get("stage_title"),
                    int(attempt.get("attempt", 0)),
                    1 if bool(attempt.get("approved", False)) else 0,
                    stage.get("status"),
                    stage.get("execution_mode"),
                    stage.get("workflow_id"),
                    _truncate_text(attempt.get("implementation_summary"), max_chars=4000),
                    _truncate_text(attempt.get("review_reason"), max_chars=2000),
                )
            )

        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO stage_outcome (
                    run_id, stage_index, stage_id, stage_title, attempt, approved,
                    status, execution_mode, workflow_id, implementation_summary, review_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_decision_traces(self, *, run_id: str, decisions: List[Dict[str, Any]]) -> None:
        if not decisions:
            return
        self._ensure_schema()

        rows = []
        for decision in decisions:
            rows.append(
                (
                    run_id,
                    str(decision.get("role", "unknown")),
                    str(decision.get("decision_key", "unknown")),
                    json.dumps(decision.get("decision_value"), ensure_ascii=True, sort_keys=True),
                    _truncate_text(decision.get("reason"), max_chars=1000),
                    _truncate_text(decision.get("source"), max_chars=200),
                )
            )

        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO decision_trace (
                    run_id, role, decision_key, decision_value, reason, source
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def create_history_store_from_env() -> Optional[HistoryStore]:
    """Create history store from environment settings, if enabled."""
    if not _is_enabled(os.getenv("ADS_HISTORY_ENABLED", "true")):
        return None

    db_path_raw = os.getenv("ADS_HISTORY_DB_PATH", ".agentic_ds_history.sqlite3").strip()
    if not db_path_raw:
        db_path_raw = ".agentic_ds_history.sqlite3"

    return HistoryStore(Path(db_path_raw))

