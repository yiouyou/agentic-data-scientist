"""Unit tests for persistent history store."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from agentic_data_scientist.core.history_store import HistoryStore, create_history_store_from_env


def test_history_store_writes_run_stage_and_decision_rows():
    db_path = Path.cwd() / f".history_test_{uuid.uuid4().hex}.sqlite3"
    store = HistoryStore(db_path)

    store.record_run(
        run_id="run-1",
        session_id="session-1",
        started_at="2026-03-04T10:00:00",
        finished_at="2026-03-04T10:01:00",
        status="completed",
        agent_type="adk",
        duration_seconds=60.0,
        events_count=42,
        files_count=3,
        total_input_tokens=1000,
        cached_input_tokens=100,
        output_tokens=300,
        working_dir="E:/tmp/work",
        error_text=None,
    )
    store.record_stage_outcomes(
        run_id="run-1",
        stage_attempts=[
            {
                "stage_index": 0,
                "attempt": 1,
                "approved": True,
                "implementation_summary": "ok",
                "review_reason": "pass",
            }
        ],
        stages_by_index={
            0: {
                "index": 0,
                "stage_id": "s1",
                "title": "Stage 1",
                "status": "approved",
                "execution_mode": "workflow",
                "workflow_id": "demo.workflow",
            }
        },
    )
    store.record_decision_traces(
        run_id="run-1",
        decisions=[
            {
                "role": "implementation_review_confirmation",
                "decision_key": "implementation_review_confirmation_decision",
                "decision_value": {"exit": True, "reason": "approved"},
                "reason": "approved",
                "source": "workflow_state",
            }
        ],
    )

    conn = sqlite3.connect(db_path)
    try:
        run_count = conn.execute("SELECT COUNT(*) FROM run_summary").fetchone()[0]
        stage_count = conn.execute("SELECT COUNT(*) FROM stage_outcome").fetchone()[0]
        decision_count = conn.execute("SELECT COUNT(*) FROM decision_trace").fetchone()[0]
    finally:
        conn.close()

    assert run_count == 1
    assert stage_count == 1
    assert decision_count == 1


def test_create_history_store_from_env_disable(monkeypatch):
    monkeypatch.setenv("ADS_HISTORY_ENABLED", "false")
    assert create_history_store_from_env() is None


def test_create_history_store_from_env_custom_path(monkeypatch):
    db_path = Path.cwd() / f".history_test_custom_{uuid.uuid4().hex}.sqlite3"
    monkeypatch.setenv("ADS_HISTORY_ENABLED", "true")
    monkeypatch.setenv("ADS_HISTORY_DB_PATH", str(db_path))

    store = create_history_store_from_env()
    assert store is not None
    assert store.db_path == db_path
