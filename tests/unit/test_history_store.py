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
                "status": "approved",
                "started_at": "2026-03-04T10:00:10.000",
                "finished_at": "2026-03-04T10:00:40.000",
                "duration_seconds": 30.0,
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
        stage_row = conn.execute(
            "SELECT status, attempt_started_at, attempt_finished_at, attempt_duration_seconds "
            "FROM stage_outcome WHERE run_id='run-1' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert run_count == 1
    assert stage_count == 1
    assert decision_count == 1
    assert stage_row is not None
    assert stage_row[0] == "approved"
    assert stage_row[1] == "2026-03-04T10:00:10.000"
    assert stage_row[2] == "2026-03-04T10:00:40.000"
    assert float(stage_row[3]) == 30.0


def test_history_store_migrates_stage_outcome_table_with_timing_columns():
    db_path = Path.cwd() / f".history_test_migrate_{uuid.uuid4().hex}.sqlite3"

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE stage_outcome (
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
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = HistoryStore(db_path)
    store.record_stage_outcomes(
        run_id="run-migrate-1",
        stage_attempts=[
            {
                "stage_index": 0,
                "attempt": 1,
                "approved": True,
                "status": "approved",
                "started_at": "2026-03-04T10:00:00.000",
                "finished_at": "2026-03-04T10:00:02.500",
                "duration_seconds": 2.5,
                "implementation_summary": "ok",
            }
        ],
        stages_by_index={0: {"stage_id": "s1", "title": "Stage 1"}},
    )

    conn = sqlite3.connect(db_path)
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(stage_outcome)").fetchall()}
        row = conn.execute(
            "SELECT attempt_started_at, attempt_finished_at, attempt_duration_seconds "
            "FROM stage_outcome WHERE run_id='run-migrate-1' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert "attempt_started_at" in columns
    assert "attempt_finished_at" in columns
    assert "attempt_duration_seconds" in columns
    assert row is not None
    assert row[0] == "2026-03-04T10:00:00.000"
    assert row[1] == "2026-03-04T10:00:02.500"
    assert float(row[2]) == 2.5


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


def test_history_store_hot_aggregate_and_similarity_queries():
    db_path = Path.cwd() / f".history_test_query_{uuid.uuid4().hex}.sqlite3"
    store = HistoryStore(db_path)

    store.record_run(
        run_id="run-rna-1",
        session_id="s1",
        started_at="2026-03-04T10:00:00",
        finished_at="2026-03-04T10:05:00",
        status="completed",
        agent_type="adk",
        duration_seconds=300.0,
        events_count=50,
        files_count=5,
        total_input_tokens=2000,
        cached_input_tokens=200,
        output_tokens=800,
        working_dir="E:/tmp/work",
    )
    store.record_stage_outcomes(
        run_id="run-rna-1",
        stage_attempts=[
            {"stage_index": 0, "attempt": 1, "approved": True},
            {"stage_index": 1, "attempt": 2, "approved": True},
        ],
        stages_by_index={
            0: {"stage_id": "s1", "title": "RNA-seq QC and alignment", "workflow_id": "bio.rnaseq.nfcore"},
            1: {"stage_id": "s2", "title": "Differential expression with DESeq2", "workflow_id": "bio.rnaseq.nfcore"},
        },
    )

    store.record_run(
        run_id="run-vision-1",
        session_id="s2",
        started_at="2026-03-04T11:00:00",
        finished_at="2026-03-04T11:03:00",
        status="completed",
        agent_type="adk",
        duration_seconds=180.0,
        events_count=30,
        files_count=2,
        total_input_tokens=1200,
        cached_input_tokens=100,
        output_tokens=400,
        working_dir="E:/tmp/work",
    )
    store.record_stage_outcomes(
        run_id="run-vision-1",
        stage_attempts=[{"stage_index": 0, "attempt": 1, "approved": True}],
        stages_by_index={0: {"stage_id": "s1", "title": "Image classification baseline", "workflow_id": ""}},
    )

    hot = store.get_hot_aggregate(recent_limit=20)
    assert hot["run_count"] >= 2
    assert hot["stage_count"] >= 3
    assert hot["stage_success_rate"] > 0.0
    assert isinstance(hot["top_workflows"], list)

    similar = store.get_topk_similar_runs(
        user_request="run rnaseq differential expression analysis",
        k=2,
        recent_limit=20,
    )
    assert similar
    assert similar[0]["run_id"] == "run-rna-1"

    advice = store.build_planner_advice(
        user_request="run rnaseq differential expression analysis",
        k=2,
        recent_limit=20,
    )
    assert "Historical Planning Signals" in advice
    assert "run-rna-1" in advice


def test_history_store_plan_selection_replay_records_and_summary():
    db_path = Path.cwd() / f".history_test_replay_{uuid.uuid4().hex}.sqlite3"
    store = HistoryStore(db_path)

    store.record_run(
        run_id="run-plan-1",
        session_id="s1",
        started_at="2026-03-04T10:00:00",
        finished_at="2026-03-04T10:03:00",
        status="completed",
        agent_type="adk",
        duration_seconds=180.0,
        events_count=20,
        files_count=1,
        total_input_tokens=1000,
        cached_input_tokens=100,
        output_tokens=200,
        working_dir="E:/tmp/work",
    )
    store.record_stage_outcomes(
        run_id="run-plan-1",
        stage_attempts=[
            {"stage_index": 0, "attempt": 1, "approved": True},
            {"stage_index": 1, "attempt": 2, "approved": True},
        ],
        stages_by_index={
            0: {"stage_id": "s1", "title": "Stage A"},
            1: {"stage_id": "s2", "title": "Stage B"},
        },
    )
    store.record_decision_traces(
        run_id="run-plan-1",
        decisions=[
            {
                "role": "plan_selector",
                "decision_key": "plan_selector_ranking",
                "decision_value": {
                    "selected_index": 1,
                    "baseline_index": 0,
                    "switch_applied": True,
                    "candidate_scores": [
                        {"index": 0, "score": 0.2},
                        {"index": 1, "score": 0.5},
                    ],
                },
                "reason": "switched_by_margin",
                "source": "workflow_state",
            }
        ],
    )

    records = store.get_plan_selection_replay_records(recent_limit=20)
    assert records
    assert records[0]["run_id"] == "run-plan-1"
    assert records[0]["switch_applied"] is True
    assert records[0]["policy_gain_proxy"] > 0.0

    report = store.run_counterfactual_replay(recent_limit=20)
    assert report["summary"]["records"] >= 1
    assert "counterfactual" in report["summary"]["note"]


def test_history_store_replay_tolerates_malformed_ranking_payload():
    db_path = Path.cwd() / f".history_test_replay_bad_{uuid.uuid4().hex}.sqlite3"
    store = HistoryStore(db_path)

    store.record_run(
        run_id="run-plan-bad",
        session_id="s1",
        started_at="2026-03-04T10:00:00",
        finished_at="2026-03-04T10:01:00",
        status="error",
        agent_type="adk",
        duration_seconds=60.0,
        events_count=5,
        files_count=0,
        total_input_tokens=200,
        cached_input_tokens=0,
        output_tokens=50,
        working_dir="E:/tmp/work",
    )

    store.record_decision_traces(
        run_id="run-plan-bad",
        decisions=[
            {
                "role": "plan_selector",
                "decision_key": "plan_selector_ranking",
                "decision_value": "not-a-dict",
                "reason": "",
                "source": "workflow_state",
            }
        ],
    )

    records = store.get_plan_selection_replay_records(recent_limit=20)
    assert records
    assert records[0]["baseline_index"] == 0
    assert records[0]["selected_index"] == 0
