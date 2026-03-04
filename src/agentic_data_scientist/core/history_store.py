"""Persistent minimal trajectory history store for planning/routing learning."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_data_scientist.core.plan_learning import replay_selection_records


def _is_enabled(raw: str) -> bool:
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _truncate_text(value: Any, *, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 27] + "... [truncated for history]"


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_EN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "than",
    "were",
    "have",
    "has",
    "had",
    "will",
    "shall",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "into",
    "onto",
    "your",
    "you",
    "our",
    "their",
    "analysis",
    "stage",
    "plan",
    "task",
}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.findall((text or "").lower()):
        token = match.strip()
        if not token:
            continue
        if token in _EN_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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

    def get_hot_aggregate(self, *, recent_limit: int = 200) -> Dict[str, Any]:
        """Return lightweight aggregate metrics over recent runs."""
        self._ensure_schema()
        recent_limit = max(1, int(recent_limit))

        with self._lock, self._connect() as conn:
            run_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS run_count,
                    AVG(duration_seconds) AS avg_duration_seconds,
                    AVG(total_input_tokens + output_tokens) AS avg_total_tokens
                FROM (
                    SELECT run_id, duration_seconds, total_input_tokens, output_tokens
                    FROM run_summary
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (recent_limit,),
            ).fetchone()

            stage_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS stage_count,
                    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) AS approved_count,
                    SUM(CASE WHEN attempt > 1 THEN 1 ELSE 0 END) AS retried_count
                FROM stage_outcome
                WHERE run_id IN (
                    SELECT run_id
                    FROM run_summary
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (recent_limit,),
            ).fetchone()

            workflow_rows = conn.execute(
                """
                SELECT
                    workflow_id,
                    COUNT(*) AS total_count,
                    AVG(CASE WHEN approved = 1 THEN 1.0 ELSE 0.0 END) AS success_rate
                FROM stage_outcome
                WHERE workflow_id IS NOT NULL
                  AND workflow_id != ''
                  AND run_id IN (
                    SELECT run_id
                    FROM run_summary
                    ORDER BY created_at DESC
                    LIMIT ?
                  )
                GROUP BY workflow_id
                ORDER BY total_count DESC, success_rate DESC
                LIMIT 5
                """,
                (recent_limit,),
            ).fetchall()

        run_count = int((run_row[0] or 0) if run_row else 0)
        avg_duration = float((run_row[1] or 0.0) if run_row else 0.0)
        avg_tokens = float((run_row[2] or 0.0) if run_row else 0.0)
        stage_count = int((stage_row[0] or 0) if stage_row else 0)
        approved_count = int((stage_row[1] or 0) if stage_row else 0)
        retried_count = int((stage_row[2] or 0) if stage_row else 0)

        stage_success_rate = float(approved_count) / float(stage_count) if stage_count else 0.0
        stage_retry_rate = float(retried_count) / float(stage_count) if stage_count else 0.0

        workflows: List[Dict[str, Any]] = []
        for row in workflow_rows:
            workflows.append(
                {
                    "workflow_id": str(row[0]),
                    "count": int(row[1] or 0),
                    "success_rate": float(row[2] or 0.0),
                }
            )

        return {
            "run_count": run_count,
            "avg_duration_seconds": avg_duration,
            "avg_total_tokens": avg_tokens,
            "stage_count": stage_count,
            "stage_success_rate": stage_success_rate,
            "stage_retry_rate": stage_retry_rate,
            "top_workflows": workflows,
        }

    def get_topk_similar_runs(
        self,
        *,
        user_request: str,
        k: int = 3,
        recent_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return top-k similar runs using lexical overlap against stage titles/workflow ids."""
        self._ensure_schema()
        k = max(1, int(k))
        recent_limit = max(k, int(recent_limit))
        request_tokens = _tokenize(user_request)
        if not request_tokens:
            return []

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    rs.run_id,
                    rs.status,
                    rs.duration_seconds,
                    rs.events_count,
                    rs.total_input_tokens,
                    rs.output_tokens,
                    COALESCE(GROUP_CONCAT(DISTINCT so.stage_title), '') AS stage_titles,
                    COALESCE(GROUP_CONCAT(DISTINCT so.workflow_id), '') AS workflow_ids
                FROM run_summary rs
                LEFT JOIN stage_outcome so
                  ON so.run_id = rs.run_id
                GROUP BY rs.run_id
                ORDER BY rs.created_at DESC
                LIMIT ?
                """,
                (recent_limit,),
            ).fetchall()

        scored: List[Dict[str, Any]] = []
        for row in rows:
            run_id = str(row[0])
            status = str(row[1] or "")
            stage_titles_csv = str(row[6] or "")
            workflow_ids_csv = str(row[7] or "")
            candidate_text = f"{stage_titles_csv} {workflow_ids_csv}".strip()
            candidate_tokens = _tokenize(candidate_text)
            score = _jaccard(request_tokens, candidate_tokens)
            if status == "completed":
                score += 0.03
            if score <= 0.0:
                continue

            stage_titles = [item.strip() for item in stage_titles_csv.split(",") if item.strip()]
            workflow_ids = [item.strip() for item in workflow_ids_csv.split(",") if item.strip()]
            scored.append(
                {
                    "run_id": run_id,
                    "score": round(score, 4),
                    "status": status,
                    "duration_seconds": float(row[2] or 0.0),
                    "events_count": int(row[3] or 0),
                    "total_tokens": int((row[4] or 0) + (row[5] or 0)),
                    "stage_titles": stage_titles[:5],
                    "workflow_ids": workflow_ids[:5],
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    def build_planner_advice(
        self,
        *,
        user_request: str,
        k: int = 3,
        recent_limit: int = 200,
    ) -> str:
        """Build concise planner advice from hot aggregates and top-k similar runs."""
        signals = self.build_planner_signals(user_request=user_request, k=k, recent_limit=recent_limit)
        hot = signals.get("hot", {})
        topk = signals.get("topk_similar_runs", [])

        run_count = int(hot.get("run_count", 0))
        if run_count <= 0:
            return ""

        lines: List[str] = []
        lines.append("Historical Planning Signals (advice-only):")
        lines.append(
            "- Recent run stats: "
            f"runs={run_count}, "
            f"avg_duration_sec={hot.get('avg_duration_seconds', 0.0):.1f}, "
            f"avg_tokens={hot.get('avg_total_tokens', 0.0):.0f}, "
            f"stage_success_rate={hot.get('stage_success_rate', 0.0):.2f}, "
            f"stage_retry_rate={hot.get('stage_retry_rate', 0.0):.2f}."
        )

        workflows = hot.get("top_workflows", [])
        if isinstance(workflows, list) and workflows:
            wf_summaries = []
            for item in workflows[:3]:
                wf_id = str(item.get("workflow_id", ""))
                count = int(item.get("count", 0))
                success = float(item.get("success_rate", 0.0))
                wf_summaries.append(f"{wf_id} (n={count}, success={success:.2f})")
            lines.append("- High-volume workflows: " + "; ".join(wf_summaries) + ".")

        if topk:
            lines.append("- Similar past runs (for reference, not mandatory):")
            for hit in topk:
                stage_titles = hit.get("stage_titles", [])
                stage_hint = ", ".join(stage_titles[:3]) if stage_titles else "(no stage titles)"
                lines.append(
                    "  "
                    f"* run={hit.get('run_id')} score={hit.get('score'):.2f} "
                    f"status={hit.get('status')} "
                    f"tokens={hit.get('total_tokens')} "
                    f"stages={stage_hint}"
                )

        lines.append(
            "- Use these signals to improve stage ordering/dependency clarity and reduce likely retries, "
            "but prioritize current user requirements over history."
        )
        return "\n".join(lines)

    def build_planner_signals(
        self,
        *,
        user_request: str,
        k: int = 3,
        recent_limit: int = 200,
    ) -> Dict[str, Any]:
        """Build structured planner signals for advice/ranking."""
        hot = self.get_hot_aggregate(recent_limit=recent_limit)
        topk = self.get_topk_similar_runs(user_request=user_request, k=k, recent_limit=recent_limit)
        return {
            "hot": hot,
            "topk_similar_runs": topk,
            "query": {
                "k": int(max(1, k)),
                "recent_limit": int(max(1, recent_limit)),
            },
        }

    def get_plan_selection_replay_records(self, *, recent_limit: int = 200) -> List[Dict[str, Any]]:
        """Load planner selection traces joined with observed run outcomes."""
        self._ensure_schema()
        recent_limit = max(1, int(recent_limit))

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    dt.run_id,
                    dt.decision_value,
                    rs.status,
                    rs.duration_seconds,
                    rs.total_input_tokens + rs.output_tokens AS total_tokens,
                    COALESCE(so.stage_count, 0) AS stage_count,
                    COALESCE(so.retry_count, 0) AS retry_count
                FROM decision_trace dt
                INNER JOIN run_summary rs
                  ON rs.run_id = dt.run_id
                LEFT JOIN (
                    SELECT
                        run_id,
                        COUNT(*) AS stage_count,
                        SUM(CASE WHEN attempt > 1 THEN 1 ELSE 0 END) AS retry_count
                    FROM stage_outcome
                    GROUP BY run_id
                ) so
                  ON so.run_id = dt.run_id
                WHERE dt.decision_key = 'plan_selector_ranking'
                ORDER BY dt.created_at DESC
                LIMIT ?
                """,
                (recent_limit,),
            ).fetchall()

        records: List[Dict[str, Any]] = []
        for row in rows:
            run_id = str(row[0] or "")
            ranking_raw = str(row[1] or "")
            status = str(row[2] or "")
            duration_seconds = float(row[3] or 0.0)
            total_tokens = int(row[4] or 0)
            stage_count = int(row[5] or 0)
            retry_count = int(row[6] or 0)

            try:
                parsed = json.loads(ranking_raw) if ranking_raw else {}
                ranking = parsed if isinstance(parsed, dict) else {}
            except Exception:
                ranking = {}

            baseline_index = _safe_int(ranking.get("baseline_index", 0), 0)
            selected_index = _safe_int(ranking.get("selected_index", baseline_index), baseline_index)
            switch_applied = bool(ranking.get("switch_applied", False))
            candidate_scores = ranking.get("candidate_scores", [])

            selected_score = 0.0
            baseline_score = 0.0
            if isinstance(candidate_scores, list):
                for item in candidate_scores:
                    if not isinstance(item, dict):
                        continue
                    idx = _safe_int(item.get("index", -1), -1)
                    score = _safe_float(item.get("score", 0.0), 0.0)
                    if idx == selected_index:
                        selected_score = score
                    if idx == baseline_index:
                        baseline_score = score

            retry_rate = float(retry_count) / float(stage_count) if stage_count > 0 else 0.0
            observed_reward = (1.0 if status == "completed" else 0.0) - (0.3 * retry_rate) - min(
                0.2, duration_seconds / 18000.0
            )
            policy_gain_proxy = selected_score - baseline_score

            records.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "switch_applied": switch_applied,
                    "selected_index": selected_index,
                    "baseline_index": baseline_index,
                    "selected_score": selected_score,
                    "baseline_score": baseline_score,
                    "policy_gain_proxy": policy_gain_proxy,
                    "observed_reward": observed_reward,
                    "total_tokens": total_tokens,
                    "retry_rate": retry_rate,
                }
            )
        return records

    def run_counterfactual_replay(self, *, recent_limit: int = 200) -> Dict[str, Any]:
        """Generate an offline counterfactual replay report for planner selection traces."""
        records = self.get_plan_selection_replay_records(recent_limit=recent_limit)
        summary = replay_selection_records(records)
        return {
            "summary": summary,
            "records": records[:20],
            "recent_limit": int(max(1, recent_limit)),
        }


def create_history_store_from_env() -> Optional[HistoryStore]:
    """Create history store from environment settings, if enabled."""
    if not _is_enabled(os.getenv("ADS_HISTORY_ENABLED", "true")):
        return None

    db_path_raw = os.getenv("ADS_HISTORY_DB_PATH", ".agentic_ds_history.sqlite3").strip()
    if not db_path_raw:
        db_path_raw = ".agentic_ds_history.sqlite3"

    return HistoryStore(Path(db_path_raw))
