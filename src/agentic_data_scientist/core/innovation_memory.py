"""Episodic innovation memory — records method card outcomes across runs.

Phase 3-C: Stores which methods were tried, their hypotheses, outcomes,
and failure reasons.  Used by method_selector to avoid repeating failures
and by negative prompting to exclude known-bad approaches.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


def _truncate(value: Any, max_chars: int = 4000) -> str:
    text = str(value or "")
    return text[:max_chars] if len(text) <= max_chars else text[: max_chars - 20] + "... [truncated]"


class InnovationMemory:
    """SQLite-backed store for method card episodic records."""

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
                    CREATE TABLE IF NOT EXISTS method_episode (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        method_id TEXT NOT NULL,
                        method_family TEXT,
                        title TEXT,
                        core_hypothesis TEXT,
                        outcome TEXT NOT NULL,
                        failure_reason TEXT,
                        negative_constraints TEXT,
                        selection_score REAL,
                        tags TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_method_episode_run
                      ON method_episode(run_id);
                    CREATE INDEX IF NOT EXISTS idx_method_episode_family
                      ON method_episode(method_family);
                    CREATE INDEX IF NOT EXISTS idx_method_episode_outcome
                      ON method_episode(outcome);
                    """
                )
            self._initialized = True

    def record_episode(
        self,
        *,
        run_id: str,
        method_id: str,
        method_family: str = "",
        title: str = "",
        core_hypothesis: str = "",
        outcome: str,
        failure_reason: str = "",
        negative_constraints: Optional[List[str]] = None,
        selection_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self._ensure_schema()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO method_episode (
                    run_id, method_id, method_family, title, core_hypothesis,
                    outcome, failure_reason, negative_constraints,
                    selection_score, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    method_id,
                    method_family,
                    _truncate(title, 500),
                    _truncate(core_hypothesis, 2000),
                    outcome,
                    _truncate(failure_reason, 2000),
                    json.dumps(negative_constraints or []),
                    selection_score,
                    json.dumps(tags or []),
                ),
            )

    def get_failed_methods(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent failed method episodes for exclusion during generation."""
        self._ensure_schema()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT method_id, method_family, title, core_hypothesis,
                       failure_reason, negative_constraints, tags
                FROM method_episode
                WHERE outcome = 'failed'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "method_id": row[0],
                    "method_family": row[1] or "",
                    "title": row[2] or "",
                    "core_hypothesis": row[3] or "",
                    "failure_reason": row[4] or "",
                    "negative_constraints": _safe_json_list(row[5]),
                    "tags": _safe_json_list(row[6]),
                }
            )
        return results

    def get_succeeded_methods(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent succeeded method episodes for positive reference."""
        self._ensure_schema()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT method_id, method_family, title, core_hypothesis,
                       selection_score, tags
                FROM method_episode
                WHERE outcome = 'succeeded'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "method_id": row[0],
                    "method_family": row[1] or "",
                    "title": row[2] or "",
                    "core_hypothesis": row[3] or "",
                    "selection_score": row[4],
                    "tags": _safe_json_list(row[5]),
                }
            )
        return results

    def build_negative_constraints_summary(self, *, limit: int = 20) -> str:
        """Build a text summary of known-bad approaches for injection into prompts."""
        failed = self.get_failed_methods(limit=limit)
        if not failed:
            return ""

        lines = ["Known failed approaches (avoid repeating):"]
        for ep in failed[:10]:
            hypothesis = ep.get("core_hypothesis", "")[:120]
            reason = ep.get("failure_reason", "")[:100]
            lines.append(f"  - [{ep.get('method_family', '?')}] {hypothesis} — failed: {reason}")
        return "\n".join(lines)

    def get_episode_count(self) -> Dict[str, int]:
        """Return counts by outcome."""
        self._ensure_schema()
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT outcome, COUNT(*) FROM method_episode GROUP BY outcome").fetchall()
        return {str(row[0]): int(row[1]) for row in rows}


def _safe_json_list(raw: Any) -> List[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return list(parsed) if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def create_innovation_memory_from_env() -> Optional[InnovationMemory]:
    """Create innovation memory from environment settings."""
    import os

    db_path_raw = os.getenv("ADS_INNOVATION_MEMORY_DB_PATH", ".agentic_ds_innovation_memory.sqlite3").strip()
    if not db_path_raw:
        return None
    return InnovationMemory(Path(db_path_raw))
