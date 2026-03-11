"""Tests for episodic innovation memory (Phase 3-C)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

from agentic_data_scientist.core.innovation_memory import (
    InnovationMemory,
    _safe_json_list,
    _truncate,
    create_innovation_memory_from_env,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_innovation_memory.sqlite3"


@pytest.fixture()
def mem(tmp_db: Path) -> InnovationMemory:
    return InnovationMemory(tmp_db)


def _record_failed(mem: InnovationMemory, n: int = 1, *, prefix: str = "fail") -> None:
    for i in range(n):
        mem.record_episode(
            run_id=f"run-{prefix}-{i}",
            method_id=f"m-{prefix}-{i}",
            method_family="baseline",
            title=f"Failed method {prefix} #{i}",
            core_hypothesis=f"Hypothesis for {prefix} #{i}",
            outcome="failed",
            failure_reason=f"Reason {prefix} #{i}",
            negative_constraints=[f"nc-{i}-a", f"nc-{i}-b"],
            tags=["t1", "t2"],
        )


def _record_succeeded(mem: InnovationMemory, n: int = 1, *, prefix: str = "ok") -> None:
    for i in range(n):
        mem.record_episode(
            run_id=f"run-{prefix}-{i}",
            method_id=f"m-{prefix}-{i}",
            method_family="negative_variant",
            title=f"Success method {prefix} #{i}",
            core_hypothesis=f"Hypothesis for {prefix} #{i}",
            outcome="succeeded",
            selection_score=0.8 + i * 0.01,
            tags=["good"],
        )


# ── Helper tests ──────────────────────────────────────────────


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("abc", 100) == "abc"

    def test_exact_boundary(self):
        text = "a" * 50
        assert _truncate(text, 50) == text

    def test_long_string_truncated(self):
        text = "x" * 200
        result = _truncate(text, 50)
        assert len(result) < 50
        assert result.endswith("... [truncated]")
        assert result.startswith("x" * 30)

    def test_none_becomes_empty(self):
        assert _truncate(None) == ""

    def test_number_converted(self):
        assert _truncate(42) == "42"


class TestSafeJsonList:
    def test_valid_json_list(self):
        assert _safe_json_list('["a", "b"]') == ["a", "b"]

    def test_empty_string(self):
        assert _safe_json_list("") == []

    def test_none(self):
        assert _safe_json_list(None) == []

    def test_invalid_json(self):
        assert _safe_json_list("not json") == []

    def test_json_dict_returns_empty(self):
        assert _safe_json_list('{"a": 1}') == []

    def test_nested_list_preserved(self):
        result = _safe_json_list('["x", "y", "z"]')
        assert result == ["x", "y", "z"]


# ── InnovationMemory core ────────────────────────────────────


class TestInnovationMemoryInit:
    def test_creates_db_file_on_first_operation(self, mem: InnovationMemory, tmp_db: Path):
        assert not tmp_db.exists()
        mem.record_episode(run_id="r1", method_id="m1", outcome="failed")
        assert tmp_db.exists()

    def test_db_path_property(self, mem: InnovationMemory, tmp_db: Path):
        assert mem.db_path == tmp_db

    def test_creates_parent_directories(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.sqlite3"
        m = InnovationMemory(deep_path)
        m.record_episode(run_id="r1", method_id="m1", outcome="failed")
        assert deep_path.exists()

    def test_schema_initialized_once(self, mem: InnovationMemory):
        assert not mem._initialized
        mem._ensure_schema()
        assert mem._initialized
        # Second call is no-op
        mem._ensure_schema()
        assert mem._initialized


class TestRecordEpisode:
    def test_basic_record(self, mem: InnovationMemory):
        mem.record_episode(
            run_id="run-1",
            method_id="m1",
            method_family="baseline",
            title="Test Method",
            core_hypothesis="Test hypothesis",
            outcome="failed",
            failure_reason="Did not converge",
            negative_constraints=["nc1", "nc2"],
            selection_score=0.75,
            tags=["tag1"],
        )
        counts = mem.get_episode_count()
        assert counts.get("failed") == 1

    def test_minimal_record(self, mem: InnovationMemory):
        mem.record_episode(run_id="r1", method_id="m1", outcome="succeeded")
        counts = mem.get_episode_count()
        assert counts.get("succeeded") == 1

    def test_multiple_records(self, mem: InnovationMemory):
        _record_failed(mem, 3)
        _record_succeeded(mem, 2)
        counts = mem.get_episode_count()
        assert counts["failed"] == 3
        assert counts["succeeded"] == 2

    def test_truncation_applied_to_long_fields(self, mem: InnovationMemory):
        long_title = "T" * 1000
        long_hypo = "H" * 5000
        long_reason = "R" * 5000
        mem.record_episode(
            run_id="r1",
            method_id="m1",
            outcome="failed",
            title=long_title,
            core_hypothesis=long_hypo,
            failure_reason=long_reason,
        )
        # Record should succeed without error (truncation happens silently)
        assert mem.get_episode_count()["failed"] == 1

    def test_none_constraints_stored_as_empty_list(self, mem: InnovationMemory):
        mem.record_episode(
            run_id="r1",
            method_id="m1",
            outcome="failed",
            negative_constraints=None,
        )
        failed = mem.get_failed_methods(limit=1)
        assert len(failed) == 1
        assert failed[0]["negative_constraints"] == []


class TestGetFailedMethods:
    def test_empty_db(self, mem: InnovationMemory):
        assert mem.get_failed_methods() == []

    def test_returns_failed_only(self, mem: InnovationMemory):
        _record_failed(mem, 2)
        _record_succeeded(mem, 3)
        failed = mem.get_failed_methods()
        assert len(failed) == 2
        for f in failed:
            assert "method_id" in f
            assert "failure_reason" in f

    def test_limit_respected(self, mem: InnovationMemory):
        _record_failed(mem, 10)
        failed = mem.get_failed_methods(limit=3)
        assert len(failed) == 3

    def test_returns_all_in_batch(self, mem: InnovationMemory):
        _record_failed(mem, 5, prefix="batch")
        failed = mem.get_failed_methods(limit=5)
        ids = {f["method_id"] for f in failed}
        assert ids == {f"m-batch-{i}" for i in range(5)}

    def test_fields_present(self, mem: InnovationMemory):
        mem.record_episode(
            run_id="r1",
            method_id="m1",
            method_family="triz_resolution",
            title="TRIZ Method",
            core_hypothesis="Resolve contradiction",
            outcome="failed",
            failure_reason="Contradiction irreconcilable",
            negative_constraints=["no_brute_force"],
            tags=["triz", "resolution"],
        )
        failed = mem.get_failed_methods(limit=1)
        assert len(failed) == 1
        ep = failed[0]
        assert ep["method_id"] == "m1"
        assert ep["method_family"] == "triz_resolution"
        assert ep["title"] == "TRIZ Method"
        assert ep["core_hypothesis"] == "Resolve contradiction"
        assert ep["failure_reason"] == "Contradiction irreconcilable"
        assert ep["negative_constraints"] == ["no_brute_force"]
        assert ep["tags"] == ["triz", "resolution"]

    def test_limit_zero_treated_as_one(self, mem: InnovationMemory):
        _record_failed(mem, 3)
        failed = mem.get_failed_methods(limit=0)
        assert len(failed) == 1  # max(1, 0) = 1


class TestGetSucceededMethods:
    def test_empty_db(self, mem: InnovationMemory):
        assert mem.get_succeeded_methods() == []

    def test_returns_succeeded_only(self, mem: InnovationMemory):
        _record_failed(mem, 3)
        _record_succeeded(mem, 2)
        succeeded = mem.get_succeeded_methods()
        assert len(succeeded) == 2
        for s in succeeded:
            assert "method_id" in s
            assert "selection_score" in s

    def test_limit_respected(self, mem: InnovationMemory):
        _record_succeeded(mem, 10)
        result = mem.get_succeeded_methods(limit=4)
        assert len(result) == 4

    def test_fields_present(self, mem: InnovationMemory):
        mem.record_episode(
            run_id="r1",
            method_id="m1",
            method_family="negative_variant",
            title="Good Method",
            core_hypothesis="Works well",
            outcome="succeeded",
            selection_score=0.92,
            tags=["good", "fast"],
        )
        results = mem.get_succeeded_methods(limit=1)
        assert len(results) == 1
        ep = results[0]
        assert ep["method_id"] == "m1"
        assert ep["method_family"] == "negative_variant"
        assert ep["title"] == "Good Method"
        assert ep["core_hypothesis"] == "Works well"
        assert ep["selection_score"] == 0.92
        assert ep["tags"] == ["good", "fast"]


class TestBuildNegativeConstraintsSummary:
    def test_empty_db_returns_empty_string(self, mem: InnovationMemory):
        assert mem.build_negative_constraints_summary() == ""

    def test_produces_text_with_failed_episodes(self, mem: InnovationMemory):
        _record_failed(mem, 3, prefix="neg")
        summary = mem.build_negative_constraints_summary()
        assert "Known failed approaches" in summary
        assert "failed:" in summary

    def test_limit_controls_entries(self, mem: InnovationMemory):
        _record_failed(mem, 20, prefix="big")
        summary = mem.build_negative_constraints_summary(limit=5)
        # build_negative_constraints_summary fetches `limit` then slices to 10
        lines = summary.strip().split("\n")
        # Header + up to 5 entries (min of limit and hardcoded 10)
        assert len(lines) <= 6

    def test_summary_includes_family_and_reason(self, mem: InnovationMemory):
        mem.record_episode(
            run_id="r1",
            method_id="m1",
            method_family="abductive_hypothesis",
            title="Test",
            core_hypothesis="Some hypothesis about hidden causes",
            outcome="failed",
            failure_reason="Data insufficient for abduction",
        )
        summary = mem.build_negative_constraints_summary()
        assert "abductive_hypothesis" in summary
        assert "Data insufficient" in summary


class TestGetEpisodeCount:
    def test_empty_db(self, mem: InnovationMemory):
        assert mem.get_episode_count() == {}

    def test_counts_by_outcome(self, mem: InnovationMemory):
        _record_failed(mem, 3)
        _record_succeeded(mem, 2)
        counts = mem.get_episode_count()
        assert counts == {"failed": 3, "succeeded": 2}

    def test_custom_outcome(self, mem: InnovationMemory):
        mem.record_episode(run_id="r1", method_id="m1", outcome="partial")
        counts = mem.get_episode_count()
        assert counts == {"partial": 1}


# ── Factory function ──────────────────────────────────────────


class TestCreateInnovationMemoryFromEnv:
    def test_default_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADS_INNOVATION_MEMORY_DB_PATH", None)
            mem = create_innovation_memory_from_env()
            assert mem is not None
            assert mem.db_path == Path(".agentic_ds_innovation_memory.sqlite3")

    def test_custom_path(self, tmp_path: Path):
        custom = str(tmp_path / "custom.sqlite3")
        with patch.dict(os.environ, {"ADS_INNOVATION_MEMORY_DB_PATH": custom}):
            mem = create_innovation_memory_from_env()
            assert mem is not None
            assert mem.db_path == Path(custom)

    def test_empty_string_returns_none(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MEMORY_DB_PATH": ""}):
            mem = create_innovation_memory_from_env()
            assert mem is None

    def test_whitespace_only_returns_none(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MEMORY_DB_PATH": "   "}):
            mem = create_innovation_memory_from_env()
            assert mem is None


# ── Thread safety smoke test ──────────────────────────────────


class TestThreadSafety:
    def test_concurrent_writes(self, mem: InnovationMemory):
        import concurrent.futures

        def _write(idx: int) -> None:
            mem.record_episode(
                run_id=f"thread-{idx}",
                method_id=f"m-{idx}",
                outcome="failed" if idx % 2 == 0 else "succeeded",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_write, range(20)))

        counts = mem.get_episode_count()
        total = sum(counts.values())
        assert total == 20


# ── Persistence test ──────────────────────────────────────────


class TestPersistence:
    def test_data_persists_across_instances(self, tmp_db: Path):
        m1 = InnovationMemory(tmp_db)
        m1.record_episode(run_id="r1", method_id="m1", outcome="failed")
        m1.record_episode(run_id="r2", method_id="m2", outcome="succeeded")

        m2 = InnovationMemory(tmp_db)
        counts = m2.get_episode_count()
        assert counts == {"failed": 1, "succeeded": 1}
