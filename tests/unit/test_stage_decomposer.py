"""Tests for stage_decomposer agent (Phase 3-B)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_data_scientist.agents.adk.stage_decomposer import (
    StageDecomposerAgent,
    apply_decomposition,
    build_decomposer_prompt,
    parse_decomposition_response,
)
from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.state_contracts import StateKeys, make_stage_record


# ── Fixtures ──────────────────────────────────────────────────


def _sample_stage(index: int = 2, title: str = "Complex ML Pipeline") -> Dict[str, Any]:
    return make_stage_record(
        index=index,
        title=title,
        description="Build and evaluate 5 ML models with cross-validation and hyperparameter tuning.",
        stage_id=f"s{index + 1}",
        inputs_required=["cleaned_data.csv"],
        outputs_produced=["best_model.pkl", "metrics.json"],
        source_method_id="m1",
        method_family="baseline",
    )


def _sample_criteria() -> List[Dict[str, Any]]:
    return [
        {"index": 0, "criteria": "Model AUC > 0.85", "met": False},
        {"index": 1, "criteria": "All results reproducible", "met": False},
    ]


def _decomposition_json(n: int = 2) -> str:
    subs = []
    for i in range(n):
        subs.append(
            {
                "title": f"Sub-stage {i + 1}",
                "description": f"Description for sub-stage {i + 1}",
                "inputs_required": [f"input_{i}.csv"],
                "outputs_produced": [f"output_{i}.csv"],
            }
        )
    return json.dumps({"decomposition_rationale": "Too complex", "sub_stages": subs})


# ── build_decomposer_prompt ──────────────────────────────────


class TestBuildDecomposerPrompt:
    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_returns_string_with_placeholders_filled(self, mock_load):
        mock_load.return_value = (
            "Title: {stage_title}\nDesc: {stage_description}\n"
            "ID: {stage_id}\nAttempts: {attempt_count}\n"
            "Failure: {failure_context}\nCriteria: {success_criteria}"
        )
        stage = _sample_stage()
        stage["attempts"] = 3
        result = build_decomposer_prompt(stage, "Review said too complex", _sample_criteria())
        assert "Complex ML Pipeline" in result
        assert "Review said too complex" in result
        assert "s3" in result
        assert "3" in result

    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_default_failure_context(self, mock_load):
        mock_load.return_value = "Failure: {failure_context}"
        result = build_decomposer_prompt(_sample_stage(), "", [])
        assert "Stage was flagged as too complex." in result


# ── parse_decomposition_response ─────────────────────────────


class TestParseDecompositionResponse:
    def test_parse_fenced_json(self):
        text = f"Here is the decomposition:\n```json\n{_decomposition_json()}\n```"
        result = parse_decomposition_response(text)
        assert result is not None
        assert len(result["sub_stages"]) == 2

    def test_parse_raw_json(self):
        result = parse_decomposition_response(_decomposition_json(3))
        assert result is not None
        assert len(result["sub_stages"]) == 3

    def test_returns_none_for_garbage(self):
        assert parse_decomposition_response("no json here") is None

    def test_returns_none_for_invalid_json(self):
        assert parse_decomposition_response("```json\n{broken\n```") is None

    def test_returns_none_for_empty_string(self):
        assert parse_decomposition_response("") is None


# ── apply_decomposition ──────────────────────────────────────


class TestApplyDecomposition:
    def _make_stages(self, n: int = 4) -> List[Dict[str, Any]]:
        return [make_stage_record(index=i, title=f"Stage {i + 1}", description=f"Desc {i + 1}") for i in range(n)]

    def test_replaces_target_stage(self):
        stages = self._make_stages(4)
        original = stages[2]
        subs = [
            {"title": "Sub A", "description": "Do A", "inputs_required": ["x"], "outputs_produced": ["y"]},
            {"title": "Sub B", "description": "Do B", "inputs_required": ["y"], "outputs_produced": ["z"]},
        ]
        result = apply_decomposition(stages, 2, subs, original)
        assert len(result) == 5  # 4 - 1 + 2
        titles = [s["title"] for s in result]
        assert "Sub A" in titles
        assert "Sub B" in titles
        assert "Stage 3" not in titles

    def test_indices_reindexed(self):
        stages = self._make_stages(3)
        original = stages[1]
        subs = [
            {"title": "X1", "description": "d1"},
            {"title": "X2", "description": "d2"},
            {"title": "X3", "description": "d3"},
        ]
        result = apply_decomposition(stages, 1, subs, original)
        assert len(result) == 5  # 3 - 1 + 3
        for i, s in enumerate(result):
            assert s["index"] == i
            assert s["stage_id"] == f"s{i + 1}"

    def test_chained_dependencies(self):
        stages = self._make_stages(2)
        original = stages[0]
        subs = [
            {"title": "A", "description": "a"},
            {"title": "B", "description": "b"},
        ]
        result = apply_decomposition(stages, 0, subs, original)
        assert result[1]["depends_on"] == [result[0]["stage_id"]]

    def test_inherits_source_method_id(self):
        stages = self._make_stages(2)
        original = stages[0]
        original["source_method_id"] = "m2"
        original["method_family"] = "triz_resolution"
        subs = [
            {"title": "A", "description": "a"},
            {"title": "B", "description": "b"},
        ]
        result = apply_decomposition(stages, 0, subs, original)
        assert result[0]["source_method_id"] == "m2"
        assert result[1]["method_family"] == "triz_resolution"

    def test_caps_at_3_sub_stages(self):
        stages = self._make_stages(2)
        original = stages[0]
        subs = [{"title": f"S{i}", "description": f"d{i}"} for i in range(5)]
        result = apply_decomposition(stages, 0, subs, original)
        decomposed_count = len(result) - 1  # original list was 2, removed 1
        assert decomposed_count == 3

    def test_returns_unchanged_if_less_than_2(self):
        stages = self._make_stages(2)
        original = stages[0]
        result = apply_decomposition(stages, 0, [{"title": "X", "description": "d"}], original)
        assert result == stages

    def test_empty_sub_stages(self):
        stages = self._make_stages(2)
        result = apply_decomposition(stages, 0, [], stages[0])
        assert result == stages

    def test_decomposed_from_tag(self):
        stages = self._make_stages(2)
        original = stages[0]
        original["stage_id"] = "s1"
        subs = [
            {"title": "A", "description": "a"},
            {"title": "B", "description": "b"},
        ]
        result = apply_decomposition(stages, 0, subs, original)
        assert result[0]["_decomposed_from"] == "s1"
        assert result[1]["_decomposed_from"] == "s1"


# ── StageDecomposerAgent (gating logic) ─────────────────────


class TestStageDecomposerAgentGating:
    def _make_ctx(self, state: Dict[str, Any]) -> MagicMock:
        ctx = MagicMock()
        ctx.session.state = state
        return ctx

    def test_skips_routine_mode(self):
        agent = StageDecomposerAgent()
        ctx = self._make_ctx({StateKeys.INNOVATION_MODE: "routine"})
        events = list(asyncio.run(_collect(agent._run_async_impl(ctx))))
        assert len(events) == 0

    def test_skips_when_budget_exhausted(self):
        budget = InnovationBudget(decomposition=1)
        budget.consume("decomposition")
        agent = StageDecomposerAgent()
        ctx = self._make_ctx(
            {
                StateKeys.INNOVATION_MODE: "innovation",
                StateKeys.INNOVATION_BUDGET: budget.to_dict(),
            }
        )
        events = list(asyncio.run(_collect(agent._run_async_impl(ctx))))
        assert len(events) == 0

    def test_skips_when_no_current_stage(self):
        agent = StageDecomposerAgent()
        ctx = self._make_ctx(
            {
                StateKeys.INNOVATION_MODE: "innovation",
                StateKeys.INNOVATION_BUDGET: InnovationBudget(decomposition=1).to_dict(),
            }
        )
        events = list(asyncio.run(_collect(agent._run_async_impl(ctx))))
        assert len(events) == 0

    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_runs_when_conditions_met(self, mock_load):
        mock_load.return_value = "Decompose: {stage_title} {stage_description} {stage_id} {attempt_count} {failure_context} {success_criteria}"

        stage = _sample_stage()
        stages = [_sample_stage(0, "S1"), _sample_stage(1, "S2"), stage, _sample_stage(3, "S4")]

        budget = InnovationBudget(decomposition=1)
        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.INNOVATION_BUDGET: budget.to_dict(),
            StateKeys.CURRENT_STAGE: dict(stage),
            StateKeys.HIGH_LEVEL_STAGES: stages,
            StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA: _sample_criteria(),
        }

        response_json = _decomposition_json(2)

        async def fake_generate(llm_request, **kwargs):
            resp = MagicMock()
            resp.content.parts = [MagicMock(text=response_json)]
            yield resp

        mock_llm = MagicMock()
        mock_llm.generate_content_async = fake_generate

        agent = StageDecomposerAgent(model=mock_llm)
        ctx = self._make_ctx(state)

        events = list(asyncio.run(_collect(agent._run_async_impl(ctx))))
        assert len(events) == 1
        assert "decomposed" in events[0].content.parts[0].text.lower()

        updated_stages = state[StateKeys.HIGH_LEVEL_STAGES]
        assert len(updated_stages) == 5  # 4 - 1 + 2

        updated_budget = InnovationBudget.from_dict(state[StateKeys.INNOVATION_BUDGET])
        assert updated_budget.remaining("decomposition") == 0

    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_handles_empty_llm_response(self, mock_load):
        mock_load.return_value = (
            "{stage_title}{stage_description}{stage_id}{attempt_count}{failure_context}{success_criteria}"
        )

        stage = _sample_stage()
        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.INNOVATION_BUDGET: InnovationBudget(decomposition=1).to_dict(),
            StateKeys.CURRENT_STAGE: dict(stage),
            StateKeys.HIGH_LEVEL_STAGES: [stage],
            StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA: [],
        }

        async def fake_generate(llm_request, **kwargs):
            resp = MagicMock()
            resp.content.parts = [MagicMock(text="")]
            yield resp

        mock_llm = MagicMock()
        mock_llm.generate_content_async = fake_generate

        agent = StageDecomposerAgent(model=mock_llm)
        ctx = self._make_ctx(state)

        events = list(asyncio.run(_collect(agent._run_async_impl(ctx))))
        assert len(events) == 0  # no decomposition event


async def _collect(agen):
    results = []
    async for event in agen:
        results.append(event)
    return results
