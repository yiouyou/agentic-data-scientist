"""Unit tests for learning-informed plan candidate selector agent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_data_scientist.agents.adk.plan_selector import PlanCandidateSelectorAgent
from agentic_data_scientist.core.state_contracts import StateKeys


@pytest.mark.asyncio
async def test_plan_selector_keeps_single_candidate_trace():
    # Single candidate should still produce trace even when rollout is disabled.
    agent = PlanCandidateSelectorAgent(name="plan_selector_test", description="test")
    state = {
        StateKeys.HIGH_LEVEL_PLAN: "1. Load data\n2. Train model",
        StateKeys.PLAN_CANDIDATES: ["1. Load data\n2. Train model"],
    }
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))

    events = [event async for event in agent._run_async_impl(ctx)]
    assert events == []
    trace = state[StateKeys.PLAN_SELECTION_TRACE]
    assert trace["switch_applied"] is False
    assert trace["reason"] == "single_candidate"


@pytest.mark.asyncio
async def test_plan_selector_switches_when_margin_is_high(monkeypatch):
    monkeypatch.setenv("ADS_PLAN_SELECTOR_ENABLED", "true")
    monkeypatch.setenv("ADS_PLAN_SELECTOR_ROLLOUT_PERCENT", "100")
    monkeypatch.setenv("ADS_PLAN_RANK_MIN_SWITCH_MARGIN", "0.01")

    baseline = "1. Load csv\n2. Clean data\n3. Plot charts"
    better = "1. RNA-seq QC and alignment\n2. Differential expression with DESeq2\n3. Pathway interpretation"
    state = {
        StateKeys.ORIGINAL_USER_INPUT: "rnaseq differential expression analysis",
        StateKeys.HIGH_LEVEL_PLAN: baseline,
        StateKeys.PLAN_CANDIDATES: [baseline, better],
        StateKeys.PLANNER_HISTORY_SIGNALS: {
            "hot": {"run_count": 30, "stage_retry_rate": 0.05, "top_workflows": []},
            "topk_similar_runs": [
                {"stage_titles": ["RNA-seq QC and alignment", "Differential expression with DESeq2"]}
            ],
        },
    }
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))
    agent = PlanCandidateSelectorAgent(name="plan_selector_test", description="test")

    events = [event async for event in agent._run_async_impl(ctx)]
    assert len(events) == 1
    assert state[StateKeys.HIGH_LEVEL_PLAN] == better
    assert state[StateKeys.PLAN_SELECTION_TRACE]["switch_applied"] is True


@pytest.mark.asyncio
async def test_plan_selector_rollout_disabled_keeps_baseline(monkeypatch):
    monkeypatch.setenv("ADS_PLAN_SELECTOR_ENABLED", "false")
    monkeypatch.setenv("ADS_PLAN_SELECTOR_ROLLOUT_PERCENT", "100")

    baseline = "1. Load csv\n2. Clean data\n3. Plot charts"
    better = "1. RNA-seq QC and alignment\n2. Differential expression with DESeq2\n3. Pathway interpretation"
    state = {
        StateKeys.ORIGINAL_USER_INPUT: "rnaseq differential expression analysis",
        StateKeys.HIGH_LEVEL_PLAN: baseline,
        StateKeys.PLAN_CANDIDATES: [baseline, better],
        StateKeys.PLANNER_HISTORY_SIGNALS: {
            "hot": {"run_count": 30, "stage_retry_rate": 0.05, "top_workflows": []},
            "topk_similar_runs": [
                {"stage_titles": ["RNA-seq QC and alignment", "Differential expression with DESeq2"]}
            ],
        },
    }
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))
    agent = PlanCandidateSelectorAgent(name="plan_selector_test", description="test")

    events = [event async for event in agent._run_async_impl(ctx)]
    assert events == []
    assert state[StateKeys.HIGH_LEVEL_PLAN] == baseline
    trace = state[StateKeys.PLAN_SELECTION_TRACE]
    assert trace["switch_applied"] is False
    assert trace["reason"] == "feature_disabled"


@pytest.mark.asyncio
async def test_plan_selector_rollout_intent_filter(monkeypatch):
    monkeypatch.setenv("ADS_PLAN_SELECTOR_ENABLED", "true")
    monkeypatch.setenv("ADS_PLAN_SELECTOR_INTENT_REGEXES", "rna-?seq,variant")

    baseline = "1. Load csv\n2. Clean data\n3. Plot charts"
    better = "1. RNA-seq QC and alignment\n2. Differential expression with DESeq2\n3. Pathway interpretation"
    state = {
        StateKeys.ORIGINAL_USER_INPUT: "sales forecast analysis",
        StateKeys.HIGH_LEVEL_PLAN: baseline,
        StateKeys.PLAN_CANDIDATES: [baseline, better],
        StateKeys.PLANNER_HISTORY_SIGNALS: {"hot": {"run_count": 30, "stage_retry_rate": 0.05}},
    }
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))
    agent = PlanCandidateSelectorAgent(name="plan_selector_test", description="test")

    events = [event async for event in agent._run_async_impl(ctx)]
    assert events == []
    trace = state[StateKeys.PLAN_SELECTION_TRACE]
    assert trace["switch_applied"] is False
    assert trace["reason"] == "intent_not_matched"
