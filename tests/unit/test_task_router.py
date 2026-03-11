"""Tests for TaskRouterAgent, route_by_rules, resolve_mode, and get_cli_override."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_data_scientist.agents.adk.task_router import (
    TaskRouterAgent,
    get_cli_override,
    resolve_mode,
    route_by_rules,
)
from agentic_data_scientist.core.state_contracts import StateKeys


class TestGetCliOverride:
    def test_no_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            assert get_cli_override() is None

    def test_routine_override(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "routine"}):
            assert get_cli_override() == "routine"

    def test_hybrid_override(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "hybrid"}):
            assert get_cli_override() == "hybrid"

    def test_innovation_override(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "innovation"}):
            assert get_cli_override() == "innovation"

    def test_auto_is_not_valid_override(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "auto"}):
            assert get_cli_override() is None

    def test_empty_string(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": ""}):
            assert get_cli_override() is None

    def test_whitespace_stripped(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "  hybrid  "}):
            assert get_cli_override() == "hybrid"

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "INNOVATION"}):
            assert get_cli_override() == "innovation"


class TestRouteByRules:
    def test_routine_processing(self):
        fp = {"task_type": "routine_processing", "recommended_mode": "routine"}
        assert route_by_rules(fp) == "routine"

    def test_discovery_task_type(self):
        fp = {"task_type": "discovery", "recommended_mode": "routine"}
        assert route_by_rules(fp) == "innovation"

    def test_contradictions_trigger_innovation(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": ["time vs accuracy tradeoff"],
            "complexity_signals": [],
        }
        assert route_by_rules(fp) == "innovation"

    def test_data_analysis_three_signals_capped_at_hybrid(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": [],
            "complexity_signals": ["multi-omics", "novel_method", "optimization"],
        }
        assert route_by_rules(fp) == "hybrid"

    def test_non_data_analysis_three_signals_trigger_innovation(self):
        fp = {
            "task_type": "modeling",
            "contradictions": [],
            "complexity_signals": ["multi-omics", "novel_method", "optimization"],
        }
        assert route_by_rules(fp) == "innovation"

    def test_exploration_three_signals_trigger_innovation(self):
        fp = {
            "task_type": "exploration",
            "contradictions": [],
            "complexity_signals": ["a", "b", "c"],
        }
        assert route_by_rules(fp) == "innovation"

    def test_two_signals_trigger_hybrid(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": [],
            "complexity_signals": ["multi-omics", "novel_method"],
        }
        assert route_by_rules(fp) == "hybrid"

    def test_modeling_with_one_signal_trigger_hybrid(self):
        fp = {
            "task_type": "modeling",
            "contradictions": [],
            "complexity_signals": ["optimization"],
        }
        assert route_by_rules(fp) == "hybrid"

    def test_exploration_with_one_signal_trigger_hybrid(self):
        fp = {
            "task_type": "exploration",
            "contradictions": [],
            "complexity_signals": ["novel_method"],
        }
        assert route_by_rules(fp) == "hybrid"

    def test_data_analysis_with_one_signal_inconclusive(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": [],
            "complexity_signals": ["one_signal"],
        }
        assert route_by_rules(fp) is None

    def test_data_analysis_with_zero_signals_inconclusive(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": [],
            "complexity_signals": [],
        }
        assert route_by_rules(fp) is None

    def test_empty_task_type_inconclusive(self):
        fp = {"task_type": "", "contradictions": [], "complexity_signals": []}
        assert route_by_rules(fp) is None

    def test_missing_fields_treated_as_empty(self):
        fp = {"task_type": "data_analysis"}
        assert route_by_rules(fp) is None

    def test_non_list_contradictions_treated_as_empty(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": "not a list",
            "complexity_signals": [],
        }
        assert route_by_rules(fp) is None

    def test_non_list_signals_treated_as_empty(self):
        fp = {
            "task_type": "data_analysis",
            "contradictions": [],
            "complexity_signals": "not a list",
        }
        assert route_by_rules(fp) is None

    def test_routine_processing_overrides_contradictions(self):
        """Rule 1 (routine_processing) has higher priority than Rule 2 (contradictions)."""
        fp = {
            "task_type": "routine_processing",
            "contradictions": ["some contradiction"],
            "complexity_signals": ["a", "b", "c"],
        }
        assert route_by_rules(fp) == "routine"


class TestResolveMode:
    def test_cli_override_wins(self):
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "innovation"}):
            fp = {"task_type": "routine_processing", "recommended_mode": "routine"}
            assert resolve_mode(fp) == "innovation"

    def test_no_framed_problem_returns_routine(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            assert resolve_mode(None) == "routine"

    def test_empty_dict_returns_routine(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            assert resolve_mode({}) == "routine"

    def test_non_dict_returns_routine(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            assert resolve_mode("not a dict") == "routine"

    def test_rules_engine_invoked(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            fp = {"task_type": "discovery", "recommended_mode": "routine"}
            assert resolve_mode(fp) == "innovation"

    def test_tiebreaker_when_rules_inconclusive(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            fp = {
                "task_type": "data_analysis",
                "contradictions": [],
                "complexity_signals": [],
                "recommended_mode": "hybrid",
            }
            assert resolve_mode(fp) == "hybrid"

    def test_default_when_all_inconclusive(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            fp = {
                "task_type": "data_analysis",
                "contradictions": [],
                "complexity_signals": [],
                "recommended_mode": "unknown_garbage",
            }
            assert resolve_mode(fp) == "routine"

    def test_recommended_mode_innovation(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            fp = {
                "task_type": "data_analysis",
                "contradictions": [],
                "complexity_signals": [],
                "recommended_mode": "innovation",
            }
            assert resolve_mode(fp) == "innovation"


def _make_ctx(state: dict):
    """Create a minimal mock InvocationContext."""
    session = MagicMock()
    session.state = state
    ctx = MagicMock()
    ctx.session = session
    return ctx


class TestTaskRouterAgent:
    @pytest.mark.asyncio
    async def test_writes_mode_to_state(self):
        state = {
            StateKeys.FRAMED_PROBLEM: {
                "task_type": "discovery",
                "contradictions": [],
                "complexity_signals": [],
                "recommended_mode": "innovation",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            agent = TaskRouterAgent(name="task_router")
            ctx = _make_ctx(state)
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)
            assert state[StateKeys.INNOVATION_MODE] == "innovation"
            assert len(events) == 1
            assert "innovation" in events[0].content.parts[0].text

    @pytest.mark.asyncio
    async def test_routine_when_no_framed_problem(self):
        state = {}
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            agent = TaskRouterAgent(name="task_router")
            ctx = _make_ctx(state)
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)
            assert state[StateKeys.INNOVATION_MODE] == "routine"

    @pytest.mark.asyncio
    async def test_cli_override_wins_over_state(self):
        state = {
            StateKeys.FRAMED_PROBLEM: {
                "task_type": "discovery",
                "contradictions": [],
                "complexity_signals": [],
                "recommended_mode": "innovation",
            }
        }
        with patch.dict(os.environ, {"ADS_INNOVATION_MODE_OVERRIDE": "routine"}):
            agent = TaskRouterAgent(name="task_router")
            ctx = _make_ctx(state)
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)
            assert state[StateKeys.INNOVATION_MODE] == "routine"

    @pytest.mark.asyncio
    async def test_hybrid_from_two_signals(self):
        state = {
            StateKeys.FRAMED_PROBLEM: {
                "task_type": "data_analysis",
                "contradictions": [],
                "complexity_signals": ["multi-omics", "novel_method"],
                "recommended_mode": "routine",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADS_INNOVATION_MODE_OVERRIDE", None)
            agent = TaskRouterAgent(name="task_router")
            ctx = _make_ctx(state)
            async for _ in agent._run_async_impl(ctx):
                pass
            assert state[StateKeys.INNOVATION_MODE] == "hybrid"


class TestModeGatedPlanningAgent:
    def _make_dummy_sub_agent(self):
        from google.adk.agents import BaseAgent

        class _DummyAgent(BaseAgent):
            async def _run_async_impl(self, ctx):
                return
                yield  # noqa: make it a generator

        return _DummyAgent(name="dummy_sub")

    @pytest.mark.asyncio
    async def test_skips_when_innovation_mode(self):
        from agentic_data_scientist.agents.adk.agent import _ModeGatedPlanningAgent

        agent = _ModeGatedPlanningAgent(name="gated", description="test", sub_agents=[self._make_dummy_sub_agent()])
        state = {StateKeys.INNOVATION_MODE: "innovation"}
        ctx = _make_ctx(state)

        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        assert events == []

    @pytest.mark.asyncio
    async def test_skips_when_hybrid_mode(self):
        from agentic_data_scientist.agents.adk.agent import _ModeGatedPlanningAgent

        agent = _ModeGatedPlanningAgent(name="gated", description="test", sub_agents=[self._make_dummy_sub_agent()])
        state = {StateKeys.INNOVATION_MODE: "hybrid"}
        ctx = _make_ctx(state)

        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        assert events == []

    @pytest.mark.asyncio
    async def test_does_not_skip_when_routine(self):
        from agentic_data_scientist.agents.adk.agent import _ModeGatedPlanningAgent

        agent = _ModeGatedPlanningAgent(name="gated", description="test", sub_agents=[self._make_dummy_sub_agent()])
        state = {StateKeys.INNOVATION_MODE: "routine"}
        ctx = _make_ctx(state)

        async def _empty_gen(*args, **kwargs):
            return
            yield

        with patch.object(
            _ModeGatedPlanningAgent.__bases__[0],
            "_run_async_impl",
            side_effect=_empty_gen,
        ):
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

    @pytest.mark.asyncio
    async def test_does_not_skip_when_mode_missing(self):
        from agentic_data_scientist.agents.adk.agent import _ModeGatedPlanningAgent

        agent = _ModeGatedPlanningAgent(name="gated", description="test", sub_agents=[self._make_dummy_sub_agent()])
        state = {}
        ctx = _make_ctx(state)

        async def _empty_gen(*args, **kwargs):
            return
            yield

        with patch.object(
            _ModeGatedPlanningAgent.__bases__[0],
            "_run_async_impl",
            side_effect=_empty_gen,
        ):
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)
            assert events == []
