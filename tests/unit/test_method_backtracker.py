"""Tests for MethodBacktrackerAgent and related utilities."""

import asyncio
from unittest.mock import MagicMock

import pytest

from agentic_data_scientist.agents.adk.method_backtracker import (
    MethodBacktrackerAgent,
    execute_backtrack,
    should_backtrack,
)
from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.state_contracts import StateKeys


def _make_critic_output(issue_type="method_failure", recommendation="backtrack", explanation="method is wrong"):
    return {
        "issue_type": issue_type,
        "confidence": 0.9,
        "evidence": ["e1"],
        "recommendation": recommendation,
        "explanation": explanation,
    }


def _make_method(method_id, title="Test Method", status="standby"):
    return {
        "method_id": method_id,
        "method_family": "negative_variant",
        "title": title,
        "core_hypothesis": "test",
        "assumptions": [],
        "invalid_if": [],
        "cheap_test": "test",
        "failure_modes": [],
        "required_capabilities": [],
        "expected_artifacts": [],
        "orthogonality_tags": [],
        "status": status,
    }


def _budget_with_backtrack(n=1):
    return InnovationBudget(backtrack=n).to_dict()


class TestShouldBacktrack:
    def test_all_conditions_met(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
        }
        assert should_backtrack(state) is True

    def test_no_critic_output(self):
        state = {
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
        }
        assert should_backtrack(state) is False

    def test_critic_says_retry(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(recommendation="retry"),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
        }
        assert should_backtrack(state) is False

    def test_critic_says_execution_failure(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(issue_type="execution_failure"),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
        }
        assert should_backtrack(state) is False

    def test_no_standby_methods(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
            StateKeys.STANDBY_METHODS: [],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
        }
        assert should_backtrack(state) is False

    def test_budget_exhausted(self):
        budget = InnovationBudget(backtrack=1)
        budget.consume("backtrack")
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: budget.to_dict(),
        }
        assert should_backtrack(state) is False

    def test_no_budget_data_defaults_to_zero(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
        }
        assert should_backtrack(state) is False


class TestExecuteBacktrack:
    def test_basic_backtrack(self):
        current = _make_method("m1", status="selected")
        standby = [_make_method("m2"), _make_method("m3")]
        state = {
            StateKeys.SELECTED_METHOD: current,
            StateKeys.STANDBY_METHODS: standby,
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(explanation="bad method"),
        }
        result = execute_backtrack(state)

        assert result is not None
        assert result["method_id"] == "m2"
        assert result["status"] == "selected"
        assert state[StateKeys.SELECTED_METHOD]["method_id"] == "m2"
        assert current["status"] == "failed"
        assert current["rejection_reason"] == "bad method"
        assert len(state[StateKeys.STANDBY_METHODS]) == 1

    def test_budget_consumed(self):
        state = {
            StateKeys.SELECTED_METHOD: _make_method("m1", status="selected"),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
        }
        execute_backtrack(state)
        budget = InnovationBudget.from_dict(state[StateKeys.INNOVATION_BUDGET])
        assert budget.remaining("backtrack") == 0

    def test_backtrack_history_created(self):
        state = {
            StateKeys.SELECTED_METHOD: _make_method("m1", status="selected"),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(explanation="wrong approach"),
        }
        execute_backtrack(state)
        history = state[StateKeys.BACKTRACK_HISTORY]
        assert len(history) == 1
        assert history[0]["from_method"] == "m1"
        assert history[0]["to_method"] == "m2"
        assert history[0]["reason"] == "wrong approach"

    def test_backtrack_history_appends(self):
        state = {
            StateKeys.SELECTED_METHOD: _make_method("m2", status="selected"),
            StateKeys.STANDBY_METHODS: [_make_method("m3")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(2),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
            StateKeys.BACKTRACK_HISTORY: [{"from_method": "m1", "to_method": "m2", "reason": "prev", "evidence": []}],
        }
        execute_backtrack(state)
        assert len(state[StateKeys.BACKTRACK_HISTORY]) == 2

    def test_empty_standby_returns_none(self):
        state = {
            StateKeys.SELECTED_METHOD: _make_method("m1", status="selected"),
            StateKeys.STANDBY_METHODS: [],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
        }
        assert execute_backtrack(state) is None

    def test_exhausted_budget_returns_none(self):
        budget = InnovationBudget(backtrack=1)
        budget.consume("backtrack")
        state = {
            StateKeys.SELECTED_METHOD: _make_method("m1", status="selected"),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: budget.to_dict(),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
        }
        assert execute_backtrack(state) is None

    def test_no_current_method(self):
        state = {
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
        }
        result = execute_backtrack(state)
        assert result is not None
        assert result["method_id"] == "m2"
        history = state[StateKeys.BACKTRACK_HISTORY]
        assert history[0]["from_method"] == "unknown"


class TestMethodBacktrackerAgentGating:
    def _make_ctx(self, state: dict):
        ctx = MagicMock()
        ctx.session.state = state
        return ctx

    def _run(self, agent, ctx):
        events = []

        async def _collect():
            async for event in agent._run_async_impl(ctx):
                events.append(event)

        asyncio.run(_collect())
        return events

    def test_skips_in_routine_mode(self):
        agent = MethodBacktrackerAgent(name="backtracker")
        state = {StateKeys.INNOVATION_MODE: "routine"}
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []

    def test_skips_when_conditions_not_met(self):
        agent = MethodBacktrackerAgent(name="backtracker")
        state = {StateKeys.INNOVATION_MODE: "hybrid"}
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []

    def test_executes_when_conditions_met(self):
        agent = MethodBacktrackerAgent(name="backtracker")
        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.METHOD_CRITIC_OUTPUT: _make_critic_output(),
            StateKeys.SELECTED_METHOD: _make_method("m1", status="selected"),
            StateKeys.STANDBY_METHODS: [_make_method("m2")],
            StateKeys.INNOVATION_BUDGET: _budget_with_backtrack(1),
        }
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert len(events) == 1
        assert "m2" in events[0].content.parts[0].text
        assert state[StateKeys.SELECTED_METHOD]["method_id"] == "m2"
