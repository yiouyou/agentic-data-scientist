"""Tests for MethodCriticAgent and related utilities."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_data_scientist.agents.adk.method_critic import (
    MethodCriticAgent,
    make_default_critic_output,
    parse_critic_output,
    sanitize_critic_output,
    validate_critic_output,
)
from agentic_data_scientist.core.state_contracts import StateKeys


class TestParseCriticOutput:
    def test_valid_json(self):
        text = '{"issue_type": "execution_failure", "confidence": 0.8, "evidence": ["e1"], "recommendation": "retry", "explanation": "Bug in code"}'
        result = parse_critic_output(text)
        assert result is not None
        assert result["issue_type"] == "execution_failure"

    def test_json_in_markdown_fences(self):
        text = '```json\n{"issue_type": "method_failure", "confidence": 0.9, "evidence": ["e1"], "recommendation": "backtrack", "explanation": "Wrong approach"}\n```'
        result = parse_critic_output(text)
        assert result is not None
        assert result["recommendation"] == "backtrack"

    def test_json_embedded_in_text(self):
        text = 'Here is my analysis: {"issue_type": "execution_failure", "confidence": 0.5, "evidence": [], "recommendation": "retry", "explanation": "try again"} end.'
        result = parse_critic_output(text)
        assert result is not None
        assert result["issue_type"] == "execution_failure"

    def test_invalid_text(self):
        assert parse_critic_output("no json here") is None

    def test_empty_string(self):
        assert parse_critic_output("") is None


class TestValidateCriticOutput:
    def test_valid_output(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.85,
            "evidence": ["e1", "e2"],
            "recommendation": "retry",
            "explanation": "The bug is a syntax error.",
        }
        assert validate_critic_output(output) == []

    def test_valid_method_failure(self):
        output = {
            "issue_type": "method_failure",
            "confidence": 0.9,
            "evidence": ["assumption violated"],
            "recommendation": "backtrack",
            "explanation": "Data violates normality assumption.",
        }
        assert validate_critic_output(output) == []

    def test_invalid_issue_type(self):
        output = {
            "issue_type": "unknown",
            "confidence": 0.5,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "test",
        }
        errors = validate_critic_output(output)
        assert any("issue_type" in e for e in errors)

    def test_invalid_recommendation(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.5,
            "evidence": [],
            "recommendation": "abort",
            "explanation": "test",
        }
        errors = validate_critic_output(output)
        assert any("recommendation" in e for e in errors)

    def test_confidence_out_of_range(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 1.5,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "test",
        }
        errors = validate_critic_output(output)
        assert any("confidence" in e for e in errors)

    def test_confidence_not_a_number(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": "high",
            "evidence": [],
            "recommendation": "retry",
            "explanation": "test",
        }
        errors = validate_critic_output(output)
        assert any("confidence" in e for e in errors)

    def test_evidence_not_a_list(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.5,
            "evidence": "not a list",
            "recommendation": "retry",
            "explanation": "test",
        }
        errors = validate_critic_output(output)
        assert any("evidence" in e for e in errors)

    def test_explanation_empty(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.5,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "",
        }
        errors = validate_critic_output(output)
        assert any("explanation" in e for e in errors)

    def test_explanation_missing(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.5,
            "evidence": [],
            "recommendation": "retry",
        }
        errors = validate_critic_output(output)
        assert any("explanation" in e for e in errors)

    def test_confidence_zero_is_valid(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.0,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "test",
        }
        assert validate_critic_output(output) == []

    def test_confidence_one_is_valid(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 1.0,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "test",
        }
        assert validate_critic_output(output) == []


class TestMakeDefaultCriticOutput:
    def test_returns_retry(self):
        result = make_default_critic_output()
        assert result["issue_type"] == "execution_failure"
        assert result["recommendation"] == "retry"
        assert result["confidence"] == 0.5
        assert isinstance(result["evidence"], list)
        assert isinstance(result["explanation"], str)


class TestSanitizeCriticOutput:
    def test_fixes_invalid_issue_type(self):
        output = {"issue_type": "bad", "confidence": 0.5, "evidence": [], "recommendation": "retry", "explanation": "x"}
        result = sanitize_critic_output(output)
        assert result["issue_type"] == "execution_failure"

    def test_fixes_invalid_recommendation(self):
        output = {
            "issue_type": "method_failure",
            "confidence": 0.5,
            "evidence": [],
            "recommendation": "abort",
            "explanation": "x",
        }
        result = sanitize_critic_output(output)
        assert result["recommendation"] == "retry"

    def test_clamps_confidence_high(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 2.0,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "x",
        }
        result = sanitize_critic_output(output)
        assert result["confidence"] == 1.0

    def test_clamps_confidence_low(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": -0.5,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "x",
        }
        result = sanitize_critic_output(output)
        assert result["confidence"] == 0.0

    def test_fixes_non_numeric_confidence(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": "high",
            "evidence": [],
            "recommendation": "retry",
            "explanation": "x",
        }
        result = sanitize_critic_output(output)
        assert result["confidence"] == 0.5

    def test_fixes_non_list_evidence(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.5,
            "evidence": "text",
            "recommendation": "retry",
            "explanation": "x",
        }
        result = sanitize_critic_output(output)
        assert result["evidence"] == []

    def test_fixes_empty_explanation(self):
        output = {
            "issue_type": "execution_failure",
            "confidence": 0.5,
            "evidence": [],
            "recommendation": "retry",
            "explanation": "",
        }
        result = sanitize_critic_output(output)
        assert result["explanation"] == "No explanation provided."

    def test_valid_output_unchanged(self):
        output = {
            "issue_type": "method_failure",
            "confidence": 0.9,
            "evidence": ["e1"],
            "recommendation": "backtrack",
            "explanation": "Method is wrong.",
        }
        result = sanitize_critic_output(output)
        assert result == output


class TestMethodCriticAgentGating:
    """Tests for the gating logic in MethodCriticAgent._run_async_impl."""

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
        agent = MethodCriticAgent(name="critic")
        state = {StateKeys.INNOVATION_MODE: "routine"}
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []
        assert StateKeys.METHOD_CRITIC_OUTPUT not in state

    def test_skips_when_review_approved(self):
        agent = MethodCriticAgent(name="critic")
        state = {
            StateKeys.INNOVATION_MODE: "hybrid",
            StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION: "approved",
        }
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []
        assert StateKeys.METHOD_CRITIC_OUTPUT not in state

    def test_skips_when_attempts_below_2(self):
        agent = MethodCriticAgent(name="critic")
        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION: "not-approved",
            StateKeys.HIGH_LEVEL_STAGES: [{"description": "stage 1", "attempts": 1}],
            StateKeys.CURRENT_STAGE_INDEX: 0,
        }
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []
        assert StateKeys.METHOD_CRITIC_OUTPUT not in state

    def test_attempts_extraction_from_stages(self):
        state = {
            StateKeys.HIGH_LEVEL_STAGES: [
                {"description": "s1", "attempts": 1},
                {"description": "s2", "attempts": 3},
            ],
            StateKeys.CURRENT_STAGE_INDEX: 1,
        }
        assert MethodCriticAgent._get_attempts(state) == 3

    def test_attempts_extraction_missing_stages(self):
        state = {}
        assert MethodCriticAgent._get_attempts(state) == 0

    def test_attempts_extraction_out_of_range_index(self):
        state = {
            StateKeys.HIGH_LEVEL_STAGES: [{"description": "s1", "attempts": 2}],
            StateKeys.CURRENT_STAGE_INDEX: 5,
        }
        assert MethodCriticAgent._get_attempts(state) == 0

    def test_stage_description_extraction(self):
        state = {
            StateKeys.HIGH_LEVEL_STAGES: [{"description": "Analyze data", "attempts": 1}],
            StateKeys.CURRENT_STAGE_INDEX: 0,
        }
        assert MethodCriticAgent._get_stage_description(state) == "Analyze data"

    def test_stage_description_fallback_to_title(self):
        state = {
            StateKeys.HIGH_LEVEL_STAGES: [{"title": "Data Analysis", "attempts": 1}],
            StateKeys.CURRENT_STAGE_INDEX: 0,
        }
        assert MethodCriticAgent._get_stage_description(state) == "Data Analysis"

    def test_stage_description_fallback_to_current_stage(self):
        state = {
            StateKeys.CURRENT_STAGE: "some stage text",
        }
        assert MethodCriticAgent._get_stage_description(state) == "some stage text"
