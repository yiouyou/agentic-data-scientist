"""Tests for DeepVerifierAgent and related utilities."""

import asyncio
from unittest.mock import MagicMock

import pytest

from agentic_data_scientist.agents.adk.deep_verifier import (
    DeepVerifierAgent,
    make_default_verification,
    parse_verification_output,
    sanitize_verification_output,
    validate_verification_output,
)
from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.state_contracts import StateKeys


class TestParseVerificationOutput:
    def test_valid_json(self):
        text = '{"overall_verdict": "pass", "confidence": 0.9, "summary": "All good"}'
        result = parse_verification_output(text)
        assert result is not None
        assert result["overall_verdict"] == "pass"

    def test_json_in_fences(self):
        text = '```json\n{"overall_verdict": "warn", "confidence": 0.7, "summary": "Minor issues"}\n```'
        result = parse_verification_output(text)
        assert result is not None
        assert result["overall_verdict"] == "warn"

    def test_embedded_json(self):
        text = 'Result: {"overall_verdict": "fail", "confidence": 0.8, "summary": "Major problems"} done.'
        result = parse_verification_output(text)
        assert result is not None

    def test_invalid_text(self):
        assert parse_verification_output("not json") is None

    def test_empty_string(self):
        assert parse_verification_output("") is None


class TestValidateVerificationOutput:
    def test_valid_output(self):
        output = {
            "overall_verdict": "pass",
            "confidence": 0.85,
            "summary": "Analysis is sound.",
            "consistency_issues": [],
            "uncovered_risks": [],
            "criteria_corrections": [],
        }
        assert validate_verification_output(output) == []

    def test_invalid_verdict(self):
        output = {"overall_verdict": "good", "confidence": 0.5, "summary": "test"}
        errors = validate_verification_output(output)
        assert any("overall_verdict" in e for e in errors)

    def test_confidence_out_of_range(self):
        output = {"overall_verdict": "pass", "confidence": 2.0, "summary": "test"}
        errors = validate_verification_output(output)
        assert any("confidence" in e for e in errors)

    def test_missing_summary(self):
        output = {"overall_verdict": "pass", "confidence": 0.5}
        errors = validate_verification_output(output)
        assert any("summary" in e for e in errors)

    def test_empty_summary(self):
        output = {"overall_verdict": "pass", "confidence": 0.5, "summary": ""}
        errors = validate_verification_output(output)
        assert any("summary" in e for e in errors)

    def test_non_list_issues(self):
        output = {
            "overall_verdict": "pass",
            "confidence": 0.5,
            "summary": "test",
            "consistency_issues": "not a list",
        }
        errors = validate_verification_output(output)
        assert any("consistency_issues" in e for e in errors)


class TestMakeDefaultVerification:
    def test_returns_pass(self):
        result = make_default_verification()
        assert result["overall_verdict"] == "pass"
        assert result["confidence"] == 0.5
        assert isinstance(result["consistency_issues"], list)
        assert isinstance(result["uncovered_risks"], list)
        assert isinstance(result["summary"], str)


class TestSanitizeVerificationOutput:
    def test_fixes_invalid_verdict(self):
        output = {"overall_verdict": "bad", "confidence": 0.5, "summary": "test"}
        result = sanitize_verification_output(output)
        assert result["overall_verdict"] == "warn"

    def test_clamps_confidence(self):
        output = {"overall_verdict": "pass", "confidence": 5.0, "summary": "test"}
        result = sanitize_verification_output(output)
        assert result["confidence"] == 1.0

    def test_fixes_non_list_fields(self):
        output = {
            "overall_verdict": "pass",
            "confidence": 0.5,
            "summary": "test",
            "consistency_issues": "string",
            "uncovered_risks": 42,
            "criteria_corrections": {},
        }
        result = sanitize_verification_output(output)
        assert result["consistency_issues"] == []
        assert result["uncovered_risks"] == []
        assert result["criteria_corrections"] == []

    def test_fixes_missing_goal_alignment(self):
        output = {"overall_verdict": "pass", "confidence": 0.5, "summary": "test"}
        result = sanitize_verification_output(output)
        assert isinstance(result["goal_alignment"], dict)

    def test_fixes_missing_method_validity(self):
        output = {"overall_verdict": "pass", "confidence": 0.5, "summary": "test"}
        result = sanitize_verification_output(output)
        assert isinstance(result["method_validity"], dict)

    def test_valid_output_unchanged(self):
        output = {
            "overall_verdict": "pass",
            "confidence": 0.9,
            "summary": "Good analysis.",
            "goal_alignment": {"score": 0.9, "notes": "good"},
            "consistency_issues": [],
            "criteria_corrections": [],
            "method_validity": {
                "assumptions_satisfied": True,
                "invalid_conditions_triggered": False,
                "notes": "ok",
            },
            "uncovered_risks": [],
        }
        result = sanitize_verification_output(output)
        assert result == output


class TestDeepVerifierAgentGating:
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
        agent = DeepVerifierAgent(name="verifier")
        state = {StateKeys.INNOVATION_MODE: "routine"}
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []
        assert StateKeys.DEEP_VERIFICATION not in state

    def test_skips_when_budget_exhausted(self):
        agent = DeepVerifierAgent(name="verifier")
        budget = InnovationBudget(verification=1)
        budget.consume("verification")
        state = {
            StateKeys.INNOVATION_MODE: "hybrid",
            StateKeys.INNOVATION_BUDGET: budget.to_dict(),
        }
        ctx = self._make_ctx(state)
        events = self._run(agent, ctx)
        assert events == []
        assert StateKeys.DEEP_VERIFICATION not in state
