"""Tests for ProblemFramerAgent and related utilities."""

import pytest

from agentic_data_scientist.agents.adk.problem_framer import (
    ProblemFramerAgent,
    _parse_framed_problem,
    make_default_framed_problem,
    validate_framed_problem,
)
from agentic_data_scientist.core.state_contracts import StateKeys


class TestParseFramedProblem:
    def test_valid_json(self):
        text = '{"research_goal": "test", "task_type": "modeling", "recommended_mode": "hybrid"}'
        result = _parse_framed_problem(text)
        assert result is not None
        assert result["research_goal"] == "test"

    def test_json_in_markdown_fences(self):
        text = '```json\n{"research_goal": "test", "task_type": "modeling", "recommended_mode": "routine"}\n```'
        result = _parse_framed_problem(text)
        assert result is not None
        assert result["task_type"] == "modeling"

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"research_goal": "x", "task_type": "discovery", "recommended_mode": "innovation"} done.'
        result = _parse_framed_problem(text)
        assert result is not None
        assert result["task_type"] == "discovery"

    def test_invalid_text(self):
        result = _parse_framed_problem("no json here")
        assert result is None

    def test_empty_string(self):
        result = _parse_framed_problem("")
        assert result is None

    def test_nested_json(self):
        text = '{"research_goal": "test", "task_type": "modeling", "recommended_mode": "hybrid", "knowns": ["a", "b"]}'
        result = _parse_framed_problem(text)
        assert result is not None
        assert result["knowns"] == ["a", "b"]


class TestValidateFramedProblem:
    def test_valid_problem(self):
        problem = {
            "research_goal": "Analyze gene expression",
            "task_type": "data_analysis",
            "recommended_mode": "routine",
            "knowns": ["data.csv"],
            "unknowns": ["differential genes"],
            "contradictions": [],
            "complexity_signals": [],
        }
        errors = validate_framed_problem(problem)
        assert errors == []

    def test_missing_research_goal(self):
        problem = {"task_type": "modeling", "recommended_mode": "hybrid"}
        errors = validate_framed_problem(problem)
        assert any("research_goal" in e for e in errors)

    def test_missing_task_type(self):
        problem = {"research_goal": "test", "recommended_mode": "hybrid"}
        errors = validate_framed_problem(problem)
        assert any("task_type" in e for e in errors)

    def test_missing_recommended_mode(self):
        problem = {"research_goal": "test", "task_type": "modeling"}
        errors = validate_framed_problem(problem)
        assert any("recommended_mode" in e for e in errors)

    def test_invalid_task_type(self):
        problem = {"research_goal": "test", "task_type": "invalid_type", "recommended_mode": "routine"}
        errors = validate_framed_problem(problem)
        assert any("Invalid task_type" in e for e in errors)

    def test_invalid_recommended_mode(self):
        problem = {"research_goal": "test", "task_type": "modeling", "recommended_mode": "turbo"}
        errors = validate_framed_problem(problem)
        assert any("Invalid recommended_mode" in e for e in errors)

    def test_list_fields_must_be_lists(self):
        problem = {
            "research_goal": "test",
            "task_type": "modeling",
            "recommended_mode": "hybrid",
            "knowns": "not a list",
        }
        errors = validate_framed_problem(problem)
        assert any("knowns must be a list" in e for e in errors)

    def test_all_list_fields_validated(self):
        problem = {
            "research_goal": "test",
            "task_type": "modeling",
            "recommended_mode": "hybrid",
            "knowns": "x",
            "unknowns": "y",
            "contradictions": 42,
            "complexity_signals": {},
        }
        errors = validate_framed_problem(problem)
        assert len(errors) == 4


class TestMakeDefaultFramedProblem:
    def test_returns_routine(self):
        result = make_default_framed_problem("some request")
        assert result["task_type"] == "routine_processing"
        assert result["recommended_mode"] == "routine"

    def test_truncates_long_input(self):
        long_input = "x" * 300
        result = make_default_framed_problem(long_input)
        assert len(result["research_goal"]) == 200

    def test_empty_lists(self):
        result = make_default_framed_problem("test")
        assert result["knowns"] == []
        assert result["unknowns"] == []
        assert result["contradictions"] == []
        assert result["complexity_signals"] == []


class TestProblemFramerStateKeys:
    def test_framed_problem_key_exists(self):
        assert hasattr(StateKeys, "FRAMED_PROBLEM")
        assert StateKeys.FRAMED_PROBLEM == "framed_problem"

    def test_phase2_keys_exist(self):
        assert hasattr(StateKeys, "INNOVATION_TRIGGER")
        assert hasattr(StateKeys, "METHOD_CRITIC_OUTPUT")
        assert hasattr(StateKeys, "BACKTRACK_HISTORY")
        assert hasattr(StateKeys, "DEEP_VERIFICATION")
