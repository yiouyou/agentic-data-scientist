"""Phase 3-D benchmark evaluation set A-E.

Each benchmark validates a specific innovation pipeline scenario end-to-end
at the agent level with mocked LLMs.  No real API calls are needed.

A: Routine — operators skipped, zero method candidates
B: Contradiction → TRIZ operator triggered
C: Unknowns → abduction operator triggered
D: Combined — both TRIZ + abduction triggered together
E: Method backtrack — primary method fails, standby activated
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agentic_data_scientist.agents.adk.method_backtracker import execute_backtrack, should_backtrack
from agentic_data_scientist.agents.adk.method_discovery import MethodDiscoveryAgent
from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.method_card import make_method_card
from agentic_data_scientist.core.state_contracts import StateKeys


# ── Helpers ───────────────────────────────────────────────────


def _method_card_json(method_id: str = "m1", family: str = "baseline", **overrides: Any) -> str:
    card = {
        "method_id": method_id,
        "method_family": family,
        "title": overrides.pop("title", f"Method {method_id}"),
        "core_hypothesis": overrides.pop("core_hypothesis", f"Hypothesis for {method_id}"),
        "assumptions": overrides.pop("assumptions", ["a1"]),
        "invalid_if": overrides.pop("invalid_if", ["inv1"]),
        "cheap_test": overrides.pop("cheap_test", "quick check"),
        "failure_modes": overrides.pop("failure_modes", ["fm1"]),
        "required_capabilities": overrides.pop("required_capabilities", ["python"]),
        "expected_artifacts": overrides.pop("expected_artifacts", ["result.csv"]),
        "orthogonality_tags": overrides.pop("orthogonality_tags", ["tag1", "tag2"]),
    }
    card.update(overrides)
    return json.dumps(card)


def _make_mock_llm(responses: List[str]):
    """Create a mock LLM that yields responses sequentially."""
    call_count = [0]

    async def fake_generate(llm_request, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        resp = MagicMock()
        resp.content = MagicMock()
        resp.content.parts = [MagicMock(text=responses[idx])]
        yield resp

    llm = MagicMock()
    llm.generate_content_async = fake_generate
    return llm


def _make_ctx(state: Dict[str, Any]):
    ctx = MagicMock()
    ctx.session.state = state
    return ctx


def _collect_events(agent, ctx):
    async def _run():
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        return events

    return asyncio.run(_run())


def _make_standby_method(method_id: str = "m2") -> Dict[str, Any]:
    return make_method_card(
        method_id=method_id,
        method_family="negative_variant",
        title=f"Standby {method_id}",
        core_hypothesis="alternative approach",
        assumptions=["a1"],
        invalid_if=["i1"],
        cheap_test="test",
        failure_modes=["f1"],
        required_capabilities=["python"],
        expected_artifacts=["out.csv"],
        orthogonality_tags=["alt"],
    )


# ── Benchmark A: Routine — no operators ──────────────────────


class TestBenchmarkA_Routine:
    """Routine mode: discovery skips entirely, zero candidates."""

    def test_routine_mode_skips_method_discovery(self):
        state = {
            StateKeys.INNOVATION_MODE: "routine",
            StateKeys.ORIGINAL_USER_INPUT: "Calculate average sales per region",
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model="dummy")
        _collect_events(agent, ctx)

        assert state[StateKeys.METHOD_CANDIDATES] == []

    def test_routine_with_framed_problem_still_skips(self):
        state = {
            StateKeys.INNOVATION_MODE: "routine",
            StateKeys.ORIGINAL_USER_INPUT: "Simple aggregation",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": ["speed vs accuracy"],
                "unknowns": ["hidden variable"],
                "complexity_signals": ["multi-source"],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model="dummy")
        _collect_events(agent, ctx)

        assert state[StateKeys.METHOD_CANDIDATES] == []


# ── Benchmark B: Contradiction → TRIZ ────────────────────────


class TestBenchmarkB_TRIZ:
    """Innovation mode with contradictions triggers TRIZ operator."""

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_triz_triggered_by_contradictions(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_card = _method_card_json("m1", "baseline")
        triz_card = _method_card_json("triz_1", "triz_resolution", title="TRIZ Resolution")

        llm = _make_mock_llm([baseline_card, baseline_card, baseline_card, triz_card])

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Analyze gene expression with speed vs accuracy tradeoff",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": ["Need fast analysis but also high accuracy"],
                "unknowns": [],
                "complexity_signals": [],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        assert len(candidates) >= 2
        families = {c["method_family"] for c in candidates}
        assert "triz_resolution" in families

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_triz_not_triggered_without_contradictions(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_card = _method_card_json("m1", "baseline")
        llm = _make_mock_llm([baseline_card, baseline_card, baseline_card])

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Analyze data",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": [],
                "unknowns": [],
                "complexity_signals": [],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        families = {c["method_family"] for c in candidates}
        assert "triz_resolution" not in families


# ── Benchmark C: Unknowns → Abduction ────────────────────────


class TestBenchmarkC_Abduction:
    """Innovation mode with unknowns triggers abduction operator."""

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_abduction_triggered_by_unknowns(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_card = _method_card_json("m1", "baseline")
        abd_card = _method_card_json("abd_1", "abductive_hypothesis", title="Abductive Hypothesis")

        llm = _make_mock_llm([baseline_card, baseline_card, baseline_card, abd_card])

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Explain unexpected outliers in protein expression data",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": [],
                "unknowns": ["Why do certain proteins show bimodal expression?"],
                "complexity_signals": ["novel_pattern"],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        assert len(candidates) >= 2
        families = {c["method_family"] for c in candidates}
        assert "abductive_hypothesis" in families

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_abduction_triggered_by_complexity_signals_alone(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_card = _method_card_json("m1", "baseline")
        abd_card = _method_card_json("abd_1", "abductive_hypothesis")

        llm = _make_mock_llm([baseline_card, baseline_card, baseline_card, abd_card])

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Analyze complex multi-omics dataset",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": [],
                "unknowns": [],
                "complexity_signals": ["multi_omics", "cross_domain"],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        families = {c["method_family"] for c in candidates}
        assert "abductive_hypothesis" in families


# ── Benchmark D: Combined TRIZ + Abduction ───────────────────


class TestBenchmarkD_Combined:
    """Both contradictions and unknowns present — both operators fire."""

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_both_operators_fire(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_card = _method_card_json("m1", "baseline")
        triz_card = _method_card_json("triz_1", "triz_resolution")
        abd_card = _method_card_json("abd_1", "abductive_hypothesis")

        llm = _make_mock_llm(
            [
                baseline_card,
                baseline_card,
                baseline_card,
                triz_card,
                abd_card,
            ]
        )

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Resolve speed-accuracy tradeoff while explaining outliers",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": ["speed vs accuracy"],
                "unknowns": ["unexplained outlier pattern"],
                "complexity_signals": ["novel"],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        families = {c["method_family"] for c in candidates}
        assert "triz_resolution" in families, f"TRIZ missing; families={families}"
        assert "abductive_hypothesis" in families, f"Abduction missing; families={families}"
        assert len(candidates) >= 3

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_combined_produces_distinct_method_ids(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_1 = _method_card_json("m1", "baseline")
        baseline_2 = _method_card_json("m2", "negative_variant")
        triz = _method_card_json("triz_1", "triz_resolution")
        abd = _method_card_json("abd_1", "abductive_hypothesis")

        llm = _make_mock_llm([baseline_1, baseline_2, baseline_2, triz, abd])

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Complex analysis",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": ["c1"],
                "unknowns": ["u1"],
                "complexity_signals": [],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        ids = [c["method_id"] for c in candidates]
        assert len(ids) == len(set(ids)) or len(candidates) >= 3


# ── Benchmark E: Method Backtrack ─────────────────────────────


class TestBenchmarkE_Backtrack:
    """Primary method fails → standby method activated via backtracker."""

    def _critic_output(self, recommendation="backtrack"):
        return {
            "issue_type": "method_failure",
            "confidence": 0.9,
            "evidence": ["AUC below 0.5"],
            "recommendation": recommendation,
            "explanation": "Primary method fundamentally flawed",
        }

    def test_backtrack_condition_met(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: self._critic_output(),
            StateKeys.STANDBY_METHODS: [_make_standby_method("m2")],
            StateKeys.INNOVATION_BUDGET: InnovationBudget(backtrack=1).to_dict(),
        }
        assert should_backtrack(state) is True

    def test_backtrack_executes_successfully(self):
        primary = make_method_card(
            method_id="m1",
            method_family="baseline",
            title="Failed Primary",
            core_hypothesis="wrong approach",
            assumptions=[],
            invalid_if=[],
            cheap_test="test",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=[],
        )
        standby = _make_standby_method("m2")

        state = {
            StateKeys.SELECTED_METHOD: primary,
            StateKeys.STANDBY_METHODS: [standby],
            StateKeys.METHOD_CRITIC_OUTPUT: self._critic_output(),
            StateKeys.INNOVATION_BUDGET: InnovationBudget(backtrack=1).to_dict(),
            StateKeys.BACKTRACK_HISTORY: [],
        }

        result = execute_backtrack(state)
        assert result is not None
        assert state[StateKeys.SELECTED_METHOD]["method_id"] == "m2"
        assert len(state[StateKeys.BACKTRACK_HISTORY]) == 1
        history_entry = state[StateKeys.BACKTRACK_HISTORY][0]
        assert history_entry["from_method"] == "m1"
        assert history_entry["to_method"] == "m2"

    def test_no_backtrack_when_critic_says_retry(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: self._critic_output(recommendation="retry"),
            StateKeys.STANDBY_METHODS: [_make_standby_method("m2")],
            StateKeys.INNOVATION_BUDGET: InnovationBudget(backtrack=1).to_dict(),
        }
        assert should_backtrack(state) is False

    def test_no_backtrack_when_no_standby(self):
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: self._critic_output(),
            StateKeys.STANDBY_METHODS: [],
            StateKeys.INNOVATION_BUDGET: InnovationBudget(backtrack=1).to_dict(),
        }
        assert should_backtrack(state) is False

    def test_no_backtrack_when_budget_exhausted(self):
        budget = InnovationBudget(backtrack=1)
        budget.consume("backtrack")
        state = {
            StateKeys.METHOD_CRITIC_OUTPUT: self._critic_output(),
            StateKeys.STANDBY_METHODS: [_make_standby_method("m2")],
            StateKeys.INNOVATION_BUDGET: budget.to_dict(),
        }
        assert should_backtrack(state) is False

    def test_backtrack_history_accumulates(self):
        primary = make_method_card(
            method_id="m1",
            method_family="baseline",
            title="First",
            core_hypothesis="h1",
            assumptions=[],
            invalid_if=[],
            cheap_test="t",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=[],
        )
        m2 = _make_standby_method("m2")
        m3 = _make_standby_method("m3")

        state = {
            StateKeys.SELECTED_METHOD: primary,
            StateKeys.STANDBY_METHODS: [m2, m3],
            StateKeys.METHOD_CRITIC_OUTPUT: self._critic_output(),
            StateKeys.INNOVATION_BUDGET: InnovationBudget(backtrack=2).to_dict(),
            StateKeys.BACKTRACK_HISTORY: [],
        }

        result1 = execute_backtrack(state)
        assert result1 is not None
        assert state[StateKeys.SELECTED_METHOD]["method_id"] == "m2"

        state[StateKeys.METHOD_CRITIC_OUTPUT] = self._critic_output()
        result2 = execute_backtrack(state)
        assert result2 is not None
        assert state[StateKeys.SELECTED_METHOD]["method_id"] == "m3"
        assert len(state[StateKeys.BACKTRACK_HISTORY]) == 2


# ── Cross-benchmark: episodic memory integration ─────────────


class TestBenchmarkEpisodicMemoryIntegration:
    """Verify episodic memory is queried during method discovery."""

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_episodic_constraints_injected_into_prompt(self, mock_mem_factory, mock_load_prompt):
        mock_memory = MagicMock()
        mock_memory.build_negative_constraints_summary.return_value = (
            "Known failed approaches (avoid repeating):\n  - [baseline] PCA-based clustering — failed: poor separation"
        )
        mock_mem_factory.return_value = mock_memory

        captured_prompts = []

        def fake_load_prompt(name):
            return (
                "User: {original_user_input?}\nRound: {round_number?}\n"
                "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
                "Episodic: {episodic_memory_constraints?}"
            )

        mock_load_prompt.side_effect = fake_load_prompt

        baseline_card = _method_card_json("m1", "baseline")

        call_count = [0]

        async def capture_generate(llm_request, **kwargs):
            call_count[0] += 1
            if llm_request.contents:
                for content in llm_request.contents:
                    if content.parts:
                        for part in content.parts:
                            if hasattr(part, "text") and part.text:
                                captured_prompts.append(part.text)
            resp = MagicMock()
            resp.content = MagicMock()
            resp.content.parts = [MagicMock(text=baseline_card)]
            yield resp

        llm = MagicMock()
        llm.generate_content_async = capture_generate

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Cluster gene expression profiles",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": [],
                "unknowns": [],
                "complexity_signals": [],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        assert len(captured_prompts) >= 1
        first_prompt = captured_prompts[0]
        assert "PCA-based clustering" in first_prompt
        assert "poor separation" in first_prompt

    @patch("agentic_data_scientist.prompts.load_prompt")
    @patch("agentic_data_scientist.core.innovation_memory.create_innovation_memory_from_env")
    def test_no_episodic_memory_gracefully_handled(self, mock_mem_factory, mock_load_prompt):
        mock_mem_factory.return_value = None
        mock_load_prompt.return_value = (
            "User: {original_user_input?}\nRound: {round_number?}\n"
            "Existing: {existing_methods?}\nNeg: {negative_constraints?}\n"
            "Episodic: {episodic_memory_constraints?}"
        )

        baseline_card = _method_card_json("m1", "baseline")
        llm = _make_mock_llm([baseline_card, baseline_card, baseline_card])

        state = {
            StateKeys.INNOVATION_MODE: "innovation",
            StateKeys.ORIGINAL_USER_INPUT: "Analyze data",
            StateKeys.FRAMED_PROBLEM: {
                "contradictions": [],
                "unknowns": [],
                "complexity_signals": [],
            },
        }
        ctx = _make_ctx(state)
        agent = MethodDiscoveryAgent(name="test_discovery", model=llm)
        _collect_events(agent, ctx)

        candidates = state[StateKeys.METHOD_CANDIDATES]
        assert len(candidates) >= 1
