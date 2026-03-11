"""Unit tests for method_selector (agents/adk/method_selector.py)."""

import pytest

from agentic_data_scientist.agents.adk.method_selector import (
    _jaccard_similarity,
    _parse_selector_response,
    score_methods_programmatic,
)
from agentic_data_scientist.core.method_card import make_method_card


class TestJaccardSimilarity:
    def test_identical_lists(self):
        assert _jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint_lists(self):
        assert _jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity(["a", "b", "c"], ["b", "c", "d"])
        assert 0.4 < sim < 0.6

    def test_empty_lists(self):
        assert _jaccard_similarity([], []) == 0.0

    def test_one_empty(self):
        assert _jaccard_similarity(["a"], []) == 0.0

    def test_case_insensitive(self):
        assert _jaccard_similarity(["ABC"], ["abc"]) == 1.0

    def test_strips_whitespace(self):
        assert _jaccard_similarity(["  a  "], ["a"]) == 1.0


class TestParseSelecterResponse:
    def test_plain_json(self):
        result = _parse_selector_response('{"selected_method_id": "m1"}')
        assert result == {"selected_method_id": "m1"}

    def test_json_in_code_block(self):
        text = '```json\n{"selected_method_id": "m1"}\n```'
        result = _parse_selector_response(text)
        assert result["selected_method_id"] == "m1"

    def test_json_with_surrounding_text(self):
        text = 'Here is my response:\n{"selected_method_id": "m2"}\nDone.'
        result = _parse_selector_response(text)
        assert result["selected_method_id"] == "m2"

    def test_invalid_json(self):
        result = _parse_selector_response("not json at all")
        assert result is None

    def test_empty_string(self):
        result = _parse_selector_response("")
        assert result is None


def _make_baseline_card():
    return make_method_card(
        method_id="m1",
        method_family="baseline",
        title="Standard Analysis",
        core_hypothesis="Standard approach works",
        assumptions=["a1"],
        invalid_if=["c1"],
        cheap_test="Check data exists and has expected columns",
        failure_modes=["f1"],
        required_capabilities=["python", "pandas", "visualization"],
        expected_artifacts=["results.csv"],
        orthogonality_tags=["parametric", "frequentist", "supervised"],
    )


def _make_alternative_card():
    return make_method_card(
        method_id="m2",
        method_family="negative_variant",
        title="Alternative Analysis",
        core_hypothesis="Non-parametric approach may be better",
        assumptions=["a2"],
        invalid_if=["c2"],
        cheap_test="Test normality assumption — if violated, this method is preferred",
        failure_modes=["f2"],
        required_capabilities=["python", "scipy", "machine_learning"],
        expected_artifacts=["alt_results.csv"],
        orthogonality_tags=["nonparametric", "bayesian", "unsupervised"],
    )


class TestScoreMethodsProgrammatic:
    def test_single_candidate(self):
        cards = [_make_baseline_card()]
        results = score_methods_programmatic(cards)
        assert len(results) == 1
        assert results[0]["method_id"] == "m1"
        assert 0.0 < results[0]["total_score"] < 1.0

    def test_two_candidates_baseline_vs_alt(self):
        cards = [_make_baseline_card(), _make_alternative_card()]
        results = score_methods_programmatic(cards)
        assert len(results) == 2
        assert results[0]["total_score"] >= results[1]["total_score"]

    def test_baseline_gets_baseline_bonus(self):
        cards = [_make_baseline_card()]
        results = score_methods_programmatic(cards)
        assert results[0]["scores"]["baseline_bonus"] == 0.70

    def test_alternative_gets_no_baseline_bonus(self):
        cards = [_make_alternative_card()]
        results = score_methods_programmatic(cards)
        assert results[0]["scores"]["baseline_bonus"] == 0.0

    def test_alternative_gets_higher_novelty(self):
        cards = [_make_baseline_card(), _make_alternative_card()]
        results = score_methods_programmatic(cards)
        baseline_result = next(r for r in results if r["method_id"] == "m1")
        alt_result = next(r for r in results if r["method_id"] == "m2")
        assert alt_result["scores"]["novelty"] > baseline_result["scores"]["novelty"]

    def test_orthogonality_high_for_disjoint_tags(self):
        cards = [_make_baseline_card(), _make_alternative_card()]
        results = score_methods_programmatic(cards)
        for r in results:
            assert r["scores"]["orthogonality"] > 0.5

    def test_similarity_penalty_for_identical_tags(self):
        card1 = _make_baseline_card()
        card2 = _make_alternative_card()
        card2["orthogonality_tags"] = card1["orthogonality_tags"].copy()
        results = score_methods_programmatic([card1, card2])
        has_penalty = any(r["similarity_penalty"] > 0 for r in results)
        assert has_penalty

    def test_results_sorted_descending(self):
        cards = [_make_baseline_card(), _make_alternative_card()]
        results = score_methods_programmatic(cards)
        scores = [r["total_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_capability_coverage_common_tools(self):
        cards = [_make_baseline_card()]
        results = score_methods_programmatic(cards)
        assert results[0]["scores"]["capability_coverage"] > 0.5

    def test_capability_coverage_exotic_tools(self):
        card = _make_baseline_card()
        card["required_capabilities"] = ["quantum_computing", "dna_synthesis"]
        results = score_methods_programmatic([card])
        assert results[0]["scores"]["capability_coverage"] == 0.0

    def test_empty_candidates(self):
        results = score_methods_programmatic([])
        assert results == []


class TestStateContractsPhase1Extensions:
    def test_stage_record_has_method_fields(self):
        from agentic_data_scientist.core.state_contracts import make_stage_record

        record = make_stage_record(
            index=0,
            title="Stage 1",
            description="Test stage",
            source_method_id="m1",
            method_family="baseline",
        )
        assert record["source_method_id"] == "m1"
        assert record["method_family"] == "baseline"

    def test_stage_record_method_fields_default_none(self):
        from agentic_data_scientist.core.state_contracts import make_stage_record

        record = make_stage_record(index=0, title="Stage 1", description="Test")
        assert record["source_method_id"] is None
        assert record["method_family"] is None

    def test_state_keys_phase1(self):
        from agentic_data_scientist.core.state_contracts import StateKeys

        assert hasattr(StateKeys, "METHOD_CANDIDATES")
        assert hasattr(StateKeys, "SELECTED_METHOD")
        assert hasattr(StateKeys, "STANDBY_METHODS")
        assert hasattr(StateKeys, "METHOD_SELECTION_TRACE")
        assert hasattr(StateKeys, "INNOVATION_MODE")
        assert hasattr(StateKeys, "INNOVATION_BUDGET")

    def test_build_initial_state_includes_innovation_mode(self):
        import os
        from agentic_data_scientist.core.state_contracts import (
            StateKeys,
            build_initial_state_delta,
        )

        os.environ["ADS_INNOVATION_MODE"] = "hybrid"
        try:
            state = build_initial_state_delta(
                original_message="test",
                rendered_prompt="test",
                agent_type="adk",
            )
            assert state[StateKeys.INNOVATION_MODE] == "hybrid"
        finally:
            os.environ.pop("ADS_INNOVATION_MODE", None)

    def test_build_initial_state_defaults_to_auto(self):
        import os
        from agentic_data_scientist.core.state_contracts import (
            StateKeys,
            build_initial_state_delta,
        )

        os.environ.pop("ADS_INNOVATION_MODE", None)
        state = build_initial_state_delta(
            original_message="test",
            rendered_prompt="test",
            agent_type="adk",
        )
        assert state.get(StateKeys.INNOVATION_MODE) == "auto"
