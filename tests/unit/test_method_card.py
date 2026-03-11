"""Unit tests for Method Card schema (core/method_card.py)."""

import pytest

from agentic_data_scientist.core.method_card import (
    make_method_card,
    method_card_summary,
    validate_method_card,
)


class TestMakeMethodCard:
    def test_basic_creation(self):
        card = make_method_card(
            method_id="m1",
            method_family="baseline",
            title="Standard DEG Analysis",
            core_hypothesis="Differential expression reveals treatment effects",
            assumptions=["Normal distribution", "Sufficient sample size"],
            invalid_if=["Sample size < 3"],
            cheap_test="Check sample count per group",
            failure_modes=["Low power"],
            required_capabilities=["python", "statistical_testing"],
            expected_artifacts=["deg_results.csv", "volcano_plot.png"],
            orthogonality_tags=["parametric", "frequentist"],
        )
        assert card["method_id"] == "m1"
        assert card["method_family"] == "baseline"
        assert card["status"] == "proposed"
        assert card["selection_score"] is None
        assert card["rejection_reason"] is None

    def test_strips_whitespace(self):
        card = make_method_card(
            method_id="  m2  ",
            method_family="  negative_variant  ",
            title="  Alt Method  ",
            core_hypothesis="  Some hypothesis  ",
            assumptions=["  a1  "],
            invalid_if=["  c1  "],
            cheap_test="  test  ",
            failure_modes=["  f1  "],
            required_capabilities=["  cap  "],
            expected_artifacts=["  art  "],
            orthogonality_tags=["  t1  "],
        )
        assert card["method_id"] == "m2"
        assert card["method_family"] == "negative_variant"
        assert card["assumptions"] == ["a1"]

    def test_invalid_status_defaults_to_proposed(self):
        card = make_method_card(
            method_id="m1",
            method_family="baseline",
            title="T",
            core_hypothesis="H",
            assumptions=[],
            invalid_if=[],
            cheap_test="C",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=[],
            status="bogus",
        )
        assert card["status"] == "proposed"

    def test_valid_statuses(self):
        for status in ("proposed", "selected", "standby", "failed", "succeeded"):
            card = make_method_card(
                method_id="m1",
                method_family="baseline",
                title="T",
                core_hypothesis="H",
                assumptions=[],
                invalid_if=[],
                cheap_test="C",
                failure_modes=[],
                required_capabilities=[],
                expected_artifacts=[],
                orthogonality_tags=[],
                status=status,
            )
            assert card["status"] == status

    def test_selection_score_float(self):
        card = make_method_card(
            method_id="m1",
            method_family="baseline",
            title="T",
            core_hypothesis="H",
            assumptions=[],
            invalid_if=[],
            cheap_test="C",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=[],
            selection_score=0.85,
        )
        assert card["selection_score"] == 0.85


class TestValidateMethodCard:
    def _valid_card(self):
        return make_method_card(
            method_id="m1",
            method_family="baseline",
            title="Valid Method",
            core_hypothesis="Some hypothesis",
            assumptions=["a"],
            invalid_if=["c"],
            cheap_test="test",
            failure_modes=["f"],
            required_capabilities=["python"],
            expected_artifacts=["out.csv"],
            orthogonality_tags=["tag1"],
        )

    def test_valid_card_no_errors(self):
        errors = validate_method_card(self._valid_card())
        assert errors == []

    def test_not_a_dict(self):
        errors = validate_method_card("not a dict")
        assert errors == ["method card must be a dict"]

    def test_missing_required_string(self):
        card = self._valid_card()
        card["title"] = ""
        errors = validate_method_card(card)
        assert any("title" in e for e in errors)

    def test_missing_required_list(self):
        card = self._valid_card()
        card["assumptions"] = "not a list"
        errors = validate_method_card(card)
        assert any("assumptions" in e for e in errors)

    def test_invalid_family(self):
        card = self._valid_card()
        card["method_family"] = "unknown_family"
        errors = validate_method_card(card)
        assert any("method_family" in e for e in errors)

    def test_invalid_status(self):
        card = self._valid_card()
        card["status"] = "bogus_status"
        errors = validate_method_card(card)
        assert any("status" in e for e in errors)


class TestMethodCardSummary:
    def test_summary_format(self):
        card = make_method_card(
            method_id="m1",
            method_family="baseline",
            title="DEG Analysis",
            core_hypothesis="Expression changes indicate treatment",
            assumptions=[],
            invalid_if=[],
            cheap_test="test",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=["parametric", "bulk_rna"],
        )
        summary = method_card_summary(card)
        assert "[m1]" in summary
        assert "DEG Analysis" in summary
        assert "parametric" in summary

    def test_summary_truncates_tags(self):
        card = make_method_card(
            method_id="m2",
            method_family="negative_variant",
            title="Alt",
            core_hypothesis="H",
            assumptions=[],
            invalid_if=[],
            cheap_test="C",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
        )
        summary = method_card_summary(card)
        assert "t6" not in summary
