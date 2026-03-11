"""Tests for innovation_trigger detection functions."""

import pytest

from agentic_data_scientist.agents.adk.innovation_trigger import (
    compute_trigger_result,
    detect_criteria_stagnation,
    detect_excessive_retries,
    detect_mediocre_review,
    detect_verifier_warnings,
)


class TestDetectMediocreReview:
    def test_none_input(self):
        assert detect_mediocre_review(None) is False

    def test_empty_string(self):
        assert detect_mediocre_review("") is False

    def test_clean_approval(self):
        assert detect_mediocre_review("Implementation looks great, all tests pass.") is False

    def test_partially_keyword(self):
        assert detect_mediocre_review("The stage was partially completed.") is True

    def test_marginal_keyword(self):
        assert detect_mediocre_review("Results show marginal improvement.") is True

    def test_could_be_improved(self):
        assert detect_mediocre_review("The approach could be improved.") is True

    def test_with_reservations(self):
        assert detect_mediocre_review("Approved with reservations about performance.") is True

    def test_minor_issues(self):
        assert detect_mediocre_review("There are minor issues in the output.") is True

    def test_barely_keyword(self):
        assert detect_mediocre_review("This barely meets the requirement.") is True

    def test_caveat_keyword(self):
        assert detect_mediocre_review("One caveat: the error rate is high.") is True

    def test_case_insensitive(self):
        assert detect_mediocre_review("PARTIALLY done, BORDERLINE acceptable.") is True

    def test_acceptable_but(self):
        assert detect_mediocre_review("The result is acceptable but needs polishing.") is True


class TestDetectExcessiveRetries:
    def test_zero_attempts(self):
        assert detect_excessive_retries(0) is False

    def test_at_threshold(self):
        assert detect_excessive_retries(2) is False

    def test_above_threshold(self):
        assert detect_excessive_retries(3) is True

    def test_custom_threshold(self):
        assert detect_excessive_retries(4, threshold=3) is True
        assert detect_excessive_retries(3, threshold=3) is False

    def test_one_attempt(self):
        assert detect_excessive_retries(1) is False


class TestDetectCriteriaStagnation:
    def test_increasing_history(self):
        assert detect_criteria_stagnation([1, 2, 3]) is False

    def test_flat_history(self):
        assert detect_criteria_stagnation([2, 2, 2]) is True

    def test_decreasing_history(self):
        assert detect_criteria_stagnation([3, 2, 1]) is True

    def test_too_short_history(self):
        assert detect_criteria_stagnation([2, 2]) is False

    def test_empty_history(self):
        assert detect_criteria_stagnation([]) is False

    def test_single_entry(self):
        assert detect_criteria_stagnation([5]) is False

    def test_stagnation_at_zero(self):
        assert detect_criteria_stagnation([0, 0, 0]) is True

    def test_recent_increase_not_stagnant(self):
        assert detect_criteria_stagnation([1, 1, 2]) is False

    def test_custom_window_3(self):
        assert detect_criteria_stagnation([1, 2, 2, 2], stagnation_window=3) is True
        assert detect_criteria_stagnation([1, 2, 2, 3], stagnation_window=3) is False

    def test_longer_history_only_checks_window(self):
        assert detect_criteria_stagnation([0, 1, 2, 3, 3, 3]) is True


class TestDetectVerifierWarnings:
    def test_none_input(self):
        assert detect_verifier_warnings(None) is False

    def test_empty_string(self):
        assert detect_verifier_warnings("") is False

    def test_pass_verdict(self):
        assert detect_verifier_warnings("Verdict: pass") is False

    def test_warn_verdict_plain(self):
        assert detect_verifier_warnings("Verdict: warn") is True

    def test_warn_verdict_json(self):
        assert detect_verifier_warnings('{"verdict": "warn", "details": "..."}') is True

    def test_case_insensitive(self):
        assert detect_verifier_warnings("VERDICT: WARN") is True


class TestComputeTriggerResult:
    def test_no_signals(self):
        result = compute_trigger_result()
        assert result["triggered"] is False
        assert result["signals"] == []
        assert result["strength"] == 0.0
        assert result["recommended_action"] == "none"

    def test_mediocre_review_only(self):
        result = compute_trigger_result(review_text="Partially done.")
        assert result["triggered"] is True
        assert "mediocre_review" in result["signals"]
        assert result["strength"] == 0.25
        assert result["recommended_action"] == "escalate_review"

    def test_excessive_retries_only(self):
        result = compute_trigger_result(attempts=3)
        assert result["triggered"] is True
        assert "excessive_retries" in result["signals"]
        assert result["strength"] == 0.35
        assert result["recommended_action"] == "escalate_review"

    def test_verifier_warnings_only(self):
        result = compute_trigger_result(verifier_summary="Verdict: warn")
        assert result["triggered"] is True
        assert "verifier_warnings" in result["signals"]
        assert result["strength"] == 0.15
        assert result["recommended_action"] == "escalate_review"

    def test_stagnation_only(self):
        result = compute_trigger_result(criteria_met_history=[2, 2, 2])
        assert result["triggered"] is True
        assert "criteria_stagnation" in result["signals"]
        assert result["strength"] == 0.25

    def test_multiple_signals_escalate_to_method_switch(self):
        result = compute_trigger_result(
            review_text="Barely acceptable.",
            attempts=3,
        )
        assert result["triggered"] is True
        assert set(result["signals"]) == {"mediocre_review", "excessive_retries"}
        assert result["strength"] == 0.6
        assert result["recommended_action"] == "consider_method_switch"

    def test_all_signals(self):
        result = compute_trigger_result(
            review_text="Marginal quality.",
            attempts=4,
            criteria_met_history=[1, 1, 1],
            verifier_summary="Verdict: warn",
        )
        assert result["triggered"] is True
        assert len(result["signals"]) == 4
        assert result["strength"] == 1.0
        assert result["recommended_action"] == "consider_method_switch"

    def test_strength_boundary_at_0_5(self):
        result = compute_trigger_result(
            review_text="Partially done.",
            criteria_met_history=[3, 3, 3],
        )
        assert result["strength"] == 0.5
        assert result["recommended_action"] == "consider_method_switch"

    def test_no_stagnation_with_none_history(self):
        result = compute_trigger_result(criteria_met_history=None)
        assert "criteria_stagnation" not in result["signals"]

    def test_custom_retry_threshold(self):
        result = compute_trigger_result(attempts=2, retry_threshold=1)
        assert "excessive_retries" in result["signals"]

    def test_custom_stagnation_window(self):
        result = compute_trigger_result(
            criteria_met_history=[1, 2, 2, 2],
            stagnation_window=3,
        )
        assert "criteria_stagnation" in result["signals"]
