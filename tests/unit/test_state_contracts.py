"""Unit tests for shared workflow state contracts."""

import pytest

from agentic_data_scientist.core.state_contracts import (
    StageStatus,
    StateKeys,
    build_initial_state_delta,
    ensure_stage_status,
    make_stage_record,
    make_success_criterion_record,
    review_confirmation_decision_key,
    set_stage_status,
)


def test_build_initial_state_delta_for_adk():
    """ADK initial state should not include implementation_task."""
    state = build_initial_state_delta(
        original_message="raw",
        rendered_prompt="rendered",
        agent_type="adk",
    )
    assert state[StateKeys.ORIGINAL_USER_INPUT] == "raw"
    assert state[StateKeys.LATEST_USER_INPUT] == "raw"
    assert state[StateKeys.RENDERED_PROMPT] == "rendered"
    assert StateKeys.IMPLEMENTATION_TASK not in state


def test_build_initial_state_delta_for_claude_code():
    """Claude code initial state should include implementation_task."""
    state = build_initial_state_delta(
        original_message="raw",
        rendered_prompt="rendered",
        agent_type="claude_code",
    )
    assert state[StateKeys.IMPLEMENTATION_TASK] == "rendered"


def test_stage_status_helpers():
    """Status helpers should normalize and validate stage status values."""
    stage = {"completed": False}
    status = ensure_stage_status(stage)
    assert status == StageStatus.PENDING
    assert stage["status"] == StageStatus.PENDING

    set_stage_status(stage, StageStatus.IN_PROGRESS)
    assert stage["status"] == StageStatus.IN_PROGRESS

    with pytest.raises(ValueError):
        set_stage_status(stage, "unknown")


def test_stage_and_criterion_record_builders():
    """Record builders should produce normalized workflow dictionaries."""
    stage = make_stage_record(index=1, title="S", description="D")
    assert stage["index"] == 1
    assert stage["stage_id"] == "s2"
    assert stage["status"] == StageStatus.PENDING
    assert stage["completed"] is False
    assert stage["implementation_result"] is None
    assert stage["depends_on"] == []
    assert stage["inputs_required"] == []
    assert stage["outputs_produced"] == []
    assert stage["evidence_refs"] == []

    criterion = make_success_criterion_record(index=0, criteria="C")
    assert criterion["index"] == 0
    assert criterion["criteria"] == "C"
    assert criterion["met"] is False
    assert criterion["evidence"] is None


def test_review_confirmation_decision_key_mapping():
    """Known review confirmation prompts should map to canonical state keys."""
    assert review_confirmation_decision_key("plan_review_confirmation") == StateKeys.PLAN_REVIEW_CONFIRMATION_DECISION
    assert (
        review_confirmation_decision_key("implementation_review_confirmation")
        == StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION
    )


def test_review_confirmation_decision_key_fallback():
    """Unknown prompt names should keep backward-compatible key generation."""
    assert review_confirmation_decision_key("custom_review") == "custom_review_decision"
