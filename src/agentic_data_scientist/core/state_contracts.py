"""Shared workflow state contracts and helpers.

This module centralizes state-key names and stage status semantics used across
API entrypoints and ADK workflow agents. Keeping these contracts in one place
reduces drift and makes state transitions easier to reason about and test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class StateKeys:
    """Canonical state keys used across the workflow."""

    ORIGINAL_USER_INPUT = "original_user_input"
    LATEST_USER_INPUT = "latest_user_input"
    RENDERED_PROMPT = "rendered_prompt"
    IMPLEMENTATION_TASK = "implementation_task"

    HIGH_LEVEL_STAGES = "high_level_stages"
    HIGH_LEVEL_SUCCESS_CRITERIA = "high_level_success_criteria"
    CURRENT_STAGE = "current_stage"
    CURRENT_STAGE_INDEX = "current_stage_index"
    CURRENT_STAGE_SKILL_RECOMMENDATIONS = "current_stage_skill_recommendations"
    CURRENT_STAGE_PARALLEL_SUBTASKS = "current_stage_parallel_subtasks"
    STAGE_IMPLEMENTATIONS = "stage_implementations"

    HIGH_LEVEL_PLAN = "high_level_plan"
    PLANNER_HISTORY_ADVICE = "planner_history_advice"
    PLANNER_HISTORY_SIGNALS = "planner_history_signals"
    PLAN_CANDIDATES = "plan_candidates"
    PLAN_SELECTION_TRACE = "plan_selection_trace"
    PLANNER_SKILL_ADVICE = "planner_skill_advice"
    PLAN_REVIEW_FEEDBACK = "plan_review_feedback"
    PARSED_PLAN_OUTPUT = "parsed_plan_output"
    CRITERIA_CHECKER_OUTPUT = "criteria_checker_output"
    STAGE_REFLECTOR_OUTPUT = "stage_reflector_output"

    IMPLEMENTATION_SUMMARY = "implementation_summary"
    REVIEW_FEEDBACK = "review_feedback"
    PLAN_REVIEW_CONFIRMATION_DECISION = "plan_review_confirmation_decision"
    IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION = "implementation_review_confirmation_decision"
    USER_MESSAGE = "user_message"


class StageStatus:
    """Allowed stage lifecycle statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RETRYING = "retrying"
    FAILED = "failed"
    APPROVED = "approved"

    ALL = {PENDING, IN_PROGRESS, RETRYING, FAILED, APPROVED}


def set_stage_status(stage: Dict[str, Any], status: str) -> None:
    """Set stage status with strict validation."""
    if status not in StageStatus.ALL:
        raise ValueError(f"Invalid stage status: {status}")
    stage["status"] = status


def ensure_stage_status(stage: Dict[str, Any]) -> str:
    """Ensure a stage has a valid status and return it."""
    status = stage.get("status")
    if status in StageStatus.ALL:
        return status

    default_status = StageStatus.APPROVED if stage.get("completed", False) else StageStatus.PENDING
    stage["status"] = default_status
    return default_status


def make_stage_record(
    *,
    index: int,
    title: str,
    description: str,
    completed: bool = False,
    implementation_result: Optional[str] = None,
    stage_id: Optional[str] = None,
    depends_on: Optional[List[str]] = None,
    inputs_required: Optional[List[str]] = None,
    outputs_produced: Optional[List[str]] = None,
    evidence_refs: Optional[List[str]] = None,
    subtasks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a normalized stage record."""
    status = StageStatus.APPROVED if completed else StageStatus.PENDING
    resolved_stage_id = stage_id.strip() if isinstance(stage_id, str) and stage_id.strip() else f"s{index + 1}"
    return {
        "index": index,
        "stage_id": resolved_stage_id,
        "title": title,
        "description": description,
        "completed": completed,
        "status": status,
        "implementation_result": implementation_result,
        "depends_on": list(depends_on or []),
        "inputs_required": list(inputs_required or []),
        "outputs_produced": list(outputs_produced or []),
        "evidence_refs": list(evidence_refs or []),
        "subtasks": list(subtasks or []),
    }


def make_success_criterion_record(*, index: int, criteria: str) -> Dict[str, Any]:
    """Create a normalized success criterion record."""
    return {
        "index": index,
        "criteria": criteria,
        "met": False,
        "evidence": None,
    }


def build_initial_state_delta(
    *,
    original_message: str,
    rendered_prompt: str,
    agent_type: str,
) -> Dict[str, str]:
    """Build the initial state delta passed to the ADK runner."""
    state = {
        StateKeys.ORIGINAL_USER_INPUT: original_message,
        StateKeys.LATEST_USER_INPUT: original_message,
        StateKeys.RENDERED_PROMPT: rendered_prompt,
    }
    if agent_type == "claude_code":
        state[StateKeys.IMPLEMENTATION_TASK] = rendered_prompt
    return state


def review_confirmation_decision_key(prompt_name: str) -> str:
    """Return the canonical decision key for a review-confirmation prompt name."""
    if prompt_name == "plan_review_confirmation":
        return StateKeys.PLAN_REVIEW_CONFIRMATION_DECISION
    if prompt_name == "implementation_review_confirmation":
        return StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION
    return f"{prompt_name}_decision"
