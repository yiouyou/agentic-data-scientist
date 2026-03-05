"""Reliability-focused tests for ADK callbacks and stage orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_data_scientist.agents.adk.agent import (
    criteria_checker_callback,
    plan_parser_callback,
    stage_reflector_callback,
)
from agentic_data_scientist.agents.adk.stage_orchestrator import StageOrchestratorAgent
from agentic_data_scientist.core.state_contracts import StageStatus, StateKeys


def _make_callback_context(state: dict):
    """Create a lightweight callback context for callback unit tests."""
    session = SimpleNamespace(state=state, events=[])
    invocation_context = SimpleNamespace(session=session)
    return SimpleNamespace(_invocation_context=invocation_context)


class TestCallbackValidation:
    """Validate strict callback acceptance/rejection behavior."""

    def test_plan_parser_rejects_partial_invalid_output(self):
        """Plan parser should reject the full payload when any stage entry is invalid."""
        state = {
            "parsed_plan_output": {
                "stages": [
                    {"title": "S1", "description": "valid"},
                    {"title": "S2"},  # invalid: missing description
                ],
                "success_criteria": [{"criteria": "C1"}],
            }
        }
        callback_context = _make_callback_context(state)
        plan_parser_callback(callback_context)

        assert "high_level_stages" not in state
        assert "high_level_success_criteria" not in state

    def test_plan_parser_sets_pending_status_for_stages(self):
        """Plan parser should initialize stage status as pending."""
        state = {
            "parsed_plan_output": {
                "stages": [{"title": "S1", "description": "desc"}],
                "success_criteria": [{"criteria": "C1"}],
            }
        }
        callback_context = _make_callback_context(state)
        plan_parser_callback(callback_context)

        assert state[StateKeys.HIGH_LEVEL_STAGES][0]["status"] == StageStatus.PENDING
        assert state["high_level_stages"][0]["completed"] is False

    def test_plan_parser_extracts_workflow_stage_hints(self):
        """Plan parser should lift workflow routing hints into normalized stage records."""
        state = {
            "parsed_plan_output": {
                "stages": [
                    {
                        "title": "Run fixed pipeline",
                        "description": (
                            "execution_mode: workflow\n"
                            "workflow_id: bio.rnaseq.nfcore_deseq2\n"
                            "workflow_version: 1.0.0\n"
                            "inputs_required: [\"artifacts/s1/counts.tsv\"]\n"
                            "outputs_produced: [\"artifacts/s2/de.tsv\"]\n"
                            "workflow_inputs: {\"sample_sheet\": \"user_data/samples.csv\"}\n"
                            "workflow_params: {\"fdr_threshold\": 0.05}\n"
                        ),
                    }
                ],
                "success_criteria": [{"criteria": "C1"}],
            }
        }
        callback_context = _make_callback_context(state)
        plan_parser_callback(callback_context)

        stage = state[StateKeys.HIGH_LEVEL_STAGES][0]
        assert stage["stage_id"] == "s1"
        assert stage["execution_mode"] == "workflow"
        assert stage["workflow_id"] == "bio.rnaseq.nfcore_deseq2"
        assert stage["workflow_version"] == "1.0.0"
        assert stage["depends_on"] == []
        assert stage["inputs_required"] == ["artifacts/s1/counts.tsv"]
        assert stage["outputs_produced"] == ["artifacts/s2/de.tsv"]
        assert stage["workflow_inputs"]["sample_sheet"] == "user_data/samples.csv"
        assert stage["workflow_params"]["fdr_threshold"] == 0.05

    def test_plan_parser_rejects_invalid_dependency_reference(self):
        """Plan parser should reject plan when stage dependency references unknown stage."""
        state = {
            "parsed_plan_output": {
                "stages": [
                    {"title": "S1", "description": "desc"},
                    {"title": "S2", "description": "desc", "depends_on": ["missing"]},
                ],
                "success_criteria": [{"criteria": "C1"}],
            }
        }
        callback_context = _make_callback_context(state)
        plan_parser_callback(callback_context)

        assert StateKeys.HIGH_LEVEL_STAGES not in state
        assert StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA not in state

    def test_criteria_checker_rejects_partial_coverage(self):
        """Criteria checker must reject updates that do not cover all criteria exactly once."""
        state = {
            StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA: [
                {"index": 0, "criteria": "C0", "met": False, "evidence": None},
                {"index": 1, "criteria": "C1", "met": False, "evidence": None},
            ],
            StateKeys.CRITERIA_CHECKER_OUTPUT: {
                "criteria_updates": [
                    {"index": 0, "met": True, "evidence": "ok"},
                ]
            },
        }
        callback_context = _make_callback_context(state)
        criteria_checker_callback(callback_context)

        assert state[StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA][0]["met"] is False
        assert state[StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA][1]["met"] is False

    def test_stage_reflector_rejects_invalid_modifications(self):
        """Stage reflector should reject a payload that tries to modify a completed stage."""
        stages = [
            {"index": 0, "title": "Done", "description": "x", "completed": True},
            {"index": 1, "title": "Todo", "description": "y", "completed": False},
        ]
        original_snapshot = [dict(s) for s in stages]
        state = {
            StateKeys.HIGH_LEVEL_STAGES: stages,
            StateKeys.STAGE_REFLECTOR_OUTPUT: {
                "stage_modifications": [{"index": 0, "new_description": "should reject"}],
                "new_stages": [{"title": "Extra", "description": "new"}],
            },
        }
        callback_context = _make_callback_context(state)
        stage_reflector_callback(callback_context)

        assert state[StateKeys.HIGH_LEVEL_STAGES] == original_snapshot


class _DummyAgent:
    """Minimal async agent stub with side effects in session state."""

    def __init__(self, role: str):
        self.role = role
        self.calls = 0

    async def run_async(self, ctx):
        self.calls += 1
        state = ctx.session.state

        if self.role == "implementation":
            # First attempt fails review, second attempt passes.
            if self.calls == 1:
                state[StateKeys.IMPLEMENTATION_SUMMARY] = "attempt-1"
                state[StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION] = {
                    "exit": False,
                    "reason": "blocking issue",
                }
            else:
                state[StateKeys.IMPLEMENTATION_SUMMARY] = "attempt-2"
                state[StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION] = {
                    "exit": True,
                    "reason": "approved",
                }

        elif self.role == "criteria_checker":
            # Mark all criteria met when this agent is called.
            for criterion in state.get(StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA, []):
                criterion["met"] = True
                criterion["evidence"] = "validated"

        # Keep as async generator shape
        if False:
            yield None


@pytest.mark.asyncio
async def test_stage_orchestrator_requires_review_approval_before_completion():
    """A stage should only become completed when implementation review decision has exit=true."""
    implementation_agent = _DummyAgent("implementation")
    criteria_checker_agent = _DummyAgent("criteria_checker")
    reflector_agent = _DummyAgent("reflector")

    orchestrator = StageOrchestratorAgent(
        implementation_loop=implementation_agent,
        criteria_checker=criteria_checker_agent,
        stage_reflector=reflector_agent,
    )

    session = SimpleNamespace(
        state={
            StateKeys.HIGH_LEVEL_STAGES: [
                {
                    "index": 0,
                    "title": "Only Stage",
                    "description": "desc",
                    "completed": False,
                    "status": StageStatus.PENDING,
                }
            ],
            StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA: [
                {"index": 0, "criteria": "C0", "met": False, "evidence": None}
            ],
            StateKeys.STAGE_IMPLEMENTATIONS: [],
        },
        events=[],
    )
    ctx = SimpleNamespace(session=session, session_service=None)

    # Drain orchestrator events
    async for _ in orchestrator._run_async_impl(ctx):
        pass

    stage = session.state[StateKeys.HIGH_LEVEL_STAGES][0]
    attempts = session.state[StateKeys.STAGE_IMPLEMENTATIONS]
    assert implementation_agent.calls >= 2
    assert criteria_checker_agent.calls == 1
    assert stage["completed"] is True
    assert stage["status"] == StageStatus.APPROVED
    assert stage["attempts"] >= 2
    assert len(attempts) >= 2
    for attempt in attempts:
        assert attempt.get("started_at")
        assert attempt.get("finished_at")
        assert float(attempt.get("duration_seconds", 0.0)) >= 0.0
        assert attempt.get("status") in {
            StageStatus.RETRYING,
            StageStatus.APPROVED,
            StageStatus.FAILED,
        }


@pytest.mark.asyncio
async def test_stage_orchestrator_keeps_stage_metadata_in_current_stage():
    """Current stage should preserve workflow metadata for downstream execution routing."""

    class _MetadataCapturingImplementationAgent:
        async def run_async(self, ctx):
            current = ctx.session.state[StateKeys.CURRENT_STAGE]
            assert current["workflow_id"] == "bio.rnaseq.nfcore_deseq2"
            assert current["execution_mode"] == "workflow"
            ctx.session.state[StateKeys.IMPLEMENTATION_SUMMARY] = "ok"
            ctx.session.state[StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION] = {
                "exit": True,
                "reason": "approved",
            }
            if False:
                yield None

    class _CriteriaCheckerAgent:
        async def run_async(self, ctx):
            for criterion in ctx.session.state.get(StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA, []):
                criterion["met"] = True
                criterion["evidence"] = "ok"
            if False:
                yield None

    class _NoopReflectorAgent:
        async def run_async(self, ctx):
            if False:
                yield None

    orchestrator = StageOrchestratorAgent(
        implementation_loop=_MetadataCapturingImplementationAgent(),
        criteria_checker=_CriteriaCheckerAgent(),
        stage_reflector=_NoopReflectorAgent(),
    )

    session = SimpleNamespace(
        state={
            StateKeys.HIGH_LEVEL_STAGES: [
                {
                    "index": 0,
                    "title": "Run workflow",
                    "description": "desc",
                    "completed": False,
                    "status": StageStatus.PENDING,
                    "execution_mode": "workflow",
                    "workflow_id": "bio.rnaseq.nfcore_deseq2",
                }
            ],
            StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA: [
                {"index": 0, "criteria": "done", "met": False, "evidence": None}
            ],
            StateKeys.STAGE_IMPLEMENTATIONS: [],
        },
        events=[],
    )
    ctx = SimpleNamespace(session=session, session_service=None)

    async for _ in orchestrator._run_async_impl(ctx):
        pass

    stage = session.state[StateKeys.HIGH_LEVEL_STAGES][0]
    assert stage["completed"] is True
    assert stage["status"] == StageStatus.APPROVED
