"""Unit tests for execution routing between skill and workflow executors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_data_scientist.agents.coding_backends import (
    create_execution_agent,
    stage_uses_workflow_execution,
)
from agentic_data_scientist.core.state_contracts import StateKeys


class _DummyExecutor:
    """Minimal executor agent for routing tests."""

    def __init__(self, *, name: str, marker: str):
        self.name = name
        self._marker = marker

    async def run_async(self, ctx):
        ctx.session.state[StateKeys.IMPLEMENTATION_SUMMARY] = self._marker
        if False:
            yield None


def test_stage_uses_workflow_execution_detection():
    """Routing hint detector should recognize workflow stages."""
    assert stage_uses_workflow_execution({"workflow_id": "bio.rnaseq"}) is True
    assert stage_uses_workflow_execution({"execution_mode": "workflow"}) is True
    assert stage_uses_workflow_execution({"execution_mode": "local_cli"}) is True
    assert stage_uses_workflow_execution({"title": "normal stage"}) is False
    assert stage_uses_workflow_execution(None) is False


@pytest.mark.asyncio
async def test_execution_agent_routes_to_skill_executor_by_default():
    """Without workflow hints, routed agent should invoke skill executor."""
    skill_executor = _DummyExecutor(name="skill", marker="skill-path")
    workflow_executor = _DummyExecutor(name="workflow", marker="workflow-path")
    execution_agent = create_execution_agent(
        name="execution_agent",
        description="test",
        skill_executor=skill_executor,
        workflow_executor=workflow_executor,
    )
    ctx = SimpleNamespace(session=SimpleNamespace(state={StateKeys.CURRENT_STAGE: {"title": "S1"}}, events=[]))

    async for _ in execution_agent._run_async_impl(ctx):
        pass

    assert ctx.session.state[StateKeys.IMPLEMENTATION_SUMMARY] == "skill-path"


@pytest.mark.asyncio
async def test_execution_agent_routes_to_workflow_executor_when_marked():
    """With workflow hints, routed agent should invoke workflow executor."""
    skill_executor = _DummyExecutor(name="skill", marker="skill-path")
    workflow_executor = _DummyExecutor(name="workflow", marker="workflow-path")
    execution_agent = create_execution_agent(
        name="execution_agent",
        description="test",
        skill_executor=skill_executor,
        workflow_executor=workflow_executor,
    )
    ctx = SimpleNamespace(
        session=SimpleNamespace(
            state={StateKeys.CURRENT_STAGE: {"title": "S2", "workflow_id": "bio.rnaseq"}},
            events=[],
        )
    )

    async for _ in execution_agent._run_async_impl(ctx):
        pass

    assert ctx.session.state[StateKeys.IMPLEMENTATION_SUMMARY] == "workflow-path"
