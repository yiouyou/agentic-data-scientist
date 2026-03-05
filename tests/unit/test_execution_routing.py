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
        self._working_dir = None

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


@pytest.mark.asyncio
async def test_execution_agent_injects_stage_skill_and_subtask_hints(monkeypatch):
    """Skill executor route should inject Top-K skill hints and parallel subtask candidates."""
    from pathlib import Path

    root = Path(".tmp") / "unit_execution_routing" / "skill_hints"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    rnaseq_dir = source / "unitomics-rnaseq"
    rnaseq_dir.mkdir(parents=True, exist_ok=True)
    (rnaseq_dir / "SKILL.md").write_text(
        "# Unitomics RNA-seq\nunitomics123 differential expression and quality control workflow.",
        encoding="utf-8",
    )

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    monkeypatch.setenv("ADS_STAGE_SKILL_TOPK", "3")
    monkeypatch.setenv("ADS_STAGE_PARALLEL_SUBTASK_MAX", "3")
    monkeypatch.setenv("ADS_EMBEDDING_ENABLED", "false")

    skill_executor = _DummyExecutor(name="skill", marker="skill-path")
    skill_executor._working_dir = str(root)
    workflow_executor = _DummyExecutor(name="workflow", marker="workflow-path")
    execution_agent = create_execution_agent(
        name="execution_agent",
        description="test",
        skill_executor=skill_executor,
        workflow_executor=workflow_executor,
    )
    ctx = SimpleNamespace(
        session=SimpleNamespace(
                state={
                    StateKeys.ORIGINAL_USER_INPUT: "Analyze unitomics123 differential expression results",
                    StateKeys.CURRENT_STAGE: {
                        "title": "unitomics123 differential expression",
                        "description": "- load counts matrix\n- run DEG analysis\n- create QC plots",
                    },
                },
                events=[],
        )
    )

    async for _ in execution_agent._run_async_impl(ctx):
        pass

    stage = ctx.session.state[StateKeys.CURRENT_STAGE]
    recommendations = ctx.session.state[StateKeys.CURRENT_STAGE_SKILL_RECOMMENDATIONS]
    subtasks = ctx.session.state[StateKeys.CURRENT_STAGE_PARALLEL_SUBTASKS]

    assert ctx.session.state[StateKeys.IMPLEMENTATION_SUMMARY] == "skill-path"
    assert isinstance(recommendations, list) and recommendations
    recommended_names = {item["skill_name"] for item in recommendations}
    assert "unitomics-rnaseq" in recommended_names
    assert isinstance(stage.get("recommended_skills"), list) and stage["recommended_skills"]
    assert isinstance(subtasks, list) and len(subtasks) >= 2
