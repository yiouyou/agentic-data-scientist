"""Integration tests for ADK workflow."""

import shutil
import uuid
from pathlib import Path

import pytest


def _new_case_dir(prefix: str) -> Path:
    root = Path(".tmp") / "integration_cases"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{prefix}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


@pytest.mark.integration
class TestADKWorkflow:
    """Test full ADK workflow integration."""

    def test_create_agent(self):
        """Test creating an ADK agent with local tools."""
        from agentic_data_scientist.agents.adk import create_agent

        case_dir = _new_case_dir("create_agent")
        try:
            agent = create_agent(working_dir=str(case_dir))
            assert agent is not None
            assert agent.name == "agentic_data_scientist_workflow"
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_agent_has_sub_agents(self):
        """Test that created agent has proper sub-agents."""
        from agentic_data_scientist.agents.adk import create_agent

        case_dir = _new_case_dir("sub_agents")
        try:
            agent = create_agent(working_dir=str(case_dir))
            # SequentialAgent has sub_agents
            assert hasattr(agent, 'sub_agents')
            assert len(agent.sub_agents) == 5  # planning_loop, selector, parser, orchestrator, summary
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_agent_plan_only_mode(self, monkeypatch):
        """ADS_PLAN_ONLY should construct planning-only root workflow."""
        from agentic_data_scientist.agents.adk import create_agent

        case_dir = _new_case_dir("plan_only")
        monkeypatch.setenv("ADS_PLAN_ONLY", "true")
        try:
            agent = create_agent(working_dir=str(case_dir))
            assert hasattr(agent, "sub_agents")
            assert [sub_agent.name for sub_agent in agent.sub_agents] == [
                "high_level_planning_loop",
                "plan_candidate_selector",
                "high_level_plan_parser",
            ]
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_agent_with_tools_integration(self):
        """Test agent creation with local tools integration."""
        from agentic_data_scientist.agents.adk import create_agent

        case_dir = _new_case_dir("tools")
        try:
            agent = create_agent(working_dir=str(case_dir))

            # Verify agent was created successfully
            assert agent is not None
            assert hasattr(agent, 'sub_agents')
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)


@pytest.mark.integration
class TestImplementationLoop:
    """Test implementation loop integration."""

    def test_make_implementation_agents(self):
        """Test creating implementation agents."""
        from agentic_data_scientist.agents.adk.implementation_loop import make_implementation_agents

        case_dir = _new_case_dir("impl_agents")
        try:
            coding_agent, review_agent, review_confirmation = make_implementation_agents(str(case_dir), [])

            assert coding_agent is not None
            assert review_agent is not None
            assert review_confirmation is not None
            assert coding_agent.name == "execution_agent"
            assert review_agent.name == "review_agent"
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_coding_agent_defaults_to_claude_code(self):
        """Default coding backend should remain ClaudeCodeAgent."""
        from agentic_data_scientist.agents.adk.implementation_loop import make_implementation_agents
        from agentic_data_scientist.agents.claude_code import ClaudeCodeAgent
        from agentic_data_scientist.agents.coding_backends import RoutedExecutionAgent

        case_dir = _new_case_dir("impl_default")
        try:
            coding_agent, review_agent, review_confirmation = make_implementation_agents(str(case_dir), [])

            assert isinstance(coding_agent, RoutedExecutionAgent)
            assert isinstance(coding_agent._skill_executor, ClaudeCodeAgent)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)
