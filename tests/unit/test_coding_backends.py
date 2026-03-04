"""Unit tests for coding backend routing and agent factory behavior."""

import agentic_data_scientist.agents.adk.utils as adk_utils
from agentic_data_scientist.agents.adk.implementation_loop import make_implementation_agents
from agentic_data_scientist.agents.coding_backends import (
    CodexCodeAgent,
    RoutedExecutionAgent,
    resolve_coding_backend_route,
    resolve_coding_executor,
)
from agentic_data_scientist.core.llm_config import LLMProfile, LLMRoutingConfig, RoleRoute


def test_resolve_coding_executor_prefers_profile_setting():
    """Profile coding_executor should override heuristics/defaults."""
    profile = LLMProfile(
        name="p1",
        provider="openai",
        model="gpt-5.2",
        coding_executor="codex",
    )
    assert resolve_coding_executor(profile, "gpt-5.2") == "codex"


def test_resolve_coding_backend_route_disables_cross_executor_fallback():
    """Cross-executor fallback should be disabled for now."""
    primary = LLMProfile(name="p1", provider="anthropic", model="claude-sonnet-4-6", coding_executor="claude_code")
    fallback = LLMProfile(name="p2", provider="openai", model="gpt-5.2", coding_executor="codex")
    route = resolve_coding_backend_route(
        primary_profile=primary,
        primary_model=primary.model,
        fallback_profile=fallback,
        fallback_model=fallback.model,
        max_retry=1,
    )
    assert route.primary_executor == "claude_code"
    assert route.fallback_executor == "codex"
    assert route.fallback_enabled is False
    assert "cross-executor fallback" in route.fallback_reason


def test_make_implementation_agents_uses_codex_when_configured(monkeypatch, tmp_working_dir):
    """Coding agent should switch to CodexCodeAgent when route profile requests codex executor."""
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    try:
        adk_utils._ROUTING_CONFIG_CACHE = LLMRoutingConfig(
            profiles={
                "code_primary": LLMProfile(
                    name="code_primary",
                    provider="openai",
                    model="gpt-5.2",
                    api_key_env="OPENAI_API_KEY",
                    coding_executor="codex",
                    enabled=True,
                ),
            },
            routing={
                "coding_agent": RoleRoute(primary="code_primary", max_retry=0),
                "review_agent": RoleRoute(primary="code_primary", max_retry=0),
            },
        )
        adk_utils._ROUTING_CONFIG_LOADED = True

        coding_agent, _, _ = make_implementation_agents(str(tmp_working_dir), [])
        assert isinstance(coding_agent, RoutedExecutionAgent)
        assert isinstance(coding_agent._skill_executor, CodexCodeAgent)
    finally:
        adk_utils._ROUTING_CONFIG_CACHE = None
        adk_utils._ROUTING_CONFIG_LOADED = False
