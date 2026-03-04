"""Unit tests for role-based runtime model routing."""

from pathlib import Path

import agentic_data_scientist.agents.adk.utils as adk_utils
from agentic_data_scientist.core.llm_circuit_breaker import get_llm_circuit_breaker
from agentic_data_scientist.core.llm_config import LLMProfile, LLMRoutingConfig, RoleRoute


def _reset_routing_cache(path: Path):
    adk_utils.LLM_ROUTING_CONFIG_PATH = str(path)
    adk_utils._ROUTING_CONFIG_CACHE = None
    adk_utils._ROUTING_CONFIG_LOADED = False


def test_role_model_candidates_use_defaults_when_no_config(tmp_path: Path):
    """When routing file is absent, defaults should be used."""
    missing = tmp_path / "missing.yaml"
    _reset_routing_cache(missing)

    result = adk_utils.get_role_model_candidates(
        role="plan_maker",
        default_primary="default-primary-model",
        default_fallback="default-fallback-model",
    )
    assert result.primary_model == "default-primary-model"
    assert result.fallback_model == "default-fallback-model"
    assert result.source == "env_default"


def test_role_model_candidates_promote_fallback_when_primary_unavailable(tmp_path: Path, monkeypatch):
    """Fallback profile should be promoted if primary is missing required key."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  primary_profile:",
                "    provider: openai",
                "    model: gpt-5.2",
                "    api_key_env: PRIMARY_KEY",
                "    enabled: true",
                "  fallback_profile:",
                "    provider: deepseek",
                "    model: deepseek-reasoner",
                "    api_key_env: FALLBACK_KEY",
                "    enabled: true",
                "agent_routing:",
                "  plan_maker: {primary: primary_profile, fallback: fallback_profile, max_retry: 1}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.setenv("FALLBACK_KEY", "ok")
    _reset_routing_cache(config_path)

    result = adk_utils.get_role_model_candidates(
        role="plan_maker",
        default_primary="default-primary-model",
    )
    assert result.primary_model == "deepseek-reasoner"
    assert result.fallback_model is None
    assert result.source == "routing_fallback_promoted"


def test_get_litellm_for_role_uses_routed_model(tmp_path: Path, monkeypatch):
    """LiteLlm builder should use routed primary model when profile is usable."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "    api_key_env: OPENAI_API_KEY",
                "    enabled: true",
                "agent_routing:",
                "  summary_agent: {primary: p1, max_retry: 1}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    _reset_routing_cache(config_path)

    model = adk_utils.get_litellm_for_role(role="summary_agent", default_model_name="fallback-model")
    assert getattr(model, "model", "") == "gpt-5.2"


def test_get_litellm_candidates_for_role_builds_fallback_model(tmp_path: Path, monkeypatch):
    """When route has usable fallback, both primary and fallback LiteLlm should be built."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "    api_key_env: OPENAI_API_KEY",
                "    enabled: true",
                "  p2:",
                "    provider: dashscope",
                "    model: openai/qwen3.5-plus",
                "    api_key_env: DASHSCOPE_API_KEY",
                "    api_base: https://coding.dashscope.aliyuncs.com/v1",
                "    enabled: true",
                "agent_routing:",
                "  summary_agent: {primary: p1, fallback: p2, max_retry: 1}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ok")
    _reset_routing_cache(config_path)

    primary_model, fallback_model, resolved = adk_utils.get_litellm_candidates_for_role(
        role="summary_agent",
        default_model_name="default-model",
    )
    assert getattr(primary_model, "model", "") == "gpt-5.2"
    assert fallback_model is not None
    assert getattr(fallback_model, "model", "") == "openai/qwen3.5-plus"
    assert resolved.source == "routing_primary_fallback"


def test_get_litellm_candidates_for_role_builds_fallback_model_in_memory(monkeypatch):
    """Fallback LiteLlm building should work from in-memory routing config."""
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ok")
    try:
        adk_utils._ROUTING_CONFIG_CACHE = LLMRoutingConfig(
            profiles={
                "p1": LLMProfile(
                    name="p1",
                    provider="openai",
                    model="gpt-5.2",
                    api_key_env="OPENAI_API_KEY",
                    enabled=True,
                ),
                "p2": LLMProfile(
                    name="p2",
                    provider="dashscope",
                    model="openai/qwen3.5-plus",
                    api_key_env="DASHSCOPE_API_KEY",
                    api_base="https://coding.dashscope.aliyuncs.com/v1",
                    enabled=True,
                ),
            },
            routing={
                "summary_agent": RoleRoute(primary="p1", fallback="p2", max_retry=1),
            },
        )
        adk_utils._ROUTING_CONFIG_LOADED = True

        primary_model, fallback_model, resolved = adk_utils.get_litellm_candidates_for_role(
            role="summary_agent",
            default_model_name="default-model",
        )
        assert getattr(primary_model, "model", "") == "gpt-5.2"
        assert fallback_model is not None
        assert getattr(fallback_model, "model", "") == "openai/qwen3.5-plus"
        assert resolved.source == "routing_primary_fallback"
        assert resolved.max_retry == 1
    finally:
        adk_utils._ROUTING_CONFIG_CACHE = None
        adk_utils._ROUTING_CONFIG_LOADED = False


def test_get_role_model_candidates_disables_runtime_fallback_when_max_retry_zero(monkeypatch):
    """Route max_retry=0 should keep primary but disable execution-time fallback."""
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ok")
    try:
        adk_utils._ROUTING_CONFIG_CACHE = LLMRoutingConfig(
            profiles={
                "p1": LLMProfile(
                    name="p1",
                    provider="openai",
                    model="gpt-5.2",
                    api_key_env="OPENAI_API_KEY",
                    enabled=True,
                ),
                "p2": LLMProfile(
                    name="p2",
                    provider="dashscope",
                    model="openai/qwen3.5-plus",
                    api_key_env="DASHSCOPE_API_KEY",
                    api_base="https://coding.dashscope.aliyuncs.com/v1",
                    enabled=True,
                ),
            },
            routing={
                "summary_agent": RoleRoute(primary="p1", fallback="p2", max_retry=0),
            },
        )
        adk_utils._ROUTING_CONFIG_LOADED = True

        resolved = adk_utils.get_role_model_candidates(
            role="summary_agent",
            default_primary="default-model",
        )
        assert resolved.primary_model == "gpt-5.2"
        assert resolved.fallback_model is None
        assert resolved.max_retry == 0
    finally:
        adk_utils._ROUTING_CONFIG_CACHE = None
        adk_utils._ROUTING_CONFIG_LOADED = False


def test_get_role_model_candidates_uses_circuit_open_fallback(monkeypatch):
    """When circuit is open for primary profile, route should bypass it to fallback."""
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ok")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "1")

    breaker = get_llm_circuit_breaker()
    breaker._states.clear()

    try:
        adk_utils._ROUTING_CONFIG_CACHE = LLMRoutingConfig(
            profiles={
                "p1": LLMProfile(
                    name="p1",
                    provider="openai",
                    model="gpt-5.2",
                    api_key_env="OPENAI_API_KEY",
                    enabled=True,
                ),
                "p2": LLMProfile(
                    name="p2",
                    provider="dashscope",
                    model="openai/qwen3.5-plus",
                    api_key_env="DASHSCOPE_API_KEY",
                    api_base="https://coding.dashscope.aliyuncs.com/v1",
                    enabled=True,
                ),
            },
            routing={
                "summary_agent": RoleRoute(primary="p1", fallback="p2", max_retry=1),
            },
        )
        adk_utils._ROUTING_CONFIG_LOADED = True

        breaker.record_retryable_failure("summary_agent", "p1", Exception("rate limit"))
        resolved = adk_utils.get_role_model_candidates(
            role="summary_agent",
            default_primary="default-model",
        )
        assert resolved.source == "circuit_open_fallback"
        assert resolved.primary_model == "openai/qwen3.5-plus"
        assert resolved.fallback_model is None
    finally:
        breaker._states.clear()
        adk_utils._ROUTING_CONFIG_CACHE = None
        adk_utils._ROUTING_CONFIG_LOADED = False


def test_execution_agent_role_uses_legacy_coding_agent_route(monkeypatch):
    """`execution_agent` lookup should resolve legacy `coding_agent` route key."""
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    try:
        adk_utils._ROUTING_CONFIG_CACHE = LLMRoutingConfig(
            profiles={
                "p1": LLMProfile(
                    name="p1",
                    provider="openai",
                    model="gpt-5.2",
                    api_key_env="OPENAI_API_KEY",
                    enabled=True,
                ),
            },
            routing={
                "coding_agent": RoleRoute(primary="p1", max_retry=1),
            },
        )
        adk_utils._ROUTING_CONFIG_LOADED = True

        resolved = adk_utils.get_role_model_candidates(
            role="execution_agent",
            default_primary="default-model",
        )
        assert resolved.primary_model == "gpt-5.2"
        assert resolved.source == "routing_primary_only"
    finally:
        adk_utils._ROUTING_CONFIG_CACHE = None
        adk_utils._ROUTING_CONFIG_LOADED = False
