"""Unit tests for LLM routing config loader/validator."""

from pathlib import Path

import pytest

from agentic_data_scientist.core.llm_config import (
    REQUIRED_ROLES,
    load_llm_routing_config,
    validate_routing_config,
    write_llm_config_template,
)


def _full_role_routing(primary: str, fallback: str = "") -> str:
    lines = []
    for role in REQUIRED_ROLES:
        if fallback:
            lines.append(f"  {role}: {{primary: {primary}, fallback: {fallback}, max_retry: 1}}")
        else:
            lines.append(f"  {role}: {{primary: {primary}, max_retry: 1}}")
    return "\n".join(lines)


def test_load_llm_routing_config_success(tmp_path: Path):
    """Loader should parse valid YAML into typed config."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openrouter",
                "    model: google/gemini-2.5-pro",
                "    api_key_env: OPENROUTER_API_KEY",
                "    capabilities: [reasoning]",
                "agent_routing:",
                _full_role_routing("p1"),
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    assert "p1" in config.profiles
    assert config.profiles["p1"].model == "google/gemini-2.5-pro"
    assert config.routing["plan_maker"].primary == "p1"


def test_validate_routing_config_unknown_profile(tmp_path: Path):
    """Validation should fail when route points to missing profile."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openrouter",
                "    model: google/gemini-2.5-pro",
                "agent_routing:",
                _full_role_routing("missing_profile"),
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    errors, warnings = validate_routing_config(config)
    assert errors
    assert any("primary profile not found" in item for item in errors)
    assert isinstance(warnings, list)


def test_validate_routing_config_missing_required_roles(tmp_path: Path):
    """Validation should flag missing required role routes."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openrouter",
                "    model: google/gemini-2.5-pro",
                "agent_routing:",
                "  plan_maker: {primary: p1, max_retry: 1}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    errors, _ = validate_routing_config(config)
    assert any("Missing route for required role" in item for item in errors)


def test_write_llm_config_template(tmp_path: Path):
    """Template writer should create YAML file."""
    config_path = tmp_path / "configs" / "llm_routing.yaml"
    written = write_llm_config_template(config_path)
    assert written.exists()
    assert "llm_profiles:" in written.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_llm_config_template(config_path)


def test_validate_routing_config_rejects_non_env_api_key_name(tmp_path: Path):
    """api_key_env must be an env var name, not a raw token-like value."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "    api_key_env: sk-abc123",
                "agent_routing:",
                _full_role_routing("p1"),
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    errors, _ = validate_routing_config(config)
    assert any("invalid api_key_env" in item for item in errors)


def test_validate_routing_config_rejects_negative_max_retry(tmp_path: Path):
    """max_retry must be non-negative."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "agent_routing:",
                "  plan_maker: {primary: p1, max_retry: -1}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    errors, _ = validate_routing_config(config)
    assert any("invalid max_retry" in item for item in errors)


def test_validate_routing_config_warns_when_max_retry_exceeds_runtime_support(tmp_path: Path):
    """max_retry >1 should be warned because runtime currently supports 0/1."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "agent_routing:",
                "  plan_maker: {primary: p1, max_retry: 3}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    _, warnings = validate_routing_config(config)
    assert any("exceeds current runtime fallback support" in item for item in warnings)


def test_load_llm_routing_config_parses_coding_executor(tmp_path: Path):
    """Loader should parse optional profile coding_executor."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "    coding_executor: codex",
                "agent_routing:",
                _full_role_routing("p1"),
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    assert config.profiles["p1"].coding_executor == "codex"


def test_validate_routing_config_rejects_unknown_coding_executor(tmp_path: Path):
    """Unknown coding_executor value should fail validation."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "    coding_executor: mystery_backend",
                "agent_routing:",
                _full_role_routing("p1"),
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    errors, _ = validate_routing_config(config)
    assert any("unsupported coding_executor" in item for item in errors)


def test_validate_routing_config_accepts_legacy_coding_agent_alias(tmp_path: Path):
    """Legacy `coding_agent` key should satisfy required `execution_agent` role."""
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm_profiles:",
                "  p1:",
                "    provider: openai",
                "    model: gpt-5.2",
                "agent_routing:",
                "  plan_maker: {primary: p1, max_retry: 1}",
                "  plan_reviewer: {primary: p1, max_retry: 1}",
                "  plan_parser: {primary: p1, max_retry: 1}",
                "  criteria_checker: {primary: p1, max_retry: 1}",
                "  stage_reflector: {primary: p1, max_retry: 1}",
                "  coding_agent: {primary: p1, max_retry: 1}",
                "  review_agent: {primary: p1, max_retry: 1}",
                "  summary_agent: {primary: p1, max_retry: 1}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_routing_config(config_path)
    errors, _ = validate_routing_config(config)
    assert not any("Missing route for required role: execution_agent" in item for item in errors)
