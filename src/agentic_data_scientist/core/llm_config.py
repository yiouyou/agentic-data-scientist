"""LLM routing configuration loader and validator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


REQUIRED_ROLES = [
    "plan_maker",
    "plan_reviewer",
    "plan_parser",
    "criteria_checker",
    "stage_reflector",
    "execution_agent",
    "review_agent",
    "summary_agent",
]

ROLE_ALIASES = {
    # Preferred name -> backward-compatible alias
    "execution_agent": "coding_agent",
    # Reverse lookup for callers still using the old name.
    "coding_agent": "execution_agent",
}


ROLE_CAPABILITY_HINTS = {
    "plan_maker": {"reasoning"},
    "plan_reviewer": {"reasoning"},
    "plan_parser": {"structured_output"},
    "criteria_checker": {"structured_output"},
    "stage_reflector": {"reasoning"},
    "execution_agent": {"tool_calling"},
    "coding_agent": {"tool_calling"},
    "review_agent": {"reasoning"},
    "summary_agent": {"long_context"},
}

SUPPORTED_CODING_EXECUTORS = {"claude_code", "codex", "opencode"}


DEFAULT_LLM_CONFIG_TEMPLATE = """# LLM routing config for agentic-data-scientist
# Copy this file to configs/llm_routing.yaml and fill your own model names + API env vars.
#
# Notes:
# - provider is informational for now; actual invocation uses model + optional api_base/api_key.
# - capabilities drive static validation hints for role assignment.
# - execution_agent fallback is currently model-only within one coding_executor (no cross-executor fallback).
# - prefer cross-provider fallback for resilience.
#
llm_profiles:
  us_openai_reasoning:
    provider: openai
    model: gpt-5.2
    api_key_env: OPENAI_API_KEY
    # Optional for coding profiles: claude_code | codex | opencode
    # coding_executor: codex
    temperature: 0.3
    capabilities: [reasoning, structured_output, long_context, tool_calling]
    enabled: true

  us_anthropic_code:
    provider: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    coding_executor: claude_code
    temperature: 0.2
    capabilities: [code, reasoning, long_context, tool_calling]
    enabled: true

  us_google_reasoning:
    provider: google
    model: gemini-3.1-pro-preview
    api_key_env: GOOGLE_API_KEY
    temperature: 0.3
    capabilities: [reasoning, structured_output, long_context, tool_calling]
    enabled: true

  cn_deepseek_reasoning:
    provider: deepseek
    model: deepseek/deepseek-reasoner
    api_key_env: DEEPSEEK_API_KEY
    api_base: https://api.deepseek.com
    temperature: 0.2
    capabilities: [reasoning]
    enabled: true

  cn_deepseek_chat:
    provider: deepseek
    model: deepseek/deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
    api_base: https://api.deepseek.com
    temperature: 0.1
    capabilities: [reasoning, structured_output, tool_calling]
    enabled: true

  cn_qwen_reasoning:
    provider: dashscope
    model: openai/qwen3.5-plus
    api_key_env: DASHSCOPE_API_KEY
    api_base: https://coding.dashscope.aliyuncs.com/v1
    temperature: 0.0
    capabilities: [reasoning, structured_output, long_context, tool_calling]
    enabled: true

  cn_qwen_code:
    provider: dashscope
    model: openai/qwen3-coder-plus
    api_key_env: DASHSCOPE_API_KEY
    api_base: https://coding.dashscope.aliyuncs.com/v1
    # Example: use OpenCode CLI as coding executor for this profile.
    # coding_executor: opencode
    temperature: 0.1
    capabilities: [code, structured_output, tool_calling]
    enabled: true

agent_routing:
  plan_maker: {primary: us_openai_reasoning, fallback: cn_deepseek_reasoning, max_retry: 1}
  plan_reviewer: {primary: cn_deepseek_reasoning, fallback: us_google_reasoning, max_retry: 1}
  plan_parser: {primary: cn_qwen_reasoning, fallback: us_openai_reasoning, max_retry: 1}
  criteria_checker: {primary: cn_qwen_reasoning, fallback: us_openai_reasoning, max_retry: 1}
  stage_reflector: {primary: us_google_reasoning, fallback: cn_deepseek_reasoning, max_retry: 1}
  # preferred role key: execution_agent (legacy alias: coding_agent)
  execution_agent: {primary: us_anthropic_code, fallback: cn_qwen_code, max_retry: 1}
  review_agent: {primary: cn_deepseek_chat, fallback: us_openai_reasoning, max_retry: 1}
  summary_agent: {primary: cn_qwen_reasoning, fallback: us_openai_reasoning, max_retry: 1}
"""


@dataclass
class LLMProfile:
    """One model profile candidate."""

    name: str
    provider: str
    model: str
    api_key_env: str = ""
    api_base: str = ""
    coding_executor: str = ""
    temperature: float = 0.0
    enabled: bool = True
    capabilities: List[str] = field(default_factory=list)


@dataclass
class RoleRoute:
    """Role-level model routing config."""

    primary: str
    fallback: str = ""
    max_retry: int = 1


@dataclass
class LLMRoutingConfig:
    """Top-level routing config."""

    profiles: Dict[str, LLMProfile]
    routing: Dict[str, RoleRoute]


def _as_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_llm_routing_config(path: str | Path) -> LLMRoutingConfig:
    """Load model profiles and role routing from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _as_dict(parsed, "root")

    raw_profiles = _as_dict(root.get("llm_profiles", {}), "llm_profiles")
    raw_routing = _as_dict(root.get("agent_routing", {}), "agent_routing")

    profiles: Dict[str, LLMProfile] = {}
    for name, raw in raw_profiles.items():
        item = _as_dict(raw, f"llm_profiles.{name}")
        model = str(item.get("model", "")).strip()
        if not model:
            raise ValueError(f"llm_profiles.{name}.model is required")

        capabilities = item.get("capabilities", [])
        if capabilities is None:
            capabilities = []
        if not isinstance(capabilities, list):
            raise ValueError(f"llm_profiles.{name}.capabilities must be a list")

        profiles[name] = LLMProfile(
            name=name,
            provider=str(item.get("provider", "")).strip() or "unknown",
            model=model,
            api_key_env=str(item.get("api_key_env", "")).strip(),
            api_base=str(item.get("api_base", "")).strip(),
            coding_executor=str(item.get("coding_executor", "")).strip().lower(),
            temperature=float(item.get("temperature", 0.0)),
            enabled=bool(item.get("enabled", True)),
            capabilities=[str(x).strip() for x in capabilities if str(x).strip()],
        )

    routing: Dict[str, RoleRoute] = {}
    for role, raw in raw_routing.items():
        item = _as_dict(raw, f"agent_routing.{role}")
        primary = str(item.get("primary", "")).strip()
        fallback = str(item.get("fallback", "")).strip()
        if not primary:
            raise ValueError(f"agent_routing.{role}.primary is required")
        routing[role] = RoleRoute(
            primary=primary,
            fallback=fallback,
            max_retry=int(item.get("max_retry", 1)),
        )

    return LLMRoutingConfig(profiles=profiles, routing=routing)


def validate_routing_config(config: LLMRoutingConfig) -> Tuple[List[str], List[str]]:
    """Validate role mappings and emit errors/warnings."""
    errors: List[str] = []
    warnings: List[str] = []

    if not config.profiles:
        errors.append("No llm_profiles configured")
        return errors, warnings

    if "execution_agent" in config.routing and "coding_agent" in config.routing:
        warnings.append(
            "Both 'execution_agent' and legacy alias 'coding_agent' are configured; "
            "prefer only 'execution_agent' to avoid ambiguity."
        )

    for role in REQUIRED_ROLES:
        if _find_route_with_alias(config.routing, role) is None:
            alias = ROLE_ALIASES.get(role)
            if alias:
                errors.append(f"Missing route for required role: {role} (or legacy alias: {alias})")
            else:
                errors.append(f"Missing route for required role: {role}")

    for role, route in config.routing.items():
        primary = config.profiles.get(route.primary)
        fallback = config.profiles.get(route.fallback) if route.fallback else None

        if primary is None:
            errors.append(f"Route '{role}' primary profile not found: {route.primary}")
            continue
        if not primary.enabled:
            warnings.append(f"Route '{role}' primary profile is disabled: {route.primary}")
        if primary.api_key_env and not _is_valid_env_var_name(primary.api_key_env):
            errors.append(
                f"Route '{role}' primary '{route.primary}' has invalid api_key_env: "
                f"{primary.api_key_env!r} (must be env var name like OPENAI_API_KEY)"
            )
        if primary.coding_executor and primary.coding_executor not in SUPPORTED_CODING_EXECUTORS:
            errors.append(
                f"Route '{role}' primary '{route.primary}' has unsupported coding_executor: "
                f"{primary.coding_executor!r} (supported: {sorted(SUPPORTED_CODING_EXECUTORS)})"
            )

        if route.fallback:
            if fallback is None:
                errors.append(f"Route '{role}' fallback profile not found: {route.fallback}")
            elif not fallback.enabled:
                warnings.append(f"Route '{role}' fallback profile is disabled: {route.fallback}")
            elif fallback.api_key_env and not _is_valid_env_var_name(fallback.api_key_env):
                errors.append(
                    f"Route '{role}' fallback '{route.fallback}' has invalid api_key_env: "
                    f"{fallback.api_key_env!r} (must be env var name like OPENAI_API_KEY)"
                )
            elif fallback.coding_executor and fallback.coding_executor not in SUPPORTED_CODING_EXECUTORS:
                errors.append(
                    f"Route '{role}' fallback '{route.fallback}' has unsupported coding_executor: "
                    f"{fallback.coding_executor!r} (supported: {sorted(SUPPORTED_CODING_EXECUTORS)})"
                )
            if route.fallback == route.primary:
                warnings.append(f"Route '{role}' has identical primary/fallback: {route.primary}")

        if route.max_retry < 0:
            errors.append(f"Route '{role}' has invalid max_retry={route.max_retry}; must be >= 0")
        elif route.max_retry > 1:
            warnings.append(
                f"Route '{role}' max_retry={route.max_retry} exceeds current runtime fallback support (0 or 1)"
            )

        expected_caps = ROLE_CAPABILITY_HINTS.get(role, set())
        if expected_caps and not expected_caps.intersection(set(primary.capabilities)):
            warnings.append(
                f"Route '{role}' primary '{route.primary}' capability mismatch "
                f"(needs one of: {sorted(expected_caps)})"
            )

    return errors, warnings


def _find_route_with_alias(routing: Dict[str, RoleRoute], role: str) -> RoleRoute | None:
    """Find one role route with backward-compatible alias lookup."""
    route = routing.get(role)
    if route is not None:
        return route
    alias = ROLE_ALIASES.get(role)
    if not alias:
        return None
    return routing.get(alias)


def write_llm_config_template(path: str | Path, overwrite: bool = False) -> Path:
    """Write default template to target path."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Config file already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_LLM_CONFIG_TEMPLATE, encoding="utf-8")
    return target


def _is_valid_env_var_name(name: str) -> bool:
    """Validate expected env var naming style."""
    if not name:
        return True
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isupper() or ch.isdigit() or ch == "_" for ch in name)


def profile_has_api_key(profile: LLMProfile) -> bool:
    """Return whether required API key env var is present."""
    if not profile.api_key_env:
        return True
    return bool(os.getenv(profile.api_key_env))


def profile_connection_kwargs(profile: LLMProfile) -> Dict[str, Any]:
    """
    Build connection kwargs for LiteLLM/OpenAI-compatible calls from profile config.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments such as api_key/api_base/custom_llm_provider.
    """
    kwargs: Dict[str, Any] = {}

    if profile.api_key_env:
        api_key = os.getenv(profile.api_key_env, "")
        if api_key:
            kwargs["api_key"] = api_key

    if profile.api_base:
        kwargs["api_base"] = profile.api_base

    # DashScope coding endpoint is OpenAI-compatible.
    if profile.provider == "dashscope":
        kwargs["custom_llm_provider"] = "openai"
    elif profile.provider == "openrouter":
        kwargs["custom_llm_provider"] = "openrouter"

    return kwargs
