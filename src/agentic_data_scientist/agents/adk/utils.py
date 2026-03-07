"""
Utility functions and configurations for ADK agents.

This module provides model configuration, helper functions, and shared settings
for the ADK agent system.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from agentic_data_scientist.core.llm_config import (
    LLMProfile,
    LLMRoutingConfig,
    ROLE_ALIASES,
    load_llm_routing_config,
    profile_connection_kwargs,
    profile_has_api_key,
    validate_routing_config,
)
from agentic_data_scientist.core.llm_circuit_breaker import get_llm_circuit_breaker


load_dotenv()

logger = logging.getLogger(__name__)


# Model configuration
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "gemini-3.1-pro-preview")
REVIEW_MODEL_NAME = os.getenv("REVIEW_MODEL", "gemini-3.1-pro-preview")
CODING_MODEL_NAME = os.getenv("CODING_MODEL", "claude-sonnet-4-6")
LLM_ROUTING_CONFIG_PATH = os.getenv("LLM_ROUTING_CONFIG_PATH", "configs/llm_routing.yaml")

logger.info(f"[AgenticDS] DEFAULT_MODEL={DEFAULT_MODEL_NAME}")
logger.info(f"[AgenticDS] REVIEW_MODEL={REVIEW_MODEL_NAME}")
logger.info(f"[AgenticDS] CODING_MODEL={CODING_MODEL_NAME}")

# Configure LiteLLM for OpenRouter
# OpenRouter requires specific environment variables and configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OR_SITE_URL = os.getenv("OR_SITE_URL", "k-dense.ai")
OR_APP_NAME = os.getenv("OR_APP_NAME", "Agentic Data Scientist")

# Export for use in event compression
__all__ = [
    'DEFAULT_MODEL',
    'REVIEW_MODEL',
    'DEFAULT_MODEL_NAME',
    'REVIEW_MODEL_NAME',  # Export model name strings
    'OPENROUTER_API_KEY',
    'OPENROUTER_API_BASE',
    'get_generate_content_config',
    'get_role_model_candidates',
    'get_litellm_candidates_for_role',
    'get_litellm_for_role',
    'exit_loop_simple',
    'is_network_disabled',
]

# Set up LiteLLM environment for OpenRouter
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    logger.info("[AgenticDS] OpenRouter API key configured")
else:
    logger.warning("[AgenticDS] OPENROUTER_API_KEY not set - using default credentials")

# Create LiteLLM model instances
# LiteLLM will automatically route through OpenRouter when model names have the provider prefix (e.g., "google/", "anthropic/")
DEFAULT_MODEL = LiteLlm(
    model=DEFAULT_MODEL_NAME,
    num_retries=10,
    timeout=300,
    api_base=OPENROUTER_API_BASE if OPENROUTER_API_KEY else None,
    custom_llm_provider="openrouter" if OPENROUTER_API_KEY else None,
)

REVIEW_MODEL = LiteLlm(
    model=REVIEW_MODEL_NAME,
    num_retries=10,
    timeout=300,
    api_base=OPENROUTER_API_BASE if OPENROUTER_API_KEY else None,
    custom_llm_provider="openrouter" if OPENROUTER_API_KEY else None,
)


@dataclass
class RoleModelSelection:
    """Resolved primary/fallback model names for one workflow role."""

    primary_model: str
    fallback_model: Optional[str]
    source: str
    max_retry: int = 1
    selected_profile: Optional[LLMProfile] = None
    fallback_profile: Optional[LLMProfile] = None


_ROUTING_CONFIG_CACHE: Optional[LLMRoutingConfig] = None
_ROUTING_CONFIG_LOADED = False


def _load_routing_config_once() -> Optional[LLMRoutingConfig]:
    """Load routing config lazily once and cache it."""
    global _ROUTING_CONFIG_CACHE, _ROUTING_CONFIG_LOADED

    if _ROUTING_CONFIG_LOADED:
        return _ROUTING_CONFIG_CACHE

    _ROUTING_CONFIG_LOADED = True
    config_path = Path(LLM_ROUTING_CONFIG_PATH)
    if not config_path.exists():
        logger.info(f"[AgenticDS] LLM routing config not found at {config_path}, using env defaults")
        _ROUTING_CONFIG_CACHE = None
        return None

    try:
        config = load_llm_routing_config(config_path)
        errors, warnings = validate_routing_config(config)
        blocking_errors = [e for e in errors if not e.startswith("Missing route for required role:")]
        non_blocking_errors = [e for e in errors if e.startswith("Missing route for required role:")]
        if blocking_errors:
            logger.warning(
                f"[AgenticDS] LLM routing config invalid at {config_path}, falling back to env defaults. "
                f"Errors: {blocking_errors}"
            )
            _ROUTING_CONFIG_CACHE = None
            return None
        if non_blocking_errors:
            logger.warning(f"[AgenticDS] LLM routing config partial role coverage: {non_blocking_errors}")
        if warnings:
            logger.warning(f"[AgenticDS] LLM routing config warnings: {warnings}")
        _ROUTING_CONFIG_CACHE = config
        logger.info(f"[AgenticDS] Loaded LLM routing config from {config_path}")
        return config
    except Exception as exc:
        logger.warning(f"[AgenticDS] Failed to load LLM routing config at {config_path}, using env defaults: {exc}")
        _ROUTING_CONFIG_CACHE = None
        return None


def _profile_is_usable(profile: Optional[LLMProfile]) -> bool:
    """Return whether profile exists, enabled, and has required API key."""
    if profile is None:
        return False
    if not profile.enabled:
        return False
    if not profile_has_api_key(profile):
        return False
    return True


def get_role_model_candidates(
    role: str,
    default_primary: str,
    default_fallback: Optional[str] = None,
) -> RoleModelSelection:
    """
    Resolve primary/fallback model names for a role.

    Resolution order:
    1) `configs/llm_routing.yaml` (or env-defined path) if present and valid
    2) fallback to provided default model names
    """
    config = _load_routing_config_once()
    if config is None:
        return RoleModelSelection(
            primary_model=default_primary,
            fallback_model=default_fallback,
            source="env_default",
            max_retry=1,
        )

    route = config.routing.get(role)
    if route is None:
        alias = ROLE_ALIASES.get(role)
        if alias:
            route = config.routing.get(alias)

    if route is None:
        return RoleModelSelection(
            primary_model=default_primary,
            fallback_model=default_fallback,
            source="env_default",
            max_retry=1,
        )

    max_retry = max(0, route.max_retry)

    primary_profile = config.profiles.get(route.primary)
    fallback_profile = config.profiles.get(route.fallback) if route.fallback else None

    primary_ready = _profile_is_usable(primary_profile)
    fallback_ready = _profile_is_usable(fallback_profile)
    primary_profile_name = primary_profile.name if primary_profile is not None else route.primary

    if primary_ready and fallback_ready:
        # Circuit-open primaries are bypassed temporarily to reduce repeated failures.
        if max_retry > 0 and get_llm_circuit_breaker().should_force_fallback(role, primary_profile_name):
            return RoleModelSelection(
                primary_model=fallback_profile.model,
                fallback_model=None,
                source="circuit_open_fallback",
                max_retry=max_retry,
                selected_profile=fallback_profile,
            )
        return RoleModelSelection(
            primary_model=primary_profile.model,
            fallback_model=fallback_profile.model if max_retry > 0 else None,
            source="routing_primary_fallback",
            max_retry=max_retry,
            selected_profile=primary_profile,
            fallback_profile=fallback_profile if max_retry > 0 else None,
        )

    if primary_ready:
        return RoleModelSelection(
            primary_model=primary_profile.model,
            fallback_model=None,
            source="routing_primary_only",
            max_retry=max_retry,
            selected_profile=primary_profile,
        )

    if fallback_ready:
        return RoleModelSelection(
            primary_model=fallback_profile.model,
            fallback_model=None,
            source="routing_fallback_promoted",
            max_retry=max_retry,
            selected_profile=fallback_profile,
        )

    return RoleModelSelection(
        primary_model=default_primary,
        fallback_model=default_fallback,
        source="env_default",
        max_retry=max_retry,
    )


def _build_litellm(model_name: str, profile: Optional[LLMProfile] = None) -> LiteLlm:
    """Build a LiteLlm instance using environment/provider defaults."""
    timeout = profile.timeout if profile is not None else 300
    kwargs = {
        "model": model_name,
        "num_retries": 10,
        "timeout": timeout,
    }
    if profile is not None:
        kwargs.update(profile_connection_kwargs(profile))
    elif OPENROUTER_API_KEY:
        kwargs["api_base"] = OPENROUTER_API_BASE
        kwargs["custom_llm_provider"] = "openrouter"

    return LiteLlm(
        **kwargs,
    )


def get_litellm_for_role(role: str, default_model_name: str) -> LiteLlm:
    """Resolve and build LiteLlm for one role."""
    primary_model, _, _ = get_litellm_candidates_for_role(
        role=role,
        default_model_name=default_model_name,
    )
    return primary_model


def get_litellm_candidates_for_role(
    role: str,
    default_model_name: str,
    default_fallback_model_name: Optional[str] = None,
) -> Tuple[LiteLlm, Optional[LiteLlm], RoleModelSelection]:
    """Resolve and build primary/fallback LiteLlm instances for one role."""
    resolved = get_role_model_candidates(
        role=role,
        default_primary=default_model_name,
        default_fallback=default_fallback_model_name,
    )
    logger.info(
        f"[AgenticDS] Role '{role}' model resolved to {resolved.primary_model} "
        f"(fallback={resolved.fallback_model or '-'}, source={resolved.source})"
    )

    primary_model = _build_litellm(resolved.primary_model, profile=resolved.selected_profile)
    fallback_model: Optional[LiteLlm] = None
    if resolved.fallback_model:
        fallback_model = _build_litellm(resolved.fallback_model, profile=resolved.fallback_profile)

    return primary_model, fallback_model, resolved


# Language requirement (empty for English-only models)
LANGUAGE_REQUIREMENT = ""


def is_network_disabled() -> bool:
    """
    Check if network access is disabled via environment variable.

    Network access is enabled by default. Set DISABLE_NETWORK_ACCESS
    to "true" or "1" to disable network tools.

    Returns
    -------
    bool
        True if network access should be disabled, False otherwise
    """
    disable_network = os.getenv("DISABLE_NETWORK_ACCESS", "").lower()
    return disable_network in ("true", "1")


# DEPRECATED: Use review_confirmation agents instead
# This function is kept for backward compatibility but should not be used in new code.
# Loop exit decisions should be made by dedicated review_confirmation agents with
# structured output and callbacks, not by direct tool calls from review agents.
def exit_loop_simple(tool_context: ToolContext):
    """
    Exit the iterative loop when no further changes are needed.

    DEPRECATED: Use review_confirmation agents instead.

    This function is called by review agents to signal that the iterative
    process should end.

    Parameters
    ----------
    tool_context : ToolContext
        The tool execution context

    Returns
    -------
    dict
        Empty dictionary (tools should return JSON-serializable output)
    """
    tool_context.actions.escalate = True
    return {}


def get_generate_content_config(temperature: float = 0.0, output_tokens: Optional[int] = None):
    """
    Create a GenerateContentConfig with retry settings.

    Parameters
    ----------
    temperature : float, optional
        Sampling temperature (default: 0.0)
    output_tokens : int, optional
        Maximum output tokens

    Returns
    -------
    types.GenerateContentConfig
        Configuration for content generation
    """
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        seed=42,
        max_output_tokens=output_tokens,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=50,
                initial_delay=1.0,
                max_delay=30,
                exp_base=1.5,
                jitter=0.5,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        ),
    )
