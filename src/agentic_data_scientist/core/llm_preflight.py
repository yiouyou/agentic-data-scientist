"""Preflight checks for configured LLM profiles and role routing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from agentic_data_scientist.core.llm_config import (
    LLMProfile,
    LLMRoutingConfig,
    ROLE_CAPABILITY_HINTS,
    profile_connection_kwargs,
    profile_has_api_key,
)


@dataclass
class ProfileCheckResult:
    """One profile preflight result."""

    profile: str
    model: str
    provider: str
    status: str
    basic_ok: bool
    structured_ok: bool
    latency_ms: int
    error: str = ""


def _extract_text(response: Any) -> str:
    """Best-effort extraction of text from LiteLLM response."""
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _litellm_completion(**kwargs: Any) -> Any:
    """Thin wrapper for easy testing via monkeypatch."""
    import litellm

    return litellm.completion(**kwargs)


def _call_model(profile: LLMProfile, messages: List[Dict[str, str]], timeout_seconds: int, structured: bool) -> str:
    """Call one model using LiteLLM and return text output."""
    kwargs: Dict[str, Any] = {
        "model": profile.model,
        "messages": messages,
        "temperature": profile.temperature,
        "max_tokens": 64,
        "timeout": timeout_seconds,
    }
    kwargs.update(profile_connection_kwargs(profile))
    if structured:
        kwargs["response_format"] = {"type": "json_object"}

    response = _litellm_completion(**kwargs)
    return _extract_text(response)


def _check_basic(profile: LLMProfile, timeout_seconds: int) -> tuple[bool, str]:
    """Run basic generation check."""
    prompt = [{"role": "user", "content": "Reply with exactly OK"}]
    try:
        text = _call_model(profile, prompt, timeout_seconds=timeout_seconds, structured=False)
    except Exception as exc:
        return False, str(exc)
    return "OK" in text.upper(), ""


def _check_structured(profile: LLMProfile, timeout_seconds: int) -> tuple[bool, str]:
    """Run JSON/structured output check."""
    prompt = [
        {
            "role": "user",
            "content": 'Return JSON only: {"ok": true, "source": "preflight"}',
        }
    ]
    try:
        text = _call_model(profile, prompt, timeout_seconds=timeout_seconds, structured=True)
    except Exception:
        # Retry without strict response_format for compatibility with providers
        try:
            text = _call_model(profile, prompt, timeout_seconds=timeout_seconds, structured=False)
        except Exception as exc:
            return False, str(exc)

    try:
        parsed = json.loads(text)
    except Exception as exc:
        return False, f"invalid JSON: {exc}"

    return bool(parsed.get("ok", False)), ""


def _status_from_checks(basic_ok: bool, structured_ok: bool) -> str:
    if basic_ok and structured_ok:
        return "ready"
    if basic_ok:
        return "degraded"
    return "unavailable"


def run_llm_preflight(config: LLMRoutingConfig, timeout_seconds: int = 20) -> Dict[str, Any]:
    """Run preflight for each enabled profile and evaluate role route readiness."""
    results: List[ProfileCheckResult] = []

    for profile_name, profile in config.profiles.items():
        if not profile.enabled:
            results.append(
                ProfileCheckResult(
                    profile=profile_name,
                    model=profile.model,
                    provider=profile.provider,
                    status="disabled",
                    basic_ok=False,
                    structured_ok=False,
                    latency_ms=0,
                    error="profile disabled",
                )
            )
            continue

        start = time.perf_counter()
        if not profile_has_api_key(profile):
            results.append(
                ProfileCheckResult(
                    profile=profile_name,
                    model=profile.model,
                    provider=profile.provider,
                    status="unavailable",
                    basic_ok=False,
                    structured_ok=False,
                    latency_ms=0,
                    error=f"missing API key env: {profile.api_key_env}",
                )
            )
            continue

        basic_ok, basic_error = _check_basic(profile, timeout_seconds=timeout_seconds)
        structured_ok, structured_error = _check_structured(profile, timeout_seconds=timeout_seconds) if basic_ok else (
            False,
            "",
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        status = _status_from_checks(basic_ok=basic_ok, structured_ok=structured_ok)
        error = basic_error or structured_error
        results.append(
            ProfileCheckResult(
                profile=profile_name,
                model=profile.model,
                provider=profile.provider,
                status=status,
                basic_ok=basic_ok,
                structured_ok=structured_ok,
                latency_ms=latency_ms,
                error=error,
            )
        )

    readiness_by_profile = {item.profile: item.status for item in results}
    result_by_profile = {item.profile: item for item in results}
    role_issues: List[str] = []
    route_uses_fallback: List[str] = []

    def _is_route_acceptable(role: str, profile_status: str, profile_result: ProfileCheckResult | None) -> bool:
        if profile_status == "ready":
            return True
        if profile_status != "degraded":
            return False
        # Structured-output critical roles must be fully ready.
        required_caps = ROLE_CAPABILITY_HINTS.get(role, set())
        if "structured_output" in required_caps:
            return False
        # Degraded means basic check passed; acceptable for non-structured roles.
        if profile_result is None:
            return False
        return bool(profile_result.basic_ok)

    for role, route in config.routing.items():
        primary_state = readiness_by_profile.get(route.primary, "unavailable")
        fallback_state = readiness_by_profile.get(route.fallback, "unavailable") if route.fallback else "unavailable"

        primary_result = result_by_profile.get(route.primary)
        fallback_result = result_by_profile.get(route.fallback) if route.fallback else None

        if _is_route_acceptable(role, primary_state, primary_result):
            continue

        if _is_route_acceptable(role, fallback_state, fallback_result):
            route_uses_fallback.append(role)
        else:
            role_issues.append(
                f"Role '{role}' has no ready route (primary={route.primary}:{primary_state}, "
                f"fallback={route.fallback or '-'}:{fallback_state})"
            )

    if role_issues:
        overall = "unavailable"
    elif route_uses_fallback:
        overall = "degraded"
    elif any(item.status == "degraded" for item in results):
        overall = "degraded"
    else:
        overall = "ready"

    return {
        "overall_status": overall,
        "profiles": [item.__dict__ for item in results],
        "route_issues": role_issues,
        "fallback_roles": route_uses_fallback,
    }


def format_preflight_report(report: Dict[str, Any]) -> str:
    """Format preflight report for CLI output."""
    lines = [f"LLM preflight overall: {report['overall_status']}"]
    lines.append("")
    lines.append("Profiles:")

    for item in report.get("profiles", []):
        lines.append(
            f"- {item['profile']}: {item['status']} "
            f"(provider={item['provider']}, model={item['model']}, "
            f"basic={item['basic_ok']}, structured={item['structured_ok']}, {item['latency_ms']}ms)"
        )
        if item.get("error"):
            lines.append(f"  error: {item['error']}")

    if report.get("fallback_roles"):
        lines.append("")
        lines.append("Roles using fallback:")
        for role in report["fallback_roles"]:
            lines.append(f"- {role}")

    if report.get("route_issues"):
        lines.append("")
        lines.append("Route issues:")
        for issue in report["route_issues"]:
            lines.append(f"- {issue}")

    return "\n".join(lines)
