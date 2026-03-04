"""Unit tests for LLM preflight checks."""

from agentic_data_scientist.core.llm_config import LLMProfile, LLMRoutingConfig, RoleRoute
from agentic_data_scientist.core.llm_preflight import format_preflight_report, run_llm_preflight


def _make_config(primary: str, fallback: str = "", profiles: dict | None = None) -> LLMRoutingConfig:
    profile_map = profiles or {}
    routing = {"plan_maker": RoleRoute(primary=primary, fallback=fallback, max_retry=1)}
    return LLMRoutingConfig(profiles=profile_map, routing=routing)


def test_run_llm_preflight_missing_api_key(monkeypatch):
    """Profiles requiring missing keys should be unavailable."""
    profile = LLMProfile(
        name="p1",
        provider="openrouter",
        model="google/gemini-2.5-pro",
        api_key_env="NOT_SET_KEY",
        enabled=True,
    )
    monkeypatch.delenv("NOT_SET_KEY", raising=False)
    report = run_llm_preflight(_make_config(primary="p1", profiles={"p1": profile}))

    assert report["overall_status"] == "unavailable"
    assert report["profiles"][0]["status"] == "unavailable"
    assert "missing API key env" in report["profiles"][0]["error"]


def test_run_llm_preflight_ready(monkeypatch):
    """Passing basic + structured checks should mark profile ready."""
    from agentic_data_scientist.core import llm_preflight

    monkeypatch.setattr(llm_preflight, "_check_basic", lambda profile, timeout_seconds: (True, ""))
    monkeypatch.setattr(llm_preflight, "_check_structured", lambda profile, timeout_seconds: (True, ""))
    profile = LLMProfile(name="p1", provider="openrouter", model="google/gemini-2.5-pro", enabled=True)

    report = run_llm_preflight(_make_config(primary="p1", profiles={"p1": profile}))
    assert report["overall_status"] == "ready"
    assert report["profiles"][0]["status"] == "ready"


def test_run_llm_preflight_fallback_route(monkeypatch):
    """If primary is unavailable but fallback is ready, overall should be degraded."""
    from agentic_data_scientist.core import llm_preflight

    def fake_basic(profile, timeout_seconds):
        return (profile.name == "fallback", "")

    monkeypatch.setattr(llm_preflight, "_check_basic", fake_basic)
    monkeypatch.setattr(llm_preflight, "_check_structured", lambda profile, timeout_seconds: (True, ""))

    primary = LLMProfile(name="primary", provider="openrouter", model="m1", enabled=True)
    fallback = LLMProfile(name="fallback", provider="openrouter", model="m2", enabled=True)
    config = _make_config(primary="primary", fallback="fallback", profiles={"primary": primary, "fallback": fallback})

    report = run_llm_preflight(config)
    assert report["overall_status"] == "degraded"
    assert "plan_maker" in report["fallback_roles"]


def test_run_llm_preflight_allows_degraded_for_non_structured_roles(monkeypatch):
    """Non-structured roles can run on degraded profiles (basic OK, structured not OK)."""
    from agentic_data_scientist.core import llm_preflight

    monkeypatch.setattr(llm_preflight, "_check_basic", lambda profile, timeout_seconds: (True, ""))
    monkeypatch.setattr(llm_preflight, "_check_structured", lambda profile, timeout_seconds: (False, "json failed"))
    profile = LLMProfile(name="p1", provider="openai", model="gpt-5.2", enabled=True)
    config = LLMRoutingConfig(
        profiles={"p1": profile},
        routing={"plan_maker": RoleRoute(primary="p1", max_retry=1)},
    )

    report = run_llm_preflight(config)
    assert report["overall_status"] == "degraded"
    assert report["route_issues"] == []


def test_run_llm_preflight_requires_ready_for_structured_roles(monkeypatch):
    """Structured roles must be fully ready; degraded should not pass."""
    from agentic_data_scientist.core import llm_preflight

    monkeypatch.setattr(llm_preflight, "_check_basic", lambda profile, timeout_seconds: (True, ""))
    monkeypatch.setattr(llm_preflight, "_check_structured", lambda profile, timeout_seconds: (False, "json failed"))
    profile = LLMProfile(name="p1", provider="openai", model="gpt-5.2", enabled=True)
    config = LLMRoutingConfig(
        profiles={"p1": profile},
        routing={"plan_parser": RoleRoute(primary="p1", max_retry=1)},
    )

    report = run_llm_preflight(config)
    assert report["overall_status"] == "unavailable"
    assert report["route_issues"]


def test_format_preflight_report_contains_sections():
    """Formatter should include profile and route issue sections."""
    report = {
        "overall_status": "unavailable",
        "profiles": [
            {
                "profile": "p1",
                "status": "unavailable",
                "provider": "openrouter",
                "model": "m",
                "basic_ok": False,
                "structured_ok": False,
                "latency_ms": 0,
                "error": "fail",
            }
        ],
        "fallback_roles": ["plan_maker"],
        "route_issues": ["Role 'plan_maker' has no ready route"],
    }
    text = format_preflight_report(report)
    assert "LLM preflight overall: unavailable" in text
    assert "Profiles:" in text
    assert "Roles using fallback:" in text
    assert "Route issues:" in text
