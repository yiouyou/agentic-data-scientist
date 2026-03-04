"""Unit tests for LLM circuit breaker behavior."""

import time

from agentic_data_scientist.core.llm_circuit_breaker import (
    LLMCircuitBreaker,
    is_retryable_llm_error,
)


def test_is_retryable_llm_error_detects_provider_markers():
    assert is_retryable_llm_error(Exception("Rate limit exceeded")) is True
    assert is_retryable_llm_error(Exception("provider timeout")) is True
    assert is_retryable_llm_error(Exception("syntax error in output")) is False


def test_circuit_opens_after_threshold_and_closes_after_cooldown(monkeypatch):
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_MAX_COOLDOWN_SECONDS", "600")

    breaker = LLMCircuitBreaker()
    role = "plan_maker"
    profile = "us_openai_reasoning"

    breaker.record_retryable_failure(role, profile, Exception("rate limit"))
    assert breaker.should_force_fallback(role, profile) is False

    breaker.record_retryable_failure(role, profile, Exception("timeout"))
    assert breaker.should_force_fallback(role, profile) is True

    # Simulate cooldown expiry.
    breaker._states[(role, profile)].open_until_ts = time.time() - 1.0
    assert breaker.should_force_fallback(role, profile) is False


def test_record_success_resets_open_circuit(monkeypatch):
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "1")
    breaker = LLMCircuitBreaker()
    role = "review_agent"
    profile = "cn_deepseek_chat"

    breaker.record_retryable_failure(role, profile, Exception("provider unavailable"))
    assert breaker.should_force_fallback(role, profile) is True

    breaker.record_success(role, profile)
    assert breaker.should_force_fallback(role, profile) is False


def test_disabled_circuit_never_forces_fallback(monkeypatch):
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_ENABLED", "false")
    breaker = LLMCircuitBreaker()
    role = "summary_agent"
    profile = "cn_qwen_reasoning"

    breaker.record_retryable_failure(role, profile, Exception("rate limit"))
    assert breaker.should_force_fallback(role, profile) is False
