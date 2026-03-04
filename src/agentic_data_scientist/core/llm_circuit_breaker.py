"""In-memory LLM circuit breaker for role/profile routing stability."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple


logger = logging.getLogger(__name__)


RETRYABLE_ERROR_MARKERS = (
    "rate limit",
    "quota",
    "not found",
    "provider",
    "authentication",
    "unauthorized",
    "timeout",
    "timed out",
    "service unavailable",
    "model",
)


def is_retryable_llm_error(error: Exception) -> bool:
    """Return whether an exception appears retryable for provider/model health."""
    message = str(error).lower()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


@dataclass
class CircuitState:
    """Mutable circuit breaker state for one (role, profile) key."""

    consecutive_failures: int = 0
    open_until_ts: float = 0.0
    cooldown_seconds: float = 120.0
    last_error: str = ""


class LLMCircuitBreaker:
    """Role/profile scoped in-memory circuit breaker with cooldown backoff."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[Tuple[str, str], CircuitState] = {}

    @property
    def enabled(self) -> bool:
        raw = os.getenv("LLM_CIRCUIT_BREAKER_ENABLED", "true").strip().lower()
        return raw not in {"0", "false", "off", "no"}

    @property
    def failure_threshold(self) -> int:
        value = int(os.getenv("LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "2"))
        return max(1, value)

    @property
    def base_cooldown_seconds(self) -> float:
        value = float(os.getenv("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "120"))
        return max(1.0, value)

    @property
    def max_cooldown_seconds(self) -> float:
        value = float(os.getenv("LLM_CIRCUIT_BREAKER_MAX_COOLDOWN_SECONDS", "1800"))
        return max(self.base_cooldown_seconds, value)

    def _key(self, role: str, profile: str) -> Tuple[str, str]:
        return (role.strip(), profile.strip())

    def should_force_fallback(self, role: str, profile: str) -> bool:
        """Return whether the primary model should be bypassed temporarily."""
        if not self.enabled:
            return False
        if not role or not profile:
            return False

        key = self._key(role, profile)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            now = time.time()
            return state.open_until_ts > now

    def record_success(self, role: str, profile: str) -> None:
        """Record a successful primary call and close/reset its circuit."""
        if not self.enabled:
            return
        if not role or not profile:
            return

        key = self._key(role, profile)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                self._states[key] = CircuitState(
                    consecutive_failures=0,
                    open_until_ts=0.0,
                    cooldown_seconds=self.base_cooldown_seconds,
                )
                return
            state.consecutive_failures = 0
            state.open_until_ts = 0.0
            state.cooldown_seconds = self.base_cooldown_seconds
            state.last_error = ""

    def record_retryable_failure(self, role: str, profile: str, error: Exception) -> None:
        """Record retryable failure and open circuit if threshold is reached."""
        if not self.enabled:
            return
        if not role or not profile:
            return

        key = self._key(role, profile)
        now = time.time()

        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = CircuitState(cooldown_seconds=self.base_cooldown_seconds)
                self._states[key] = state

            # If currently open, keep it open; do not inflate counters repeatedly.
            if state.open_until_ts > now:
                return

            state.consecutive_failures += 1
            state.last_error = str(error)[:300]
            if state.consecutive_failures < self.failure_threshold:
                return

            cooldown = state.cooldown_seconds or self.base_cooldown_seconds
            state.open_until_ts = now + cooldown
            state.consecutive_failures = 0
            state.cooldown_seconds = min(cooldown * 2.0, self.max_cooldown_seconds)

            logger.warning(
                f"[AgenticDS] Circuit opened for role='{role}', profile='{profile}' "
                f"for {cooldown:.1f}s due to repeated retryable failures."
            )


_GLOBAL_BREAKER = LLMCircuitBreaker()


def get_llm_circuit_breaker() -> LLMCircuitBreaker:
    """Return the process-global breaker instance."""
    return _GLOBAL_BREAKER
