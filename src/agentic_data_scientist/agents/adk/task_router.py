"""Task router: decides innovation mode from framed problem using rules engine.

Design principles:
  - Rules-first, LLM-fallback (near-zero latency for routine tasks).
  - CLI override (ADS_INNOVATION_MODE_OVERRIDE) always wins.
  - When rules are inconclusive, uses framed_problem.recommended_mode as tiebreaker.
  - LLM fallback is reserved for genuinely ambiguous cases (not yet implemented — deferred
    until real-world data shows how often the rules engine is inconclusive).

Priority order:
  1. CLI override  → forced mode
  2. No framed_problem → routine
  3. Rules engine  → deterministic decision
  4. recommended_mode tiebreaker
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)

_VALID_MODES = {"routine", "hybrid", "innovation"}


def get_cli_override() -> Optional[str]:
    """Return CLI override mode if user explicitly set --innovation-mode.

    The override is communicated via ``ADS_INNOVATION_MODE_OVERRIDE``.
    When the user passes ``--innovation-mode auto`` (the default), the override
    env var is **not** set — so auto-routing proceeds normally.
    When the user passes ``--innovation-mode routine|hybrid|innovation``,
    the env var is set to that value and this function returns it.
    """
    raw = os.getenv("ADS_INNOVATION_MODE_OVERRIDE", "").strip().lower()
    if raw in _VALID_MODES:
        return raw
    return None


def route_by_rules(framed_problem: Dict[str, Any]) -> Optional[str]:
    """Deterministic rules engine over framed-problem fields.

    Returns a mode string or ``None`` if the rules are inconclusive.

    Rules (evaluated top-to-bottom, first match wins):
      1. task_type == "routine_processing"                           → routine
      2. task_type == "discovery" OR contradictions is non-empty      → innovation
      3. task_type == "data_analysis" AND signals >= 2               → hybrid (capped)
      4. len(complexity_signals) >= 3                                 → innovation
      5. len(complexity_signals) >= 2                                 → hybrid
      6. task_type in ("modeling", "exploration") AND signals >= 1    → hybrid
      7. None (inconclusive)
    """
    task_type = str(framed_problem.get("task_type", "")).strip().lower()

    contradictions = framed_problem.get("contradictions") or []
    if not isinstance(contradictions, list):
        contradictions = []

    complexity_signals = framed_problem.get("complexity_signals") or []
    if not isinstance(complexity_signals, list):
        complexity_signals = []

    n_signals = len(complexity_signals)
    has_contradictions = len(contradictions) > 0

    if task_type == "routine_processing":
        return "routine"

    if task_type == "discovery" or has_contradictions:
        return "innovation"

    if task_type == "data_analysis" and n_signals >= 2:
        return "hybrid"

    if n_signals >= 3:
        return "innovation"

    if n_signals >= 2:
        return "hybrid"

    if task_type in ("modeling", "exploration") and n_signals >= 1:
        return "hybrid"

    return None


def resolve_mode(framed_problem: Optional[Dict[str, Any]]) -> str:
    """Full resolution pipeline: override → rules → tiebreaker → default.

    This is the pure-function core that ``TaskRouterAgent`` calls.
    Separated for easy unit-testing.
    """
    override = get_cli_override()
    if override is not None:
        logger.info("[TaskRouter] CLI override → %s", override)
        return override

    if not framed_problem or not isinstance(framed_problem, dict):
        logger.info("[TaskRouter] No framed_problem → routine")
        return "routine"

    rule_result = route_by_rules(framed_problem)
    if rule_result is not None:
        logger.info("[TaskRouter] Rules engine → %s", rule_result)
        return rule_result

    recommended = str(framed_problem.get("recommended_mode", "")).strip().lower()
    if recommended in _VALID_MODES:
        logger.info("[TaskRouter] Tiebreaker (recommended_mode) → %s", recommended)
        return recommended

    logger.info("[TaskRouter] Default fallback → routine")
    return "routine"


class TaskRouterAgent(BaseAgent):
    """Reads ``StateKeys.FRAMED_PROBLEM`` and writes ``StateKeys.INNOVATION_MODE``.

    This agent is purely deterministic (no LLM call).  It sits immediately
    after ``ProblemFramerAgent`` in the auto-routing workflow prefix.
    """

    model: Any = ""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        framed_problem = state.get(StateKeys.FRAMED_PROBLEM)
        resolved = resolve_mode(framed_problem)

        state[StateKeys.INNOVATION_MODE] = resolved

        logger.info(
            "[TaskRouter] Resolved mode=%s (task_type=%s, signals=%d, contradictions=%d)",
            resolved,
            (framed_problem or {}).get("task_type", "N/A"),
            len((framed_problem or {}).get("complexity_signals", [])),
            len((framed_problem or {}).get("contradictions", [])),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=f"Task routing: innovation_mode={resolved}")],
            ),
        )

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for TaskRouterAgent.")
