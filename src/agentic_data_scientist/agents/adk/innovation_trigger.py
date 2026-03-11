"""Programmatic innovation trigger detection.

All detectors are pure functions — zero LLM cost.  The module is designed to
be called from ``StageOrchestratorAgent`` after the criteria_checker and before
the stage_reflector, only when ``innovation_mode != "routine"``.

Detection dimensions:
  - mediocre_review:     review approved but with reservations (keyword scan)
  - excessive_retries:   stage attempts exceeded threshold
  - criteria_stagnation: criteria-met count unchanged for N consecutive stages
  - verifier_warnings:   programmatic verifier verdict == "warn"
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MEDIOCRE_KEYWORDS: re.Pattern = re.compile(
    r"\b("
    r"partially|somewhat|marginal|borderline|acceptable\s+but|"
    r"could\s+be\s+improved|minor\s+issues?|with\s+reservations?|"
    r"not\s+ideal|barely|passable|adequate\s+but|caveat"
    r")\b",
    re.IGNORECASE,
)

_SIGNAL_WEIGHTS: Dict[str, float] = {
    "mediocre_review": 0.25,
    "excessive_retries": 0.35,
    "criteria_stagnation": 0.25,
    "verifier_warnings": 0.15,
}


def detect_mediocre_review(review_text: Optional[str]) -> bool:
    if not review_text:
        return False
    return bool(_MEDIOCRE_KEYWORDS.search(review_text))


def detect_excessive_retries(attempts: int, threshold: int = 2) -> bool:
    return attempts > threshold


def detect_criteria_stagnation(
    criteria_met_history: List[int],
    stagnation_window: int = 2,
) -> bool:
    if len(criteria_met_history) < stagnation_window + 1:
        return False
    tail = criteria_met_history[-stagnation_window:]
    return all(v <= tail[0] for v in tail[1:])


def detect_verifier_warnings(verifier_summary: Optional[str]) -> bool:
    if not verifier_summary:
        return False
    lower = verifier_summary.lower()
    return "verdict: warn" in lower or '"verdict": "warn"' in lower


def compute_trigger_result(
    *,
    review_text: Optional[str] = None,
    attempts: int = 0,
    retry_threshold: int = 2,
    criteria_met_history: Optional[List[int]] = None,
    stagnation_window: int = 2,
    verifier_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all detectors and return the aggregated trigger result.

    Returns a dict matching the ``StateKeys.INNOVATION_TRIGGER`` schema::

        {
            "triggered": bool,
            "signals": [...],
            "strength": float,          # 0.0 – 1.0
            "recommended_action": str,  # none | escalate_review | consider_method_switch
        }
    """
    signals: List[str] = []

    if detect_mediocre_review(review_text):
        signals.append("mediocre_review")

    if detect_excessive_retries(attempts, threshold=retry_threshold):
        signals.append("excessive_retries")

    if criteria_met_history is not None and detect_criteria_stagnation(
        criteria_met_history, stagnation_window=stagnation_window
    ):
        signals.append("criteria_stagnation")

    if detect_verifier_warnings(verifier_summary):
        signals.append("verifier_warnings")

    strength = round(sum(_SIGNAL_WEIGHTS.get(s, 0.0) for s in signals), 4)
    triggered = len(signals) > 0

    if strength >= 0.5:
        action = "consider_method_switch"
    elif triggered:
        action = "escalate_review"
    else:
        action = "none"

    result: Dict[str, Any] = {
        "triggered": triggered,
        "signals": signals,
        "strength": strength,
        "recommended_action": action,
    }

    logger.info(
        "[InnovationTrigger] triggered=%s signals=%s strength=%.2f action=%s",
        triggered,
        signals,
        strength,
        action,
    )

    return result
