"""Plan candidate selector: learning-informed, advice-only ranking."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agentic_data_scientist.core.plan_learning import rank_plan_candidates
from agentic_data_scientist.core.state_contracts import StateKeys


logger = logging.getLogger(__name__)


def _is_enabled(raw: str, *, default: bool = False) -> bool:
    value = str(raw or "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "off", "no"}


def _parse_csv(raw: str) -> List[str]:
    values: List[str] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if token:
            values.append(token)
    return values


def _get_rollout_percent() -> int:
    raw = os.getenv("ADS_PLAN_SELECTOR_ROLLOUT_PERCENT", "100").strip()
    try:
        value = int(raw)
    except Exception:
        value = 100
    return max(0, min(100, value))


def _get_min_switch_margin() -> float:
    raw = os.getenv("ADS_PLAN_RANK_MIN_SWITCH_MARGIN", "0.12").strip()
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.12


def _is_intent_allowed(user_request: str) -> bool:
    patterns = _parse_csv(os.getenv("ADS_PLAN_SELECTOR_INTENT_REGEXES", ""))
    if not patterns:
        return True
    request = str(user_request or "")
    for pattern in patterns:
        try:
            if re.search(pattern, request, flags=re.IGNORECASE):
                return True
        except re.error:
            logger.warning("Invalid ADS_PLAN_SELECTOR_INTENT_REGEXES pattern: %s", pattern)
            continue
    return False


def _calc_rollout_bucket(ctx: InvocationContext, user_request: str) -> int:
    salt = os.getenv("ADS_PLAN_SELECTOR_ROLLOUT_SALT", "").strip()
    session = getattr(ctx, "session", None)
    session_id = (
        str(getattr(session, "id", "") or getattr(session, "session_id", "") or "").strip()
    )
    key = f"{session_id}|{user_request}|{salt}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _resolve_rollout(ctx: InvocationContext, user_request: str) -> Dict[str, Any]:
    if not _is_enabled(os.getenv("ADS_PLAN_SELECTOR_ENABLED", "false"), default=False):
        return {"enabled": False, "reason": "feature_disabled", "rollout_percent": 0, "bucket": None}

    if not _is_intent_allowed(user_request):
        return {"enabled": False, "reason": "intent_not_matched", "rollout_percent": 0, "bucket": None}

    rollout_percent = _get_rollout_percent()
    if rollout_percent <= 0:
        return {"enabled": False, "reason": "rollout_percent_zero", "rollout_percent": 0, "bucket": None}
    if rollout_percent >= 100:
        return {"enabled": True, "reason": "rollout_full", "rollout_percent": rollout_percent, "bucket": 0}

    bucket = _calc_rollout_bucket(ctx, user_request)
    enabled = bucket < rollout_percent
    reason = "rollout_bucket_enabled" if enabled else "rollout_bucket_holdout"
    return {
        "enabled": enabled,
        "reason": reason,
        "rollout_percent": rollout_percent,
        "bucket": bucket,
    }


def _extract_history_signals(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


class PlanCandidateSelectorAgent(BaseAgent):
    """Select final high-level plan from collected candidates using learning signals."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        current_plan = str(state.get(StateKeys.HIGH_LEVEL_PLAN, "") or "").strip()
        if not current_plan:
            return

        raw_candidates = state.get(StateKeys.PLAN_CANDIDATES, [])
        candidates: List[str] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                text = str(item or "").strip()
                if text:
                    candidates.append(text)

        if current_plan not in candidates:
            candidates.append(current_plan)

        # Dedupe while preserving order
        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        candidates = deduped

        baseline_index = 0
        if current_plan in candidates:
            baseline_index = candidates.index(current_plan)

        user_request = str(state.get(StateKeys.ORIGINAL_USER_INPUT, "") or "")
        rollout = _resolve_rollout(ctx, user_request)

        if len(candidates) <= 1:
            state[StateKeys.PLAN_SELECTION_TRACE] = {
                "selected_index": baseline_index,
                "baseline_index": baseline_index,
                "candidate_scores": [{"index": baseline_index, "score": 1.0}],
                "switch_applied": False,
                "reason": "single_candidate",
                "rollout": rollout,
            }
            return

        if not rollout.get("enabled", False):
            state[StateKeys.HIGH_LEVEL_PLAN] = candidates[baseline_index]
            state[StateKeys.PLAN_CANDIDATES] = candidates
            state[StateKeys.PLAN_SELECTION_TRACE] = {
                "selected_index": baseline_index,
                "baseline_index": baseline_index,
                "candidate_scores": [{"index": baseline_index, "score": 1.0}],
                "switch_applied": False,
                "reason": str(rollout.get("reason", "rollout_disabled")),
                "rollout": rollout,
            }
            return

        history_signals = _extract_history_signals(state.get(StateKeys.PLANNER_HISTORY_SIGNALS))
        ranking = rank_plan_candidates(
            user_request=user_request,
            candidates=candidates,
            history_signals=history_signals,
            baseline_index=baseline_index,
            min_switch_margin=_get_min_switch_margin(),
        )
        ranking["rollout"] = rollout

        selected_index = int(ranking.get("selected_index", baseline_index))
        selected_index = min(max(0, selected_index), len(candidates) - 1)
        selected_plan = candidates[selected_index]

        state[StateKeys.HIGH_LEVEL_PLAN] = selected_plan
        state[StateKeys.PLAN_SELECTION_TRACE] = ranking
        state[StateKeys.PLAN_CANDIDATES] = candidates

        reason = str(ranking.get("reason", "ranked"))
        switch_applied = bool(ranking.get("switch_applied", False))
        margin = float(ranking.get("margin_vs_baseline", 0.0) or 0.0)
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Plan selection (advice-only): "
                            f"candidates={len(candidates)}, "
                            f"switch_applied={switch_applied}, "
                            f"margin={margin:.3f}, "
                            f"reason={reason}."
                        )
                    )
                ],
            ),
        )

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for PlanCandidateSelectorAgent.")
