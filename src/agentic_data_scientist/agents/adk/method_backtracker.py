"""Method backtracker: switches to standby method when critic recommends backtrack."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)


def should_backtrack(state: Dict[str, Any]) -> bool:
    critic_output = state.get(StateKeys.METHOD_CRITIC_OUTPUT)
    if not isinstance(critic_output, dict):
        return False
    if critic_output.get("issue_type") != "method_failure":
        return False
    if critic_output.get("recommendation") != "backtrack":
        return False

    standby = state.get(StateKeys.STANDBY_METHODS) or []
    if not standby:
        return False

    budget_data = state.get(StateKeys.INNOVATION_BUDGET)
    if budget_data:
        budget = InnovationBudget.from_dict(budget_data)
    else:
        budget = InnovationBudget()
    if budget.remaining("backtrack") <= 0:
        return False

    return True


def execute_backtrack(state: Dict[str, Any]) -> Dict[str, Any] | None:
    standby: List[Dict[str, Any]] = state.get(StateKeys.STANDBY_METHODS) or []
    if not standby:
        return None

    current_method = state.get(StateKeys.SELECTED_METHOD)
    critic_output = state.get(StateKeys.METHOD_CRITIC_OUTPUT) or {}

    budget_data = state.get(StateKeys.INNOVATION_BUDGET)
    budget = InnovationBudget.from_dict(budget_data) if budget_data else InnovationBudget()
    if not budget.consume("backtrack"):
        return None

    if isinstance(current_method, dict):
        current_method["status"] = "failed"
        current_method["rejection_reason"] = critic_output.get("explanation", "Method failure detected by critic")

    new_method = standby.pop(0)
    new_method["status"] = "selected"
    state[StateKeys.SELECTED_METHOD] = new_method
    state[StateKeys.STANDBY_METHODS] = standby
    state[StateKeys.INNOVATION_BUDGET] = budget.to_dict()

    history: List[Dict[str, Any]] = state.get(StateKeys.BACKTRACK_HISTORY) or []
    history.append(
        {
            "from_method": current_method.get("method_id", "unknown")
            if isinstance(current_method, dict)
            else "unknown",
            "to_method": new_method.get("method_id", "unknown"),
            "reason": critic_output.get("explanation", ""),
            "evidence": critic_output.get("evidence", []),
        }
    )
    state[StateKeys.BACKTRACK_HISTORY] = history

    return new_method


class MethodBacktrackerAgent(BaseAgent):
    """Switches to a standby method when the method critic recommends backtracking."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        innovation_mode = str(state.get(StateKeys.INNOVATION_MODE, "routine") or "routine").strip()
        if innovation_mode == "routine":
            logger.info("[MethodBacktracker] Routine mode — skipping")
            return

        if not should_backtrack(state):
            logger.info("[MethodBacktracker] Backtrack conditions not met — skipping")
            return

        new_method = execute_backtrack(state)
        if new_method is None:
            logger.warning("[MethodBacktracker] Backtrack execution failed")
            return

        logger.info(
            "[MethodBacktracker] Switched to method: %s (%s)",
            new_method.get("method_id"),
            new_method.get("title"),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Method backtrack: switched to {new_method.get('method_id')} "
                            f"({new_method.get('title', 'untitled')})"
                        )
                    )
                ],
            ),
        )

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for MethodBacktrackerAgent.")
