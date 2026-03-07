"""
Review Confirmation Agent.

This module provides a specialized agent that determines whether to exit a review
loop based on the review feedback. It parses JSON from the model's text output
(without relying on provider-specific response_format support) and does not have
access to any tools.
"""

import json
import logging
import re
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field

from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.utils import REVIEW_MODEL, get_generate_content_config
from agentic_data_scientist.core.state_contracts import review_confirmation_decision_key
from agentic_data_scientist.prompts import load_prompt


logger = logging.getLogger(__name__)


def _parse_decision_from_text(raw: str) -> Optional[dict]:
    """Extract {exit, reason} dict from model text that may contain markdown fences or preamble."""
    text = raw.strip()
    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Try to find the first JSON object in the text
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None
    if "exit" not in parsed:
        return None
    return {"exit": bool(parsed["exit"]), "reason": str(parsed.get("reason", ""))}


def _create_clear_decision_callback(state_key: str):
    """
    Factory function to create a before-agent callback that clears stale decisions.

    This callback clears any previous decision from the state before the agent runs,
    preventing state pollution from previous invocations or other agents.

    Parameters
    ----------
    state_key : str
        The state key to clear

    Returns
    -------
    Callable
        A callback function that clears the specified state key
    """

    def clear_decision_callback(callback_context: CallbackContext):
        """Clear stale decision from state before agent runs."""
        ctx = callback_context._invocation_context
        state = ctx.session.state

        if state_key in state:
            old_value = state[state_key]
            del state[state_key]
            logger.debug(f"[ReviewConfirmation] Cleared stale decision from state key '{state_key}': {old_value}")
        else:
            logger.debug(f"[ReviewConfirmation] No stale decision to clear for key '{state_key}'")

    return clear_decision_callback


def _create_exit_loop_callback(state_key: str):
    """
    Factory function to create an after-agent callback that conditionally exits the loop.

    This callback is invoked after the agent completes. It reads the agent's
    structured output from the specified state key and only sets the escalate
    flag if the agent decided to exit.

    To ensure the escalate flag is properly propagated, this callback returns
    an empty Content object, which triggers event creation in ADK's
    _handle_after_agent_callback method.

    Parameters
    ----------
    state_key : str
        The state key to read the decision from

    Returns
    -------
    Callable
        A callback function that reads from the specified state key and conditionally escalates
    """

    def exit_loop_callback(callback_context: CallbackContext):
        """
        After-agent callback that conditionally exits the loop by escalating.

        Parameters
        ----------
        callback_context : CallbackContext
            The callback context with invocation context access

        Returns
        -------
        Optional[types.Content]
            Empty content to trigger event creation when exiting the loop
        """
        ctx = callback_context._invocation_context
        state = ctx.session.state

        raw_decision = state.get(state_key)

        if not raw_decision:
            logger.warning(f"[ReviewConfirmation] No decision found in state key '{state_key}' - not exiting loop")
            return None

        # Without output_schema, ADK stores raw text in state.
        # Parse JSON from text; also accept pre-parsed dicts for backward compat.
        if isinstance(raw_decision, dict):
            decision = raw_decision
        elif isinstance(raw_decision, str):
            decision = _parse_decision_from_text(raw_decision)
            if decision is None:
                logger.error(f"[ReviewConfirmation] Could not parse JSON from text in '{state_key}' - not exiting loop")
                return None
        else:
            logger.error(
                f"[ReviewConfirmation] Unexpected decision type in '{state_key}': {type(raw_decision)} - not exiting loop"
            )
            return None

        # Check if the agent decided to exit
        should_exit = decision.get("exit", False)
        reason = decision.get("reason", "No reason provided")

        if should_exit:
            logger.info(f"[ReviewConfirmation] Exiting loop (key='{state_key}') - Reason: {reason}")
            # Set escalate flag on the event_actions
            if hasattr(callback_context, '_event_actions') and callback_context._event_actions:
                callback_context._event_actions.escalate = True
            else:
                logger.warning("[ReviewConfirmation] No event_actions available - cannot escalate")
                return None

            # Return empty content to trigger event creation with the escalate flag
            # This ensures NonEscalatingLoopAgent receives the escalate signal
            return types.Content(role="model", parts=[])
        else:
            logger.info(f"[ReviewConfirmation] Continuing loop (key='{state_key}') - Reason: {reason}")
            return None

    return exit_loop_callback


# Output schema for review confirmation (Pydantic BaseModel)
class ReviewConfirmationOutput(BaseModel):
    """Schema for review confirmation decision."""

    exit: bool = Field(
        description="Whether to exit the review loop. True if implementation is approved, False if more work is needed."
    )
    reason: str = Field(description="Brief explanation of the decision to exit or continue.")


# Keep for backwards compatibility
REVIEW_CONFIRMATION_OUTPUT_SCHEMA = ReviewConfirmationOutput


def create_review_confirmation_agent(
    auto_exit_on_completion: bool = False,
    prompt_name: str = "plan_review_confirmation",
    model: Any = REVIEW_MODEL,
    fallback_model: Optional[Any] = None,
    fallback_max_retries: int = 1,
    routing_role: str = "",
    primary_profile_name: str = "",
) -> LoopDetectionAgent:
    """
    Create a review confirmation agent with structured output.

    This agent analyzes review feedback and determines whether the review loop
    should exit. It uses structured output (output_schema) to ensure consistent
    JSON responses and does not have access to any tools.

    Each agent instance uses a unique state key based on the prompt_name to prevent
    state pollution between different review confirmation agents (e.g., plan review
    vs implementation review).

    Parameters
    ----------
    auto_exit_on_completion : bool, optional
        If True, automatically exit the loop after agent completion by escalating.
        This uses an after_agent_callback to set escalate=True. Defaults to False.
    prompt_name : str, optional
        Name of the prompt file to load (default: "plan_review_confirmation").
        This is also used to generate a unique state key for this agent instance.
    model : Any, optional
        Model instance or model identifier to use for this confirmation agent.
    fallback_model : Any, optional
        Optional fallback model instance/identifier used for one-shot retry.
    fallback_max_retries : int, optional
        Maximum fallback retries. Current execution path supports 0 (disabled) or 1 (enabled).
    routing_role : str, optional
        Routing role identifier for per-role circuit breaker state.
    primary_profile_name : str, optional
        Primary profile identifier for per-role circuit breaker state.

    Returns
    -------
    LoopDetectionAgent
        The configured review confirmation agent

    Examples
    --------
    >>> agent = create_review_confirmation_agent()
    >>> # Agent will output structured JSON like:
    >>> # {"exit": true, "reason": "All issues resolved"}
    >>> # or
    >>> # {"exit": false, "reason": "Critical bugs remain"}
    >>>
    >>> # With auto-exit enabled:
    >>> agent = create_review_confirmation_agent(auto_exit_on_completion=True)
    >>> # Agent will automatically exit the loop after completion

    Notes
    -----
    The agent uses a unique state key derived from the prompt_name to prevent
    cross-contamination between different review confirmation agents. For example:
    - "plan_review_confirmation" -> state key: "plan_review_confirmation_decision"
    - "implementation_review_confirmation" -> state key: "implementation_review_confirmation_decision"

    A before_agent_callback is used to clear any stale decisions before the agent runs,
    providing defense-in-depth against state pollution.
    """
    logger.info(f"[AgenticDS] Creating review confirmation agent (prompt={prompt_name})")

    instruction = load_prompt(prompt_name)

    # Create unique state key per agent instance to prevent cross-contamination
    state_key = review_confirmation_decision_key(prompt_name)
    logger.debug(f"[AgenticDS] Using state key: {state_key}")

    # Create agent-specific callbacks using factory functions
    # These closures capture the state_key for this specific agent instance
    before_callback = _create_clear_decision_callback(state_key)
    after_callback = _create_exit_loop_callback(state_key) if auto_exit_on_completion else None

    agent = LoopDetectionAgent(
        name=f"{prompt_name}_agent",
        model=model,
        fallback_model=fallback_model,
        fallback_max_retries=max(0, int(fallback_max_retries)),
        routing_role=routing_role,
        primary_profile_name=primary_profile_name,
        description="Determines whether to exit the review loop based on implementation status.",
        instruction=instruction,
        tools=[],  # No tools - structured output only
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        ),
        generate_content_config=get_generate_content_config(temperature=0.0),
        # output_schema omitted: some providers (DeepSeek) reject response_format
        output_key=state_key,
        before_agent_callback=before_callback,  # Clear stale decisions before agent runs
        after_agent_callback=after_callback,  # Conditionally escalate after agent completes
    )

    logger.info(
        f"[AgenticDS] Review confirmation agent created successfully "
        f"(prompt={prompt_name}, state_key={state_key}, auto_exit_on_completion={auto_exit_on_completion})"
    )

    return agent
