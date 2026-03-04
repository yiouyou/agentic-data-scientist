"""
Implementation loop agents for ADK system.

This module creates the coding and review agents that work together in an
iterative loop to implement and verify solutions.
"""

import logging

from google.adk.planners import BuiltInPlanner
from google.adk.tools.tool_context import CallbackContext
from google.genai import types

from agentic_data_scientist.agents.adk.event_compression import create_compression_callback
from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.review_confirmation import create_review_confirmation_agent
from agentic_data_scientist.agents.adk.utils import (
    CODING_MODEL_NAME,
    REVIEW_MODEL_NAME,
    get_generate_content_config,
    get_litellm_candidates_for_role,
    get_role_model_candidates,
)
from agentic_data_scientist.agents.coding_backends import (
    create_execution_agent,
    create_coding_agent,
    resolve_coding_backend_route,
)
from agentic_data_scientist.agents.workflow_execution import WorkflowExecutionAgent
from agentic_data_scientist.core.state_contracts import StateKeys
from agentic_data_scientist.prompts import load_prompt


logger = logging.getLogger(__name__)

# Configuration
TOOL_LOOP_LIMIT = 5
CODING_EVENT_LIMIT = 100
MAX_EVENTS_TO_KEEP = 20


def trim_history_to_recent_events(callback_context: CallbackContext, max_events: int = CODING_EVENT_LIMIT):
    """
    Simple callback to keep only the most recent events in history.

    This prevents token overflow by maintaining a fixed window of context.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context with session access
    max_events : int, optional
        Maximum number of events to keep (default: CODING_EVENT_LIMIT)
    """
    session = callback_context._invocation_context.session
    events = session.events

    logger.info(f"[DEBUG] trim_history_to_recent_events called: {len(events)} events, max={max_events}")

    if len(events) > max_events:
        # Keep only the most recent events
        events_to_remove = len(events) - max_events
        logger.info(f"[DEBUG] Trimming {events_to_remove} old events, keeping {max_events} most recent")

        # Remove oldest events
        for _ in range(events_to_remove):
            events.pop(0)

        logger.info(f"[DEBUG] After trimming: {len(events)} events remain")


def make_implementation_agents(working_dir: str, tools: list):
    """
    Create the implementation agents (coding + review + confirmation).

    Parameters
    ----------
    working_dir : str
        Working directory for the session
    tools : list
        List of tool functions available to agents

    Returns
    -------
    tuple
        (coding_agent, review_agent, review_confirmation_agent)
    """
    logger.info(f"[AgenticDS] Initializing implementation agents with {len(tools)} tools")

    # Create compression callback for coding agent
    coding_compression_callback = create_compression_callback(
        event_threshold=40,
        overlap_size=20,
    )
    coding_model_selection = get_role_model_candidates(
        role="execution_agent",
        default_primary=CODING_MODEL_NAME,
    )

    coding_backend_route = resolve_coding_backend_route(
        primary_profile=coding_model_selection.selected_profile,
        primary_model=coding_model_selection.primary_model,
        fallback_profile=coding_model_selection.fallback_profile,
        fallback_model=coding_model_selection.fallback_model,
        max_retry=coding_model_selection.max_retry,
    )
    if coding_backend_route.fallback_reason:
        logger.info(f"[AgenticDS] execution_agent fallback: {coding_backend_route.fallback_reason}")

    skill_execution_agent = create_coding_agent(
        executor=coding_backend_route.primary_executor,
        name="skill_execution_agent",
        description=(
            f"A coding agent that uses {coding_backend_route.primary_executor} to implement plans."
        ),
        working_dir=working_dir,
        model=coding_model_selection.primary_model,
        fallback_model=(
            coding_model_selection.fallback_model
            if coding_backend_route.fallback_enabled
            else None
        ),
        fallback_max_retries=(
            coding_model_selection.max_retry if coding_backend_route.fallback_enabled else 0
        ),
        routing_role="execution_agent",
        primary_profile_name=(
            coding_model_selection.selected_profile.name
            if coding_model_selection.selected_profile is not None
            else coding_model_selection.primary_model
        ),
        output_key=StateKeys.IMPLEMENTATION_SUMMARY,
    )

    workflow_execution_agent = WorkflowExecutionAgent(
        name="workflow_execution_agent",
        description="Executes fixed workflow manifests via local_cli or remote_api adapters.",
        working_dir=working_dir,
        output_key=StateKeys.IMPLEMENTATION_SUMMARY,
    )

    coding_agent = create_execution_agent(
        name="execution_agent",
        description="Routes execution to skill executor or workflow executor based on stage metadata.",
        skill_executor=skill_execution_agent,
        workflow_executor=workflow_execution_agent,
        after_agent_callback=coding_compression_callback,
    )

    # Review Agent - Uses ADK with loop detection
    logger.info("[AgenticDS] Creating review agent")

    # Load review prompt
    review_prompt = load_prompt("coding_review")

    # Create compression callback for review agent
    review_compression_callback = create_compression_callback(
        event_threshold=40,
        overlap_size=20,
    )

    review_model, review_fallback_model, review_model_selection = get_litellm_candidates_for_role(
        role="review_agent",
        default_model_name=REVIEW_MODEL_NAME,
    )

    review_agent = LoopDetectionAgent(
        name="review_agent",
        description="Reviews implementation and provides feedback or approval.",
        instruction=review_prompt,
        model=review_model,
        fallback_model=review_fallback_model,
        fallback_max_retries=review_model_selection.max_retry,
        routing_role="review_agent",
        primary_profile_name=(
            review_model_selection.selected_profile.name
            if review_model_selection.selected_profile is not None
            else review_model_selection.primary_model
        ),
        tools=tools,
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        ),
        generate_content_config=get_generate_content_config(temperature=0.0),
        output_key=StateKeys.REVIEW_FEEDBACK,
        include_contents="none",
        after_agent_callback=review_compression_callback,
    )

    logger.info("[AgenticDS] Implementation agents created successfully")

    # Return coding agent, review agent, AND review confirmation agent
    return (
        coding_agent,
        review_agent,
        create_review_confirmation_agent(
            auto_exit_on_completion=True,
            prompt_name="implementation_review_confirmation",
            model=review_model,
            fallback_model=review_fallback_model,
            fallback_max_retries=review_model_selection.max_retry,
            routing_role="review_agent",
            primary_profile_name=(
                review_model_selection.selected_profile.name
                if review_model_selection.selected_profile is not None
                else review_model_selection.primary_model
            ),
        ),
    )
