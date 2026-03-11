"""Problem framer: analyzes user request into structured problem statement."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.genai import types

from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {"research_goal", "task_type", "recommended_mode"}
_VALID_TASK_TYPES = {"data_analysis", "modeling", "exploration", "discovery", "routine_processing"}
_VALID_MODES = {"routine", "hybrid", "innovation"}


def _parse_framed_problem(text: str) -> Dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text) - 1, start, -1):
                if text[end] == "}":
                    try:
                        parsed = json.loads(text[start : end + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
    return None


def validate_framed_problem(problem: Dict[str, Any]) -> list[str]:
    errors = []
    for field in _REQUIRED_FIELDS:
        if field not in problem or not problem[field]:
            errors.append(f"Missing required field: {field}")
    if problem.get("task_type") and problem["task_type"] not in _VALID_TASK_TYPES:
        errors.append(f"Invalid task_type: {problem['task_type']}")
    if problem.get("recommended_mode") and problem["recommended_mode"] not in _VALID_MODES:
        errors.append(f"Invalid recommended_mode: {problem['recommended_mode']}")
    for list_field in ("knowns", "unknowns", "contradictions", "complexity_signals"):
        if list_field in problem and not isinstance(problem[list_field], list):
            errors.append(f"{list_field} must be a list")
    return errors


def make_default_framed_problem(user_request: str) -> Dict[str, Any]:
    return {
        "research_goal": user_request[:200],
        "task_type": "routine_processing",
        "knowns": [],
        "unknowns": [],
        "contradictions": [],
        "complexity_signals": [],
        "recommended_mode": "routine",
    }


class ProblemFramerAgent(BaseAgent):
    """Analyzes user request into a structured problem statement via LLM."""

    model: Any = ""
    fallback_model: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        user_request = str(state.get(StateKeys.ORIGINAL_USER_INPUT, "") or "")
        if not user_request.strip():
            state[StateKeys.FRAMED_PROBLEM] = make_default_framed_problem("")
            return

        from agentic_data_scientist.agents.adk.utils import _build_litellm
        from agentic_data_scientist.prompts import load_prompt

        llm = self.model if hasattr(self.model, "generate_content_async") else _build_litellm(self.model)

        raw_prompt = load_prompt("problem_framer")
        raw_prompt = raw_prompt.replace("{original_user_input?}", user_request)
        raw_prompt = raw_prompt.replace("{original_user_input}", user_request)

        logger.info("[ProblemFramer] Analyzing user request")

        try:
            llm_request = LlmRequest(
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=raw_prompt)])],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            response_text = ""
            async for llm_response in llm.generate_content_async(llm_request):
                if llm_response.content and llm_response.content.parts:
                    for part in llm_response.content.parts:
                        if part.text:
                            response_text += part.text
        except Exception as e:
            logger.warning(f"[ProblemFramer] LLM call failed: {e}")
            state[StateKeys.FRAMED_PROBLEM] = make_default_framed_problem(user_request)
            return

        parsed = _parse_framed_problem(response_text)
        if parsed is None:
            logger.warning("[ProblemFramer] Failed to parse LLM response")
            state[StateKeys.FRAMED_PROBLEM] = make_default_framed_problem(user_request)
            return

        errors = validate_framed_problem(parsed)
        if errors:
            logger.warning(f"[ProblemFramer] Validation errors: {errors}")
            for list_field in ("knowns", "unknowns", "contradictions", "complexity_signals"):
                if not isinstance(parsed.get(list_field), list):
                    parsed[list_field] = []
            if parsed.get("task_type") not in _VALID_TASK_TYPES:
                parsed["task_type"] = "routine_processing"
            if parsed.get("recommended_mode") not in _VALID_MODES:
                parsed["recommended_mode"] = "routine"
            if not parsed.get("research_goal"):
                parsed["research_goal"] = user_request[:200]

        state[StateKeys.FRAMED_PROBLEM] = parsed

        logger.info(
            "[ProblemFramer] Result: task_type=%s, recommended_mode=%s, signals=%d, contradictions=%d",
            parsed.get("task_type"),
            parsed.get("recommended_mode"),
            len(parsed.get("complexity_signals", [])),
            len(parsed.get("contradictions", [])),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Problem analysis: task_type={parsed.get('task_type')}, "
                            f"recommended_mode={parsed.get('recommended_mode')}, "
                            f"complexity_signals={len(parsed.get('complexity_signals', []))}"
                        )
                    )
                ],
            ),
        )

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for ProblemFramerAgent.")
