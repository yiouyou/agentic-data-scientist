"""Method critic: diagnoses whether stage failures stem from execution or method issues."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.genai import types

from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)

_VALID_ISSUE_TYPES = {"execution_failure", "method_failure"}
_VALID_RECOMMENDATIONS = {"retry", "backtrack", "continue"}


def parse_critic_output(text: str) -> Dict[str, Any] | None:
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


def validate_critic_output(output: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if output.get("issue_type") not in _VALID_ISSUE_TYPES:
        errors.append(f"Invalid issue_type: {output.get('issue_type')}")
    if output.get("recommendation") not in _VALID_RECOMMENDATIONS:
        errors.append(f"Invalid recommendation: {output.get('recommendation')}")
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        errors.append(f"confidence must be a float in [0, 1], got {confidence}")
    if not isinstance(output.get("evidence"), list):
        errors.append("evidence must be a list")
    if not isinstance(output.get("explanation"), str) or not output.get("explanation"):
        errors.append("explanation must be a non-empty string")
    return errors


def make_default_critic_output() -> Dict[str, Any]:
    return {
        "issue_type": "execution_failure",
        "confidence": 0.5,
        "evidence": [],
        "recommendation": "retry",
        "explanation": "Unable to diagnose — defaulting to retry.",
    }


def sanitize_critic_output(output: Dict[str, Any]) -> Dict[str, Any]:
    if output.get("issue_type") not in _VALID_ISSUE_TYPES:
        output["issue_type"] = "execution_failure"
    if output.get("recommendation") not in _VALID_RECOMMENDATIONS:
        output["recommendation"] = "retry"
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)):
        output["confidence"] = 0.5
    else:
        output["confidence"] = max(0.0, min(1.0, float(confidence)))
    if not isinstance(output.get("evidence"), list):
        output["evidence"] = []
    if not isinstance(output.get("explanation"), str) or not output.get("explanation"):
        output["explanation"] = "No explanation provided."
    return output


class MethodCriticAgent(BaseAgent):
    """Diagnoses stage failures as execution-level or method-level issues."""

    model: Any = ""
    fallback_model: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        innovation_mode = str(state.get(StateKeys.INNOVATION_MODE, "routine") or "routine").strip()
        if innovation_mode == "routine":
            logger.info("[MethodCritic] Routine mode — skipping")
            return

        review_decision = str(
            state.get(StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION, "approved") or "approved"
        ).strip()
        if review_decision == "approved":
            logger.info("[MethodCritic] Review approved — no critique needed")
            return

        attempts = self._get_attempts(state)
        if attempts < 2:
            logger.info("[MethodCritic] Only %d attempts — too early to critique", attempts)
            return

        stage_description = self._get_stage_description(state)
        review_feedback = str(state.get(StateKeys.REVIEW_FEEDBACK, "") or "")
        implementation_summary = str(state.get(StateKeys.IMPLEMENTATION_SUMMARY, "") or "")
        selected_method = state.get(StateKeys.SELECTED_METHOD)
        selected_method_str = json.dumps(selected_method, ensure_ascii=False) if selected_method else "{}"

        from agentic_data_scientist.agents.adk.utils import _build_litellm
        from agentic_data_scientist.prompts import load_prompt

        llm = self.model if hasattr(self.model, "generate_content_async") else _build_litellm(self.model)

        raw_prompt = load_prompt("method_critic")
        replacements = {
            "stage_description": stage_description,
            "review_feedback": review_feedback,
            "attempts": str(attempts),
            "selected_method": selected_method_str,
            "implementation_summary": implementation_summary,
        }
        for key, val in replacements.items():
            raw_prompt = raw_prompt.replace(f"{{{key}?}}", val)
            raw_prompt = raw_prompt.replace(f"{{{key}}}", val)

        logger.info("[MethodCritic] Diagnosing failure (attempts=%d)", attempts)

        try:
            llm_request = LlmRequest(
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=raw_prompt)])],
                config=types.GenerateContentConfig(temperature=0.3),
            )
            response_text = ""
            async for llm_response in llm.generate_content_async(llm_request):
                if llm_response.content and llm_response.content.parts:
                    for part in llm_response.content.parts:
                        if part.text:
                            response_text += part.text
        except Exception as e:
            logger.warning("[MethodCritic] LLM call failed: %s", e)
            state[StateKeys.METHOD_CRITIC_OUTPUT] = make_default_critic_output()
            return

        parsed = parse_critic_output(response_text)
        if parsed is None:
            logger.warning("[MethodCritic] Failed to parse LLM response")
            state[StateKeys.METHOD_CRITIC_OUTPUT] = make_default_critic_output()
            return

        errors = validate_critic_output(parsed)
        if errors:
            logger.warning("[MethodCritic] Validation errors: %s — sanitizing", errors)
            parsed = sanitize_critic_output(parsed)

        state[StateKeys.METHOD_CRITIC_OUTPUT] = parsed

        logger.info(
            "[MethodCritic] Diagnosis: issue_type=%s, recommendation=%s, confidence=%.2f",
            parsed.get("issue_type"),
            parsed.get("recommendation"),
            parsed.get("confidence", 0),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Method critique: issue_type={parsed.get('issue_type')}, "
                            f"recommendation={parsed.get('recommendation')}, "
                            f"confidence={parsed.get('confidence')}"
                        )
                    )
                ],
            ),
        )

    @staticmethod
    def _get_attempts(state: Dict[str, Any]) -> int:
        stages = state.get(StateKeys.HIGH_LEVEL_STAGES) or []
        idx = state.get(StateKeys.CURRENT_STAGE_INDEX, 0) or 0
        if isinstance(stages, list) and 0 <= idx < len(stages):
            return stages[idx].get("attempts", 0)
        return 0

    @staticmethod
    def _get_stage_description(state: Dict[str, Any]) -> str:
        stages = state.get(StateKeys.HIGH_LEVEL_STAGES) or []
        idx = state.get(StateKeys.CURRENT_STAGE_INDEX, 0) or 0
        if isinstance(stages, list) and 0 <= idx < len(stages):
            stage = stages[idx]
            return stage.get("description", stage.get("title", ""))
        return str(state.get(StateKeys.CURRENT_STAGE, "") or "")

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for MethodCriticAgent.")
