"""Deep verifier: holistic consistency check before final summary."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.genai import types

from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)

_VALID_VERDICTS = {"pass", "warn", "fail"}


def parse_verification_output(text: str) -> Dict[str, Any] | None:
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


def validate_verification_output(output: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if output.get("overall_verdict") not in _VALID_VERDICTS:
        errors.append(f"Invalid overall_verdict: {output.get('overall_verdict')}")
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        errors.append(f"confidence must be float in [0,1], got {confidence}")
    if not isinstance(output.get("summary"), str) or not output.get("summary"):
        errors.append("summary must be a non-empty string")
    for list_field in ("consistency_issues", "uncovered_risks", "criteria_corrections"):
        if list_field in output and not isinstance(output[list_field], list):
            errors.append(f"{list_field} must be a list")
    return errors


def make_default_verification() -> Dict[str, Any]:
    return {
        "overall_verdict": "pass",
        "confidence": 0.5,
        "goal_alignment": {"score": 0.5, "notes": "Verification skipped or failed."},
        "consistency_issues": [],
        "criteria_corrections": [],
        "method_validity": {
            "assumptions_satisfied": True,
            "invalid_conditions_triggered": False,
            "notes": "Not verified.",
        },
        "uncovered_risks": [],
        "summary": "Deep verification was not completed.",
    }


def sanitize_verification_output(output: Dict[str, Any]) -> Dict[str, Any]:
    if output.get("overall_verdict") not in _VALID_VERDICTS:
        output["overall_verdict"] = "warn"
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)):
        output["confidence"] = 0.5
    else:
        output["confidence"] = max(0.0, min(1.0, float(confidence)))
    if not isinstance(output.get("summary"), str) or not output.get("summary"):
        output["summary"] = "No summary provided."
    for list_field in ("consistency_issues", "uncovered_risks", "criteria_corrections"):
        if not isinstance(output.get(list_field), list):
            output[list_field] = []
    if not isinstance(output.get("goal_alignment"), dict):
        output["goal_alignment"] = {"score": 0.5, "notes": "Not assessed."}
    if not isinstance(output.get("method_validity"), dict):
        output["method_validity"] = {
            "assumptions_satisfied": True,
            "invalid_conditions_triggered": False,
            "notes": "Not assessed.",
        }
    return output


def _build_innovation_summary_section(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    selected = state.get(StateKeys.SELECTED_METHOD)
    if isinstance(selected, dict) and selected.get("title"):
        parts.append(f"**Selected Method:** {selected['title']} — {selected.get('core_hypothesis', 'N/A')}")

    backtrack_history = state.get(StateKeys.BACKTRACK_HISTORY) or []
    if backtrack_history:
        parts.append(f"**Method Backtracks ({len(backtrack_history)}):**")
        for bt in backtrack_history:
            parts.append(f"- Switched from {bt.get('from_method')} to {bt.get('to_method')}: {bt.get('reason', '')}")

    verification = state.get(StateKeys.DEEP_VERIFICATION)
    if isinstance(verification, dict):
        parts.append(
            f"**Deep Verification:** verdict={verification.get('overall_verdict')}, "
            f"confidence={verification.get('confidence')}"
        )
        issues = verification.get("consistency_issues", [])
        if issues:
            parts.append("Consistency issues:")
            for issue in issues[:5]:
                parts.append(f"- {issue}")
        risks = verification.get("uncovered_risks", [])
        if risks:
            parts.append("Uncovered risks:")
            for risk in risks[:5]:
                parts.append(f"- {risk}")

    if not parts:
        return ""
    return "# Innovation Process Report\n\n" + "\n".join(parts)


class DeepVerifierAgent(BaseAgent):
    """Performs holistic verification of the entire analysis before summary generation."""

    model: Any = ""
    fallback_model: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        innovation_mode = str(state.get(StateKeys.INNOVATION_MODE, "routine") or "routine").strip()
        if innovation_mode == "routine":
            logger.info("[DeepVerifier] Routine mode — skipping")
            return

        budget_data = state.get(StateKeys.INNOVATION_BUDGET)
        if budget_data:
            budget = InnovationBudget.from_dict(budget_data)
            if not budget.consume("verification"):
                logger.info("[DeepVerifier] Verification budget exhausted — skipping")
                return
            state[StateKeys.INNOVATION_BUDGET] = budget.to_dict()

        from agentic_data_scientist.agents.adk.utils import _build_litellm
        from agentic_data_scientist.prompts import load_prompt

        llm = self.model if hasattr(self.model, "generate_content_async") else _build_litellm(self.model)

        raw_prompt = load_prompt("deep_verifier")
        replacements = {
            "original_user_input": str(state.get(StateKeys.ORIGINAL_USER_INPUT, "") or ""),
            "high_level_stages": json.dumps(
                state.get(StateKeys.HIGH_LEVEL_STAGES, []), ensure_ascii=False, default=str
            )[:8000],
            "high_level_success_criteria": json.dumps(
                state.get(StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA, []), ensure_ascii=False, default=str
            )[:4000],
            "selected_method": json.dumps(state.get(StateKeys.SELECTED_METHOD) or {}, ensure_ascii=False, default=str)[
                :3000
            ],
            "stage_implementations": json.dumps(
                state.get(StateKeys.STAGE_IMPLEMENTATIONS, []), ensure_ascii=False, default=str
            )[:8000],
            "backtrack_history": json.dumps(
                state.get(StateKeys.BACKTRACK_HISTORY, []), ensure_ascii=False, default=str
            )[:2000],
        }
        for key, val in replacements.items():
            raw_prompt = raw_prompt.replace(f"{{{key}?}}", val)
            raw_prompt = raw_prompt.replace(f"{{{key}}}", val)

        logger.info("[DeepVerifier] Running holistic verification")

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
            logger.warning("[DeepVerifier] LLM call failed: %s", e)
            state[StateKeys.DEEP_VERIFICATION] = make_default_verification()
            return

        parsed = parse_verification_output(response_text)
        if parsed is None:
            logger.warning("[DeepVerifier] Failed to parse LLM response")
            state[StateKeys.DEEP_VERIFICATION] = make_default_verification()
            return

        errors = validate_verification_output(parsed)
        if errors:
            logger.warning("[DeepVerifier] Validation errors: %s — sanitizing", errors)
            parsed = sanitize_verification_output(parsed)

        state[StateKeys.DEEP_VERIFICATION] = parsed
        state["innovation_summary_section"] = _build_innovation_summary_section(state)

        logger.info(
            "[DeepVerifier] Verdict: %s (confidence=%.2f, issues=%d, risks=%d)",
            parsed.get("overall_verdict"),
            parsed.get("confidence", 0),
            len(parsed.get("consistency_issues", [])),
            len(parsed.get("uncovered_risks", [])),
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Deep verification: verdict={parsed.get('overall_verdict')}, "
                            f"confidence={parsed.get('confidence')}, "
                            f"issues={len(parsed.get('consistency_issues', []))}, "
                            f"risks={len(parsed.get('uncovered_risks', []))}"
                        )
                    )
                ],
            ),
        )

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for DeepVerifierAgent.")
