"""Plan instantiator: converts a selected Method Card into a concrete plan."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.genai import types

from agentic_data_scientist.core.method_card import method_card_summary
from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)


class PlanInstantiatorAgent(BaseAgent):
    """Converts a selected Method Card into a high-level plan via LLM."""

    model: Any = ""
    fallback_model: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        selected_method = state.get(StateKeys.SELECTED_METHOD)
        if not selected_method or not isinstance(selected_method, dict):
            logger.warning("[PlanInstantiator] No selected method found, skipping")
            return

        user_request = str(state.get(StateKeys.ORIGINAL_USER_INPUT, "") or "")
        standby_methods = state.get(StateKeys.STANDBY_METHODS, [])

        standby_summaries = []
        if isinstance(standby_methods, list):
            for m in standby_methods:
                if isinstance(m, dict):
                    standby_summaries.append(method_card_summary(m))

        from agentic_data_scientist.agents.adk.utils import _build_litellm
        from agentic_data_scientist.prompts import load_prompt

        llm = self.model if hasattr(self.model, "generate_content_async") else _build_litellm(self.model)

        logger.info(
            "[PlanInstantiator] Generating plan from method %s",
            selected_method.get("method_id", "?"),
        )

        method_json = json.dumps(selected_method, ensure_ascii=False, indent=2)
        standby_text = json.dumps(standby_summaries, ensure_ascii=False) if standby_summaries else "[]"

        raw_prompt = load_prompt("plan_instantiator")
        raw_prompt = raw_prompt.replace("{original_user_input?}", user_request)
        raw_prompt = raw_prompt.replace("{original_user_input}", user_request)
        raw_prompt = raw_prompt.replace("{selected_method?}", method_json)
        raw_prompt = raw_prompt.replace("{selected_method}", method_json)
        raw_prompt = raw_prompt.replace("{standby_methods?}", standby_text)
        raw_prompt = raw_prompt.replace("{standby_methods}", standby_text)

        try:
            llm_request = LlmRequest(
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=raw_prompt)])],
                config=types.GenerateContentConfig(temperature=0.4),
            )
            plan_text = ""
            async for llm_response in llm.generate_content_async(llm_request):
                if llm_response.content and llm_response.content.parts:
                    for part in llm_response.content.parts:
                        if part.text:
                            plan_text += part.text
            plan_text = plan_text.strip()
        except Exception as e:
            logger.error(f"[PlanInstantiator] LLM call failed: {e}")
            plan_text = self._fallback_plan(selected_method, user_request)

        if not plan_text:
            plan_text = self._fallback_plan(selected_method, user_request)

        state[StateKeys.HIGH_LEVEL_PLAN] = plan_text
        state[StateKeys.PLAN_CANDIDATES] = [plan_text]

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=plan_text)],
            ),
        )

    @staticmethod
    def _fallback_plan(method: Dict[str, Any], user_request: str) -> str:
        """Minimal plan skeleton when LLM fails."""
        title = method.get("title", "Unknown Method")
        hypothesis = method.get("core_hypothesis", "N/A")
        cheap_test = method.get("cheap_test", "N/A")
        artifacts = method.get("expected_artifacts", [])
        mid = method.get("method_id", "m1")

        stages = [
            f"Stage 1: Validation — {cheap_test} (source_method_id: {mid})",
            f"Stage 2: Data preparation and preprocessing (source_method_id: {mid})",
            f"Stage 3: Core analysis — {title} (source_method_id: {mid})",
            f"Stage 4: Results and visualization (source_method_id: {mid})",
        ]

        criteria = [
            f"Core hypothesis tested: {hypothesis}",
            "Data preprocessed and validated",
            f"Expected artifacts produced: {', '.join(artifacts[:5])}",
            "Results documented with visualizations",
        ]

        plan = f"# Analysis Plan: {title}\n\n"
        plan += f"**Objective:** {user_request}\n\n"
        plan += f"**Method:** {title} — {hypothesis}\n\n"
        plan += "## Analysis Stages\n\n"
        for i, s in enumerate(stages, 1):
            plan += f"{i}. {s}\n"
        plan += "\n## Success Criteria\n\n"
        for i, c in enumerate(criteria, 1):
            plan += f"{i}. {c}\n"

        return plan

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for PlanInstantiatorAgent.")
