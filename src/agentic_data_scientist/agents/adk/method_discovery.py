"""Method discovery layer: generates diverse method candidates via LLM."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.genai import types

from agentic_data_scientist.core.budget_controller import InnovationBudget, get_budget_for_mode
from agentic_data_scientist.core.method_card import make_method_card, method_card_summary, validate_method_card
from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)

_METHOD_CARD_KEYS = (
    "method_id",
    "method_family",
    "title",
    "core_hypothesis",
    "assumptions",
    "invalid_if",
    "cheap_test",
    "failure_modes",
    "required_capabilities",
    "expected_artifacts",
    "orthogonality_tags",
)

_OPERATOR_LIST_FIELDS = (
    "assumptions",
    "invalid_if",
    "failure_modes",
    "required_capabilities",
    "expected_artifacts",
    "orthogonality_tags",
)


def _parse_method_card_from_text(text: str, round_number: int) -> Dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            method_id = parsed.get("method_id", f"m{round_number}")
            family = "baseline" if round_number == 1 else "negative_variant"
            parsed.setdefault("method_id", method_id)
            parsed.setdefault("method_family", family)
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
                            parsed.setdefault("method_id", f"m{round_number}")
                            parsed.setdefault("method_family", "baseline" if round_number == 1 else "negative_variant")
                            return parsed
                    except json.JSONDecodeError:
                        continue
    return None


class MethodDiscoveryAgent(BaseAgent):
    """Generates method card candidates via iterative LLM calls with negative prompting."""

    model: Any = ""
    fallback_model: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        innovation_mode = str(state.get(StateKeys.INNOVATION_MODE, "routine") or "routine").strip()
        if innovation_mode == "routine":
            logger.info("[MethodDiscovery] Routine mode — skipping method discovery")
            state[StateKeys.METHOD_CANDIDATES] = []
            return

        budget = get_budget_for_mode(innovation_mode)
        max_methods = budget.method_generation
        if max_methods <= 0:
            state[StateKeys.METHOD_CANDIDATES] = []
            return

        user_request = str(state.get(StateKeys.ORIGINAL_USER_INPUT, "") or "")
        if not user_request.strip():
            state[StateKeys.METHOD_CANDIDATES] = []
            return

        candidates: List[Dict[str, Any]] = []
        negative_constraints: List[str] = []
        existing_summaries: List[str] = []

        from agentic_data_scientist.agents.adk.utils import _build_litellm

        llm = self.model if hasattr(self.model, "generate_content_async") else _build_litellm(self.model)

        # Phase 3-C: load episodic memory for cross-run negative constraints
        episodic_summary = ""
        try:
            from agentic_data_scientist.core.innovation_memory import create_innovation_memory_from_env

            inno_mem = create_innovation_memory_from_env()
            if inno_mem is not None:
                episodic_summary = inno_mem.build_negative_constraints_summary(limit=10)
                if episodic_summary:
                    logger.info(
                        "[MethodDiscovery] Loaded %d chars of episodic memory constraints", len(episodic_summary)
                    )
        except Exception as e:
            logger.debug("[MethodDiscovery] Episodic memory unavailable: %s", e)

        for round_num in range(1, max_methods + 1):
            if not budget.consume("method_generation"):
                break

            prompt_vars = {
                "original_user_input": user_request,
                "round_number": str(round_num),
                "existing_methods": json.dumps(existing_summaries, ensure_ascii=False) if existing_summaries else "[]",
                "negative_constraints": "\n".join(f"- {c}" for c in negative_constraints)
                if negative_constraints
                else "None",
                "episodic_memory_constraints": episodic_summary or "None",
            }

            from agentic_data_scientist.prompts import load_prompt

            raw_prompt = load_prompt("method_discovery")
            for key, val in prompt_vars.items():
                raw_prompt = raw_prompt.replace(f"{{{key}?}}", val)
                raw_prompt = raw_prompt.replace(f"{{{key}}}", val)

            logger.info(f"[MethodDiscovery] Round {round_num}/{max_methods}: generating method card")

            try:
                llm_request = LlmRequest(
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=raw_prompt)])],
                    config=types.GenerateContentConfig(temperature=0.7),
                )
                response_text = ""
                async for llm_response in llm.generate_content_async(llm_request):
                    if llm_response.content and llm_response.content.parts:
                        for part in llm_response.content.parts:
                            if part.text:
                                response_text += part.text
            except Exception as e:
                logger.warning(f"[MethodDiscovery] LLM call failed in round {round_num}: {e}")
                continue

            card_data = _parse_method_card_from_text(response_text, round_num)
            if card_data is None:
                logger.warning(f"[MethodDiscovery] Failed to parse method card from round {round_num}")
                continue

            card_data["method_id"] = f"m{round_num}"
            card_data["method_family"] = "baseline" if round_num == 1 else "negative_variant"

            errors = validate_method_card(card_data)
            if errors:
                logger.warning(f"[MethodDiscovery] Validation errors in round {round_num}: {errors}")
                for field_name in _OPERATOR_LIST_FIELDS:
                    if not isinstance(card_data.get(field_name), list):
                        card_data[field_name] = []

            card = make_method_card(**{k: card_data[k] for k in _METHOD_CARD_KEYS if k in card_data})
            candidates.append(card)

            summary = method_card_summary(card)
            existing_summaries.append(summary)
            negative_constraints.append(card.get("core_hypothesis", ""))
            tags = card.get("orthogonality_tags", [])
            if tags:
                negative_constraints.extend(tags[:3])

            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"Method discovery round {round_num}: {summary}")],
                ),
            )

        # --- Phase 3-A: TRIZ + abduction operator rounds ---
        framed_problem = state.get(StateKeys.FRAMED_PROBLEM) or {}
        contradictions = framed_problem.get("contradictions", [])
        unknowns = framed_problem.get("unknowns", [])
        complexity_signals = framed_problem.get("complexity_signals", [])

        if contradictions:
            from agentic_data_scientist.agents.adk.operators.triz import generate_triz_candidates

            logger.info("[MethodDiscovery] TRIZ operator: %d contradictions found", len(contradictions))
            try:
                triz_cards = await generate_triz_candidates(
                    contradictions=contradictions,
                    user_request=user_request,
                    existing_summaries=existing_summaries,
                    llm=llm,
                    max_cards=1,
                )
                for tc in triz_cards:
                    tc.setdefault("method_family", "triz_resolution")
                    for field_name in _OPERATOR_LIST_FIELDS:
                        if not isinstance(tc.get(field_name), list):
                            tc[field_name] = []
                    card = make_method_card(**{k: tc[k] for k in _METHOD_CARD_KEYS if k in tc})
                    candidates.append(card)
                    existing_summaries.append(method_card_summary(card))

                    yield Event(
                        author=self.name,
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=f"TRIZ operator: {method_card_summary(card)}")],
                        ),
                    )
            except Exception as e:
                logger.warning("[MethodDiscovery] TRIZ operator failed: %s", e)

        if unknowns or complexity_signals:
            from agentic_data_scientist.agents.adk.operators.abduction import generate_abduction_candidates

            logger.info(
                "[MethodDiscovery] Abduction operator: %d unknowns, %d complexity_signals",
                len(unknowns),
                len(complexity_signals),
            )
            try:
                abd_cards = await generate_abduction_candidates(
                    unknowns=unknowns,
                    complexity_signals=complexity_signals,
                    user_request=user_request,
                    existing_summaries=existing_summaries,
                    llm=llm,
                    max_cards=1,
                )
                for ac in abd_cards:
                    ac.setdefault("method_family", "abductive_hypothesis")
                    for field_name in _OPERATOR_LIST_FIELDS:
                        if not isinstance(ac.get(field_name), list):
                            ac[field_name] = []
                    card = make_method_card(**{k: ac[k] for k in _METHOD_CARD_KEYS if k in ac})
                    candidates.append(card)
                    existing_summaries.append(method_card_summary(card))

                    yield Event(
                        author=self.name,
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=f"Abduction operator: {method_card_summary(card)}")],
                        ),
                    )
            except Exception as e:
                logger.warning("[MethodDiscovery] Abduction operator failed: %s", e)

        state[StateKeys.METHOD_CANDIDATES] = candidates
        state[StateKeys.INNOVATION_BUDGET] = budget.to_dict()

        logger.info(f"[MethodDiscovery] Generated {len(candidates)} method candidates (incl. operator rounds)")

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for MethodDiscoveryAgent.")
