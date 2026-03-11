"""Method candidate selector: scores and ranks method cards."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.genai import types

from agentic_data_scientist.core.method_card import validate_method_card
from agentic_data_scientist.core.state_contracts import StateKeys

logger = logging.getLogger(__name__)

# Scoring weights per the development plan
_WEIGHTS = {
    "feasibility": 0.30,
    "orthogonality": 0.20,
    "cheap_testability": 0.15,
    "capability_coverage": 0.15,
    "novelty": 0.10,
    "baseline_bonus": 0.10,
}


def _jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two tag lists."""
    set_a = set(t.lower().strip() for t in a if t.strip())
    set_b = set(t.lower().strip() for t in b if t.strip())
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _parse_selector_response(text: str) -> Dict[str, Any] | None:
    """Try to extract JSON from LLM response."""
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
    # Fallback: find outermost { ... }
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


def score_methods_programmatic(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pure-programmatic fallback scoring when LLM selection fails.

    Returns a list of dicts with method_id, total_score, scores, similarity_penalty.
    """
    results = []
    for i, card in enumerate(candidates):
        scores: Dict[str, float] = {}

        # feasibility: assume all methods are feasible by default
        scores["feasibility"] = 0.70

        # orthogonality: compare against all other candidates
        max_sim = 0.0
        tags_i = card.get("orthogonality_tags", [])
        for j, other in enumerate(candidates):
            if i == j:
                continue
            tags_j = other.get("orthogonality_tags", [])
            sim = _jaccard_similarity(tags_i, tags_j)
            max_sim = max(max_sim, sim)
        scores["orthogonality"] = max(0.0, 1.0 - max_sim)

        # cheap_testability: longer cheap_test = more concrete
        cheap_test = str(card.get("cheap_test", "") or "")
        scores["cheap_testability"] = min(1.0, len(cheap_test) / 100.0) if cheap_test.strip() else 0.2

        # capability_coverage: fraction of common capabilities
        _COMMON = {
            "python",
            "pandas",
            "numpy",
            "scipy",
            "scikit-learn",
            "sklearn",
            "matplotlib",
            "seaborn",
            "plotly",
            "statistical_testing",
            "visualization",
            "data_cleaning",
            "eda",
            "machine_learning",
            "deep_learning",
            "r",
            "statistics",
            "regression",
            "classification",
        }
        caps = card.get("required_capabilities", [])
        if caps:
            covered = sum(1 for c in caps if c.lower().strip() in _COMMON)
            scores["capability_coverage"] = covered / len(caps)
        else:
            scores["capability_coverage"] = 0.5

        # novelty: negative_variant gets higher novelty
        family = str(card.get("method_family", "") or "")
        scores["novelty"] = 0.70 if family == "negative_variant" else 0.30

        # baseline_bonus
        scores["baseline_bonus"] = 0.70 if family == "baseline" else 0.0

        # similarity penalty
        similarity_penalty = 0.05 if max_sim > 0.6 else 0.0

        total = sum(scores[k] * _WEIGHTS[k] for k in _WEIGHTS) - similarity_penalty

        results.append(
            {
                "method_id": card.get("method_id", f"m{i + 1}"),
                "scores": scores,
                "similarity_penalty": similarity_penalty,
                "total_score": round(total, 4),
                "rationale": f"Programmatic scoring for {card.get('title', 'unknown')}",
            }
        )

    results.sort(key=lambda r: r["total_score"], reverse=True)
    return results


class MethodCandidateSelectorAgent(BaseAgent):
    """Scores method candidates and selects the best one."""

    model: Any = ""
    fallback_model: Any = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        candidates = state.get(StateKeys.METHOD_CANDIDATES, [])
        if not isinstance(candidates, list) or not candidates:
            logger.info("[MethodSelector] No method candidates to select from")
            return

        # If only one candidate, auto-select it
        if len(candidates) == 1:
            card = candidates[0]
            card["status"] = "selected"
            card["selection_score"] = 1.0
            state[StateKeys.SELECTED_METHOD] = card
            state[StateKeys.STANDBY_METHODS] = []
            state[StateKeys.METHOD_SELECTION_TRACE] = {
                "rankings": [{"method_id": card.get("method_id"), "total_score": 1.0}],
                "selected_method_id": card.get("method_id"),
                "selection_rationale": "Single candidate auto-selected",
            }
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=f"Method selection: auto-selected {card.get('method_id')} (single candidate)"
                        )
                    ],
                ),
            )
            return

        # Try LLM-based selection
        llm_result = None
        if self.model:
            try:
                llm_result = await self._llm_select(state, candidates)
            except Exception as e:
                logger.warning(f"[MethodSelector] LLM selection failed: {e}")

        if llm_result and isinstance(llm_result.get("rankings"), list) and llm_result["rankings"]:
            rankings = llm_result["rankings"]
            selected_id = str(llm_result.get("selected_method_id", rankings[0].get("method_id", "")))
        else:
            # Fallback to programmatic scoring
            logger.info("[MethodSelector] Using programmatic fallback scoring")
            rankings = score_methods_programmatic(candidates)
            selected_id = rankings[0]["method_id"] if rankings else candidates[0].get("method_id", "m1")
            llm_result = {
                "rankings": rankings,
                "selected_method_id": selected_id,
                "selection_rationale": "Programmatic fallback scoring",
            }

        # Apply selection to candidates
        selected_card = None
        standby = []
        for card in candidates:
            mid = card.get("method_id", "")
            if mid == selected_id:
                card["status"] = "selected"
                # Find score from rankings
                for r in rankings:
                    if r.get("method_id") == mid:
                        card["selection_score"] = r.get("total_score")
                        break
                selected_card = card
            else:
                card["status"] = "standby"
                for r in rankings:
                    if r.get("method_id") == mid:
                        card["selection_score"] = r.get("total_score")
                        card["rejection_reason"] = r.get("rationale", "Not top-ranked")
                        break
                standby.append(card)

        if selected_card is None:
            # Fallback: just pick the first candidate
            selected_card = candidates[0]
            selected_card["status"] = "selected"
            standby = candidates[1:]

        state[StateKeys.SELECTED_METHOD] = selected_card
        state[StateKeys.STANDBY_METHODS] = standby
        state[StateKeys.METHOD_SELECTION_TRACE] = llm_result

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Method selection: chose {selected_card.get('method_id')} "
                            f"({selected_card.get('title', '?')}) "
                            f"score={selected_card.get('selection_score', '?')}, "
                            f"standby={len(standby)}"
                        )
                    )
                ],
            ),
        )

    async def _llm_select(
        self,
        state: Any,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Use LLM to score and rank method candidates."""
        from agentic_data_scientist.agents.adk.utils import _build_litellm
        from agentic_data_scientist.prompts import load_prompt

        llm = self.model if hasattr(self.model, "generate_content_async") else _build_litellm(self.model)

        user_request = str(state.get(StateKeys.ORIGINAL_USER_INPUT, "") or "")
        candidates_json = json.dumps(
            [
                {
                    "method_id": c.get("method_id"),
                    "title": c.get("title"),
                    "core_hypothesis": c.get("core_hypothesis"),
                    "assumptions": c.get("assumptions", []),
                    "cheap_test": c.get("cheap_test"),
                    "required_capabilities": c.get("required_capabilities", []),
                    "orthogonality_tags": c.get("orthogonality_tags", []),
                    "method_family": c.get("method_family"),
                }
                for c in candidates
            ],
            ensure_ascii=False,
            indent=2,
        )

        raw_prompt = load_prompt("method_selector")
        raw_prompt = raw_prompt.replace("{original_user_input?}", user_request)
        raw_prompt = raw_prompt.replace("{original_user_input}", user_request)
        raw_prompt = raw_prompt.replace("{method_candidates?}", candidates_json)
        raw_prompt = raw_prompt.replace("{method_candidates}", candidates_json)

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

        if not response_text:
            return None

        return _parse_selector_response(response_text)

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError("Live mode is not supported for MethodCandidateSelectorAgent.")
