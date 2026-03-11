"""TRIZ contradiction-resolution operator for method discovery.

Uses TRIZ-inspired inventive principles to generate method cards that
resolve contradictions identified in the framed problem.  This operator
runs as an *additional generation round* inside ``MethodDiscoveryAgent``
— it is NOT a standalone ADK agent.

Key TRIZ principles mapped to data-science contexts:
  - Separation (in space/time/scale)   → split data, temporal holdout
  - Inversion                          → reverse hypothesis
  - Dynamization                       → adaptive / online approach
  - Prior counteraction                → pre-correct known bias
  - Nesting / Segmentation             → hierarchical / ensemble models
  - Universality                       → transfer learning / foundation model
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from google.adk.models import LlmRequest
from google.genai import types

logger = logging.getLogger(__name__)

# The 6 TRIZ-derived resolution strategies most applicable to data science.
TRIZ_PRINCIPLES: List[Dict[str, str]] = [
    {
        "id": "separation",
        "name": "Separation",
        "description": (
            "Resolve the contradiction by separating conflicting requirements "
            "across different subsets, time windows, or scales of the data."
        ),
    },
    {
        "id": "inversion",
        "name": "Inversion",
        "description": (
            "Reverse the usual assumption — if the standard approach assumes X, design a method around NOT-X."
        ),
    },
    {
        "id": "dynamization",
        "name": "Dynamization",
        "description": (
            "Make the rigid part flexible — replace static models or parameters "
            "with adaptive, online, or context-dependent ones."
        ),
    },
    {
        "id": "prior_counteraction",
        "name": "Prior Counteraction",
        "description": (
            "Pre-correct for the expected source of failure — apply bias "
            "correction, calibration, or augmentation before analysis."
        ),
    },
    {
        "id": "nesting",
        "name": "Nesting / Segmentation",
        "description": (
            "Decompose the problem into nested sub-problems or use hierarchical "
            "/ ensemble methods so each layer resolves part of the contradiction."
        ),
    },
    {
        "id": "universality",
        "name": "Universality",
        "description": (
            "Use a general-purpose, pre-trained, or transfer-learning approach "
            "that side-steps the contradiction by leveraging external knowledge."
        ),
    },
]


def select_relevant_principles(
    contradictions: List[str],
    *,
    max_principles: int = 3,
) -> List[Dict[str, str]]:
    """Select the most relevant TRIZ principles for the given contradictions.

    For now this is a simple heuristic — return the first *max_principles*
    principles.  A future version could use embedding similarity.
    """
    # Always include at least separation + inversion as the two most
    # general-purpose strategies; fill up to *max_principles* from the rest.
    if max_principles >= len(TRIZ_PRINCIPLES):
        return list(TRIZ_PRINCIPLES)
    return TRIZ_PRINCIPLES[:max_principles]


def build_triz_prompt(
    *,
    contradictions: List[str],
    user_request: str,
    existing_summaries: List[str],
    principles: List[Dict[str, str]],
    round_label: str = "triz_1",
) -> str:
    """Build the LLM prompt for TRIZ-based method card generation."""
    from agentic_data_scientist.prompts import load_prompt

    raw = load_prompt("triz_operator")

    replacements = {
        "original_user_input": user_request,
        "contradictions": json.dumps(contradictions, ensure_ascii=False),
        "existing_methods": (json.dumps(existing_summaries, ensure_ascii=False) if existing_summaries else "[]"),
        "triz_principles": json.dumps(
            [{"id": p["id"], "name": p["name"], "description": p["description"]} for p in principles],
            ensure_ascii=False,
            indent=2,
        ),
        "round_label": round_label,
    }
    for key, val in replacements.items():
        raw = raw.replace(f"{{{key}?}}", val)
        raw = raw.replace(f"{{{key}}}", val)
    return raw


def parse_triz_method_card(
    text: str,
    round_label: str = "triz_1",
) -> Optional[Dict[str, Any]]:
    """Parse a TRIZ method card from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try full text first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed.setdefault("method_id", round_label)
            parsed.setdefault("method_family", "triz_resolution")
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object
    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text) - 1, start, -1):
                if text[end] == "}":
                    try:
                        parsed = json.loads(text[start : end + 1])
                        if isinstance(parsed, dict):
                            parsed.setdefault("method_id", round_label)
                            parsed.setdefault("method_family", "triz_resolution")
                            return parsed
                    except json.JSONDecodeError:
                        continue
    return None


async def generate_triz_candidates(
    *,
    contradictions: List[str],
    user_request: str,
    existing_summaries: List[str],
    llm: Any,
    max_cards: int = 1,
) -> List[Dict[str, Any]]:
    """Generate method cards by applying TRIZ principles to contradictions.

    Parameters
    ----------
    contradictions:
        List of contradiction strings from ``FRAMED_PROBLEM.contradictions``.
    user_request:
        The original user query.
    existing_summaries:
        One-line summaries of previously generated method cards (for negative
        prompting / avoiding duplication).
    llm:
        A LiteLlm-compatible model with ``generate_content_async``.
    max_cards:
        Maximum number of TRIZ cards to generate (default 1).

    Returns
    -------
    List of method card dicts (un-validated — caller should validate).
    """
    if not contradictions:
        logger.debug("[TRIZ] No contradictions — skipping")
        return []

    principles = select_relevant_principles(contradictions)
    cards: List[Dict[str, Any]] = []

    for i in range(1, max_cards + 1):
        round_label = f"triz_{i}"
        prompt = build_triz_prompt(
            contradictions=contradictions,
            user_request=user_request,
            existing_summaries=existing_summaries,
            principles=principles,
            round_label=round_label,
        )

        logger.info(f"[TRIZ] Generating card {i}/{max_cards}")

        try:
            llm_request = LlmRequest(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.8),
            )
            response_text = ""
            async for llm_response in llm.generate_content_async(llm_request):
                if llm_response.content and llm_response.content.parts:
                    for part in llm_response.content.parts:
                        if part.text:
                            response_text += part.text
        except Exception as e:
            logger.warning(f"[TRIZ] LLM call failed for round {round_label}: {e}")
            continue

        card_data = parse_triz_method_card(response_text, round_label)
        if card_data is None:
            logger.warning(f"[TRIZ] Failed to parse card from round {round_label}")
            continue

        card_data["method_id"] = round_label
        card_data["method_family"] = "triz_resolution"
        cards.append(card_data)

        # Add to existing summaries for next round's negative prompting
        title = card_data.get("title", "?")
        hypothesis = card_data.get("core_hypothesis", "?")
        existing_summaries = list(existing_summaries) + [f"[{round_label}] {title} — {hypothesis}"]

    logger.info(f"[TRIZ] Generated {len(cards)} TRIZ-based method cards")
    return cards
