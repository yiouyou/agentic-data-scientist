"""Abductive hypothesis operator for method discovery.

Generates competing-hypothesis method cards from unknowns and complexity
signals in the framed problem.  Each hypothesis proposes a different
*explanation* for the unknowns and derives a research method from it.

Design note (Phase 3-A, Option 1 — planning-time):
    This operator runs at planning time using ``FRAMED_PROBLEM.unknowns``
    and ``FRAMED_PROBLEM.complexity_signals`` as input.  A future Option 2
    enhancement could run at execution time using ``INNOVATION_TRIGGER``
    signals for more precise, observation-driven abduction.  See
    ``docs/innovation_os_development_plan.md`` for details.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from google.adk.models import LlmRequest
from google.genai import types

logger = logging.getLogger(__name__)


def build_abduction_prompt(
    *,
    unknowns: List[str],
    complexity_signals: List[str],
    user_request: str,
    existing_summaries: List[str],
    round_label: str = "abd_1",
) -> str:
    """Build the LLM prompt for abductive hypothesis method card generation."""
    from agentic_data_scientist.prompts import load_prompt

    raw = load_prompt("abduction_operator")

    replacements = {
        "original_user_input": user_request,
        "unknowns": json.dumps(unknowns, ensure_ascii=False),
        "complexity_signals": json.dumps(complexity_signals, ensure_ascii=False),
        "existing_methods": (json.dumps(existing_summaries, ensure_ascii=False) if existing_summaries else "[]"),
        "round_label": round_label,
    }
    for key, val in replacements.items():
        raw = raw.replace(f"{{{key}?}}", val)
        raw = raw.replace(f"{{{key}}}", val)
    return raw


def parse_abduction_method_card(
    text: str,
    round_label: str = "abd_1",
) -> Optional[Dict[str, Any]]:
    """Parse an abduction method card from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed.setdefault("method_id", round_label)
            parsed.setdefault("method_family", "abductive_hypothesis")
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
                            parsed.setdefault("method_id", round_label)
                            parsed.setdefault("method_family", "abductive_hypothesis")
                            return parsed
                    except json.JSONDecodeError:
                        continue
    return None


async def generate_abduction_candidates(
    *,
    unknowns: List[str],
    complexity_signals: List[str],
    user_request: str,
    existing_summaries: List[str],
    llm: Any,
    max_cards: int = 1,
) -> List[Dict[str, Any]]:
    """Generate method cards via abductive reasoning over unknowns.

    Parameters
    ----------
    unknowns:
        List of unknown strings from ``FRAMED_PROBLEM.unknowns``.
    complexity_signals:
        List of complexity signal strings from ``FRAMED_PROBLEM.complexity_signals``.
    user_request:
        The original user query.
    existing_summaries:
        One-line summaries of previously generated method cards.
    llm:
        A LiteLlm-compatible model with ``generate_content_async``.
    max_cards:
        Maximum number of abduction cards to generate (default 1).

    Returns
    -------
    List of method card dicts (un-validated — caller should validate).
    """
    if not unknowns and not complexity_signals:
        logger.debug("[Abduction] No unknowns or complexity signals — skipping")
        return []

    cards: List[Dict[str, Any]] = []

    for i in range(1, max_cards + 1):
        round_label = f"abd_{i}"
        prompt = build_abduction_prompt(
            unknowns=unknowns,
            complexity_signals=complexity_signals,
            user_request=user_request,
            existing_summaries=existing_summaries,
            round_label=round_label,
        )

        logger.info(f"[Abduction] Generating card {i}/{max_cards}")

        try:
            llm_request = LlmRequest(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.9),
            )
            response_text = ""
            async for llm_response in llm.generate_content_async(llm_request):
                if llm_response.content and llm_response.content.parts:
                    for part in llm_response.content.parts:
                        if part.text:
                            response_text += part.text
        except Exception as e:
            logger.warning(f"[Abduction] LLM call failed for round {round_label}: {e}")
            continue

        card_data = parse_abduction_method_card(response_text, round_label)
        if card_data is None:
            logger.warning(f"[Abduction] Failed to parse card from round {round_label}")
            continue

        card_data["method_id"] = round_label
        card_data["method_family"] = "abductive_hypothesis"
        cards.append(card_data)

        title = card_data.get("title", "?")
        hypothesis = card_data.get("core_hypothesis", "?")
        existing_summaries = list(existing_summaries) + [f"[{round_label}] {title} — {hypothesis}"]

    logger.info(f"[Abduction] Generated {len(cards)} abductive method cards")
    return cards
