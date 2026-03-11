"""Stage decomposer agent — splits complex stages into sub-stages.

Phase 3-B: When a stage fails repeatedly or is flagged as too complex,
this agent decomposes it into 2-3 smaller, independently implementable
sub-stages. Controlled by ``budget.decomposition``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import PrivateAttr

from agentic_data_scientist.core.budget_controller import InnovationBudget
from agentic_data_scientist.core.state_contracts import StateKeys, make_stage_record

logger = logging.getLogger(__name__)


def build_decomposer_prompt(
    stage: Dict[str, Any],
    failure_context: str,
    success_criteria: List[Dict[str, Any]],
) -> str:
    """Build the prompt for stage decomposition LLM call."""
    from agentic_data_scientist.prompts import load_prompt

    criteria_text = "\n".join(f"  - [{c.get('index', '?')}] {c.get('criteria', '')}" for c in success_criteria)

    template = load_prompt("stage_decomposer")
    return (
        template.replace("{stage_title}", stage.get("title", ""))
        .replace("{stage_description}", stage.get("description", ""))
        .replace("{stage_id}", stage.get("stage_id", f"s{stage.get('index', 0) + 1}"))
        .replace("{attempt_count}", str(stage.get("attempts", 0)))
        .replace("{failure_context}", failure_context or "Stage was flagged as too complex.")
        .replace("{success_criteria}", criteria_text or "(none)")
    )


def parse_decomposition_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON decomposition payload from LLM response text."""
    # Try to find JSON block in markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find raw JSON object
    m = re.search(r"\{.*\"sub_stages\".*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def apply_decomposition(
    stages: List[Dict[str, Any]],
    stage_index: int,
    sub_stages_raw: List[Dict[str, Any]],
    original_stage: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Replace the target stage with decomposed sub-stages in the stage list.

    The original stage is removed and 2-3 sub-stages are inserted at its position.
    All subsequent stage indices are shifted accordingly.

    Parameters
    ----------
    stages : list
        The full stage list (mutable).
    stage_index : int
        The *list position* (0-based index into ``stages``) of the stage being replaced.
    sub_stages_raw : list
        Raw sub-stage dicts from LLM (title, description, inputs_required, outputs_produced).
    original_stage : dict
        The original stage being decomposed — used to inherit metadata.

    Returns
    -------
    list
        The updated stage list with sub-stages spliced in.
    """
    if not sub_stages_raw or len(sub_stages_raw) < 2:
        return stages

    source_method_id = original_stage.get("source_method_id")
    method_family = original_stage.get("method_family")
    orig_depends = original_stage.get("depends_on", [])

    new_sub_stages: List[Dict[str, Any]] = []
    base_idx = original_stage.get("index", stage_index)

    for i, raw in enumerate(sub_stages_raw[:3]):  # cap at 3
        sub = make_stage_record(
            index=base_idx,  # will be re-indexed below
            title=raw.get("title", f"Sub-stage {i + 1}"),
            description=raw.get("description", ""),
            inputs_required=raw.get("inputs_required"),
            outputs_produced=raw.get("outputs_produced"),
            source_method_id=source_method_id,
            method_family=method_family,
            depends_on=orig_depends if i == 0 else None,
        )
        sub["_decomposed_from"] = original_stage.get("stage_id", f"s{base_idx + 1}")
        new_sub_stages.append(sub)

    # Chain dependencies: each sub-stage depends on the previous
    for i in range(1, len(new_sub_stages)):
        prev_id = new_sub_stages[i - 1]["stage_id"]
        new_sub_stages[i]["depends_on"] = [prev_id]

    # Splice into the list
    result = list(stages[:stage_index]) + new_sub_stages + list(stages[stage_index + 1 :])

    # Re-index all stages and assign fresh stage_ids
    for pos, stg in enumerate(result):
        stg["index"] = pos
        stg["stage_id"] = f"s{pos + 1}"

    return result


class StageDecomposerAgent(BaseAgent):
    """Decomposes a complex stage into 2-3 sub-stages via LLM call.

    This agent is called by the stage orchestrator when a stage is deemed
    too complex.  It reads the current stage from state, calls the LLM
    to produce sub-stages, and rewrites ``StateKeys.HIGH_LEVEL_STAGES``.

    Gating:
    - ``INNOVATION_MODE != "routine"``
    - ``budget.decomposition`` has remaining capacity
    """

    _model: Any = PrivateAttr(default=None)
    _fallback_model: Any = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        name: str = "stage_decomposer",
        model: Any = None,
        fallback_model: Any = None,
        description: str = "Decomposes complex stages into sub-stages.",
    ):
        super().__init__(name=name, description=description)
        self._model = model
        self._fallback_model = fallback_model

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        innovation_mode = str(state.get(StateKeys.INNOVATION_MODE, "routine") or "routine").strip()

        # Gate: routine mode — skip
        if innovation_mode == "routine":
            logger.info("[StageDecomposer] Routine mode — skipping")
            return

        # Gate: budget check
        budget_raw = state.get(StateKeys.INNOVATION_BUDGET)
        if isinstance(budget_raw, dict):
            budget = InnovationBudget.from_dict(budget_raw)
        else:
            budget = InnovationBudget()

        if budget.is_exhausted("decomposition"):
            logger.info("[StageDecomposer] Decomposition budget exhausted — skipping")
            return

        # Read current stage
        current_stage = state.get(StateKeys.CURRENT_STAGE)
        if not isinstance(current_stage, dict):
            logger.warning("[StageDecomposer] No current stage in state — skipping")
            return

        stages = state.get(StateKeys.HIGH_LEVEL_STAGES, [])
        criteria = state.get(StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA, [])

        # Build failure context from review feedback + critic output
        failure_parts: list[str] = []
        review_fb = state.get(StateKeys.REVIEW_FEEDBACK)
        if review_fb:
            failure_parts.append(f"Review feedback: {review_fb}")
        critic_out = state.get(StateKeys.METHOD_CRITIC_OUTPUT)
        if isinstance(critic_out, dict):
            failure_parts.append(f"Critic diagnosis: {critic_out.get('diagnosis', '')}")
        failure_context = "\n".join(failure_parts) or "Stage failed on multiple attempts."

        prompt_text = build_decomposer_prompt(current_stage, failure_context, criteria)

        logger.info(
            "[StageDecomposer] Decomposing stage %s (%s)",
            current_stage.get("stage_id", "?"),
            current_stage.get("title", "?"),
        )

        # LLM call
        from google.genai.types import Content, Part

        llm = self._model
        if llm is None:
            from agentic_data_scientist.agents.adk.utils import _build_litellm

            llm = _build_litellm("openai/gpt-4o-mini")

        if not hasattr(llm, "generate_content_async"):
            from agentic_data_scientist.agents.adk.utils import _build_litellm

            llm = _build_litellm(str(llm))

        from google.adk.models.llm_request import LlmRequest

        llm_request = LlmRequest(
            contents=[Content(role="user", parts=[Part(text=prompt_text)])],
        )

        response_text = ""
        try:
            async for resp in llm.generate_content_async(llm_request):
                if resp and resp.content and resp.content.parts:
                    for part in resp.content.parts:
                        if part.text:
                            response_text += part.text
        except Exception as e:
            logger.error("[StageDecomposer] LLM call failed: %s", e)
            if self._fallback_model is not None:
                fallback = self._fallback_model
                if not isinstance(fallback, LiteLlm):
                    from agentic_data_scientist.agents.adk.utils import _build_litellm

                    fallback = _build_litellm(str(fallback))
                try:
                    llm_request2 = LlmRequest(
                        contents=[Content(role="user", parts=[Part(text=prompt_text)])],
                    )
                    async for resp in fallback.generate_content_async(llm_request2):
                        if resp and resp.content and resp.content.parts:
                            for part in resp.content.parts:
                                if part.text:
                                    response_text += part.text
                except Exception as e2:
                    logger.error("[StageDecomposer] Fallback LLM also failed: %s", e2)
                    return
            else:
                return

        if not response_text.strip():
            logger.warning("[StageDecomposer] Empty LLM response — skipping")
            return

        # Parse response
        parsed = parse_decomposition_response(response_text)
        if parsed is None or "sub_stages" not in parsed:
            logger.warning("[StageDecomposer] Failed to parse decomposition response")
            return

        sub_stages_raw = parsed["sub_stages"]
        if not isinstance(sub_stages_raw, list) or len(sub_stages_raw) < 2:
            logger.warning("[StageDecomposer] Not enough sub-stages (%d) — skipping", len(sub_stages_raw or []))
            return

        # Find current stage position in list
        target_pos = None
        for pos, s in enumerate(stages):
            if s.get("stage_id") == current_stage.get("stage_id") or s.get("index") == current_stage.get("index"):
                target_pos = pos
                break

        if target_pos is None:
            logger.warning("[StageDecomposer] Could not locate stage in list — skipping")
            return

        # Apply decomposition
        new_stages = apply_decomposition(stages, target_pos, sub_stages_raw, current_stage)
        state[StateKeys.HIGH_LEVEL_STAGES] = new_stages

        # Consume budget
        budget.consume("decomposition")
        state[StateKeys.INNOVATION_BUDGET] = budget.to_dict()

        rationale = parsed.get("decomposition_rationale", "Stage was too complex.")
        n_sub = min(len(sub_stages_raw), 3)

        logger.info(
            "[StageDecomposer] Decomposed into %d sub-stages. Rationale: %s",
            n_sub,
            rationale[:120],
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"\n\n🔬 Stage decomposed into {n_sub} sub-stages.\n"
                            f"Rationale: {rationale}\n"
                            f"Sub-stages: {', '.join(s.get('title', '?') for s in sub_stages_raw[:3])}\n\n"
                        )
                    )
                ],
            ),
            turn_complete=False,
        )
