"""
Main ADK agent factory for Agentic Data Scientist.

This module creates the multi-agent system with planning, orchestration,
implementation, and verification agents.
"""

import logging
import json
import re
import warnings
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

from dotenv import load_dotenv
from google.adk.agents import InvocationContext, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.events import Event
from google.adk.planners import BuiltInPlanner
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from pydantic import BaseModel, Field
from typing_extensions import override

from agentic_data_scientist.agents.adk.event_compression import create_compression_callback
from agentic_data_scientist.agents.adk.implementation_loop import make_implementation_agents
from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.review_confirmation import create_review_confirmation_agent
from agentic_data_scientist.agents.adk.utils import (
    DEFAULT_MODEL_NAME,
    REVIEW_MODEL_NAME,
    get_generate_content_config,
    get_litellm_candidates_for_role,
    is_network_disabled,
)
from agentic_data_scientist.core.state_contracts import (
    StateKeys,
    make_stage_record,
    make_success_criterion_record,
)
from agentic_data_scientist.core.knowledge_constraints import normalize_and_validate_stage_constraints
from agentic_data_scientist.prompts import load_prompt


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Suppress experimental feature warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.adk.tools.mcp_tool")

# Suppress verbose JSON Schema conversion logs
logging.getLogger("google_genai.types").setLevel(logging.WARNING)


# ========================= Output Schemas (Pydantic BaseModel) =========================


class Stage(BaseModel):
    """A high-level implementation stage."""

    title: str = Field(description="Stage title")
    description: str = Field(description="Detailed stage description")
    stage_id: Optional[str] = Field(default=None, description="Optional stable stage id")
    depends_on: Optional[List[str]] = Field(
        default=None, description="Optional list of stage ids/indexes this stage depends on"
    )
    inputs_required: Optional[List[str]] = Field(
        default=None, description="Optional required inputs/artifacts for this stage"
    )
    outputs_produced: Optional[List[str]] = Field(
        default=None, description="Optional outputs/artifacts produced by this stage"
    )
    evidence_refs: Optional[List[str]] = Field(
        default=None, description="Optional evidence references supporting stage completion"
    )


class SuccessCriterion(BaseModel):
    """A success criterion for completion."""

    criteria: str = Field(description="Success criterion description")


class PlanParserOutput(BaseModel):
    """Parsed high-level plan into stages and success criteria."""

    stages: List[Stage] = Field(description="List of high-level stages to implement progressively")
    success_criteria: List[SuccessCriterion] = Field(description="Definitive checklist for overall analysis completion")


class CriteriaUpdate(BaseModel):
    """Update for a specific success criterion."""

    index: int = Field(description="Criterion index")
    met: bool = Field(description="Whether criterion is met")
    evidence: str = Field(description="Evidence or reason for the status (file paths, metrics, etc.)")


class CriteriaCheckerOutput(BaseModel):
    """Updated success criteria status."""

    criteria_updates: List[CriteriaUpdate] = Field(description="List of criteria with updated met status and evidence")


class StageModification(BaseModel):
    """Modification to an existing stage."""

    index: int = Field(description="Stage index to modify")
    new_description: str = Field(description="Updated stage description (or empty if no change)")


class NewStage(BaseModel):
    """A new stage to add."""

    title: str = Field(description="New stage title")
    description: str = Field(description="New stage description")
    stage_id: Optional[str] = Field(default=None, description="Optional stable stage id")
    depends_on: Optional[List[str]] = Field(default=None, description="Optional dependency stage ids/indexes")
    inputs_required: Optional[List[str]] = Field(default=None, description="Optional required inputs for the stage")
    outputs_produced: Optional[List[str]] = Field(default=None, description="Optional outputs produced by the stage")
    evidence_refs: Optional[List[str]] = Field(default=None, description="Optional evidence references")


class StageReflectorOutput(BaseModel):
    """Reflection on remaining stages with optional modifications."""

    stage_modifications: List[StageModification] = Field(description="Modifications to existing uncompleted stages")
    new_stages: List[NewStage] = Field(description="New stages to add to the end of the stage list")


# Keep for backwards compatibility
PLAN_PARSER_OUTPUT_SCHEMA = PlanParserOutput
CRITERIA_CHECKER_OUTPUT_SCHEMA = CriteriaCheckerOutput
STAGE_REFLECTOR_OUTPUT_SCHEMA = StageReflectorOutput


# ========================= Callbacks =========================


_WORKFLOW_ID_PATTERN = re.compile(r"(?im)\bworkflow_id\s*[:=]\s*([A-Za-z0-9_.\-\/]+)")
_WORKFLOW_TAG_PATTERN = re.compile(r"(?im)\[\s*workflow\s*[:=]\s*([A-Za-z0-9_.\-\/]+)\s*\]")
_WORKFLOW_VERSION_PATTERN = re.compile(r"(?im)\bworkflow_version\s*[:=]\s*([A-Za-z0-9_.\-]+)")
_EXECUTION_MODE_PATTERN = re.compile(r"(?im)\bexecution_mode\s*[:=]\s*([A-Za-z0-9_.\-]+)")
_WORKFLOW_INPUTS_PATTERN = re.compile(r"(?im)\bworkflow_inputs\s*[:=]\s*(\{.*\})")
_WORKFLOW_PARAMS_PATTERN = re.compile(r"(?im)\bworkflow_params\s*[:=]\s*(\{.*\})")
_WORKFLOW_OUTDIR_PATTERN = re.compile(r"(?im)\bworkflow_outdir\s*[:=]\s*([^\n]+)")
_DEPENDS_ON_PATTERN = re.compile(r"(?im)\bdepends_on\s*[:=]\s*(\[[^\n]*\])")
_INPUTS_REQUIRED_PATTERN = re.compile(r"(?im)\binputs_required\s*[:=]\s*(\[[^\n]*\])")
_OUTPUTS_PRODUCED_PATTERN = re.compile(r"(?im)\boutputs_produced\s*[:=]\s*(\[[^\n]*\])")
_EVIDENCE_REFS_PATTERN = re.compile(r"(?im)\bevidence_refs\s*[:=]\s*(\[[^\n]*\])")


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    return str(match.group(1)).strip()


def _parse_json_object(text: str) -> dict:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _parse_json_array(text: str) -> list:
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
            return values
    except Exception:
        pass
    return []


def _extract_stage_execution_hints(*, title: str, description: str) -> dict:
    """Extract optional workflow execution hints from stage text."""
    combined = f"{title}\n{description}"
    hints: dict = {}

    workflow_id = _first_match(_WORKFLOW_ID_PATTERN, combined)
    if not workflow_id:
        workflow_id = _first_match(_WORKFLOW_TAG_PATTERN, combined)
    if workflow_id:
        hints["workflow_id"] = workflow_id

    workflow_version = _first_match(_WORKFLOW_VERSION_PATTERN, combined)
    if workflow_version:
        hints["workflow_version"] = workflow_version

    execution_mode = _first_match(_EXECUTION_MODE_PATTERN, combined).lower()
    if execution_mode:
        hints["execution_mode"] = execution_mode

    workflow_outdir = _first_match(_WORKFLOW_OUTDIR_PATTERN, combined)
    if workflow_outdir:
        hints["workflow_outdir"] = workflow_outdir

    workflow_inputs = _parse_json_object(_first_match(_WORKFLOW_INPUTS_PATTERN, combined))
    if workflow_inputs:
        hints["workflow_inputs"] = workflow_inputs

    workflow_params = _parse_json_object(_first_match(_WORKFLOW_PARAMS_PATTERN, combined))
    if workflow_params:
        hints["workflow_params"] = workflow_params

    depends_on = _parse_json_array(_first_match(_DEPENDS_ON_PATTERN, combined))
    if depends_on:
        hints["depends_on"] = depends_on

    inputs_required = _parse_json_array(_first_match(_INPUTS_REQUIRED_PATTERN, combined))
    if inputs_required:
        hints["inputs_required"] = inputs_required

    outputs_produced = _parse_json_array(_first_match(_OUTPUTS_PRODUCED_PATTERN, combined))
    if outputs_produced:
        hints["outputs_produced"] = outputs_produced

    evidence_refs = _parse_json_array(_first_match(_EVIDENCE_REFS_PATTERN, combined))
    if evidence_refs:
        hints["evidence_refs"] = evidence_refs

    if hints.get("workflow_id") and "execution_mode" not in hints:
        hints["execution_mode"] = "workflow"

    return hints


def plan_parser_callback(callback_context: CallbackContext):
    """
    Transform parsed output into structured stage/criteria lists.

    This callback processes the plan parser output and initializes
    high_level_stages and high_level_success_criteria with proper tracking fields.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context with invocation context access
    """

    ctx = callback_context._invocation_context
    state = ctx.session.state

    # Get the output from the agent
    parsed_output = state.get(StateKeys.PARSED_PLAN_OUTPUT)

    if not parsed_output:
        logger.error("[PlanParser] No parsed output found in state")
        return

    # Validate structure
    if not isinstance(parsed_output, dict):
        logger.error(f"[PlanParser] Invalid parsed output type: {type(parsed_output)}")
        return

    stages_data = parsed_output.get("stages", [])
    criteria_data = parsed_output.get("success_criteria", [])

    if not isinstance(stages_data, list):
        logger.error(f"[PlanParser] stages is not a list: {type(stages_data)}")
        return

    if not isinstance(criteria_data, list):
        logger.error(f"[PlanParser] success_criteria is not a list: {type(criteria_data)}")
        return

    logger.info("[PlanParser] Processing parsed plan output")

    if not stages_data:
        logger.error("[PlanParser] Empty stages list - rejecting parsed output")
        return

    if not criteria_data:
        logger.error("[PlanParser] Empty success_criteria list - rejecting parsed output")
        return

    # Strict validation: reject entire update if any item is invalid
    stages = []
    for idx, stage in enumerate(stages_data):
        if not isinstance(stage, dict):
            logger.error(f"[PlanParser] Invalid stage type at index {idx}: {type(stage)}")
            return

        title = stage.get("title")
        description = stage.get("description")
        if not isinstance(title, str) or not title.strip():
            logger.error(f"[PlanParser] Invalid stage title at index {idx}: {title!r}")
            return
        if not isinstance(description, str) or not description.strip():
            logger.error(f"[PlanParser] Invalid stage description at index {idx}: {description!r}")
            return

        stage_record = make_stage_record(
            index=idx,
            title=title,
            description=description,
            completed=False,
            stage_id=stage.get("stage_id"),
            depends_on=stage.get("depends_on"),
            inputs_required=stage.get("inputs_required"),
            outputs_produced=stage.get("outputs_produced"),
            evidence_refs=stage.get("evidence_refs"),
        )
        hints = _extract_stage_execution_hints(title=title, description=description)
        if hints:
            stage_record.update(hints)
        stages.append(stage_record)

    criteria = []
    for idx, crit in enumerate(criteria_data):
        if not isinstance(crit, dict):
            logger.error(f"[PlanParser] Invalid criterion type at index {idx}: {type(crit)}")
            return

        criteria_text = crit.get("criteria")
        if not isinstance(criteria_text, str) or not criteria_text.strip():
            logger.error(f"[PlanParser] Invalid criterion text at index {idx}: {criteria_text!r}")
            return

        criteria.append(
            make_success_criterion_record(
                index=idx,
                criteria=criteria_text,
            )
        )

    constraints = normalize_and_validate_stage_constraints(stages, apply_sequential_defaults=True)
    if constraints.errors:
        for error in constraints.errors:
            logger.error(f"[PlanParser] Constraint validation failed: {error}")
        return
    for warning in constraints.warnings:
        logger.warning(f"[PlanParser] Constraint warning: {warning}")

    state[StateKeys.HIGH_LEVEL_STAGES] = constraints.stages
    state[StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA] = criteria
    state[StateKeys.CURRENT_STAGE_INDEX] = 0

    logger.info(f"[PlanParser] Initialized {len(constraints.stages)} stages and {len(criteria)} criteria")


def criteria_checker_callback(callback_context: CallbackContext):
    """
    Update criteria met status based on checker output.

    This callback updates the high_level_success_criteria in state based on
    the criteria checker's assessment.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context with invocation context access
    """

    ctx = callback_context._invocation_context
    state = ctx.session.state

    criteria_output = state.get(StateKeys.CRITERIA_CHECKER_OUTPUT)
    criteria = state.get(StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA, [])

    if not criteria_output:
        logger.error("[CriteriaChecker] No output found in state")
        return

    if not isinstance(criteria_output, dict) or "criteria_updates" not in criteria_output:
        logger.error("[CriteriaChecker] Invalid output structure")
        return

    updates = criteria_output["criteria_updates"]
    if not isinstance(updates, list):
        logger.error(f"[CriteriaChecker] criteria_updates is not a list: {type(updates)}")
        return

    if not isinstance(criteria, list) or not criteria:
        logger.error("[CriteriaChecker] Missing or empty high_level_success_criteria in state")
        return

    expected_indices = set(range(len(criteria)))
    seen_indices = set()
    parsed_updates = []

    # Strict coverage rule: checker must return exactly one update per criterion
    if len(updates) != len(criteria):
        logger.error(
            f"[CriteriaChecker] Rejecting updates: expected {len(criteria)} updates, got {len(updates)}"
        )
        return

    for update in updates:
        if not isinstance(update, dict):
            logger.error(f"[CriteriaChecker] Invalid update structure (not dict): {update}")
            return

        if "index" not in update or "met" not in update or "evidence" not in update:
            logger.error(f"[CriteriaChecker] Invalid update structure (missing fields): {update}")
            return

        idx = update["index"]
        met = update["met"]
        evidence = update["evidence"]

        if not isinstance(idx, int):
            logger.error(f"[CriteriaChecker] Invalid index type: {type(idx)}")
            return
        if idx not in expected_indices:
            logger.error(f"[CriteriaChecker] Invalid criterion index: {idx}")
            return
        if idx in seen_indices:
            logger.error(f"[CriteriaChecker] Duplicate criterion index: {idx}")
            return
        if not isinstance(met, bool):
            logger.error(f"[CriteriaChecker] Invalid met type at index {idx}: {type(met)}")
            return
        if not isinstance(evidence, str) or not evidence.strip():
            logger.error(f"[CriteriaChecker] Invalid evidence at index {idx}: {evidence!r}")
            return

        seen_indices.add(idx)
        parsed_updates.append((idx, met, evidence))

    if seen_indices != expected_indices:
        logger.error(
            f"[CriteriaChecker] Rejecting updates: missing indices {sorted(expected_indices - seen_indices)}"
        )
        return

    logger.info("[CriteriaChecker] Updating criteria status")
    for idx, met, evidence in parsed_updates:
        criteria[idx]["met"] = met
        criteria[idx]["evidence"] = evidence
        status = "✅ MET" if met else "❌ NOT MET"
        criteria_text = criteria[idx].get("criteria", "Unknown")
        logger.info(f"[CriteriaChecker] Criterion {idx}: {status}")
        logger.info(f"[CriteriaChecker]   └─ Criteria: {criteria_text}")
        logger.info(f"[CriteriaChecker]   └─ Evidence: {evidence}")

    met_count = sum(1 for c in criteria if c.get("met", False))
    logger.info(f"[CriteriaChecker] Summary: {met_count}/{len(criteria)} criteria met")
    state[StateKeys.HIGH_LEVEL_SUCCESS_CRITERIA] = criteria


def stage_reflector_callback(callback_context: CallbackContext):
    """
    Apply stage modifications and add new stages.

    This callback updates the high_level_stages in state based on the
    stage reflector's recommendations.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context with invocation context access
    """

    ctx = callback_context._invocation_context
    state = ctx.session.state

    reflector_output = state.get(StateKeys.STAGE_REFLECTOR_OUTPUT)
    stages = state.get(StateKeys.HIGH_LEVEL_STAGES, [])

    if not reflector_output:
        logger.error("[StageReflector] No output found in state")
        return

    if not isinstance(reflector_output, dict):
        logger.error(f"[StageReflector] Invalid output type: {type(reflector_output)}")
        return

    if not isinstance(stages, list):
        logger.error(f"[StageReflector] Invalid stages in state: {type(stages)}")
        return

    logger.info("[StageReflector] Processing stage reflections")

    modifications = reflector_output.get("stage_modifications", [])
    new_stages = reflector_output.get("new_stages", [])

    if not isinstance(modifications, list):
        logger.error(f"[StageReflector] stage_modifications is not a list: {type(modifications)}")
        return
    if not isinstance(new_stages, list):
        logger.error(f"[StageReflector] new_stages is not a list: {type(new_stages)}")
        return

    # Strict validation before any state mutation
    validated_modifications = []
    for mod in modifications:
        if not isinstance(mod, dict):
            logger.error(f"[StageReflector] Invalid modification structure: {mod}")
            return
        if "index" not in mod or "new_description" not in mod:
            logger.error(f"[StageReflector] Missing fields in modification: {mod}")
            return

        idx = mod["index"]
        new_desc = mod["new_description"]
        if not isinstance(idx, int):
            logger.error(f"[StageReflector] Invalid modification index type: {type(idx)}")
            return
        if not isinstance(new_desc, str):
            logger.error(f"[StageReflector] Invalid new_description type for stage {idx}: {type(new_desc)}")
            return
        if idx < 0 or idx >= len(stages):
            logger.error(f"[StageReflector] Invalid stage index for modification: {idx}")
            return
        if stages[idx].get("completed", False):
            logger.error(f"[StageReflector] Modification attempted on completed stage {idx}")
            return

        # Empty description means no-op (allowed by schema)
        if new_desc.strip():
            validated_modifications.append((idx, new_desc))

    validated_new_stages: List[Dict[str, Any]] = []
    for new_stage in new_stages:
        if not isinstance(new_stage, dict):
            logger.error(f"[StageReflector] Invalid new stage structure: {new_stage}")
            return
        if "title" not in new_stage or "description" not in new_stage:
            logger.error(f"[StageReflector] Missing fields in new stage: {new_stage}")
            return

        title = new_stage["title"]
        description = new_stage["description"]
        if not isinstance(title, str) or not title.strip():
            logger.error(f"[StageReflector] Invalid new stage title: {title!r}")
            return
        if not isinstance(description, str) or not description.strip():
            logger.error(f"[StageReflector] Invalid new stage description: {description!r}")
            return

        validated_new_stages.append(
            {
                "title": title,
                "description": description,
                "stage_id": new_stage.get("stage_id"),
                "depends_on": new_stage.get("depends_on"),
                "inputs_required": new_stage.get("inputs_required"),
                "outputs_produced": new_stage.get("outputs_produced"),
                "evidence_refs": new_stage.get("evidence_refs"),
            }
        )

    # Apply validated modifications and additions
    for idx, new_desc in validated_modifications:
        stages[idx]["description"] = new_desc
        logger.info(f"[StageReflector] Modified stage {idx} description")

    for new_stage in validated_new_stages:
        new_idx = len(stages)
        stages.append(
            make_stage_record(
                index=new_idx,
                title=new_stage["title"],
                description=new_stage["description"],
                completed=False,
                stage_id=new_stage["stage_id"],
                depends_on=new_stage["depends_on"],
                inputs_required=new_stage["inputs_required"],
                outputs_produced=new_stage["outputs_produced"],
                evidence_refs=new_stage["evidence_refs"],
            )
        )
        logger.info(f"[StageReflector] Added new stage {new_idx}: {new_stage['title']}")

    constraints = normalize_and_validate_stage_constraints(stages, apply_sequential_defaults=True)
    if constraints.errors:
        for error in constraints.errors:
            logger.error(f"[StageReflector] Constraint validation failed: {error}")
        return
    for warning in constraints.warnings:
        logger.warning(f"[StageReflector] Constraint warning: {warning}")

    state[StateKeys.HIGH_LEVEL_STAGES] = constraints.stages


class NonEscalatingLoopAgent(LoopAgent):
    """A loop agent that does not propagate escalate flags upward."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        times_looped = 0
        while not self.max_iterations or times_looped < self.max_iterations:
            for sub_agent in self.sub_agents:
                should_exit = False
                async with Aclosing(sub_agent.run_async(ctx)) as agen:
                    async for event in agen:
                        if event.actions.escalate:
                            event.actions.escalate = False
                            should_exit = True
                        yield event
                        if should_exit:
                            break

                if should_exit:
                    return
            times_looped += 1
        return


def create_agent(
    working_dir: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
) -> LoopDetectionAgent:
    """
    Factory function to create an Agentic Data Scientist ADK agent.

    Parameters
    ----------
    working_dir : str, optional
        Working directory for the session
    mcp_servers : List[str], optional
        List of MCP servers to enable for tools

    Returns
    -------
    LoopDetectionAgent
        The configured root agent
    """
    # Create working directory if not provided
    if working_dir is None:
        import tempfile

        working_dir = tempfile.mkdtemp(prefix="agentic_ds_")

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[AgenticDS] Creating ADK agent with working_dir={working_dir}")

    # Create local tools with working_dir bound via wrapper functions
    from agentic_data_scientist.tools import (
        directory_tree,
        fetch_url,
        get_file_info,
        list_directory,
        read_file,
        read_media_file,
        search_files,
    )

    # Bind working_dir using wrapper functions that completely hide the parameter
    # This ensures ADK sees the correct signature without working_dir
    working_dir_str = str(working_dir)

    def read_file_bound(path: str, head: Optional[int] = None, tail: Optional[int] = None) -> str:
        """Read file contents with optional head/tail line limits."""
        return read_file(path, working_dir_str, head, tail)

    def read_media_file_bound(path: str) -> str:
        """Read binary/media files and return base64 encoded data."""
        return read_media_file(path, working_dir_str)

    def list_directory_bound(path: str = ".", show_sizes: bool = False, sort_by: str = "name") -> str:
        """List directory contents with optional size display and sorting."""
        return list_directory(path, working_dir_str, show_sizes, sort_by)

    def directory_tree_bound(path: str = ".", exclude_patterns: Optional[list[str]] = None) -> str:
        """Generate a recursive directory tree view."""
        return directory_tree(path, working_dir_str, exclude_patterns)

    def search_files_bound(pattern: str, path: str = ".", exclude_patterns: Optional[list[str]] = None) -> str:
        """Search for files matching a pattern."""
        return search_files(pattern, working_dir_str, path, exclude_patterns)

    def get_file_info_bound(path: str) -> str:
        """Get detailed metadata about a file."""
        return get_file_info(path, working_dir_str)

    tools = [
        read_file_bound,
        read_media_file_bound,
        list_directory_bound,
        directory_tree_bound,
        search_files_bound,
        get_file_info_bound,
    ]

    # Only add fetch_url if network access is not disabled
    if not is_network_disabled():
        tools.append(fetch_url)

    logger.info(f"[AgenticDS] Configured {len(tools)} local tools")

    # Resolve role-based models (with routing + startup fallback).
    summary_model, summary_fallback_model, summary_selection = get_litellm_candidates_for_role(
        role="summary_agent",
        default_model_name=DEFAULT_MODEL_NAME,
    )
    plan_maker_model, plan_maker_fallback_model, plan_maker_selection = get_litellm_candidates_for_role(
        role="plan_maker",
        default_model_name=DEFAULT_MODEL_NAME,
    )
    plan_reviewer_model, plan_reviewer_fallback_model, plan_reviewer_selection = get_litellm_candidates_for_role(
        role="plan_reviewer",
        default_model_name=REVIEW_MODEL_NAME,
    )
    plan_parser_model, plan_parser_fallback_model, plan_parser_selection = get_litellm_candidates_for_role(
        role="plan_parser",
        default_model_name=DEFAULT_MODEL_NAME,
    )
    criteria_checker_model, criteria_checker_fallback_model, criteria_checker_selection = get_litellm_candidates_for_role(
        role="criteria_checker",
        default_model_name=REVIEW_MODEL_NAME,
    )
    stage_reflector_model, stage_reflector_fallback_model, stage_reflector_selection = get_litellm_candidates_for_role(
        role="stage_reflector",
        default_model_name=DEFAULT_MODEL_NAME,
    )

    # ------------------------- Implementation Loop -------------------------

    coding_agent, review_agent, review_confirmation = make_implementation_agents(str(working_dir), tools)

    # LoopAgent wrapper for implementation
    implementation_loop = NonEscalatingLoopAgent(
        name="implementation_loop",
        description="Iterative implementation-review-confirmation loop for each stage.",
        sub_agents=[coding_agent, review_agent, review_confirmation],
        max_iterations=5,
    )

    # ------------------------- Summary Agent -------------------------

    logger.info("[AgenticDS] Loading summary_agent prompt")
    summary_agent_instructions = load_prompt("summary")

    logger.info(f"[AgenticDS] Creating summary_agent with model={summary_model}")

    summary_agent = LoopDetectionAgent(
        name="summary_agent",
        model=summary_model,
        fallback_model=summary_fallback_model,
        fallback_max_retries=summary_selection.max_retry,
        routing_role="summary_agent",
        primary_profile_name=(
            summary_selection.selected_profile.name
            if summary_selection.selected_profile is not None
            else summary_selection.primary_model
        ),
        description="Summarizes results into a comprehensive pure text report.",
        instruction=summary_agent_instructions,
        tools=tools,  # Needs tools to read files
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        ),
        generate_content_config=get_generate_content_config(temperature=0.3),
    )

    # ------------------------- High Level Planning Agents -------------------------

    logger.info("[AgenticDS] Loading plan maker agent prompt")
    plan_maker_instructions = load_prompt("plan_maker")

    logger.info(f"[AgenticDS] Creating plan maker agent with model={plan_maker_model}")

    plan_maker_compression = create_compression_callback(
        event_threshold=40,
        overlap_size=20,
        model_name=str(getattr(plan_maker_model, "model", DEFAULT_MODEL_NAME)),
    )

    plan_maker_agent = LoopDetectionAgent(
        name="plan_maker_agent",
        model=plan_maker_model,
        fallback_model=plan_maker_fallback_model,
        fallback_max_retries=plan_maker_selection.max_retry,
        routing_role="plan_maker",
        primary_profile_name=(
            plan_maker_selection.selected_profile.name
            if plan_maker_selection.selected_profile is not None
            else plan_maker_selection.primary_model
        ),
        description="Plan maker agent - creates high-level plans for complex tasks.",
        instruction=plan_maker_instructions,
        tools=tools,
        output_key=StateKeys.HIGH_LEVEL_PLAN,
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        ),
        generate_content_config=get_generate_content_config(temperature=0.6),
        after_agent_callback=plan_maker_compression,
    )

    logger.info("[AgenticDS] Loading plan reviewer agent prompt")
    plan_reviewer_instructions = load_prompt("plan_reviewer")

    logger.info(f"[AgenticDS] Creating plan reviewer agent with model={plan_reviewer_model}")

    plan_reviewer_compression = create_compression_callback(
        event_threshold=40,
        overlap_size=20,
        model_name=str(getattr(plan_reviewer_model, "model", REVIEW_MODEL_NAME)),
    )

    plan_reviewer_agent = LoopDetectionAgent(
        name="plan_reviewer_agent",
        model=plan_reviewer_model,
        fallback_model=plan_reviewer_fallback_model,
        fallback_max_retries=plan_reviewer_selection.max_retry,
        routing_role="plan_reviewer",
        primary_profile_name=(
            plan_reviewer_selection.selected_profile.name
            if plan_reviewer_selection.selected_profile is not None
            else plan_reviewer_selection.primary_model
        ),
        description="Plan reviewer agent - reviews high-level plans for completeness and correctness.",
        instruction=plan_reviewer_instructions,
        tools=tools,
        output_key=StateKeys.PLAN_REVIEW_FEEDBACK,
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        ),
        generate_content_config=get_generate_content_config(temperature=0.3),
        after_agent_callback=plan_reviewer_compression,
    )

    high_level_planning_loop = NonEscalatingLoopAgent(
        name="high_level_planning_loop",
        description="Carries out high-level planning through multiple iterations.",
        sub_agents=[
            plan_maker_agent,
            plan_reviewer_agent,
            create_review_confirmation_agent(
                auto_exit_on_completion=True,
                prompt_name="plan_review_confirmation",
                model=plan_reviewer_model,
                fallback_model=plan_reviewer_fallback_model,
                fallback_max_retries=plan_reviewer_selection.max_retry,
                routing_role="plan_reviewer",
                primary_profile_name=(
                    plan_reviewer_selection.selected_profile.name
                    if plan_reviewer_selection.selected_profile is not None
                    else plan_reviewer_selection.primary_model
                ),
            ),
        ],
        max_iterations=10,
    )

    # ------------------------- High Level Plan Parser -------------------------

    logger.info("[AgenticDS] Loading plan parser prompt")
    plan_parser_instructions = load_prompt("plan_parser")

    logger.info(f"[AgenticDS] Creating plan parser agent with model={plan_parser_model}")

    high_level_plan_parser = LoopDetectionAgent(
        name="high_level_plan_parser",
        model=plan_parser_model,
        fallback_model=plan_parser_fallback_model,
        fallback_max_retries=plan_parser_selection.max_retry,
        routing_role="plan_parser",
        primary_profile_name=(
            plan_parser_selection.selected_profile.name
            if plan_parser_selection.selected_profile is not None
            else plan_parser_selection.primary_model
        ),
        description="Parses high-level plan into stages and success criteria.",
        instruction=plan_parser_instructions,
        tools=[],  # NO TOOLS - pure JSON parsing
        output_schema=PLAN_PARSER_OUTPUT_SCHEMA,
        output_key=StateKeys.PARSED_PLAN_OUTPUT,
        after_agent_callback=plan_parser_callback,
        generate_content_config=get_generate_content_config(temperature=0.0),
    )

    # ------------------------- Success Criteria Checker -------------------------

    logger.info("[AgenticDS] Loading criteria checker prompt")
    criteria_checker_instructions = load_prompt("criteria_checker")

    logger.info(f"[AgenticDS] Creating criteria checker agent with model={criteria_checker_model}")

    criteria_checker_compression = create_compression_callback(
        event_threshold=40,
        overlap_size=20,
        model_name=str(getattr(criteria_checker_model, "model", REVIEW_MODEL_NAME)),
    )

    # Combine compression callback with criteria checker callback
    async def combined_criteria_callback(callback_context):
        # Run criteria checker callback first
        criteria_checker_callback(callback_context)
        # Then run compression callback (async)
        await criteria_checker_compression(callback_context)

    success_criteria_checker = LoopDetectionAgent(
        name="success_criteria_checker",
        model=criteria_checker_model,
        fallback_model=criteria_checker_fallback_model,
        fallback_max_retries=criteria_checker_selection.max_retry,
        routing_role="criteria_checker",
        primary_profile_name=(
            criteria_checker_selection.selected_profile.name
            if criteria_checker_selection.selected_profile is not None
            else criteria_checker_selection.primary_model
        ),
        description="Checks which high-level success criteria have been met.",
        instruction=criteria_checker_instructions,
        tools=tools,  # NEEDS TOOLS to inspect files
        output_schema=CRITERIA_CHECKER_OUTPUT_SCHEMA,
        output_key=StateKeys.CRITERIA_CHECKER_OUTPUT,
        after_agent_callback=combined_criteria_callback,
        generate_content_config=get_generate_content_config(temperature=0.0),
    )

    # ------------------------- Stage Reflector -------------------------

    logger.info("[AgenticDS] Loading stage reflector prompt")
    stage_reflector_instructions = load_prompt("stage_reflector")

    logger.info(f"[AgenticDS] Creating stage reflector agent with model={stage_reflector_model}")

    stage_reflector_compression = create_compression_callback(
        event_threshold=40,
        overlap_size=20,
        model_name=str(getattr(stage_reflector_model, "model", DEFAULT_MODEL_NAME)),
    )

    # Combine compression callback with stage reflector callback
    async def combined_reflector_callback(callback_context):
        # Run stage reflector callback first
        stage_reflector_callback(callback_context)
        # Then run compression callback (async)
        await stage_reflector_compression(callback_context)

    stage_reflector = LoopDetectionAgent(
        name="stage_reflector",
        model=stage_reflector_model,
        fallback_model=stage_reflector_fallback_model,
        fallback_max_retries=stage_reflector_selection.max_retry,
        routing_role="stage_reflector",
        primary_profile_name=(
            stage_reflector_selection.selected_profile.name
            if stage_reflector_selection.selected_profile is not None
            else stage_reflector_selection.primary_model
        ),
        description="Reflects on and adapts remaining implementation stages.",
        instruction=stage_reflector_instructions,
        tools=tools,  # NEEDS TOOLS for context
        output_schema=STAGE_REFLECTOR_OUTPUT_SCHEMA,
        output_key=StateKeys.STAGE_REFLECTOR_OUTPUT,
        after_agent_callback=combined_reflector_callback,
        generate_content_config=get_generate_content_config(temperature=0.4),
    )

    # ------------------------- Stage Orchestrator -------------------------

    logger.info("[AgenticDS] Creating stage orchestrator")

    from agentic_data_scientist.agents.adk.stage_orchestrator import StageOrchestratorAgent

    stage_orchestrator = StageOrchestratorAgent(
        implementation_loop=implementation_loop,
        criteria_checker=success_criteria_checker,
        stage_reflector=stage_reflector,
        name="stage_orchestrator",
        description="Orchestrates stage-by-stage implementation with adaptive planning.",
    )

    # ------------------------- Root Workflow -------------------------

    logger.info("[AgenticDS] Creating root workflow")

    workflow = SequentialAgent(
        name="agentic_data_scientist_workflow",
        description="Complete Agentic Data Scientist workflow with adaptive stage-wise implementation.",
        sub_agents=[
            high_level_planning_loop,
            high_level_plan_parser,
            stage_orchestrator,
            summary_agent,
        ],
    )

    logger.info("[AgenticDS] Agent creation complete")

    return workflow


def create_app(
    working_dir: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
) -> App:
    """
    Create an App instance with context management for the ADK agent.

    Parameters
    ----------
    working_dir : str, optional
        Working directory for the session
    mcp_servers : List[str], optional
        List of MCP servers to enable for tools

    Returns
    -------
    App
        The configured App with context caching and compression
    """
    # Create the root agent
    root_agent = create_agent(working_dir=working_dir, mcp_servers=mcp_servers)

    # Configure context caching (just creating the config enables caching)
    cache_config = ContextCacheConfig()

    # Create App with context caching
    # Note: Event compression is now handled via custom callbacks
    app = App(
        name="agentic-data-scientist",
        root_agent=root_agent,
        context_cache_config=cache_config,
    )

    logger.info("[AgenticDS] Created App with context caching enabled")
    logger.info("[AgenticDS] Event compression will be handled via custom callbacks")

    return app
