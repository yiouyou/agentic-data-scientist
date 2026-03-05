"""
Core API for Agentic Data Scientist - Simplified stateless interface.

This module provides the main DataScientist class for running agents
with optional conversation context and file handling.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentic_data_scientist.core.events import (
    CompletedEvent,
    ErrorEvent,
    FunctionCallEvent,
    FunctionResponseEvent,
    MessageEvent,
    UsageEvent,
    event_to_dict,
)
from agentic_data_scientist.core.history_store import create_history_store_from_env
from agentic_data_scientist.core.skill_registry import discover_skills, format_skill_advice, recommend_skills
from agentic_data_scientist.core.state_contracts import StateKeys, build_initial_state_delta


# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("google_adk.google.adk.tools.base_authenticated_tool").setLevel(logging.ERROR)


@dataclass
class SessionConfig:
    """Configuration for an Agentic Data Scientist session."""

    agent_type: str = "adk"  # "adk" or "claude_code"
    mcp_servers: Optional[List[str]] = None
    max_llm_calls: int = 1024
    session_id: Optional[str] = None
    working_dir: Optional[str] = None
    auto_cleanup: bool = True


@dataclass
class FileInfo:
    """Information about an uploaded file."""

    name: str
    path: str
    size_kb: float


@dataclass
class Result:
    """Result from running an agent."""

    session_id: str
    status: str
    run_id: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    duration: Optional[float] = None
    events_count: int = 0


class DataScientist:
    """
    Simplified stateless API for Agentic Data Scientist agents.

    This class provides a clean interface for running ADK workflows or direct coding agents
    with optional conversation context and file handling.

    Parameters
    ----------
    agent_type : str, optional
        Type of agent to use: "adk" or "claude_code" (default: "adk")
    mcp_servers : List[str], optional
        List of MCP servers to enable
    working_dir : str, optional
        Working directory for the session. If not provided, defaults to
        "./agentic_output/" in the current directory
    auto_cleanup : bool, optional
        Whether to automatically cleanup the working directory after completion.
        Defaults to False (files are preserved)
    """

    def __init__(
        self,
        agent_type: str = "adk",
        mcp_servers: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        auto_cleanup: Optional[bool] = None,
    ):
        """Initialize Agentic Data Scientist core with configuration."""
        # Generate session ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        self.session_id = f"session_{timestamp}_{unique_id}"

        # Set up working directory
        if working_dir:
            self.working_dir = Path(working_dir)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            self._user_provided_dir = True
            # Default: don't cleanup user-provided directories
            self.auto_cleanup = auto_cleanup if auto_cleanup is not None else False
        else:
            # Default to ./agentic_output/ subdirectory in current directory
            self.working_dir = Path("./agentic_output")
            self.working_dir.mkdir(parents=True, exist_ok=True)
            self._user_provided_dir = False
            # Default: don't cleanup default directory
            self.auto_cleanup = auto_cleanup if auto_cleanup is not None else False

        self.config = SessionConfig(
            agent_type=agent_type,
            mcp_servers=mcp_servers,
            working_dir=str(self.working_dir),
            auto_cleanup=self.auto_cleanup,
        )

        # ADK components
        self.agent = None
        self.app = None  # Will store App instance for ADK agents
        self.session_service = None
        self.runner = None
        self._history_store = create_history_store_from_env()

        logger.info(f"Initialized Agentic Data Scientist session: {self.session_id}")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Auto-cleanup enabled: {self.auto_cleanup}")
        if self._history_store is not None:
            logger.info(f"History store enabled: {self._history_store.db_path}")

    def _planner_advice_enabled(self) -> bool:
        raw = os.getenv("ADS_LEARNING_ADVICE_ENABLED", "true").strip().lower()
        return raw not in {"0", "false", "off", "no"}

    def _build_planner_history_advice(self, *, user_message: str) -> str:
        """Build advice-only planner guidance from history store."""
        if self._history_store is None:
            return ""
        if not self._planner_advice_enabled():
            return ""
        if not user_message.strip():
            return ""

        top_k_raw = os.getenv("ADS_LEARNING_TOPK", "3").strip()
        recent_limit_raw = os.getenv("ADS_LEARNING_RECENT_RUNS", "200").strip()
        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            top_k = 3
        try:
            recent_limit = max(top_k, int(recent_limit_raw))
        except Exception:
            recent_limit = 200

        try:
            return self._history_store.build_planner_advice(
                user_request=user_message,
                k=top_k,
                recent_limit=recent_limit,
            )
        except Exception as advice_error:
            logger.warning(f"Failed to build planner history advice: {advice_error}")
            return ""

    def _build_planner_history_signals(self, *, user_message: str) -> Dict[str, Any]:
        """Build structured planner signals from history store."""
        if self._history_store is None:
            return {}
        if not self._planner_advice_enabled():
            return {}
        if not user_message.strip():
            return {}

        top_k_raw = os.getenv("ADS_LEARNING_TOPK", "3").strip()
        recent_limit_raw = os.getenv("ADS_LEARNING_RECENT_RUNS", "200").strip()
        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            top_k = 3
        try:
            recent_limit = max(top_k, int(recent_limit_raw))
        except Exception:
            recent_limit = 200

        try:
            return self._history_store.build_planner_signals(
                user_request=user_message,
                k=top_k,
                recent_limit=recent_limit,
            )
        except Exception as signal_error:
            logger.warning(f"Failed to build planner history signals: {signal_error}")
            return {}

    def _planner_skill_advice_enabled(self) -> bool:
        raw = os.getenv("ADS_PLANNER_SKILL_ADVICE_ENABLED", "true").strip().lower()
        return raw not in {"0", "false", "off", "no"}

    def _build_planner_skill_advice(self, *, user_message: str) -> str:
        """Build skill-inventory guidance for plan generation."""
        if not self._planner_skill_advice_enabled():
            return ""
        if not user_message.strip():
            return ""

        top_k_raw = os.getenv("ADS_PLANNER_SKILL_TOPK", "8").strip()
        min_score_raw = os.getenv("ADS_PLANNER_SKILL_MIN_SCORE", "0.12").strip()
        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            top_k = 8
        try:
            min_score = max(0.0, float(min_score_raw))
        except Exception:
            min_score = 0.12

        try:
            recommendations = recommend_skills(
                query=user_message,
                working_dir=str(self.working_dir),
                top_k=top_k,
                min_score=min_score,
            )
            inventory_count = len(discover_skills(working_dir=str(self.working_dir)))
            return format_skill_advice(recommendations, total_skills=inventory_count)
        except Exception as advice_error:
            logger.warning(f"Failed to build planner skill advice: {advice_error}")
            return ""

    def _collect_decision_traces(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect compact decision traces from canonical state keys."""
        decisions: List[Dict[str, Any]] = []
        if not isinstance(state, dict):
            return decisions

        plan_decision = state.get(StateKeys.PLAN_REVIEW_CONFIRMATION_DECISION)
        if plan_decision is not None:
            reason = plan_decision.get("reason") if isinstance(plan_decision, dict) else ""
            decisions.append(
                {
                    "role": "plan_review_confirmation",
                    "decision_key": StateKeys.PLAN_REVIEW_CONFIRMATION_DECISION,
                    "decision_value": plan_decision,
                    "reason": reason,
                    "source": "workflow_state",
                }
            )

        impl_decision = state.get(StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION)
        if impl_decision is not None:
            reason = impl_decision.get("reason") if isinstance(impl_decision, dict) else ""
            decisions.append(
                {
                    "role": "implementation_review_confirmation",
                    "decision_key": StateKeys.IMPLEMENTATION_REVIEW_CONFIRMATION_DECISION,
                    "decision_value": impl_decision,
                    "reason": reason,
                    "source": "workflow_state",
                }
            )
        plan_selection = state.get(StateKeys.PLAN_SELECTION_TRACE)
        if plan_selection is not None:
            reason = plan_selection.get("reason") if isinstance(plan_selection, dict) else ""
            decisions.append(
                {
                    "role": "plan_selector",
                    "decision_key": "plan_selector_ranking",
                    "decision_value": plan_selection,
                    "reason": reason,
                    "source": "workflow_state",
                }
            )
        return decisions

    async def _persist_history(
        self,
        *,
        run_id: Optional[str],
        start_time: datetime,
        status: str,
        duration: float,
        events_count: int,
        files_created: List[str],
        usage_totals: Dict[str, int],
        error_text: Optional[str] = None,
    ) -> None:
        """Persist compact run/stage/decision history (best effort)."""
        if self._history_store is None or not run_id:
            return
        if self.session_service is None:
            return

        try:
            app_name = self.app.name if self.app else "agentic_data_scientist"
            session = await self.session_service.get_session(
                app_name=app_name,
                user_id="default_user",
                session_id=self.session_id,
            )
            state = session.state if session is not None else {}
            if not isinstance(state, dict):
                state = {}

            self._history_store.record_run(
                run_id=run_id,
                session_id=self.session_id,
                started_at=start_time.isoformat(timespec="seconds"),
                finished_at=datetime.now().isoformat(timespec="seconds"),
                status=status,
                agent_type=self.config.agent_type,
                duration_seconds=duration,
                events_count=events_count,
                files_count=len(files_created),
                total_input_tokens=int(usage_totals.get("total_input_tokens", 0)),
                cached_input_tokens=int(usage_totals.get("cached_input_tokens", 0)),
                output_tokens=int(usage_totals.get("output_tokens", 0)),
                error_text=error_text,
                working_dir=str(self.working_dir),
            )

            stage_attempts = state.get(StateKeys.STAGE_IMPLEMENTATIONS, [])
            if isinstance(stage_attempts, list):
                stages = state.get(StateKeys.HIGH_LEVEL_STAGES, [])
                stages_by_index: Dict[int, Dict[str, Any]] = {}
                if isinstance(stages, list):
                    for stage in stages:
                        if not isinstance(stage, dict):
                            continue
                        try:
                            stage_index = int(stage.get("index", -1))
                        except Exception:
                            continue
                        stages_by_index[stage_index] = stage
                self._history_store.record_stage_outcomes(
                    run_id=run_id,
                    stage_attempts=stage_attempts,
                    stages_by_index=stages_by_index,
                )

            decisions = self._collect_decision_traces(state)
            self._history_store.record_decision_traces(run_id=run_id, decisions=decisions)

        except Exception as history_error:
            logger.warning(f"Failed to persist history for run {run_id}: {history_error}")

    async def _setup_agent(self):
        """Set up the agent and session service."""
        if self.agent is not None:
            return  # Already set up

        if self.config.agent_type == "adk":
            from agentic_data_scientist.agents.adk import create_app

            # Create App instead of bare agent
            app = create_app(
                working_dir=str(self.working_dir),
                mcp_servers=self.config.mcp_servers,
            )

            # Store both app and agent references
            self.app = app
            self.agent = app.root_agent  # For compatibility

        elif self.config.agent_type == "claude_code":
            from google.adk.apps import App
            from google.adk.apps.app import EventsCompactionConfig

            from agentic_data_scientist.agents.coding_backends import create_coding_agent

            # Create direct coding agent (executor defaults to claude_code)
            direct_executor = os.getenv("DIRECT_CODING_EXECUTOR", "claude_code")
            coding_agent = create_coding_agent(
                executor=direct_executor,
                working_dir=str(self.working_dir),
                name="direct_coding_agent",
                description=(
                    "Direct coding agent for simple mode. "
                    f"Executor={direct_executor}"
                ),
                model=os.getenv("CODING_MODEL", "claude-sonnet-4-6"),
                fallback_model=None,
                fallback_max_retries=0,
                routing_role="execution_agent",
                primary_profile_name=os.getenv("CODING_MODEL", "claude-sonnet-4-6"),
                output_key=StateKeys.IMPLEMENTATION_SUMMARY,
            )
            self.agent = coding_agent

            # Create App with compression config (no caching for claude_code)
            compression_config = EventsCompactionConfig(
                summarizer=None,
                compaction_interval=3,  # Compress every 3 invocations
                overlap_size=2,
            )

            self.app = App(
                name="agentic-data-scientist-coding",
                root_agent=coding_agent,
                events_compaction_config=compression_config,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")

        # Create session service
        self.session_service = InMemorySessionService()

        # Get app_name from app if available, otherwise use default
        app_name = self.app.name if self.app else "agentic_data_scientist"

        # Pre-create the session
        session = await self.session_service.create_session(
            app_name=app_name,
            user_id="default_user",
            session_id=self.session_id,
        )
        self.session = session

        # Create runner with App if available
        if self.app:
            self.runner = Runner(
                app=self.app,  # Pass App instead of agent
                session_service=self.session_service,
            )
        else:
            # Fallback for claude_code (though we should always have app now)
            self.runner = Runner(
                agent=self.agent,
                app_name="agentic_data_scientist",
                session_service=self.session_service,
            )

        logger.info(f"Agent setup complete: {self.config.agent_type}")

    def save_files(self, files: List[tuple]) -> List[FileInfo]:
        """
        Save files to the working directory.

        Parameters
        ----------
        files : List[tuple]
            List of (filename, content) tuples where content can be bytes or Path

        Returns
        -------
        List[FileInfo]
            List of saved file information
        """
        user_data_dir = self.working_dir / "user_data"
        user_data_dir.mkdir(parents=True, exist_ok=True)

        file_info_list = []
        for filename, content in files:
            file_path = user_data_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(content, (bytes, bytearray)):
                file_path.write_bytes(content)
                size_kb = len(content) / 1024
            elif isinstance(content, (str, Path)):
                source_path = Path(content)
                if not source_path.exists():
                    raise FileNotFoundError(f"Source file not found: {source_path}")
                file_path.write_bytes(source_path.read_bytes())
                size_kb = source_path.stat().st_size / 1024
            else:
                raise TypeError(f"Invalid content type for {filename}: {type(content)}")

            file_info = FileInfo(name=filename, path=str(file_path), size_kb=size_kb)
            file_info_list.append(file_info)
            logger.info(f"Saved file: {filename} ({size_kb:.1f} KB)")

        return file_info_list

    def prepare_prompt(self, message: str, file_info: Optional[List[FileInfo]] = None) -> str:
        """
        Prepare the prompt with optional file information.

        Parameters
        ----------
        message : str
            User's message
        file_info : List[FileInfo], optional
            List of uploaded files

        Returns
        -------
        str
            Complete prompt with file information
        """
        if not file_info:
            return message

        prompt_parts = [message, "", "=" * 60, "USER DATA FILES:"]
        prompt_parts.append(f"The following files are available in your workspace at: {self.working_dir}/user_data/")
        prompt_parts.append("")

        for info in file_info:
            prompt_parts.append(f"- user_data/{info.name} ({info.size_kb:.1f} KB)")

        prompt_parts.extend(
            [
                "",
                "These files are in your workspace under the 'user_data' folder.",
                "You can directly read and analyze them.",
                "=" * 60,
                "",
            ]
        )

        return "\n".join(prompt_parts)

    async def run_async(
        self,
        message: str,
        files: Optional[List[tuple]] = None,
        stream: bool = False,
        context: Optional[Dict] = None,
    ) -> Union[Result, AsyncGenerator[Dict[str, Any], None]]:
        """
        Run agent asynchronously.

        Parameters
        ----------
        message : str
            User's message/prompt
        files : List[tuple], optional
            List of (filename, content) tuples
        stream : bool, optional
            If True, return an async generator for streaming responses
        context : Dict, optional
            Optional conversation context (not implemented yet)

        Returns
        -------
        Union[Result, AsyncGenerator]
            Result if stream=False, or AsyncGenerator if stream=True
        """
        start_time = datetime.now()
        run_id = f"run_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        try:
            # Set up agent if not already done
            await self._setup_agent()

            # Initialize session state EARLY before any agent execution
            # Get the session from session_service to ensure we're modifying the right instance
            app_name = self.app.name if self.app else "agentic_data_scientist"
            session = await self.session_service.get_session(
                app_name=app_name, user_id="default_user", session_id=self.session_id
            )

            # Keep semantic separation:
            # - original/latest_user_input: raw user message only
            # - rendered_prompt: message enriched with file context/instructions
            session.state[StateKeys.ORIGINAL_USER_INPUT] = message
            session.state[StateKeys.LATEST_USER_INPUT] = message
            planner_signals = self._build_planner_history_signals(user_message=message)
            planner_advice = self._build_planner_history_advice(user_message=message)
            planner_skill_advice = self._build_planner_skill_advice(user_message=message)
            session.state[StateKeys.PLANNER_HISTORY_SIGNALS] = (
                json.dumps(planner_signals, ensure_ascii=True) if planner_signals else ""
            )
            session.state[StateKeys.PLANNER_HISTORY_ADVICE] = planner_advice
            session.state[StateKeys.PLANNER_SKILL_ADVICE] = planner_skill_advice

            # Save files if provided
            file_info = self.save_files(files) if files else None

            # Prepare prompt
            full_prompt = self.prepare_prompt(message, file_info)

            # Save rendered prompt after file metadata has been attached
            session.state[StateKeys.RENDERED_PROMPT] = full_prompt
            if self.config.agent_type == "claude_code":
                session.state[StateKeys.IMPLEMENTATION_TASK] = full_prompt

            logger.info(f"[API] Set session state keys: {list(session.state.keys())}")
            logger.info(
                f"[API] implementation_task = {session.state.get(StateKeys.IMPLEMENTATION_TASK, 'NOT SET')[:50]}..."
            )
            if session.state.get(StateKeys.PLANNER_HISTORY_ADVICE):
                logger.info("[API] planner history advice injected for planning agents")
            if session.state.get(StateKeys.PLANNER_SKILL_ADVICE):
                logger.info("[API] planner skill advice injected for planning agents")

            if stream:
                return self._stream_responses(message, full_prompt, start_time, run_id=run_id)
            else:
                return await self._collect_responses(message, full_prompt, start_time, run_id=run_id)

        except Exception as e:
            logger.error(f"Error in run_async: {e}", exc_info=True)
            if not stream:
                return Result(
                    session_id=self.session_id,
                    status="error",
                    run_id=run_id,
                    error=str(e),
                    duration=(datetime.now() - start_time).total_seconds(),
                )
            else:
                raise

    async def _stream_responses(
        self,
        original_message: str,
        prompt: str,
        start_time: datetime,
        run_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from the agent."""
        event_count = 0
        message_event_number = 0
        responses = []
        usage_totals = {
            "total_input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
        }

        try:
            # Pass initial state to runner via state_delta
            initial_state = build_initial_state_delta(
                original_message=original_message,
                rendered_prompt=prompt,
                agent_type=self.config.agent_type,
            )

            async for event in self.runner.run_async(
                user_id="default_user",
                session_id=self.session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                state_delta=initial_state,
            ):
                event_count += 1

                # Process event content
                if hasattr(event, 'author') and hasattr(event, 'content'):
                    if event.content and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            # Handle text content
                            if hasattr(part, 'text') and part.text:
                                is_thought = hasattr(part, 'thought') and part.thought is True
                                is_partial = getattr(event, 'partial', False)

                                message_event_number += 1
                                msg_event = MessageEvent(
                                    content=part.text,
                                    author=event.author,
                                    timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                    is_thought=is_thought,
                                    is_partial=is_partial,
                                    event_number=message_event_number,
                                )
                                yield event_to_dict(msg_event)
                                responses.append(f"[{event.author}]: {part.text}")

                            # Handle function calls
                            if hasattr(part, 'function_call') and part.function_call:
                                fc = part.function_call
                                if fc.name and fc.name.strip():
                                    args = {}
                                    if hasattr(fc, 'args') and fc.args:
                                        try:
                                            import json

                                            args = json.loads(fc.args) if isinstance(fc.args, str) else fc.args
                                        except Exception:
                                            args = {'raw': str(fc.args)}

                                    message_event_number += 1
                                    func_call_event = FunctionCallEvent(
                                        name=fc.name,
                                        arguments=args,
                                        author=event.author,
                                        timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                        event_number=message_event_number,
                                    )
                                    yield event_to_dict(func_call_event)

                            # Handle function responses
                            if hasattr(part, 'function_response') and part.function_response:
                                fr = part.function_response
                                message_event_number += 1
                                func_resp_event = FunctionResponseEvent(
                                    name=fr.name,
                                    response=fr.response,
                                    author=event.author,
                                    timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                    event_number=message_event_number,
                                )
                                yield event_to_dict(func_resp_event)

                # Handle usage metadata
                if hasattr(event, 'usage_metadata') and event.usage_metadata:
                    usage = event.usage_metadata
                    if isinstance(usage, types.GenerateContentResponseUsageMetadata):
                        usage_info = {
                            'total_input_tokens': usage.total_token_count,
                            'cached_input_tokens': usage.cached_content_token_count,
                            'output_tokens': usage.candidates_token_count,
                        }
                        usage_event = UsageEvent(
                            usage=usage_info, timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        )
                        yield event_to_dict(usage_event)
                        usage_totals["total_input_tokens"] += int(usage_info.get("total_input_tokens") or 0)
                        usage_totals["cached_input_tokens"] += int(usage_info.get("cached_input_tokens") or 0)
                        usage_totals["output_tokens"] += int(usage_info.get("output_tokens") or 0)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Find created files (exclude hidden directories like .venv, .claude)
            files_created = []
            if self.working_dir.exists():
                for file_path in self.working_dir.rglob('*'):
                    if file_path.is_file() and 'user_data' not in file_path.parts:
                        # Exclude hidden directories (starting with .)
                        if not any(part.startswith('.') for part in file_path.parts):
                            relative_path = file_path.relative_to(self.working_dir)
                            files_created.append(str(relative_path))

            # Final completed event
            completed_event = CompletedEvent(
                session_id=self.session_id,
                duration=duration,
                total_events=message_event_number,
                files_created=files_created,
                files_count=len(files_created),
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            )
            await self._persist_history(
                run_id=run_id,
                start_time=start_time,
                status="completed",
                duration=duration,
                events_count=event_count,
                files_created=files_created,
                usage_totals=usage_totals,
            )
            yield event_to_dict(completed_event)

        except Exception as e:
            logger.error(f"Error in stream: {e}", exc_info=True)
            await self._persist_history(
                run_id=run_id,
                start_time=start_time,
                status="error",
                duration=(datetime.now() - start_time).total_seconds(),
                events_count=event_count,
                files_created=[],
                usage_totals=usage_totals,
                error_text=str(e),
            )
            error_event = ErrorEvent(content=str(e), timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3])
            yield event_to_dict(error_event)

    async def _collect_responses(
        self,
        original_message: str,
        prompt: str,
        start_time: datetime,
        run_id: Optional[str] = None,
    ) -> Result:
        """Collect all responses and return a complete result."""
        responses = []
        event_count = 0
        usage_totals = {
            "total_input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
        }

        try:
            # Pass initial state to runner via state_delta
            initial_state = build_initial_state_delta(
                original_message=original_message,
                rendered_prompt=prompt,
                agent_type=self.config.agent_type,
            )

            async for event in self.runner.run_async(
                user_id="default_user",
                session_id=self.session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                state_delta=initial_state,
            ):
                event_count += 1

                # Collect text outputs
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                is_thought = hasattr(part, 'thought') and part.thought is True
                                author = getattr(event, 'author', 'agent')
                                prefix = f"[{author}]" if not is_thought else f"[{author} - THINKING]"
                                responses.append(f"{prefix}: {part.text}")

                if hasattr(event, 'usage_metadata') and event.usage_metadata:
                    usage = event.usage_metadata
                    if isinstance(usage, types.GenerateContentResponseUsageMetadata):
                        usage_totals["total_input_tokens"] += int(usage.total_token_count or 0)
                        usage_totals["cached_input_tokens"] += int(usage.cached_content_token_count or 0)
                        usage_totals["output_tokens"] += int(usage.candidates_token_count or 0)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Find created files (exclude hidden directories like .venv, .claude)
            files_created = []
            if self.working_dir.exists():
                for file_path in self.working_dir.rglob('*'):
                    if file_path.is_file() and 'user_data' not in file_path.parts:
                        # Exclude hidden directories (starting with .)
                        if not any(part.startswith('.') for part in file_path.parts):
                            relative_path = file_path.relative_to(self.working_dir)
                            files_created.append(str(relative_path))

            await self._persist_history(
                run_id=run_id,
                start_time=start_time,
                status="completed",
                duration=duration,
                events_count=event_count,
                files_created=files_created,
                usage_totals=usage_totals,
            )
            return Result(
                session_id=self.session_id,
                status="completed",
                run_id=run_id,
                response="\n".join(responses),
                files_created=files_created,
                duration=duration,
                events_count=event_count,
            )

        except Exception as e:
            logger.error(f"Error collecting responses: {e}", exc_info=True)
            await self._persist_history(
                run_id=run_id,
                start_time=start_time,
                status="error",
                duration=(datetime.now() - start_time).total_seconds(),
                events_count=event_count,
                files_created=[],
                usage_totals=usage_totals,
                error_text=str(e),
            )
            return Result(
                session_id=self.session_id,
                status="error",
                run_id=run_id,
                error=str(e),
                duration=(datetime.now() - start_time).total_seconds(),
            )

    def run(self, message: str, files: Optional[List[tuple]] = None, **kwargs) -> Result:
        """
        Synchronous wrapper for run_async.

        Parameters
        ----------
        message : str
            User's message/prompt
        files : List[tuple], optional
            List of (filename, content) tuples
        **kwargs
            Additional arguments passed to run_async

        Returns
        -------
        Result
            The complete response
        """
        return asyncio.run(self.run_async(message, files, stream=False, **kwargs))

    def cleanup(self):
        """Clean up working directory if auto_cleanup is enabled."""
        if not self.auto_cleanup:
            logger.info(f"Auto-cleanup disabled. Working directory preserved at: {self.working_dir}")
            return

        if self.working_dir and self.working_dir.exists():
            import shutil

            try:
                shutil.rmtree(self.working_dir)
                logger.info(f"Cleaned up working directory: {self.working_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up working directory: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.cleanup()
