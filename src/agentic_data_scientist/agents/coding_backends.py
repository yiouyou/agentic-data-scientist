"""Coding backend selection and non-Claude coding agent implementations."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence

from google.adk.agents import Agent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agentic_data_scientist.core.llm_circuit_breaker import (
    get_llm_circuit_breaker,
    is_retryable_llm_error,
)
from agentic_data_scientist.core.llm_config import LLMProfile, SUPPORTED_CODING_EXECUTORS
from agentic_data_scientist.core.state_contracts import StateKeys


logger = logging.getLogger(__name__)

DEFAULT_CODING_EXECUTOR = "claude_code"


def normalize_coding_executor(executor: str | None) -> str:
    """Normalize coding executor name with backward-compatible aliases."""
    value = (executor or "").strip().lower()
    if not value:
        return ""
    aliases = {
        "claude": "claude_code",
        "claudecode": "claude_code",
        "claude_code": "claude_code",
        "codex": "codex",
        "openai_codex": "codex",
        "opencode": "opencode",
    }
    normalized = aliases.get(value, value)
    if normalized not in SUPPORTED_CODING_EXECUTORS:
        return ""
    return normalized


def resolve_coding_executor(
    profile: Optional[LLMProfile],
    model_name: str,
    default_executor: str = DEFAULT_CODING_EXECUTOR,
) -> str:
    """
    Resolve which coding executor should run one coding profile/model.

    Priority:
    1) `profile.coding_executor` if configured
    2) anthropic/claude heuristic
    3) `DEFAULT_CODING_EXECUTOR` env var (if valid)
    4) hard default: `claude_code`
    """
    configured = normalize_coding_executor(profile.coding_executor if profile is not None else "")
    if configured:
        return configured

    provider = (profile.provider if profile is not None else "").strip().lower()
    model = (model_name or "").strip().lower()
    if provider == "anthropic" or model.startswith("claude"):
        return "claude_code"

    env_default = normalize_coding_executor(os.getenv("DEFAULT_CODING_EXECUTOR", default_executor))
    return env_default or DEFAULT_CODING_EXECUTOR


@dataclass
class CodingBackendRoute:
    """Resolved coding executor(s) for primary and optional fallback model."""

    primary_executor: str
    fallback_executor: Optional[str]
    fallback_enabled: bool
    fallback_reason: str = ""


def resolve_coding_backend_route(
    *,
    primary_profile: Optional[LLMProfile],
    primary_model: str,
    fallback_profile: Optional[LLMProfile],
    fallback_model: Optional[str],
    max_retry: int,
) -> CodingBackendRoute:
    """Resolve backend route and decide whether fallback is executable."""
    primary_executor = resolve_coding_executor(primary_profile, primary_model)

    if not fallback_model or max_retry <= 0:
        return CodingBackendRoute(
            primary_executor=primary_executor,
            fallback_executor=None,
            fallback_enabled=False,
            fallback_reason="fallback disabled by route config",
        )

    fallback_executor = resolve_coding_executor(fallback_profile, fallback_model)
    if fallback_executor != primary_executor:
        return CodingBackendRoute(
            primary_executor=primary_executor,
            fallback_executor=fallback_executor,
            fallback_enabled=False,
            fallback_reason=(
                f"cross-executor fallback is not supported yet: "
                f"{primary_executor} -> {fallback_executor}"
            ),
        )

    return CodingBackendRoute(
        primary_executor=primary_executor,
        fallback_executor=fallback_executor,
        fallback_enabled=True,
        fallback_reason="",
    )


class ExternalCLICodeAgent(Agent):
    """Coding agent that delegates execution to an external CLI tool."""

    model_config = {"extra": "allow"}

    _working_dir: Optional[str] = None
    _output_key: str = "implementation_summary"
    _fallback_model: Optional[str] = None
    _fallback_max_retries: int = 1
    _fallback_retrying: bool = False
    _routing_role: Optional[str] = None
    _primary_profile_name: Optional[str] = None
    _primary_model: Optional[str] = None
    _backend_name: str = "external_cli"
    _command_template_env: str = ""
    _default_command_template: str = ""

    def __init__(
        self,
        *,
        name: str,
        description: Optional[str],
        working_dir: Optional[str],
        model: Optional[str],
        fallback_model: Optional[str],
        fallback_max_retries: int,
        routing_role: Optional[str],
        primary_profile_name: Optional[str],
        output_key: str,
        after_agent_callback: Optional[Any] = None,
        **kwargs: Any,
    ):
        resolved_model = model or os.getenv("CODING_MODEL", "claude-sonnet-4-6")
        super().__init__(
            name=name,
            description=description or "A coding agent that uses an external coding CLI",
            model=resolved_model,
            after_agent_callback=after_agent_callback,
            **kwargs,
        )
        self._working_dir = working_dir
        self._output_key = output_key
        self._fallback_model = fallback_model
        self._fallback_max_retries = max(0, int(fallback_max_retries))
        self._routing_role = routing_role
        self._primary_profile_name = primary_profile_name
        self._primary_model = str(resolved_model)

    def _truncate_summary(self, summary: str) -> str:
        max_chars = 40000
        if not summary or len(summary) <= max_chars:
            return summary
        keep_start = max_chars * 3 // 4
        keep_end = max_chars // 4
        return (
            summary[:keep_start]
            + "\n\n[... middle section truncated to fit token limits ...]\n\n"
            + summary[-keep_end:]
        )

    def _should_retry_with_fallback(self, error: Exception) -> bool:
        if self._fallback_max_retries <= 0:
            return False
        if not self._fallback_model:
            return False
        if self._fallback_retrying:
            return False
        if self._fallback_model == self.model:
            return False
        return is_retryable_llm_error(error)

    def _build_prompt(self, ctx: InvocationContext, working_dir: str) -> str:
        state = ctx.session.state
        current_stage = state.get(StateKeys.CURRENT_STAGE)
        if current_stage:
            from agentic_data_scientist.agents.claude_code.templates import get_claude_context

            stage_info = (
                f"Stage {current_stage.get('index', 0) + 1}: {current_stage.get('title', 'Unknown')}\n\n"
                f"{current_stage.get('description', '')}"
            )
            prompt = get_claude_context(
                implementation_plan=stage_info,
                working_dir=working_dir,
                original_request=state.get(StateKeys.ORIGINAL_USER_INPUT, ""),
                completed_stages=state.get(StateKeys.STAGE_IMPLEMENTATIONS, []),
                all_stages=state.get(StateKeys.HIGH_LEVEL_STAGES, []),
            )
        else:
            task_prompt = (
                state.get(StateKeys.IMPLEMENTATION_TASK, "")
                or state.get(StateKeys.ORIGINAL_USER_INPUT, "")
                or state.get(StateKeys.LATEST_USER_INPUT, "")
                or state.get(StateKeys.USER_MESSAGE, "")
            )
            if not task_prompt and hasattr(ctx, "initial_message"):
                initial_msg = ctx.initial_message
                if initial_msg and hasattr(initial_msg, "parts"):
                    for part in initial_msg.parts:
                        if hasattr(part, "text"):
                            task_prompt = part.text
                            break
            if not task_prompt:
                raise ValueError("No implementation task or plan found in state.")
            prompt = f"""Create and execute a comprehensive implementation plan.

User Request: {task_prompt}

Working directory: {working_dir}

Requirements:
1. Analyze the request and create a structured plan
2. Execute the plan step by step
3. Save all outputs with descriptive filenames
4. Generate comprehensive documentation
5. Create final execution summary when done"""

        # Add explicit skill guidance for non-Claude native skill runtimes.
        skill_guidance = """

SKILL EXECUTION MODE:
- Discover reusable skill instructions from:
  - .claude/skills/
  - .codex/skills/ (if present)
  - ~/.codex/skills/ (if accessible)
- Before implementing, inspect relevant SKILL.md files and follow them.
- In README.md, document which skills were used and how.
"""
        return prompt + skill_guidance

    def _render_command(self, prompt: str, model: str, working_dir: str) -> tuple[Sequence[str], bytes | None]:
        template = os.getenv(self._command_template_env, self._default_command_template).strip()
        if not template:
            raise RuntimeError(
                f"{self._backend_name} command template is not configured. "
                f"Set {self._command_template_env}."
            )

        send_stdin: bytes | None = prompt.encode("utf-8")
        if "{prompt}" in template:
            rendered = template.format(model=model, working_dir=working_dir, prompt=prompt)
            send_stdin = None
        else:
            rendered = template.format(model=model, working_dir=working_dir)

        command = shlex.split(rendered, posix=(os.name != "nt"))
        if not command:
            raise RuntimeError(f"{self._backend_name} resolved command is empty")
        return command, send_stdin

    async def _invoke_external_cli(self, *, prompt: str, model: str, working_dir: str) -> tuple[str, str]:
        command, stdin_payload = self._render_command(prompt=prompt, model=model, working_dir=working_dir)
        logger.info(f"[CodingBackend] Running {self._backend_name} command: {' '.join(command)}")

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=working_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        stdout_raw, stderr_raw = await process.communicate(input=stdin_payload)
        stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_raw.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            error_tail = (stderr_text or stdout_text or "no output")[-1200:]
            raise RuntimeError(
                f"{self._backend_name} command failed with exit code {process.returncode}. "
                f"Tail output: {error_tail}"
            )

        return stdout_text, stderr_text

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        primary_model = self._primary_model or str(self.model)
        primary_identity = self._primary_profile_name or primary_model
        forced_fallback = False
        had_exception = False
        retryable_primary_failure = False

        if self._fallback_retrying:
            forced_fallback = True
        else:
            self.model = primary_model
            if (
                self._routing_role
                and self._fallback_model
                and self._fallback_max_retries > 0
                and get_llm_circuit_breaker().should_force_fallback(self._routing_role, primary_identity)
            ):
                self.model = str(self._fallback_model)
                forced_fallback = True

        try:
            working_dir = self._working_dir
            if not working_dir:
                import tempfile

                working_dir = tempfile.mkdtemp(prefix="coding_session_")

            from agentic_data_scientist.agents.claude_code.agent import setup_working_directory
            from agentic_data_scientist.agents.claude_code.templates import get_claude_instructions

            setup_working_directory(working_dir)

            system_instructions = get_claude_instructions(state=state, working_dir=working_dir)
            prompt = self._build_prompt(ctx, working_dir)
            full_prompt = f"{system_instructions}\n\n{prompt}"

            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                f"Starting {self._backend_name} coding backend with model: {self.model}"
                            )
                        )
                    ],
                ),
            )

            stdout_text, stderr_text = await self._invoke_external_cli(
                prompt=full_prompt,
                model=str(self.model),
                working_dir=working_dir,
            )

            combined_output = stdout_text
            if stderr_text:
                combined_output = (combined_output + "\n\n[stderr]\n" + stderr_text).strip()

            if not combined_output:
                combined_output = f"{self._backend_name} completed with no textual output."

            state[self._output_key] = self._truncate_summary(combined_output)
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=combined_output)],
                ),
            )
            return

        except Exception as e:
            had_exception = True
            if not forced_fallback and self._routing_role and is_retryable_llm_error(e):
                retryable_primary_failure = True
                get_llm_circuit_breaker().record_retryable_failure(
                    role=self._routing_role,
                    profile=primary_identity,
                    error=e,
                )

            if self._should_retry_with_fallback(e):
                current_model = str(self.model)
                fallback_model = str(self._fallback_model)
                self._fallback_retrying = True
                self.model = fallback_model
                try:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        f"Primary coding model failed ({current_model}). "
                                        f"Retrying with fallback model ({fallback_model})..."
                                    )
                                )
                            ],
                        ),
                    )
                    async for fallback_event in self._run_async_impl(ctx):
                        yield fallback_event
                    return
                finally:
                    self.model = current_model
                    self._fallback_retrying = False

            state[self._output_key] = self._truncate_summary(f"Error: {str(e)}")
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"Error: {str(e)}")],
                ),
            )
        finally:
            if not self._fallback_retrying:
                self.model = primary_model
            if (
                self._routing_role
                and not forced_fallback
                and not had_exception
                and not retryable_primary_failure
                and not self._fallback_retrying
            ):
                get_llm_circuit_breaker().record_success(
                    role=self._routing_role,
                    profile=primary_identity,
                )


class CodexCodeAgent(ExternalCLICodeAgent):
    """Coding agent powered by Codex CLI."""

    _backend_name = "codex"
    _command_template_env = "CODEX_COMMAND_TEMPLATE"
    _default_command_template = "codex exec --model {model}"


class OpenCodeAgent(ExternalCLICodeAgent):
    """Coding agent powered by OpenCode CLI."""

    _backend_name = "opencode"
    _command_template_env = "OPENCODE_COMMAND_TEMPLATE"
    _default_command_template = "opencode run --model {model}"


def stage_uses_workflow_execution(stage: Mapping[str, Any] | None) -> bool:
    """Return whether current stage should route to workflow executor."""
    if not isinstance(stage, Mapping):
        return False

    workflow_id = str(stage.get("workflow_id", "")).strip()
    if workflow_id:
        return True

    mode = str(stage.get("execution_mode", "")).strip().lower()
    if mode in {"workflow", "local_cli", "remote_api", "managed_platform"}:
        return True

    return False


class RoutedExecutionAgent(Agent):
    """Route execution to skill executor or workflow executor based on stage hints."""

    model_config = {"extra": "allow"}

    _skill_executor: Agent
    _workflow_executor: Agent

    def __init__(
        self,
        *,
        name: str,
        description: str,
        skill_executor: Agent,
        workflow_executor: Agent,
        after_agent_callback: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            description=description,
            model="execution-router",
            after_agent_callback=after_agent_callback,
            **kwargs,
        )
        self._skill_executor = skill_executor
        self._workflow_executor = workflow_executor

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        stage = ctx.session.state.get(StateKeys.CURRENT_STAGE)
        use_workflow = stage_uses_workflow_execution(stage)
        delegate = self._workflow_executor if use_workflow else self._skill_executor

        route_label = "workflow executor" if use_workflow else "skill executor"
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Execution routing: {route_label} ({delegate.name})."
                    )
                ],
            ),
        )
        async for event in delegate.run_async(ctx):
            yield event


def create_execution_agent(
    *,
    name: str,
    description: str,
    skill_executor: Agent,
    workflow_executor: Agent,
    after_agent_callback: Optional[Any] = None,
) -> Agent:
    """Create routed execution agent wrapper."""
    return RoutedExecutionAgent(
        name=name,
        description=description,
        skill_executor=skill_executor,
        workflow_executor=workflow_executor,
        after_agent_callback=after_agent_callback,
    )


def create_coding_agent(
    *,
    executor: str,
    name: str,
    description: str,
    working_dir: str,
    model: str,
    fallback_model: Optional[str],
    fallback_max_retries: int,
    routing_role: Optional[str],
    primary_profile_name: Optional[str],
    output_key: str,
    after_agent_callback: Optional[Any] = None,
) -> Agent:
    """Factory for coding agents across supported coding executors."""
    normalized = normalize_coding_executor(executor) or DEFAULT_CODING_EXECUTOR
    common_kwargs = {
        "name": name,
        "description": description,
        "working_dir": working_dir,
        "model": model,
        "fallback_model": fallback_model,
        "fallback_max_retries": fallback_max_retries,
        "routing_role": routing_role,
        "primary_profile_name": primary_profile_name,
        "output_key": output_key,
        "after_agent_callback": after_agent_callback,
    }

    if normalized == "claude_code":
        from agentic_data_scientist.agents.claude_code.agent import ClaudeCodeAgent

        return ClaudeCodeAgent(**common_kwargs)
    if normalized == "codex":
        return CodexCodeAgent(**common_kwargs)
    if normalized == "opencode":
        return OpenCodeAgent(**common_kwargs)

    logger.warning(
        f"[CodingBackend] Unsupported executor '{executor}', falling back to {DEFAULT_CODING_EXECUTOR}"
    )
    from agentic_data_scientist.agents.claude_code.agent import ClaudeCodeAgent

    return ClaudeCodeAgent(**common_kwargs)
