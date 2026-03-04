"""Workflow execution agent for fixed pipeline manifests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from google.adk.agents import Agent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agentic_data_scientist.core.state_contracts import StateKeys
from agentic_data_scientist.workflows.executors import (
    WorkflowExecutionError,
    WorkflowExecutionRequest,
    build_workflow_executor,
)
from agentic_data_scientist.workflows.registry import WorkflowRegistry


logger = logging.getLogger(__name__)


class WorkflowExecutionAgent(Agent):
    """Execution agent that runs declarative workflow manifests."""

    model_config = {"extra": "allow"}

    _working_dir: Optional[str] = None
    _output_key: str = StateKeys.IMPLEMENTATION_SUMMARY
    _registry: WorkflowRegistry
    _discover_on_start: bool = True

    def __init__(
        self,
        *,
        name: str = "workflow_execution_agent",
        description: Optional[str] = None,
        working_dir: Optional[str] = None,
        output_key: str = StateKeys.IMPLEMENTATION_SUMMARY,
        manifest_dirs: Optional[list[str]] = None,
        discover_on_start: bool = True,
        after_agent_callback: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            description=description or "Runs fixed pipelines defined by workflow manifests.",
            model="workflow-executor",
            after_agent_callback=after_agent_callback,
            **kwargs,
        )
        self._working_dir = working_dir
        self._output_key = output_key
        self._registry = WorkflowRegistry(manifest_dirs=manifest_dirs)
        self._discover_on_start = discover_on_start

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

    def _resolve_working_dir(self) -> str:
        if self._working_dir:
            Path(self._working_dir).mkdir(parents=True, exist_ok=True)
            return self._working_dir
        import tempfile

        path = tempfile.mkdtemp(prefix="workflow_exec_")
        return path

    def _read_stage(self, ctx: InvocationContext) -> Dict[str, Any]:
        stage = ctx.session.state.get(StateKeys.CURRENT_STAGE)
        if isinstance(stage, dict):
            return dict(stage)
        return {}

    def _resolve_workflow_target(self, stage: Dict[str, Any]) -> tuple[str, str | None]:
        workflow_id = str(stage.get("workflow_id", "")).strip()
        workflow_version = str(stage.get("workflow_version", "")).strip() or None
        return workflow_id, workflow_version

    def _extract_mapping(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _resolve_runtime_outdir(self, stage: Dict[str, Any], working_dir: str, workflow_id: str) -> str:
        configured = str(stage.get("workflow_outdir", "")).strip()
        if configured:
            Path(configured).mkdir(parents=True, exist_ok=True)
            return configured
        safe_name = workflow_id.replace("/", "_").replace("\\", "_").replace(".", "_")
        outdir = Path(working_dir) / "results" / safe_name
        outdir.mkdir(parents=True, exist_ok=True)
        return str(outdir)

    def _refresh_registry(self) -> None:
        discovered = self._registry.discover()
        if discovered.errors:
            logger.warning(
                f"[WorkflowExecution] Manifest discovery had {len(discovered.errors)} errors"
            )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        stage = self._read_stage(ctx)
        workflow_id, workflow_version = self._resolve_workflow_target(stage)

        if not workflow_id:
            message = "Workflow stage missing `workflow_id`; cannot execute fixed pipeline."
            state[self._output_key] = message
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part.from_text(text=message)]),
            )
            return

        if self._discover_on_start:
            self._refresh_registry()

        manifest = self._registry.get(workflow_id, workflow_version)
        if manifest is None:
            message = (
                f"Workflow manifest not found: id={workflow_id!r}, version={workflow_version or 'latest'}."
            )
            state[self._output_key] = message
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part.from_text(text=message)]),
            )
            return

        working_dir = self._resolve_working_dir()
        runtime_outdir = self._resolve_runtime_outdir(stage, working_dir, workflow_id)
        request = WorkflowExecutionRequest(
            manifest=manifest,
            working_dir=working_dir,
            inputs=self._extract_mapping(stage.get("workflow_inputs")),
            params=self._extract_mapping(stage.get("workflow_params")),
            runtime_outdir=runtime_outdir,
        )
        executor = build_workflow_executor(manifest)

        start_message = (
            f"Executing workflow `{manifest.metadata.id}@{manifest.metadata.version}` "
            f"via `{manifest.executor.type}/{manifest.executor.adapter}`."
        )
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part.from_text(text=start_message)]),
        )

        try:
            result = await executor.execute(request)
            summary_obj = {
                "workflow_id": manifest.metadata.id,
                "workflow_version": manifest.metadata.version,
                "executor": {
                    "type": manifest.executor.type,
                    "adapter": manifest.executor.adapter,
                    "profile": manifest.executor.profile,
                },
                "success": result.success,
                "status": result.status,
                "exit_code": result.exit_code,
                "job_id": result.job_id,
                "artifacts": result.artifacts,
                "runtime_outdir": runtime_outdir,
                "metadata": result.metadata,
            }
            summary = self._truncate_summary(json.dumps(summary_obj, ensure_ascii=False, indent=2))
            state[self._output_key] = summary
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part.from_text(text=summary)]),
            )
        except WorkflowExecutionError as exc:
            message = self._truncate_summary(f"Workflow execution failed: {str(exc)}")
            state[self._output_key] = message
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part.from_text(text=message)]),
            )
        except Exception as exc:
            message = self._truncate_summary(f"Unexpected workflow execution error: {str(exc)}")
            state[self._output_key] = message
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part.from_text(text=message)]),
            )
