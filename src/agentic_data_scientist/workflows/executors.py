"""Workflow execution adapters (local CLI and remote API)."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import requests

from agentic_data_scientist.workflows.manifest import WorkflowManifest


logger = logging.getLogger(__name__)


class WorkflowExecutionError(RuntimeError):
    """Raised when workflow execution fails."""


@dataclass
class WorkflowExecutionRequest:
    """Runtime inputs for workflow execution."""

    manifest: WorkflowManifest
    working_dir: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    runtime_outdir: str = ""


@dataclass
class WorkflowExecutionResult:
    """Structured workflow execution result."""

    success: bool
    status: str
    exit_code: int | None = None
    job_id: str = ""
    stdout: str = ""
    stderr: str = ""
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseWorkflowExecutor:
    """Base workflow executor interface."""

    async def execute(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResult:
        raise NotImplementedError


class LocalCLIWorkflowExecutor(BaseWorkflowExecutor):
    """Execute workflow by local command line."""

    async def execute(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResult:
        manifest = request.manifest
        entrypoint = manifest.spec.entrypoint
        command_template = str(entrypoint.get("command", "")).strip()
        if not command_template:
            raise WorkflowExecutionError("spec.entrypoint.command is required for local_cli execution")

        context = {
            "inputs": request.inputs,
            "params": request.params,
            "runtime_outdir": request.runtime_outdir,
            "working_dir": request.working_dir,
        }
        command = _render_string(command_template, context)
        shell_mode = bool(entrypoint.get("shell", True))

        working_dir = Path(request.working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        if request.runtime_outdir:
            Path(request.runtime_outdir).mkdir(parents=True, exist_ok=True)

        timeout = int(manifest.runtime.timeout_seconds)
        runtime_env = _build_runtime_env(manifest.runtime.env)
        stdout_text, stderr_text, exit_code = await _run_subprocess(
            command=command,
            cwd=str(working_dir),
            shell=shell_mode,
            timeout_seconds=timeout,
            extra_env=runtime_env,
        )

        artifacts = _collect_artifacts(working_dir, manifest.artifacts.collect)
        success = (exit_code == 0)
        status = "success" if success else "failed"
        return WorkflowExecutionResult(
            success=success,
            status=status,
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            artifacts=artifacts,
            metadata={"command": command, "shell": shell_mode, "runtime_env_keys": sorted(runtime_env.keys())},
        )


class RemoteAPIWorkflowExecutor(BaseWorkflowExecutor):
    """Execute workflow by remote HTTP job API."""

    SUCCESS_STATUSES = {"success", "succeeded", "completed", "done", "ok"}
    FAILURE_STATUSES = {"failed", "error", "cancelled", "canceled", "timeout"}

    async def execute(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResult:
        manifest = request.manifest
        entrypoint = manifest.spec.entrypoint
        submit = _as_dict(entrypoint.get("submit"), "spec.entrypoint.submit")
        status_cfg = _as_dict(entrypoint.get("status", {}), "spec.entrypoint.status")
        result_cfg = _as_dict(entrypoint.get("result", {}), "spec.entrypoint.result")

        context = {
            "inputs": request.inputs,
            "params": request.params,
            "runtime_outdir": request.runtime_outdir,
            "working_dir": request.working_dir,
        }

        submit_method = str(submit.get("method", "POST")).upper().strip()
        submit_url = _render_string(str(submit.get("url", "")).strip(), context)
        submit_body = _render_structure(submit.get("body_template", {}), context)
        submit_headers = _render_mapping(submit.get("headers_template", {}), context)
        submit_timeout = int(submit.get("timeout_seconds", 60))
        submit_job_id_path = str(submit.get("job_id_path", "")).strip()
        if not submit_url:
            raise WorkflowExecutionError("Remote workflow submit URL is empty")

        submit_resp = await _request_json(
            method=submit_method,
            url=submit_url,
            body=submit_body,
            headers=submit_headers,
            timeout_seconds=submit_timeout,
        )
        job_id = _extract_job_id(submit_resp, explicit_path=submit_job_id_path)
        if not job_id:
            raise WorkflowExecutionError("Remote submit response missing job id")

        final_status = "submitted"
        details: Dict[str, Any] = {"submit_response": submit_resp, "job_id": job_id}

        timeout_seconds = int(manifest.runtime.timeout_seconds)
        interval_seconds = int(manifest.monitoring.interval_seconds)
        started = time.monotonic()

        if status_cfg:
            status_method = str(status_cfg.get("method", "GET")).upper().strip()
            status_url_template = str(status_cfg.get("url_template", "")).strip()
            status_headers = _render_mapping(status_cfg.get("headers_template", {}), context)
            status_timeout = int(status_cfg.get("timeout_seconds", 60))
            status_path = str(status_cfg.get("status_path", "")).strip()
            if not status_url_template:
                raise WorkflowExecutionError("spec.entrypoint.status.url_template is required for polling")

            while True:
                if time.monotonic() - started > timeout_seconds:
                    raise WorkflowExecutionError(
                        f"Remote workflow timed out after {timeout_seconds}s (job_id={job_id})"
                    )

                poll_context = dict(context)
                poll_context["job_id"] = job_id
                status_url = _render_string(status_url_template, poll_context)
                status_resp = await _request_json(
                    method=status_method,
                    url=status_url,
                    body=None,
                    headers=status_headers,
                    timeout_seconds=status_timeout,
                )
                details["last_status_response"] = status_resp
                status_value = _extract_status(status_resp, explicit_path=status_path)
                if status_value:
                    normalized = status_value.strip().lower()
                    final_status = normalized
                    if normalized in self.SUCCESS_STATUSES:
                        break
                    if normalized in self.FAILURE_STATUSES:
                        return WorkflowExecutionResult(
                            success=False,
                            status=normalized,
                            job_id=job_id,
                            metadata=details,
                        )
                await asyncio.sleep(max(1, interval_seconds))
        else:
            final_status = "success"

        artifacts: List[str] = []
        if result_cfg:
            result_method = str(result_cfg.get("method", "GET")).upper().strip()
            result_url_template = str(result_cfg.get("url_template", "")).strip()
            result_headers = _render_mapping(result_cfg.get("headers_template", {}), context)
            result_timeout = int(result_cfg.get("timeout_seconds", 60))
            result_artifacts_path = str(result_cfg.get("artifacts_path", "")).strip()
            if result_url_template:
                result_context = dict(context)
                result_context["job_id"] = job_id
                result_url = _render_string(result_url_template, result_context)
                result_resp = await _request_json(
                    method=result_method,
                    url=result_url,
                    body=None,
                    headers=result_headers,
                    timeout_seconds=result_timeout,
                )
                details["result_response"] = result_resp
                artifacts = _extract_artifacts_from_result(
                    result_resp,
                    explicit_path=result_artifacts_path,
                )

        return WorkflowExecutionResult(
            success=True,
            status=final_status or "success",
            job_id=job_id,
            artifacts=artifacts,
            metadata=details,
        )


def build_workflow_executor(manifest: WorkflowManifest) -> BaseWorkflowExecutor:
    """Create executor adapter instance based on manifest executor type."""
    executor_type = manifest.executor.type
    if executor_type == "local_cli":
        return LocalCLIWorkflowExecutor()
    if executor_type in {"remote_api", "managed_platform"}:
        return RemoteAPIWorkflowExecutor()
    raise WorkflowExecutionError(f"Unsupported executor type: {executor_type}")


async def _run_subprocess(
    command: str,
    cwd: str,
    shell: bool,
    timeout_seconds: int,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[str, str, int]:
    """Run subprocess command and return stdout/stderr/exit-code."""
    env = os.environ.copy()
    if extra_env:
        env.update(dict(extra_env))
    if shell:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command, posix=(os.name != "nt")),
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise WorkflowExecutionError(f"Local workflow command timeout after {timeout_seconds}s")

    stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
    stderr_text = stderr_raw.decode("utf-8", errors="replace").strip()
    return stdout_text, stderr_text, int(process.returncode)


async def _request_json(
    method: str,
    url: str,
    body: Any,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Perform HTTP request in thread and parse JSON response."""
    return await asyncio.to_thread(
        _request_json_sync,
        method,
        url,
        body,
        dict(headers or {}),
        timeout_seconds,
    )


def _request_json_sync(
    method: str,
    url: str,
    body: Any,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Blocking HTTP request helper."""
    kwargs: Dict[str, Any] = {"timeout": timeout_seconds}
    if headers:
        kwargs["headers"] = dict(headers)
    if body is not None:
        kwargs["json"] = body
    response = requests.request(method=method, url=url, **kwargs)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        return {}
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return {"data": payload}
    except Exception:
        return {"raw_text": text}


def _extract_job_id(payload: Mapping[str, Any], explicit_path: str = "") -> str:
    """Best-effort job id extraction from submit response."""
    if explicit_path:
        explicit = _resolve_path(payload, explicit_path)
        if explicit is not None:
            value = str(explicit).strip()
            if value:
                return value

    candidates = [
        _resolve_path(payload, "job_id"),
        _resolve_path(payload, "id"),
        _resolve_path(payload, "jobId"),
        _resolve_path(payload, "data.job_id"),
        _resolve_path(payload, "data.id"),
    ]
    for item in candidates:
        if item is None:
            continue
        value = str(item).strip()
        if value:
            return value
    return ""


def _extract_status(payload: Mapping[str, Any], explicit_path: str = "") -> str:
    """Best-effort status extraction from polling response."""
    if explicit_path:
        explicit = _resolve_path(payload, explicit_path)
        if explicit is not None:
            value = str(explicit).strip()
            if value:
                return value

    candidates = [
        _resolve_path(payload, "status"),
        _resolve_path(payload, "state"),
        _resolve_path(payload, "phase"),
        _resolve_path(payload, "data.status"),
        _resolve_path(payload, "data.state"),
    ]
    for item in candidates:
        if item is None:
            continue
        value = str(item).strip()
        if value:
            return value
    return ""


def _extract_artifacts_from_result(payload: Mapping[str, Any], explicit_path: str = "") -> List[str]:
    """Best-effort artifact list extraction from result response."""
    if explicit_path:
        explicit = _resolve_path(payload, explicit_path)
        if isinstance(explicit, list):
            return [str(item).strip() for item in explicit if str(item).strip()]

    artifacts = _resolve_path(payload, "artifacts")
    if isinstance(artifacts, list):
        return [str(item).strip() for item in artifacts if str(item).strip()]
    files = _resolve_path(payload, "files")
    if isinstance(files, list):
        return [str(item).strip() for item in files if str(item).strip()]
    return []


def _collect_artifacts(base_dir: Path, patterns: List[str]) -> List[str]:
    """Collect artifacts by glob patterns relative to base directory."""
    if not patterns:
        return []
    results: List[str] = []
    seen = set()
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if not path.exists():
                continue
            rel = path.relative_to(base_dir).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            results.append(rel)
    results.sort()
    return results


def _render_structure(value: Any, context: Dict[str, Any]) -> Any:
    """Render templated values recursively for dict/list/string structures."""
    if isinstance(value, str):
        return _render_string(value, context)
    if isinstance(value, list):
        return [_render_structure(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_structure(item, context) for key, item in value.items()}
    return value


def _render_mapping(value: Any, context: Dict[str, Any]) -> Dict[str, str]:
    """Render mapping structure to `Dict[str, str]`."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkflowExecutionError("Rendered mapping template must be a mapping")

    rendered: Dict[str, str] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        item_value = _render_structure(item, context)
        if item_value is None:
            continue
        rendered[key_text] = str(item_value)
    return rendered


def _build_runtime_env(env_vars: List[Any]) -> Dict[str, str]:
    """Resolve runtime env vars declared in manifest."""
    resolved: Dict[str, str] = {}
    for item in env_vars:
        name = str(getattr(item, "name", "")).strip()
        if not name:
            continue
        value = str(getattr(item, "value", "")).strip()
        value_from_env = str(getattr(item, "value_from_env", "")).strip()
        if value_from_env:
            env_value = os.getenv(value_from_env, "")
            if env_value:
                resolved[name] = env_value
            continue
        resolved[name] = value
    return resolved


def _render_string(template: str, context: Dict[str, Any]) -> str:
    """Render {path.to.value} placeholders using context dictionaries."""
    import re

    pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\.]*)\}")

    def _replace(match: re.Match[str]) -> str:
        key_path = match.group(1)
        value = _resolve_path(context, key_path)
        return "" if value is None else str(value)

    return pattern.sub(_replace, template)


def _resolve_path(data: Mapping[str, Any], path: str) -> Any:
    """Resolve dotted path from nested mapping."""
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def _as_dict(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise WorkflowExecutionError(f"{label} must be a mapping")
    return value
