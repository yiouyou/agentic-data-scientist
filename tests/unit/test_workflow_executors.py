"""Unit tests for workflow execution adapters."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest

import agentic_data_scientist.workflows.executors as executors_mod
from agentic_data_scientist.workflows import build_workflow_executor, parse_workflow_manifest
from agentic_data_scientist.workflows.executors import WorkflowExecutionRequest


def _new_case_dir(prefix: str) -> Path:
    root = Path(".tmp") / "workflow_executor_cases"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{prefix}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _make_local_manifest():
    return parse_workflow_manifest(
        {
            "apiVersion": "ads.workflow/v1",
            "kind": "WorkflowManifest",
            "metadata": {
                "id": "local.test",
                "name": "Local Test",
                "domain": "test",
                "version": "1.0.0",
            },
            "executor": {"type": "local_cli", "adapter": "shell", "profile": "local"},
            "spec": {
                "entrypoint": {
                    "command": (
                        'python -c "import os; from pathlib import Path; '
                        "Path('results').mkdir(exist_ok=True); "
                        "Path('results/out.txt').write_text(os.getenv('WF_FLAG', ''), encoding='utf-8'); "
                        "print('ok')\""
                    ),
                    "shell": True,
                },
                "outputs": [
                    {"name": "report", "type": "file", "path": "results/out.txt"},
                ],
            },
            "runtime": {
                "timeout_seconds": 60,
                "env": [
                    {"name": "WF_FLAG", "value": "enabled"},
                ],
            },
            "artifacts": {"collect": ["results/**/*.txt", "results/*.txt"]},
        }
    )


def _make_remote_manifest():
    return parse_workflow_manifest(
        {
            "apiVersion": "ads.workflow/v1",
            "kind": "WorkflowManifest",
            "metadata": {
                "id": "remote.test",
                "name": "Remote Test",
                "domain": "test",
                "version": "1.0.0",
            },
            "executor": {"type": "remote_api", "adapter": "http_job", "profile": "cloud"},
            "spec": {
                "entrypoint": {
                    "submit": {
                        "method": "POST",
                        "url": "https://api.example.com/jobs",
                        "body_template": {
                            "workflow": "remote.test",
                            "inputs": "{inputs}",
                            "params": "{params}",
                        },
                        "headers_template": {"Authorization": "Bearer {params.token}"},
                        "job_id_path": "data.job.id",
                    },
                    "status": {
                        "method": "GET",
                        "url_template": "https://api.example.com/jobs/{job_id}",
                        "status_path": "data.state",
                    },
                    "result": {
                        "method": "GET",
                        "url_template": "https://api.example.com/jobs/{job_id}/artifacts",
                        "artifacts_path": "data.files",
                    },
                },
                "outputs": [
                    {"name": "metrics", "type": "file", "path": "results/metrics.json"},
                ],
            },
            "runtime": {"timeout_seconds": 60},
            "monitoring": {"mode": "poll", "interval_seconds": 1},
        }
    )


@pytest.mark.asyncio
async def test_local_cli_executor_runs_and_collects_artifacts(monkeypatch):
    """Local CLI adapter should run command, apply runtime env, and collect artifacts."""
    case_dir = _new_case_dir("local")
    try:
        manifest = _make_local_manifest()
        executor = build_workflow_executor(manifest)
        request = WorkflowExecutionRequest(
            manifest=manifest,
            working_dir=str(case_dir),
            runtime_outdir=str(case_dir / "results"),
        )
        async def _fake_run_subprocess(command, cwd, shell, timeout_seconds, extra_env=None):
            out_file = Path(cwd) / "results" / "out.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(str((extra_env or {}).get("WF_FLAG", "")), encoding="utf-8")
            return "ok", "", 0

        monkeypatch.setattr(executors_mod, "_run_subprocess", _fake_run_subprocess)

        result = await executor.execute(request)

        assert result.success is True
        assert result.exit_code == 0
        assert "results/out.txt" in result.artifacts
        assert "WF_FLAG" in result.metadata.get("runtime_env_keys", [])
        assert (case_dir / "results" / "out.txt").read_text(encoding="utf-8") == "enabled"
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_remote_api_executor_submit_poll_result_flow(monkeypatch):
    """Remote API adapter should submit job, poll status, and read result artifacts."""
    manifest = _make_remote_manifest()
    executor = build_workflow_executor(manifest)
    calls: list[tuple[str, str, dict | None, dict | None]] = []

    async def _fake_request_json(method, url, body, headers=None, timeout_seconds=60):
        calls.append((method, url, body, dict(headers or {})))
        if url.endswith("/jobs"):
            return {"data": {"job": {"id": "job-123"}}}
        if url.endswith("/jobs/job-123"):
            return {"data": {"state": "succeeded"}}
        if url.endswith("/jobs/job-123/artifacts"):
            return {"data": {"files": ["results/metrics.json", "results/equity_curve.csv"]}}
        return {}

    monkeypatch.setattr(executors_mod, "_request_json", _fake_request_json)

    case_dir = _new_case_dir("remote_ok")
    try:
        request = WorkflowExecutionRequest(
            manifest=manifest,
            working_dir=str(case_dir),
            params={"token": "abc123"},
        )
        result = await executor.execute(request)

        assert result.success is True
        assert result.status == "succeeded"
        assert result.job_id == "job-123"
        assert result.artifacts == ["results/metrics.json", "results/equity_curve.csv"]
        assert calls[0][0] == "POST"
        assert calls[0][3].get("Authorization") == "Bearer abc123"
        assert calls[1][1].endswith("/jobs/job-123")
        assert calls[2][1].endswith("/jobs/job-123/artifacts")
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_remote_api_executor_returns_failed_status(monkeypatch):
    """Remote API adapter should return failure result on failed terminal status."""
    manifest = _make_remote_manifest()
    executor = build_workflow_executor(manifest)

    async def _fake_request_json(method, url, body, headers=None, timeout_seconds=60):
        if url.endswith("/jobs"):
            return {"data": {"job": {"id": "job-456"}}}
        if url.endswith("/jobs/job-456"):
            return {"data": {"state": "failed"}}
        return {}

    monkeypatch.setattr(executors_mod, "_request_json", _fake_request_json)

    case_dir = _new_case_dir("remote_fail")
    try:
        request = WorkflowExecutionRequest(
            manifest=manifest,
            working_dir=str(case_dir),
        )
        result = await executor.execute(request)

        assert result.success is False
        assert result.status == "failed"
        assert result.job_id == "job-456"
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
