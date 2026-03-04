"""Unit tests for WorkflowExecutionAgent."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentic_data_scientist.workflows.executors as executors_mod
from agentic_data_scientist.agents.workflow_execution import WorkflowExecutionAgent
from agentic_data_scientist.core.state_contracts import StateKeys


def _new_case_dir(prefix: str) -> Path:
    root = Path(".tmp") / "workflow_agent_cases"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{prefix}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _write_manifest(case_dir: Path) -> Path:
    manifest_dir = case_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "demo.local.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "apiVersion: ads.workflow/v1",
                "kind: WorkflowManifest",
                "metadata:",
                "  id: test.workflow.agent",
                "  name: Test Workflow Agent",
                "  domain: test",
                "  version: 1.0.0",
                "executor:",
                "  type: local_cli",
                "  adapter: shell",
                "spec:",
                "  entrypoint:",
                "    command: python -c \"print('ok')\"",
                "    shell: true",
                "  outputs:",
                "    - name: report",
                "      type: file",
                "      path: results/report.txt",
                "runtime:",
                "  timeout_seconds: 60",
                "monitoring:",
                "  mode: poll",
                "  interval_seconds: 1",
                "  success:",
                "    exit_code: 0",
                "artifacts:",
                "  collect: [results/*.txt]",
                "security:",
                "  secrets: []",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_manifest_with_args(case_dir: Path) -> Path:
    manifest_dir = case_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "demo.args.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "apiVersion: ads.workflow/v1",
                "kind: WorkflowManifest",
                "metadata:",
                "  id: test.workflow.args",
                "  name: Test Workflow Args",
                "  domain: test",
                "  version: 1.0.0",
                "executor:",
                "  type: local_cli",
                "  adapter: shell",
                "spec:",
                "  entrypoint:",
                (
                    "    command: python -c \"print('sample={inputs.sample_id}|threshold={params.threshold}')\""
                ),
                "    shell: true",
                "  inputs:",
                "    - name: sample_id",
                "      type: string",
                "      required: true",
                "  params:",
                "    - name: threshold",
                "      type: number",
                "      required: true",
                "  outputs:",
                "    - name: report",
                "      type: file",
                "      path: results/report.txt",
                "runtime:",
                "  timeout_seconds: 60",
                "monitoring:",
                "  mode: poll",
                "  interval_seconds: 1",
                "  success:",
                "    exit_code: 0",
                "artifacts:",
                "  collect: [results/*.txt]",
                "security:",
                "  secrets: []",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.asyncio
async def test_workflow_execution_agent_runs_workflow_and_writes_summary(monkeypatch):
    """Agent should execute workflow by workflow_id and persist structured summary."""
    case_dir = _new_case_dir("workflow_agent")
    try:
        _write_manifest(case_dir)

        async def _fake_run_subprocess(command, cwd, shell, timeout_seconds, extra_env=None):
            out_file = Path(cwd) / "results" / "report.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("workflow agent ok", encoding="utf-8")
            return "workflow agent ok", "", 0

        monkeypatch.setattr(executors_mod, "_run_subprocess", _fake_run_subprocess)

        agent = WorkflowExecutionAgent(
            working_dir=str(case_dir / "workdir"),
            manifest_dirs=[str(case_dir / "manifests")],
        )
        state = {
            StateKeys.CURRENT_STAGE: {
                "index": 0,
                "title": "Run fixed workflow",
                "description": "smoke",
                "execution_mode": "workflow",
                "workflow_id": "test.workflow.agent",
            }
        }
        ctx = SimpleNamespace(session=SimpleNamespace(state=state, events=[]))

        async for _ in agent._run_async_impl(ctx):
            pass

        summary = state[StateKeys.IMPLEMENTATION_SUMMARY]
        assert '"workflow_id": "test.workflow.agent"' in summary
        assert '"success": true' in summary.lower()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_workflow_execution_agent_passes_stage_inputs_and_params(monkeypatch):
    """Agent should pass workflow_inputs/workflow_params into command rendering."""
    case_dir = _new_case_dir("workflow_agent_args")
    try:
        _write_manifest_with_args(case_dir)
        captured = {"command": ""}

        async def _fake_run_subprocess(command, cwd, shell, timeout_seconds, extra_env=None):
            captured["command"] = command
            out_file = Path(cwd) / "results" / "report.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("ok", encoding="utf-8")
            return "ok", "", 0

        monkeypatch.setattr(executors_mod, "_run_subprocess", _fake_run_subprocess)

        agent = WorkflowExecutionAgent(
            working_dir=str(case_dir / "workdir"),
            manifest_dirs=[str(case_dir / "manifests")],
        )
        state = {
            StateKeys.CURRENT_STAGE: {
                "index": 0,
                "title": "Run workflow with args",
                "description": "smoke args",
                "execution_mode": "workflow",
                "workflow_id": "test.workflow.args",
                "workflow_inputs": {"sample_id": "S-42"},
                "workflow_params": {"threshold": 0.75},
            }
        }
        ctx = SimpleNamespace(session=SimpleNamespace(state=state, events=[]))

        async for _ in agent._run_async_impl(ctx):
            pass

        assert "sample=S-42|threshold=0.75" in captured["command"]
        summary = state[StateKeys.IMPLEMENTATION_SUMMARY]
        assert '"workflow_id": "test.workflow.args"' in summary
        assert '"success": true' in summary.lower()
        assert "sample=S-42|threshold=0.75" in summary
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
