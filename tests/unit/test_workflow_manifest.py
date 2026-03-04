"""Unit tests for workflow manifest schema validation."""

import shutil
import uuid
from pathlib import Path

import pytest

from agentic_data_scientist.workflows import ManifestValidationError, load_workflow_manifest


def _make_case_dir() -> Path:
    root = Path(".test_workflow_tmp")
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"manifest_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _write_manifest(case_dir: Path, text: str, filename: str = "workflow.yaml"):
    path = case_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def test_load_workflow_manifest_success_local_cli():
    """Valid local CLI manifest should load into typed object."""
    case_dir = _make_case_dir()
    try:
        manifest_path = _write_manifest(
            case_dir,
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: demo.simple.echo
  name: Demo Echo
  domain: general
  version: 1.0.0
executor:
  type: local_cli
  adapter: generic_cli
  profile: local
spec:
  entrypoint:
    command: "echo hello"
  inputs:
    - name: input_file
      type: file
      required: true
  params:
    - name: max_rows
      type: integer
      default: 100
  outputs:
    - name: report
      type: file
      path: results/report.txt
monitoring:
  mode: poll
  interval_seconds: 10
  success:
    exit_code: 0
    required_outputs: [report]
            """.strip(),
        )

        manifest = load_workflow_manifest(manifest_path)
        assert manifest.metadata.id == "demo.simple.echo"
        assert manifest.executor.type == "local_cli"
        assert manifest.spec.entrypoint["command"] == "echo hello"
        assert manifest.spec.params[0].default == 100
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_load_workflow_manifest_requires_submit_for_remote():
    """Remote API manifest must include submit method/url config."""
    case_dir = _make_case_dir()
    try:
        manifest_path = _write_manifest(
            case_dir,
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: demo.remote.job
  name: Demo Remote
  domain: general
  version: 1.0.0
executor:
  type: remote_api
  adapter: http_job
spec:
  entrypoint:
    status:
      method: GET
      url_template: "https://api.example/jobs/{job_id}"
  outputs:
    - name: result
      type: file
      path: results/out.json
            """.strip(),
        )

        with pytest.raises(ManifestValidationError, match="entrypoint.submit"):
            load_workflow_manifest(manifest_path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_load_workflow_manifest_rejects_invalid_default_type():
    """Default value type must match declared parameter type."""
    case_dir = _make_case_dir()
    try:
        manifest_path = _write_manifest(
            case_dir,
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: demo.invalid.default
  name: Demo Invalid Default
  domain: general
  version: 1.0.0
executor:
  type: local_cli
  adapter: generic_cli
spec:
  entrypoint:
    command: "echo test"
  params:
    - name: max_rows
      type: integer
      default: "one hundred"
  outputs:
    - name: report
      type: file
      path: results/report.txt
            """.strip(),
        )

        with pytest.raises(ManifestValidationError, match="must be an integer"):
            load_workflow_manifest(manifest_path)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
