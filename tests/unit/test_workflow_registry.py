"""Unit tests for workflow registry discovery and lookup."""

import shutil
import uuid
from pathlib import Path

from agentic_data_scientist.workflows import WorkflowRegistry


def _make_case_dir() -> Path:
    root = Path(".test_workflow_tmp")
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"registry_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _write(case_dir: Path, relative_path: str, content: str):
    path = case_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip(), encoding="utf-8")
    return path


def test_registry_discovers_and_filters():
    """Registry should discover manifests and support id/domain lookups."""
    case_dir = _make_case_dir()
    try:
        _write(
            case_dir,
            "bio/rnaseq_v1.yaml",
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: bio.rnaseq.pipeline
  name: RNAseq
  domain: bioinformatics
  version: 1.0.0
executor:
  type: local_cli
  adapter: nextflow
spec:
  entrypoint:
    command: "nextflow run main.nf"
  outputs:
    - name: de_genes
      type: file
      path: results/de.tsv
            """,
        )
        _write(
            case_dir,
            "bio/rnaseq_v2.yaml",
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: bio.rnaseq.pipeline
  name: RNAseq
  domain: bioinformatics
  version: 1.1.0
executor:
  type: local_cli
  adapter: nextflow
spec:
  entrypoint:
    command: "nextflow run main.nf -profile docker"
  outputs:
    - name: de_genes
      type: file
      path: results/de.tsv
            """,
        )
        _write(
            case_dir,
            "finance/factor.yaml",
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: quant.factor.backtest
  name: Factor Backtest
  domain: finance
  version: 2.0.0
executor:
  type: remote_api
  adapter: http_job
spec:
  entrypoint:
    submit:
      method: POST
      url: https://api.example/jobs
  outputs:
    - name: metrics
      type: file
      path: results/metrics.json
            """,
        )

        registry = WorkflowRegistry(manifest_dirs=[case_dir])
        result = registry.discover()

        assert len(result.errors) == 0
        assert len(result.manifests) == 3
        assert registry.get("bio.rnaseq.pipeline").metadata.version == "1.1.0"
        assert len(registry.find_by_domain("bioinformatics")) == 2
        assert len(registry.find_by_domain("finance")) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_registry_records_invalid_manifest_errors():
    """Registry should skip invalid manifest and keep error details."""
    case_dir = _make_case_dir()
    try:
        _write(
            case_dir,
            "ok.yaml",
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: demo.ok
  name: Demo OK
  domain: general
  version: 1.0.0
executor:
  type: local_cli
  adapter: generic_cli
spec:
  entrypoint:
    command: "echo ok"
  outputs:
    - name: out
      type: file
      path: out.txt
            """,
        )
        _write(
            case_dir,
            "bad.yaml",
            """
apiVersion: ads.workflow/v1
kind: WorkflowManifest
metadata:
  id: demo.bad
  name: Demo Bad
  domain: general
  version: 1.0.0
executor:
  type: local_cli
  adapter: generic_cli
spec:
  entrypoint: {}
            """,
        )

        registry = WorkflowRegistry(manifest_dirs=[case_dir])
        result = registry.discover()

        assert len(result.manifests) == 1
        assert result.manifests[0].metadata.id == "demo.ok"
        assert len(result.errors) == 1
        assert "entrypoint.command" in result.errors[0].error
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
