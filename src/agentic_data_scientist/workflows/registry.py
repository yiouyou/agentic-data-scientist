"""Workflow manifest discovery and lookup registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from agentic_data_scientist.workflows.manifest import ManifestValidationError, WorkflowManifest, load_workflow_manifest


@dataclass
class WorkflowRegistryError:
    """One manifest discovery/load failure."""

    path: str
    error: str


@dataclass
class WorkflowDiscoveryResult:
    """Manifest discovery output."""

    manifests: List[WorkflowManifest] = field(default_factory=list)
    errors: List[WorkflowRegistryError] = field(default_factory=list)


class WorkflowRegistry:
    """Workflow registry loaded from one or more manifest directories."""

    def __init__(self, manifest_dirs: Sequence[str | Path] | None = None):
        self.manifest_dirs = [Path(item) for item in (manifest_dirs or _default_manifest_dirs())]
        self._manifests: Dict[Tuple[str, str], WorkflowManifest] = {}
        self._errors: List[WorkflowRegistryError] = []

    @property
    def errors(self) -> List[WorkflowRegistryError]:
        """Return discovery errors from the latest `discover` call."""
        return list(self._errors)

    def discover(self) -> WorkflowDiscoveryResult:
        """Discover and load workflow manifests from configured directories."""
        self._manifests.clear()
        self._errors.clear()

        for directory in self.manifest_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                try:
                    manifest = load_workflow_manifest(path)
                    self.register(manifest)
                except (ManifestValidationError, OSError, ValueError) as exc:
                    self._errors.append(WorkflowRegistryError(path=str(path), error=str(exc)))

        return WorkflowDiscoveryResult(manifests=self.list(), errors=self.errors)

    def register(self, manifest: WorkflowManifest) -> None:
        """Register one manifest object."""
        key = (manifest.metadata.id, manifest.metadata.version)
        self._manifests[key] = manifest

    def list(self) -> List[WorkflowManifest]:
        """List all manifests sorted by `(domain, id, version)`."""
        manifests = list(self._manifests.values())
        manifests.sort(key=lambda item: (item.metadata.domain, item.metadata.id, _version_key(item.metadata.version)))
        return manifests

    def get(self, workflow_id: str, version: str | None = None) -> WorkflowManifest | None:
        """Get one workflow by id and optional version."""
        if version:
            return self._manifests.get((workflow_id, version))

        matches = [item for (wf_id, _), item in self._manifests.items() if wf_id == workflow_id]
        if not matches:
            return None
        matches.sort(key=lambda item: _version_key(item.metadata.version))
        return matches[-1]

    def find_by_domain(self, domain: str) -> List[WorkflowManifest]:
        """List workflows for one domain."""
        items = [item for item in self._manifests.values() if item.metadata.domain == domain]
        items.sort(key=lambda item: (item.metadata.id, _version_key(item.metadata.version)))
        return items


def _default_manifest_dirs() -> List[Path]:
    """Resolve default workflow manifest directories."""
    env_paths = os.getenv("WORKFLOW_MANIFEST_PATHS", "").strip()
    if env_paths:
        return [Path(item.strip()) for item in env_paths.split(os.pathsep) if item.strip()]
    return [Path("configs/workflows"), Path("workflows/manifests")]


def _version_key(version: str) -> Tuple[int, ...]:
    """Best-effort semantic-ish version sort key."""
    parts = str(version).strip().lstrip("v").split(".")
    result: List[int] = []
    for part in parts:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        result.append(int(digits) if digits else 0)
    return tuple(result)
