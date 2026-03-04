"""Workflow manifest schema and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


MANIFEST_API_VERSION = "ads.workflow/v1"
MANIFEST_KIND = "WorkflowManifest"

SUPPORTED_EXECUTOR_TYPES = {"local_cli", "remote_api", "managed_platform"}
SUPPORTED_MONITOR_MODES = {"poll", "stream"}
SUPPORTED_VALUE_TYPES = {
    "string",
    "integer",
    "number",
    "boolean",
    "file",
    "directory",
    "array",
    "object",
}


class ManifestValidationError(ValueError):
    """Raised when workflow manifest is invalid."""


@dataclass
class ManifestMetadata:
    """Workflow identity metadata."""

    id: str
    name: str
    domain: str
    version: str
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkflowExecutorConfig:
    """Execution backend selector."""

    type: str
    adapter: str
    profile: str = "default"


@dataclass
class ValueSpec:
    """Input/parameter specification."""

    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Any = None
    enum: List[Any] = field(default_factory=list)


@dataclass
class OutputSpec:
    """Output artifact declaration."""

    name: str
    type: str
    path: str
    description: str = ""


@dataclass
class WorkflowSpec:
    """Workflow payload definition."""

    entrypoint: Dict[str, Any]
    inputs: List[ValueSpec] = field(default_factory=list)
    params: List[ValueSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)


@dataclass
class RuntimeEnvVar:
    """Runtime environment variable mapping."""

    name: str
    value: str = ""
    value_from_env: str = ""


@dataclass
class RuntimeConfig:
    """Runtime control knobs."""

    timeout_seconds: int = 3600
    max_retries: int = 0
    resources: Dict[str, Any] = field(default_factory=dict)
    env: List[RuntimeEnvVar] = field(default_factory=list)


@dataclass
class MonitoringSuccess:
    """Success-condition settings."""

    exit_code: int = 0
    required_outputs: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring behavior settings."""

    mode: str = "poll"
    interval_seconds: int = 30
    success: MonitoringSuccess = field(default_factory=MonitoringSuccess)


@dataclass
class ArtifactsConfig:
    """Artifact collection settings."""

    collect: List[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security/secrets settings."""

    secrets: List[str] = field(default_factory=list)


@dataclass
class WorkflowManifest:
    """Strongly-typed workflow manifest."""

    api_version: str
    kind: str
    metadata: ManifestMetadata
    executor: WorkflowExecutorConfig
    spec: WorkflowSpec
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    source_path: str = ""


def load_workflow_manifest(path: str | Path) -> WorkflowManifest:
    """Load and validate one workflow manifest file."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Workflow manifest not found: {manifest_path}")

    raw = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        parsed = json.loads(raw)
    else:
        parsed = yaml.safe_load(raw)
    return parse_workflow_manifest(parsed, source_path=str(manifest_path))


def parse_workflow_manifest(parsed: Dict[str, Any], source_path: str = "") -> WorkflowManifest:
    """Parse and validate workflow manifest payload."""
    root = _as_dict(parsed, "root")

    api_version = str(root.get("apiVersion", "")).strip()
    kind = str(root.get("kind", "")).strip()
    if api_version != MANIFEST_API_VERSION:
        raise ManifestValidationError(
            f"apiVersion must be '{MANIFEST_API_VERSION}', got: {api_version!r}"
        )
    if kind != MANIFEST_KIND:
        raise ManifestValidationError(f"kind must be '{MANIFEST_KIND}', got: {kind!r}")

    metadata = _parse_metadata(_as_dict(root.get("metadata", {}), "metadata"))
    executor = _parse_executor(_as_dict(root.get("executor", {}), "executor"))
    spec = _parse_spec(_as_dict(root.get("spec", {}), "spec"), executor)
    runtime = _parse_runtime(_as_dict(root.get("runtime", {}), "runtime"))
    monitoring = _parse_monitoring(_as_dict(root.get("monitoring", {}), "monitoring"))
    artifacts = _parse_artifacts(_as_dict(root.get("artifacts", {}), "artifacts"))
    security = _parse_security(_as_dict(root.get("security", {}), "security"))

    _validate_cross_sections(spec=spec, monitoring=monitoring)

    return WorkflowManifest(
        api_version=api_version,
        kind=kind,
        metadata=metadata,
        executor=executor,
        spec=spec,
        runtime=runtime,
        monitoring=monitoring,
        artifacts=artifacts,
        security=security,
        source_path=source_path,
    )


def _parse_metadata(raw: Dict[str, Any]) -> ManifestMetadata:
    workflow_id = str(raw.get("id", "")).strip()
    name = str(raw.get("name", "")).strip()
    domain = str(raw.get("domain", "")).strip()
    version = str(raw.get("version", "")).strip()
    tags_raw = raw.get("tags", [])

    if not workflow_id:
        raise ManifestValidationError("metadata.id is required")
    if not name:
        raise ManifestValidationError("metadata.name is required")
    if not domain:
        raise ManifestValidationError("metadata.domain is required")
    if not version:
        raise ManifestValidationError("metadata.version is required")
    if not isinstance(tags_raw, list):
        raise ManifestValidationError("metadata.tags must be a list")

    tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
    return ManifestMetadata(id=workflow_id, name=name, domain=domain, version=version, tags=tags)


def _parse_executor(raw: Dict[str, Any]) -> WorkflowExecutorConfig:
    executor_type = str(raw.get("type", "")).strip()
    adapter = str(raw.get("adapter", "")).strip()
    profile = str(raw.get("profile", "default")).strip() or "default"

    if executor_type not in SUPPORTED_EXECUTOR_TYPES:
        raise ManifestValidationError(
            f"executor.type must be one of {sorted(SUPPORTED_EXECUTOR_TYPES)}, got: {executor_type!r}"
        )
    if not adapter:
        raise ManifestValidationError("executor.adapter is required")

    return WorkflowExecutorConfig(type=executor_type, adapter=adapter, profile=profile)


def _parse_spec(raw: Dict[str, Any], executor: WorkflowExecutorConfig) -> WorkflowSpec:
    entrypoint = _as_dict(raw.get("entrypoint", {}), "spec.entrypoint")
    _validate_entrypoint(entrypoint, executor)

    inputs = _parse_value_specs(raw.get("inputs", []), "spec.inputs")
    params = _parse_value_specs(raw.get("params", []), "spec.params")
    outputs = _parse_output_specs(raw.get("outputs", []), "spec.outputs")

    _ensure_unique_names([item.name for item in inputs], "spec.inputs")
    _ensure_unique_names([item.name for item in params], "spec.params")
    _ensure_unique_names([item.name for item in outputs], "spec.outputs")

    return WorkflowSpec(entrypoint=entrypoint, inputs=inputs, params=params, outputs=outputs)


def _parse_value_specs(raw: Any, label: str) -> List[ValueSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ManifestValidationError(f"{label} must be a list")

    specs: List[ValueSpec] = []
    for idx, item in enumerate(raw):
        row = _as_dict(item, f"{label}[{idx}]")
        name = str(row.get("name", "")).strip()
        value_type = str(row.get("type", "")).strip()
        required = bool(row.get("required", True))
        description = str(row.get("description", "")).strip()
        default = row.get("default")
        enum_raw = row.get("enum", [])
        if enum_raw is None:
            enum_raw = []
        if not isinstance(enum_raw, list):
            raise ManifestValidationError(f"{label}[{idx}].enum must be a list")

        if not name:
            raise ManifestValidationError(f"{label}[{idx}].name is required")
        _validate_value_type(value_type, f"{label}[{idx}].type")
        if default is not None:
            _validate_scalar_type(default, value_type, f"{label}[{idx}].default")

        enum_values = [value for value in enum_raw]
        for value in enum_values:
            _validate_scalar_type(value, value_type, f"{label}[{idx}].enum")

        specs.append(
            ValueSpec(
                name=name,
                type=value_type,
                required=required,
                description=description,
                default=default,
                enum=enum_values,
            )
        )
    return specs


def _parse_output_specs(raw: Any, label: str) -> List[OutputSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ManifestValidationError(f"{label} must be a list")

    outputs: List[OutputSpec] = []
    for idx, item in enumerate(raw):
        row = _as_dict(item, f"{label}[{idx}]")
        name = str(row.get("name", "")).strip()
        value_type = str(row.get("type", "")).strip()
        path = str(row.get("path", "")).strip()
        description = str(row.get("description", "")).strip()

        if not name:
            raise ManifestValidationError(f"{label}[{idx}].name is required")
        _validate_value_type(value_type, f"{label}[{idx}].type")
        if not path:
            raise ManifestValidationError(f"{label}[{idx}].path is required")

        outputs.append(OutputSpec(name=name, type=value_type, path=path, description=description))
    return outputs


def _parse_runtime(raw: Dict[str, Any]) -> RuntimeConfig:
    timeout_seconds = int(raw.get("timeout_seconds", 3600))
    max_retries = int(raw.get("max_retries", 0))
    resources = raw.get("resources", {})
    env_raw = raw.get("env", [])

    if timeout_seconds <= 0:
        raise ManifestValidationError("runtime.timeout_seconds must be > 0")
    if max_retries < 0:
        raise ManifestValidationError("runtime.max_retries must be >= 0")
    if not isinstance(resources, dict):
        raise ManifestValidationError("runtime.resources must be a mapping")
    if env_raw is None:
        env_raw = []
    if not isinstance(env_raw, list):
        raise ManifestValidationError("runtime.env must be a list")

    env: List[RuntimeEnvVar] = []
    for idx, item in enumerate(env_raw):
        row = _as_dict(item, f"runtime.env[{idx}]")
        name = str(row.get("name", "")).strip()
        value = str(row.get("value", "")).strip()
        value_from_env = str(row.get("value_from_env", "")).strip()
        if not name:
            raise ManifestValidationError(f"runtime.env[{idx}].name is required")
        if value and value_from_env:
            raise ManifestValidationError(
                f"runtime.env[{idx}] must set only one of value or value_from_env"
            )
        env.append(RuntimeEnvVar(name=name, value=value, value_from_env=value_from_env))

    return RuntimeConfig(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        resources=resources,
        env=env,
    )


def _parse_monitoring(raw: Dict[str, Any]) -> MonitoringConfig:
    mode = str(raw.get("mode", "poll")).strip() or "poll"
    interval_seconds = int(raw.get("interval_seconds", 30))
    success_raw = _as_dict(raw.get("success", {}), "monitoring.success")

    if mode not in SUPPORTED_MONITOR_MODES:
        raise ManifestValidationError(
            f"monitoring.mode must be one of {sorted(SUPPORTED_MONITOR_MODES)}, got: {mode!r}"
        )
    if interval_seconds <= 0:
        raise ManifestValidationError("monitoring.interval_seconds must be > 0")

    exit_code = int(success_raw.get("exit_code", 0))
    required_outputs_raw = success_raw.get("required_outputs", [])
    if required_outputs_raw is None:
        required_outputs_raw = []
    if not isinstance(required_outputs_raw, list):
        raise ManifestValidationError("monitoring.success.required_outputs must be a list")
    required_outputs = [str(name).strip() for name in required_outputs_raw if str(name).strip()]

    success = MonitoringSuccess(exit_code=exit_code, required_outputs=required_outputs)
    return MonitoringConfig(mode=mode, interval_seconds=interval_seconds, success=success)


def _parse_artifacts(raw: Dict[str, Any]) -> ArtifactsConfig:
    collect_raw = raw.get("collect", [])
    if collect_raw is None:
        collect_raw = []
    if not isinstance(collect_raw, list):
        raise ManifestValidationError("artifacts.collect must be a list")
    collect = [str(item).strip() for item in collect_raw if str(item).strip()]
    return ArtifactsConfig(collect=collect)


def _parse_security(raw: Dict[str, Any]) -> SecurityConfig:
    secrets_raw = raw.get("secrets", [])
    if secrets_raw is None:
        secrets_raw = []
    if not isinstance(secrets_raw, list):
        raise ManifestValidationError("security.secrets must be a list")
    secrets = [str(item).strip() for item in secrets_raw if str(item).strip()]
    return SecurityConfig(secrets=secrets)


def _validate_entrypoint(entrypoint: Dict[str, Any], executor: WorkflowExecutorConfig) -> None:
    if executor.type == "local_cli":
        command = str(entrypoint.get("command", "")).strip()
        if not command:
            raise ManifestValidationError("spec.entrypoint.command is required for local_cli executor")
        return

    submit = entrypoint.get("submit")
    if not isinstance(submit, dict):
        raise ManifestValidationError("spec.entrypoint.submit is required for remote/managed executors")

    method = str(submit.get("method", "")).strip()
    url = str(submit.get("url", "")).strip()
    if not method or not url:
        raise ManifestValidationError("spec.entrypoint.submit requires method and url")


def _validate_cross_sections(spec: WorkflowSpec, monitoring: MonitoringConfig) -> None:
    output_names = {item.name for item in spec.outputs}
    for name in monitoring.success.required_outputs:
        if name not in output_names:
            raise ManifestValidationError(
                f"monitoring.success.required_outputs references unknown output: {name!r}"
            )


def _validate_value_type(value_type: str, label: str) -> None:
    if value_type not in SUPPORTED_VALUE_TYPES:
        raise ManifestValidationError(
            f"{label} must be one of {sorted(SUPPORTED_VALUE_TYPES)}, got: {value_type!r}"
        )


def _validate_scalar_type(value: Any, value_type: str, label: str) -> None:
    if value_type in {"string", "file", "directory"} and not isinstance(value, str):
        raise ManifestValidationError(f"{label} must be a string")
    if value_type == "integer" and not isinstance(value, int):
        raise ManifestValidationError(f"{label} must be an integer")
    if value_type == "number" and not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{label} must be a number")
    if value_type == "boolean" and not isinstance(value, bool):
        raise ManifestValidationError(f"{label} must be a boolean")
    if value_type == "array" and not isinstance(value, list):
        raise ManifestValidationError(f"{label} must be an array")
    if value_type == "object" and not isinstance(value, dict):
        raise ManifestValidationError(f"{label} must be an object")


def _ensure_unique_names(names: List[str], label: str) -> None:
    seen = set()
    duplicates = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ManifestValidationError(f"{label} has duplicate names: {sorted(duplicates)}")


def _as_dict(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{label} must be a mapping")
    return value
