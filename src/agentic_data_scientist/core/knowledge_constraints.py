"""Minimal knowledge-constraint normalization and validation for stage plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


_STAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SN_PATTERN = re.compile(r"^s(\d+)$", re.IGNORECASE)


@dataclass
class ConstraintValidationResult:
    """Normalization and validation result for stage constraints."""

    stages: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _ordered_dedupe(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return _ordered_dedupe(values)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values = [str(item).strip() for item in parsed if str(item).strip()]
                    return _ordered_dedupe(values)
            except Exception:
                pass
        values = [item.strip() for item in raw.split(",") if item.strip()]
        return _ordered_dedupe(values)

    return [str(value).strip()] if str(value).strip() else []


def _normalize_path_list(value: Any) -> List[str]:
    paths = _coerce_string_list(value)
    normalized = [path.replace("\\", "/") for path in paths]
    return _ordered_dedupe(normalized)


def _normalize_stage_id(raw_stage_id: Any, *, default_stage_id: str) -> str:
    if isinstance(raw_stage_id, str):
        candidate = raw_stage_id.strip()
        if candidate and _STAGE_ID_PATTERN.fullmatch(candidate):
            return candidate
    return default_stage_id


def _resolve_dep_token(token: str, stages: List[Dict[str, Any]], id_to_stage: Dict[str, Dict[str, Any]]) -> str | None:
    if token in id_to_stage:
        return token

    lower_token = token.lower()
    for stage in stages:
        title = str(stage.get("title", "")).strip().lower()
        if title and title == lower_token:
            return str(stage["stage_id"])

    if token.isdigit():
        idx = int(token)
        if 0 <= idx < len(stages):
            return str(stages[idx]["stage_id"])

    sn_match = _SN_PATTERN.fullmatch(token.strip())
    if sn_match:
        idx = int(sn_match.group(1)) - 1
        if 0 <= idx < len(stages):
            return str(stages[idx]["stage_id"])

    return None


def _find_cycle(graph: Dict[str, List[str]]) -> List[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> List[str]:
        if node in visiting:
            try:
                start = stack.index(node)
            except ValueError:
                start = 0
            return stack[start:] + [node]

        if node in visited:
            return []

        visiting.add(node)
        stack.append(node)
        for dep in graph.get(node, []):
            cycle = dfs(dep)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return []

    for node in graph:
        cycle = dfs(node)
        if cycle:
            return cycle
    return []


def normalize_and_validate_stage_constraints(
    stages: List[Dict[str, Any]],
    *,
    apply_sequential_defaults: bool = True,
) -> ConstraintValidationResult:
    """
    Normalize and validate minimal stage knowledge constraints.

    Enforced fields:
    - stage_id
    - depends_on
    - inputs_required
    - outputs_produced
    - evidence_refs
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(stages, list):
        return ConstraintValidationResult(stages=[], errors=["stages must be a list"], warnings=[])

    normalized: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for idx, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            errors.append(f"stage[{idx}] must be an object")
            continue

        stage_copy = dict(stage)
        default_stage_id = f"s{idx + 1}"
        stage_id = _normalize_stage_id(stage_copy.get("stage_id"), default_stage_id=default_stage_id)

        if stage_id in used_ids:
            errors.append(f"duplicate stage_id: {stage_id!r}")
            continue
        used_ids.add(stage_id)

        stage_copy["stage_id"] = stage_id
        stage_copy["inputs_required"] = _normalize_path_list(stage_copy.get("inputs_required"))
        stage_copy["outputs_produced"] = _normalize_path_list(stage_copy.get("outputs_produced"))
        stage_copy["evidence_refs"] = _coerce_string_list(stage_copy.get("evidence_refs"))
        normalized.append(stage_copy)

    if errors:
        return ConstraintValidationResult(stages=normalized, errors=errors, warnings=warnings)

    id_to_stage = {str(stage["stage_id"]): stage for stage in normalized}
    graph: Dict[str, List[str]] = {}

    for idx, stage in enumerate(normalized):
        stage_id = str(stage["stage_id"])
        raw_deps = stage.get("depends_on")
        dep_tokens = _coerce_string_list(raw_deps)

        if raw_deps is None and apply_sequential_defaults and idx > 0:
            dep_tokens = [str(normalized[idx - 1]["stage_id"])]

        resolved_deps: List[str] = []
        for token in dep_tokens:
            resolved = _resolve_dep_token(token, normalized, id_to_stage)
            if not resolved:
                errors.append(f"stage {stage_id!r} depends_on references unknown stage: {token!r}")
                continue
            if resolved == stage_id:
                errors.append(f"stage {stage_id!r} cannot depend on itself")
                continue
            resolved_deps.append(resolved)

        stage["depends_on"] = _ordered_dedupe(resolved_deps)
        graph[stage_id] = list(stage["depends_on"])

        if not stage["outputs_produced"]:
            warnings.append(f"stage {stage_id!r} has empty outputs_produced")

    if errors:
        return ConstraintValidationResult(stages=normalized, errors=errors, warnings=warnings)

    cycle = _find_cycle(graph)
    if cycle:
        errors.append(f"dependency cycle detected: {' -> '.join(cycle)}")

    output_owner: Dict[str, str] = {}
    for stage in normalized:
        stage_id = str(stage["stage_id"])
        for output_path in stage.get("outputs_produced", []):
            owner = output_owner.get(output_path)
            if owner and owner != stage_id:
                errors.append(
                    f"outputs_produced collision for {output_path!r}: {owner!r} and {stage_id!r}"
                )
            else:
                output_owner[output_path] = stage_id

    return ConstraintValidationResult(stages=normalized, errors=errors, warnings=warnings)

