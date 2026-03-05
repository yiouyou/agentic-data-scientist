"""Stage-level helper functions for skill-aware execution hints."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping


_BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")
_SPLIT_PATTERN = re.compile(r"[\n;]+")
_SEQUENTIAL_MARKERS = (
    " then ",
    " after ",
    " once ",
    " based on ",
    " using output",
    " finally ",
)


def _normalize_subtask(item: Mapping[str, Any], index: int) -> Dict[str, Any]:
    title = str(item.get("title", "")).strip()
    description = str(item.get("description", "")).strip()
    if not title:
        title = description or f"subtask_{index + 1}"
    if not description:
        description = title

    depends_on_raw = item.get("depends_on", [])
    depends_on: List[str] = []
    if isinstance(depends_on_raw, list):
        depends_on = [str(token).strip() for token in depends_on_raw if str(token).strip()]

    can_parallel = bool(item.get("can_parallel", False))
    return {
        "id": f"t{index + 1}",
        "title": title,
        "description": description,
        "can_parallel": can_parallel and not depends_on,
        "depends_on": depends_on,
    }


def _derive_text_subtasks(description: str, max_subtasks: int) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    candidates: List[str] = []

    for line in lines:
        match = _BULLET_PATTERN.match(line)
        if match:
            candidates.append(match.group(1).strip())

    if len(candidates) < 2:
        pieces = []
        for chunk in _SPLIT_PATTERN.split(description):
            text = str(chunk).strip()
            if not text:
                continue
            if len(text.split()) < 4:
                continue
            pieces.append(text)
        candidates = pieces

    results: List[Dict[str, Any]] = []
    for idx, text in enumerate(candidates[: max(1, int(max_subtasks))]):
        lowered = f" {text.lower()} "
        is_sequential = any(marker in lowered for marker in _SEQUENTIAL_MARKERS)
        depends_on = [f"t{idx}"] if is_sequential and idx > 0 else []
        results.append(
            {
                "id": f"t{idx + 1}",
                "title": text[:120],
                "description": text,
                "can_parallel": (not is_sequential) and not depends_on,
                "depends_on": depends_on,
            }
        )
    return results


def derive_parallel_subtasks(
    *,
    stage: Mapping[str, Any],
    max_subtasks: int = 5,
) -> List[Dict[str, Any]]:
    """Derive stage-level subtask candidates and mark parallel-safe ones."""
    subtasks_raw = stage.get("subtasks")
    if isinstance(subtasks_raw, list) and subtasks_raw:
        normalized = []
        for idx, item in enumerate(subtasks_raw):
            if not isinstance(item, Mapping):
                continue
            normalized.append(_normalize_subtask(item, idx))
        if normalized:
            return normalized[: max(1, int(max_subtasks))]

    description = str(stage.get("description", "")).strip()
    if not description:
        return []
    return _derive_text_subtasks(description=description, max_subtasks=max_subtasks)


def render_stage_info(stage: Mapping[str, Any]) -> str:
    """Render one stage with optional skill/subtask hints for coding executors."""
    index = int(stage.get("index", 0)) + 1
    title = str(stage.get("title", "Unknown")).strip() or "Unknown"
    description = str(stage.get("description", "")).strip()
    lines = [f"Stage {index}: {title}", "", description]

    recommendations = stage.get("recommended_skills")
    if isinstance(recommendations, list) and recommendations:
        lines.append("")
        lines.append("Recommended Skills (Top-K):")
        for rec in recommendations:
            if not isinstance(rec, Mapping):
                continue
            name = str(rec.get("skill_name", "")).strip()
            summary = str(rec.get("summary", "")).strip()
            matched = rec.get("matched_terms", [])
            matched_hint = ", ".join(str(token) for token in matched[:4]) if isinstance(matched, list) else ""
            bullet = f"- {name}" if name else "- (unknown skill)"
            if summary:
                bullet += f": {summary}"
            if matched_hint:
                bullet += f" (matched: {matched_hint})"
            lines.append(bullet)

    parallel_subtasks = stage.get("parallel_subtasks")
    if isinstance(parallel_subtasks, list) and parallel_subtasks:
        lines.append("")
        lines.append("Subtask Execution Hints:")
        lines.append("- Prefer running independent subtasks in parallel when safe.")
        for sub in parallel_subtasks:
            if not isinstance(sub, Mapping):
                continue
            sub_id = str(sub.get("id", "")).strip() or "t?"
            sub_title = str(sub.get("title", "")).strip() or str(sub.get("description", "")).strip()
            depends_on = sub.get("depends_on", [])
            deps = ", ".join(str(dep) for dep in depends_on) if isinstance(depends_on, list) else ""
            mode = "parallel-candidate" if bool(sub.get("can_parallel", False)) and not deps else "sequential"
            line = f"- [{sub_id}] ({mode}) {sub_title}"
            if deps:
                line += f" | depends_on={deps}"
            lines.append(line)

    return "\n".join(line for line in lines if line is not None).strip()
