"""Method Card schema for Innovation OS.

A Method Card is a structured representation of a research approach —
it captures the hypothesis, assumptions, failure modes, and expected
artifacts so the system can compare, select, and trace methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_VALID_FAMILIES = {"baseline", "negative_variant", "triz_resolution", "abductive_hypothesis"}
_VALID_STATUSES = {"proposed", "selected", "standby", "failed", "succeeded"}

_REQUIRED_STRING_FIELDS = ("method_id", "method_family", "title", "core_hypothesis", "cheap_test")
_REQUIRED_LIST_FIELDS = (
    "assumptions",
    "invalid_if",
    "failure_modes",
    "required_capabilities",
    "expected_artifacts",
    "orthogonality_tags",
)


def make_method_card(
    *,
    method_id: str,
    method_family: str,
    title: str,
    core_hypothesis: str,
    assumptions: List[str],
    invalid_if: List[str],
    cheap_test: str,
    failure_modes: List[str],
    required_capabilities: List[str],
    expected_artifacts: List[str],
    orthogonality_tags: List[str],
    status: str = "proposed",
    selection_score: Optional[float] = None,
    rejection_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a normalized method card record."""
    return {
        "method_id": str(method_id).strip(),
        "method_family": str(method_family).strip(),
        "title": str(title).strip(),
        "core_hypothesis": str(core_hypothesis).strip(),
        "assumptions": [str(a).strip() for a in (assumptions or [])],
        "invalid_if": [str(c).strip() for c in (invalid_if or [])],
        "cheap_test": str(cheap_test).strip(),
        "failure_modes": [str(f).strip() for f in (failure_modes or [])],
        "required_capabilities": [str(c).strip() for c in (required_capabilities or [])],
        "expected_artifacts": [str(a).strip() for a in (expected_artifacts or [])],
        "orthogonality_tags": [str(t).strip() for t in (orthogonality_tags or [])],
        "status": str(status).strip() if status in _VALID_STATUSES else "proposed",
        "selection_score": float(selection_score) if selection_score is not None else None,
        "rejection_reason": str(rejection_reason).strip() if rejection_reason else None,
    }


def validate_method_card(card: Dict[str, Any]) -> List[str]:
    """Validate method card structure, return list of errors (empty = valid)."""
    errors: List[str] = []

    if not isinstance(card, dict):
        return ["method card must be a dict"]

    for field_name in _REQUIRED_STRING_FIELDS:
        val = card.get(field_name)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"missing or empty required field: {field_name}")

    for field_name in _REQUIRED_LIST_FIELDS:
        val = card.get(field_name)
        if not isinstance(val, list):
            errors.append(f"field must be a list: {field_name}")

    family = card.get("method_family", "")
    if isinstance(family, str) and family.strip() and family.strip() not in _VALID_FAMILIES:
        errors.append(f"invalid method_family: {family!r} (expected one of {_VALID_FAMILIES})")

    status = card.get("status", "")
    if isinstance(status, str) and status.strip() and status.strip() not in _VALID_STATUSES:
        errors.append(f"invalid status: {status!r} (expected one of {_VALID_STATUSES})")

    return errors


def method_card_summary(card: Dict[str, Any]) -> str:
    """One-line summary of a method card for use in negative prompting."""
    title = card.get("title", "?")
    hypothesis = card.get("core_hypothesis", "?")
    tags = ", ".join(card.get("orthogonality_tags", [])[:5])
    return f"[{card.get('method_id', '?')}] {title} — {hypothesis} (tags: {tags})"
