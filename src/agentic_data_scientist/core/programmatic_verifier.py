"""Programmatic verification of stage execution results.

Runs deterministic checks on stage outputs *before* the LLM-based
criteria_checker so that the checker receives factual signals.

Two tiers:
- **Hard checks** (blocking): file existence, schema validity, no tracebacks.
- **Soft signals** (advisory): metric extraction, image validity, artifact coverage.

Verdict: ``"pass"`` | ``"warn"`` | ``"fail"``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|"
    r"^\s*File \".+\", line \d+|"
    r"^\w+Error:|"
    r"^\w+Exception:",
    re.MULTILINE,
)

_METRIC_PATTERNS: Dict[str, re.Pattern] = {
    "accuracy": re.compile(r"(?:accuracy|acc)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "auc": re.compile(r"(?:auc|auroc|roc.auc)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "f1": re.compile(r"(?:f1|f1.score)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "p_value": re.compile(r"(?:p.val(?:ue)?|p_val)\s*[:=]\s*([\d.eE\-+]+)", re.IGNORECASE),
    "r_squared": re.compile(r"(?:r2|r.squared|r\u00b2)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "rmse": re.compile(r"(?:rmse)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "mae": re.compile(r"(?:mae)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "loss": re.compile(r"(?:loss)\s*[:=]\s*([\d.eE\-+]+)", re.IGNORECASE),
    "precision": re.compile(r"(?:precision)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    "recall": re.compile(r"(?:recall|sensitivity)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
}

# Magic bytes → format for binary image validation
_IMAGE_MAGIC: List[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),
    (b"BM", "bmp"),
]

_SVG_PATTERN = re.compile(r"<svg\b", re.IGNORECASE)


@dataclass
class HardChecks:
    """Blocking verification results — any failure ⇒ verdict ``"fail"``."""

    file_exists: bool = True
    schema_valid: bool = True
    no_traceback: bool = True

    missing_files: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    traceback_files: List[str] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return self.file_exists and self.schema_valid and self.no_traceback


@dataclass
class SoftSignals:
    """Advisory signals — used for ``"warn"`` verdicts and criteria_checker context."""

    metrics: Dict[str, float] = field(default_factory=dict)
    image_valid: Dict[str, bool] = field(default_factory=dict)
    artifact_coverage: float = 1.0

    @property
    def has_warnings(self) -> bool:
        if self.artifact_coverage < 0.8:
            return True
        if any(not v for v in self.image_valid.values()):
            return True
        return False


@dataclass
class VerificationResult:
    """Combined hard-check + soft-signal result with overall verdict."""

    hard_checks: HardChecks = field(default_factory=HardChecks)
    soft_signals: SoftSignals = field(default_factory=SoftSignals)
    verdict: str = "pass"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_summary(self) -> str:
        """Human-readable summary for injection into criteria_checker prompt."""
        lines = [f"Programmatic Verification: **{self.verdict.upper()}**"]

        hc = self.hard_checks
        if not hc.file_exists:
            lines.append(f"  - Missing files: {', '.join(hc.missing_files)}")
        if not hc.schema_valid:
            lines.append(f"  - Schema errors: {'; '.join(hc.schema_errors)}")
        if not hc.no_traceback:
            lines.append(f"  - Tracebacks found in: {', '.join(hc.traceback_files)}")

        ss = self.soft_signals
        if ss.metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in ss.metrics.items())
            lines.append(f"  - Extracted metrics: {metrics_str}")
        if ss.image_valid:
            invalid = [k for k, v in ss.image_valid.items() if not v]
            if invalid:
                lines.append(f"  - Invalid images: {', '.join(invalid)}")
        if ss.artifact_coverage < 1.0:
            lines.append(f"  - Artifact coverage: {ss.artifact_coverage:.0%}")

        return "\n".join(lines)


def check_files_exist(
    working_dir: str | Path,
    expected_files: List[str],
) -> tuple[bool, List[str]]:
    """Return ``(all_exist, missing_list)`` for expected output files."""
    base = Path(working_dir)
    missing: List[str] = []
    for fpath in expected_files:
        resolved = Path(fpath) if Path(fpath).is_absolute() else base / fpath
        if not resolved.exists():
            missing.append(fpath)
    return len(missing) == 0, missing


def check_csv_schema(
    filepath: str | Path,
    *,
    min_rows: int = 0,
    expected_columns: Optional[List[str]] = None,
) -> tuple[bool, str]:
    """Return ``(valid, error_msg)`` after parsing a CSV and optionally checking columns/row-count."""
    path = Path(filepath)
    if not path.exists():
        return False, f"File not found: {path}"
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return False, f"Empty CSV file: {path.name}"

            if expected_columns:
                header_lower = [h.strip().lower() for h in header]
                missing_cols = [c for c in expected_columns if c.strip().lower() not in header_lower]
                if missing_cols:
                    return False, (f"{path.name}: missing columns {missing_cols}. Found: {header[:10]}")

            row_count = sum(1 for _ in reader)
            if row_count < min_rows:
                return False, (f"{path.name}: only {row_count} data rows, expected >= {min_rows}")
        return True, ""
    except Exception as exc:
        return False, f"{path.name}: CSV parse error: {exc}"


def check_json_schema(filepath: str | Path) -> tuple[bool, str]:
    """Return ``(valid, error_msg)`` after attempting to parse JSON."""
    path = Path(filepath)
    if not path.exists():
        return False, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            json.load(f)
        return True, ""
    except json.JSONDecodeError as exc:
        return False, f"{path.name}: JSON parse error: {exc}"
    except Exception as exc:
        return False, f"{path.name}: read error: {exc}"


def check_no_traceback(
    filepath: str | Path,
    *,
    max_bytes: int = 512_000,
) -> tuple[bool, str]:
    """Return ``(clean, snippet)`` — reads last *max_bytes* of a text file for traceback patterns."""
    path = Path(filepath)
    if not path.exists():
        return True, ""
    try:
        size = path.stat().st_size
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            content = f.read()
        match = _TRACEBACK_RE.search(content)
        if match:
            start = max(0, match.start() - 40)
            end = min(len(content), match.end() + 120)
            snippet = content[start:end].strip()
            return False, snippet
        return True, ""
    except Exception:
        return True, ""


def extract_metrics_from_text(text: str) -> Dict[str, float]:
    """Extract known numeric metrics (accuracy, AUC, p-value, …) from free-form text."""
    found: Dict[str, float] = {}
    for name, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            try:
                found[name] = float(match.group(1))
            except (ValueError, IndexError):
                pass
    return found


def extract_metrics_from_file(filepath: str | Path, *, max_bytes: int = 256_000) -> Dict[str, float]:
    """Extract metrics from a text file (reads up to *max_bytes*)."""
    path = Path(filepath)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
        return extract_metrics_from_text(content)
    except Exception:
        return {}


def validate_image_file(filepath: str | Path) -> bool:
    """Check if a file is a valid image by inspecting magic bytes (PNG/JPEG/GIF/WebP/BMP/SVG)."""
    path = Path(filepath)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(16)

        for magic, _ in _IMAGE_MAGIC:
            if header.startswith(magic):
                return True

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head_text = f.read(512)
            if _SVG_PATTERN.search(head_text):
                return True
        except Exception:
            pass

        return False
    except Exception:
        return False


def run_programmatic_checks(
    working_dir: str | Path,
    stage: Dict[str, Any],
    criteria: List[Dict[str, Any]],
) -> VerificationResult:
    """Run all programmatic checks for a completed stage.

    Parameters
    ----------
    working_dir
        Root working directory for the analysis.
    stage
        Completed stage record (from ``make_stage_record``).
    criteria
        All success criteria (from ``make_success_criterion_record``).

    Returns
    -------
    VerificationResult
        Combined result with ``verdict`` in ``{"pass", "warn", "fail"}``.
    """
    result = VerificationResult()
    base = Path(working_dir)

    if not base.exists():
        result.hard_checks.file_exists = False
        result.hard_checks.missing_files.append(str(working_dir))
        result.verdict = "fail"
        return result

    outputs_produced = stage.get("outputs_produced", [])
    implementation_result = stage.get("implementation_result", "") or ""

    # Hard Check 1: File existence
    if outputs_produced:
        all_exist, missing = check_files_exist(base, outputs_produced)
        if not all_exist:
            result.hard_checks.file_exists = False
            result.hard_checks.missing_files = missing

    # Hard Check 2: Schema validity for CSV/JSON
    for output_path in outputs_produced:
        resolved = Path(output_path) if Path(output_path).is_absolute() else base / output_path
        if not resolved.exists():
            continue

        suffix = resolved.suffix.lower()
        if suffix == ".csv":
            valid, err = check_csv_schema(resolved)
            if not valid:
                result.hard_checks.schema_valid = False
                result.hard_checks.schema_errors.append(err)
        elif suffix == ".json":
            valid, err = check_json_schema(resolved)
            if not valid:
                result.hard_checks.schema_valid = False
                result.hard_checks.schema_errors.append(err)

    # Hard Check 3: Traceback detection
    _scan_extensions = {".py", ".log", ".txt", ".md", ".stderr", ".out"}
    for output_path in outputs_produced:
        resolved = Path(output_path) if Path(output_path).is_absolute() else base / output_path
        if not resolved.exists():
            continue
        if resolved.suffix.lower() in _scan_extensions:
            clean, snippet = check_no_traceback(resolved)
            if not clean:
                result.hard_checks.no_traceback = False
                result.hard_checks.traceback_files.append(output_path)
                logger.warning(
                    "[ProgrammaticVerifier] Traceback in %s: %s",
                    output_path,
                    snippet[:200],
                )

    if implementation_result:
        match = _TRACEBACK_RE.search(implementation_result)
        if match:
            result.hard_checks.no_traceback = False
            result.hard_checks.traceback_files.append("<implementation_result>")

    # Soft Signal 1: Metric extraction
    all_metrics: Dict[str, float] = {}
    if implementation_result:
        all_metrics.update(extract_metrics_from_text(implementation_result))
    _text_extensions = {".txt", ".log", ".md", ".json", ".csv", ".tsv", ".out"}
    for output_path in outputs_produced:
        resolved = Path(output_path) if Path(output_path).is_absolute() else base / output_path
        if resolved.exists() and resolved.suffix.lower() in _text_extensions:
            file_metrics = extract_metrics_from_file(resolved)
            for k, v in file_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = v
    result.soft_signals.metrics = all_metrics

    # Soft Signal 2: Image validation
    _image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}
    for output_path in outputs_produced:
        resolved = Path(output_path) if Path(output_path).is_absolute() else base / output_path
        if resolved.suffix.lower() in _image_extensions:
            result.soft_signals.image_valid[output_path] = validate_image_file(resolved)

    # Soft Signal 3: Artifact coverage
    if outputs_produced:
        found = sum(1 for p in outputs_produced if (base / p).exists() or Path(p).exists())
        result.soft_signals.artifact_coverage = found / len(outputs_produced)
    else:
        result.soft_signals.artifact_coverage = 1.0

    # Determine verdict
    if not result.hard_checks.all_pass:
        result.verdict = "fail"
    elif result.soft_signals.has_warnings:
        result.verdict = "warn"
    else:
        result.verdict = "pass"

    logger.info(
        "[ProgrammaticVerifier] Stage '%s' verdict=%s (hard=%s, coverage=%.0f%%)",
        stage.get("title", "?"),
        result.verdict,
        "PASS" if result.hard_checks.all_pass else "FAIL",
        result.soft_signals.artifact_coverage * 100,
    )

    return result
