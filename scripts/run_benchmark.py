#!/usr/bin/env python
"""
Run synthetic benchmark suites for Agentic Data Scientist.

Default behavior is dry-run planning only. Use --execute to run commands.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunRecord:
    task_id: str
    title: str
    run_index: int
    mode: str
    command: list[str]
    status: str
    exit_code: int | None
    duration_sec: float
    checks_passed: int
    checks_total: int
    output_dir: str
    log_file: str | None
    notes: list[str]


def _load_suite(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid suite format: {path}")
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks.yaml must contain a non-empty 'tasks' list")
    return data


def _resolve_tasks(
    suite: dict[str, Any],
    task_ids: set[str] | None,
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    tasks = suite["tasks"]
    selected: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            continue
        if task_ids and task_id not in task_ids:
            continue
        selected.append(task)
    if max_tasks is not None and max_tasks > 0:
        selected = selected[:max_tasks]
    return selected


def _build_command(
    *,
    task: dict[str, Any],
    defaults: dict[str, Any],
    fixtures_dir: Path,
    run_output_dir: Path,
    use_uv: bool,
    cli_bin: str,
) -> tuple[list[str], Path | None, str]:
    task_type = str(task.get("type", "query")).strip().lower()
    mode = str(task.get("mode", defaults.get("mode", "orchestrated"))).strip()
    if task_type == "history_replay":
        command = [cli_bin, "--history-replay", "--history-replay-limit", str(task.get("history_replay_limit", 200))]
        if use_uv:
            command = ["uv", "run", *command]
        return command, None, "history_replay"

    prompt = str(task.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Task {task.get('id')} missing prompt")

    log_file = run_output_dir / ".agentic_ds.log"
    command = [cli_bin, prompt, "--mode", mode, "--working-dir", str(run_output_dir), "--log-file", str(log_file)]
    for rel_path in task.get("files", []) or []:
        fixture_path = fixtures_dir / str(rel_path)
        command.extend(["--files", str(fixture_path)])

    for extra_arg in task.get("extra_args", []) or []:
        command.append(str(extra_arg))

    if use_uv:
        command = ["uv", "run", *command]
    return command, log_file, mode


def _run_command(command: list[str], env: dict[str, str], timeout_sec: int) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
            check=False,
        )
        duration = time.perf_counter() - start
        return proc.returncode, proc.stdout or "", proc.stderr or "", duration
    except subprocess.TimeoutExpired as timeout_error:
        duration = time.perf_counter() - start
        stdout_text = timeout_error.stdout or ""
        stderr_text = timeout_error.stderr or ""
        if isinstance(stdout_text, bytes):
            stdout_text = stdout_text.decode("utf-8", errors="ignore")
        if isinstance(stderr_text, bytes):
            stderr_text = stderr_text.decode("utf-8", errors="ignore")
        timeout_note = f"[benchmark] timeout after {timeout_sec}s"
        if timeout_note not in stderr_text:
            stderr_text = f"{stderr_text}\n{timeout_note}".strip()
        return 124, stdout_text, stderr_text, duration


def _evaluate_checks(
    *,
    task: dict[str, Any],
    stdout_text: str,
    log_file: Path | None,
    run_output_dir: Path,
) -> tuple[int, int, list[str]]:
    expected = task.get("expected", {}) or {}
    checks_total = 0
    checks_passed = 0
    notes: list[str] = []

    lowered_stdout = stdout_text.lower()
    for token in expected.get("response_contains", []) or []:
        checks_total += 1
        token_text = str(token).lower()
        ok = token_text in lowered_stdout
        checks_passed += int(ok)
        if not ok:
            notes.append(f"missing response token: {token}")

    log_content = ""
    if log_file and log_file.exists():
        log_content = log_file.read_text(encoding="utf-8", errors="ignore").lower()
    for token in expected.get("log_contains", []) or []:
        checks_total += 1
        token_text = str(token).lower()
        ok = token_text in log_content
        checks_passed += int(ok)
        if not ok:
            notes.append(f"missing log token: {token}")

    for rel_path in expected.get("files_exist", []) or []:
        checks_total += 1
        file_path = run_output_dir / str(rel_path)
        ok = file_path.exists()
        checks_passed += int(ok)
        if not ok:
            notes.append(f"missing file: {rel_path}")

    return checks_passed, checks_total, notes


def _write_summary(records: list[RunRecord], output_root: Path, suite_name: str, execute: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"

    payload = {
        "suite": suite_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "execute": execute,
        "records": [record.__dict__ for record in records],
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    total = len(records)
    passed = sum(1 for r in records if r.status == "passed")
    failed = sum(1 for r in records if r.status == "failed")
    planned = sum(1 for r in records if r.status == "planned")

    lines = [
        "# Benchmark Summary",
        "",
        f"- Suite: `{suite_name}`",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Mode: `{'execute' if execute else 'dry-run'}`",
        f"- Runs: `{total}` (`passed={passed}`, `failed={failed}`, `planned={planned}`)",
        "",
        "| task_id | run | status | checks | duration_sec | output_dir |",
        "|---|---:|---|---:|---:|---|",
    ]
    for rec in records:
        lines.append(
            "| "
            f"{rec.task_id} | {rec.run_index} | {rec.status} | "
            f"{rec.checks_passed}/{rec.checks_total} | {rec.duration_sec:.2f} | {rec.output_dir} |"
        )
    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic benchmark tasks.")
    parser.add_argument("--tasks", default="benchmarks/tasks.yaml", help="Path to benchmark tasks yaml.")
    parser.add_argument("--fixtures-dir", default="benchmarks/fixtures", help="Path to fixture directory.")
    parser.add_argument(
        "--output-dir",
        default=".tmp/benchmark_runs",
        help="Root directory for run outputs and summary artifacts.",
    )
    parser.add_argument("--task-id", action="append", default=[], help="Run only specific task id(s), repeatable.")
    parser.add_argument("--max-tasks", type=int, default=0, help="Run at most N selected tasks (0 means all).")
    parser.add_argument("--execute", action="store_true", help="Execute commands. Default is dry-run only.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Set ADS_PLAN_ONLY=true for query tasks to benchmark planning without execution stage.",
    )
    parser.add_argument("--timeout-sec", type=int, default=1200, help="Per-run timeout in seconds.")
    parser.add_argument("--cli-bin", default="agentic-data-scientist", help="CLI binary name.")
    parser.add_argument("--no-uv", action="store_true", help="Do not prefix commands with 'uv run'.")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    fixtures_dir = Path(args.fixtures_dir)
    output_root = Path(args.output_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    suite = _load_suite(tasks_path)
    defaults = suite.get("defaults", {}) or {}
    task_ids = {task_id.strip() for task_id in args.task_id if task_id.strip()} or None
    selected_tasks = _resolve_tasks(suite, task_ids=task_ids, max_tasks=(args.max_tasks or None))
    if not selected_tasks:
        print("No tasks selected.")
        return 2

    records: list[RunRecord] = []
    for task in selected_tasks:
        task_id = str(task.get("id"))
        title = str(task.get("title", task_id))
        repeat = int(task.get("repeat", defaults.get("repeat", 1)) or 1)
        timeout_sec = int(task.get("timeout_sec", args.timeout_sec) or args.timeout_sec)
        repeat = max(1, repeat)

        for run_idx in range(1, repeat + 1):
            run_dir = run_root / task_id / f"run_{run_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                command, log_file, mode = _build_command(
                    task=task,
                    defaults=defaults,
                    fixtures_dir=fixtures_dir,
                    run_output_dir=run_dir,
                    use_uv=not args.no_uv,
                    cli_bin=args.cli_bin,
                )
            except Exception as e:
                records.append(
                    RunRecord(
                        task_id=task_id,
                        title=title,
                        run_index=run_idx,
                        mode="invalid",
                        command=[],
                        status="failed",
                        exit_code=None,
                        duration_sec=0.0,
                        checks_passed=0,
                        checks_total=0,
                        output_dir=str(run_dir),
                        log_file=None,
                        notes=[f"build_command_error: {e}"],
                    )
                )
                continue

            env = os.environ.copy()
            task_env = task.get("env", {}) or {}
            for key, value in task_env.items():
                env[str(key)] = str(value)
            if args.plan_only and str(task.get("type", "query")).strip().lower() != "history_replay":
                env["ADS_PLAN_ONLY"] = "true"

            command_file = run_dir / "command.txt"
            command_file.write_text(" ".join(command), encoding="utf-8")
            env_file = run_dir / "env_overrides.json"
            env_overrides = {str(k): str(v) for k, v in task_env.items()}
            if args.plan_only and str(task.get("type", "query")).strip().lower() != "history_replay":
                env_overrides["ADS_PLAN_ONLY"] = "true"
            if env_overrides:
                env_file.write_text(json.dumps(env_overrides, ensure_ascii=True, indent=2), encoding="utf-8")

            if not args.execute:
                records.append(
                    RunRecord(
                        task_id=task_id,
                        title=title,
                        run_index=run_idx,
                        mode=mode,
                        command=command,
                        status="planned",
                        exit_code=None,
                        duration_sec=0.0,
                        checks_passed=0,
                        checks_total=0,
                        output_dir=str(run_dir),
                        log_file=str(log_file) if log_file else None,
                        notes=[],
                    )
                )
                continue

            exit_code, stdout_text, stderr_text, duration = _run_command(command, env=env, timeout_sec=timeout_sec)
            (run_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
            (run_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")
            checks_passed, checks_total, notes = _evaluate_checks(
                task=task,
                stdout_text=stdout_text,
                log_file=log_file,
                run_output_dir=run_dir,
            )

            status = "passed"
            if exit_code != 0:
                status = "failed"
                notes.append(f"nonzero_exit: {exit_code}")
                if exit_code == 124:
                    notes.append(f"timeout_sec: {timeout_sec}")
            elif checks_total > 0 and checks_passed < checks_total:
                status = "failed"

            records.append(
                RunRecord(
                    task_id=task_id,
                    title=title,
                    run_index=run_idx,
                    mode=mode,
                    command=command,
                    status=status,
                    exit_code=exit_code,
                    duration_sec=duration,
                    checks_passed=checks_passed,
                    checks_total=checks_total,
                    output_dir=str(run_dir),
                    log_file=str(log_file) if log_file else None,
                    notes=notes,
                )
            )

    _write_summary(records, run_root, suite_name=str(suite.get("name", "benchmark-suite")), execute=args.execute)
    print(f"Benchmark summary written to: {run_root}")

    if args.execute and any(r.status == "failed" for r in records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
