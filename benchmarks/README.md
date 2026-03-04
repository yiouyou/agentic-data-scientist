# Synthetic Benchmark Suite

This folder contains a lightweight benchmark suite for fast validation when real tasks are unavailable.

## Files

- `tasks.yaml`: benchmark task definitions
- `fixtures/`: small synthetic input files for benchmark tasks

## Quick Start

Dry-run (plan commands only, no model calls):

```bash
uv run python scripts/run_benchmark.py
```

Execute selected tasks:

```bash
uv run python scripts/run_benchmark.py --execute --task-id bio_rnaseq_nfcore_route --task-id history_replay_report
```

## Notes

- Default output root is `.tmp/benchmark_runs/<timestamp>/`.
- Each run stores `command.txt`, and when executed also stores `stdout.txt` and `stderr.txt`.
- A consolidated `summary.json` and `summary.md` are generated per suite run.
- `history_replay_report` task requires prior history data to produce non-zero records.
