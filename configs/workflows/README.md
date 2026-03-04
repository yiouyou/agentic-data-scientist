# Workflow Manifest Examples

This directory contains example workflow manifests for the `ads.workflow/v1` schema.

- `demo.smoke.local_echo.yaml`: zero-dependency local smoke workflow for execution routing tests.
- `demo.smoke.local_args.yaml`: local smoke workflow proving `workflow_inputs/workflow_params` passthrough.
- `bio.rnaseq.nfcore_deseq2.yaml`: local CLI/Nextflow workflow example.
- `finance.factor_backtest.remote.yaml`: remote API job workflow example.

Default registry lookup paths include this directory (`configs/workflows`).
