"""Unit tests for learning-driven plan ranking helpers."""

from agentic_data_scientist.core.plan_learning import (
    extract_stage_titles_from_plan,
    rank_plan_candidates,
    replay_selection_records,
)


def test_extract_stage_titles_from_plan_numbered_lines():
    plan = """
1. **Data QC** - validate files
2. Feature Engineering - derive covariates
3. Model Training - fit baseline model
"""
    titles = extract_stage_titles_from_plan(plan)
    assert titles == ["Data QC", "Feature Engineering - derive covariates", "Model Training - fit baseline model"]


def test_rank_plan_candidates_prefers_higher_historical_alignment():
    user_request = "rnaseq differential expression analysis"
    candidates = [
        "1. Load data\n2. Clean data\n3. Plot charts",
        "1. RNA-seq QC and alignment\n2. Differential expression with DESeq2\n3. Interpret pathways",
    ]
    history_signals = {
        "hot": {"run_count": 20, "stage_retry_rate": 0.1, "top_workflows": []},
        "topk_similar_runs": [
            {"stage_titles": ["RNA-seq QC and alignment", "Differential expression with DESeq2"]},
        ],
    }
    ranking = rank_plan_candidates(
        user_request=user_request,
        candidates=candidates,
        history_signals=history_signals,
        baseline_index=0,
        min_switch_margin=0.01,
    )
    assert ranking["selected_index"] == 1
    assert ranking["switch_applied"] is True


def test_replay_selection_records_returns_summary():
    records = [
        {"switch_applied": True, "observed_reward": 0.8, "policy_gain_proxy": 0.2},
        {"switch_applied": False, "observed_reward": 0.6, "policy_gain_proxy": -0.1},
    ]
    summary = replay_selection_records(records)
    assert summary["records"] == 2
    assert summary["switch_rate"] == 0.5
    assert summary["avg_observed_reward"] > 0.0
