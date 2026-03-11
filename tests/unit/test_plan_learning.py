"""Unit tests for learning-driven plan ranking helpers."""

import pytest

from agentic_data_scientist.core.plan_learning import (
    _dag_validity_score,
    _dataflow_coverage_score,
    _granularity_uniformity_score,
    extract_stage_titles_from_plan,
    rank_plan_candidates,
    replay_selection_records,
    score_plan_candidate,
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


def test_dag_validity_valid_linear_chain():
    stages = [
        {"stage_id": "s1", "index": 0, "depends_on": []},
        {"stage_id": "s2", "index": 1, "depends_on": ["s1"]},
        {"stage_id": "s3", "index": 2, "depends_on": ["s2"]},
    ]
    assert _dag_validity_score(stages) == 0.10


def test_dag_validity_cycle_detected():
    stages = [
        {"stage_id": "s1", "index": 0, "depends_on": ["s3"]},
        {"stage_id": "s2", "index": 1, "depends_on": ["s1"]},
        {"stage_id": "s3", "index": 2, "depends_on": ["s2"]},
    ]
    assert _dag_validity_score(stages) == -0.10


def test_dag_validity_dangling_reference():
    stages = [
        {"stage_id": "s1", "index": 0, "depends_on": ["nonexistent"]},
    ]
    assert _dag_validity_score(stages) == -0.10


def test_dag_validity_no_deps():
    stages = [
        {"stage_id": "s1", "index": 0, "depends_on": []},
        {"stage_id": "s2", "index": 1, "depends_on": []},
    ]
    assert _dag_validity_score(stages) == 0.0


def test_dag_validity_empty():
    assert _dag_validity_score([]) == 0.0


def test_dataflow_coverage_full():
    stages = [
        {"stage_id": "s1", "inputs_required": [], "outputs_produced": ["data.csv"]},
        {"stage_id": "s2", "inputs_required": ["data.csv"], "outputs_produced": ["model.pkl"]},
    ]
    assert _dataflow_coverage_score(stages) == 0.10


def test_dataflow_coverage_partial():
    stages = [
        {"stage_id": "s1", "inputs_required": [], "outputs_produced": ["data.csv"]},
        {"stage_id": "s2", "inputs_required": ["data.csv", "extra.csv"], "outputs_produced": []},
    ]
    score = _dataflow_coverage_score(stages)
    assert 0.0 < score < 0.10


def test_dataflow_coverage_none_satisfied():
    stages = [
        {"stage_id": "s1", "inputs_required": ["missing.csv"], "outputs_produced": []},
    ]
    assert _dataflow_coverage_score(stages) == 0.0


def test_dataflow_coverage_no_metadata():
    stages = [
        {"stage_id": "s1"},
        {"stage_id": "s2"},
    ]
    assert _dataflow_coverage_score(stages) == 0.0


def test_granularity_uniformity_uniform():
    stages = [
        {"description": "A" * 100},
        {"description": "B" * 110},
        {"description": "C" * 95},
    ]
    assert _granularity_uniformity_score(stages) == pytest.approx(0.06, abs=0.01)


def test_granularity_uniformity_wildly_different():
    stages = [
        {"description": "short"},
        {"description": "x" * 1000},
    ]
    score = _granularity_uniformity_score(stages)
    assert score < 0.03


def test_granularity_uniformity_single_stage():
    stages = [{"description": "anything"}]
    assert _granularity_uniformity_score(stages) == 0.06


def test_score_plan_candidate_with_parsed_stages():
    parsed_stages = [
        {
            "stage_id": "s1",
            "index": 0,
            "depends_on": [],
            "inputs_required": [],
            "outputs_produced": ["data.csv"],
            "description": "Load data and validate",
        },
        {
            "stage_id": "s2",
            "index": 1,
            "depends_on": ["s1"],
            "inputs_required": ["data.csv"],
            "outputs_produced": ["model.pkl"],
            "description": "Train model on dataset",
        },
    ]
    result = score_plan_candidate(
        user_request="train a model on data",
        candidate_plan="1. Load data\n2. Train model",
        history_signals={"hot": {"run_count": 20, "stage_retry_rate": 0.0}, "topk_similar_runs": []},
        is_baseline=False,
        parsed_stages=parsed_stages,
    )
    assert "structural_scores" in result
    assert result["structural_scores"]["dag_validity"] == 0.10
    assert result["structural_scores"]["dataflow_coverage"] == 0.10


def test_score_plan_candidate_without_parsed_stages_backward_compat():
    result = score_plan_candidate(
        user_request="analyze data",
        candidate_plan="1. Load data\n2. Analyze\n3. Report",
        history_signals={"hot": {"run_count": 20, "stage_retry_rate": 0.0}, "topk_similar_runs": []},
        is_baseline=True,
    )
    assert "structural_scores" not in result
    assert "score" in result


# ========================= Phase 0 Plan-Only Tests =========================


class TestP0T1_PlanScoringDifferentiatesGoodBadPlans:
    """P0-T1: Verify scoring dimensions correctly differentiate good vs bad plans."""

    COMMON_REQUEST = "Analyze sales_2023.csv sales trends, build forecast model, and generate report"
    COMMON_HISTORY = {"hot": {"run_count": 20, "stage_retry_rate": 0.0}, "topk_similar_runs": []}

    GOOD_STAGES = [
        {
            "stage_id": "s1",
            "index": 0,
            "depends_on": [],
            "inputs_required": ["sales_2023.csv"],
            "outputs_produced": ["cleaned_sales.csv", "eda_report.html"],
            "description": "Load sales_2023.csv, validate schema, handle missing values, generate EDA summary",
        },
        {
            "stage_id": "s2",
            "index": 1,
            "depends_on": ["s1"],
            "inputs_required": ["cleaned_sales.csv"],
            "outputs_produced": ["feature_matrix.csv"],
            "description": "Engineer time-based features: monthly aggregation, rolling averages, seasonal indicators",
        },
        {
            "stage_id": "s3",
            "index": 2,
            "depends_on": ["s2"],
            "inputs_required": ["feature_matrix.csv"],
            "outputs_produced": ["forecast_model.pkl", "evaluation_metrics.json"],
            "description": "Train ARIMA and Prophet models, evaluate with RMSE and MAPE on held-out data",
        },
        {
            "stage_id": "s4",
            "index": 3,
            "depends_on": ["s3", "s1"],
            "inputs_required": ["forecast_model.pkl", "evaluation_metrics.json", "eda_report.html"],
            "outputs_produced": ["final_report.html"],
            "description": "Synthesize EDA findings and model results into comprehensive trend analysis report",
        },
    ]

    BAD_STAGES_NO_DEPS = [
        {
            "stage_id": "s1",
            "index": 0,
            "depends_on": [],
            "inputs_required": [],
            "outputs_produced": [],
            "description": "Do data stuff",
        },
        {
            "stage_id": "s2",
            "index": 1,
            "depends_on": [],
            "inputs_required": [],
            "outputs_produced": [],
            "description": "Make a really complicated and extremely long-winded model that spans many "
            "paragraphs of description text to create uneven granularity" * 5,
        },
        {
            "stage_id": "s3",
            "index": 2,
            "depends_on": [],
            "inputs_required": [],
            "outputs_produced": [],
            "description": "Report",
        },
    ]

    BAD_STAGES_CYCLE = [
        {
            "stage_id": "s1",
            "index": 0,
            "depends_on": ["s2"],
            "inputs_required": ["data.csv"],
            "outputs_produced": ["output_a.csv"],
            "description": "Stage A depends on Stage B which also depends on Stage A forming a cycle",
        },
        {
            "stage_id": "s2",
            "index": 1,
            "depends_on": ["s1"],
            "inputs_required": ["output_a.csv"],
            "outputs_produced": ["data.csv"],
            "description": "Stage B depends on Stage A creating a circular dependency graph structure",
        },
    ]

    def _score(self, stages, plan_text=None):
        if plan_text is None:
            plan_text = "\n".join(f"{i + 1}. {s['description']}" for i, s in enumerate(stages))
        return score_plan_candidate(
            user_request=self.COMMON_REQUEST,
            candidate_plan=plan_text,
            history_signals=self.COMMON_HISTORY,
            is_baseline=False,
            parsed_stages=stages,
        )

    def test_good_plan_scores_higher_than_bad(self):
        good = self._score(self.GOOD_STAGES)
        bad = self._score(self.BAD_STAGES_NO_DEPS)
        assert good["score"] > bad["score"], f"Good {good['score']:.3f} should beat bad {bad['score']:.3f}"

    def test_good_plan_has_valid_dag(self):
        result = self._score(self.GOOD_STAGES)
        assert result["structural_scores"]["dag_validity"] == 0.10

    def test_good_plan_has_full_dataflow_coverage(self):
        result = self._score(self.GOOD_STAGES)
        assert result["structural_scores"]["dataflow_coverage"] > 0.05

    def test_bad_plan_no_deps_gets_zero_dag(self):
        result = self._score(self.BAD_STAGES_NO_DEPS)
        assert result["structural_scores"]["dag_validity"] == 0.0

    def test_bad_plan_no_deps_gets_zero_dataflow(self):
        result = self._score(self.BAD_STAGES_NO_DEPS)
        assert result["structural_scores"]["dataflow_coverage"] == 0.0

    def test_bad_plan_wildly_uneven_granularity(self):
        result = self._score(self.BAD_STAGES_NO_DEPS)
        assert result["structural_scores"]["granularity_uniformity"] < 0.03

    def test_good_plan_uniform_granularity(self):
        result = self._score(self.GOOD_STAGES)
        assert result["structural_scores"]["granularity_uniformity"] > 0.03

    def test_cyclic_dag_penalized(self):
        result = self._score(self.BAD_STAGES_CYCLE)
        assert result["structural_scores"]["dag_validity"] == -0.10


class TestP0T2_MultiStageComplexDependencyScoring:
    """P0-T2: Complex multi-omics query — verify dependency graph scoring distinguishes plan quality."""

    MULTI_OMICS_REQUEST = (
        "Integrate RNA-seq, proteomics, and metabolomics data from 200 patient samples. "
        "Perform QC, normalization, multi-omics factor analysis (MOFA+), classification, "
        "and survival analysis with comprehensive visualization."
    )
    COMMON_HISTORY = {"hot": {"run_count": 15, "stage_retry_rate": 0.05}, "topk_similar_runs": []}

    WELL_STRUCTURED_STAGES = [
        {
            "stage_id": "s1",
            "index": 0,
            "depends_on": [],
            "inputs_required": ["rnaseq_counts.csv", "proteomics_data.csv", "metabolomics_data.csv"],
            "outputs_produced": [
                "qc_report.html",
                "normalized_rnaseq.csv",
                "normalized_proteomics.csv",
                "normalized_metabolomics.csv",
            ],
            "description": "Quality control and normalization of all three omics datasets with batch effect correction",
        },
        {
            "stage_id": "s2",
            "index": 1,
            "depends_on": ["s1"],
            "inputs_required": ["normalized_rnaseq.csv"],
            "outputs_produced": ["deg_results.csv", "volcano_plot.png"],
            "description": "Differential expression analysis on normalized RNA-seq data using DESeq2",
        },
        {
            "stage_id": "s3",
            "index": 2,
            "depends_on": ["s1"],
            "inputs_required": ["normalized_rnaseq.csv", "normalized_proteomics.csv", "normalized_metabolomics.csv"],
            "outputs_produced": ["mofa_model.hdf5", "mofa_factors.csv", "variance_explained.png"],
            "description": "MOFA+ integrative factor analysis across all three normalized omics layers",
        },
        {
            "stage_id": "s4",
            "index": 3,
            "depends_on": ["s3", "s2"],
            "inputs_required": ["mofa_factors.csv", "deg_results.csv"],
            "outputs_produced": ["classifier.pkl", "classification_metrics.json", "roc_curve.png"],
            "description": "Train random forest classifier using MOFA factors and top DEGs as features",
        },
        {
            "stage_id": "s5",
            "index": 4,
            "depends_on": ["s4", "s3"],
            "inputs_required": ["mofa_factors.csv", "classifier.pkl", "classification_metrics.json"],
            "outputs_produced": ["survival_curves.png", "final_report.html"],
            "description": "Kaplan-Meier survival analysis and comprehensive visualization report",
        },
    ]

    FLAT_NO_STRUCTURE_STAGES = [
        {
            "stage_id": "s1",
            "index": 0,
            "depends_on": [],
            "inputs_required": [],
            "outputs_produced": [],
            "description": "Process all the omics data and do QC normalization DEG MOFA classification",
        },
        {
            "stage_id": "s2",
            "index": 1,
            "depends_on": [],
            "inputs_required": [],
            "outputs_produced": [],
            "description": "Make survival curves and visualization and a report about results",
        },
    ]

    def _score(self, stages):
        plan_text = "\n".join(f"{i + 1}. {s['description']}" for i, s in enumerate(stages))
        return score_plan_candidate(
            user_request=self.MULTI_OMICS_REQUEST,
            candidate_plan=plan_text,
            history_signals=self.COMMON_HISTORY,
            is_baseline=False,
            parsed_stages=stages,
        )

    def test_well_structured_beats_flat(self):
        good = self._score(self.WELL_STRUCTURED_STAGES)
        bad = self._score(self.FLAT_NO_STRUCTURE_STAGES)
        assert good["score"] > bad["score"], f"Structured {good['score']:.3f} should beat flat {bad['score']:.3f}"

    def test_well_structured_has_valid_dag(self):
        result = self._score(self.WELL_STRUCTURED_STAGES)
        assert result["structural_scores"]["dag_validity"] == 0.10

    def test_well_structured_has_high_dataflow_coverage(self):
        result = self._score(self.WELL_STRUCTURED_STAGES)
        assert result["structural_scores"]["dataflow_coverage"] > 0.07

    def test_flat_plan_zero_dataflow(self):
        result = self._score(self.FLAT_NO_STRUCTURE_STAGES)
        assert result["structural_scores"]["dataflow_coverage"] == 0.0

    def test_flat_plan_zero_dag(self):
        result = self._score(self.FLAT_NO_STRUCTURE_STAGES)
        assert result["structural_scores"]["dag_validity"] == 0.0

    def test_parallel_branches_valid(self):
        """s2 and s3 both depend on s1 independently — should still be valid DAG."""
        result = self._score(self.WELL_STRUCTURED_STAGES)
        assert result["structural_scores"]["dag_validity"] == 0.10

    def test_score_margin_meaningful(self):
        """The score difference should be large enough to reliably prefer the structured plan."""
        good = self._score(self.WELL_STRUCTURED_STAGES)
        bad = self._score(self.FLAT_NO_STRUCTURE_STAGES)
        margin = good["score"] - bad["score"]
        assert margin > 0.10, f"Margin {margin:.3f} should be > 0.10 for reliable differentiation"
