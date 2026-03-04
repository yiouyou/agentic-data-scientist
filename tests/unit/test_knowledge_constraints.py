"""Unit tests for minimal stage knowledge-constraint normalization."""

from agentic_data_scientist.core.knowledge_constraints import normalize_and_validate_stage_constraints


def test_normalize_constraints_adds_default_stage_ids_and_dependencies():
    stages = [
        {"index": 0, "title": "Load data", "description": "stage 1"},
        {"index": 1, "title": "Train model", "description": "stage 2"},
    ]

    result = normalize_and_validate_stage_constraints(stages, apply_sequential_defaults=True)

    assert result.ok is True
    assert result.stages[0]["stage_id"] == "s1"
    assert result.stages[1]["stage_id"] == "s2"
    assert result.stages[0]["depends_on"] == []
    assert result.stages[1]["depends_on"] == ["s1"]
    assert result.stages[0]["inputs_required"] == []
    assert result.stages[0]["outputs_produced"] == []
    assert result.stages[0]["evidence_refs"] == []


def test_normalize_constraints_rejects_unknown_dependencies():
    stages = [
        {"index": 0, "title": "A", "description": "x", "stage_id": "sA"},
        {"index": 1, "title": "B", "description": "y", "depends_on": ["missing_stage"]},
    ]

    result = normalize_and_validate_stage_constraints(stages, apply_sequential_defaults=True)

    assert result.ok is False
    assert any("unknown stage" in err for err in result.errors)


def test_normalize_constraints_rejects_dependency_cycle():
    stages = [
        {"index": 0, "title": "A", "description": "x", "stage_id": "s1", "depends_on": ["s2"]},
        {"index": 1, "title": "B", "description": "y", "stage_id": "s2", "depends_on": ["s1"]},
    ]

    result = normalize_and_validate_stage_constraints(stages, apply_sequential_defaults=False)

    assert result.ok is False
    assert any("cycle" in err for err in result.errors)


def test_normalize_constraints_rejects_output_collisions():
    stages = [
        {
            "index": 0,
            "title": "A",
            "description": "x",
            "stage_id": "s1",
            "outputs_produced": ["artifacts/result.tsv"],
        },
        {
            "index": 1,
            "title": "B",
            "description": "y",
            "stage_id": "s2",
            "outputs_produced": ["artifacts/result.tsv"],
        },
    ]

    result = normalize_and_validate_stage_constraints(stages, apply_sequential_defaults=False)

    assert result.ok is False
    assert any("collision" in err for err in result.errors)

