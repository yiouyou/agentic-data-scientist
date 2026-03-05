"""Unit tests for skill registry discovery and top-k retrieval."""

from __future__ import annotations

from pathlib import Path

import agentic_data_scientist.core.skill_registry as skill_registry
from agentic_data_scientist.core.skill_registry import discover_skills, recommend_skills


def _make_skill(root: Path, name: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def test_discover_skills_from_local_source(monkeypatch):
    root = Path(".tmp") / "unit_skill_registry" / "discover"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    _make_skill(source, "rna-seq-analysis", "# RNA-seq\nDifferential expression workflow.")
    _make_skill(source, "citation-management", "# Citations\nManage and validate references.")

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    cards = discover_skills(working_dir=str(root))
    names = {card.skill_name for card in cards}
    assert "rna-seq-analysis" in names
    assert "citation-management" in names


def test_recommend_skills_top_k(monkeypatch):
    root = Path(".tmp") / "unit_skill_registry" / "recommend"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    _make_skill(
        source,
        "rna-seq-analysis",
        "# RNA-seq Analysis\nAnalyze RNA sequencing differential expression and QC.",
    )
    _make_skill(
        source,
        "financial-forecasting",
        "# Financial Forecasting\nForecast macro indicators and market time series.",
    )

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    results = recommend_skills(
        query="Need RNA sequencing differential expression analysis",
        working_dir=str(root),
        top_k=1,
        min_score=0.12,
    )
    assert len(results) == 1
    assert results[0]["skill_name"] == "rna-seq-analysis"


def test_recommend_skills_handles_rnaseq_alias(monkeypatch):
    root = Path(".tmp") / "unit_skill_registry" / "alias"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    _make_skill(
        source,
        "rna-seq-analysis",
        "# RNA-seq Analysis\nDifferential expression and sequencing QC.",
    )
    _make_skill(
        source,
        "general-writing",
        "# Writing\nGeneral scientific writing support.",
    )

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    results = recommend_skills(
        query="Run rnaseq DEG pipeline",
        working_dir=str(root),
        top_k=1,
        min_score=0.05,
    )
    assert results
    assert results[0]["skill_name"] == "rna-seq-analysis"


def test_recommend_skills_with_mmr_diversifies(monkeypatch):
    root = Path(".tmp") / "unit_skill_registry" / "mmr"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    _make_skill(source, "unit-alpha-a", "# Unit Alpha A\nunitalpha999 unitdelta999 analysis pipeline")
    _make_skill(source, "unit-alpha-b", "# Unit Alpha B\nunitalpha999 unitepsilon999 analysis workflow")
    _make_skill(source, "unit-zeta", "# Unit Zeta\nunitzeta999 report synthesis")

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    monkeypatch.setenv("ADS_SKILL_MMR_ENABLED", "false")
    monkeypatch.setenv("ADS_SKILL_CANDIDATE_POOL", "6")
    query = "Need unitalpha999 and unitzeta999 outputs"

    no_mmr_results = recommend_skills(
        query=query,
        working_dir=str(root),
        top_k=2,
        min_score=0.01,
    )
    no_mmr_names = [item["skill_name"] for item in no_mmr_results]

    monkeypatch.setenv("ADS_SKILL_MMR_ENABLED", "true")
    monkeypatch.setenv("ADS_SKILL_MMR_LAMBDA", "0.2")
    mmr_results = recommend_skills(
        query=query,
        working_dir=str(root),
        top_k=2,
        min_score=0.01,
    )
    mmr_names = [item["skill_name"] for item in mmr_results]

    assert len(no_mmr_results) == 2
    assert len(mmr_results) == 2
    assert no_mmr_names != mmr_names
    assert "unit-zeta" in mmr_names
    assert all("mmr_score" in item for item in mmr_results)


def test_recommend_skills_prefers_external_embedding_signal(monkeypatch):
    root = Path(".tmp") / "unit_skill_registry" / "external"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    _make_skill(source, "unit-a", "# Unit A\nunitembed111 common tokens")
    _make_skill(source, "unit-b", "# Unit B\nunitembed111 common tokens")

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    monkeypatch.setenv("ADS_EMBEDDING_ENABLED", "true")

    def fake_external(*, texts, task, provider=None):
        if task == "query":
            return "gemini", [[1.0, 0.0]]
        if task == "document":
            vectors = []
            for idx, _ in enumerate(texts):
                vectors.append([1.0, 0.0] if idx == 0 else [0.0, 1.0])
            return "gemini", vectors
        return None

    monkeypatch.setattr(skill_registry, "_get_external_embeddings", fake_external)
    results = recommend_skills(
        query="unitembed111",
        working_dir=str(root),
        top_k=1,
        min_score=0.01,
    )
    assert results
    assert results[0]["skill_name"] == "unit-a"
    assert results[0]["embedding_provider"] == "gemini"


def test_recommend_skills_falls_back_to_local_when_external_unavailable(monkeypatch):
    root = Path(".tmp") / "unit_skill_registry" / "external_fallback"
    if root.exists():
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    source = root / "scientific-skills"
    source.mkdir(parents=True, exist_ok=True)
    _make_skill(source, "unit-local", "# Unit Local\nunitlocal777 analysis")

    monkeypatch.setenv("ADS_LOCAL_SKILLS_SOURCE", str(source))
    monkeypatch.setenv("ADS_EMBEDDING_ENABLED", "true")
    monkeypatch.setattr(skill_registry, "_get_external_embeddings", lambda **kwargs: None)

    results = recommend_skills(
        query="unitlocal777 analysis",
        working_dir=str(root),
        top_k=1,
        min_score=0.01,
    )
    assert results
    assert results[0]["skill_name"] == "unit-local"
    assert results[0]["embedding_provider"] == "local"
