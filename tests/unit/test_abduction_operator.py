"""Unit tests for abductive hypothesis operator."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_data_scientist.agents.adk.operators.abduction import (
    build_abduction_prompt,
    generate_abduction_candidates,
    parse_abduction_method_card,
)


class TestBuildAbductionPrompt:
    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_replaces_all_placeholders(self, mock_load):
        mock_load.return_value = (
            "User: {original_user_input?}\n"
            "Unknowns: {unknowns?}\n"
            "Signals: {complexity_signals?}\n"
            "Existing: {existing_methods?}\n"
            "Round: {round_label?}"
        )
        result = build_abduction_prompt(
            unknowns=["optimal cluster count"],
            complexity_signals=["high dimensionality"],
            user_request="Cluster analysis",
            existing_summaries=["[m1] Baseline"],
            round_label="abd_1",
        )
        assert "Cluster analysis" in result
        assert "optimal cluster count" in result
        assert "high dimensionality" in result
        assert "[m1] Baseline" in result
        assert "abd_1" in result

    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_handles_empty_inputs(self, mock_load):
        mock_load.return_value = "Unknowns: {unknowns?}\nSignals: {complexity_signals?}\nExisting: {existing_methods?}"
        result = build_abduction_prompt(
            unknowns=[],
            complexity_signals=[],
            user_request="query",
            existing_summaries=[],
        )
        assert "[]" in result


class TestParseAbductionMethodCard:
    def test_parses_valid_json(self):
        data = {
            "method_id": "abd_1",
            "method_family": "abductive_hypothesis",
            "title": "Latent-subgroup hypothesis",
            "core_hypothesis": "Unknown clusters arise from latent disease subtypes",
        }
        result = parse_abduction_method_card(json.dumps(data))
        assert result is not None
        assert result["method_family"] == "abductive_hypothesis"

    def test_parses_json_with_markdown_fences(self):
        data = {"title": "Test", "core_hypothesis": "H"}
        text = f"```json\n{json.dumps(data)}\n```"
        result = parse_abduction_method_card(text)
        assert result is not None
        assert result["title"] == "Test"

    def test_parses_json_embedded_in_text(self):
        data = {"title": "Test", "core_hypothesis": "H"}
        text = f"Some preamble {json.dumps(data)} trailing"
        result = parse_abduction_method_card(text)
        assert result is not None

    def test_returns_none_for_invalid_json(self):
        assert parse_abduction_method_card("garbage") is None

    def test_returns_none_for_empty_string(self):
        assert parse_abduction_method_card("") is None

    def test_sets_defaults(self):
        data = {"title": "Test"}
        result = parse_abduction_method_card(json.dumps(data), round_label="abd_3")
        assert result is not None
        assert result["method_id"] == "abd_3"
        assert result["method_family"] == "abductive_hypothesis"

    def test_preserves_existing_fields(self):
        data = {"method_id": "custom", "method_family": "abductive_hypothesis"}
        result = parse_abduction_method_card(json.dumps(data))
        assert result["method_id"] == "custom"


class TestGenerateAbductionCandidates:
    def _make_mock_llm(self, response_text: str):
        mock_llm = MagicMock()

        async def fake_generate(llm_request, **kwargs):
            part = MagicMock()
            part.text = response_text
            content = MagicMock()
            content.parts = [part]
            resp = MagicMock()
            resp.content = content
            yield resp

        mock_llm.generate_content_async = fake_generate
        return mock_llm

    def test_returns_empty_when_no_unknowns_or_signals(self):
        mock_llm = self._make_mock_llm("")
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=[],
                complexity_signals=[],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert result == []

    def test_generates_card_from_unknowns_only(self):
        card_json = json.dumps(
            {
                "title": "Subtype Hypothesis",
                "core_hypothesis": "Unknown variance is due to disease subtypes",
                "assumptions": ["multiple subtypes exist"],
                "invalid_if": ["single homogeneous population"],
                "cheap_test": "Check bimodality of top PC",
                "failure_modes": ["insufficient sample size"],
                "required_capabilities": ["python", "clustering"],
                "expected_artifacts": ["subtype_clusters.csv"],
                "orthogonality_tags": ["abduction", "subtype", "clustering"],
                "competing_hypotheses": [
                    "Disease subtypes (chosen)",
                    "Batch effects (rejected — already corrected)",
                ],
                "unknown_addressed": "unexplained variance in gene expression",
            }
        )
        mock_llm = self._make_mock_llm(card_json)
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=["unexplained variance in gene expression"],
                complexity_signals=[],
                user_request="Analyze expression data",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert len(result) == 1
        assert result[0]["method_family"] == "abductive_hypothesis"
        assert result[0]["method_id"] == "abd_1"

    def test_generates_card_from_complexity_signals_only(self):
        card_json = json.dumps(
            {
                "title": "Dimensionality Hypothesis",
                "core_hypothesis": "High dimensionality masks true signal",
                "assumptions": [],
                "invalid_if": [],
                "cheap_test": "PCA variance explained",
                "failure_modes": [],
                "required_capabilities": [],
                "expected_artifacts": [],
                "orthogonality_tags": ["abduction"],
            }
        )
        mock_llm = self._make_mock_llm(card_json)
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=[],
                complexity_signals=["high dimensionality"],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert len(result) == 1

    def test_generates_multiple_cards(self):
        card_json = json.dumps(
            {
                "title": "Hypothesis",
                "core_hypothesis": "H",
                "assumptions": [],
                "invalid_if": [],
                "cheap_test": "t",
                "failure_modes": [],
                "required_capabilities": [],
                "expected_artifacts": [],
                "orthogonality_tags": ["abduction"],
            }
        )
        mock_llm = self._make_mock_llm(card_json)
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=["u1"],
                complexity_signals=["s1"],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
                max_cards=3,
            )
        )
        assert len(result) == 3
        assert result[0]["method_id"] == "abd_1"
        assert result[1]["method_id"] == "abd_2"
        assert result[2]["method_id"] == "abd_3"

    def test_handles_llm_failure(self):
        mock_llm = MagicMock()

        async def fail_generate(llm_request, **kwargs):
            raise RuntimeError("LLM down")
            yield  # noqa: unreachable

        mock_llm.generate_content_async = fail_generate
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=["u1"],
                complexity_signals=[],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert result == []

    def test_handles_parse_failure(self):
        mock_llm = self._make_mock_llm("not json")
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=["u1"],
                complexity_signals=[],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert result == []

    def test_card_family_always_abductive(self):
        card_json = json.dumps(
            {
                "method_family": "baseline",
                "title": "Test",
                "core_hypothesis": "H",
                "assumptions": [],
                "invalid_if": [],
                "cheap_test": "t",
                "failure_modes": [],
                "required_capabilities": [],
                "expected_artifacts": [],
                "orthogonality_tags": [],
            }
        )
        mock_llm = self._make_mock_llm(card_json)
        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=["u1"],
                complexity_signals=[],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert len(result) == 1
        assert result[0]["method_family"] == "abductive_hypothesis"

    def test_accumulates_summaries_across_rounds(self):
        call_count = 0

        async def counting_generate(llm_request, **kwargs):
            nonlocal call_count
            call_count += 1
            card = {
                "title": f"Method {call_count}",
                "core_hypothesis": f"H{call_count}",
                "assumptions": [],
                "invalid_if": [],
                "cheap_test": "t",
                "failure_modes": [],
                "required_capabilities": [],
                "expected_artifacts": [],
                "orthogonality_tags": ["abduction"],
            }
            part = MagicMock()
            part.text = json.dumps(card)
            content = MagicMock()
            content.parts = [part]
            resp = MagicMock()
            resp.content = content
            yield resp

        mock_llm = MagicMock()
        mock_llm.generate_content_async = counting_generate

        result = asyncio.run(
            generate_abduction_candidates(
                unknowns=["u1"],
                complexity_signals=[],
                user_request="query",
                existing_summaries=["[m1] Existing"],
                llm=mock_llm,
                max_cards=2,
            )
        )
        assert len(result) == 2
        assert call_count == 2
