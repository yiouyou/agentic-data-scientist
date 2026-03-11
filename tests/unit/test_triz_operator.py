"""Unit tests for TRIZ contradiction-resolution operator."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_data_scientist.agents.adk.operators.triz import (
    TRIZ_PRINCIPLES,
    build_triz_prompt,
    generate_triz_candidates,
    parse_triz_method_card,
    select_relevant_principles,
)


class TestTrizPrinciples:
    def test_principles_count(self):
        assert len(TRIZ_PRINCIPLES) == 6

    def test_principles_have_required_fields(self):
        for p in TRIZ_PRINCIPLES:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert isinstance(p["id"], str)
            assert isinstance(p["name"], str)
            assert len(p["description"]) > 10

    def test_principle_ids_unique(self):
        ids = [p["id"] for p in TRIZ_PRINCIPLES]
        assert len(ids) == len(set(ids))


class TestSelectRelevantPrinciples:
    def test_returns_max_principles(self):
        result = select_relevant_principles(["contradiction A"], max_principles=2)
        assert len(result) == 2

    def test_returns_all_when_max_exceeds(self):
        result = select_relevant_principles(["c1"], max_principles=100)
        assert len(result) == len(TRIZ_PRINCIPLES)

    def test_returns_all_when_max_equals(self):
        result = select_relevant_principles(["c1"], max_principles=len(TRIZ_PRINCIPLES))
        assert len(result) == len(TRIZ_PRINCIPLES)

    def test_default_max_is_three(self):
        result = select_relevant_principles(["c1"])
        assert len(result) == 3

    def test_empty_contradictions_still_works(self):
        result = select_relevant_principles([])
        assert len(result) == 3


class TestBuildTrizPrompt:
    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_replaces_all_placeholders(self, mock_load):
        mock_load.return_value = (
            "User: {original_user_input?}\n"
            "Contradictions: {contradictions?}\n"
            "Existing: {existing_methods?}\n"
            "Principles: {triz_principles?}\n"
            "Round: {round_label?}"
        )
        result = build_triz_prompt(
            contradictions=["speed vs accuracy"],
            user_request="Analyze data",
            existing_summaries=["[m1] Baseline"],
            principles=[{"id": "separation", "name": "Separation", "description": "desc"}],
            round_label="triz_1",
        )
        assert "Analyze data" in result
        assert "speed vs accuracy" in result
        assert "[m1] Baseline" in result
        assert "separation" in result
        assert "triz_1" in result
        assert "{" not in result or "}" not in result.replace("{", "").replace("}", "")

    @patch("agentic_data_scientist.prompts.load_prompt")
    def test_handles_empty_existing_methods(self, mock_load):
        mock_load.return_value = "Existing: {existing_methods?}"
        result = build_triz_prompt(
            contradictions=["c1"],
            user_request="query",
            existing_summaries=[],
            principles=[],
            round_label="triz_1",
        )
        assert "[]" in result


class TestParseTrizMethodCard:
    def test_parses_valid_json(self):
        data = {
            "method_id": "triz_1",
            "method_family": "triz_resolution",
            "title": "Separation-based DEG",
            "core_hypothesis": "Splitting by time resolves the accuracy-speed tradeoff",
        }
        result = parse_triz_method_card(json.dumps(data))
        assert result is not None
        assert result["method_family"] == "triz_resolution"
        assert result["title"] == "Separation-based DEG"

    def test_parses_json_with_markdown_fences(self):
        data = {"method_id": "triz_1", "title": "Test", "core_hypothesis": "H"}
        text = f"```json\n{json.dumps(data)}\n```"
        result = parse_triz_method_card(text)
        assert result is not None
        assert result["title"] == "Test"

    def test_parses_json_embedded_in_text(self):
        data = {"method_id": "triz_1", "title": "Test", "core_hypothesis": "H"}
        text = f"Here is the method card: {json.dumps(data)} -- end"
        result = parse_triz_method_card(text)
        assert result is not None
        assert result["title"] == "Test"

    def test_returns_none_for_invalid_json(self):
        assert parse_triz_method_card("not json at all") is None

    def test_returns_none_for_empty_string(self):
        assert parse_triz_method_card("") is None

    def test_sets_defaults(self):
        data = {"title": "Test", "core_hypothesis": "H"}
        result = parse_triz_method_card(json.dumps(data), round_label="triz_2")
        assert result is not None
        assert result["method_id"] == "triz_2"
        assert result["method_family"] == "triz_resolution"

    def test_preserves_existing_method_id(self):
        data = {"method_id": "custom_id", "title": "Test"}
        result = parse_triz_method_card(json.dumps(data))
        assert result["method_id"] == "custom_id"


class TestGenerateTrizCandidates:
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

    def test_returns_empty_when_no_contradictions(self):
        mock_llm = self._make_mock_llm("")
        result = asyncio.run(
            generate_triz_candidates(
                contradictions=[],
                user_request="Analyze data",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert result == []

    def test_generates_one_card(self):
        card_json = json.dumps(
            {
                "method_id": "triz_1",
                "method_family": "triz_resolution",
                "title": "Separation DEG",
                "core_hypothesis": "Temporal separation resolves batch effects",
                "assumptions": ["time-dependent signal"],
                "invalid_if": ["no temporal structure"],
                "cheap_test": "Check temporal autocorrelation",
                "failure_modes": ["insufficient time points"],
                "required_capabilities": ["python"],
                "expected_artifacts": ["temporal_deg.csv"],
                "orthogonality_tags": ["triz", "separation", "temporal"],
                "triz_principle_used": "separation",
                "contradiction_addressed": "batch effects vs signal preservation",
            }
        )
        mock_llm = self._make_mock_llm(card_json)
        result = asyncio.run(
            generate_triz_candidates(
                contradictions=["batch effects vs signal preservation"],
                user_request="Analyze gene expression",
                existing_summaries=[],
                llm=mock_llm,
                max_cards=1,
            )
        )
        assert len(result) == 1
        assert result[0]["method_family"] == "triz_resolution"
        assert result[0]["method_id"] == "triz_1"

    def test_generates_multiple_cards(self):
        card_json = json.dumps(
            {
                "title": "TRIZ Method",
                "core_hypothesis": "Hypothesis",
                "assumptions": [],
                "invalid_if": [],
                "cheap_test": "test",
                "failure_modes": [],
                "required_capabilities": [],
                "expected_artifacts": [],
                "orthogonality_tags": ["triz"],
            }
        )
        mock_llm = self._make_mock_llm(card_json)
        result = asyncio.run(
            generate_triz_candidates(
                contradictions=["c1", "c2"],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
                max_cards=2,
            )
        )
        assert len(result) == 2
        assert result[0]["method_id"] == "triz_1"
        assert result[1]["method_id"] == "triz_2"

    def test_handles_llm_failure(self):
        mock_llm = MagicMock()

        async def fail_generate(llm_request, **kwargs):
            raise RuntimeError("LLM unavailable")
            yield  # noqa: unreachable - makes this an async generator

        mock_llm.generate_content_async = fail_generate
        result = asyncio.run(
            generate_triz_candidates(
                contradictions=["c1"],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert result == []

    def test_handles_parse_failure(self):
        mock_llm = self._make_mock_llm("not valid json response")
        result = asyncio.run(
            generate_triz_candidates(
                contradictions=["c1"],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert result == []

    def test_card_family_always_triz_resolution(self):
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
            generate_triz_candidates(
                contradictions=["c1"],
                user_request="query",
                existing_summaries=[],
                llm=mock_llm,
            )
        )
        assert len(result) == 1
        assert result[0]["method_family"] == "triz_resolution"


class TestMethodCardFamilyExpansion:
    def test_triz_resolution_is_valid_family(self):
        from agentic_data_scientist.core.method_card import _VALID_FAMILIES

        assert "triz_resolution" in _VALID_FAMILIES

    def test_abductive_hypothesis_is_valid_family(self):
        from agentic_data_scientist.core.method_card import _VALID_FAMILIES

        assert "abductive_hypothesis" in _VALID_FAMILIES

    def test_make_card_with_triz_family(self):
        from agentic_data_scientist.core.method_card import make_method_card

        card = make_method_card(
            method_id="triz_1",
            method_family="triz_resolution",
            title="TRIZ Method",
            core_hypothesis="Hypothesis",
            assumptions=[],
            invalid_if=[],
            cheap_test="test",
            failure_modes=[],
            required_capabilities=[],
            expected_artifacts=[],
            orthogonality_tags=["triz"],
        )
        assert card["method_family"] == "triz_resolution"

    def test_validate_card_accepts_triz_family(self):
        from agentic_data_scientist.core.method_card import validate_method_card

        card = {
            "method_id": "triz_1",
            "method_family": "triz_resolution",
            "title": "Test",
            "core_hypothesis": "H",
            "cheap_test": "t",
            "assumptions": [],
            "invalid_if": [],
            "failure_modes": [],
            "required_capabilities": [],
            "expected_artifacts": [],
            "orthogonality_tags": [],
        }
        errors = validate_method_card(card)
        assert not errors

    def test_validate_card_accepts_abductive_family(self):
        from agentic_data_scientist.core.method_card import validate_method_card

        card = {
            "method_id": "abd_1",
            "method_family": "abductive_hypothesis",
            "title": "Test",
            "core_hypothesis": "H",
            "cheap_test": "t",
            "assumptions": [],
            "invalid_if": [],
            "failure_modes": [],
            "required_capabilities": [],
            "expected_artifacts": [],
            "orthogonality_tags": [],
        }
        errors = validate_method_card(card)
        assert not errors
