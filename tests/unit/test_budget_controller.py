"""Unit tests for budget_controller (core/budget_controller.py)."""

import pytest

from agentic_data_scientist.core.budget_controller import (
    BUDGET_PRESETS,
    InnovationBudget,
    get_budget_for_mode,
)


class TestInnovationBudget:
    def test_default_values(self):
        b = InnovationBudget()
        assert b.method_generation == 2
        assert b.backtrack == 0
        assert b.decomposition == 0
        assert b.verification == 1

    def test_remaining_initial(self):
        b = InnovationBudget(method_generation=3)
        assert b.remaining("method_generation") == 3
        assert b.remaining("backtrack") == 0

    def test_consume_success(self):
        b = InnovationBudget(method_generation=2)
        assert b.consume("method_generation") is True
        assert b.remaining("method_generation") == 1
        assert b.consume("method_generation") is True
        assert b.remaining("method_generation") == 0

    def test_consume_fails_when_exhausted(self):
        b = InnovationBudget(method_generation=1)
        assert b.consume("method_generation") is True
        assert b.consume("method_generation") is False
        assert b.remaining("method_generation") == 0

    def test_consume_amount_greater_than_one(self):
        b = InnovationBudget(method_generation=5)
        assert b.consume("method_generation", 3) is True
        assert b.remaining("method_generation") == 2
        assert b.consume("method_generation", 3) is False

    def test_is_exhausted(self):
        b = InnovationBudget(method_generation=1)
        assert b.is_exhausted("method_generation") is False
        b.consume("method_generation")
        assert b.is_exhausted("method_generation") is True

    def test_is_exhausted_zero_budget(self):
        b = InnovationBudget(backtrack=0)
        assert b.is_exhausted("backtrack") is True

    def test_unknown_dimension_returns_zero(self):
        b = InnovationBudget()
        assert b.remaining("nonexistent") == 0
        assert b.consume("nonexistent") is False

    def test_to_dict(self):
        b = InnovationBudget(method_generation=3, backtrack=1)
        b.consume("method_generation")
        d = b.to_dict()
        assert d["method_generation"] == 3
        assert d["backtrack"] == 1
        assert d["consumed"]["method_generation"] == 1

    def test_from_dict_roundtrip(self):
        original = InnovationBudget(method_generation=3, backtrack=2)
        original.consume("method_generation")
        original.consume("backtrack")

        restored = InnovationBudget.from_dict(original.to_dict())
        assert restored.method_generation == 3
        assert restored.backtrack == 2
        assert restored.remaining("method_generation") == 2
        assert restored.remaining("backtrack") == 1

    def test_from_dict_missing_fields(self):
        restored = InnovationBudget.from_dict({})
        assert restored.method_generation == 2
        assert restored.backtrack == 0


class TestBudgetPresets:
    def test_routine_preset(self):
        p = BUDGET_PRESETS["routine"]
        assert p.method_generation == 0
        assert p.backtrack == 0

    def test_hybrid_preset(self):
        p = BUDGET_PRESETS["hybrid"]
        assert p.method_generation == 2
        assert p.backtrack == 1

    def test_innovation_preset(self):
        p = BUDGET_PRESETS["innovation"]
        assert p.method_generation == 3
        assert p.decomposition == 1


class TestGetBudgetForMode:
    def test_known_modes(self):
        for mode in ("routine", "hybrid", "innovation"):
            b = get_budget_for_mode(mode)
            assert isinstance(b, InnovationBudget)

    def test_routine_returns_zero_method_generation(self):
        b = get_budget_for_mode("routine")
        assert b.method_generation == 0

    def test_unknown_mode_returns_default(self):
        b = get_budget_for_mode("unknown_mode")
        assert isinstance(b, InnovationBudget)
        assert b.method_generation == 2

    def test_returned_budget_is_independent_copy(self):
        b1 = get_budget_for_mode("hybrid")
        b2 = get_budget_for_mode("hybrid")
        b1.consume("method_generation")
        assert b2.remaining("method_generation") == 2
