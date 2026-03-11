"""Budget controller for Innovation OS discovery operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class InnovationBudget:
    method_generation: int = 2
    backtrack: int = 0
    decomposition: int = 0
    verification: int = 1

    _consumed: Dict[str, int] = field(default_factory=dict, repr=False)

    def remaining(self, dimension: str) -> int:
        limit = getattr(self, dimension, 0)
        used = self._consumed.get(dimension, 0)
        return max(0, limit - used)

    def consume(self, dimension: str, amount: int = 1) -> bool:
        if self.remaining(dimension) < amount:
            return False
        self._consumed[dimension] = self._consumed.get(dimension, 0) + amount
        return True

    def is_exhausted(self, dimension: str) -> bool:
        return self.remaining(dimension) <= 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_generation": self.method_generation,
            "backtrack": self.backtrack,
            "decomposition": self.decomposition,
            "verification": self.verification,
            "consumed": dict(self._consumed),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InnovationBudget:
        consumed = data.get("consumed", {})
        budget = cls(
            method_generation=int(data.get("method_generation", 2)),
            backtrack=int(data.get("backtrack", 0)),
            decomposition=int(data.get("decomposition", 0)),
            verification=int(data.get("verification", 1)),
        )
        budget._consumed = dict(consumed)
        return budget


BUDGET_PRESETS: Dict[str, InnovationBudget] = {
    "routine": InnovationBudget(method_generation=0, verification=1),
    "hybrid": InnovationBudget(method_generation=2, backtrack=1, verification=2),
    "innovation": InnovationBudget(method_generation=3, backtrack=1, decomposition=1, verification=3),
}


def get_budget_for_mode(mode: str) -> InnovationBudget:
    preset = BUDGET_PRESETS.get(mode)
    if preset is None:
        return InnovationBudget()
    return InnovationBudget(
        method_generation=preset.method_generation,
        backtrack=preset.backtrack,
        decomposition=preset.decomposition,
        verification=preset.verification,
    )
