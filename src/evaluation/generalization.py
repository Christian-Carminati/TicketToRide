"""Generalization benchmarks on unseen / procedural maps."""

from dataclasses import dataclass

from src.agents.base_agent import BaseAgent


@dataclass
class GeneralizationResult:
    train_map_score: float = 0.0
    unseen_map_score: float = 0.0
    generalization_gap: float = 0.0


class GeneralizationEvaluator:
    """Evaluates whether an agent generalizes across procedural maps (Phase 10)."""

    def __init__(self, test_seeds: list[int]) -> None:
        self.test_seeds = test_seeds

    def evaluate_generalization(self, agent: BaseAgent) -> GeneralizationResult:
        # Skeleton generalization test - implemented in Phase 10
        return GeneralizationResult()
