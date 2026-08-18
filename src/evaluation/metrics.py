"""Structured evaluation metrics."""

from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    """Standard metrics collected during evaluation and tournaments."""

    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    avg_score: float = 0.0
    avg_score_diff: float = 0.0
    avg_turns: float = 0.0
    ticket_completion_rate: float = 0.0
    custom_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0
