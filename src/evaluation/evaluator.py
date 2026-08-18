"""Agent Evaluation framework."""


from src.agents.base_agent import BaseAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.metrics import EvaluationMetrics


class Evaluator:
    """Evaluates an agent against baseline opponents across N episodes."""

    def __init__(self, env: TicketToRideEnv | None = None) -> None:
        self.env = env or TicketToRideEnv()

    def evaluate(
        self,
        agent: BaseAgent,
        opponent: BaseAgent,
        num_episodes: int = 100,
        seed: int = 42,
    ) -> EvaluationMetrics:
        # Skeleton evaluation loop - implemented in Phase 2
        return EvaluationMetrics(total_games=num_episodes)
