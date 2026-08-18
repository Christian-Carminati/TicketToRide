"""Round-robin and Swiss tournament orchestrator."""


from src.agents.base_agent import BaseAgent
from src.evaluation.elo import EloSystem
from src.evaluation.metrics import EvaluationMetrics


class Tournament:
    """Orchestrates deterministic tournaments among pools of agents."""

    def __init__(self, agents: list[BaseAgent], games_per_pair: int = 100) -> None:
        self.agents = agents
        self.games_per_pair = games_per_pair
        self.elo_system = EloSystem()

    def run(self, seed: int = 42) -> dict[str, EvaluationMetrics]:
        # Skeleton tournament runner - implemented in Phase 2
        results: dict[str, EvaluationMetrics] = {}
        for agent in self.agents:
            results[agent.name] = EvaluationMetrics()
        return results
