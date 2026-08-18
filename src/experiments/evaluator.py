"""Multi-Opponent Evaluator for RL agents during and after training."""

from collections.abc import Sequence
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.evaluation.evaluator import Evaluator
from src.game.board import Board
from src.game.ticket import DestinationTicket


class MultiOpponentEvaluator:
    """Evaluates an agent against multiple baseline opponents."""

    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        seed: int = 42,
    ) -> None:
        self.board = board
        self.tickets_deck = tickets_deck
        self.seed = seed
        self.evaluator = Evaluator(board=self.board, tickets_deck=self.tickets_deck, seed=self.seed)

    def _get_opponent(self, name: str) -> BaseAgent:
        name_lower = name.lower()
        if name_lower == "random":
            return RandomAgent(name="RandomOpponent")
        if name_lower == "greedy":
            return GreedyAgent(name="GreedyOpponent")
        if name_lower == "strategic":
            return StrategicAgent(name="StrategicOpponent")
        raise ValueError(f"Unknown opponent type: {name}")

    def evaluate(
        self,
        agent: BaseAgent,
        opponents: Sequence[str] = ("random", "greedy", "strategic"),
        games_per_opponent: int = 20,
    ) -> dict[str, float]:
        """Run round-robin evaluation against multiple baseline opponents."""
        metrics: dict[str, float] = {}
        for opp_name in opponents:
            opp_agent = self._get_opponent(opp_name)
            result = self.evaluator.evaluate_head_to_head(
                agent_a=agent,
                agent_b=opp_agent,
                num_games=games_per_opponent,
            )
            metrics[f"win_rate_vs_{opp_name}"] = result["agent_a_win_rate"]
            metrics[f"score_diff_vs_{opp_name}"] = (
                result["agent_a_mean_score"] - result["agent_b_mean_score"]
            )
            metrics[f"mean_score_vs_{opp_name}"] = result["agent_a_mean_score"]
        return metrics
