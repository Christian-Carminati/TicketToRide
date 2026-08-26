"""Mixed / Random Pool Opponent Agent for training against diverse strategies."""

from typing import Any
import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.game.action import Action
from src.game.board import Board
from src.game.state import GameState


class MixedOpponentAgent(BaseAgent):
    """Opponent agent that randomly switches between baseline agents on each reset.
    
    Prevents RL policies from overfitting to a single deterministic baseline.
    """

    def __init__(self, name: str = "MixedPoolBot", seed: int | None = None) -> None:
        super().__init__(name=name)
        self._rng = np.random.default_rng(seed)
        self.agents: list[BaseAgent] = [
            GreedyAgent(name="GreedyBot"),
            StrategicAgent(name="StrategicBot"),
            RandomAgent(name="RandomBot", seed=seed if seed is not None else 42),
        ]
        self.active_agent: BaseAgent = self.agents[0]
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Pick a new opponent policy from the pool for the upcoming episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        choice_idx = int(self._rng.integers(0, len(self.agents)))
        self.active_agent = self.agents[choice_idx]
        self.name = f"MixedBot ({self.active_agent.name})"
        if hasattr(self.active_agent, "reset"):
            self.active_agent.reset(seed=seed)

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        return self.active_agent.act(state, valid_actions, board=board)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool = True,
        info: dict[str, Any] | None = None,
    ) -> int:
        return self.active_agent.select_action(
            observation, action_mask=action_mask, deterministic=deterministic, info=info
        )
