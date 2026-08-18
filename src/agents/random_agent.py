"""Random baseline agent."""

import random
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.game.action import Action
from src.game.board import Board
from src.game.state import GameState


class RandomAgent(BaseAgent):
    """Uniformly samples from valid actions using seeded RNG."""

    def __init__(self, seed: int = 42, name: str = "RandomAgent") -> None:
        super().__init__(name=name)
        self._initial_seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        new_seed = seed if seed is not None else self._initial_seed
        self.rng = random.Random(new_seed)

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("Cannot select an action from an empty valid_actions list.")
        return self.rng.choice(valid_actions)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(self.rng.choice(valid_indices))
        return 0
