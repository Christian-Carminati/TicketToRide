"""Random baseline agent."""

import random
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Uniformly samples from valid actions using action mask."""

    def __init__(self, seed: int = 42, name: str = "RandomAgent") -> None:
        super().__init__(name=name)
        self.rng = random.Random(seed)

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
