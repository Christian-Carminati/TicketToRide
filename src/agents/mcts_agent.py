"""MCTS Agent interface."""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent


class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search agent."""

    def __init__(self, name: str = "MCTSAgent", num_simulations: int = 100) -> None:
        super().__init__(name=name)
        self.num_simulations = num_simulations

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(valid_indices[0])
        return 0
