"""Base Agent interface supporting both native Game Core and Gymnasium."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.game.action import Action
from src.game.board import Board
from src.game.state import GameState


class BaseAgent(ABC):
    """Abstract interface for all agents (heuristics, RL, tree search)."""

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

    @abstractmethod
    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        """Select a domain Action given current game state and legal actions."""
        pass

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Select an action index for Gymnasium environments."""
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(valid_indices[0])
        return 0

    def reset(self, seed: int | None = None) -> None:
        """Reset internal agent state or random number generator."""
        pass
