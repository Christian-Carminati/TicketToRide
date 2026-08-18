"""Base Agent interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Abstract interface for all agents (heuristics, RL, tree search)."""

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Select an action given current observation and optional action mask."""
