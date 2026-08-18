"""Rollout buffer for on-policy algorithms (PPO)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    """On-policy trajectory storage for PPO rollouts."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.observations)
