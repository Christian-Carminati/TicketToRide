"""Rollout buffer for on-policy PPO trajectories with Action Masking."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """On-policy trajectory storage with mini-batch generation."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.action_masks.append(np.asarray(action_mask, dtype=bool))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.action_masks.clear()

    def generate_minibatches(
        self,
        batch_size: int,
        advantages: np.ndarray,
        returns: np.ndarray,
        device: str = "cpu",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield randomized minibatches for multi-epoch PPO optimization."""
        total_steps = len(self.observations)
        indices = np.random.permutation(total_steps)

        obs_arr = np.array(self.observations, dtype=np.float32)
        act_arr = np.array(self.actions, dtype=np.int64)
        lp_arr = np.array(self.log_probs, dtype=np.float32)
        mask_arr = np.array(self.action_masks, dtype=bool)

        for start in range(0, total_steps, batch_size):
            mb_indices = indices[start : start + batch_size]
            yield {
                "obs": torch.tensor(obs_arr[mb_indices], dtype=torch.float32, device=device),
                "actions": torch.tensor(act_arr[mb_indices], dtype=torch.int64, device=device),
                "old_log_probs": torch.tensor(lp_arr[mb_indices], dtype=torch.float32, device=device),
                "advantages": torch.tensor(advantages[mb_indices], dtype=torch.float32, device=device),
                "returns": torch.tensor(returns[mb_indices], dtype=torch.float32, device=device),
                "action_masks": torch.tensor(mask_arr[mb_indices], dtype=torch.bool, device=device),
            }

    def __len__(self) -> int:
        return len(self.observations)
