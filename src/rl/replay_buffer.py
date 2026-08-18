"""Experience Replay Buffer for DQN with Action Masking."""

import random
from collections import deque
from typing import Any

import numpy as np
import torch


class ReplayBatch:
    """Batch container supporting both dictionary key access and tuple unpacking."""

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        next_action_masks: torch.Tensor,
    ) -> None:
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.next_obs = next_obs
        self.dones = dones
        self.next_action_masks = next_action_masks

    def __getitem__(self, item: str) -> torch.Tensor:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"ReplayBatch has no key '{item}'")

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def __iter__(self):
        # Enables tuple unpacking (obs, actions, rewards, next_obs, dones)
        yield self.obs.cpu().numpy()
        yield self.actions.cpu().numpy()
        yield self.rewards.cpu().numpy()
        yield self.next_obs.cpu().numpy()
        yield self.dones.cpu().numpy()


class ReplayBuffer:
    """Fixed-capacity experience replay buffer storing transitions with next action masks."""

    def __init__(self, capacity: int = 100000) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        if next_action_mask is None:
            # Default empty mask or single element
            next_action_mask = np.ones(1, dtype=np.int8)
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
                np.asarray(next_action_mask, dtype=bool),
            )
        )

    def sample(self, batch_size: int, device: str = "cpu") -> ReplayBatch:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)

        return ReplayBatch(
            obs=torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            actions=torch.tensor(np.array(actions), dtype=torch.int64, device=device),
            rewards=torch.tensor(np.array(rewards), dtype=torch.float32, device=device),
            next_obs=torch.tensor(np.array(next_obs), dtype=torch.float32, device=device),
            dones=torch.tensor(np.array(dones), dtype=torch.float32, device=device),
            next_action_masks=torch.tensor(np.array(next_masks), dtype=torch.bool, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)
