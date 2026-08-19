"""Experience Replay Buffer for DQN with Action Masking and Vectorized Storage."""

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
    """Fixed-capacity vectorized experience replay buffer with circular pre-allocation."""

    def __init__(self, capacity: int = 100000, obs_dim: int = 0, action_dim: int = 0) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self.initialized = False

        self.obs_buf: np.ndarray | None = None
        self.next_obs_buf: np.ndarray | None = None
        self.actions_buf = np.zeros(capacity, dtype=np.int64)
        self.rewards_buf = np.zeros(capacity, dtype=np.float32)
        self.dones_buf = np.zeros(capacity, dtype=np.float32)
        self.masks_buf: np.ndarray | None = None

        if obs_dim > 0 and action_dim > 0:
            self._lazy_init(obs_dim, action_dim)

    def _lazy_init(self, obs_dim: int, action_mask_dim: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_mask_dim
        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.masks_buf = np.zeros((self.capacity, action_mask_dim), dtype=bool)
        self.initialized = True

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).ravel()
        mask_arr = np.asarray(next_action_mask if next_action_mask is not None else [True], dtype=bool).ravel()

        if not self.initialized or self.obs_buf is None or self.masks_buf is None:
            self._lazy_init(len(obs_arr), len(mask_arr))

        self.obs_buf[self.ptr] = obs_arr
        self.actions_buf[self.ptr] = int(action)
        self.rewards_buf[self.ptr] = float(reward)
        self.next_obs_buf[self.ptr] = next_obs_arr
        self.dones_buf[self.ptr] = float(done)
        self.masks_buf[self.ptr] = mask_arr

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> ReplayBatch:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions_buf[idxs], dtype=torch.int64, device=device),
            rewards=torch.as_tensor(self.rewards_buf[idxs], dtype=torch.float32, device=device),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones_buf[idxs], dtype=torch.float32, device=device),
            next_action_masks=torch.as_tensor(self.masks_buf[idxs], dtype=torch.bool, device=device),
        )

    def __len__(self) -> int:
        return self.size
