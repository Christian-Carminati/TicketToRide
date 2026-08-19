"""Rollout buffer for on-policy PPO trajectories with Action Masking."""

from collections.abc import Iterator
import numpy as np
import torch


class RolloutBuffer:
    """On-policy trajectory storage with optimized mini-batch generation."""

    def __init__(self, capacity: int = 2048, obs_dim: int = 0, action_dim: int = 0) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self.initialized = False

        self.obs_buf: np.ndarray | None = None
        self.actions_buf = np.zeros(capacity, dtype=np.int64)
        self.rewards_buf = np.zeros(capacity, dtype=np.float32)
        self.values_buf = np.zeros(capacity, dtype=np.float32)
        self.log_probs_buf = np.zeros(capacity, dtype=np.float32)
        self.dones_buf = np.zeros(capacity, dtype=bool)
        self.masks_buf: np.ndarray | None = None

        if obs_dim > 0 and action_dim > 0:
            self._lazy_init(obs_dim, action_dim)

    def _lazy_init(self, obs_dim: int, action_dim: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.masks_buf = np.zeros((self.capacity, action_dim), dtype=bool)
        self.initialized = True

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
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        mask_arr = np.asarray(action_mask, dtype=bool).ravel()

        if not self.initialized or self.obs_buf is None or self.masks_buf is None:
            self._lazy_init(len(obs_arr), len(mask_arr))

        if self.ptr >= self.capacity:
            # Expand capacity if needed
            new_cap = self.capacity * 2
            new_obs = np.zeros((new_cap, self.obs_dim), dtype=np.float32)
            new_masks = np.zeros((new_cap, self.action_dim), dtype=bool)
            new_obs[:self.capacity] = self.obs_buf
            new_masks[:self.capacity] = self.masks_buf
            self.obs_buf = new_obs
            self.masks_buf = new_masks

            self.actions_buf = np.pad(self.actions_buf, (0, self.capacity))
            self.rewards_buf = np.pad(self.rewards_buf, (0, self.capacity))
            self.values_buf = np.pad(self.values_buf, (0, self.capacity))
            self.log_probs_buf = np.pad(self.log_probs_buf, (0, self.capacity))
            self.dones_buf = np.pad(self.dones_buf, (0, self.capacity))
            self.capacity = new_cap

        self.obs_buf[self.ptr] = obs_arr
        self.actions_buf[self.ptr] = int(action)
        self.rewards_buf[self.ptr] = float(reward)
        self.values_buf[self.ptr] = float(value)
        self.log_probs_buf[self.ptr] = float(log_prob)
        self.dones_buf[self.ptr] = bool(done)
        self.masks_buf[self.ptr] = mask_arr

        self.ptr += 1
        self.size = self.ptr

    @property
    def observations(self) -> list[np.ndarray]:
        if self.obs_buf is None:
            return []
        return [self.obs_buf[i] for i in range(self.size)]

    @property
    def actions(self) -> list[int]:
        return list(self.actions_buf[:self.size])

    @property
    def rewards(self) -> list[float]:
        return list(self.rewards_buf[:self.size])

    @property
    def values(self) -> list[float]:
        return list(self.values_buf[:self.size])

    @property
    def log_probs(self) -> list[float]:
        return list(self.log_probs_buf[:self.size])

    @property
    def dones(self) -> list[bool]:
        return list(self.dones_buf[:self.size])

    @property
    def action_masks(self) -> list[np.ndarray]:
        if self.masks_buf is None:
            return []
        return [self.masks_buf[i] for i in range(self.size)]

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    def generate_minibatches(
        self,
        batch_size: int,
        advantages: np.ndarray,
        returns: np.ndarray,
        device: str = "cpu",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield randomized minibatches for multi-epoch PPO optimization."""
        total_steps = self.size
        if total_steps == 0 or self.obs_buf is None or self.masks_buf is None:
            return

        indices = np.random.permutation(total_steps)

        # Convert full rollout once for all minibatches in this epoch
        t_obs = torch.as_tensor(self.obs_buf[:total_steps], dtype=torch.float32, device=device)
        t_act = torch.as_tensor(self.actions_buf[:total_steps], dtype=torch.int64, device=device)
        t_lp = torch.as_tensor(self.log_probs_buf[:total_steps], dtype=torch.float32, device=device)
        t_adv = torch.as_tensor(advantages[:total_steps], dtype=torch.float32, device=device)
        t_ret = torch.as_tensor(returns[:total_steps], dtype=torch.float32, device=device)
        t_mask = torch.as_tensor(self.masks_buf[:total_steps], dtype=torch.bool, device=device)

        for start in range(0, total_steps, batch_size):
            mb_idx = indices[start : start + batch_size]
            yield {
                "obs": t_obs[mb_idx],
                "actions": t_act[mb_idx],
                "old_log_probs": t_lp[mb_idx],
                "advantages": t_adv[mb_idx],
                "returns": t_ret[mb_idx],
                "action_masks": t_mask[mb_idx],
            }

    def __len__(self) -> int:
        return self.size
