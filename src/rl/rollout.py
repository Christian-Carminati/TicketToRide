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
        if not self.initialized or self.obs_buf is None or self.masks_buf is None:
            self._lazy_init(len(obs), len(action_mask))

        assert self.obs_buf is not None and self.masks_buf is not None

        if self.ptr >= self.capacity:
            new_cap = self.capacity * 2
            new_obs = np.zeros((new_cap, self.obs_dim), dtype=np.float32)
            new_masks = np.zeros((new_cap, self.action_dim), dtype=bool)
            new_obs[: self.capacity] = self.obs_buf
            new_masks[: self.capacity] = self.masks_buf
            self.obs_buf = new_obs
            self.masks_buf = new_masks

            self.actions_buf = np.pad(self.actions_buf, (0, self.capacity))
            self.rewards_buf = np.pad(self.rewards_buf, (0, self.capacity))
            self.values_buf = np.pad(self.values_buf, (0, self.capacity))
            self.log_probs_buf = np.pad(self.log_probs_buf, (0, self.capacity))
            self.dones_buf = np.pad(self.dones_buf, (0, self.capacity))
            self.capacity = new_cap

        self.obs_buf[self.ptr] = obs
        self.actions_buf[self.ptr] = int(action)
        self.rewards_buf[self.ptr] = float(reward)
        self.values_buf[self.ptr] = float(value)
        self.log_probs_buf[self.ptr] = float(log_prob)
        self.dones_buf[self.ptr] = bool(done)
        self.masks_buf[self.ptr] = action_mask

        self.ptr += 1
        self.size = self.ptr

    @property
    def observations(self) -> list[np.ndarray]:
        if self.obs_buf is None:
            return []
        return [self.obs_buf[i] for i in range(self.size)]

    @property
    def actions(self) -> list[int]:
        return list(self.actions_buf[: self.size])

    @property
    def rewards(self) -> list[float]:
        return list(self.rewards_buf[: self.size])

    @property
    def values(self) -> list[float]:
        return list(self.values_buf[: self.size])

    @property
    def log_probs(self) -> list[float]:
        return list(self.log_probs_buf[: self.size])

    @property
    def dones(self) -> list[bool]:
        return list(self.dones_buf[: self.size])

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
        t_obs = torch.from_numpy(self.obs_buf[:total_steps]).to(device=device)
        t_act = torch.from_numpy(self.actions_buf[:total_steps]).to(device=device)
        t_lp = torch.from_numpy(self.log_probs_buf[:total_steps]).to(device=device)
        t_val = torch.from_numpy(self.values_buf[:total_steps]).to(device=device)
        t_adv = torch.from_numpy(advantages[:total_steps]).to(device=device)
        t_ret = torch.from_numpy(returns[:total_steps]).to(device=device)
        t_mask = torch.from_numpy(self.masks_buf[:total_steps]).to(device=device)

        for start in range(0, total_steps, batch_size):
            mb_idx = indices[start : start + batch_size]
            yield {
                "obs": t_obs[mb_idx],
                "actions": t_act[mb_idx],
                "old_log_probs": t_lp[mb_idx],
                "values": t_val[mb_idx],
                "advantages": t_adv[mb_idx],
                "returns": t_ret[mb_idx],
                "action_masks": t_mask[mb_idx],
            }

    def __len__(self) -> int:
        return self.size


class VectorRolloutBuffer:
    """Multidimensional On-policy trajectory storage for single or vectorized environments."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs_buf = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.rewards_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones_buf = np.zeros((num_steps, num_envs), dtype=bool)
        self.masks_buf = np.zeros((num_steps, num_envs, action_dim), dtype=bool)

        self.step = 0

    def clear(self) -> None:
        self.step = 0

    def is_full(self) -> bool:
        return self.step >= self.num_steps

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | int | list[int],
        reward: np.ndarray | float | list[float],
        value: np.ndarray | float | list[float],
        log_prob: np.ndarray | float | list[float],
        done: np.ndarray | bool | list[bool],
        action_mask: np.ndarray,
    ) -> None:
        """Add a step transition across all environments."""
        if self.step >= self.num_steps:
            raise IndexError("VectorRolloutBuffer is full. Call clear() before adding more steps.")

        self.obs_buf[self.step] = np.asarray(obs, dtype=np.float32)
        self.actions_buf[self.step] = np.asarray(action, dtype=np.int64)
        self.rewards_buf[self.step] = np.asarray(reward, dtype=np.float32)
        self.values_buf[self.step] = np.asarray(value, dtype=np.float32)
        self.log_probs_buf[self.step] = np.asarray(log_prob, dtype=np.float32)
        self.dones_buf[self.step] = np.asarray(done, dtype=bool)
        self.masks_buf[self.step] = np.asarray(action_mask, dtype=bool)

        self.step += 1

    def compute_returns_and_advantages(
        self,
        next_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute vectorized GAE advantages and returns for current rollout."""
        from src.rl.advantage import compute_gae_vectorized

        return compute_gae_vectorized(
            rewards=self.rewards_buf[: self.step],
            values=self.values_buf[: self.step],
            dones=self.dones_buf[: self.step],
            next_values=np.asarray(next_values, dtype=np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def generate_minibatches(
        self,
        batch_size: int,
        advantages: np.ndarray,
        returns: np.ndarray,
        device: str = "cpu",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield flattened, randomized minibatches for multi-epoch PPO updates."""
        total_samples = self.step * self.num_envs
        if total_samples == 0:
            return

        b_obs = self.obs_buf[: self.step].reshape(-1, self.obs_dim)
        b_actions = self.actions_buf[: self.step].reshape(-1)
        b_log_probs = self.log_probs_buf[: self.step].reshape(-1)
        b_values = self.values_buf[: self.step].reshape(-1)
        b_masks = self.masks_buf[: self.step].reshape(-1, self.action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        indices = np.random.permutation(total_samples)

        t_obs = torch.from_numpy(b_obs).to(device=device)
        t_actions = torch.from_numpy(b_actions).to(device=device)
        t_log_probs = torch.from_numpy(b_log_probs).to(device=device)
        t_values = torch.from_numpy(b_values).to(device=device)
        t_masks = torch.from_numpy(b_masks).to(device=device)
        t_advantages = torch.from_numpy(b_advantages).to(device=device)
        t_returns = torch.from_numpy(b_returns).to(device=device)

        for start in range(0, total_samples, batch_size):
            mb_idx = indices[start : start + batch_size]
            yield {
                "obs": t_obs[mb_idx],
                "actions": t_actions[mb_idx],
                "old_log_probs": t_log_probs[mb_idx],
                "values": t_values[mb_idx],
                "advantages": t_advantages[mb_idx],
                "returns": t_returns[mb_idx],
                "action_masks": t_masks[mb_idx],
            }

    def __len__(self) -> int:
        return self.step * self.num_envs


class RecurrentRolloutBuffer:
    """Trajectory storage for recurrent PPO supporting sequential chunk mini-batches."""

    def __init__(
        self,
        capacity: int = 2048,
        obs_dim: int = 0,
        action_dim: int = 0,
        lstm_hidden_dim: int = 128,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
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
        self.h_buf: np.ndarray | None = None
        self.c_buf: np.ndarray | None = None
        self.advantages_buf = np.zeros(capacity, dtype=np.float32)
        self.returns_buf = np.zeros(capacity, dtype=np.float32)

        if obs_dim > 0 and action_dim > 0:
            self._lazy_init(obs_dim, action_dim)

    def _lazy_init(self, obs_dim: int, action_dim: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.masks_buf = np.zeros((self.capacity, action_dim), dtype=bool)
        self.h_buf = np.zeros((self.capacity, self.lstm_hidden_dim), dtype=np.float32)
        self.c_buf = np.zeros((self.capacity, self.lstm_hidden_dim), dtype=np.float32)
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
        h: np.ndarray,
        c: np.ndarray,
    ) -> None:
        if (
            not self.initialized
            or self.obs_buf is None
            or self.masks_buf is None
            or self.h_buf is None
            or self.c_buf is None
        ):
            self._lazy_init(len(obs), len(action_mask))

        assert (
            self.obs_buf is not None
            and self.masks_buf is not None
            and self.h_buf is not None
            and self.c_buf is not None
        )

        if self.ptr >= self.capacity:
            new_cap = self.capacity * 2
            new_obs = np.zeros((new_cap, self.obs_dim), dtype=np.float32)
            new_masks = np.zeros((new_cap, self.action_dim), dtype=bool)
            new_h = np.zeros((new_cap, self.lstm_hidden_dim), dtype=np.float32)
            new_c = np.zeros((new_cap, self.lstm_hidden_dim), dtype=np.float32)
            new_obs[: self.capacity] = self.obs_buf
            new_masks[: self.capacity] = self.masks_buf
            new_h[: self.capacity] = self.h_buf
            new_c[: self.capacity] = self.c_buf
            self.obs_buf = new_obs
            self.masks_buf = new_masks
            self.h_buf = new_h
            self.c_buf = new_c

            self.actions_buf = np.pad(self.actions_buf, (0, self.capacity))
            self.rewards_buf = np.pad(self.rewards_buf, (0, self.capacity))
            self.values_buf = np.pad(self.values_buf, (0, self.capacity))
            self.log_probs_buf = np.pad(self.log_probs_buf, (0, self.capacity))
            self.dones_buf = np.pad(self.dones_buf, (0, self.capacity))
            self.advantages_buf = np.pad(self.advantages_buf, (0, self.capacity))
            self.returns_buf = np.pad(self.returns_buf, (0, self.capacity))
            self.capacity = new_cap

        self.obs_buf[self.ptr] = obs
        self.actions_buf[self.ptr] = int(action)
        self.rewards_buf[self.ptr] = float(reward)
        self.values_buf[self.ptr] = float(value)
        self.log_probs_buf[self.ptr] = float(log_prob)
        self.dones_buf[self.ptr] = bool(done)
        self.masks_buf[self.ptr] = action_mask
        self.h_buf[self.ptr] = h
        self.c_buf[self.ptr] = c

        self.ptr += 1
        self.size = self.ptr

    def set_advantages_and_returns(self, advantages: np.ndarray, returns: np.ndarray) -> None:
        self.advantages_buf[: self.size] = advantages[: self.size]
        self.returns_buf[: self.size] = returns[: self.size]

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0

    def generate_recurrent_chunks(
        self,
        seq_len: int = 8,
        batch_size: int = 64,
        num_epochs: int = 4,
        device: torch.device | str = "cpu",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield mini-batches of contiguous sequence chunks for recurrent BPTT."""
        num_chunks = self.size // seq_len
        if (
            num_chunks == 0
            or self.obs_buf is None
            or self.masks_buf is None
            or self.h_buf is None
            or self.c_buf is None
        ):
            return

        chunk_starts = np.arange(0, num_chunks * seq_len, seq_len)

        for _ in range(num_epochs):
            np.random.shuffle(chunk_starts)
            for i in range(0, len(chunk_starts), batch_size):
                batch_starts = chunk_starts[i : i + batch_size]
                actual_batch_size = len(batch_starts)

                batch_obs = np.zeros((actual_batch_size, seq_len, self.obs_dim), dtype=np.float32)
                batch_actions = np.zeros((actual_batch_size, seq_len), dtype=np.int64)
                batch_log_probs = np.zeros((actual_batch_size, seq_len), dtype=np.float32)
                batch_values = np.zeros((actual_batch_size, seq_len), dtype=np.float32)
                batch_advs = np.zeros((actual_batch_size, seq_len), dtype=np.float32)
                batch_returns = np.zeros((actual_batch_size, seq_len), dtype=np.float32)
                batch_masks = np.zeros((actual_batch_size, seq_len, self.action_dim), dtype=bool)
                batch_dones = np.zeros((actual_batch_size, seq_len), dtype=bool)
                batch_h = np.zeros((1, actual_batch_size, self.lstm_hidden_dim), dtype=np.float32)
                batch_c = np.zeros((1, actual_batch_size, self.lstm_hidden_dim), dtype=np.float32)

                for b_idx, start in enumerate(batch_starts):
                    end = start + seq_len
                    batch_obs[b_idx] = self.obs_buf[start:end]
                    batch_actions[b_idx] = self.actions_buf[start:end]
                    batch_log_probs[b_idx] = self.log_probs_buf[start:end]
                    batch_values[b_idx] = self.values_buf[start:end]
                    batch_advs[b_idx] = self.advantages_buf[start:end]
                    batch_returns[b_idx] = self.returns_buf[start:end]
                    batch_masks[b_idx] = self.masks_buf[start:end]
                    batch_dones[b_idx] = self.dones_buf[start:end]
                    batch_h[0, b_idx] = self.h_buf[start]
                    batch_c[0, b_idx] = self.c_buf[start]

                yield {
                    "obs": torch.as_tensor(batch_obs, device=device),
                    "actions": torch.as_tensor(batch_actions, device=device),
                    "old_log_probs": torch.as_tensor(batch_log_probs, device=device),
                    "values": torch.as_tensor(batch_values, device=device),
                    "advantages": torch.as_tensor(batch_advs, device=device),
                    "returns": torch.as_tensor(batch_returns, device=device),
                    "action_masks": torch.as_tensor(batch_masks, device=device),
                    "dones": torch.as_tensor(batch_dones, device=device),
                    "initial_h": torch.as_tensor(batch_h, device=device),
                    "initial_c": torch.as_tensor(batch_c, device=device),
                }

    def __len__(self) -> int:
        return self.size
