"""Masked Double-DQN Trainer implementation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.environment.env import TicketToRideEnv
from src.rl.networks import MaskedQNetwork
from src.rl.replay_buffer import ReplayBuffer


class MaskedDQNTrainer:
    """Double Deep Q-Network Trainer with Action Masking."""

    def __init__(self, env: TicketToRideEnv, config: dict[str, Any] | None = None) -> None:
        self.env = env
        self.config = config or {}

        self.gamma: float = self.config.get("gamma", 0.99)
        self.lr: float = self.config.get("lr", 5e-4)
        self.batch_size: int = self.config.get("batch_size", 64)
        self.buffer_size: int = self.config.get("buffer_size", 50000)
        self.target_update_freq: int = self.config.get("target_update_freq", 500)
        self.epsilon_start: float = self.config.get("epsilon_start", 1.0)
        self.epsilon_end: float = self.config.get("epsilon_end", 0.05)
        self.epsilon_decay_steps: int = self.config.get("epsilon_decay_steps", 20000)
        self.learning_starts: int = self.config.get("learning_starts", 500)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 1.0)
        self.device: str = self.config.get("device", "cpu")

        obs_dim = self.env.observation_space.shape[0]
        action_dim = int(self.env.action_space.n)

        self.policy_net = MaskedQNetwork(input_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.target_net = MaskedQNetwork(input_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.total_timesteps = 0
        self.current_obs, self.current_info = self.env.reset()

    def get_epsilon(self) -> float:
        """Compute linearly decayed epsilon."""
        progress = min(1.0, self.total_timesteps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def step(self, epsilon: float | None = None) -> tuple[float, bool]:
        """Execute a single environment step with epsilon-greedy action selection."""
        eps = self.get_epsilon() if epsilon is None else epsilon
        action_mask = self.current_info["action_mask"]

        obs_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device)
        action = self.policy_net.select_action(obs_tensor, action_mask, epsilon=eps)

        next_obs, reward, terminated, truncated, next_info = self.env.step(action)
        done = terminated or truncated

        self.replay_buffer.push(
            obs=self.current_obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            next_action_mask=next_info["action_mask"],
        )

        self.total_timesteps += 1

        if done:
            self.current_obs, self.current_info = self.env.reset()
        else:
            self.current_obs = next_obs
            self.current_info = next_info

        return reward, done

    def train_step(self) -> dict[str, float]:
        """Sample a batch and update Q-network weights with Double-DQN."""
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return {"loss": 0.0, "q_mean": 0.0, "epsilon": self.get_epsilon()}

        batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones
        next_masks = batch.next_action_masks

        # Current Q(s, a)
        q_values = self.policy_net(obs)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target:
        # 1. Best action chosen by policy_net over valid next actions
        with torch.no_grad():
            next_q_policy = self.policy_net(next_obs)
            masked_next_q = next_q_policy.masked_fill(~next_masks, -1e9)
            best_next_actions = torch.argmax(masked_next_q, dim=1, keepdim=True)

            # 2. Value estimated by target_net for best_next_actions
            next_q_target = self.target_net(next_obs)
            next_state_values = next_q_target.gather(1, best_next_actions).squeeze(1)
            target_values = rewards + (self.gamma * next_state_values * (1.0 - dones))

        loss = F.smooth_l1_loss(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.total_timesteps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(state_action_values.mean().item()),
            "epsilon": self.get_epsilon(),
        }

    def train(
        self,
        total_timesteps: int,
        eval_callback: Callable[[int, "MaskedDQNTrainer"], None] | None = None,
        eval_freq: int = 5000,
    ) -> dict[str, Any]:
        """Train agent for a specified number of environment steps."""
        losses: list[float] = []
        q_means: list[float] = []

        while self.total_timesteps < total_timesteps:
            self.step()
            metrics = self.train_step()
            if metrics["loss"] > 0.0:
                losses.append(metrics["loss"])
                q_means.append(metrics["q_mean"])

            if eval_callback is not None and self.total_timesteps % eval_freq == 0:
                eval_callback(self.total_timesteps, self)

        return {
            "total_timesteps": self.total_timesteps,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_q": float(np.mean(q_means)) if q_means else 0.0,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)


# Backwards compatibility alias
DQNTrainer = MaskedDQNTrainer
