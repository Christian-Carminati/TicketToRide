"""Masked Proximal Policy Optimization (PPO) Trainer implementation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.environment.env import TicketToRideEnv
from src.rl.advantage import compute_gae
from src.rl.networks import MaskedActorCritic
from src.rl.rollout import RolloutBuffer


class MaskedPPOTrainer:
    """Trainer for Masked Proximal Policy Optimization."""

    def __init__(self, env: TicketToRideEnv, config: dict[str, Any] | None = None) -> None:
        self.env = env
        self.config = config or {}

        self.gamma: float = self.config.get("gamma", 0.99)
        self.gae_lambda: float = self.config.get("gae_lambda", 0.95)
        self.clip_eps: float = self.config.get("clip_eps", 0.2)
        self.vf_coef: float = self.config.get("vf_coef", 0.5)
        self.ent_coef: float = self.config.get("ent_coef", 0.01)
        self.lr: float = self.config.get("lr", 3e-4)
        self.rollout_steps: int = self.config.get("rollout_steps", 512)
        self.num_epochs: int = self.config.get("num_epochs", 4)
        self.minibatch_size: int = self.config.get("minibatch_size", 64)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 0.5)
        self.device: str = self.config.get("device", "cpu")

        obs_dim = self.env.observation_space.shape[0]
        action_dim = int(self.env.action_space.n)

        self.actor_critic = MaskedActorCritic(input_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(
            capacity=self.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.total_timesteps = 0
        self.current_obs, self.current_info = self.env.reset()

    def collect_rollout(self) -> dict[str, float]:
        """Collect rollout_steps transitions using current policy."""
        self.rollout_buffer.clear()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        for _ in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.as_tensor(self.current_info["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.actor_critic.get_action_and_value(
                    obs_tensor, action_mask=mask_tensor
                )

            act_item = int(action.item())
            next_obs, reward, terminated, truncated, next_info = self.env.step(act_item)
            done = terminated or truncated

            self.rollout_buffer.add(
                obs=self.current_obs,
                action=act_item,
                reward=reward,
                value=float(value.item()),
                log_prob=float(log_prob.item()),
                done=done,
                action_mask=self.current_info["action_mask"],
            )

            current_ep_reward += reward
            self.total_timesteps += 1

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                self.current_obs, self.current_info = self.env.reset()
            else:
                self.current_obs = next_obs
                self.current_info = next_info

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return {"mean_rollout_reward": mean_reward, "episodes": float(len(episode_rewards))}

    def train_epoch(self) -> dict[str, float]:
        """Perform PPO optimization on collected rollout."""
        # Estimate next state value for GAE boundary
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_val = self.actor_critic(obs_tensor)
            next_value = float(next_val.item())

        n_steps = self.rollout_buffer.size
        advantages, returns = compute_gae(
            rewards=self.rollout_buffer.rewards_buf[:n_steps],
            values=self.rollout_buffer.values_buf[:n_steps],
            dones=self.rollout_buffer.dones_buf[:n_steps],
            next_value=next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.num_epochs):
            for mb in self.rollout_buffer.generate_minibatches(
                batch_size=self.minibatch_size,
                advantages=norm_advantages,
                returns=returns,
                device=self.device,
            ):
                _, new_log_prob, entropy, new_value = self.actor_critic.get_action_and_value(
                    mb["obs"],
                    action_mask=mb["action_masks"],
                    action=mb["actions"],
                )

                log_ratio = new_log_prob - mb["old_log_probs"]
                ratio = torch.exp(log_ratio)

                # Clipped Policy Objective
                surr1 = ratio * mb["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(new_value.squeeze(1), mb["returns"])

                # Entropy Bonus
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

        y_true = returns
        y_pred = self.rollout_buffer.values_buf[:n_steps]
        var_y = np.var(y_true)
        explained_var = float(np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y)

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "explained_var": explained_var,
        }

    def train(
        self,
        total_timesteps: int,
        eval_callback: Callable[[int, "MaskedPPOTrainer"], None] | None = None,
        eval_freq: int = 5000,
    ) -> dict[str, Any]:
        """Train agent for a specified number of environment steps."""
        last_eval_step = 0
        all_policy_losses = []
        all_value_losses = []

        while self.total_timesteps < total_timesteps:
            self.collect_rollout()
            metrics = self.train_epoch()
            all_policy_losses.append(metrics["policy_loss"])
            all_value_losses.append(metrics["value_loss"])

            if eval_callback is not None and (self.total_timesteps - last_eval_step >= eval_freq):
                eval_callback(self.total_timesteps, self)
                last_eval_step = self.total_timesteps

        return {
            "total_timesteps": self.total_timesteps,
            "mean_policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "mean_value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_timesteps": self.total_timesteps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)


# Backwards compatibility alias
PPOTrainer = MaskedPPOTrainer
