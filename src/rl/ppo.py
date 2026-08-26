"""Masked Proximal Policy Optimization (PPO) Trainer implementation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.rl.advantage import compute_gae
from src.rl.networks import MaskedActorCritic
from src.rl.rollout import RolloutBuffer


class MaskedPPOTrainer:
    """Proximal Policy Optimization (PPO) trainer with explicit categorical action masking."""

    def __init__(self, env: Any, config: dict[str, Any] | None = None) -> None:
        self.env = env
        self.config = config or {}

        self.gamma: float = self.config.get("gamma", 0.99)
        self.gae_lambda: float = self.config.get("gae_lambda", 0.95)
        self.clip_eps: float = self.config.get("clip_eps", 0.2)
        self.vf_coef: float = self.config.get("vf_coef", 0.5)
        self.ent_coef: float = self.config.get("ent_coef", 0.01)
        self.initial_lr: float = self.config.get("lr", 3e-4)
        self.lr: float = self.initial_lr
        self.anneal_lr: bool = self.config.get("anneal_lr", False)
        self.clip_vloss: bool = self.config.get("clip_vloss", False)
        self.vf_clip_eps: float = self.config.get("vf_clip_eps", 0.2)
        self.target_kl: float | None = self.config.get("target_kl", None)
        self.norm_adv: bool = self.config.get("norm_adv", True)
        self.rollout_steps: int = self.config.get("rollout_steps", 512)
        self.num_epochs: int = self.config.get("num_epochs", 4)
        self.minibatch_size: int = self.config.get("minibatch_size", 64)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 0.5)
        self.orthogonal_init: bool = self.config.get("orthogonal_init", True)
        self.device: str = self.config.get("device", "cpu")

        if self.device == "cpu" and torch.get_num_threads() > 2:
            torch.set_num_threads(2)

        obs_shape = self.env.observation_space.shape
        obs_dim = obs_shape[0] if obs_shape is not None else 180
        action_dim = int(getattr(self.env.action_space, "n", 150))

        self.actor_critic = MaskedActorCritic(
            input_dim=obs_dim,
            action_dim=action_dim,
            orthogonal_init=self.orthogonal_init,
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(
            capacity=self.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.total_timesteps = 0
        self.current_obs, self.current_info = self.env.reset()

    def update_learning_rate(self, total_timesteps: int) -> float:
        """Update optimizer learning rate following linear annealing schedule."""
        frac = 1.0 - (self.total_timesteps / max(1, total_timesteps))
        lr_now = max(0.0, frac * self.initial_lr)
        self.lr = lr_now
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now
        return lr_now

    def collect_rollout(self) -> dict[str, float]:
        """Collect rollout_steps transitions using current policy."""
        self.rollout_buffer.clear()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        for _ in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device).unsqueeze(0)
            mask_tensor = torch.as_tensor(
                self.current_info["action_mask"], device=self.device
            ).unsqueeze(0)

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

    def train_epoch(self) -> dict[str, Any]:
        """Perform PPO optimization on collected rollout with CleanRL stability features."""
        # Estimate next state value for GAE boundary
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device).unsqueeze(0)
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

        # Normalize advantages across full rollout if enabled
        if self.norm_adv:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            norm_advantages = (advantages - adv_mean) / adv_std
        else:
            norm_advantages = advantages

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        early_stopped = False
        epochs_completed = 0

        for _ in range(self.num_epochs):
            epoch_kls: list[float] = []
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
                mb_advantages = mb["advantages"]
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (Clipped or Unclipped)
                new_val_flat = new_value.squeeze(1) if new_value.dim() > 1 else new_value
                if self.clip_vloss and "values" in mb:
                    old_v = mb["values"]
                    v_loss_unclipped = (new_val_flat - mb["returns"]) ** 2
                    v_clipped = old_v + torch.clamp(
                        new_val_flat - old_v, -self.vf_clip_eps, self.vf_clip_eps
                    )
                    v_loss_clipped = (v_clipped - mb["returns"]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    value_loss = v_loss_max.mean()
                else:
                    value_loss = F.mse_loss(new_val_flat, mb["returns"])

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
                    epoch_kls.append(approx_kl.item())
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    clip_fractions.append(clip_frac.item())

            epochs_completed += 1
            if (
                self.target_kl is not None
                and len(epoch_kls) > 0
                and float(np.mean(epoch_kls)) > self.target_kl
            ):
                early_stopped = True
                break

        y_true = returns
        y_pred = self.rollout_buffer.values_buf[:n_steps]
        var_y = np.var(y_true)
        explained_var = float(np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y)

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "explained_var": explained_var,
            "early_stopped": early_stopped,
            "epochs_completed": epochs_completed,
            "lr": self.lr,
        }

    def train(
        self,
        total_timesteps: int,
        eval_callback: Callable[[int, "MaskedPPOTrainer"], None] | None = None,
        eval_freq: int = 5000,
    ) -> dict[str, Any]:
        """Train agent for a specified number of environment steps."""
        last_eval_step = 0
        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_approx_kls: list[float] = []
        all_clip_fracs: list[float] = []

        while self.total_timesteps < total_timesteps:
            if self.anneal_lr:
                self.update_learning_rate(total_timesteps)

            self.collect_rollout()
            metrics = self.train_epoch()
            all_policy_losses.append(metrics["policy_loss"])
            all_value_losses.append(metrics["value_loss"])
            all_approx_kls.append(metrics["approx_kl"])
            all_clip_fracs.append(metrics["clip_fraction"])

            if eval_callback is not None and (self.total_timesteps - last_eval_step >= eval_freq):
                eval_callback(self.total_timesteps, self)
                last_eval_step = self.total_timesteps

        return {
            "total_timesteps": self.total_timesteps,
            "mean_policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "mean_value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "mean_approx_kl": float(np.mean(all_approx_kls)) if all_approx_kls else 0.0,
            "mean_clip_fraction": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
            "final_lr": self.lr,
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
