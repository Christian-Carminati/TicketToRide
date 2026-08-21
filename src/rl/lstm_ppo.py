"""Recurrent Actor-Critic network architecture and PPO trainer for POMDPs."""

from collections.abc import Callable
import math
from typing import Any
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
import torch.nn.functional as F

from src.environment.env import TicketToRideEnv
from src.rl.advantage import compute_gae
from src.rl.rollout import RecurrentRolloutBuffer


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layers with orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RecurrentMaskedActorCritic(nn.Module):
    """Recurrent Actor-Critic architecture with LSTM memory and action masking."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        if orthogonal_init:
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim), gain=np.sqrt(2)),
                nn.Tanh(),
            )
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            self.actor = layer_init(nn.Linear(lstm_hidden_dim, action_dim), gain=0.01)
            self.critic = layer_init(nn.Linear(lstm_hidden_dim, 1), gain=1.0)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            )
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            self.actor = nn.Linear(lstm_hidden_dim, action_dim)
            self.critic = nn.Linear(lstm_hidden_dim, 1)

    def get_initial_hidden(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero initial hidden states (h, c) for LSTM."""
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, dtype=torch.float32, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, dtype=torch.float32, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass across either single step or sequence tensor."""
        if x.dim() == 2:
            # (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        features = self.encoder(x)
        lstm_out, new_hidden = self.lstm(features, hidden)
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)

        if squeeze_output:
            return logits.squeeze(1), values.squeeze(1), new_hidden
        return logits, values, new_hidden

    def get_action_and_value(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute action, log_prob, entropy, and value with optional action masking."""
        logits, values, new_hidden = self.forward(x, hidden)

        if action_mask is not None:
            if action_mask.dim() == 2 and logits.dim() == 3:
                action_mask = action_mask.unsqueeze(1)
            masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))
        else:
            masked_logits = logits

        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values, new_hidden

    def get_value(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute state value and return updated hidden state."""
        _, values, new_hidden = self.forward(x, hidden)
        return values, new_hidden


class MaskedRecurrentPPOTrainer:
    """Trainer for Recurrent PPO with LSTM memory and CleanRL standard optimizations."""

    def __init__(
        self,
        env: TicketToRideEnv,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.config = config or {}
        self.seed = seed if seed is not None else self.config.get("seed")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

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
        self.seq_len: int = self.config.get("seq_len", 8)
        self.minibatch_chunks: int = self.config.get("minibatch_chunks", 8)
        self.num_epochs: int = self.config.get("num_epochs", 4)
        self.max_grad_norm: float = self.config.get("max_grad_norm", 0.5)
        self.hidden_dim: int = self.config.get("hidden_dim", 128)
        self.lstm_hidden_dim: int = self.config.get("lstm_hidden_dim", 128)
        self.orthogonal_init: bool = self.config.get("orthogonal_init", True)
        self.device: str = self.config.get("device", "cpu")

        if self.device == "cpu" and torch.get_num_threads() > 2:
            torch.set_num_threads(2)

        obs_dim = self.env.observation_space.shape[0]
        action_dim = int(self.env.action_space.n)

        self.actor_critic = RecurrentMaskedActorCritic(
            input_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            orthogonal_init=self.orthogonal_init,
        ).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        self.rollout_buffer = RecurrentRolloutBuffer(
            capacity=self.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
        )

        self.total_timesteps = 0
        self.current_obs, self.current_info = self.env.reset()
        self.current_hidden = self.actor_critic.get_initial_hidden(batch_size=1, device=self.device)

    def collect_rollout(self) -> dict[str, float]:
        """Collect rollout transitions while tracking recurrent states."""
        self.rollout_buffer.clear()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        for _ in range(self.rollout_steps):
            obs_tensor = torch.from_numpy(self.current_obs).unsqueeze(0).to(device=self.device)
            mask_np = self.current_info.get("action_mask")
            if mask_np is None:
                mask_np = np.ones(self.env.action_space.n, dtype=bool)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device=self.device)

            h_step = self.current_hidden[0][0, 0].cpu().numpy().copy()
            c_step = self.current_hidden[1][0, 0].cpu().numpy().copy()

            with torch.no_grad():
                action_tensor, log_prob_tensor, _, val_tensor, next_hidden = (
                    self.actor_critic.get_action_and_value(
                        obs_tensor,
                        self.current_hidden,
                        action_mask=mask_tensor,
                    )
                )

            action = int(action_tensor.item())
            log_prob = float(log_prob_tensor.item())
            val = float(val_tensor.item())

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated
            current_ep_reward += reward

            self.rollout_buffer.add(
                obs=self.current_obs,
                action=action,
                reward=reward,
                value=val,
                log_prob=log_prob,
                done=done,
                action_mask=mask_np,
                h=h_step,
                c=c_step,
            )

            self.total_timesteps += 1

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                self.current_obs, self.current_info = self.env.reset()
                # Boundary reset on done
                self.current_hidden = self.actor_critic.get_initial_hidden(batch_size=1, device=self.device)
            else:
                self.current_obs = next_obs
                self.current_info = next_info
                self.current_hidden = next_hidden

        # Bootstrap value for unfinished rollout
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(self.current_obs).unsqueeze(0).to(device=self.device)
            last_val_tensor, _ = self.actor_critic.get_value(last_obs_tensor, self.current_hidden)
            last_value = float(last_val_tensor.item())

        rewards = np.array(self.rollout_buffer.rewards_buf[:self.rollout_buffer.size], dtype=np.float32)
        values = np.array(self.rollout_buffer.values_buf[:self.rollout_buffer.size], dtype=np.float32)
        dones = np.array(self.rollout_buffer.dones_buf[:self.rollout_buffer.size], dtype=bool)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self.rollout_buffer.set_advantages_and_returns(advantages, returns)

        return {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "episodes_completed": len(episode_rewards),
        }

    def train_step(self) -> dict[str, float]:
        """Perform one complete rollout collection and optimization cycle."""
        rollout_stats = self.collect_rollout()

        if self.rollout_buffer.size < self.seq_len:
            return rollout_stats

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []

        chunk_generator = self.rollout_buffer.generate_recurrent_chunks(
            seq_len=self.seq_len,
            batch_size=self.minibatch_chunks,
            num_epochs=self.num_epochs,
            device=self.device,
        )

        for batch in chunk_generator:
            obs = batch["obs"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            old_values = batch["values"]
            advantages = batch["advantages"]
            returns = batch["returns"]
            action_masks = batch["action_masks"]
            init_hidden = (batch["initial_h"], batch["initial_c"])

            if self.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            _, new_log_probs, entropy, new_values, _ = self.actor_critic.get_action_and_value(
                obs,
                init_hidden,
                action_mask=action_masks,
                action=actions,
            )

            # Flatten sequence dimensions for loss calculation
            new_log_probs = new_log_probs.view(-1)
            old_log_probs = old_log_probs.view(-1)
            advantages = advantages.view(-1)
            returns = returns.view(-1)
            new_values = new_values.view(-1)
            old_values = old_values.view(-1)
            entropy = entropy.view(-1)

            logratio = new_log_probs - old_log_probs
            ratio = torch.exp(logratio)

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
                approx_kls.append(approx_kl)

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

            # Clipped surrogate objective
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            if self.clip_vloss:
                v_clipped = old_values + torch.clamp(new_values - old_values, -self.vf_clip_eps, self.vf_clip_eps)
                v_loss1 = (new_values - returns) ** 2
                v_loss2 = (v_clipped - returns) ** 2
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                v_loss = 0.5 * ((new_values - returns) ** 2).mean()

            ent_loss = entropy.mean()

            loss = pg_loss - self.ent_coef * ent_loss + self.vf_coef * v_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            policy_losses.append(pg_loss.item())
            value_losses.append(v_loss.item())
            entropy_losses.append(ent_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "mean_reward": rollout_stats["mean_reward"],
            "total_timesteps": self.total_timesteps,
        }

    def train(
        self, total_timesteps: int, callback: Callable[[dict[str, float]], None] | None = None
    ) -> list[dict[str, float]]:
        """Run training until total_timesteps is reached."""
        logs = []
        while self.total_timesteps < total_timesteps:
            if self.anneal_lr:
                frac = 1.0 - (self.total_timesteps / max(1, total_timesteps))
                self.lr = max(0.0, frac * self.initial_lr)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr

            metrics = self.train_step()
            logs.append(metrics)
            if callback:
                callback(metrics)
        return logs

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "total_timesteps": self.total_timesteps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        elif "actor_critic_state_dict" in checkpoint:
            self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)


# Aliases for backward compatibility
RecurrentPPOActorCritic = RecurrentMaskedActorCritic
RecurrentPPOTrainer = MaskedRecurrentPPOTrainer

__all__ = [
    "MaskedRecurrentPPOTrainer",
    "RecurrentMaskedActorCritic",
    "RecurrentPPOActorCritic",
    "RecurrentPPOTrainer",
    "RecurrentRolloutBuffer",
    "layer_init",
]
