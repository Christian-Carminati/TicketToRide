"""Historical policy pool management and matchmaking for self-play training."""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.environment.env import TicketToRideEnv
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer, RecurrentMaskedActorCritic
from src.rl.networks import MaskedActorCritic
from src.rl.ppo import MaskedPPOTrainer


@dataclass
class PolicySnapshot:
    """Frozen snapshot of an agent policy checkpoint."""

    generation: int
    step: int
    name: str
    state_dict: dict[str, torch.Tensor]
    is_recurrent: bool = False
    hidden_dim: int = 128
    lstm_hidden_dim: int = 128
    input_dim: int = 50
    action_dim: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyPool:
    """Historical pool of frozen policy checkpoints with capacity limits and retention."""

    max_size: int = 50
    _snapshots: list[PolicySnapshot] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[PolicySnapshot]:
        return list(self._snapshots)

    def add_policy(
        self,
        model: nn.Module,
        step: int,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot:
        gen = len(self._snapshots)
        snap_name = name or f"gen_{gen:03d}_step_{step}"
        is_recurrent = isinstance(model, RecurrentMaskedActorCritic) or hasattr(model, "lstm")

        # Clone state_dict to CPU
        cpu_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        hidden_dim = getattr(model, "hidden_dim", 128)
        lstm_hidden_dim = getattr(model, "lstm_hidden_dim", 128)
        input_dim = getattr(model, "input_dim", 50)
        action_dim = getattr(model, "action_dim", 10)

        if is_recurrent:
            if "encoder.0.weight" in cpu_state_dict:
                input_dim = cpu_state_dict["encoder.0.weight"].shape[1]
                hidden_dim = cpu_state_dict["encoder.0.weight"].shape[0]
            if "critic.weight" in cpu_state_dict:
                lstm_hidden_dim = cpu_state_dict["critic.weight"].shape[1]
            if "actor.weight" in cpu_state_dict:
                action_dim = cpu_state_dict["actor.weight"].shape[0]
        else:
            if "actor.0.weight" in cpu_state_dict:
                input_dim = cpu_state_dict["actor.0.weight"].shape[1]
                hidden_dim = cpu_state_dict["actor.0.weight"].shape[0]
            if "actor.4.weight" in cpu_state_dict:
                action_dim = cpu_state_dict["actor.4.weight"].shape[0]

        snapshot = PolicySnapshot(
            generation=gen,
            step=step,
            name=snap_name,
            state_dict=cpu_state_dict,
            is_recurrent=is_recurrent,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            input_dim=input_dim,
            action_dim=action_dim,
            metadata=metadata or {},
        )

        if len(self._snapshots) >= self.max_size:
            # Preserve generation 0 as the anchor baseline, remove second oldest
            if len(self._snapshots) > 1:
                self._snapshots.pop(1)
            else:
                self._snapshots.pop(0)

        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshot(self, index_or_name: int | str) -> PolicySnapshot:
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self._snapshots):
                return self._snapshots[index_or_name]
            raise IndexError(f"Snapshot index {index_or_name} out of bounds (size {len(self._snapshots)})")

        for snap in self._snapshots:
            if snap.name == index_or_name:
                return snap
        raise KeyError(f"Snapshot '{index_or_name}' not found in policy pool")

    def create_agent(
        self,
        index_or_name: int | str,
        board: Any = None,
        tickets: Any = None,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> BaseAgent:
        snapshot = self.get_snapshot(index_or_name)
        if snapshot.is_recurrent:
            model = RecurrentMaskedActorCritic(
                input_dim=snapshot.input_dim,
                action_dim=snapshot.action_dim,
                hidden_dim=snapshot.hidden_dim,
                lstm_hidden_dim=snapshot.lstm_hidden_dim,
            )
            model.load_state_dict(snapshot.state_dict)
            model.eval()
            return RecurrentPPOAgent(
                model=model,
                board=board,
                tickets=tickets,
                deterministic=deterministic,
                device=device,
                name=snapshot.name,
            )
        else:
            model = MaskedActorCritic(
                input_dim=snapshot.input_dim,
                action_dim=snapshot.action_dim,
                hidden_dim=snapshot.hidden_dim,
            )
            model.load_state_dict(snapshot.state_dict)
            model.eval()
            return PPOAgent(
                model=model,
                board=board,
                tickets=tickets,
                device=device,
                name=snapshot.name,
            )

    def save_pool(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        manifest = []
        for snap in self._snapshots:
            snap_file = f"{snap.name}.pt"
            snap_path = os.path.join(directory, snap_file)
            torch.save(
                {
                    "generation": snap.generation,
                    "step": snap.step,
                    "name": snap.name,
                    "state_dict": snap.state_dict,
                    "is_recurrent": snap.is_recurrent,
                    "hidden_dim": snap.hidden_dim,
                    "lstm_hidden_dim": snap.lstm_hidden_dim,
                    "input_dim": snap.input_dim,
                    "action_dim": snap.action_dim,
                    "metadata": snap.metadata,
                },
                snap_path,
            )
            manifest.append({
                "name": snap.name,
                "generation": snap.generation,
                "step": snap.step,
                "file": snap_file,
            })
        with open(os.path.join(directory, "pool_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def load_pool(self, directory: str) -> None:
        manifest_path = os.path.join(directory, "pool_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self._snapshots.clear()
        for item in manifest:
            snap_path = os.path.join(directory, item["file"])
            ckpt = torch.load(snap_path, map_location="cpu")
            snapshot = PolicySnapshot(
                generation=ckpt["generation"],
                step=ckpt["step"],
                name=ckpt["name"],
                state_dict=ckpt["state_dict"],
                is_recurrent=ckpt.get("is_recurrent", False),
                hidden_dim=ckpt.get("hidden_dim", 128),
                lstm_hidden_dim=ckpt.get("lstm_hidden_dim", 128),
                input_dim=ckpt.get("input_dim", 50),
                action_dim=ckpt.get("action_dim", 10),
                metadata=ckpt.get("metadata", {}),
            )
            self._snapshots.append(snapshot)


class SelfPlayOpponentSampler:
    """Dynamic matchmaking sampler supporting Uniform, Latest-biased, PFSP, and Baseline Mix-in."""

    def __init__(
        self,
        strategy: str = "latest_biased",
        baseline_mix_rate: float = 0.15,
        pfsp_exponent: float = 1.0,
        baseline_agents: list[BaseAgent] | None = None,
        seed: int = 42,
    ) -> None:
        self.strategy = strategy.lower()
        self.baseline_mix_rate = max(0.0, min(1.0, baseline_mix_rate))
        self.pfsp_exponent = pfsp_exponent
        self.rng = random.Random(seed)
        self.match_records: dict[str, dict[str, int]] = {}
        if baseline_agents is not None:
            self.baseline_agents = baseline_agents
        else:
            from src.agents.greedy_agent import GreedyAgent
            from src.agents.heuristic_agent import StrategicAgent
            from src.agents.random_agent import RandomAgent

            self.baseline_agents = [
                RandomAgent(name="RandomBot", seed=seed),
                GreedyAgent(name="GreedyBot"),
                StrategicAgent(name="StrategicBot"),
            ]

    def record_match(self, opponent_name: str, trainee_won: bool) -> None:
        if opponent_name not in self.match_records:
            self.match_records[opponent_name] = {"trainee_wins": 0, "total_games": 0}
        self.match_records[opponent_name]["total_games"] += 1
        if trainee_won:
            self.match_records[opponent_name]["trainee_wins"] += 1

    def get_opponent_weights(self, pool: PolicyPool) -> dict[str, float]:
        if pool.size == 0:
            return {}
        names = [s.name for s in pool.snapshots]
        n = len(names)

        if self.strategy == "uniform" or n == 1:
            return {name: 1.0 / n for name in names}

        if self.strategy == "latest_biased":
            p_latest = 0.5
            p_hist = (1.0 - p_latest) / (n - 1) if n > 1 else 0.0
            weights = {name: p_hist for name in names}
            weights[names[-1]] = p_latest if n > 1 else 1.0
            return weights

        if self.strategy == "pfsp":
            raw_weights = []
            for name in names:
                rec = self.match_records.get(name, {"trainee_wins": 0, "total_games": 0})
                if rec["total_games"] == 0:
                    win_rate = 0.5
                else:
                    win_rate = rec["trainee_wins"] / rec["total_games"]
                loss_rate = 1.0 - win_rate
                score = (loss_rate ** self.pfsp_exponent) + 0.05
                raw_weights.append(score)
            total = sum(raw_weights)
            return {name: raw_weights[i] / total for i, name in enumerate(names)}

        # Fallback to uniform
        return {name: 1.0 / n for name in names}

    def sample_opponent(
        self,
        pool: PolicyPool,
        board: Any = None,
        tickets: Any = None,
        deterministic: bool = True,
    ) -> BaseAgent:
        # 1. Check if baseline mix-in triggers
        if self.baseline_agents and self.rng.random() < self.baseline_mix_rate:
            return self.rng.choice(self.baseline_agents)

        # 2. Fallback to random bot if pool is empty
        if pool.size == 0:
            from src.agents.random_agent import RandomAgent

            return self.baseline_agents[0] if self.baseline_agents else RandomAgent()

        # 3. Sample historical policy from pool
        weights_dict = self.get_opponent_weights(pool)
        names = list(weights_dict.keys())
        probs = [weights_dict[name] for name in names]

        chosen_name = self.rng.choices(names, weights=probs, k=1)[0]
        return pool.create_agent(chosen_name, board=board, tickets=tickets, deterministic=deterministic)


class SelfPlayPPOTrainer(MaskedPPOTrainer):
    """Self-Play PPO Trainer updating opponent per episode and saving policy snapshots."""

    def __init__(
        self,
        env: TicketToRideEnv,
        config: dict[str, Any] | None = None,
        pool: PolicyPool | None = None,
        sampler: SelfPlayOpponentSampler | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(env=env, config=config)
        self.seed = seed
        self.pool = pool or PolicyPool(max_size=self.config.get("pool_max_size", 50))
        self.sampler = sampler or SelfPlayOpponentSampler(
            strategy=self.config.get("sampling_strategy", "latest_biased"),
            baseline_mix_rate=self.config.get("baseline_mix_rate", 0.15),
            pfsp_exponent=self.config.get("pfsp_exponent", 1.0),
            seed=seed,
        )
        self.snapshot_interval: int = self.config.get("snapshot_interval", 5000)
        self.last_snapshot_step: int = 0

        # Snapshot generation 0 (initial policy)
        self.pool.add_policy(self.actor_critic, step=0, name="gen_000_initial")

    def _switch_opponent_for_new_episode(self) -> None:
        """Sample and assign next opponent to the environment."""
        new_opp = self.sampler.sample_opponent(
            pool=self.pool,
            board=self.env.board,
            tickets=self.env.initial_tickets,
            deterministic=True,
        )
        self.env.opponent = new_opp

    def collect_rollout(self) -> dict[str, float]:
        self.rollout_buffer.clear()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        for _ in range(self.rollout_steps):
            obs_tensor = torch.from_numpy(self.current_obs).unsqueeze(0).to(device=self.device)
            mask_tensor = torch.from_numpy(self.current_info["action_mask"]).unsqueeze(0).to(device=self.device)

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
                # Record result for PFSP
                trainee_won = (next_info.get("winner_id") == 0)
                opp_name = getattr(self.env.opponent, "name", "Opponent")
                self.sampler.record_match(opp_name, trainee_won=trainee_won)

                current_ep_reward = 0.0
                # Matchmaking switch for the next episode
                self._switch_opponent_for_new_episode()
                self.current_obs, self.current_info = self.env.reset()
            else:
                self.current_obs = next_obs
                self.current_info = next_info

            # Check snapshot interval
            if self.total_timesteps - self.last_snapshot_step >= self.snapshot_interval:
                gen_idx = self.pool.size
                self.pool.add_policy(
                    self.actor_critic,
                    step=self.total_timesteps,
                    name=f"gen_{gen_idx:03d}_step_{self.total_timesteps}",
                )
                self.last_snapshot_step = self.total_timesteps

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return {"mean_rollout_reward": mean_reward, "episodes": float(len(episode_rewards))}


class SelfPlayRecurrentPPOTrainer(MaskedRecurrentPPOTrainer):
    """Self-Play Recurrent PPO Trainer with LSTM memory and dynamic matchmaking."""

    def __init__(
        self,
        env: TicketToRideEnv,
        config: dict[str, Any] | None = None,
        pool: PolicyPool | None = None,
        sampler: SelfPlayOpponentSampler | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(env=env, config=config, seed=seed)
        self.pool = pool or PolicyPool(max_size=self.config.get("pool_max_size", 50))
        self.sampler = sampler or SelfPlayOpponentSampler(
            strategy=self.config.get("sampling_strategy", "latest_biased"),
            baseline_mix_rate=self.config.get("baseline_mix_rate", 0.15),
            pfsp_exponent=self.config.get("pfsp_exponent", 1.0),
            seed=seed,
        )
        self.snapshot_interval: int = self.config.get("snapshot_interval", 5000)
        self.last_snapshot_step: int = 0

        # Snapshot initial policy
        self.pool.add_policy(self.actor_critic, step=0, name="gen_000_initial")

    def _switch_opponent_for_new_episode(self) -> None:
        new_opp = self.sampler.sample_opponent(
            pool=self.pool,
            board=self.env.board,
            tickets=self.env.initial_tickets,
            deterministic=True,
        )
        self.env.opponent = new_opp

    def collect_rollout(self) -> dict[str, float]:
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
                trainee_won = (next_info.get("winner_id") == 0)
                opp_name = getattr(self.env.opponent, "name", "Opponent")
                self.sampler.record_match(opp_name, trainee_won=trainee_won)

                current_ep_reward = 0.0
                self._switch_opponent_for_new_episode()
                self.current_obs, self.current_info = self.env.reset()
                self.current_hidden = self.actor_critic.get_initial_hidden(batch_size=1, device=self.device)
            else:
                self.current_obs = next_obs
                self.current_info = next_info
                self.current_hidden = next_hidden

            if self.total_timesteps - self.last_snapshot_step >= self.snapshot_interval:
                gen_idx = self.pool.size
                self.pool.add_policy(
                    self.actor_critic,
                    step=self.total_timesteps,
                    name=f"gen_{gen_idx:03d}_step_{self.total_timesteps}",
                )
                self.last_snapshot_step = self.total_timesteps

        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(self.current_obs).unsqueeze(0).to(device=self.device)
            last_val_tensor, _ = self.actor_critic.get_value(last_obs_tensor, self.current_hidden)
            last_value = float(last_val_tensor.item())

        rewards = np.array(self.rollout_buffer.rewards_buf[:self.rollout_buffer.size], dtype=np.float32)
        values = np.array(self.rollout_buffer.values_buf[:self.rollout_buffer.size], dtype=np.float32)
        dones = np.array(self.rollout_buffer.dones_buf[:self.rollout_buffer.size], dtype=bool)

        from src.rl.advantage import compute_gae

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


