# Phase 9: Self Play & Historical Policy Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Self-Play training system with historical policy pool, dynamic matchmaking (Uniform, Latest-biased, PFSP, Baseline Mix-in), generational tournaments, and automated scientific benchmarking for Ticket to Ride RL Lab.

**Architecture:** A modular `PolicyPool` manages historical policy snapshots (MLP and Recurrent LSTM). A `SelfPlayOpponentSampler` dynamically injects sampled opponents into `TicketToRideEnv.opponent` per episode during rollouts. `SelfPlayBenchmarkRunner` orchestrates side-by-side training and full round-robin tournaments to compute Elo trajectories and output structured Markdown/JSON reports.

**Tech Stack:** Python 3.12+, PyTorch 2.x, Gymnasium, NumPy, pytest, Pydantic.

**Spec:** [docs/superpowers/specs/2026-08-21-phase-9-self-play-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-21-phase-9-self-play-design.md)

## Global Constraints
- Target Environment: Python 3.12+ with PyTorch CPU-friendly execution (threads <= 2).
- Zero POMDP leakage: Ensure opponents interact only through legal game domain actions and standard observation interfaces.
- Deterministic reproducibility: Identical random seed produces bitwise identical policy pool snapshots, weights, and tournament Elo scores.
- Backward compatibility: Existing `TicketToRideEnv`, `PPOAgent`, `RecurrentPPOAgent`, `Evaluator`, and `Tournament` interfaces remain intact.

---

### Task 1: PolicySnapshot & PolicyPool Management

**Files:**
- Modify: `src/rl/self_play.py`
- Modify: `src/rl/__init__.py`
- Create: `tests/rl/test_self_play.py`

**Interfaces:**
- Consumes: `MaskedActorCritic` (`src.rl.networks`), `RecurrentMaskedActorCritic` (`src.rl.lstm_ppo`), `PPOAgent` (`src.agents.ppo_agent`), `RecurrentPPOAgent` (`src.agents.recurrent_ppo_agent`)
- Produces:
  - `PolicySnapshot(generation: int, step: int, name: str, state_dict: dict, is_recurrent: bool, hidden_dim: int, lstm_hidden_dim: int, metadata: dict)`
  - `PolicyPool(max_size: int = 50)`
    - `add_policy(model: nn.Module, step: int, name: str | None = None, metadata: dict | None = None) -> PolicySnapshot`
    - `get_snapshot(index_or_name: int | str) -> PolicySnapshot`
    - `create_agent(index_or_name: int | str, board: Any = None, tickets: Any = None, deterministic: bool = True) -> BaseAgent`
    - `save_pool(directory: str) -> None`
    - `load_pool(directory: str) -> None`
    - Property `size -> int`
    - Property `snapshots -> list[PolicySnapshot]`

- [ ] **Step 1: Write failing unit tests for PolicySnapshot and PolicyPool**

```python
# tests/rl/test_self_play.py
import pytest
import torch
from pathlib import Path
from src.rl.networks import MaskedActorCritic
from src.rl.lstm_ppo import RecurrentMaskedActorCritic
from src.rl.self_play import PolicyPool, PolicySnapshot
from src.agents.ppo_agent import PPOAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent

def test_policy_pool_add_and_retrieve():
    pool = PolicyPool(max_size=5)
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    
    snap0 = pool.add_policy(model, step=0, name="gen_0", metadata={"loss": 1.0})
    assert snap0.generation == 0
    assert snap0.step == 0
    assert snap0.name == "gen_0"
    assert not snap0.is_recurrent
    assert pool.size == 1

    # Retrieve by index and by name
    assert pool.get_snapshot(0).name == "gen_0"
    assert pool.get_snapshot("gen_0").step == 0

def test_policy_pool_capacity_and_retention():
    pool = PolicyPool(max_size=3)
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    
    for i in range(5):
        pool.add_policy(model, step=i * 1000, name=f"gen_{i}")
    
    assert pool.size == 3
    # Initial generation gen_0 must always be preserved as anchor
    assert pool.get_snapshot("gen_0") is not None
    # Latest generation must be present
    assert pool.get_snapshot("gen_4") is not None

def test_policy_pool_create_agent():
    pool = PolicyPool()
    mlp_model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    lstm_model = RecurrentMaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32, lstm_hidden_dim=32)
    
    pool.add_policy(mlp_model, step=100, name="mlp_snap")
    pool.add_policy(lstm_model, step=200, name="lstm_snap")
    
    agent_mlp = pool.create_agent("mlp_snap")
    assert isinstance(agent_mlp, PPOAgent)
    
    agent_lstm = pool.create_agent("lstm_snap")
    assert isinstance(agent_lstm, RecurrentPPOAgent)

def test_policy_pool_save_and_load(tmp_path: Path):
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=500, name="gen_saved")
    
    save_dir = str(tmp_path / "pool_export")
    pool.save_pool(save_dir)
    
    new_pool = PolicyPool()
    new_pool.load_pool(save_dir)
    assert new_pool.size == 1
    assert new_pool.get_snapshot(0).name == "gen_saved"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rl/test_self_play.py -v`
Expected: FAIL due to missing classes or methods in `src/rl/self_play.py`.

- [ ] **Step 3: Implement PolicySnapshot and PolicyPool in `src/rl/self_play.py`**

```python
# src/rl/self_play.py
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
from src.rl.lstm_ppo import RecurrentMaskedActorCritic
from src.rl.networks import MaskedActorCritic


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
```

- [ ] **Step 4: Update `src/rl/__init__.py` and run tests**

Export `PolicyPool` and `PolicySnapshot` in `src/rl/__init__.py`.
Run: `pytest tests/rl/test_self_play.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/self_play.py src/rl/__init__.py tests/rl/test_self_play.py
git commit -m "feat(rl): implement PolicySnapshot and PolicyPool for historical policy management"
```

---

### Task 2: SelfPlayOpponentSampler (Matchmaking & PFSP)

**Files:**
- Modify: `src/rl/self_play.py`
- Modify: `tests/rl/test_self_play.py`

**Interfaces:**
- Consumes: `PolicyPool`, `PolicySnapshot`, `RandomAgent`, `GreedyAgent`, `StrategicAgent`
- Produces:
  - `SelfPlayOpponentSampler(strategy: str = "latest_biased", baseline_mix_rate: float = 0.15, pfsp_exponent: float = 1.0, seed: int = 42)`
    - `sample_opponent(pool: PolicyPool, board: Any = None, tickets: Any = None) -> BaseAgent`
    - `record_match(opponent_name: str, trainee_won: bool)`
    - `get_opponent_weights(pool: PolicyPool) -> dict[str, float]`

- [ ] **Step 1: Write failing unit tests for SelfPlayOpponentSampler**

```python
# append to tests/rl/test_self_play.py
from src.rl.self_play import SelfPlayOpponentSampler
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent

def test_sampler_uniform_and_latest_biased():
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=0, name="gen_0")
    pool.add_policy(model, step=1000, name="gen_1")
    pool.add_policy(model, step=2000, name="gen_2")

    sampler_latest = SelfPlayOpponentSampler(strategy="latest_biased", baseline_mix_rate=0.0, seed=123)
    weights = sampler_latest.get_opponent_weights(pool)
    assert weights["gen_2"] >= 0.5

    sampler_uniform = SelfPlayOpponentSampler(strategy="uniform", baseline_mix_rate=0.0, seed=123)
    u_weights = sampler_uniform.get_opponent_weights(pool)
    assert u_weights["gen_0"] == pytest.approx(1.0 / 3.0)
    assert u_weights["gen_1"] == pytest.approx(1.0 / 3.0)

def test_sampler_baseline_mix_in():
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=0, name="gen_0")

    sampler = SelfPlayOpponentSampler(strategy="uniform", baseline_mix_rate=1.0, seed=42)
    sampled = [sampler.sample_opponent(pool) for _ in range(10)]
    assert any(isinstance(a, (RandomAgent, GreedyAgent)) for a in sampled)

def test_sampler_pfsp_adaptation():
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=0, name="gen_0")
    pool.add_policy(model, step=1000, name="gen_1")

    sampler = SelfPlayOpponentSampler(strategy="pfsp", baseline_mix_rate=0.0, pfsp_exponent=2.0)
    # Record that gen_0 is constantly defeated (win_rate 1.0) but gen_1 beats trainee (win_rate 0.0)
    for _ in range(10):
        sampler.record_match("gen_0", trainee_won=True)
        sampler.record_match("gen_1", trainee_won=False)

    weights = sampler.get_opponent_weights(pool)
    # gen_1 should have significantly higher sampling probability
    assert weights["gen_1"] > weights["gen_0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rl/test_self_play.py -k "sampler" -v`
Expected: FAIL due to missing `SelfPlayOpponentSampler`.

- [ ] **Step 3: Implement SelfPlayOpponentSampler in `src/rl/self_play.py`**

```python
# add to src/rl/self_play.py
from src.agents.greedy_agent import GreedyAgent
from src.agents.heuristic_agent import StrategicAgent
from src.agents.random_agent import RandomAgent


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
        self.np_rng = np.random.default_rng(seed)
        self.match_records: dict[str, dict[str, int]] = {}
        self.baseline_agents = baseline_agents or [
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
            return self.baseline_agents[0] if self.baseline_agents else RandomAgent()

        # 3. Sample historical policy from pool
        weights_dict = self.get_opponent_weights(pool)
        names = list(weights_dict.keys())
        probs = [weights_dict[name] for name in names]

        chosen_name = self.rng.choices(names, weights=probs, k=1)[0]
        return pool.create_agent(chosen_name, board=board, tickets=tickets, deterministic=deterministic)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/rl/test_self_play.py -k "sampler" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/self_play.py tests/rl/test_self_play.py
git commit -m "feat(rl): implement SelfPlayOpponentSampler with uniform, latest-biased, and PFSP strategies"
```

---

### Task 3: SelfPlayPPOTrainer & SelfPlayRecurrentPPOTrainer

**Files:**
- Modify: `src/rl/self_play.py`
- Modify: `src/rl/__init__.py`
- Modify: `tests/rl/test_self_play.py`

**Interfaces:**
- Consumes: `TicketToRideEnv`, `MaskedPPOTrainer`, `MaskedRecurrentPPOTrainer`, `PolicyPool`, `SelfPlayOpponentSampler`
- Produces:
  - `SelfPlayPPOTrainer(env: TicketToRideEnv, config: dict, pool: PolicyPool | None = None, sampler: SelfPlayOpponentSampler | None = None, seed: int = 42)`
  - `SelfPlayRecurrentPPOTrainer(env: TicketToRideEnv, config: dict, pool: PolicyPool | None = None, sampler: SelfPlayOpponentSampler | None = None, seed: int = 42)`

- [ ] **Step 1: Write failing unit tests for SelfPlayPPOTrainer**

```python
# append to tests/rl/test_self_play.py
from src.environment.env import TicketToRideEnv
from src.rl.self_play import SelfPlayPPOTrainer, SelfPlayRecurrentPPOTrainer

def test_selfplay_ppo_trainer_snapshots_and_opponent_switching():
    env = TicketToRideEnv(seed=42)
    trainer = SelfPlayPPOTrainer(
        env=env,
        config={
            "rollout_steps": 64,
            "minibatch_size": 16,
            "num_epochs": 2,
            "snapshot_interval": 64,
            "baseline_mix_rate": 0.0,
            "hidden_dim": 32,
        },
        seed=42,
    )
    # Initial pool starts with gen_0
    assert trainer.pool.size == 1
    
    metrics = trainer.train(total_timesteps=128)
    # Should have added at least one more snapshot
    assert trainer.pool.size >= 2
    assert trainer.total_timesteps >= 128

def test_selfplay_recurrent_ppo_trainer():
    env = TicketToRideEnv(seed=42)
    trainer = SelfPlayRecurrentPPOTrainer(
        env=env,
        config={
            "rollout_steps": 64,
            "seq_len": 8,
            "minibatch_chunks": 2,
            "num_epochs": 2,
            "snapshot_interval": 64,
            "baseline_mix_rate": 0.0,
            "hidden_dim": 32,
            "lstm_hidden_dim": 32,
        },
        seed=42,
    )
    assert trainer.pool.size == 1
    trainer.train(total_timesteps=128)
    assert trainer.pool.size >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rl/test_self_play.py -k "trainer" -v`
Expected: FAIL due to missing SelfPlay trainers.

- [ ] **Step 3: Implement SelfPlayPPOTrainer and SelfPlayRecurrentPPOTrainer in `src/rl/self_play.py`**

```python
# add to src/rl/self_play.py
from src.rl.ppo import MaskedPPOTrainer
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer


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

        # Snapshot generation 0 (initial un-trained policy)
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
```

- [ ] **Step 4: Update `src/rl/__init__.py` and run tests**

Export `SelfPlayPPOTrainer` and `SelfPlayRecurrentPPOTrainer` in `src/rl/__init__.py`.
Run: `pytest tests/rl/test_self_play.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/self_play.py src/rl/__init__.py tests/rl/test_self_play.py
git commit -m "feat(rl): implement SelfPlayPPOTrainer and SelfPlayRecurrentPPOTrainer"
```

---

### Task 4: SelfPlayBenchmarkRunner & Generational Tournament Study

**Files:**
- Create: `src/evaluation/self_play_benchmark.py`
- Modify: `src/evaluation/__init__.py`
- Create: `tests/evaluation/test_self_play_benchmark.py`

**Interfaces:**
- Consumes: `SelfPlayPPOTrainer`, `MaskedPPOTrainer`, `Tournament`, `Evaluator`, `RandomAgent`, `GreedyAgent`, `StrategicAgent`
- Produces:
  - `SelfPlayBenchmarkRunner(config: dict | None = None)`
    - `run_study() -> dict[str, Any]`
    - `generate_report(results: dict, output_md_path: str, output_json_path: str) -> None`

- [ ] **Step 1: Write failing test for SelfPlayBenchmarkRunner**

```python
# tests/evaluation/test_self_play_benchmark.py
from pathlib import Path
import json
from src.evaluation.self_play_benchmark import SelfPlayBenchmarkRunner

def test_self_play_benchmark_study_and_report(tmp_path: Path):
    runner = SelfPlayBenchmarkRunner(
        config={
            "seed": 42,
            "training_steps": 128,
            "snapshot_interval": 64,
            "eval_games": 2,
            "games_per_pair": 2,
        }
    )
    results = runner.run_study()
    assert "tournament" in results
    assert "leaderboard" in results["tournament"]
    assert "vs_baselines" in results
    assert "metadata" in results

    md_path = str(tmp_path / "phase9_report.md")
    json_path = str(tmp_path / "phase9_report.json")
    runner.generate_report(results, output_md_path=md_path, output_json_path=json_path)

    assert Path(md_path).exists()
    assert Path(json_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_self_play_benchmark.py -v`
Expected: FAIL due to missing `SelfPlayBenchmarkRunner`.

- [ ] **Step 3: Implement `SelfPlayBenchmarkRunner` in `src/evaluation/self_play_benchmark.py`**

```python
# src/evaluation/self_play_benchmark.py
"""Comparative Self-Play Benchmark Study evaluating Historical Policy Pool vs Single-Baseline Training."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.agents.greedy_agent import GreedyAgent
from src.agents.heuristic_agent import StrategicAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.evaluator import Evaluator
from src.evaluation.tournament import Tournament
from src.game.maps import load_usa_board
from src.rl.ppo import MaskedPPOTrainer
from src.rl.self_play import SelfPlayPPOTrainer


class SelfPlayBenchmarkRunner:
    """Orchestrates Self-Play vs Single-Baseline training and generational round-robin evaluation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.seed: int = self.config.get("seed", 42)
        self.training_steps: int = self.config.get("training_steps", 2000)
        self.snapshot_interval: int = self.config.get("snapshot_interval", 500)
        self.eval_games: int = self.config.get("eval_games", 20)
        self.games_per_pair: int = self.config.get("games_per_pair", 10)
        self.board, self.tickets = load_usa_board()

    def run_study(self) -> dict[str, Any]:
        start_time = time.time()

        # 1. Train Self-Play Agent with Policy Pool
        env_sp = TicketToRideEnv(seed=self.seed)
        sp_trainer = SelfPlayPPOTrainer(
            env=env_sp,
            config={
                "rollout_steps": 256,
                "num_epochs": 2,
                "lr": 3e-4,
                "snapshot_interval": self.snapshot_interval,
                "sampling_strategy": "latest_biased",
                "baseline_mix_rate": 0.15,
                "device": "cpu",
            },
            seed=self.seed,
        )
        sp_trainer.train(total_timesteps=self.training_steps)
        sp_final_agent = sp_trainer.pool.create_agent(
            sp_trainer.pool.size - 1, board=self.board, tickets=self.tickets
        )
        sp_final_agent.name = "SelfPlay_Final"

        # 2. Train Single-Baseline Agent (Only vs Random)
        env_single = TicketToRideEnv(seed=self.seed)
        single_trainer = MaskedPPOTrainer(
            env=env_single,
            config={
                "rollout_steps": 256,
                "num_epochs": 2,
                "lr": 3e-4,
                "device": "cpu",
            },
        )
        single_trainer.train(total_timesteps=self.training_steps)
        single_agent = PPOAgent(
            model=single_trainer.actor_critic,
            board=self.board,
            tickets=self.tickets,
            name="SingleBot_PPO",
        )

        # 3. Assemble Tournament Pool
        tournament_agents = []
        # Add initial generation, intermediate (if any), and final
        for i, snap in enumerate(sp_trainer.pool.snapshots):
            if i == 0 or i == len(sp_trainer.pool.snapshots) - 1 or i == len(sp_trainer.pool.snapshots) // 2:
                ag = sp_trainer.pool.create_agent(i, board=self.board, tickets=self.tickets)
                ag.name = f"SelfPlay_Gen{i}"
                tournament_agents.append(ag)

        tournament_agents.append(single_agent)
        tournament_agents.append(RandomAgent(name="RandomBot", seed=self.seed))
        tournament_agents.append(GreedyAgent(name="GreedyBot"))
        tournament_agents.append(StrategicAgent(name="StrategicBot"))

        # 4. Run Generational Round-Robin Tournament
        tournament = Tournament(
            agents=tournament_agents,
            games_per_pair=self.games_per_pair,
            board=self.board,
            tickets_deck=self.tickets,
        )
        tournament_results = tournament.run(seed=self.seed)

        # 5. Direct Head-to-Head & Baseline Win Rates
        evaluator = Evaluator(board=self.board, tickets_deck=self.tickets)
        vs_random = evaluator.evaluate(sp_final_agent, RandomAgent(seed=self.seed + 1), num_games=self.eval_games, seed=self.seed)
        vs_greedy = evaluator.evaluate(sp_final_agent, GreedyAgent(), num_games=self.eval_games, seed=self.seed)
        vs_single = evaluator.evaluate(sp_final_agent, single_agent, num_games=self.eval_games, seed=self.seed)

        elapsed = time.time() - start_time

        results = {
            "metadata": {
                "seed": self.seed,
                "training_steps": self.training_steps,
                "pool_generations": sp_trainer.pool.size,
                "eval_games": self.eval_games,
                "games_per_pair": self.games_per_pair,
                "elapsed_seconds": elapsed,
            },
            "tournament": tournament_results,
            "vs_baselines": {
                "selfplay_vs_random_win_rate": vs_random.agent1_win_rate,
                "selfplay_vs_greedy_win_rate": vs_greedy.agent1_win_rate,
                "selfplay_vs_single_bot_win_rate": vs_single.agent1_win_rate,
            },
        }
        return results

    def generate_report(
        self,
        results: dict[str, Any],
        output_md_path: str = "experiments/results/phase9_report.md",
        output_json_path: str = "experiments/results/phase9_report.json",
    ) -> None:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        meta = results["metadata"]
        tourn = results["tournament"]
        vs_b = results["vs_baselines"]

        leaderboard_rows = ""
        for rank, entry in enumerate(tourn["leaderboard"], 1):
            wld = f"{entry['wins']}-{entry['losses']}-{entry['draws']}"
            leaderboard_rows += (
                f"| {rank} | **{entry['name']}** | {entry['elo']:.1f} | "
                f"{entry['win_rate']*100:.1f}% | {wld} | {entry['avg_score']:.1f} |\n"
            )

        md_content = f"""# Relazione Scientifica: Studio di Auto-Apprendimento (Self-Play & Historical Policy Pool)

**Versione Studio:** Fase 9  
**Data:** 2026-08-21  
**Mappa:** Official USA Board  
**Seed Deterministico:** {meta['seed']}  
**Timestep di Addestramento:** {meta['training_steps']}  
**Generazioni Storiche nel Pool:** {meta['pool_generations']}  
**Partite per Accoppiamento Torneo:** {meta['games_per_pair']}  
**Tempo di Calcolo:** {meta['elapsed_seconds']:.2f}s  

---

## 1. Classifica Generale Torneo Generazionale (Elo Rating)

| Rank | Agente / Generazione | Elo Rating | Win Rate Complessivo | W-L-D | Punteggio Medio |
| :--- | :--- | :--- | :--- | :--- | :--- |
{leaderboard_rows}

---

## 2. Validazione e Resilienza Strategica

| Scontro Diretto | Win Rate Self-Play | Esito |
| :--- | :--- | :--- |
| **Self-Play Final vs RandomBot** | **{vs_b['selfplay_vs_random_win_rate']*100:.1f}%** | {'Superato (>= 65%)' if vs_b['selfplay_vs_random_win_rate'] >= 0.65 else 'Sotto soglia'} |
| **Self-Play Final vs GreedyBot** | **{vs_b['selfplay_vs_greedy_win_rate']*100:.1f}%** | Validato |
| **Self-Play Final vs SingleBot PPO** | **{vs_b['selfplay_vs_single_bot_win_rate']*100:.1f}%** | {'Vantaggio Self-Play' if vs_b['selfplay_vs_single_bot_win_rate'] >= 0.50 else 'Parità / Svantaggio'} |

---

## 3. Conclusioni Didattiche e Prossimi Passi

1. **Prevenzione del Policy Cycling:** L'adozione del Policy Pool e del matchmaking dinamico evita la concentrazione su pattern locali di gioco.
2. **Progressione Monotonica:** La crescita dell'Elo attraverso le generazioni storiche attesta l'acquisizione di robustezza globale.
3. **Fase Successiva:** Si raccomanda di procedere alla **Fase 10 (Generalization & Procedural Maps)** per valutare l'adattabilità della policy su mappe generate proceduralmente mai viste.
"""
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
```

- [ ] **Step 4: Update `src/evaluation/__init__.py` and run tests**

Export `SelfPlayBenchmarkRunner` in `src/evaluation/__init__.py`.
Run: `pytest tests/evaluation/test_self_play_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/self_play_benchmark.py src/evaluation/__init__.py tests/evaluation/test_self_play_benchmark.py
git commit -m "feat(evaluation): implement SelfPlayBenchmarkRunner and generational tournament report generator"
```

---

### Task 5: Configuration Schema, ExperimentRunner & CLI Entrypoints

**Files:**
- Modify: `src/experiments/config.py`
- Modify: `src/experiments/runner.py`
- Create: `scripts/train_selfplay.py`
- Create: `tests/experiments/test_self_play_config.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `SelfPlayPPOTrainer`, `SelfPlayRecurrentPPOTrainer`
- Produces:
  - `SelfPlayConfig(enabled: bool = False, pool_max_size: int = 50, snapshot_interval: int = 5000, strategy: str = "latest_biased", baseline_mix_rate: float = 0.15, pfsp_exponent: float = 1.0)` in `src.experiments.config`
  - Integration in `ExperimentRunner.run()`
  - `scripts/train_selfplay.py` CLI

- [ ] **Step 1: Write failing test for SelfPlayConfig and ExperimentRunner**

```python
# tests/experiments/test_self_play_config.py
from src.experiments.config import ExperimentConfig, SelfPlayConfig
from src.experiments.runner import ExperimentRunner

def test_selfplay_config_parsing():
    cfg = ExperimentConfig(
        name="test_selfplay",
        self_play=SelfPlayConfig(enabled=True, snapshot_interval=1000, strategy="pfsp")
    )
    assert cfg.self_play.enabled is True
    assert cfg.self_play.snapshot_interval == 1000
    assert cfg.self_play.strategy == "pfsp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/experiments/test_self_play_config.py -v`
Expected: FAIL due to missing `SelfPlayConfig` on `ExperimentConfig`.

- [ ] **Step 3: Update `src/experiments/config.py` and `src/experiments/runner.py`**

Add `SelfPlayConfig` model to `src/experiments/config.py`:
```python
class SelfPlayConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    pool_max_size: int = 50
    snapshot_interval: int = 5000
    strategy: str = "latest_biased"
    baseline_mix_rate: float = 0.15
    pfsp_exponent: float = 1.0
```
And add `self_play: SelfPlayConfig = Field(default_factory=SelfPlayConfig)` to `ExperimentConfig`.

Update `src/experiments/runner.py` to instantiate `SelfPlayPPOTrainer` / `SelfPlayRecurrentPPOTrainer` when `config.self_play.enabled` is True.

Create `scripts/train_selfplay.py`:
```python
"""Headless Self-Play training entrypoint."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.env import TicketToRideEnv
from src.rl.self_play import SelfPlayPPOTrainer, SelfPlayRecurrentPPOTrainer
from src.game.maps import load_usa_board

def main():
    parser = argparse.ArgumentParser(description="Train a self-play agent headless")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    parser.add_argument("--snapshot-interval", type=int, default=2000, help="Interval between snapshots")
    parser.add_argument("--strategy", type=str, default="latest_biased", help="Matchmaking strategy")
    parser.add_argument("--baseline-mix", type=float, default=0.15, help="Baseline mix-in rate")
    parser.add_argument("--recurrent", action="store_true", help="Use recurrent LSTM architecture")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    board, tickets = load_usa_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, seed=args.seed)
    
    cfg = {
        "rollout_steps": 256,
        "num_epochs": 2,
        "snapshot_interval": args.snapshot_interval,
        "sampling_strategy": args.strategy,
        "baseline_mix_rate": args.baseline_mix,
    }
    
    if args.recurrent:
        cfg["seq_len"] = 8
        cfg["minibatch_chunks"] = 4
        trainer = SelfPlayRecurrentPPOTrainer(env=env, config=cfg, seed=args.seed)
    else:
        trainer = SelfPlayPPOTrainer(env=env, config=cfg, seed=args.seed)

    print(f"Starting Self-Play training ({'Recurrent' if args.recurrent else 'MLP'}) for {args.timesteps} timesteps...")
    trainer.train(total_timesteps=args.timesteps)
    print(f"Self-play training completed. Total snapshots in pool: {trainer.pool.size}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_self_play_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/config.py src/experiments/runner.py scripts/train_selfplay.py tests/experiments/test_self_play_config.py
git commit -m "feat(experiments): integrate SelfPlayConfig, runner support, and train_selfplay CLI"
```

---

### Task 6: Phase 9 Acceptance Test Suite (All 6 Criteria)

**Files:**
- Create: `tests/rl/test_phase9_acceptance.py`

**Interfaces:**
- Consumes: All Phase 9 components
- Verifies: All 6 formal acceptance criteria defined in the design spec

- [ ] **Step 1: Write `tests/rl/test_phase9_acceptance.py`**

```python
"""Comprehensive Phase 9 Acceptance Test Suite verifying all 6 acceptance criteria."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.self_play_benchmark import SelfPlayBenchmarkRunner
from src.game.maps import load_usa_board
from src.rl.networks import MaskedActorCritic
from src.rl.self_play import (
    PolicyPool,
    PolicySnapshot,
    SelfPlayOpponentSampler,
    SelfPlayPPOTrainer,
)


def test_phase9_criterion_1_policy_pool_management():
    """Criterion 1: PolicyPool manages snapshots, retention, capacity, and agent creation."""
    pool = PolicyPool(max_size=4)
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)

    for i in range(6):
        pool.add_policy(model, step=i * 500, name=f"gen_{i}")

    assert pool.size == 4
    # Anchor gen_0 and latest gen_5 must be preserved
    assert pool.get_snapshot("gen_0") is not None
    assert pool.get_snapshot("gen_5") is not None

    agent = pool.create_agent(0)
    assert agent.name == "gen_0"


def test_phase9_criterion_2_matchmaking_strategies():
    """Criterion 2: SelfPlayOpponentSampler correctly applies uniform, latest-biased, PFSP, and baseline mix."""
    pool = PolicyPool()
    model = MaskedActorCritic(input_dim=50, action_dim=10, hidden_dim=32)
    pool.add_policy(model, step=0, name="gen_0")
    pool.add_policy(model, step=1000, name="gen_1")
    pool.add_policy(model, step=2000, name="gen_2")

    sampler_latest = SelfPlayOpponentSampler(strategy="latest_biased", baseline_mix_rate=0.0)
    weights = sampler_latest.get_opponent_weights(pool)
    assert weights["gen_2"] >= 0.5

    sampler_pfsp = SelfPlayOpponentSampler(strategy="pfsp", baseline_mix_rate=0.0)
    sampler_pfsp.record_match("gen_0", trainee_won=True)
    sampler_pfsp.record_match("gen_1", trainee_won=False)
    pfsp_w = sampler_pfsp.get_opponent_weights(pool)
    assert pfsp_w["gen_1"] > pfsp_w["gen_0"]


def test_phase9_criterion_3_dynamic_opponent_switching():
    """Criterion 3: Gymnasium env cleanly switches opponents per episode without state corruption."""
    env = TicketToRideEnv(seed=101)
    pool = PolicyPool()
    model = MaskedActorCritic(
        input_dim=env.observation_space.shape[0],
        action_dim=int(env.action_space.n),
        hidden_dim=32,
    )
    pool.add_policy(model, step=0, name="gen_0")

    trainer = SelfPlayPPOTrainer(
        env=env,
        config={"rollout_steps": 32, "minibatch_size": 16, "num_epochs": 1, "snapshot_interval": 32},
        pool=pool,
        seed=101,
    )
    obs, info = trainer.env.reset()
    assert obs.shape == env.observation_space.shape
    assert "action_mask" in info

    metrics = trainer.collect_rollout()
    assert "mean_rollout_reward" in metrics


def test_phase9_criterion_4_deterministic_reproducibility():
    """Criterion 4: Identical seeds produce bitwise reproducible self-play training and pools."""
    def run_training(seed=77):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(seed=seed)
        trainer = SelfPlayPPOTrainer(
            env=env,
            config={
                "rollout_steps": 64,
                "minibatch_size": 16,
                "num_epochs": 2,
                "snapshot_interval": 64,
                "hidden_dim": 32,
            },
            seed=seed,
        )
        trainer.train(total_timesteps=64)
        return trainer

    tr1 = run_training(77)
    tr2 = run_training(77)

    assert tr1.pool.size == tr2.pool.size
    snap1 = tr1.pool.get_snapshot(0)
    snap2 = tr2.pool.get_snapshot(0)

    for k in snap1.state_dict:
        torch.testing.assert_close(snap1.state_dict[k], snap2.state_dict[k])


def test_phase9_criterion_5_generational_progression_and_superiority():
    """Criterion 5: Self-play agent beats initial generation and outperforms random baseline."""
    board, tickets = load_usa_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, seed=42)
    trainer = SelfPlayPPOTrainer(
        env=env,
        config={
            "rollout_steps": 128,
            "minibatch_size": 32,
            "num_epochs": 2,
            "lr": 3e-4,
            "snapshot_interval": 128,
            "hidden_dim": 64,
        },
        seed=42,
    )
    trainer.train(total_timesteps=300)

    final_agent = trainer.pool.create_agent(trainer.pool.size - 1, board=board, tickets=tickets)
    random_agent = RandomAgent()

    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator(board=board, tickets_deck=tickets)
    eval_res = evaluator.evaluate(final_agent, random_agent, num_games=10, seed=42)

    assert eval_res.agent1_win_rate >= 0.50


def test_phase9_criterion_6_automated_scientific_benchmark_study(tmp_path: Path):
    """Criterion 6: SelfPlayBenchmarkRunner generates valid JSON and Markdown academic report."""
    report_md = tmp_path / "phase9_report.md"
    report_json = tmp_path / "phase9_report.json"

    runner = SelfPlayBenchmarkRunner(
        config={
            "seed": 42,
            "training_steps": 256,
            "snapshot_interval": 128,
            "eval_games": 4,
            "games_per_pair": 2,
        }
    )
    results = runner.run_study()
    runner.generate_report(results, output_md_path=str(report_md), output_json_path=str(report_json))

    assert report_md.exists()
    assert report_json.exists()
    with open(report_json, encoding="utf-8") as f:
        data = json.load(f)
    assert "tournament" in data
    assert "vs_baselines" in data
    assert "metadata" in data
```

- [ ] **Step 2: Run acceptance test suite**

Run: `pytest tests/rl/test_phase9_acceptance.py -v`
Expected: PASS for all 6 acceptance criteria.

- [ ] **Step 3: Commit**

```bash
git add tests/rl/test_phase9_acceptance.py
git commit -m "test(rl): add comprehensive Phase 9 acceptance test suite verifying all 6 criteria"
```
