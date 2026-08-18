# Phase 4: First RL (DQN & PPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement native, pedagogical PyTorch Reinforcement Learning engines from scratch for Masked Double-DQN and Masked PPO, inference agents, multi-opponent periodic evaluation against Random, Greedy, and Strategic baselines, headless training runners, and verify that trained RL agents consistently outperform Random baselines with win rate $\ge 75\%$.

**Architecture:** Educational PyTorch implementation without external black boxes. `MaskedQNetwork` and `MaskedActorCritic` in `src/rl/networks.py` apply exact mathematical action masking prior to argmax/softmax. `ReplayBuffer` and `RolloutBuffer` store transitions with action masks. `MaskedDQNTrainer` implements Double DQN with Target Network. `MaskedPPOTrainer` implements on-policy GAE rollouts, clipped surrogate objective, entropy bonus, and diagnostics. `DQNAgent` and `PPOAgent` wrap models for standalone tournament play and `.pt` checkpointing. `MultiOpponentEvaluator` and `ExperimentRunner` orchestrate training and benchmarks.

**Tech Stack:** Python 3.12+, PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.distributions`), Gymnasium, NumPy, PyYAML.

**Spec:** `docs/superpowers/specs/2026-08-18-phase-4-first-rl-design.md`

## Global Constraints
- Pure PyTorch implementation with zero black-box training libraries for core algorithms.
- Strict Action Masking: invalid domain actions must have $-\infty$ in Q-values and exact $0.0$ probability in Actor-Critic.
- Deterministic reproducibility: same seed $\implies$ identical weights, transitions, and metrics.
- Periodic evaluation against all specified baselines (`RandomAgent`, `GreedyAgent`, `StrategicAgent`).
- All tests must pass with `.venv/bin/pytest`.

---

### Task 1: Neural Networks with Action Masking (`src/rl/networks.py`)

**Files:**
- Modify: `src/rl/networks.py`
- Create: `tests/rl/test_networks.py`

**Interfaces:**
- Produces:
  - `MaskedQNetwork(input_dim: int, action_dim: int, hidden_dim: int = 128)`:
    - `forward(obs: torch.Tensor) -> torch.Tensor`
    - `select_action(obs: torch.Tensor, action_mask: np.ndarray, epsilon: float = 0.0) -> int`
  - `MaskedActorCritic(input_dim: int, action_dim: int, hidden_dim: int = 128)`:
    - `forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`
    - `get_action_and_value(obs: torch.Tensor, action_mask: torch.Tensor | None = None, action: torch.Tensor | None = None, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing unit tests for `MaskedQNetwork` and `MaskedActorCritic`**

Create `tests/rl/test_networks.py`:
```python
import numpy as np
import torch

from src.rl.networks import MaskedActorCritic, MaskedQNetwork


def test_masked_q_network_forward_and_action_selection() -> None:
    input_dim = 20
    action_dim = 5
    q_net = MaskedQNetwork(input_dim=input_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn(input_dim)
    mask = np.array([1, 0, 1, 0, 0], dtype=np.int8)

    # Deterministic greedy action selection (epsilon=0.0)
    action = q_net.select_action(obs, mask, epsilon=0.0)
    assert action in [0, 2]

    # Random exploration with mask (epsilon=1.0)
    actions = [q_net.select_action(obs, mask, epsilon=1.0) for _ in range(50)]
    assert all(a in [0, 2] for a in actions)
    assert 0 in actions and 2 in actions


def test_masked_actor_critic_distribution_and_masking() -> None:
    input_dim = 20
    action_dim = 6
    ac = MaskedActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn(2, input_dim)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]], dtype=torch.bool)

    actions, log_probs, entropy, values = ac.get_action_and_value(obs, action_mask=mask)

    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert entropy.shape == (2,)
    assert values.shape == (2, 1)

    assert actions[0].item() in [0, 2]
    assert actions[1].item() in [1, 3]

    # Test evaluated log_prob on given action
    given_actions = torch.tensor([0, 1])
    _, eval_log_probs, _, _ = ac.get_action_and_value(obs, action_mask=mask, action=given_actions)
    assert torch.all(torch.isfinite(eval_log_probs))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/rl/test_networks.py -v`
Expected: FAIL (missing methods/attributes)

- [ ] **Step 3: Implement `MaskedQNetwork` and `MaskedActorCritic` in `src/rl/networks.py`**

Edit `src/rl/networks.py`:
```python
"""PyTorch Neural Network architectures for DQN and PPO with Action Masking."""

import random
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


class MaskedQNetwork(nn.Module):
    """Deep Q-Network with explicit Action Masking and epsilon-greedy exploration."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unmasked Q-values for all actions."""
        return self.net(x)

    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        """Select an action using epsilon-greedy strategy over legally valid actions."""
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            return 0

        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            q_values = self.forward(obs).squeeze(0).clone()
            # Mask invalid actions with -infinity
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=q_values.device)
            q_values[~mask_tensor] = -1e9
            return int(torch.argmax(q_values).item())


class MaskedActorCritic(nn.Module):
    """Discrete Actor-Critic architecture with Action Masking for PPO."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action sampling, log-probabilities, entropy, and state value."""
        logits = self.actor(obs)
        value = self.critic(obs)

        if action_mask is not None:
            # Mask out invalid actions with large negative value before softmax
            masked_logits = torch.where(
                action_mask,
                logits,
                torch.tensor(-1e8, dtype=logits.dtype, device=logits.device),
            )
        else:
            masked_logits = logits

        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/rl/test_networks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/networks.py tests/rl/test_networks.py
git commit -m "feat: implement MaskedQNetwork and MaskedActorCritic with action masking"
```

---

### Task 2: Experience Buffers with Action Masking (`src/rl/replay_buffer.py` & `src/rl/rollout.py`)

**Files:**
- Modify: `src/rl/replay_buffer.py`
- Modify: `src/rl/rollout.py`
- Create: `tests/rl/test_buffers.py`

**Interfaces:**
- Produces:
  - `ReplayBuffer`:
    - `push(obs, action, reward, next_obs, done, next_action_mask)`
    - `sample(batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]`
  - `RolloutBuffer`:
    - `add(obs, action, reward, value, log_prob, done, action_mask)`
    - `generate_minibatches(batch_size: int, advantages: np.ndarray, returns: np.ndarray)`

- [ ] **Step 1: Write failing unit tests for ReplayBuffer and RolloutBuffer**

Create `tests/rl/test_buffers.py`:
```python
import numpy as np
import torch

from src.rl.replay_buffer import ReplayBuffer
from src.rl.rollout import RolloutBuffer


def test_replay_buffer_push_sample_tensors() -> None:
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        obs = np.ones(5, dtype=np.float32) * i
        action = i % 3
        reward = float(i)
        next_obs = np.ones(5, dtype=np.float32) * (i + 1)
        done = i == 9
        next_mask = np.array([1, 0, 1], dtype=np.int8)
        buffer.push(obs, action, reward, next_obs, done, next_mask)

    assert len(buffer) == 10
    batch = buffer.sample(batch_size=4)
    assert batch["obs"].shape == (4, 5)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["next_obs"].shape == (4, 5)
    assert batch["dones"].shape == (4,)
    assert batch["next_action_masks"].shape == (4, 3)
    assert batch["next_action_masks"].dtype == torch.bool


def test_rollout_buffer_minibatch_generator() -> None:
    rollout = RolloutBuffer()
    for i in range(16):
        rollout.add(
            obs=np.ones(4, dtype=np.float32) * i,
            action=i % 2,
            reward=1.0,
            value=0.5,
            log_prob=-0.69,
            done=False,
            action_mask=np.array([1, 1], dtype=np.int8),
        )

    advantages = np.ones(16, dtype=np.float32) * 2.0
    returns = np.ones(16, dtype=np.float32) * 3.0

    minibatches = list(rollout.generate_minibatches(batch_size=4, advantages=advantages, returns=returns))
    assert len(minibatches) == 4
    for mb in minibatches:
        assert mb["obs"].shape == (4, 4)
        assert mb["actions"].shape == (4,)
        assert mb["advantages"].shape == (4,)
        assert mb["returns"].shape == (4,)
        assert mb["action_masks"].shape == (4, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/rl/test_buffers.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `ReplayBuffer` and `RolloutBuffer`**

Edit `src/rl/replay_buffer.py`:
```python
"""Experience Replay Buffer for DQN with Action Masking."""

import random
from collections import deque
from typing import Any

import numpy as np
import torch


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
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done, next_action_mask))

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)

        return {
            "obs": torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            "actions": torch.tensor(actions, dtype=torch.int64, device=device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_obs": torch.tensor(np.array(next_obs), dtype=torch.float32, device=device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device),
            "next_action_masks": torch.tensor(np.array(next_masks), dtype=torch.bool, device=device),
        }

    def __len__(self) -> int:
        return len(self.buffer)
```

Edit `src/rl/rollout.py`:
```python
"""Rollout buffer for on-policy PPO trajectories with Action Masking."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """On-policy trajectory storage with mini-batch generation."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)

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
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.action_masks.append(np.asarray(action_mask, dtype=bool))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.action_masks.clear()

    def generate_minibatches(
        self,
        batch_size: int,
        advantages: np.ndarray,
        returns: np.ndarray,
        device: str = "cpu",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield randomized minibatches for multi-epoch PPO optimization."""
        total_steps = len(self.observations)
        indices = np.random.permutation(total_steps)

        obs_arr = np.array(self.observations, dtype=np.float32)
        act_arr = np.array(self.actions, dtype=np.int64)
        lp_arr = np.array(self.log_probs, dtype=np.float32)
        mask_arr = np.array(self.action_masks, dtype=bool)

        for start in range(0, total_steps, batch_size):
            mb_indices = indices[start : start + batch_size]
            yield {
                "obs": torch.tensor(obs_arr[mb_indices], dtype=torch.float32, device=device),
                "actions": torch.tensor(act_arr[mb_indices], dtype=torch.int64, device=device),
                "old_log_probs": torch.tensor(lp_arr[mb_indices], dtype=torch.float32, device=device),
                "advantages": torch.tensor(advantages[mb_indices], dtype=torch.float32, device=device),
                "returns": torch.tensor(returns[mb_indices], dtype=torch.float32, device=device),
                "action_masks": torch.tensor(mask_arr[mb_indices], dtype=torch.bool, device=device),
            }

    def __len__(self) -> int:
        return len(self.observations)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/rl/test_buffers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/replay_buffer.py src/rl/rollout.py tests/rl/test_buffers.py
git commit -m "feat: implement ReplayBuffer and RolloutBuffer with action masking and minibatch generation"
```

---

### Task 3: Generalized Advantage Estimation (`src/rl/advantage.py`)

**Files:**
- Modify: `src/rl/advantage.py`
- Create: `tests/rl/test_advantage.py`

**Interfaces:**
- Produces:
  - `compute_gae(rewards: list[float] | np.ndarray, values: list[float] | np.ndarray, dones: list[bool] | np.ndarray, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write failing unit test for GAE and returns computation**

Create `tests/rl/test_advantage.py`:
```python
import numpy as np

from src.rl.advantage import compute_gae


def test_compute_gae_simple_trajectory() -> None:
    rewards = [1.0, 1.0, 2.0]
    values = [0.5, 0.8, 1.0]
    dones = [False, False, True]
    next_value = 0.0

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert len(advantages) == 3
    assert len(returns) == 3
    # Step 2: terminal step (done=True), delta = 2.0 + 0 - 1.0 = 1.0
    assert np.isclose(advantages[2], 1.0, atol=1e-4)
    assert np.isclose(returns[2], 2.0, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes/fails**

Run: `.venv/bin/pytest tests/rl/test_advantage.py -v`

- [ ] **Step 3: Ensure robust GAE implementation in `src/rl/advantage.py`**

Edit `src/rl/advantage.py`:
```python
"""Generalized Advantage Estimation (GAE) for Actor-Critic methods."""

from collections.abc import Sequence

import numpy as np


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and Discounted Returns.

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
    Returns_t = A_t + V(s_t)
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        v_next = next_value if t == n_steps - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * v_next * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/rl/test_advantage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/advantage.py tests/rl/test_advantage.py
git commit -m "feat: implement robust Generalized Advantage Estimation (GAE)"
```

---

### Task 4: Masked Double-DQN Trainer (`src/rl/dqn.py`)

**Files:**
- Modify: `src/rl/dqn.py`
- Create: `tests/rl/test_dqn_trainer.py`

**Interfaces:**
- Consumes: `MaskedQNetwork`, `ReplayBuffer`, `TicketToRideEnv`
- Produces:
  - `MaskedDQNTrainer(env: TicketToRideEnv, config: dict[str, Any])`:
    - `train_step() -> dict[str, float]`
    - `step(epsilon: float) -> tuple[float, bool]`
    - `train(total_timesteps: int, eval_callback=None) -> dict[str, Any]`
    - `save(path: str)`
    - `load(path: str)`

- [ ] **Step 1: Write failing unit test for `MaskedDQNTrainer`**

Create `tests/rl/test_dqn_trainer.py`:
```python
from src.environment.env import TicketToRideEnv
from src.game.board import create_synthetic_mini_board
from src.rl.dqn import MaskedDQNTrainer


def test_dqn_trainer_initialization_and_train_step() -> None:
    board = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, opponent_policy="random", seed=42)

    config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "buffer_size": 1000,
        "batch_size": 16,
        "target_update_freq": 100,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 1000,
        "learning_starts": 10,
    }

    trainer = MaskedDQNTrainer(env=env, config=config)

    # Populate buffer with initial steps
    for _ in range(20):
        trainer.step(epsilon=1.0)

    assert len(trainer.replay_buffer) >= 20

    # Execute training step
    metrics = trainer.train_step()
    assert "loss" in metrics
    assert "q_mean" in metrics
    assert metrics["loss"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/rl/test_dqn_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `MaskedDQNTrainer` in `src/rl/dqn.py`**

Edit `src/rl/dqn.py`:
```python
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
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)

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

        obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32, device=self.device)
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
            return {"loss": 0.0, "q_mean": 0.0}

        batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        next_masks = batch["next_action_masks"]

        # Current Q(s, a)
        q_values = self.policy_net(obs)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target:
        # 1. Best action chosen by policy_net over valid next actions
        with torch.no_grad():
            next_q_policy = self.policy_net(next_obs).clone()
            next_q_policy[~next_masks] = -1e9
            best_next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)

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

        return {"loss": float(loss.item()), "q_mean": float(state_action_values.mean().item())}

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/rl/test_dqn_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/dqn.py tests/rl/test_dqn_trainer.py
git commit -m "feat: implement MaskedDQNTrainer with Double DQN and target updates"
```

---

### Task 5: Masked PPO Trainer (`src/rl/ppo.py`)

**Files:**
- Modify: `src/rl/ppo.py`
- Create: `tests/rl/test_ppo_trainer.py`

**Interfaces:**
- Consumes: `MaskedActorCritic`, `RolloutBuffer`, `compute_gae`, `TicketToRideEnv`
- Produces:
  - `MaskedPPOTrainer(env: TicketToRideEnv, config: dict[str, Any])`:
    - `collect_rollout() -> dict[str, float]`
    - `train_epoch() -> dict[str, float]`
    - `train(total_timesteps: int, eval_callback=None) -> dict[str, Any]`
    - `save(path: str)`
    - `load(path: str)`

- [ ] **Step 1: Write failing unit test for `MaskedPPOTrainer`**

Create `tests/rl/test_ppo_trainer.py`:
```python
from src.environment.env import TicketToRideEnv
from src.game.board import create_synthetic_mini_board
from src.rl.ppo import MaskedPPOTrainer


def test_ppo_trainer_rollout_and_train_epoch() -> None:
    board = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, opponent_policy="random", seed=42)

    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rollout_steps": 64,
        "num_epochs": 2,
        "minibatch_size": 16,
    }

    trainer = MaskedPPOTrainer(env=env, config=config)

    rollout_metrics = trainer.collect_rollout()
    assert len(trainer.rollout_buffer) == 64

    train_metrics = trainer.train_epoch()
    assert "policy_loss" in train_metrics
    assert "value_loss" in train_metrics
    assert "entropy" in train_metrics
    assert "approx_kl" in train_metrics
    assert "explained_var" in train_metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/rl/test_ppo_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `MaskedPPOTrainer` in `src/rl/ppo.py`**

Edit `src/rl/ppo.py`:
```python
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
        self.rollout_buffer = RolloutBuffer()

        self.total_timesteps = 0
        self.current_obs, self.current_info = self.env.reset()

    def collect_rollout(self) -> dict[str, float]:
        """Collect rollout_steps transitions using current policy."""
        self.rollout_buffer.clear()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0

        for _ in range(self.rollout_steps):
            obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.tensor(self.current_info["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)

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
            obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_val = self.actor_critic(obs_tensor)
            next_value = float(next_val.item())

        advantages, returns = compute_gae(
            rewards=self.rollout_buffer.rewards,
            values=self.rollout_buffer.values,
            dones=self.rollout_buffer.dones,
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
        y_pred = np.array(self.rollout_buffer.values, dtype=np.float32)
        var_y = np.var(y_true)
        explained_var = float(np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y)

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "explained_var": explained_var,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/rl/test_ppo_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/ppo.py tests/rl/test_ppo_trainer.py
git commit -m "feat: implement MaskedPPOTrainer with GAE, clipped surrogate objective and diagnostics"
```

---

### Task 6: Inference RL Agents (`src/agents/dqn_agent.py` & `src/agents/ppo_agent.py`)

**Files:**
- Modify: `src/agents/dqn_agent.py`
- Modify: `src/agents/ppo_agent.py`
- Create: `tests/agents/test_rl_agents.py`

**Interfaces:**
- Produces:
  - `DQNAgent(name: str = "DQNAgent", model_path: str | None = None, input_dim: int = 20, action_dim: int = 10)`
  - `PPOAgent(name: str = "PPOAgent", model_path: str | None = None, input_dim: int = 20, action_dim: int = 10)`
  - Both support `select_action(observation, action_mask, info) -> int`, `load(path)`, `save(path)`

- [ ] **Step 1: Write failing unit tests for `DQNAgent` and `PPOAgent`**

Create `tests/agents/test_rl_agents.py`:
```python
import numpy as np
import pytest

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent


def test_dqn_agent_inference_and_masking() -> None:
    agent = DQNAgent(name="TestDQN", input_dim=10, action_dim=5)
    obs = np.random.randn(10).astype(np.float32)
    mask = np.array([0, 1, 0, 1, 0], dtype=np.int8)

    action = agent.select_action(obs, action_mask=mask)
    assert action in [1, 3]


def test_ppo_agent_inference_and_masking() -> None:
    agent = PPOAgent(name="TestPPO", input_dim=10, action_dim=5)
    obs = np.random.randn(10).astype(np.float32)
    mask = np.array([1, 0, 0, 0, 0], dtype=np.int8)

    action = agent.select_action(obs, action_mask=mask)
    assert action == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agents/test_rl_agents.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `DQNAgent` and `PPOAgent`**

Edit `src/agents/dqn_agent.py`:
```python
"""DQN Agent for inference and evaluation."""

from typing import Any

import numpy as np
import torch

from src.agents.base_agent import BaseAgent
from src.rl.networks import MaskedQNetwork


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent."""

    def __init__(
        self,
        name: str = "DQNAgent",
        model_path: str | None = None,
        input_dim: int = 100,
        action_dim: int = 56,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__(name=name)
        self.device = device
        self.q_net = MaskedQNetwork(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.q_net.eval()
        if model_path is not None:
            self.load(model_path)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        if action_mask is None:
            action_mask = np.ones(self.q_net.net[-1].out_features, dtype=np.int8)

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        return self.q_net.select_action(obs_tensor, action_mask=action_mask, epsilon=0.0)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if "policy_state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["policy_state_dict"])
        else:
            self.q_net.load_state_dict(checkpoint)
        self.q_net.eval()
```

Edit `src/agents/ppo_agent.py`:
```python
"""PPO Agent for inference and evaluation."""

from typing import Any

import numpy as np
import torch

from src.agents.base_agent import BaseAgent
from src.rl.networks import MaskedActorCritic


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent."""

    def __init__(
        self,
        name: str = "PPOAgent",
        model_path: str | None = None,
        input_dim: int = 100,
        action_dim: int = 56,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__(name=name)
        self.device = device
        self.actor_critic = MaskedActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.actor_critic.eval()
        if model_path is not None:
            self.load(model_path)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = (
            torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            if action_mask is not None
            else None
        )

        with torch.no_grad():
            action, _, _, _ = self.actor_critic.get_action_and_value(
                obs_tensor, action_mask=mask_tensor, deterministic=True
            )
        return int(action.item())

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if "actor_critic_state_dict" in checkpoint:
            self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        else:
            self.actor_critic.load_state_dict(checkpoint)
        self.actor_critic.eval()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/agents/test_rl_agents.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/dqn_agent.py src/agents/ppo_agent.py tests/agents/test_rl_agents.py
git commit -m "feat: implement DQNAgent and PPOAgent inference classes with checkpoint loading"
```

---

### Task 7: Multi-Opponent Periodic Evaluator, YAML Configs, and Experiment Runner (`src/experiments/`)

**Files:**
- Create: `src/experiments/evaluator.py`
- Modify: `src/experiments/runner.py`
- Modify: `src/experiments/config.py`
- Create: `experiments/configs/dqn_mini.yaml`
- Create: `experiments/configs/ppo_mini.yaml`
- Create: `experiments/configs/ppo_usa.yaml`
- Create: `tests/experiments/test_rl_runner.py`

**Interfaces:**
- Produces:
  - `MultiOpponentEvaluator.evaluate(agent: BaseAgent, opponents: list[str], games_per_opponent: int, board) -> dict[str, float]`
  - `ExperimentRunner.run() -> ExperimentRecord` with full training loop, evaluation callbacks, and checkpointing.

- [ ] **Step 1: Write failing unit test for `MultiOpponentEvaluator` and `ExperimentRunner`**

Create `tests/experiments/test_rl_runner.py`:
```python
from src.agents.random_agent import RandomAgent
from src.experiments.config import ExperimentConfig
from src.experiments.evaluator import MultiOpponentEvaluator
from src.experiments.runner import ExperimentRunner
from src.game.board import create_synthetic_mini_board


def test_multi_opponent_evaluator_basic() -> None:
    board = create_synthetic_mini_board()
    evaluator = MultiOpponentEvaluator(board=board, seed=42)
    agent = RandomAgent(name="EvalCandidate")

    results = evaluator.evaluate(agent=agent, opponents=["random", "greedy"], games_per_opponent=4)

    assert "win_rate_vs_random" in results
    assert "win_rate_vs_greedy" in results
    assert "score_diff_vs_random" in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/experiments/test_rl_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `MultiOpponentEvaluator`, update `ExperimentRunner`, and create YAML configs**

Create `src/experiments/evaluator.py`:
```python
"""Multi-Opponent Evaluator for RL agents during and after training."""

from collections.abc import Sequence
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.evaluation.evaluator import Evaluator
from src.game.board import Board


class MultiOpponentEvaluator:
    """Evaluates an agent against multiple baseline opponents."""

    def __init__(self, board: Board, seed: int = 42) -> None:
        self.board = board
        self.seed = seed
        self.evaluator = Evaluator(board=self.board, seed=self.seed)

    def _get_opponent(self, name: str) -> BaseAgent:
        name_lower = name.lower()
        if name_lower == "random":
            return RandomAgent(name="RandomOpponent")
        if name_lower == "greedy":
            return GreedyAgent(name="GreedyOpponent")
        if name_lower == "strategic":
            return StrategicAgent(name="StrategicOpponent")
        raise ValueError(f"Unknown opponent type: {name}")

    def evaluate(
        self,
        agent: BaseAgent,
        opponents: Sequence[str] = ("random", "greedy", "strategic"),
        games_per_opponent: int = 20,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for opp_name in opponents:
            opp_agent = self._get_opponent(opp_name)
            result = self.evaluator.evaluate_head_to_head(
                agent_a=agent,
                agent_b=opp_agent,
                num_games=games_per_opponent,
            )
            metrics[f"win_rate_vs_{opp_name}"] = result["agent_a_win_rate"]
            metrics[f"score_diff_vs_{opp_name}"] = (
                result["agent_a_mean_score"] - result["agent_b_mean_score"]
            )
            metrics[f"mean_score_vs_{opp_name}"] = result["agent_a_mean_score"]
        return metrics
```

Update `src/experiments/config.py` with full Pydantic schema for training, evaluation, and algorithms.
Update `src/experiments/runner.py` to orchestrate DQN/PPO training, logging, periodic evaluation, and `.pt` checkpointing.
Create `experiments/configs/dqn_mini.yaml`, `experiments/configs/ppo_mini.yaml`, and `experiments/configs/ppo_usa.yaml`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/experiments/test_rl_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/evaluator.py src/experiments/runner.py src/experiments/config.py experiments/configs/ tests/experiments/test_rl_runner.py
git commit -m "feat: implement MultiOpponentEvaluator, experiment runner orchestration, and YAML configs"
```

---

### Task 8: Phase 4 Acceptance Suite & Training Verification (`tests/rl/test_phase4_acceptance.py`)

**Files:**
- Create: `tests/rl/test_phase4_acceptance.py`

**Acceptance Requirements:**
1. DQN training on Mini Board beats RandomAgent consistently with win rate $\ge 75\%$.
2. PPO training on Mini Board beats RandomAgent consistently with win rate $\ge 75\%$.
3. Multi-opponent evaluation benchmark returns complete metrics against Random, Greedy, Strategic.
4. Deterministic reproducibility: training with identical seeds produces identical weights and results.

- [ ] **Step 1: Write Phase 4 acceptance tests in `tests/rl/test_phase4_acceptance.py`**

```python
import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.evaluation.evaluator import Evaluator
from src.experiments.evaluator import MultiOpponentEvaluator
from src.game.board import create_synthetic_mini_board
from src.rl.dqn import MaskedDQNTrainer
from src.rl.ppo import MaskedPPOTrainer


def test_phase4_dqn_beats_random_acceptance() -> None:
    board = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, opponent_policy="random", seed=42)

    config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "buffer_size": 20000,
        "batch_size": 32,
        "target_update_freq": 100,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 3000,
        "learning_starts": 100,
    }

    trainer = MaskedDQNTrainer(env=env, config=config)

    # Train for 5,000 steps on Mini Map (< 10 seconds)
    for _ in range(5000):
        trainer.step()
        trainer.train_step()

    agent = DQNAgent(input_dim=env.observation_space.shape[0], action_dim=int(env.action_space.n))
    agent.q_net.load_state_dict(trainer.policy_net.state_dict())

    evaluator = Evaluator(board=board, seed=123)
    results = evaluator.evaluate_head_to_head(
        agent_a=agent,
        agent_b=RandomAgent(name="RandomOpponent"),
        num_games=40,
    )

    assert results["agent_a_win_rate"] >= 0.75, (
        f"DQN win rate {results['agent_a_win_rate']} must be >= 0.75"
    )


def test_phase4_ppo_beats_random_acceptance() -> None:
    board = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, opponent_policy="random", seed=42)

    config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rollout_steps": 256,
        "num_epochs": 4,
        "minibatch_size": 32,
    }

    trainer = MaskedPPOTrainer(env=env, config=config)

    # Train for 20 rollouts (~5,120 steps on Mini Map)
    for _ in range(20):
        trainer.collect_rollout()
        trainer.train_epoch()

    agent = PPOAgent(input_dim=env.observation_space.shape[0], action_dim=int(env.action_space.n))
    agent.actor_critic.load_state_dict(trainer.actor_critic.state_dict())

    evaluator = Evaluator(board=board, seed=123)
    results = evaluator.evaluate_head_to_head(
        agent_a=agent,
        agent_b=RandomAgent(name="RandomOpponent"),
        num_games=40,
    )

    assert results["agent_a_win_rate"] >= 0.75, (
        f"PPO win rate {results['agent_a_win_rate']} must be >= 0.75"
    )


def test_phase4_multi_opponent_evaluation_suite() -> None:
    board = create_synthetic_mini_board()
    evaluator = MultiOpponentEvaluator(board=board, seed=42)
    agent = RandomAgent(name="TestAgent")

    metrics = evaluator.evaluate(agent=agent, opponents=["random", "greedy", "strategic"], games_per_opponent=10)
    for opp in ["random", "greedy", "strategic"]:
        assert f"win_rate_vs_{opp}" in metrics
        assert f"score_diff_vs_{opp}" in metrics


def test_phase4_deterministic_reproducibility() -> None:
    board = create_synthetic_mini_board()

    def run_training_run(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = TicketToRideEnv(board=board, opponent_policy="random", seed=seed)
        trainer = MaskedPPOTrainer(env=env, config={"rollout_steps": 64, "num_epochs": 2, "minibatch_size": 16})
        trainer.collect_rollout()
        metrics = trainer.train_epoch()
        weights = [p.clone().detach().numpy() for p in trainer.actor_critic.parameters()]
        return metrics, weights

    m1, w1 = run_training_run(seed=999)
    m2, w2 = run_training_run(seed=999)

    assert np.isclose(m1["policy_loss"], m2["policy_loss"], atol=1e-6)
    for p1, p2 in zip(w1, w2):
        assert np.allclose(p1, p2, atol=1e-6)
```

- [ ] **Step 2: Run Phase 4 acceptance test suite**

Run: `.venv/bin/pytest tests/rl/test_phase4_acceptance.py -v`
Expected: PASS

- [ ] **Step 3: Run full project test suite**

Run: `.venv/bin/pytest -v`
Expected: PASS (all tests pass)

- [ ] **Step 4: Commit**

```bash
git add tests/rl/test_phase4_acceptance.py
git commit -m "test: add Phase 4 acceptance test suite for DQN and PPO agents"
```
