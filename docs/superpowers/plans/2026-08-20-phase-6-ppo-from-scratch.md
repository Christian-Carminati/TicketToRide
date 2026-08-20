# Phase 6: PPO From Scratch & Advanced Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reference-grade, CleanRL-standard Proximal Policy Optimization (PPO) engine from scratch with orthogonal initialization, value function clipping, linear learning rate annealing, target KL early stopping, vectorized rollout buffers, and an automated multi-opponent benchmark & ablation suite.

**Architecture:** Refactor and upgrade the neural network architectures in `src/rl/networks.py` with orthogonal layer initialization; implement `VectorRolloutBuffer` in `src/rl/rollout.py` with multidimensional multi-environment GAE in `src/rl/advantage.py`; enhance `MaskedPPOTrainer` in `src/rl/ppo.py` with CleanRL mathematical details; build `PPOBenchmarkRunner` and ablation engine in `src/evaluation/benchmark.py` with CLI script in `scripts/benchmark_ppo.py`; ensure full backward compatibility with Web Lab telemetry and acceptance test suite.

**Tech Stack:** Python 3.12+, PyTorch, Gymnasium, NumPy, Pydantic, FastAPI, WebSockets, Pytest.

**Spec:** [`docs/superpowers/specs/2026-08-20-phase-6-ppo-from-scratch-design.md`](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-20-phase-6-ppo-from-scratch-design.md)

## Global Constraints

- **Python Version**: 3.12+
- **Frameworks**: Pure PyTorch implementation; no black-box RL libraries (Stable-Baselines3, Ray/RLlib, CleanRL package) in production training path.
- **Action Masking**: Categorical distribution logits masking with $-10^8$ prior to Softmax; no penalty rewards for illegal moves.
- **Determinism**: Fully deterministic execution given explicit seeds.
- **Backwards Compatibility**: Maintain compatibility with existing `DQNAgent`, `PPOAgent`, `TicketToRideEnv`, `TrainerService`, and Web Lab WebSocket telemetry.

---

### Task 1: Orthogonal Initialization & Network Enhancements in `MaskedActorCritic`

**Files:**
- Modify: `src/rl/networks.py`
- Test: `tests/rl/test_networks.py`

**Interfaces:**
- Consumes: PyTorch `nn.Module`, `nn.Linear`, `nn.init.orthogonal_`, `nn.init.constant_`
- Produces: `layer_init(layer, std, bias_const)` helper, `MaskedActorCritic(input_dim, action_dim, hidden_dim, orthogonal_init=True)`

- [ ] **Step 1: Write the failing test for orthogonal initialization**

Add tests to `tests/rl/test_networks.py` verifying that layers initialized with `orthogonal_init=True` have orthogonal weight matrices and exact gain scalings (hidden $\approx \sqrt{2}$, actor $\approx 0.01$, critic $\approx 1.0$) and zero biases:

```python
def test_masked_actor_critic_orthogonal_initialization():
    input_dim = 64
    action_dim = 10
    hidden_dim = 128
    model = MaskedActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim, orthogonal_init=True)
    
    # Test hidden layer weights are orthogonal
    for layer in [model.actor[0], model.actor[2], model.critic[0], model.critic[2]]:
        w = layer.weight.data
        # W * W^T should be close to gain^2 * I
        gram = torch.mm(w, w.t())
        expected_scale = 2.0  # gain = sqrt(2), gain^2 = 2.0
        identity = torch.eye(w.shape[0])
        # Diagonal check
        assert torch.allclose(torch.diag(gram), torch.full((w.shape[0],), expected_scale), atol=1e-1)
        assert torch.allclose(layer.bias.data, torch.zeros_like(layer.bias.data))

    # Test actor output head has gain 0.01
    actor_out = model.actor[4]
    w_act = actor_out.weight.data
    gram_act = torch.mm(w_act, w_act.t())
    assert torch.allclose(torch.diag(gram_act), torch.full((w_act.shape[0],), 0.01 ** 2), atol=1e-3)

    # Test critic output head has gain 1.0
    critic_out = model.critic[4]
    assert torch.allclose(torch.norm(critic_out.weight.data), torch.tensor(1.0), atol=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_networks.py::test_masked_actor_critic_orthogonal_initialization -v`
Expected: FAIL (argument or orthogonal check not implemented yet)

- [ ] **Step 3: Implement `layer_init` and orthogonal initialization in `src/rl/networks.py`**

In `src/rl/networks.py`:
```python
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layer weights with orthogonal matrix and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer
```
Update `MaskedActorCritic.__init__` to optionally apply `layer_init`:
- Actor hidden: `layer_init(nn.Linear(input_dim, hidden_dim), np.sqrt(2))` and `layer_init(nn.Linear(hidden_dim, hidden_dim), np.sqrt(2))`
- Actor head: `layer_init(nn.Linear(hidden_dim, action_dim), 0.01)`
- Critic hidden: `layer_init(nn.Linear(input_dim, hidden_dim), np.sqrt(2))` and `layer_init(nn.Linear(hidden_dim, hidden_dim), np.sqrt(2))`
- Critic head: `layer_init(nn.Linear(hidden_dim, 1), 1.0)`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rl/test_networks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/networks.py tests/rl/test_networks.py
git commit -m "feat(rl): add orthogonal layer initialization with calibrated gains to MaskedActorCritic"
```

---

### Task 2: Vectorized Rollout Buffer & Multi-Environment GAE Engine

**Files:**
- Modify: `src/rl/rollout.py`
- Modify: `src/rl/advantage.py`
- Create: `tests/rl/test_vector_rollout.py`
- Modify: `tests/rl/test_buffers.py`

**Interfaces:**
- Consumes: PyTorch / NumPy arrays, `compute_gae`
- Produces: `VectorRolloutBuffer(num_steps, num_envs, obs_dim, action_dim)`, `compute_gae_vectorized(rewards, values, dones, next_values, gamma, gae_lambda)`

- [ ] **Step 1: Write failing tests for VectorRolloutBuffer and vectorized GAE**

Create `tests/rl/test_vector_rollout.py`:
```python
import numpy as np
import torch
from src.rl.rollout import VectorRolloutBuffer
from src.rl.advantage import compute_gae_vectorized

def test_vector_rollout_buffer_shapes_and_minibatches():
    num_steps = 16
    num_envs = 4
    obs_dim = 10
    action_dim = 5
    
    buf = VectorRolloutBuffer(num_steps=num_steps, num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim)
    
    for step in range(num_steps):
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        act = np.random.randint(0, action_dim, size=(num_envs,))
        rew = np.random.randn(num_envs).astype(np.float32)
        val = np.random.randn(num_envs).astype(np.float32)
        logp = np.random.randn(num_envs).astype(np.float32)
        done = np.zeros(num_envs, dtype=bool)
        masks = np.ones((num_envs, action_dim), dtype=bool)
        
        buf.add(obs=obs, action=act, reward=rew, value=val, log_prob=logp, done=done, action_mask=masks)
        
    assert buf.step == num_steps
    assert buf.is_full()
    
    next_values = np.zeros(num_envs, dtype=np.float32)
    advs, returns = buf.compute_returns_and_advantages(next_values=next_values, gamma=0.99, gae_lambda=0.95)
    
    assert advs.shape == (num_steps, num_envs)
    assert returns.shape == (num_steps, num_envs)
    
    minibatches = list(buf.generate_minibatches(batch_size=16, advantages=advs, returns=returns, device="cpu"))
    # Total samples = 16 * 4 = 64. Batch size = 16 => 4 minibatches
    assert len(minibatches) == 4
    mb0 = minibatches[0]
    assert mb0["obs"].shape == (16, obs_dim)
    assert mb0["actions"].shape == (16,)
    assert mb0["action_masks"].shape == (16, action_dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_vector_rollout.py -v`
Expected: FAIL (module/class missing)

- [ ] **Step 3: Implement `compute_gae_vectorized` and `VectorRolloutBuffer`**

In `src/rl/advantage.py`, add vectorized GAE calculation supporting `(T, N)` arrays:
```python
def compute_gae_vectorized(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) across vectorized environments.
    
    Args:
        rewards: Shape (num_steps, num_envs)
        values: Shape (num_steps, num_envs)
        dones: Shape (num_steps, num_envs)
        next_values: Shape (num_envs,)
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
    Returns:
        advantages: Shape (num_steps, num_envs)
        returns: Shape (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t].astype(np.float32)
            next_val = next_values
        else:
            next_non_terminal = 1.0 - dones[t].astype(np.float32)
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
        
    returns = advantages + values
    return advantages, returns
```

In `src/rl/rollout.py`, implement `VectorRolloutBuffer` preserving backward compatibility with `RolloutBuffer`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rl/test_vector_rollout.py tests/rl/test_buffers.py tests/rl/test_advantage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/advantage.py src/rl/rollout.py tests/rl/test_vector_rollout.py
git commit -m "feat(rl): implement VectorRolloutBuffer and vectorized GAE computation"
```

---

### Task 3: CleanRL PPO Optimization Engine (Value Loss Clipping, LR Annealing, Target KL Early Stopping)

**Files:**
- Modify: `src/rl/ppo.py`
- Create: `tests/rl/test_ppo_cleanrl_details.py`
- Modify: `tests/rl/test_ppo_trainer.py`

**Interfaces:**
- Consumes: `MaskedActorCritic`, `VectorRolloutBuffer`, `compute_gae_vectorized`
- Produces: `MaskedPPOTrainer` with options `anneal_lr=True`, `clip_vloss=True`, `target_kl=0.02`, rich training telemetry metrics dictionary.

- [ ] **Step 1: Write failing tests for CleanRL details**

Create `tests/rl/test_ppo_cleanrl_details.py`:
```python
import numpy as np
import torch
from src.environment.env import TicketToRideEnv
from src.rl.ppo import MaskedPPOTrainer

def test_ppo_learning_rate_annealing():
    env = TicketToRideEnv(board_type="mini")
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "lr": 1e-3,
            "anneal_lr": True,
            "total_timesteps": 1000,
            "rollout_steps": 100,
        }
    )
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    assert np.isclose(initial_lr, 1e-3)
    
    # Simulate step advance to halfway
    trainer.total_timesteps = 500
    trainer.update_learning_rate(total_timesteps=1000)
    current_lr = trainer.optimizer.param_groups[0]["lr"]
    assert np.isclose(current_lr, 5e-4, atol=1e-5)

def test_ppo_target_kl_early_stopping():
    env = TicketToRideEnv(board_type="mini")
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "target_kl": 0.001,  # very low threshold to trigger early stopping
            "num_epochs": 10,
            "rollout_steps": 64,
        }
    )
    trainer.collect_rollout()
    metrics = trainer.train_epoch()
    assert "approx_kl" in metrics
    assert "early_stopped" in metrics
    assert metrics["early_stopped"] is True or metrics["epoch_completed"] <= 10

def test_ppo_value_loss_clipping():
    env = TicketToRideEnv(board_type="mini")
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "clip_vloss": True,
            "vf_clip_eps": 0.2,
            "rollout_steps": 64,
        }
    )
    trainer.collect_rollout()
    metrics = trainer.train_epoch()
    assert "value_loss" in metrics
    assert not np.isnan(metrics["value_loss"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rl/test_ppo_cleanrl_details.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CleanRL features in `src/rl/ppo.py`**

Update `MaskedPPOTrainer`:
1. Add `anneal_lr`, `clip_vloss`, `vf_clip_eps`, `target_kl` to `__init__`.
2. Add method `update_learning_rate(total_timesteps: int)`.
3. In `train_epoch()`:
   - Compute clipped value loss:
     ```python
     v_loss_unclipped = (new_value.squeeze(1) - mb["returns"]) ** 2
     v_clipped = mb["values"] + torch.clamp(new_value.squeeze(1) - mb["values"], -self.vf_clip_eps, self.vf_clip_eps)
     v_loss_clipped = (v_clipped - mb["returns"]) ** 2
     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
     value_loss = 0.5 * v_loss_max.mean() if self.clip_vloss else F.mse_loss(new_value.squeeze(1), mb["returns"])
     ```
   - Calculate `clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()`.
   - Calculate `approx_kl = ((ratio - 1.0) - log_ratio).mean()`.
   - Check `if self.target_kl is not None and approx_kl.item() > self.target_kl: early_stopped = True; break`.
4. Return comprehensive metrics dictionary with `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`, `explained_var`, `lr`, `early_stopped`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rl/test_ppo_cleanrl_details.py tests/rl/test_ppo_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/ppo.py tests/rl/test_ppo_cleanrl_details.py tests/rl/test_ppo_trainer.py
git commit -m "feat(rl): add CleanRL value clipping, learning rate annealing, and target KL early stopping"
```

---

### Task 4: Comprehensive Benchmark Suite & Ablation Engine

**Files:**
- Create: `src/evaluation/benchmark.py`
- Create: `scripts/benchmark_ppo.py`
- Create: `tests/evaluation/test_benchmark_suite.py`

**Interfaces:**
- Consumes: `TicketToRideEnv`, `MaskedPPOTrainer`, `RandomAgent`, `GreedyAgent`, `StrategicAgent`, `Evaluator`
- Produces: `PPOBenchmarkRunner`, `run_ppo_benchmark()`, `run_ablation_study()`, export to JSON and Markdown.

- [ ] **Step 1: Write failing test for benchmark suite**

Create `tests/evaluation/test_benchmark_suite.py`:
```python
import json
import pytest
from src.evaluation.benchmark import PPOBenchmarkRunner

def test_benchmark_runner_quick_evaluation(tmp_path):
    json_path = tmp_path / "benchmark_test.json"
    md_path = tmp_path / "benchmark_test.md"
    
    runner = PPOBenchmarkRunner(
        board_type="mini",
        games_per_opponent=5,
        total_training_steps=500,
        output_json=str(json_path),
        output_md=str(md_path),
    )
    
    results = runner.run_benchmark(run_ablation=True, ablation_steps=200)
    
    assert "opponents" in results
    assert "ablation" in results
    assert "random" in results["opponents"]
    assert "greedy" in results["opponents"]
    assert "strategic" in results["opponents"]
    
    # Check JSON output written
    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
    assert data["board_type"] == "mini"
    
    # Check Markdown output written
    assert md_path.exists()
    content = md_path.read_text()
    assert "# PPO Benchmark & Ablation Report" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evaluation/test_benchmark_suite.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `PPOBenchmarkRunner` in `src/evaluation/benchmark.py` and CLI script `scripts/benchmark_ppo.py`**

In `src/evaluation/benchmark.py`:
Implement `PPOBenchmarkRunner`:
- `train_agent(config, steps)`
- `evaluate_against_baselines(agent, games_per_opp)`
- `run_ablation_study(ablation_steps)` comparing `ppo_full`, `ppo_no_ortho`, `ppo_no_vf_clip`, `ppo_no_lr_anneal`
- `export_json(path)` and `export_markdown(path)` generating formatted Markdown tables.

In `scripts/benchmark_ppo.py`:
Create CLI entry point with `argparse`:
`--board`, `--games`, `--steps`, `--ablation`, `--output-json`, `--output-md`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/evaluation/test_benchmark_suite.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark.py scripts/benchmark_ppo.py tests/evaluation/test_benchmark_suite.py
git commit -m "feat(eval): add PPOBenchmarkRunner, ablation study engine, and CLI benchmark script"
```

---

### Task 5: Web Lab & Telemetry Integration Verification

**Files:**
- Modify: `src/api/trainer_service.py`
- Modify: `tests/api/test_trainer_service.py`
- Modify: `tests/api/test_websocket_telemetry.py`

**Interfaces:**
- Consumes: `MaskedPPOTrainer` updated metrics
- Produces: Live WebSocket telemetry events with `approx_kl`, `clip_fraction`, `entropy`, `explained_var`, `lr`.

- [ ] **Step 1: Write test for enriched telemetry broadcast in TrainerService**

In `tests/api/test_trainer_service.py`, add verification for new PPO telemetry fields:
```python
def test_trainer_service_ppo_enriched_metrics():
    # Verify that telemetry payload emitted by MaskedPPOTrainer contains CleanRL metrics
    # and is consumed without error by TrainerService
    ...
```

- [ ] **Step 2: Run test to verify compatibility**

Run: `uv run pytest tests/api/test_trainer_service.py tests/api/test_websocket_telemetry.py -v`

- [ ] **Step 3: Update `src/api/trainer_service.py` to forward all new metrics**

Ensure `TrainerService` forwards `approx_kl`, `clip_fraction`, and `explained_var` to WebSocket broadcast payloads.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/api/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/trainer_service.py tests/api/test_trainer_service.py tests/api/test_websocket_telemetry.py
git commit -m "feat(api): forward enriched CleanRL telemetry metrics to Web Lab WebSocket streams"
```

---

### Task 6: Phase 6 Acceptance Test & End-to-End Scientific Validation

**Files:**
- Create: `tests/rl/test_phase6_acceptance.py`

**Interfaces:**
- Consumes: All Phase 6 modules
- Produces: Complete acceptance verification of PPO beating baseline agents consistently, maintaining deterministic reproducibility, and generating benchmark reports.

- [ ] **Step 1: Write `tests/rl/test_phase6_acceptance.py`**

```python
import pytest
from src.environment.env import TicketToRideEnv
from src.rl.ppo import MaskedPPOTrainer
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.evaluation.evaluator import Evaluator

def test_phase6_ppo_cleanrl_training_and_acceptance():
    """Verify that custom CleanRL PPO trains effectively and outperforms RandomAgent."""
    env = TicketToRideEnv(board_type="mini", seed=42)
    trainer = MaskedPPOTrainer(
        env=env,
        config={
            "lr": 3e-4,
            "anneal_lr": True,
            "clip_vloss": True,
            "target_kl": 0.03,
            "rollout_steps": 256,
            "num_epochs": 4,
            "minibatch_size": 64,
        }
    )
    
    # Train for small budget
    trainer.train(total_timesteps=2000)
    
    agent = PPOAgent(
        actor_critic=trainer.actor_critic,
        deterministic=True,
        action_space=env.action_space,
    )
    random_agent = RandomAgent(action_space=env.action_space)
    
    evaluator = Evaluator(board_type="mini")
    metrics = evaluator.evaluate(agent=agent, opponent=random_agent, num_games=30, seed=123)
    
    assert metrics["win_rate"] >= 0.70
    assert metrics["agent_avg_score"] > metrics["opponent_avg_score"]

def test_phase6_deterministic_reproducibility():
    """Verify that two PPO trainers with identical seeds produce identical losses and weights."""
    env1 = TicketToRideEnv(board_type="mini", seed=99)
    env2 = TicketToRideEnv(board_type="mini", seed=99)
    
    trainer1 = MaskedPPOTrainer(env=env1, config={"rollout_steps": 64, "num_epochs": 2})
    trainer2 = MaskedPPOTrainer(env=env2, config={"rollout_steps": 64, "num_epochs": 2})
    
    m1 = trainer1.train(total_timesteps=128)
    m2 = trainer2.train(total_timesteps=128)
    
    assert np.isclose(m1["policy_loss"], m2["policy_loss"], atol=1e-5)
```

- [ ] **Step 2: Run acceptance tests**

Run: `uv run pytest tests/rl/test_phase6_acceptance.py -v`
Expected: PASS

- [ ] **Step 3: Run entire project test suite**

Run: `uv run pytest -v`
Expected: All 130+ tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/rl/test_phase6_acceptance.py
git commit -m "test(acceptance): add Phase 6 acceptance tests and scientific validation"
```
