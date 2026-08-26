# Comprehensive RL Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate training throughput across all Reinforcement Learning pipelines (AlphaZero, CleanRL PPO, Recurrent LSTM PPO, Double-DQN, and Self-Play) using MCTS Subtree Reuse, Zero-Copy Circular Replay Buffers, DQN Train Frequency Decoupling, and Zero-Overhead Rollout Transfers, concluding with automated before/after comparative gain reporting.

**Architecture:** MCTS preserves and shifts search tree subnodes across game turns (`root = root.children[action]`) eliminating redundant tree re-expansion; `SelfPlayReplayBuffer` and `ReplayBuffer` adopt pre-allocated contiguous 2D NumPy memory arrays with vectorized mini-batch tensor generation; `MaskedDQNTrainer` decouples environment step collection from gradient steps (`train_frequency=4`); `RolloutBuffer` writes directly into C-contiguous buffers with zero Python tensor allocations per step.

**Tech Stack:** Python 3.12+, PyTorch 2.2+, NumPy, Gymnasium, Pytest.

**Spec:** [`docs/superpowers/specs/2026-08-26-comprehensive-training-optimization-design.md`](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-26-comprehensive-training-optimization-design.md)

## Global Constraints
- Deterministic reproducibility under seeded pseudo-random number generators.
- POMDP anti-leakage invariants strictly preserved across all observation encoders.
- 100% backward compatibility with all 97+ unit and acceptance tests.
- Side-by-side benchmark reporting comparing against `benchmark_baseline_pre_optimization.json`.

---

### Task 1: Zero-Copy Circular Replay Buffer for AlphaZero

**Files:**
- Modify: `src/rl/alphazero_trainer.py:26-73`
- Test: `tests/rl/test_alphazero_trainer.py`

**Interfaces:**
- Produces: `SelfPlayReplayBuffer` with pre-allocated contiguous 2D NumPy arrays `(capacity, obs_dim)`, `(capacity, action_dim)` and vector slicing in `sample(batch_size)`.

- [ ] **Step 1: Write unit test for Zero-Copy SelfPlayReplayBuffer**

Add to `tests/rl/test_alphazero_trainer.py`:
```python
def test_selfplay_replay_buffer_zero_copy_contiguous():
    from src.rl.alphazero_trainer import SelfPlayReplayBuffer
    import numpy as np
    import torch

    buffer = SelfPlayReplayBuffer(capacity=100)
    obs = np.ones(464, dtype=np.float32)
    mask = np.ones(150, dtype=np.float32)
    pi = np.ones(150, dtype=np.float32) / 150.0
    z = 1.0

    for _ in range(50):
        buffer.add(obs, mask, pi, z)

    assert len(buffer) == 50
    obs_b, mask_b, pi_b, z_b = buffer.sample(16)
    assert obs_b.shape == (16, 464)
    assert mask_b.shape == (16, 150)
    assert pi_b.shape == (16, 150)
    assert z_b.shape == (16, 1)
    assert isinstance(obs_b, torch.Tensor)
```

- [ ] **Step 2: Run test to verify current state**

Run: `uv run pytest tests/rl/test_alphazero_trainer.py -k test_selfplay_replay_buffer_zero_copy_contiguous -v`
Expected: PASS or verify existing behavior.

- [ ] **Step 3: Implement Zero-Copy contiguous array allocation in `SelfPlayReplayBuffer`**

Update `SelfPlayReplayBuffer` in `src/rl/alphazero_trainer.py`:
```python
class SelfPlayReplayBuffer:
    """Experience replay storing (obs, action_mask, pi_mcts, outcome_z) with pre-allocated contiguous arrays."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.obs_buf: np.ndarray | None = None
        self.mask_buf: np.ndarray | None = None
        self.pi_buf: np.ndarray | None = None
        self.z_buf: np.ndarray | None = None
        self.ptr: int = 0
        self.size: int = 0

    def __len__(self) -> int:
        return self.size

    def _lazy_init(self, obs_dim: int, action_dim: int) -> None:
        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.mask_buf = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.pi_buf = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.z_buf = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, mask: np.ndarray, pi: np.ndarray, z: float) -> None:
        if self.obs_buf is None:
            self._lazy_init(len(obs), len(mask))

        assert self.obs_buf is not None and self.mask_buf is not None and self.pi_buf is not None and self.z_buf is not None

        self.obs_buf[self.ptr] = obs
        self.mask_buf[self.ptr] = mask
        self.pi_buf[self.ptr] = pi
        self.z_buf[self.ptr, 0] = z

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.obs_buf is not None and self.mask_buf is not None and self.pi_buf is not None and self.z_buf is not None
        b_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, size=b_size, replace=False)
        obs_b = torch.from_numpy(self.obs_buf[indices])
        mask_b = torch.from_numpy(self.mask_buf[indices])
        pi_b = torch.from_numpy(self.pi_buf[indices])
        z_b = torch.from_numpy(self.z_buf[indices])
        return obs_b, mask_b, pi_b, z_b
```

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/rl/test_alphazero_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl/alphazero_trainer.py tests/rl/test_alphazero_trainer.py
git commit -m "perf(alphazero): implement zero-copy contiguous replay buffer"
```

---

### Task 2: Subtree Reuse in AlphaZero Neural MCTS Engine

**Files:**
- Modify: `src/rl/alphazero_search.py:137-257`
- Modify: `src/rl/alphazero_trainer.py:115-160`
- Test: `tests/rl/test_alphazero_search.py`

**Interfaces:**
- Produces: `NeuralMCTSEngine.search(..., previous_root=None, action_taken=None)` supporting subtree inheritance.

- [ ] **Step 1: Write test for Subtree Reuse**

Add to `tests/rl/test_alphazero_search.py`:
```python
def test_subtree_reuse_preserves_accumulated_visits():
    from src.rl.alphazero_search import NeuralMCTSEngine, NeuralMCTSNode
    from src.rl.policy_value_net import PolicyValueNetwork
    from src.game.game import Game

    net = PolicyValueNetwork(obs_dim=464, action_dim=150, hidden_dim=64, num_res_blocks=1)
    engine = NeuralMCTSEngine(net=net, num_simulations=10)

    game = Game(num_players=2)
    game.reset(seed=42)

    best_act, pi, val, root_node = engine.search_with_root(game, "player_0")
    assert root_node.total_visits >= 10
    assert best_act in root_node.children

    # Subsequent search reusing the chosen subtree
    child_node = root_node.children[best_act]
    child_visits_before = sum(child_node.visits.values())
    game.step(engine.action_space.to_action(best_act))

    best_act_2, pi_2, val_2, root_node_2 = engine.search_with_root(game, "player_1", previous_root=root_node, action_taken=best_act)
    assert root_node_2.total_visits >= child_visits_before
```

- [ ] **Step 2: Run test to verify it fails initially**

Run: `uv run pytest tests/rl/test_alphazero_search.py -k test_subtree_reuse_preserves_accumulated_visits -v`
Expected: FAIL with attribute error or method missing.

- [ ] **Step 3: Implement `search_with_root` and Subtree Reuse in `NeuralMCTSEngine`**

Update `NeuralMCTSEngine` in `src/rl/alphazero_search.py`:
```python
    def search_with_root(
        self,
        game: Game,
        player_id: str,
        is_root_exploration: bool = True,
        custom_determinizer: Callable[[Game, str, SeededRNG], Game] | None = None,
        previous_root: NeuralMCTSNode | None = None,
        action_taken: int | None = None,
    ) -> tuple[int, np.ndarray, float, NeuralMCTSNode]:
        curr_p = game.state.current_player
        curr_player_id = curr_p.id if curr_p is not None else player_id

        # Subtree reuse: inherit child node if available
        if previous_root is not None and action_taken is not None and action_taken in previous_root.children:
            root = previous_root.children[action_taken]
            root.parent = None
            root.action_from_parent = None
            root.state_player = curr_player_id
        else:
            root = NeuralMCTSNode(state_player=curr_player_id)

        if not root.is_expanded:
            obs, mask, legal_actions = self._get_obs_and_mask(game, curr_player_id)
            priors, _root_v = self.net.evaluate_state(obs, mask)
            if is_root_exploration and len(legal_actions) > 1:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
                for idx, a in enumerate(legal_actions):
                    root.priors[a] = float(
                        (1.0 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * noise[idx]
                    )
            else:
                for a in legal_actions:
                    root.priors[a] = float(priors[a])

            root.legal_actions = legal_actions
            for a in legal_actions:
                root.visits[a] = 0
                root.values[a] = 0.0
            root.is_expanded = True

        # Run simulations
        for _ in range(self.num_simulations):
            if custom_determinizer is not None:
                sim_game = custom_determinizer(game, player_id, self.rng)
            else:
                sim_game = determinize_game(game, player_id, self.rng)

            node = root
            search_path: list[tuple[NeuralMCTSNode, int]] = []

            while node.is_expanded and not sim_game.state.is_game_over:
                action_idx = node.select_puct_action(self.c_puct)
                search_path.append((node, action_idx))
                game_action = self.action_space.to_action(action_idx)
                sim_game.step(game_action)

                if action_idx not in node.children:
                    next_p = sim_game.state.current_player
                    next_player_id = next_p.id if next_p is not None else curr_player_id
                    node.children[action_idx] = NeuralMCTSNode(
                        state_player=next_player_id, parent=node, action_from_parent=action_idx
                    )
                node = node.children[action_idx]

            if sim_game.state.is_game_over:
                scores = {p.id: p.score for p in sim_game.state.players}
                p0_score = scores.get(player_id, 0)
                opp_ids = [pid for pid in scores if pid != player_id]
                opp_score = scores.get(opp_ids[0], 0) if opp_ids else 0
                if p0_score > opp_score:
                    leaf_val = 1.0
                elif p0_score < opp_score:
                    leaf_val = -1.0
                else:
                    leaf_val = 0.0
            else:
                s_p = sim_game.state.current_player
                s_player_id = s_p.id if s_p is not None else player_id
                obs_leaf, mask_leaf, legal_leaf = self._get_obs_and_mask(sim_game, s_player_id)

                s_hash = hash(obs_leaf.tobytes())
                if s_hash in self.tt_cache:
                    priors_leaf, val_leaf = self.tt_cache[s_hash]
                else:
                    priors_leaf, val_leaf = self.net.evaluate_state(obs_leaf, mask_leaf)
                    self.tt_cache[s_hash] = (priors_leaf, val_leaf)

                node.legal_actions = legal_leaf
                for a in legal_leaf:
                    node.priors[a] = float(priors_leaf[a])
                    node.visits[a] = 0
                    node.values[a] = 0.0
                node.is_expanded = True
                leaf_val = val_leaf if s_player_id == player_id else -val_leaf

            for parent_node, action_taken_sim in reversed(search_path):
                val_for_parent = leaf_val if parent_node.state_player == player_id else -leaf_val
                parent_node.visits[action_taken_sim] += 1
                parent_node.values[action_taken_sim] += val_for_parent

        pi = np.zeros(self.net.action_dim, dtype=np.float32)
        for a in root.legal_actions:
            pi[a] = root.visits.get(a, 0)

        tot_visits = pi.sum()
        if tot_visits > 0:
            pi /= tot_visits
        else:
            for a in root.legal_actions:
                pi[a] = 1.0 / len(root.legal_actions)

        best_action = int(np.argmax(pi))
        root_val = sum(root.values.values()) / max(1, root.total_visits)
        return best_action, pi, root_val, root
```

- [ ] **Step 4: Connect subtree tracking in `AlphaZeroTrainer.collect_self_play_games`**

```python
            prev_root = None
            last_action = None
            while not game.state.is_game_over and turns < max_turns:
                curr_p = game.state.current_player
                curr_player_id = curr_p.id if curr_p is not None else "player_0"
                obs, mask = self._get_obs_and_mask(game, curr_player_id)

                best_action, pi_target, _, root_node = self.search_engine.search_with_root(
                    game, curr_player_id, is_root_exploration=True,
                    previous_root=prev_root, action_taken=last_action
                )

                history.append((curr_player_id, obs, mask, pi_target))
                game_action = self.action_space.to_action(best_action)
                game.step(game_action)
                prev_root = root_node
                last_action = best_action
                turns += 1
```

- [ ] **Step 5: Run tests and verify**

Run: `uv run pytest tests/rl/test_alphazero_search.py tests/rl/test_alphazero_trainer.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/rl/alphazero_search.py src/rl/alphazero_trainer.py tests/rl/test_alphazero_search.py
git commit -m "perf(alphazero): implement MCTS subtree reuse across game turns"
```

---

### Task 3: DQN Training Frequency Decoupling & Fast Replay Buffer

**Files:**
- Modify: `src/rl/dqn.py:20-130`
- Modify: `src/rl/replay_buffer.py:10-90`
- Modify: `src/api/trainer_service.py:375-425`
- Test: `tests/rl/test_dqn_trainer.py`

**Interfaces:**
- Produces: `MaskedDQNTrainer(..., train_frequency=4)` executing gradient steps every $k$ environment interactions.

- [ ] **Step 1: Write unit test for DQN train frequency decoupling**

Add to `tests/rl/test_dqn_trainer.py`:
```python
def test_dqn_train_frequency_decoupling():
    from src.environment.env import TicketToRideEnv
    from src.rl.dqn import MaskedDQNTrainer

    env = TicketToRideEnv()
    trainer = MaskedDQNTrainer(env=env, config={"train_frequency": 4, "batch_size": 16, "learning_starts": 10})
    
    assert trainer.train_frequency == 4
    # Step 1-3 should not trigger gradient step if step_count % 4 != 0
    res = trainer.step()
    assert res is not None
```

- [ ] **Step 2: Implement `train_frequency` in `src/rl/dqn.py` and `src/api/trainer_service.py`**

In `src/rl/dqn.py`:
```python
        self.train_frequency: int = self.config.get("train_frequency", 4)
```

In `src/api/trainer_service.py` under `elif algo == "dqn":`:
```python
                dqn_config = {
                    "lr": getattr(config.algorithm, "learning_rate", 5e-4),
                    "gamma": getattr(config.algorithm, "gamma", 0.99),
                    "batch_size": getattr(config.algorithm, "batch_size", 32),
                    "buffer_size": getattr(config.algorithm, "buffer_size", 10000),
                    "target_update_freq": getattr(config.algorithm, "target_update_freq", 200),
                    "epsilon_start": getattr(config.algorithm, "epsilon_start", 1.0),
                    "epsilon_end": getattr(config.algorithm, "epsilon_end", 0.05),
                    "epsilon_decay_steps": getattr(config.algorithm, "epsilon_decay_steps", 5000),
                    "learning_starts": getattr(config.algorithm, "learning_starts", 50),
                    "train_frequency": getattr(config.algorithm, "train_frequency", 4),
                }
                trainer_dqn = TrainerFactory.create("dqn", env=env, config=dqn_config)

                episode_rewards: list[float] = []
                current_ep_reward = 0.0
                metrics = {"loss": 0.0, "epsilon": 1.0}

                for step in range(1, total_steps + 1):
                    if self._stop_requested:
                        break

                    reward, done = trainer_dqn.step()
                    current_ep_reward += reward

                    if done:
                        episode_count += 1
                        episode_rewards.append(current_ep_reward)
                        current_ep_reward = 0.0

                    if step % trainer_dqn.train_frequency == 0:
                        metrics = trainer_dqn.train_step()
```

- [ ] **Step 3: Run DQN tests**

Run: `uv run pytest tests/rl/test_dqn_trainer.py tests/api/test_trainer_service.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/rl/dqn.py src/api/trainer_service.py tests/rl/test_dqn_trainer.py
git commit -m "perf(dqn): decouple environment step from training gradient frequency"
```

---

### Task 4: Zero-Overhead Memory & Vector GAE for PPO Rollout Buffer

**Files:**
- Modify: `src/rl/rollout.py:65-150`
- Modify: `src/rl/ppo.py:70-130`
- Test: `tests/rl/test_ppo_trainer.py`

**Interfaces:**
- Produces: `RolloutBuffer` with contiguous NumPy views for PyTorch mini-batches without intermediate list allocations.

- [ ] **Step 1: Write test for RolloutBuffer slicing**

Add to `tests/rl/test_buffers.py`:
```python
def test_rollout_buffer_contiguous_tensor_batches():
    from src.rl.rollout import RolloutBuffer
    import numpy as np
    import torch

    buf = RolloutBuffer(capacity=128, obs_dim=100, action_dim=20)
    for _ in range(64):
        buf.add(np.zeros(100), 1, 1.0, 0.5, -0.5, False, np.ones(20))

    batches = list(buf.get_minibatches(minibatch_size=32, device="cpu"))
    assert len(batches) == 2
    b0 = batches[0]
    assert isinstance(b0.obs, torch.Tensor)
    assert b0.obs.shape == (32, 100)
```

- [ ] **Step 2: Optimize mini-batch generation in `RolloutBuffer`**

In `src/rl/rollout.py`:
Ensure `get_minibatches()` slices directly with `torch.from_numpy()` over randomized permutation indices `np.random.permutation(self.size)`.

- [ ] **Step 3: Run PPO and Buffer tests**

Run: `uv run pytest tests/rl/test_buffers.py tests/rl/test_ppo_trainer.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/rl/rollout.py src/rl/ppo.py tests/rl/test_buffers.py
git commit -m "perf(ppo): optimize rollout mini-batch memory transfers"
```

---

### Task 5: Full Benchmark Suite & Side-by-Side Comparative Gain Report

**Files:**
- Create: `scripts/benchmark_training_suite.py`
- Generate: `benchmark_comparison_report.txt`
- Test: `tests/evaluation/test_benchmark_suite.py`

**Interfaces:**
- Produces: Comparative benchmark report evaluating baseline vs optimized throughput and convergence.

- [ ] **Step 1: Create `scripts/benchmark_training_suite.py`**

Script measures current FPS for all 5 algorithms, loads `benchmark_baseline_pre_optimization.json`, computes percentage gains, asserts correctness, and outputs formatted table to console and `benchmark_comparison_report.txt`.

- [ ] **Step 2: Run benchmark suite**

Run: `uv run python scripts/benchmark_training_suite.py`
Expected: PASS and generate report file.

- [ ] **Step 3: Run all unit tests for complete non-regression**

Run: `uv run pytest tests/game/ tests/environment/ tests/agents/ tests/rl/ tests/api/test_trainer_service.py`
Expected: 100% PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_training_suite.py benchmark_comparison_report.txt
git commit -m "bench(training): add automated training benchmark suite and comparative report"
```
