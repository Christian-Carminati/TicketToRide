# Full-Stack Performance & Stability Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the entire TicketToRide RL Lab application stack (Game Core, PPO/DQN RL engine, backend WebSocket telemetry streaming, and React frontend rendering) to eliminate browser crashes, drop CPU usage, and achieve massive speedups (> +50% PPO FPS, > +150% DQN FPS, > 50% test suite duration reduction) backed by scientific benchmarks.

**Architecture:** 
1. Pre-allocate contiguous ring buffers and vectorize GAE/advantage calculations for PPO and DQN, removing per-step PyTorch tensor allocations.
2. Replace dynamic string-based graph BFS traversals in Observation encoding and Reward calculation with integer-indexed adjacency and incremental connectivity.
3. Add adaptive telemetry rate-limiting (20 FPS max) on the backend and eliminate artificial sleep penalties.
4. Centralize WebSocket connections on the frontend, isolate telemetry state updates from the global workspace tree, memoize heavy visual components (`BoardCanvas`, `BrainInspector`), and throttle chart rendering via `requestAnimationFrame`.

**Tech Stack:** Python 3.12, PyTorch 2.2+, Gymnasium, NumPy, FastAPI, WebSockets, React 18, TypeScript, Vite.

**Spec:** [`docs/superpowers/specs/2026-08-19-performance-optimization-design.md`](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-19-performance-optimization-design.md)

## Global Constraints
- Preserve 100% deterministic reproducibility for given `(seed, state, actions)`.
- Strictly enforce anti-leakage POMDP observation guarantees.
- Maintain agent win-rate hierarchy: Strategic > Greedy > Random.
- All 119 existing unit and acceptance tests must pass.
- Produce scientific before/after benchmark evidence with percentage improvements.

---

### Task 1: Pre-allocated Fast ReplayBuffer & Vectorized Double-DQN Trainer

**Files:**
- Modify: `src/rl/replay_buffer.py`
- Modify: `src/rl/dqn.py`
- Modify: `src/rl/networks.py`
- Test: `tests/rl/test_buffers.py`
- Test: `tests/rl/test_dqn_trainer.py`

**Interfaces:**
- `ReplayBuffer(capacity: int, obs_dim: int = 0, action_dim: int = 0)`
- `ReplayBuffer.push(obs, action, reward, next_obs, done, next_action_mask)`
- `ReplayBuffer.sample(batch_size: int, device: str = "cpu") -> ReplayBatch`
- `MaskedQNetwork.select_action(obs: torch.Tensor, action_mask: np.ndarray, epsilon: float = 0.0) -> int`

- [ ] **Step 1: Write test for pre-allocated ReplayBuffer with contiguous sampling**

In `tests/rl/test_buffers.py`:
```python
def test_preallocated_replay_buffer_fast_sample():
    from src.rl.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=1000, obs_dim=10, action_dim=5)
    for i in range(100):
        obs = np.ones(10, dtype=np.float32) * i
        next_obs = np.ones(10, dtype=np.float32) * (i + 1)
        mask = np.zeros(5, dtype=bool)
        mask[i % 5] = True
        buf.push(obs, i % 5, float(i), next_obs, i % 20 == 0, mask)
    
    assert len(buf) == 100
    batch = buf.sample(batch_size=16, device="cpu")
    assert batch.obs.shape == (16, 10)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.next_obs.shape == (16, 10)
    assert batch.dones.shape == (16,)
    assert batch.next_action_masks.shape == (16, 5)
```

- [ ] **Step 2: Run test to verify failure or need for pre-allocation support**

Run: `./.venv/bin/pytest tests/rl/test_buffers.py -v`

- [ ] **Step 3: Implement pre-allocated circular arrays in `ReplayBuffer` and fast batch conversion**

In `src/rl/replay_buffer.py`:
```python
class ReplayBuffer:
    """Fixed-capacity vectorized experience replay buffer with circular pre-allocation."""

    def __init__(self, capacity: int = 100000, obs_dim: int = 0, action_dim: int = 0) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self.initialized = False
        self.obs_buf: np.ndarray | None = None
        self.next_obs_buf: np.ndarray | None = None
        self.actions_buf = np.zeros(capacity, dtype=np.int64)
        self.rewards_buf = np.zeros(capacity, dtype=np.float32)
        self.dones_buf = np.zeros(capacity, dtype=bool)
        self.masks_buf: np.ndarray | None = None

    def _lazy_init(self, obs_dim: int, action_mask_dim: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_mask_dim
        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.masks_buf = np.zeros((self.capacity, action_mask_dim), dtype=bool)
        self.initialized = True

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32).ravel()
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).ravel()
        mask_arr = np.asarray(next_action_mask if next_action_mask is not None else [True], dtype=bool).ravel()

        if not self.initialized or self.obs_buf is None or self.masks_buf is None:
            self._lazy_init(len(obs_arr), len(mask_arr))

        self.obs_buf[self.ptr] = obs_arr
        self.actions_buf[self.ptr] = int(action)
        self.rewards_buf[self.ptr] = float(reward)
        self.next_obs_buf[self.ptr] = next_obs_arr
        self.dones_buf[self.ptr] = bool(done)
        self.masks_buf[self.ptr] = mask_arr

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> ReplayBatch:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions_buf[idxs], dtype=torch.int64, device=device),
            rewards=torch.as_tensor(self.rewards_buf[idxs], dtype=torch.float32, device=device),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones_buf[idxs], dtype=torch.float32, device=device),
            next_action_masks=torch.as_tensor(self.masks_buf[idxs], dtype=torch.bool, device=device),
        )

    def __len__(self) -> int:
        return self.size
```

- [ ] **Step 4: Optimize `MaskedQNetwork.select_action` and `MaskedDQNTrainer`**

In `src/rl/networks.py`:
Avoid cloning and redundant tensor creations by using `torch.as_tensor` and direct indexing.
In `src/rl/dqn.py`:
Pre-allocate buffer with observation and action dimensions.

- [ ] **Step 5: Run tests to verify all DQN and buffer tests pass**

Run: `./.venv/bin/pytest tests/rl/test_buffers.py tests/rl/test_dqn_trainer.py tests/rl/test_networks.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/rl/replay_buffer.py src/rl/dqn.py src/rl/networks.py tests/rl/test_buffers.py
git commit -m "perf(rl): implement pre-allocated vectorized replay buffer and optimized DQN trainer"
```

---

### Task 2: Pre-allocated RolloutBuffer & Vectorized GAE for Masked PPO

**Files:**
- Modify: `src/rl/rollout.py`
- Modify: `src/rl/advantage.py`
- Modify: `src/rl/ppo.py`
- Modify: `src/rl/networks.py`
- Test: `tests/rl/test_advantage.py`
- Test: `tests/rl/test_ppo_trainer.py`

**Interfaces:**
- `RolloutBuffer.generate_minibatches(batch_size: int, advantages: np.ndarray, returns: np.ndarray, device: str = "cpu")`
- `compute_gae(rewards, values, dones, next_value, gamma, gae_lambda) -> tuple[np.ndarray, np.ndarray]`
- `MaskedActorCritic.get_action_and_value(...)`

- [ ] **Step 1: Write test for vectorized RolloutBuffer and GAE efficiency**

In `tests/rl/test_advantage.py`:
Verify exact equivalence and numerical precision of vectorized GAE against multi-step references.

- [ ] **Step 2: Run test to check baseline**

Run: `./.venv/bin/pytest tests/rl/test_advantage.py tests/rl/test_ppo_trainer.py -v`

- [ ] **Step 3: Implement vectorized `RolloutBuffer` and vectorized GAE**

In `src/rl/rollout.py`:
Store transitions in pre-allocated or directly vectorized arrays so `generate_minibatches` converts to tensors once per epoch instead of on each slice.
In `src/rl/advantage.py`:
Vectorize advantage computation with fast NumPy backwards recurrence without per-element Python type casts.
In `src/rl/networks.py`:
In `MaskedActorCritic.get_action_and_value`, replace `torch.tensor(-1e8, ...)` with `logits.masked_fill(~action_mask, -1e8)`.

- [ ] **Step 4: Optimize `MaskedPPOTrainer.collect_rollout` and `train_epoch`**

In `src/rl/ppo.py`:
Keep observation and mask tensors on device or use `torch.as_tensor` efficiently without `.item()` bottlenecks.

- [ ] **Step 5: Run tests to verify all PPO and advantage tests pass**

Run: `./.venv/bin/pytest tests/rl/test_advantage.py tests/rl/test_ppo_trainer.py tests/rl/test_phase4_acceptance.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/rl/rollout.py src/rl/advantage.py src/rl/ppo.py src/rl/networks.py
git commit -m "perf(rl): vectorize GAE computation, pre-allocate rollout buffer, and optimize ActorCritic masking"
```

---

### Task 3: Integer-Indexed Graph Connectivity & Observation/Reward Fast-Paths

**Files:**
- Modify: `src/game/graph.py`
- Modify: `src/environment/observation.py`
- Modify: `src/environment/reward.py`
- Test: `tests/game/test_game_engine.py`
- Test: `tests/environment/test_observation.py`
- Test: `tests/environment/test_reward.py`

**Interfaces:**
- `check_ticket_completed(player_routes: list[Route], ticket: DestinationTicket) -> bool`
- `ObservationV1.encode(state: GameState, player_index: int) -> np.ndarray`
- `DefaultRewardCalculator.calculate(prev_state, action, next_state, player_index) -> float`

- [ ] **Step 1: Write benchmark and correctness tests for incremental connectivity in `test_game_engine.py`**

Test that both Disjoint-Set / fast BFS and legacy queries produce identical boolean results for any set of player routes.

- [ ] **Step 2: Run test to verify**

Run: `./.venv/bin/pytest tests/game/test_game_engine.py tests/environment/test_observation.py tests/environment/test_reward.py -v`

- [ ] **Step 3: Implement fast connectivity in `graph.py`, `observation.py`, and `reward.py`**

In `src/game/graph.py`:
Optimize `check_ticket_completed` with integer city IDs / adjacency sets and fast BFS.
In `src/environment/observation.py`:
Pre-compute route ID to index lookups, ticket ID to index lookups, and use an efficient incremental connectivity graph to encode ticket completion in `O(1)`.
In `src/environment/reward.py`:
Skip ticket recalculations when the action is not `CLAIM_ROUTE` and player route count hasn't changed.

- [ ] **Step 4: Run tests to ensure 100% acceptance across game, environment, and evaluation**

Run: `./.venv/bin/pytest tests/game/ tests/environment/ tests/evaluation/ -v`

- [ ] **Step 5: Commit**

```bash
git add src/game/graph.py src/environment/observation.py src/environment/reward.py
git commit -m "perf(game): optimize graph ticket connectivity, observation encoding, and reward calculations"
```

---

### Task 4: Backend Telemetry Rate-Limiting & Adaptive Batching

**Files:**
- Modify: `src/api/trainer_service.py`
- Modify: `src/api/websocket.py`
- Test: `tests/api/test_trainer_service.py`
- Test: `tests/api/test_websocket_telemetry.py`

**Interfaces:**
- `TrainerService._run_training(...)`
- `ConnectionManager.broadcast_sync(...)`

- [ ] **Step 1: Write test for rate-limited telemetry broadcasting**

In `tests/api/test_trainer_service.py`:
Verify that running training at 10,000 steps broadcasts telemetry at regular intervals without blocking the training thread.

- [ ] **Step 2: Run test to check current behavior**

Run: `./.venv/bin/pytest tests/api/test_trainer_service.py -v`

- [ ] **Step 3: Implement adaptive rate-limiter in `TrainerService`**

In `src/api/trainer_service.py`:
- Use `last_broadcast_time = time.time()` and broadcast telemetry only when `now - last_broadcast_time >= 0.05` (20 FPS max) or on step == total_steps or done.
- Remove arbitrary blocking `time.sleep(0.04)` and `time.sleep(0.005)` delays during training, allowing training to achieve full CPU/GPU throughput while broadcasting smoothly.

- [ ] **Step 4: Run all API tests**

Run: `./.venv/bin/pytest tests/api/ -v`

- [ ] **Step 5: Commit**

```bash
git add src/api/trainer_service.py src/api/websocket.py
git commit -m "perf(api): add adaptive telemetry rate limiting and remove artificial training loop sleep"
```

---

### Task 5: Frontend Single WebSocket, Decoupled State & Component Memoization

**Files:**
- Modify: `frontend/src/context/WorkbenchContext.tsx`
- Modify: `frontend/src/hooks/useWebSocket.ts`
- Modify: `frontend/src/hooks/useTrainingStream.ts`
- Modify: `frontend/src/components/layout/WorkbenchShell.tsx`
- Modify: `frontend/src/views/TrainingView.tsx`
- Modify: `frontend/src/components/charts/LineChartSVG.tsx`
- Modify: `frontend/src/components/board/BoardCanvas.tsx`
- Modify: `frontend/src/components/brain/BrainInspectorPane.tsx`

- [ ] **Step 1: Unify WebSocket connection & prevent multiple socket creations**

In `frontend/src/context/WorkbenchContext.tsx` and `useTrainingStream.ts`:
- Make WebSocket connection a single managed instance.
- In `useTrainingStream.ts`, consume the shared telemetry stream or API client rather than spawning an independent WebSocket.

- [ ] **Step 2: Memoize visual components with `React.memo`**

In `BoardCanvas.tsx`, `BrainInspectorPane.tsx`, `StudioHeader.tsx`, `ScrubberTransportBar.tsx`:
- Wrap components in `React.memo` with appropriate prop comparisons so that telemetry stream dispatches do not trigger re-renders of the game board or brain inspector.

- [ ] **Step 3: Throttle chart updates and optimize `LineChartSVG`**

In `TrainingView.tsx` and `LineChartSVG.tsx`:
- Throttle telemetry buffer updates with `requestAnimationFrame` / 30ms window.
- Decimate historical points when length exceeds 100 points for chart rendering.
- Memoize SVG coordinate strings in `LineChartSVG`.

- [ ] **Step 4: Run frontend TypeScript check and build**

Run: `npm --prefix frontend run build`
Expected: Build succeeds with 0 errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "perf(frontend): decouple telemetry stream, memoize UI components, and throttle chart rendering"
```

---

### Task 6: Scientific Benchmark Verification & Comparative Analysis

**Files:**
- Create: `scripts/benchmark_optimized.py`
- Create: `scripts/benchmark_comparison.py`

- [ ] **Step 1: Create `scripts/benchmark_optimized.py` mirroring `benchmark_baseline.py`**
- [ ] **Step 2: Execute benchmark and generate `benchmark_optimized.json`**

Run: `PYTHONPATH=. ./.venv/bin/python scripts/benchmark_optimized.py`

- [ ] **Step 3: Create and run `scripts/benchmark_comparison.py`**

Compute exact percentage improvements across all metrics:
- Game Core (games/sec, ms/step)
- ObservationV1 (calls/sec, µs/call)
- ActionMasker (calls/sec, µs/call)
- RewardCalculator (calls/sec, µs/call)
- Masked PPO Training (FPS, ms/step)
- Masked DQN Training (FPS, ms/step)
- Tournament (games/sec)
- Full Test Suite Duration (seconds before vs after)

- [ ] **Step 4: Run full test suite and verify 100% tests pass**

Run: `./.venv/bin/pytest --durations=10`

- [ ] **Step 5: Commit benchmark reports and artifacts**

```bash
git add scripts/ benchmark_optimized.json
git commit -m "test(perf): add comparative benchmark suite and verify percentage performance gains"
```
