# TicketToRide RL Lab: Full-Stack Performance & Stability Optimization Spec

- **Author**: Antigravity & Christian Carminati
- **Date**: 2026-08-19
- **Status**: Approved
- **Scope**: Game Engine, RL Simulation (PPO/DQN), Backend WebSocket/Telemetry Streaming, Frontend React Performance & Re-render Prevention.

---

## 1. Executive Summary & Problem Statement

### 1.1 Symptoms
1. **Frontend Crash / High CPU Under Training**: When launching RL training with active WebSocket streaming, the browser CPU usage hits 100%, causing UI freezing, frame drops, and browser crashes.
2. **Slow Simulation and Training**:
   - RL test suite took **166.57s** (nearly 3 minutes) for 119 tests.
   - Masked DQN training achieved only ~315 FPS due to repetitive Python allocations and per-step tensor reconstructions.
   - PPO training suffered from multi-pass allocations across minibatches in each training epoch.
   - Graph BFS traversals in observation encoding and reward calculations were executed repeatedly on every environment step.

### 1.2 Quantitative Baseline (Measured on 2026-08-19)
| Component | Metric Baseline | Bottleneck Cause |
| :--- | :--- | :--- |
| **Masked DQN Training** | **315.2 steps/s** (3.173 ms/step) | Dynamic `torch.tensor` creations from Python tuples inside `ReplayBuffer.sample()`. |
| **Masked PPO Training** | **1,034.9 steps/s** (0.966 ms/step) | Re-array conversions per epoch in `RolloutBuffer`, per-step tensor instantiation. |
| **Game Core Simulation** | **75.45 games/s** (USA map) | BFS graph traversals and repetitive list/dict allocations in `valid_actions`. |
| **ObservationV1 Encode** | **53,273 calls/s** (18.77 µs/call) | Dynamic graph building and BFS check on string city names for ticket completion. |
| **Tournament (300 games)** | **59.91 games/s** (5.007s) | Sequential agent action evaluation and path recalculations. |
| **Total Test Suite** | **166.57s** | Heavy training runs in test cases without optimized fast paths. |
| **Frontend WebSockets** | **2 active sockets** + **Full Tree Re-render** | Multiple sockets + global context re-renders on every telemetry packet. |

---

## 2. Architecture & Subsystem Redesign

### 2.1 Frontend Architecture & Re-render Isolation
- **Single WebSocket Connection**:
  - Unify WebSocket management into a dedicated singleton service / hook within `WorkbenchProvider`.
  - Eliminate duplicate connection in `useTrainingStream` / `WorkbenchShell`.
- **Decoupled State Management**:
  - Split high-frequency telemetry state from structural workspace state.
  - Wrap `BoardCanvas`, `BrainInspectorPane`, `ScrubberTransportBar`, and `StudioHeader` with `React.memo` and scoped selectors.
  - When live training is streaming, non-training visual panes receive **0 re-renders**.
- **Throttling & Decimated SVG Rendering**:
  - Throttle chart state updates in `TrainingView` via `requestAnimationFrame` / 30ms throttle.
  - In `LineChartSVG`, memoize path and axis calculations to prevent unnecessary SVG recomputations.

### 2.2 Backend Telemetry Streaming & Rate Limiting
- **Adaptive Telemetry Throttler**:
  - In `TrainerService`, cap WebSocket broadcast frequency to a maximum of 20 FPS (every 50ms) regardless of training speed (e.g. 5,000+ FPS).
  - Training loop executes at maximum bare-metal speed without artificial `time.sleep` bottlenecks, while UI receives clean, smooth, non-saturating updates.

### 2.3 RL Engine & Vectorized Buffers
- **Pre-allocated ReplayBuffer (DQN)**:
  - Replace `collections.deque[tuple]` with pre-allocated circular NumPy arrays:
    - `obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)`
    - `next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)`
    - `actions_buf = np.zeros(capacity, dtype=np.int64)`
    - `rewards_buf = np.zeros(capacity, dtype=np.float32)`
    - `dones_buf = np.zeros(capacity, dtype=bool)`
    - `masks_buf = np.zeros((capacity, action_dim), dtype=bool)`
  - Sampling utilizes random integer index slicing `np.random.randint`, converting to torch tensors in a single vector operation.
- **Optimized RolloutBuffer (PPO)**:
  - Pre-allocate contiguous rollout arrays for `rollout_steps`.
  - Eliminate multi-epoch array re-allocation in `generate_minibatches`.
  - In `MaskedActorCritic` / `MaskedQNetwork`, replace per-call `torch.tensor(-1e8)` allocations with scalar literals or masked fill.
- **Vectorized GAE**:
  - Optimize `compute_gae` to minimize indexing overhead and avoid redundant Python float conversions.

### 2.4 Game Core, Observation & Reward Fast-Paths
- **Integer & Bitset Graph Connectivity**:
  - Map cities to integer indices `0..N-1`.
  - Maintain player connectivity with an incremental Disjoint Set Union (DSU / Union-Find) or bitset adjacency representation so `check_ticket_completed` is `O(1)` or `O(alpha(V))`.
- **Fast Observation & Reward**:
  - In `ObservationV1.encode` and `DefaultRewardCalculator.calculate`, query the pre-indexed connectivity structure, eliminating dynamic BFS allocations over string names.
- **Action Validation Optimization**:
  - In `GameRules.get_valid_actions`, optimize route filtering and affordance loops to avoid temporary dictionary creations.

---

## 3. Performance Goals & Scientific Benchmarking Plan

### 3.1 Quantitative Targets
1. **PPO Training**: >= +50% FPS increase (> 1,500 FPS).
2. **DQN Training**: >= +150% FPS increase (> 800 FPS).
3. **Observation & Reward**: >= +50% throughput increase.
4. **Test Suite Duration**: Total pytest duration reduced by >= 50% (< 80s).
5. **Frontend Stability**:
   - Zero crashes under extended training runs.
   - CPU utilization drop from 100% to <= 20% on typical workloads.
   - UI frame rate maintained at a steady 60 FPS.

### 3.2 Verification Protocol
- `scripts/benchmark_baseline.py` and `scripts/benchmark_optimized.py` execute side-by-side comparative benchmarks outputting JSON records and exact percentage deltas.
- All 119 unit and acceptance tests must pass cleanly.
- Determinism invariants and Elo hierarchies (Strategic > Greedy > Random) must remain strictly preserved.
