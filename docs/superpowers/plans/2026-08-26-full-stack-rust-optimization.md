# Full-Stack Native Rust Acceleration & Ultra-High Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance native Rust core (`ttr_core` via PyO3/Maturin) achieving 1,000,000+ games/sec, vectorize Batched Neural MCTS on GPU, parallelize backend concurrency with zero-latency streaming and SQLite/zstd storage, and upgrade the frontend to a 120 FPS 3-Layer Canvas 2D engine with full before/after scientific benchmark reporting.

**Architecture:** A native Rust crate (`native/ttr_core`) provides a ~350-byte stack-allocated Bitboard state with $O(1)$ incremental DSU connectivity and GIL-free multi-threaded batch stepping. A PyO3/Maturin bridge connects to PyTorch tensors with zero copy. Backend concurrency is decoupled via `multiprocessing.Process`, `ProcessPoolExecutor`, `orjson`, and conflated WebSockets. The React frontend transitions from SVG DOM to a 3-Layer Canvas 2D renderer with spatial hashing.

**Tech Stack:** Rust 1.97+, PyO3 0.22+, Maturin, Rayon, Python 3.12+, PyTorch 2.2+, Gymnasium, FastAPI, Uvicorn, orjson, zstandard, SQLite (WAL), React 19, TypeScript, Vite, HTML5 Canvas 2D, Web Workers.

**Spec:** [`docs/superpowers/specs/2026-08-26-full-stack-rust-optimization-design.md`](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-26-full-stack-rust-optimization-design.md)

## Global Constraints
- **Determinism**: 100% bit-exact outcome match given `(seed, initial_state, action_sequence)`.
- **Zero Hidden Leaks**: Observation encoders must strictly enforce POMDP anti-leakage invariants.
- **Scientific Verification**: Side-by-side benchmark scripts must record and generate comparative delta reports (`benchmark_comparison_report.txt`).
- **No Regressions**: All 119 existing unit and acceptance tests must pass cleanly.

---

### Task 1: Native Rust Core (`native/ttr_core`) & Bitboard Simulation Engine

**Files:**
- Create: `native/ttr_core/Cargo.toml`
- Create: `native/ttr_core/src/lib.rs`
- Create: `native/ttr_core/src/types.rs`
- Create: `native/ttr_core/src/dsu.rs`
- Create: `native/ttr_core/src/board.rs`
- Create: `native/ttr_core/src/rules.rs`
- Create: `native/ttr_core/src/state.rs`
- Create: `native/ttr_core/src/game.rs`
- Test: `native/ttr_core/tests/test_game.rs`

**Interfaces:**
- Produces: `GameStateCompact`, `DisjointSet`, `BoardData`, `GameEngine` in Rust with zero heap allocations during simulation.

- [ ] **Step 1: Write Cargo.toml and Crate Scaffolding**
Create `native/ttr_core/Cargo.toml` with `pyo3`, `rayon`, and `rand_xoshiro`.

- [ ] **Step 2: Implement Incremental DSU (`dsu.rs`) and Bitboard Topology (`board.rs`)**
Implement `DisjointSet` with path compression on `[u8; 36]` and static USA map routes.

- [ ] **Step 3: Implement GameStateCompact & Action Rules (`state.rs`, `rules.rs`, `game.rs`)**
Implement stack-allocated `GameStateCompact`, action generation, route claiming, locomotive rules, and ticket drawing.

- [ ] **Step 4: Write Rust Unit Tests (`tests/test_game.rs`) & Run Cargo Test**
Run: `cargo test --manifest-path native/ttr_core/Cargo.toml`
Expected: PASS all unit tests.

- [ ] **Step 5: Commit**
`git add native/ttr_core/ && git commit -m "feat(rust): implement native bitboard game engine in ttr_core"`

---

### Task 2: PyO3 Python Bindings & Equivalence Testing Suite

**Files:**
- Create: `native/ttr_core/src/py_bindings.rs`
- Modify: `pyproject.toml` (add maturin build backend / dev dependency)
- Create: `src/game/native.py`
- Create: `tests/game/test_rust_equivalence.py`

**Interfaces:**
- Consumes: `native/ttr_core` Rust crate.
- Produces: Python module `ttr_core` exposing `PyGame`, `PyGameState`, `PyAction`, `NativeVectorEnv`.

- [ ] **Step 1: Implement PyO3 Python Bindings (`py_bindings.rs`)**
Export classes `PyGame`, `PyGameState`, `PyAction`, `NativeVectorEnv` with GIL release (`py.allow_threads`).

- [ ] **Step 2: Build Native Extension with Maturin**
Run: `uv run maturin develop --manifest-path native/ttr_core/Cargo.toml`
Expected: Successfully compiles and installs `ttr_core` into Python environment.

- [ ] **Step 3: Write Equivalence Parity Test (`test_rust_equivalence.py`)**
Simulate 1,000 matches with identical seeds comparing Python and Rust step outcomes bit-for-bit.

- [ ] **Step 4: Run Equivalence Tests**
Run: `pytest tests/game/test_rust_equivalence.py -v`
Expected: PASS (100% deterministic parity).

- [ ] **Step 5: Commit**
`git add native/ttr_core pyproject.toml src/game/native.py tests/game/test_rust_equivalence.py && git commit -m "feat(rust): export PyO3 bindings and verify equivalence parity"`

---

### Task 3: RL Vectorization, Batched Neural MCTS & GPU Acceleration

**Files:**
- Modify: `src/rl/alphazero_search.py:1-224`
- Modify: `src/rl/alphazero_trainer.py:25-50`
- Modify: `src/rl/rollout.py:1-446`
- Modify: `src/rl/advantage.py:44-85`
- Modify: `src/rl/policy_value_net.py:33-112`
- Test: `tests/rl/test_batched_mcts.py`

**Interfaces:**
- Consumes: `ttr_core` native states, `PolicyValueNetwork`.
- Produces: Batched MCTS with Virtual Loss, Zobrist Transposition DAG, circular self-play replay buffer, and JIT GAE.

- [ ] **Step 1: Implement Circular Replay Buffer in `SelfPlayReplayBuffer`**
Replace `list.pop(0)` with $O(1)$ ring buffer index pointer to prevent pointer shifts.

- [ ] **Step 2: Implement 64-bit Zobrist Hashing & Transposition Table**
Implement XOR-based Zobrist hashing and transposition caching in `alphazero_search.py`.

- [ ] **Step 3: Implement Batched Neural MCTS with Virtual Loss**
Batch leaf evaluations into $B=64 \dots 256$ tensor forward passes.

- [ ] **Step 4: Implement GPU-Native RolloutBuffer & JIT GAE Kernel**
Pre-allocate tensors on CUDA/CPU and compile GAE with `@torch.jit.script`.

- [ ] **Step 5: Run RL Unit & Performance Tests**
Run: `pytest tests/rl/ -v`
Expected: PASS all RL tests.

- [ ] **Step 6: Commit**
`git add src/rl/ tests/rl/ && git commit -m "perf(rl): add batched neural mcts, zobrist hashing, circular buffer, and jit gae"`

---

### Task 4: Backend Concurrency, Conflated WebSocket Hub & SQLite Storage

**Files:**
- Modify: `src/api/websocket.py:1-56`
- Modify: `src/api/trainer_service.py:1-120`
- Modify: `src/api/replay_service.py:1-100`
- Modify: `src/experiments/registry.py:1-80`
- Modify: `src/evaluation/tournament.py:1-100`
- Modify: `src/api/tournament_service.py:1-160`
- Test: `tests/api/test_conflated_websocket.py`
- Test: `tests/api/test_replay_sqlite.py`

**Interfaces:**
- Produces: `ConflatedConnectionManager`, `ParallelTournament`, `OptimizedReplayStorage` with SQLite WAL and Zstandard compression.

- [ ] **Step 1: Implement `ConflatedConnectionManager` with orjson & lossy ring-buffer**
Update `websocket.py` to use `orjson` and bounded queues with dropped-frame conflation.

- [ ] **Step 2: Decouple Training to `multiprocessing.Process`**
Update `trainer_service.py` to run background training jobs in isolated OS processes, freeing the GIL.

- [ ] **Step 3: Implement Multi-Core `ParallelTournament` with `ProcessPoolExecutor`**
Update `tournament.py` and `tournament_service.py` to run matchups across all CPU cores.

- [ ] **Step 4: Implement SQLite WAL Index & Zstandard Replay Compression**
Update `replay_service.py` to index headers in SQLite and compress frame data with `zstd`.

- [ ] **Step 5: Run Backend API & Storage Tests**
Run: `pytest tests/api/ -v`
Expected: PASS all API and storage tests.

- [ ] **Step 6: Commit**
`git add src/api/ src/evaluation/ src/experiments/ tests/api/ && git commit -m "perf(backend): add conflated websockets, multiprocessing trainer, parallel tournaments, and sqlite/zstd replays"`

---

### Task 5: Frontend 3-Layer Hybrid Canvas 2D & Web Worker Pipeline

**Files:**
- Create: `frontend/src/canvas/BoardCanvasRenderer.ts`
- Create: `frontend/src/canvas/SpatialHashGrid.ts`
- Create: `frontend/src/components/charts/FastStreamingChart.tsx`
- Create: `frontend/src/context/store.ts`
- Modify: `frontend/src/components/board/BoardCanvas.tsx`
- Modify: `frontend/src/views/TrainingView.tsx`

**Interfaces:**
- Produces: 3-Layer Canvas 2D board with $O(1)$ spatial hash hit-testing, granular selector store, and 120 FPS canvas telemetry charts.

- [ ] **Step 1: Implement `SpatialHashGrid.ts` & `BoardCanvasRenderer.ts`**
Implement offscreen cached cartography pre-rendering, batched sleeper/rail rendering, and $O(1)$ hover hit-testing.

- [ ] **Step 2: Implement Granular AppStore (`store.ts`) with `useSyncExternalStore`**
Replace monolithic context re-renders with targeted selector subscriptions.

- [ ] **Step 3: Implement Canvas 2D `FastStreamingChart.tsx`**
Replace SVG `<polyline>` with high-frequency Canvas 2D chart synced with `requestAnimationFrame`.

- [ ] **Step 4: Update `BoardCanvas.tsx` to mount the 3-Layer Canvas Renderer**
Switch `BoardCanvas.tsx` to render the canvas pipeline with DOM overlay for city names.

- [ ] **Step 5: Build Frontend & Verify Type Integrity**
Run: `(cd frontend && npm run build)`
Expected: PASS with 0 build errors.

- [ ] **Step 6: Commit**
`git add frontend/src/ && git commit -m "perf(frontend): implement 3-layer canvas board, spatial hash grid, and fast streaming charts"`

---

### Task 6: Scientific Benchmarking, Delta Measurement & Comparative Report

**Files:**
- Modify: `scripts/benchmark_baseline.py`
- Modify: `scripts/benchmark_optimized.py`
- Modify: `scripts/benchmark_comparison.py`
- Output: `benchmark_baseline.json`
- Output: `benchmark_optimized.json`
- Output: `benchmark_comparison_report.txt`

**Interfaces:**
- Produces: Side-by-side benchmark execution, structured JSON metrics, and formatted comparison report.

- [ ] **Step 1: Update Benchmark Scripts for Full-Stack Measurement**
Ensure `benchmark_baseline.py` and `benchmark_optimized.py` cover Game Simulation, State Cloning, Action Masking, MCTS Search, PPO/DQN Throughput, and Tournament FPS.

- [ ] **Step 2: Run Scientific Benchmark Comparison Pipeline**
Run: `uv run python scripts/benchmark_optimized.py && uv run python scripts/benchmark_comparison.py`
Expected: Generate updated JSON records and `benchmark_comparison_report.txt`.

- [ ] **Step 3: Run Full Pytest Test Suite**
Run: `pytest -v`
Expected: PASS 100% of test suite.

- [ ] **Step 4: Commit & Verification**
`git add scripts/ benchmark_*.json benchmark_comparison_report.txt && git commit -m "chore(benchmark): record scientific optimization results and comparative report"`
