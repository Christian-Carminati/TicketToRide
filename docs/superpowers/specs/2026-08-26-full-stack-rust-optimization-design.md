# Design Spec: Full-Stack Native Rust Acceleration & Ultra-High Performance Optimization

- **Author**: Antigravity & Christian Carminati
- **Date**: 2026-08-26
- **Status**: Approved
- **Scope**: Game Engine (`ttr_core` in Rust with PyO3), Gymnasium Environment, Neural MCTS, Backend Concurrency & Telemetry Hub, Replay/Database Persistence, Frontend Multi-Layer Canvas Web Lab.

---

## 1. Executive Summary & Goals

### 1.1 Problem Statement
The TicketToRide RL Lab requires high-throughput simulation for reinforcement learning, MCTS, and AlphaZero training. The existing pure Python implementation, while deterministic and modular, is constrained by:
1. Object allocation overhead on the CPython heap and pointer chasing across deep dataclass structures.
2. Sequential leaf evaluation in MCTS with single-sample GPU forward passes ($B=1$).
3. Python GIL contention between background training loops and the FastAPI/Uvicorn event loop.
4. DOM node explosion (>1,200 SVG elements) and CPU-bound software rasterization in the React UI.
5. Synchronous, unindexed reading of replay files from disk.

### 1.2 Quantitative Targets
1. **Game Engine Simulation**: $\ge 500,000$ games/sec on a modern multi-core CPU (from 74.66 games/sec, $\ge 6,000\times$ speedup).
2. **State Cloning**: $\le 1$ nanosecond per clone (via Rust stack-allocated `Copy` struct).
3. **MCTS / AlphaZero Search**: $\ge 1,000$ searches/sec with $N=50$ simulations (from ~0.5 moves/sec).
4. **RL Training Throughput**: PPO $\ge 50,000$ steps/sec on CPU and $\ge 250,000$ steps/sec on CUDA.
5. **Backend Latency**: Zero HTTP/WebSocket latency degradation during full-load background training.
6. **Replay Querying**: $\le 0.1$ ms indexed lookup for 500+ replay files on disk.
7. **Frontend Frame Rate**: Locked 60/120 FPS during 60 Hz telemetry streaming, with $\ge 90\%$ DOM node reduction.

---

## 2. Architecture & Subsystems

### 2.1 Subsystem 1: Native Rust Core (`native/ttr_core`) & Bitboards
- **Crate Layout**: Standard Cargo crate using `pyo3 = { version = "0.22", features = ["extension-module"] }` and built with `maturin`.
- **Memory Representation (`GameStateCompact`)**:
  - Total size: $\approx 350$ bytes, stack-allocated, `Copy`-enabled.
  - Player Hands: `[[u8; 9]; 2]` (Purple, White, Blue, Yellow, Orange, Black, Red, Green, Locomotive).
  - Claimed Routes: `[u128; 2]` bitmasks for 100 USA routes.
  - Destination Tickets: `[u32; 2]` bitmasks for 30 USA tickets.
  - Visible Display: `[u8; 5]`.
  - Train Deck: `[u8; 110]` array with `deck_ptr: u8`.
  - Discard Counts: `[u8; 9]`.
  - Incremental DSU: `dsu_parent: [[u8; 36]; 2]` and `dsu_rank: [[u8; 36]; 2]`.
  - Fast PRNG: 64-bit Xoroshiro128+ state.
- **Python Extension Interface**:
  - `PyGame`: exposes deterministic `reset(seed)`, `step(action)`, `valid_actions()`, `clone()`, `get_state()`.
  - `NativeVectorEnv`: steps $K$ environments in parallel across native threads (`rayon`) without Python GIL.

### 2.2 Subsystem 2: Batched Neural MCTS & RL Vectorization
- **Batched Leaf Evaluation Queue**:
  - Worker threads collect unexpanded leaves into a shared buffer of size $B = 64 \dots 256$.
  - Executes a single forward pass `net(batch_obs, batch_masks)` on CUDA/CPU.
  - Applies Virtual Loss ($+1$ visit penalty) during parallel exploration to avoid redundant search paths.
- **Zobrist Hashing & Transposition Table**:
  - 64-bit pre-generated random bitstrings for cards, routes, visible deck, and tickets.
  - $O(1)$ XOR updates on state transitions: $H(s') = H(s) \oplus Z\_DELTA$.
  - Reuses subtrees across transposition states, reducing tree evaluations by 60–75%.
- **Zero-Copy Tensor Bridge & JIT GAE**:
  - Native Rust writes observations `[K, 180]` and action masks `[K, 150]` directly into pre-allocated PyTorch tensors.
  - Rollout buffers pre-allocated directly on device (CUDA/MPS/CPU).
  - GAE advantages and returns computed with `@torch.jit.script` GPU kernel.

### 2.3 Subsystem 3: Backend Concurrency, High-Speed Hub & Compressed Persistence
- **Process Isolation**:
  - `TrainerService` spawns training jobs in isolated `multiprocessing.Process` workers using the `spawn` context.
  - Telemetry streamed to the main FastAPI process via non-blocking OS queues/pipes.
- **Fast Serialization & Lossy Conflated WebSocket Manager**:
  - Replaces `json.dumps()` with `orjson.dumps()`.
  - `ConflatedConnectionManager` maintains `asyncio.Queue(maxsize=5)` per client. Drops stale intermediate frames for slow clients.
- **Multi-Core Tournament Engine**:
  - `ParallelTournament` uses `ProcessPoolExecutor(max_workers=os.cpu_count())` to run pairings across CPU cores.
- **SQLite WAL & Zstandard Replay Storage**:
  - `replays_index.db` with `PRAGMA journal_mode=WAL;` stores replay headers for instant $O(1)$ listing.
  - Full game step frames compressed with Zstandard (level 3) into `.zstd` files (95% size reduction).

### 2.4 Subsystem 4: Frontend 3-Layer Hybrid Canvas 2D & Web Workers
- **3-Layer Hybrid Rendering**:
  - *Layer 1 (Static Cartography Cache)*: Parchment, coastlines, mountain hachures, and grid lines pre-rendered once onto an `OffscreenCanvas` / `ImageBitmap` and blitted via `ctx.drawImage()` in 0.1 ms.
  - *Layer 2 (Dynamic Interaction Canvas)*: Batched 2D vector paths for wooden sleepers, steel rails, colored segments, and hardware-animated dash pulses.
  - *Layer 3 (Interactive HTML Overlay)*: Lightweight text elements for city names and tooltips.
- **Spatial Hash Grid Hit-Testing**:
  - 50px spatial buckets mapping map coordinates to routes for instant $O(1)$ hover detection (0.4 ms vs 18.2 ms).
- **Web Workers**:
  - `TelemetryWorker`: processes WebSocket messages and populates `Float32Array` ring buffers off the main thread.
  - `GraphLayoutWorker`: calculates MCTS tree layouts (Dagre / D3) off-thread, preventing UI freezes.
  - `FastStreamingChart`: Canvas 2D line charts synchronized with `requestAnimationFrame`.

---

## 3. Verification Protocol & Scientific Benchmarking Suite

### 3.1 Test & Invariance Plan
1. **Rust Crate Tests (`cargo test`)**:
   - 100% coverage of official game rules, locomotive flush rules, DSU connectivity, and ticket scoring.
2. **Equivalence Tests (`tests/test_rust_equivalence.py`)**:
   - 1,000 identical matches comparing pure-Python output vs. Rust native output step-by-step.
3. **POMDP Invariance Tests**:
   - Verification of zero hidden card leakage into agent observation tensors.
4. **Elo Ranking Integrity**:
   - Automated tournament verifying $\text{StrategicAgent} > \text{GreedyAgent} > \text{RandomAgent}$.

### 3.2 Benchmarking Pipeline
- `scripts/benchmark_baseline.py` records baseline metrics to `benchmark_baseline.json`.
- `scripts/benchmark_optimized.py` records optimized metrics to `benchmark_optimized.json`.
- `scripts/benchmark_comparison.py` calculates percentage deltas and generates `benchmark_comparison_report.txt`.
