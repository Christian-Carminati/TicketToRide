# TicketToRide RL Lab — Development Guidelines

This document outlines core principles, development workflow, and rules for agents and contributors working on **TicketToRide RL Lab**.

---

## 1. Project Mission & Identity

TicketToRide RL Lab is **not** a board game with a simple bot. It is an **experimental laboratory for Reinforcement Learning (RL)** built from scratch to investigate and benchmark:
- DQN, PPO, Recurrent PPO (LSTM)
- Partial Observability / POMDP
- Reward Engineering & Shaping
- Action Masking
- Self-Play & Historical Policy Pools
- Opponent Modeling
- Curriculum Learning & Procedural Maps
- MCTS & Neural MCTS (AlphaZero-style)
- Live Introspection, Metrics & Replay Visualization

---

## 2. Core Architectural Principles

1. **RL-First**: The game core is designed from the ground up to serve as a high-performance, deterministic RL environment.
2. **Learning-First**: Implement algorithms and components with clarity and pedagogical value. No black boxes without reason.
3. **Deterministic Core**: Given `(seed, initial_state, action_sequence)`, execution must be 100% deterministic.
4. **Strict Separation of Concerns**:
   - `Game Core` (`src/game/`): Pure Python rules, board, state, actions. No dependencies on `torch`, `gymnasium`, `fastapi`, or `frontend`.
   - `Environment` (`src/environment/`): Gymnasium wrapper, state-to-observation encoding, action mapping, action masking, reward calculators.
   - `Agents` (`src/agents/`): Random, Heuristic, DQN, PPO, MCTS agent abstractions.
   - `RL Engine` (`src/rl/`): Networks, buffers, GAE, PPO, DQN, self-play algorithms.
   - `Evaluation` (`src/evaluation/`): Tournament runners, Elo calculation, win rates, generalization benchmarks.
   - `Experiments` (`src/experiments/`): Configuration schemas, runner, registry, reproducibility metadata.
   - `API & Web Lab` (`src/api/`, `frontend/`): Visualization, real-time WebSocket streaming, replay analysis. The game and training run completely headless without web dependencies.
5. **No Hidden Information Leaks**: Opponent hidden cards/tickets or deck order must NEVER leak into agent observations.

---

## 3. Phased Roadmap

We follow strict phased incremental development:
- **Phase 0 — Project Skeleton & Foundations** (Current)
- **Phase 1 — Game Core** (Pure game logic, rules, validation, deterministic seed, serialization)
- **Phase 2 — Baseline Agents** (Random, Greedy, Heuristic, Tournament benchmark)
- **Phase 3 — Gymnasium Environment** (Observations, action space, action masking, reward module)
- **Phase 4 — First RL** (DQN & PPO baseline validation)
- **Phase 5 — Web Lab** (Board viewer, agent brain viewer, live training stream, replay player)
- **Phase 6 — PPO From Scratch** (Custom Actor-Critic, GAE, clipped objective, entropy)
- **Phase 7 — Reward Research** (A/B testing reward versions and behavioral analysis)
- **Phase 8 — Partial Observability & Recurrent PPO** (POMDP benchmarking, LSTM PPO)
- **Phase 9 — Self Play** (Historical policy pool, matchmaking, Elo tracking)
- **Phase 10 — Generalization & Procedural Maps** (Map generators, unseen map testing)
- **Phase 11 — MCTS** (Selection, expansion, rollout, backprop)
- **Phase 12 — Advanced Research** (Neural MCTS, opponent modeling)

---

## 4. Development Rules for AI Agents

1. **One Phase at a Time**: Never implement multiple phases at once unless explicitly requested.
2. **Small, Testable Changes**: Every new module or feature must be accompanied by unit tests.
3. **No Premature RL**: Do not start RL implementations before the Game Core and Environment are fully validated.
4. **Reproducibility**: Experiments must be configuration-driven with explicit seeds, metrics, and artifact tracking.
5. **Debugging Priority**: When an RL agent fails to learn, debug in order:
   `Environment -> Action Space -> Action Masking -> Observation -> Reward -> Termination -> Algorithm -> Network -> Hyperparameters`.
