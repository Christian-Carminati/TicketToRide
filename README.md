# 🎫 TicketToRide RL Lab

**TicketToRide RL Lab** is an experimental reinforcement learning laboratory and game engine built from scratch in Python and TypeScript.

The goal of the project is not simply "a board game with an AI bot", but a complete **research and experimentation environment** to build, train, inspect, benchmark, and evaluate Reinforcement Learning agents across diverse paradigms:

* **Algorithms**: DQN, PPO, Recurrent PPO (LSTM), Self-Play, MCTS, Neural MCTS (AlphaZero-style)
* **Mechanisms**: Action Masking, Reward Shaping, Partial Observability (POMDP), Opponent Modeling, Curriculum Learning, Procedural Map Generalization
* **Introspection & Web Lab**: Live training streaming, neural policy visualization, interactive game replay, and tournament benchmarking

---

## 🏛 Architecture

```text
                           ┌───────────────────────┐
                           │       WEB LAB         │
                           │  Board / Brain / Replay│
                           └───────────┬───────────┘
                                       │ WebSocket / REST API
                           ┌───────────▼───────────┐
                           │       API LAYER       │
                           └───────────┬───────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
       ┌────────────┐           ┌─────────────┐          ┌────────────┐
       │ GAME CORE  │           │  RL ENGINE  │          │ EVALUATION │
       │ Pure Rules │           │ PPO/DQN/MCTS│          │ Tournament │
       └────────────┘           └─────────────┘          └────────────┘
```

### Key Modules:
- `src/game/`: Pure deterministic game engine (Board, Route, Card, Ticket, Player, GameState, Rules). Zero ML/web dependencies.
- `src/environment/`: Gymnasium wrapper, observation encoders (V1, V2, ...), action masking, reward calculators.
- `src/agents/`: Agent implementations (`RandomAgent`, `GreedyAgent`, `StrategicHeuristicAgent`, `DQNAgent`, `PPOAgent`, `MCTSAgent`).
- `src/rl/`: Neural networks, replay buffers, rollouts, advantage estimation (GAE), PPO/DQN algorithms, self-play pool.
- `src/evaluation/`: Tournaments, Elo rating system, head-to-head Evaluator, generalization metrics.
- `src/experiments/`: Config validation, experiment runner, reproducibility tracking.
- `src/api/`: FastAPI + WebSockets for real-time telemetry and control.
- `frontend/`: React + TypeScript + Canvas/SVG web lab for live inspection.

---

## 🚀 Quick Start

### Python Environment (Python 3.12+)

```bash
# Set up virtual environment
python3 -m venv venv_py312
source venv_py312/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### Running Head-to-Head Evaluations & Tournaments

```bash
# Evaluate Strategic vs Greedy head-to-head across 100 games
python scripts/evaluate.py --agent1 strategic --agent2 greedy --games 100 --seed 42

# Run a round-robin tournament among all baseline agents
python scripts/tournament.py --agents random,greedy,strategic --games-per-pair 50 --seed 42
```

### Running Tests

```bash
pytest
```

### Frontend Web Lab

```bash
cd frontend
npm install
npm run dev
```

---

## 🗺 Development Roadmap

- [x] **Phase 0 — Project Skeleton & Foundations**
- [x] **Phase 1 — Game Core** (Deterministic engine, rules, tickets, 100% test invariants)
- [x] **Phase 2 — Baseline Agents & Tournament System** (`RandomAgent`, `GreedyAgent`, `StrategicHeuristicAgent`, `Evaluator`, `Tournament`, `EloSystem`)
- [x] **Phase 3 — Gymnasium Environment & Action Masking** (`TicketToRideEnv`, `ObservationV1`, `ActionMasker`, `RewardV1`)
- [ ] **Phase 4 — First RL (DQN & PPO baseline)**
- [ ] **Phase 5 — Web Lab (Interactive Viewer & Brain Introspection)**
- [ ] **Phase 6 — Custom PPO From Scratch**
- [ ] **Phase 7 — Reward Engineering Experiments**
- [ ] **Phase 8 — Partial Observability & Recurrent PPO (LSTM)**
- [ ] **Phase 9 — Self-Play & Historical Policy Pool**
- [ ] **Phase 10 — Generalization & Procedural Maps**
- [ ] **Phase 11 — Monte Carlo Tree Search (MCTS)**
- [ ] **Phase 12 — Neural MCTS & Advanced Research**

---

## 📖 Documentation

- [Design & Specification](DESIGN.md)
- [Phase 2 Baseline Agents Design](docs/superpowers/specs/2026-08-18-phase-2-baseline-agents-design.md)
- [Phase 2 Implementation Plan](docs/superpowers/plans/2026-08-18-phase-2-baseline-agents.md)
- [Agent & Contributor Rules](CLAUDE.md)
