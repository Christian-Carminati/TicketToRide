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
- `src/agents/`: Agent implementations (Random, Heuristic, DQN, PPO, MCTS).
- `src/rl/`: Neural networks, replay buffers, rollouts, advantage estimation (GAE), PPO/DQN algorithms, self-play pool.
- `src/evaluation/`: Tournaments, Elo rating system, generalization metrics.
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
- [ ] **Phase 1 — Game Core**
- [ ] **Phase 2 — Baseline Agents**
- [ ] **Phase 3 — Gymnasium Environment & Action Masking**
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
- [Agent & Contributor Rules](CLAUDE.md)
