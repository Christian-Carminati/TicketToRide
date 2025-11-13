# Ticket to Ride AI Toolkit

Comprehensive toolkit to simulate Ticket to Ride (USA map), experiment with classical solvers, and train reinforcement learning agents that can plan routes, manage cards, and compete against scripted opponents.

## Quick Start

```bash
git clone https://github.com/<your-account>/TicketToRide.git
cd TicketToRide
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
# (optional) pip install -r requirements.txt
```

### Recommended Dependencies (Linux/macOS)

```bash
pip install networkx pandas numpy gymnasium pettingzoo supersuit \
  stable-baselines3[extra] tensorboard ray[rllib] matplotlib
```

> Tip: run `PYTHONPATH=src` from the project root when launching scripts, or work directly inside the `src/` directory.

## Main Features

- **Full game simulation** (`src/game.py`, `src/map/`): official rules, card management, tickets, and scoring.
- **Classical solvers** (`src/best_solution/`): heuristics, tabu search, simulated annealing, genetic algorithms, and branch & bound.
- **Reinforcement learning module** (`src/rl/`):
  - Gymnasium and PettingZoo environments with consistent action masking.
  - Scripted opponents (`opponents.py`) to benchmark agents.
  - Training scripts for Stable-Baselines3 (`train_full_game.py`, `train_sb3.py`) and RLlib (`train_rllib.py`).
  - Evaluation tools (`eval_full_game.py`) and shared scoring utilities (`scoring.py`).
- **Technical documentation** (`docs/`): diagrams, architecture notes, and tuning guides.

## Tools Used

- Python 3.10+
- Gymnasium & PettingZoo for environment interfaces
- Stable-Baselines3 and RLlib for RL algorithms
- NetworkX, NumPy, and pandas for graph operations and data handling
- TensorBoard and Matplotlib for logging and visualization

## Training an RL Agent (PPO by default)

```bash
cd src
PYTHONPATH=. python -m rl.train_full_game \
  --algorithm ppo \
  --timesteps 200000 \
  --opponent greedy \
  --log-dir runs/full_game
```

Key details:

- `FullGameSingleAgentEnv` handles draws correctly (one call equals one turn) and discourages random card collection that does not unlock new routes.
- Reward shaping rewards efficient routes, long connections, completed tickets, and the final score differential.
- Checkpoints are saved in `runs/full_game/checkpoints/`, with the best model stored in `runs/full_game/best_model/`.

To monitor training progress:

```bash
tensorboard --logdir runs/full_game/tensorboard
```

## Evaluating a Trained Model

```bash
cd src
PYTHONPATH=. python -m rl.eval_full_game \
  --model-path runs/full_game/final_model.zip \
  --episodes 20 \
  --render 0
```

The report summarizes average rewards, player vs. opponent scores, and win rate.

## Playing with Classical Solvers

```bash
cd src
PYTHONPATH=. python -m best_solution.main --solver greedy
PYTHONPATH=. python -m best_solution.main --solver tabu
```

Each solver prints the chosen routes, final score, and computation statistics.

## Project Structure

```text
src/
  best_solution/    # Classical and optimal solvers
  core/             # Shared constants and utilities
  map/              # USA map dataset + loader
  rl/               # Environments, scoring, RL training scripts
  runs/             # Example checkpoints and logs
docs/
  ARCHITECTURE.md   # High-level architecture
  RL_FULL_GAME.md   # Full-game RL environment notes
  RL_FINE_TUNING.md # Hyperparameter and reward tuning tips
```

## Helpful Resources

- `docs/RL_FULL_GAME.md`: environment observations, actions, and flow.
- `docs/RL_FINE_TUNING.md`: practical advice for tuning rewards and hyperparameters.
- `docs/VISUALIZATION.md`: tools to visualize simulated games.

Questions, ideas, or contributions? Open an issue or reach out—new strategies are always welcome!

