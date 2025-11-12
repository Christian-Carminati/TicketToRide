# Ticket to Ride AI Toolkit - Documentation

Welcome to the Ticket to Ride AI Toolkit documentation. This project provides a comprehensive toolkit for studying and automating Ticket to Ride (USA map) games using classical solvers and Reinforcement Learning agents.

## Documentation Index

### Core Documentation

- **[Architecture Overview](ARCHITECTURE.md)** - System architecture and design patterns
- **[Entity Relationship Diagram](ER_Diagram.md)** - Database/class structure visualization
- **[API Reference](API.md)** - Detailed API documentation for all modules

### Solver Documentation

- **[Heuristic Solvers](solver_heuristic.md)** - Greedy, Local Search, Simulated Annealing, Tabu Search, and Genetic Algorithms
- **[Optimal Solver](solver_optimal.md)** - Branch and Bound algorithm for guaranteed optimal solutions

### Reinforcement Learning

- **[RL Environments](RL_ENVIRONMENTS.md)** - Gymnasium and PettingZoo environment documentation
- **[Training Guide](RL_TRAINING.md)** - Guide for training RL agents with Stable Baselines3 and RLlib

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install networkx pandas matplotlib gymnasium pettingzoo \
  stable-baselines3[extra] ray[rllib] tensorboard
```

### Running Solvers

**Heuristic Solvers:**

```bash
python -m src.best_solution.main --solver greedy
python -m src.best_solution.main --solver tabu
python -m src.best_solution.main --solver simulated_annealing
```

**Optimal Solver:**

```bash
python -m src.best_solution.optimal_solver
```

**RL Training:**

```bash
python -m src.rl.train_sb3 --algorithm ppo --timesteps 200000
python -m src.rl.train_rllib --iterations 300
```

## Project Structure

```text
src/
├── core/              # Core utilities (graph, routes, scoring, constants)
├── best_solution/     # Classical solvers (heuristic and optimal)
├── rl/                # Reinforcement Learning environments and training
├── map/               # Map data files (cities, routes, tickets)
├── game.py            # Game state and logic
└── map.py             # Map loading and visualization
```

## Key Features

- **Multiple Solver Algorithms**: Greedy, Local Search, Simulated Annealing, Tabu Search, Genetic Algorithms, and Branch & Bound

- **Reinforcement Learning**: Single-agent and multi-agent (self-play) environments

- **Comprehensive Scoring**: Route points, longest path bonus, and destination ticket completion

- **Modular Design**: Clean separation of concerns with shared core utilities

## Contributing

When contributing to this project, please:

1. Follow the existing code structure and patterns
2. Add documentation for new features
3. Update relevant markdown files
4. Ensure type hints and docstrings are complete

## License

[Add your license information here]
