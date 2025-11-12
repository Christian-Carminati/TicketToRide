# Architecture Overview

This document describes the architecture and design patterns used in the Ticket to Ride AI Toolkit.

## System Architecture

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  (best_solution/main.py, rl/train_*.py)                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                   Domain Layer                          │
│  (game.py, map.py)                                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                    Core Layer                           │
│  (core/graph.py, core/routes.py, core/scoring.py)      │
└─────────────────────────────────────────────────────────┘
```

## Module Organization

### Core Module (`src/core/`)

The core module provides shared utilities used across all solvers and environments:

- **`constants.py`**: Game constants (TOTAL_TRAINS, POINTS_TABLE, LONGEST_PATH_BONUS)
- **`graph.py`**: Graph loading utilities from CSV/JSON files
- **`routes.py`**: Route extraction and validation functions
- **`scoring.py`**: Scoring calculations (route points, longest path, tickets)
- **`solution.py`**: Solution dataclass shared across modules

**Design Principle**: No dependencies on other project modules, pure utility functions.

### Best Solution Module (`src/best_solution/`)

Contains classical optimization algorithms:

- **`heuristic_solver.py`**: Multiple heuristic algorithms (Greedy, Local Search, SA, Tabu, GA)
- **`optimal_solver.py`**: Branch and Bound for optimal solutions
- **`main.py`**: Comparison and benchmarking script
- **`solution.py`**: Compatibility wrapper for Solution dataclass

**Design Principle**: Algorithms depend on core utilities, not on each other.

### RL Module (`src/rl/`)

Reinforcement Learning components:

- **`envs.py`**: Gymnasium and PettingZoo environments
- **`opponents.py`**: Scripted opponent policies
- **`scoring.py`**: RL-specific scoring utilities
- **`train_sb3.py`**: Stable Baselines3 training script
- **`train_rllib.py`**: RLlib multi-agent training script

**Design Principle**: Environments use core utilities for scoring, opponents can use heuristic solvers.

### Game Module (`src/game.py`)

Full game simulation with:
- Player management
- Card deck and hand management
- Route claiming logic
- Destination ticket validation
- Game state tracking

**Design Principle**: Complete game logic, can be used independently or by RL environments.

## Design Patterns

### 1. Shared Utilities Pattern

Common functionality (scoring, route extraction) is centralized in `core/` to avoid duplication:

```python
# Before refactoring: Duplicated in multiple files
def calculate_route_points(routes):
    return sum(POINTS_TABLE[r[2]] for r in routes)

# After refactoring: Single source of truth
from core.scoring import calculate_route_points
```

### 2. Strategy Pattern

Different solver algorithms implement the same interface:

```python
class HeuristicSolver:
    def greedy_solve(self) -> Solution: ...
    def local_search_solve(self) -> Solution: ...
    def simulated_annealing_solve(self) -> Solution: ...
```

### 3. Factory Pattern

Opponent policies are created via factory functions:

```python
def build_opponent(name: str) -> OpponentPolicyProtocol:
    if name == "greedy":
        return GreedyOpponent()
    elif name == "random":
        return RandomOpponent()
    ...
```

## Data Flow

### Solver Execution Flow

```
1. Load Map Data (CSV/JSON)
   ↓
2. Build NetworkX Graph
   ↓
3. Extract Routes (core/routes.py)
   ↓
4. Run Solver Algorithm
   ↓
5. Evaluate Solution (core/scoring.py)
   ↓
6. Return Solution Object
```

### RL Training Flow

```
1. Create Environment (rl/envs.py)
   ↓
2. Agent Selects Action
   ↓
3. Environment Updates State
   ↓
4. Calculate Reward (core/scoring.py)
   ↓
5. Update Agent Policy
   ↓
6. Repeat for N Episodes
```

## Key Abstractions

### RouteTuple

Standard representation of a route:
```python
RouteTuple = Tuple[str, str, int, str]  # (city1, city2, length, color)
```

### Solution

Standard solution representation:
```python
@dataclass
class Solution:
    routes: List[RouteTuple]
    route_points: int
    longest_path_length: int
    longest_bonus: int
    ticket_points: int
    total_score: int
    trains_used: int
    computation_time: float
    algorithm: str
```

### ScoreBreakdown

Detailed score components:
```python
@dataclass(frozen=True)
class ScoreBreakdown:
    route_points: int
    longest_path_length: int
    longest_bonus: int
    ticket_points: int
    total_score: int  # Computed property
```

## Dependencies

### External Libraries

- **NetworkX**: Graph representation and algorithms
- **Pandas**: Data loading and manipulation
- **Matplotlib**: Visualization
- **Gymnasium**: Single-agent RL environments
- **PettingZoo**: Multi-agent RL environments
- **Stable Baselines3**: RL algorithms (PPO, DQN)
- **Ray RLlib**: Distributed RL training

### Internal Dependencies

```
best_solution/ → core/
rl/ → core/ + best_solution/
game.py → map.py
map.py → core/graph.py
```

## Extension Points

### Adding a New Solver

1. Add method to `HeuristicSolver` class
2. Use `core.scoring.score_routes()` for evaluation
3. Return `Solution` object
4. Add to `compare_all_heuristics()` function

### Adding a New RL Environment

1. Inherit from `gym.Env` or `AECEnv`
2. Use `core.scoring` for reward calculation
3. Implement `reset()` and `step()` methods
4. Register with training scripts

### Adding a New Opponent

1. Implement `OpponentPolicyProtocol`
2. Add factory function in `rl/opponents.py`
3. Use in `SingleAgentTicketToRideEnv`

## Performance Considerations

- **Graph Loading**: Cached after first load
- **Route Extraction**: O(E) where E is number of edges
- **Scoring**: O(V + E) for longest path calculation
- **Solver Complexity**: Varies by algorithm (see solver documentation)

## Testing Strategy

- Unit tests for core utilities
- Integration tests for solvers
- Performance benchmarks for algorithm comparison
- RL environment validation tests

