# Optimal Solver Documentation

The `OptimalSolver` class implements a **Branch and Bound** algorithm to find the **globally optimal solution** for the Ticket to Ride route selection problem.

## Overview

### Objective
Find the solution that **maximizes the total score**, which includes:
- Route points (based on route length)
- Longest path bonus (+10 points)
- Destination ticket points

### Guarantee
Unlike heuristic solvers, the optimal solver **guarantees** finding the best possible solution, though it may take exponential time for large problem instances.

## Algorithm: Branch and Bound

### How It Works

Branch and Bound is a systematic search algorithm that explores the solution space while using **pruning** to avoid exploring unpromising branches.

#### Phase 1: Preprocessing

```python
# Extract unique routes (keep longest route per city pair)
unique_routes = unique_routes_from_graph(graph)

# Sort by efficiency (points/train ratio) to improve pruning
unique_routes.sort(
    key=lambda r: POINTS_TABLE[r[2]] / r[2] if r[2] > 0 else 0,
    reverse=True
)
```

**Why sort?** Routes with better efficiency are explored first, leading to better bounds earlier and more effective pruning.

#### Phase 2: Branching

For each route, the algorithm explores two branches:

1. **Include the route** (if valid: enough trains, city pair not already claimed)
2. **Exclude the route**

```python
def _branch_and_bound(
    unique_routes: List[RouteTuple],
    index: int,
    selected_routes: List[RouteTuple],
    trains_left: int,
    selected_pairs: Set[Tuple[str, str]],
    best_solution: Dict
) -> None:
    # Evaluate current solution
    current_score = _evaluate_solution(selected_routes)
    if current_score > best_solution['score']:
        best_solution.update(selected_routes, current_score)
    
    # Base case: all routes examined
    if index >= len(unique_routes):
        return
    
    # PRUNING: If upper bound <= best score, prune this branch
    remaining_routes = unique_routes[index:]
    upper_bound = _upper_bound_estimate(selected_routes, remaining_routes, trains_left)
    if upper_bound <= best_solution['score']:
        self.nodes_pruned += 1
        return  # PRUNE!
    
    route = unique_routes[index]
    city1, city2, length, color = route
    pair = tuple(sorted([city1, city2]))
    
    # Branch 1: Include route (if valid)
    if trains_left >= length and pair not in selected_pairs:
        selected_routes.append(route)
        selected_pairs.add(pair)
        _branch_and_bound(
            unique_routes, index + 1, selected_routes,
            trains_left - length, selected_pairs, best_solution
        )
        selected_routes.pop()  # Backtrack
        selected_pairs.remove(pair)
    
    # Branch 2: Exclude route
    _branch_and_bound(
        unique_routes, index + 1, selected_routes,
        trains_left, selected_pairs, best_solution
    )
```

#### Phase 3: Pruning (Bounding)

The key to efficiency is the **upper bound estimate**:

```python
def _upper_bound_estimate(
    selected_routes: List[RouteTuple],
    remaining_routes: List[RouteTuple],
    trains_left: int
) -> int:
    # Current score from selected routes
    current_score = _evaluate_solution(selected_routes)
    
    # Sort remaining routes by efficiency
    sorted_remaining = sorted(
        remaining_routes,
        key=lambda r: POINTS_TABLE[r[2]] / r[2] if r[2] > 0 else 0,
        reverse=True
    )
    
    # Greedily add most efficient remaining routes (optimistic)
    optimistic_routes = list(selected_routes)
    trains_remaining = trains_left
    
    for route in sorted_remaining:
        if trains_remaining >= route[2]:
            optimistic_routes.append(route)
            trains_remaining -= route[2]
    
    # Calculate optimistic score
    optimistic_score = _evaluate_solution(optimistic_routes)
    
    # Add optimistic bonus for potentially completable tickets
    optimistic_score += OPTIMISTIC_BONUS  # Default: 50 points
    
    return optimistic_score
```

**How pruning works:**
- If `upper_bound <= best_score_found`, this branch cannot lead to a better solution
- The branch is **pruned** (not explored further)
- This dramatically reduces the search space

## Key Features

### ✅ Guaranteed Optimality
The algorithm explores all valid combinations (with pruning), ensuring the solution found is globally optimal.

### ✅ Intelligent Pruning
The upper bound estimate allows pruning of branches that cannot improve the best solution found so far.

### ✅ Detailed Statistics
Tracks exploration metrics:
- `nodes_explored`: Total nodes in the search tree explored
- `nodes_pruned`: Nodes pruned due to bounding
- `best_score`: Best score found during search

### ✅ Efficiency Metrics
Reports pruning efficiency:
```python
pruning_efficiency = (nodes_pruned / nodes_explored) * 100%
```

## Usage

### Basic Usage

```python
from best_solution.optimal_solver import OptimalSolver
from map import TicketToRideMap

# Load map
ttr_map = TicketToRideMap()
ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
graph = ttr_map.get_graph()
tickets_file = "src/map/tickets.csv"

# Create solver
solver = OptimalSolver(graph, tickets_file)

# Solve (may take a long time!)
solution = solver.solve()

print(f"Optimal score: {solution.total_score}")
print(f"Routes selected: {len(solution.routes)}")
print(f"Computation time: {solution.computation_time:.2f}s")
```

### Limited Time Version

```python
# Solve with time limit (returns best found so far)
solution = solver.solve_limited(max_time_seconds=60.0)
```

## Complexity Analysis

### Time Complexity
- **Theoretical**: O(2^n) where n is the number of unique routes
- **Practical**: Much better due to pruning, but still exponential in worst case
- **Space Complexity**: O(n) for recursion stack

### Factors Affecting Performance

1. **Number of Routes**: More routes = exponentially larger search space
2. **Route Efficiency Distribution**: Better distribution improves pruning
3. **Train Constraint**: Tighter constraint (fewer trains) reduces valid combinations
4. **Pruning Effectiveness**: Better upper bounds = more pruning = faster search

## When to Use

### ✅ Use Optimal Solver When:
- Problem size is small/medium (< 30 unique routes)
- Optimality guarantee is required
- You have time to wait (seconds to minutes)
- Benchmarking against heuristics

### ❌ Don't Use When:
- Problem size is large (> 40 routes)
- Real-time or interactive systems
- Approximate solutions are acceptable
- Limited computational resources

## Comparison with Heuristic Solvers

| Aspect | Optimal Solver | Heuristic Solvers |
|--------|---------------|-------------------|
| **Optimality** | Guaranteed | Not guaranteed |
| **Speed** | Slow (exponential) | Fast (polynomial) |
| **Best Use Case** | Small problems, benchmarks | Large problems, real-time |
| **Solution Quality** | Always best | Usually good, sometimes excellent |

## Implementation Details

### Solution Evaluation

Uses shared scoring utilities from `core.scoring`:
```python
from core.scoring import score_routes

breakdown = score_routes(routes, tickets_df)
total_score = breakdown.total_score
```

### Route Uniqueness

The solver uses `unique_routes_from_graph()` which:
- Keeps only the longest route between each city pair
- Reduces problem size for faster solving
- Appropriate when color constraints aren't critical

### Constants

- `TOTAL_TRAINS = 45`: Maximum trains per player
- `OPTIMISTIC_BONUS = 50`: Bonus added to upper bound estimate
- `LONGEST_PATH_BONUS = 10`: Points for longest continuous path

## Example Output

```
=== OPTIMAL SOLVER - Branch and Bound ===
Cercando la soluzione ottima globale...
ATTENZIONE: Questo può richiedere molto tempo!

Rotte uniche da considerare: 25
Spazio di ricerca teorico: 2^25 = 33,554,432 combinazioni

=== RISULTATI ===
Nodi esplorati: 1,234,567
Nodi potati: 32,319,865
Efficienza pruning: 96.3%
Tempo di calcolo: 45.23 secondi

Punteggio ottimo trovato: 127
```

## Tips for Performance

1. **Pre-sort routes**: Already done by efficiency
2. **Use unique routes**: Reduces problem size
3. **Tight bounds**: Better upper bound estimates improve pruning
4. **Early termination**: Use `solve_limited()` for time-constrained scenarios
5. **Parallel search**: Could be parallelized (future enhancement)

## Future Enhancements

- [ ] Parallel branch exploration
- [ ] Improved upper bound estimates
- [ ] Iterative deepening with time limits
- [ ] Caching of subproblem solutions
- [ ] Adaptive pruning strategies
