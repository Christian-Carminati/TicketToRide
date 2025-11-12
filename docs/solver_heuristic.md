# Heuristic Solvers Documentation

The `HeuristicSolver` class implements several strategies to quickly find high-quality solutions for the **Ticket to Ride (TTR)** optimization problem, bypassing the exponential complexity of an optimal search.

## Overview

Heuristic solvers trade optimality guarantees for speed, making them suitable for:
- Large problem instances (> 30 routes)
- Real-time or interactive systems
- Quick exploration of solution space
- Initial solutions for other algorithms

All solvers use shared utilities from `core.scoring` for consistent evaluation:
- Route points calculation
- Longest path detection
- Destination ticket validation
- Total score computation

---

## 1. Greedy Algorithm

The **Greedy Algorithm** is the fastest and simplest approach. At each step, it selects the decision that appears best in the immediate term, ignoring future consequences.

### Mechanism
1.  **Selection Metric:** Routes are sorted by **Points/Train Car** efficiency.
2.  **Process:** It iteratively selects the most efficient route until all 45 train cars are used.
3.  **Result:** **Fast** ($O(N \log N)$), but often converges to a poor local optimum because it **fails to consider connection bonuses** (Ticket Points, Longest Path).

---

## 2. Local Search

**Local Search** improves an initial solution by exploring its "neighborhood" until no better solution can be found, resulting in a **local optimum**.

### Mechanism
1.  **Initialization:** Starts from an initial solution (e.g., the Greedy result).
2.  **Neighborhood:** Explores neighboring solutions using simple operators: **Add**, **Remove**, or **Swap** routes.
3.  **Rule:** Accepts the first neighboring solution that yields a **strictly better score**.
4.  **Limitation:** It is highly susceptible to getting **stuck in the first local optimum** it encounters.

---

## 3. Simulated Annealing (SA)

**Simulated Annealing** is a metaheuristic that extends Local Search by enabling it to escape local optima, inspired by the thermal process of annealing metals.

### Mechanism
1.  **Temperature ($T$):** Starts high and gradually decreases (cooling schedule).
2.  **Acceptance:**
    * **Better Solutions ($\Delta > 0$):** Always accepted.
    * **Worse Solutions ($\Delta \leq 0$):** Accepted with a decreasing probability $P = e^{\frac{\Delta}{T}}$.
3.  **Phases:**
    * **High $T$ (Exploration):** High probability of accepting worse moves, allowing the search to broadly explore the solution space.
    * **Low $T$ (Exploitation):** Probability drops, and the search intensifies near the best solutions found.

---

## 4. Tabu Search (TS)

**Tabu Search** is an adaptive Local Search that uses memory (the Tabu List) to prevent cycling and escape local optima.

### Mechanism
1.  **Tabu List:** Stores recently performed moves (e.g., *adding route A* or *removing route B*).
2.  **Rule:** Moves that reverse a recent action are **forbidden** (Tabu) for a specific number of iterations. This forces the search to explore new areas.
3.  **Aspiration Criterion:** Allows a Tabu move if it results in a score **better than the best solution found so far**, overriding the prohibition.

---

## 5. Genetic Algorithm (GA)

The **Genetic Algorithm** models biological evolution to optimize a solution set. It works on a **population** of solutions (individuals).

### Mechanism
1.  **Initialization:** Create a diverse initial population of valid route sets.
2.  **Fitness:** Evaluate the total score (fitness) of each individual.
3.  **Operators:** New populations are generated using:
    * **Selection:** Choosing high-scoring individuals as "parents."
    * **Crossover:** Combining parents' route sets to create "children."
    * **Mutation:** Randomly altering a child's route set to maintain **diversity**.
4.  **Evolution:** The process repeats for many **generations**, driving the population towards higher scores.

---

## 🎯 Specialized Algorithm: Ticket-Centric Genetic Algorithm (Focus on Bonus Points)

To prioritize the high-value **Ticket Points** and **Longest Path Bonus**, the GA operators are specialized to favor connected route structures:

### A. Weighted Fitness
The selection process uses a weighted score to favor connectivity and bonuses:
$$\text{Fitness}_{\text{weighted}} = (1.0 \times \text{Route Points}) + (1.5 \times \text{Ticket Points}) + (1.5 \times \text{Longest Bonus})$$
This ensures individuals with good connectivity are more likely to become parents.

### B. Targeted Mutation
The mutation step includes an operation that specifically tries to **add a short route** (length 1-2) that connects to existing routes, extending the current **longest continuous path** and directly targeting the high-value +10 bonus.

## Usage

### Basic Usage

```python
from best_solution.heuristic_solver import HeuristicSolver
from map import TicketToRideMap

# Load map
ttr_map = TicketToRideMap()
ttr_map.load_graph("src/map/city_locations.json", "src/map/routes.csv")
graph = ttr_map.get_graph()
tickets_file = "src/map/tickets.csv"

# Create solver
solver = HeuristicSolver(graph, tickets_file)

# Run different algorithms
greedy_solution = solver.greedy_solve()
sa_solution = solver.simulated_annealing_solve(
    initial_temp=100.0,
    cooling_rate=0.95,
    max_iterations=2000
)
ga_solution = solver.genetic_algorithm_solve(
    population_size=50,
    generations=100,
    mutation_rate=0.1
)
```

### Comparing All Algorithms

```python
from best_solution.heuristic_solver import compare_all_heuristics

results = compare_all_heuristics(graph, tickets_file)
# Automatically runs: greedy, local_search, simulated_annealing, 
# genetic, and ticket_centric_genetic algorithms
```

## Algorithm Comparison

| Algorithm | Speed | Solution Quality | Best For |
|-----------|-------|------------------|----------|
| **Greedy** | Very Fast | Low-Medium | Quick baseline, initialization |
| **Local Search** | Fast | Medium | Improving initial solutions |
| **Simulated Annealing** | Medium | Medium-High | Escaping local optima |
| **Tabu Search** | Medium | Medium-High | Avoiding cycling |
| **Genetic Algorithm** | Slow | High | Complex solution spaces |
| **Ticket-Centric GA** | Slow | Very High | Prioritizing connectivity |

## Implementation Notes

### Shared Scoring

All algorithms use the same scoring functions from `core.scoring`:
- `score_routes()`: Complete score breakdown
- `calculate_route_points()`: Points from route lengths
- `calculate_longest_path()`: Longest continuous path
- `calculate_ticket_points()`: Points from completed tickets

### Solution Representation

All solvers return a `Solution` object:
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

### Route Uniqueness

The solver uses `unique_routes_from_graph()` which keeps only the longest route between each city pair, reducing problem size while maintaining solution quality for most cases.

## Performance Tips

1. **Start with Greedy**: Use as initialization for other algorithms
2. **Tune Parameters**: Adjust temperature, population size, etc. for your problem
3. **Combine Algorithms**: Use Greedy → Local Search → Simulated Annealing pipeline
4. **Use Ticket-Centric GA**: When connectivity is more important than raw points