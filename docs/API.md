# API Reference

Complete API documentation for the Ticket to Ride AI Toolkit.

## Core Module

### `core.constants`

Game constants and configuration.

#### `TOTAL_TRAINS: int = 45`
Total number of train pieces each player starts with.

#### `POINTS_TABLE: PointsTable`
Points awarded for routes of different lengths.

```python
POINTS_TABLE[1]  # 1 point
POINTS_TABLE[2]  # 2 points
POINTS_TABLE[3]  # 4 points
POINTS_TABLE[4]  # 7 points
POINTS_TABLE[5]  # 10 points
POINTS_TABLE[6]  # 15 points
```

#### `LONGEST_PATH_BONUS: int = 10`
Bonus points awarded for having the longest continuous path.

---

### `core.graph`

Graph loading utilities.

#### `load_ticket_to_ride_graph(city_locations_file: Path | str, routes_file: Path | str) -> nx.MultiGraph`

Loads a Ticket to Ride map from JSON and CSV files.

**Parameters:**
- `city_locations_file`: Path to JSON file with city coordinates
- `routes_file`: Path to CSV file with route data

**Returns:**
- NetworkX MultiGraph with nodes (cities) and edges (routes)

**Example:**
```python
from core.graph import load_ticket_to_ride_graph

graph = load_ticket_to_ride_graph(
    "src/map/city_locations.json",
    "src/map/routes.csv"
)
```

---

### `core.routes`

Route manipulation utilities.

#### `RouteTuple`
Type alias: `Tuple[str, str, int, str]` representing `(city1, city2, length, color)`

#### `extract_routes(graph: nx.MultiGraph) -> List[RouteTuple]`

Extracts all routes from a NetworkX MultiGraph.

**Parameters:**
- `graph`: NetworkX MultiGraph representing the game map

**Returns:**
- List of all routes as RouteTuple

#### `unique_routes_from_graph(graph: nx.MultiGraph) -> List[RouteTuple]`

Extracts unique routes, keeping only the longest route between each city pair.

**Parameters:**
- `graph`: NetworkX MultiGraph representing the game map

**Returns:**
- List of unique routes (one per city pair, the longest)

#### `trains_used(routes: Iterable[RouteTuple]) -> int`

Calculates total number of trains used by a collection of routes.

**Parameters:**
- `routes`: Iterable of RouteTuple objects

**Returns:**
- Total number of trains (sum of route lengths)

#### `can_claim_route(route: RouteTuple, claimed_pairs: Dict[Tuple[str, str], str], trains_remaining: int) -> bool`

Checks if a route can be claimed given current constraints.

**Parameters:**
- `route`: Route to check
- `claimed_pairs`: Dictionary of already claimed city pairs
- `trains_remaining`: Number of trains the player has left

**Returns:**
- `True` if the route can be claimed, `False` otherwise

---

### `core.scoring`

Scoring calculation utilities.

#### `ScoreBreakdown`

Dataclass containing score components.

```python
@dataclass(frozen=True)
class ScoreBreakdown:
    route_points: int              # Points from claimed routes
    longest_path_length: int       # Length of longest continuous path
    longest_bonus: int             # Bonus points (10 if longest > 0)
    ticket_points: int             # Points from completed tickets
    
    @property
    def total_score(self) -> int   # Sum of all components
```

#### `calculate_route_points(routes: Sequence[RouteTuple]) -> int`

Calculates points from route lengths using the points table.

**Parameters:**
- `routes`: Sequence of routes

**Returns:**
- Total route points

#### `calculate_longest_path(routes: Sequence[RouteTuple]) -> int`

Calculates the length of the longest continuous path in the route network.

**Parameters:**
- `routes`: Sequence of routes

**Returns:**
- Length of longest path (0 if no routes)

#### `calculate_ticket_points(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> int`

Calculates points from completed destination tickets.

**Parameters:**
- `routes`: Sequence of claimed routes
- `tickets_df`: DataFrame with columns 'From', 'To', 'Points'

**Returns:**
- Total ticket points

#### `score_routes(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> ScoreBreakdown`

Calculates complete score breakdown for a set of routes.

**Parameters:**
- `routes`: Sequence of routes claimed by the player
- `tickets_df`: DataFrame containing destination tickets

**Returns:**
- `ScoreBreakdown` with all score components

#### `final_score(routes: Sequence[RouteTuple], tickets_df: pd.DataFrame) -> int`

Convenience function to get total score.

**Parameters:**
- `routes`: Sequence of routes
- `tickets_df`: DataFrame with tickets

**Returns:**
- Total score (route_points + longest_bonus + ticket_points)

---

## Best Solution Module

### `best_solution.heuristic_solver`

#### `HeuristicSolver`

Main class for heuristic solvers.

```python
class HeuristicSolver:
    def __init__(self, graph: nx.MultiGraph, tickets_file: str)
```

**Methods:**

##### `greedy_solve() -> Solution`

Greedy algorithm: selects routes with best points/train ratio.

**Returns:**
- `Solution` object with greedy solution

##### `local_search_solve(initial_solution: List[RouteTuple] = [], max_iterations: int = 1000) -> Solution`

Local search: iteratively improves solution using neighborhood operators.

**Parameters:**
- `initial_solution`: Starting solution (uses greedy if empty)
- `max_iterations`: Maximum iterations to run

**Returns:**
- `Solution` object with improved solution

##### `simulated_annealing_solve(initial_temp: float = 100.0, cooling_rate: float = 0.95, max_iterations: int = 2000) -> Solution`

Simulated annealing: probabilistic search that accepts worse solutions.

**Parameters:**
- `initial_temp`: Starting temperature
- `cooling_rate`: Temperature reduction factor per iteration
- `max_iterations`: Maximum iterations

**Returns:**
- `Solution` object

##### `tabu_search_solve(initial_solution: List[RouteTuple] = [], max_iterations: int = 2000, tabu_list_size: int = 10) -> Solution`

Tabu search: uses memory to avoid local optima.

**Parameters:**
- `initial_solution`: Starting solution
- `max_iterations`: Maximum iterations
- `tabu_list_size`: Size of tabu list

**Returns:**
- `Solution` object

##### `genetic_algorithm_solve(population_size: int = 50, generations: int = 100, mutation_rate: float = 0.1) -> Solution`

Genetic algorithm: evolutionary approach with crossover and mutation.

**Parameters:**
- `population_size`: Number of individuals in population
- `generations`: Number of generations to evolve
- `mutation_rate`: Probability of mutation

**Returns:**
- `Solution` object

##### `ticket_centric_genetic_algorithm_solve(population_size: int = 50, generations: int = 100, mutation_rate: float = 0.1) -> Solution`

Specialized GA that prioritizes connectivity and bonus points.

**Parameters:**
- Same as `genetic_algorithm_solve()`

**Returns:**
- `Solution` object

#### `compare_all_heuristics(graph: nx.MultiGraph, tickets_file: str) -> Dict[str, Solution]`

Runs all heuristic algorithms and compares results.

**Parameters:**
- `graph`: NetworkX MultiGraph
- `tickets_file`: Path to tickets CSV

**Returns:**
- Dictionary mapping algorithm names to `Solution` objects

---

### `best_solution.optimal_solver`

#### `OptimalSolver`

Branch and Bound solver for optimal solutions.

```python
class OptimalSolver:
    def __init__(self, graph: nx.MultiGraph, tickets_file: str)
```

**Attributes:**
- `nodes_explored`: Number of nodes explored
- `nodes_pruned`: Number of nodes pruned
- `best_score`: Best score found so far

**Methods:**

##### `solve() -> Solution`

Finds the globally optimal solution using Branch and Bound.

**Returns:**
- `Solution` object with optimal solution

**Note:** May take exponential time for large problems.

##### `solve_limited(max_time_seconds: float = 60.0) -> Solution`

Limited time version that returns best solution found within time limit.

**Parameters:**
- `max_time_seconds`: Maximum time to search

**Returns:**
- `Solution` object with best solution found

---

### `best_solution.solution`

#### `Solution`

Dataclass representing a complete solution.

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
    algorithm: str = ""
    episodes: int = 0
```

---

## Game Module

### `game`

Full game simulation classes.

#### `Game`

Main game controller.

```python
class Game:
    def __init__(self, number_of_players: int, graph: nx.MultiGraph)
```

**Methods:**

##### `setup() -> None`

Initializes the game: sets up deck, deals initial cards and tickets.

##### `current_player() -> Player`

Returns the player whose turn it is.

##### `next_turn() -> None`

Advances to the next player's turn.

##### `draw_card_action(face_up_card_index: Optional[int]) -> bool`

Player draws cards (from face-up or deck).

##### `claim_route_action(city1: str, city2: str, color: Color, color_to_use: Color) -> bool`

Player claims a route.

##### `draw_destination_ticket_action(keep_tickets: List[int]) -> bool`

Player draws destination tickets.

##### `calculate_winner() -> Tuple[Player, Dict[Color, int]]`

Calculates final scores and determines winner.

---

## Map Module

### `map`

#### `TicketToRideMap`

Map loading and visualization.

```python
class TicketToRideMap:
    def __init__(self)
    def load_graph(city_locations_file: str | Path, routes_file: str | Path) -> None
    def get_graph() -> nx.MultiGraph
    def draw_graph(figsize: Tuple[int, int] = (15, 10)) -> None
```

**Methods:**

##### `load_graph(city_locations_file, routes_file) -> None`

Loads map data from JSON and CSV files.

##### `get_graph() -> nx.MultiGraph`

Returns the loaded graph.

##### `draw_graph(figsize=(15, 10)) -> None`

Visualizes the map using matplotlib.

---

## Type Definitions

### `RouteTuple`
```python
RouteTuple = Tuple[str, str, int, str]  # (city1, city2, length, color)
```

### `Color` (enum)
```python
class Color(enum.Enum):
    RED = 'red'
    BLUE = 'blue'
    GREEN = 'green'
    YELLOW = 'yellow'
    ORANGE = 'orange'
    BLACK = 'black'
    SILVER = 'silver'
    PURPLE = 'purple'
    JOLLY = 'jolly'
    GREY = 'grey'
```

