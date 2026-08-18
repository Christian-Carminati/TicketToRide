# Phase 2 — Baseline Agents & Evaluation System Design Specification

## 1. Context & Mission

Following the completion of Phase 1 (Deterministic Game Core), Phase 2 delivers the **Baseline Agents and Evaluation Suite** for the *TicketToRide RL Lab*. 

The objective of Phase 2 is:
1. Provide a hierarchy of non-RL benchmark opponents:
   - **`RandomAgent`**: Uniform random stochastic baseline for sanity checks.
   - **`GreedyAgent`**: Myopic scoring heuristic maximizing immediate point gain per turn.
   - **`StrategicHeuristicAgent`**: Global graph-aware agent using Dijkstra/BFS shortest paths, ticket synergy optimization, card deficiency tracking, and critical route prioritization.
2. Establish a high-throughput, deterministic evaluation engine (`Evaluator`) and round-robin tournament system (`Tournament`) with dynamic Elo rating computation (`EloSystem`).
3. Provide headless CLI scripts (`scripts/evaluate.py`, `scripts/tournament.py`) and a comprehensive automated test suite capable of running a 1,000-game tournament deterministically in seconds.

---

## 2. Agent Interface Architecture (`src/agents/`)

To support ultra-fast headless simulation on the native Game Core (> 5,000 steps/sec in pure Python) while maintaining full compatibility with future Gymnasium environments (Phase 3), the agent abstraction provides dual interfaces:

```python
class BaseAgent(ABC):
    """Abstract base class for all game agents (heuristic, search, RL)."""

    def __init__(self, name: str = "BaseAgent") -> None:
        self.name = name

    @abstractmethod
    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        """Select a legal domain Action given the game state and valid actions.
        
        Used for direct Game Core execution, benchmarks, and tournaments.
        """
        pass

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Select a discrete integer action index for Gymnasium environments.
        
        Adapter for RL environments (Phase 3+).
        """
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(valid_indices[0])
        return 0

    def reset(self, seed: int | None = None) -> None:
        """Reset agent internal state or random generator if applicable."""
        pass
```

---

## 3. Detailed Baseline Agent Specifications

### 3.1 `RandomAgent`
* **File:** `src/agents/random_agent.py`
* **Purpose:** Pure stochastic baseline and environment stress-tester.
* **Mechanism:**
  - Holds an internal seeded `random.Random(seed)` instance.
  - In `act(state, valid_actions, board)`: returns `self.rng.choice(valid_actions)`.
  - In `select_action(obs, action_mask, info)`: uniformly samples among `np.where(action_mask)[0]`.
  - Fully deterministic given a seed.

### 3.2 `GreedyAgent`
* **File:** `src/agents/greedy_agent.py` (and aliased in `heuristic_agent.py` for backwards compatibility)
* **Purpose:** Deterministic myopic agent optimizing short-term score gains without global map lookahead.
* **Decision Hierarchy:**
  1. **Ticket Keeping (`KEEP_TICKETS`)**:
     - During initial and in-game ticket selection, selects the valid ticket combination with the highest total point value.
  2. **Claim Route (`CLAIM_ROUTE`)**:
     - Evaluates all legal `CLAIM_ROUTE` actions in `valid_actions`.
     - Assigns points based on standard route length rules:
       - Length 6 $\to$ 15 pts, Length 5 $\to$ 10 pts, Length 4 $\to$ 7 pts, Length 3 $\to$ 4 pts, Length 2 $\to$ 2 pts, Length 1 $\to$ 1 pt.
     - Selects the action yielding the maximum immediate points. Ties are broken deterministically by lower route ID or fewer locomotives required.
  3. **Draw Cards (`DRAW_VISIBLE_CARD` / `DRAW_HIDDEN_CARD`)**:
     - If no routes can be claimed, checks player's hand to find their majority non-locomotive card color.
     - Inspects visible face-up cards:
       - If a Locomotive is visible and legal to draw, draws it.
       - Else if a card matching the hand's majority color is visible, draws it.
       - Otherwise, draws from the face-down hidden deck (`DRAW_HIDDEN_CARD`).
  4. **Draw Tickets (`DRAW_TICKETS`)**:
     - If `DRAW_TICKETS` is legal and remaining trains $> 20$ and no high-value routes are affordable, draws tickets. Otherwise defaults to drawing train cards.

### 3.3 `StrategicHeuristicAgent`
* **File:** `src/agents/strategic_agent.py`
* **Purpose:** High-level strategic baseline combining graph algorithms (Dijkstra shortest path) with resource optimization.
* **Algorithmic Components:**
  1. **Graph Representation & Dynamic Routing:**
     - Builds a weighted network of the board where:
       - Routes already claimed by the agent have weight $W = 0$.
       - Unclaimed routes have weight $W = \text{length}$.
       - Routes claimed by opponents have weight $W = \infty$ (blocked).
     - For each incomplete active destination ticket $(C_1, C_2)$, computes the shortest path using Dijkstra's algorithm.
     - If a ticket's path is completely blocked, marks the ticket as unachievable and skips routing for it.
  2. **Ticket Synergy & Selection:**
     - During initial ticket selection (choosing 2 or 3 out of 3):
       - Computes the combined union cost (in train cars) of completing all possible subsets of tickets.
       - Selects the subset with the highest ratio:
         $$\text{Efficiency} = \frac{\sum \text{Points}}{\text{Union Train Cost}}$$
  3. **Card Deficiency Vector:**
     - Identifies all unclaimed route segments on the shortest paths of active tickets.
     - Aggregates the required cards per color: $\text{Needed}(c) = \sum_{r \in \text{Paths}} \text{cost}(r, c)$.
     - Calculates deficiency:
       $$\Delta C(c) = \max(0, \text{Needed}(c) - \text{Hand}(c))$$
  4. **Strategic Action Priority:**
     - **Priority 1 (Critical Claim):** If a route on an active ticket's shortest path is currently affordable and legal to claim, claims it immediately. Prioritizes single-track bottlenecks (where opponents could block) and higher-length segments.
     - **Priority 2 (Targeted Card Draw):** If cards are deficient ($\sum \Delta C(c) > 0$):
       - Checks the 5 visible cards for colors with $\Delta C(c) > 0$ or Locomotives. Draws the needed visible card.
       - If no deficient color is visible, draws from the face-down deck.
     - **Priority 3 (Greedy Opportunism / Endgame):**
       - If all tickets are completed and trains remain $> 10$, draws additional destination tickets.
       - If trains $\le 10$, claims the highest-scoring available route on the board to exhaust trains and maximize route points.

---

## 4. Evaluation & Tournament Engine (`src/evaluation/`)

### 4.1 `Evaluator` (`src/evaluation/evaluator.py`)
Runs head-to-head matches between two agents ($A$ and $B$) across $N$ games.
- **First-Player Balance:** Alternates starting player (Game $i$: $A$ is `player_0` if $i$ is even, $B$ is `player_0` if $i$ is odd).
- **Deterministic Seeding:** Game $i$ uses `seed = base_seed + i`.
- **Metrics Collected:**
  - Total games, wins for $A$, wins for $B$, ties.
  - Average scores and score differential ($\mu, \sigma, \min, \max$).
  - Ticket completion rates (tickets completed / tickets held).
  - Average turns per game and average game duration.

### 4.2 `EloSystem` (`src/evaluation/elo.py`)
Computes Elo ratings for multi-agent rankings:
- Default rating: $1200.0$.
- Default K-factor: $32.0$.
- Expected score: $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$.
- Rating update after match: $R_A' = R_A + K \cdot (S_A - E_A)$, where $S_A \in \{1.0, 0.5, 0.0\}$.

### 4.3 `Tournament` (`src/evaluation/tournament.py`)
Orchestrates round-robin tournaments among a pool of $K$ agents:
- Runs $M$ games for each unique pair $(A_i, A_j)$.
- Dynamically updates Elo ratings match-by-match.
- Produces comprehensive tournament summary tables and exportable structured dict/JSON results.

---

## 5. Command-Line Tools (`scripts/`)

### 5.1 `scripts/evaluate.py`
```bash
python scripts/evaluate.py --agent1 strategic --agent2 greedy --games 100 --seed 42
```
Outputs formatted head-to-head match statistics, score distributions, and ticket completion rates.

### 5.2 `scripts/tournament.py`
```bash
python scripts/tournament.py --agents random,greedy,strategic --games-per-pair 100 --seed 42 --export-json results.json
```
Runs a round-robin tournament across all specified baseline agents, prints an ASCII/rich leaderboard, and saves JSON results.

---

## 6. Acceptance Criteria

1. **Benchmark Hierarchy**: In a 100-game matchup on the standard USA board:
   - `GreedyAgent` achieves $> 80\%$ win rate against `RandomAgent`.
   - `StrategicHeuristicAgent` achieves $> 75\%$ win rate against `GreedyAgent` and $> 95\%$ against `RandomAgent`.
2. **Deterministic Reproducibility**: Running a 1,000-game tournament with `--seed 42` twice produces bit-exact identical game histories, scores, and Elo ratings.
3. **High Performance**: A 1,000-game tournament completes in $< 10$ seconds on modern multi-core CPU.
4. **Test Suite Coverage**: $100\%$ pass rate on all agent unit tests, evaluator tests, tournament tests, and edge cases.
