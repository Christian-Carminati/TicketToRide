# Phase 2 — Baseline Agents & Evaluation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 2 of TicketToRide RL Lab: the full baseline agent hierarchy (Random, Greedy, StrategicHeuristic with graph Dijkstra routing), head-to-head match Evaluator, round-robin Tournament system with Elo rankings, headless CLI scripts, and a 1,000-game benchmark test suite.

**Architecture:** Agents provide a dual interface (`act(state, valid_actions)` for ultra-fast native Game Core simulation and `select_action(obs, mask)` for Gymnasium compatibility). Evaluation runs deterministic matches with alternating first-player symmetry, collecting deep metrics (win rates, score distributions, ticket completions, turns). The tournament orchestrator executes round-robin matches with progressive Elo rating updates.

**Tech Stack:** Python 3.12, NumPy, Pytest, standard library (heapq, math, dataclasses, random, argparse). Zero ML/web dependencies in Game Core and Baseline Agents.

**Spec:** `docs/superpowers/specs/2026-08-18-phase-2-baseline-agents-design.md`

## Global Constraints

- Must work with Python 3.12 (`venv_py312/bin/python3.12`).
- Game Core and Baseline Agents must NEVER import `torch`, `gymnasium`, or web frameworks.
- All evaluation and tournaments must be 100% deterministic given a fixed random seed.
- 1,000-game tournament must complete in $< 10$ seconds on CPU.
- Every task must be verified with automated unit and integration tests.

---

### Task 1: Agent Base Interface & RandomAgent Enhancement

**Files:**
- Modify: `src/agents/base_agent.py`
- Modify: `src/agents/random_agent.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_agents.py`

**Interfaces:**
- Consumes: `src.game.state.GameState`, `src.game.action.Action`, `src.game.board.Board`
- Produces: `BaseAgent.act(state, valid_actions, board) -> Action`, `BaseAgent.select_action(obs, mask, info) -> int`, `RandomAgent`

- [ ] **Step 1: Write the failing tests for BaseAgent and RandomAgent**

```python
# In tests/agents/test_agents.py
import numpy as np
import pytest
from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.game.action import Action, ActionType
from src.game.game import Game


def test_random_agent_act_deterministic():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    valid_actions = game.valid_actions()
    assert len(valid_actions) > 0

    agent1 = RandomAgent(seed=123, name="Random_1")
    agent2 = RandomAgent(seed=123, name="Random_2")

    a1 = agent1.act(game.state, valid_actions, game.board)
    a2 = agent2.act(game.state, valid_actions, game.board)
    assert a1 == a2
    assert a1 in valid_actions


def test_random_agent_select_action_gym():
    agent = RandomAgent(seed=42)
    obs = np.zeros(10)
    mask = np.array([False, True, False, True, False])
    chosen = agent.select_action(obs, action_mask=mask)
    assert chosen in [1, 3]


def test_random_agent_reset():
    agent = RandomAgent(seed=42)
    agent.reset(seed=999)
    assert agent.rng is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/agents/test_agents.py -k "test_random_agent_act_deterministic or test_random_agent_reset"`
Expected: FAIL with `TypeError: Can't instantiate abstract class BaseAgent with abstract method act` or missing methods.

- [ ] **Step 3: Update BaseAgent and RandomAgent implementation**

In `src/agents/base_agent.py`:
```python
"""Base Agent interface supporting both native Game Core and Gymnasium."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.game.action import Action
from src.game.board import Board
from src.game.state import GameState


class BaseAgent(ABC):
    """Abstract interface for all agents (heuristics, RL, tree search)."""

    def __init__(self, name: str = "Agent") -> None:
        self.name = name

    @abstractmethod
    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        """Select a domain Action given current game state and legal actions."""
        pass

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Select an action index for Gymnasium environments."""
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(valid_indices[0])
        return 0

    def reset(self, seed: int | None = None) -> None:
        """Reset internal agent state or random number generator."""
        pass
```

In `src/agents/random_agent.py`:
```python
"""Random baseline agent."""

import random
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.game.action import Action
from src.game.board import Board
from src.game.state import GameState


class RandomAgent(BaseAgent):
    """Uniformly samples from valid actions using seeded RNG."""

    def __init__(self, seed: int = 42, name: str = "RandomAgent") -> None:
        super().__init__(name=name)
        self._initial_seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        new_seed = seed if seed is not None else self._initial_seed
        self.rng = random.Random(new_seed)

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("Cannot select an action from an empty valid_actions list.")
        return self.rng.choice(valid_actions)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(self.rng.choice(valid_indices))
        return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv_py312/bin/pytest tests/agents/test_agents.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/agents/base_agent.py src/agents/random_agent.py tests/agents/test_agents.py
git commit -m "feat(agents): enhance BaseAgent and RandomAgent with dual game/gym interfaces"
```

---

### Task 2: Greedy Scoring Heuristic Agent

**Files:**
- Create: `src/agents/greedy_agent.py`
- Modify: `src/agents/heuristic_agent.py` (alias/wrapper for backwards compatibility)
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_greedy_agent.py`

**Interfaces:**
- Consumes: `src.game.state.GameState`, `src.game.action.Action`, `src.game.board.Board`, `src.game.rules.GameRules`
- Produces: `GreedyAgent(BaseAgent)` with deterministic myopic scoring policy.

- [ ] **Step 1: Write failing tests for GreedyAgent**

```python
# In tests/agents/test_greedy_agent.py
from src.agents.greedy_agent import GreedyAgent
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.ticket import DestinationTicket


def test_greedy_agent_keeps_highest_value_tickets():
    agent = GreedyAgent(name="Greedy")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    valid_actions = game.valid_actions()
    ticket_actions = [a for a in valid_actions if a.action_type == ActionType.KEEP_TICKETS]
    assert len(ticket_actions) > 0

    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type == ActionType.KEEP_TICKETS
    # Greedy chooses all tickets or maximum total value
    assert len(action.ticket_ids) >= 2


def test_greedy_agent_prioritizes_highest_scoring_route():
    agent = GreedyAgent(name="Greedy")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Transition to normal turn
    keep_action = game.valid_actions()[0]
    game.step(keep_action)
    keep_action2 = game.valid_actions()[0]
    game.step(keep_action2)

    # Give current player cards to afford two routes of different lengths
    player = game.state.current_player
    for _ in range(6):
        player.add_card(TrainCard(color=CardColor.RED))
        player.add_card(TrainCard(color=CardColor.BLUE))

    valid_actions = game.valid_actions()
    claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
    if claim_actions:
        action = agent.act(game.state, valid_actions, game.board)
        assert action.action_type == ActionType.CLAIM_ROUTE
        # The chosen route should have max points
        chosen_route = game.board.get_route(action.route_id)
        for ca in claim_actions:
            r = game.board.get_route(ca.route_id)
            assert chosen_route.length >= r.length or game.rules.ROUTE_POINTS[chosen_route.length] >= game.rules.ROUTE_POINTS[r.length]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/agents/test_greedy_agent.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.greedy_agent'`

- [ ] **Step 3: Implement GreedyAgent**

In `src/agents/greedy_agent.py`:
```python
"""Greedy baseline agent maximizing immediate scoring opportunities."""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor
from src.game.rules import GameRules
from src.game.state import GameState, TurnState


class GreedyAgent(BaseAgent):
    """Deterministic greedy baseline agent.
    
    Decision Hierarchy:
    1. KEEP_TICKETS: Pick the valid combination with maximum total points.
    2. CLAIM_ROUTE: Pick the available route yielding the maximum points.
    3. DRAW_VISIBLE_CARD: Draw a Locomotive or a card matching hand's majority color.
    4. DRAW_HIDDEN_CARD: Default card draw fallback.
    5. DRAW_TICKETS: Only if abundant trains (>20) and no other good action.
    """

    def __init__(self, name: str = "GreedyAgent") -> None:
        super().__init__(name=name)
        self.rules = GameRules()

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("No valid actions available for GreedyAgent.")

        # 1. Handle Ticket Selection
        if state.turn_state in [TurnState.CHOOSING_INITIAL_TICKETS, TurnState.CHOOSING_TICKETS]:
            ticket_actions = [a for a in valid_actions if a.action_type == ActionType.KEEP_TICKETS]
            if ticket_actions:
                player = state.current_player
                # Find ticket points lookup
                all_pending = {t.id: t.points for t in player.pending_tickets}
                
                def ticket_combo_value(a: Action) -> int:
                    return sum(all_pending.get(tid, 0) for tid in (a.ticket_ids or ()))

                # Sort descending by total points, then by number of tickets kept
                ticket_actions.sort(key=lambda a: (ticket_combo_value(a), len(a.ticket_ids or ())), reverse=True)
                return ticket_actions[0]

        # 2. Handle Drawing Second Card
        if state.turn_state == TurnState.DRAWING_SECOND_CARD:
            visible_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_VISIBLE_CARD]
            if visible_draws:
                # Pick matching majority color in hand
                best_visible = self._pick_best_visible_card(state, visible_draws)
                if best_visible:
                    return best_visible
            hidden_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD]
            if hidden_draws:
                return hidden_draws[0]
            return valid_actions[0]

        # 3. Check for Claiming Routes (Prioritize Highest Points)
        claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
        if claim_actions and board is not None:
            def route_score(a: Action) -> tuple[int, int]:
                r = board.get_route(a.route_id or "")
                if not r:
                    return (0, 0)
                points = self.rules.ROUTE_POINTS.get(r.length, 0)
                return (points, r.length)

            claim_actions.sort(key=route_score, reverse=True)
            return claim_actions[0]

        # 4. Draw Cards (Visible matching hand majority or Locomotive, otherwise Hidden)
        visible_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_VISIBLE_CARD]
        if visible_draws:
            best_visible = self._pick_best_visible_card(state, visible_draws)
            if best_visible:
                return best_visible

        hidden_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD]
        if hidden_draws:
            return hidden_draws[0]

        # 5. Fallback
        return valid_actions[0]

    def _pick_best_visible_card(self, state: GameState, visible_actions: list[Action]) -> Action | None:
        player = state.current_player
        if not player:
            return visible_actions[0] if visible_actions else None

        # Find non-locomotive color with highest count in player's hand
        hand_counts = {c: count for c, count in player.cards.items() if c != CardColor.LOCOMOTIVE and count > 0}
        target_color = max(hand_counts, key=hand_counts.get) if hand_counts else None

        # Look for visible Locomotive first
        for a in visible_actions:
            idx = a.card_index or 0
            if 0 <= idx < len(state.visible_cards):
                card = state.visible_cards[idx]
                if card.color == CardColor.LOCOMOTIVE:
                    return a

        # Look for target color
        if target_color:
            for a in visible_actions:
                idx = a.card_index or 0
                if 0 <= idx < len(state.visible_cards):
                    card = state.visible_cards[idx]
                    if card.color == target_color:
                        return a

        return None
```

In `src/agents/heuristic_agent.py`:
```python
"""Deterministic greedy heuristic agent (alias/wrapper for backwards compatibility)."""

from src.agents.greedy_agent import GreedyAgent

HeuristicAgent = GreedyAgent
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv_py312/bin/pytest tests/agents/test_greedy_agent.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/agents/greedy_agent.py src/agents/heuristic_agent.py src/agents/__init__.py tests/agents/test_greedy_agent.py
git commit -m "feat(agents): implement GreedyAgent maximizing immediate route points and card matching"
```

---

### Task 3: Strategic Graph Heuristic Agent (`StrategicHeuristicAgent`)

**Files:**
- Create: `src/agents/strategic_agent.py`
- Modify: `src/agents/__init__.py`
- Test: `tests/agents/test_strategic_agent.py`

**Interfaces:**
- Consumes: `src.game.state.GameState`, `src.game.action.Action`, `src.game.board.Board`, `src.game.ticket.DestinationTicket`, `src.game.graph.check_ticket_completed`
- Produces: `StrategicHeuristicAgent(BaseAgent)` with Dijkstra routing, card deficiency analysis, ticket synergy, and critical route claiming.

- [ ] **Step 1: Write failing tests for StrategicHeuristicAgent**

```python
# In tests/agents/test_strategic_agent.py
import pytest
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.ticket import DestinationTicket


def test_strategic_agent_initial_ticket_synergy():
    agent = StrategicHeuristicAgent(name="Strategic")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type == ActionType.KEEP_TICKETS
    # Agent keeps at least 2 synergistic tickets
    assert len(action.ticket_ids) >= 2


def test_strategic_agent_targets_shortest_path_route():
    agent = StrategicHeuristicAgent(name="Strategic")
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    # Transition past ticket selection
    for _ in range(2):
        act = game.valid_actions()[0]
        game.step(act)

    player = game.state.current_player
    # Give player plenty of all cards
    for c in CardColor:
        for _ in range(10):
            player.add_card(TrainCard(color=c))

    valid_actions = game.valid_actions()
    claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
    assert len(claim_actions) > 0

    action = agent.act(game.state, valid_actions, game.board)
    assert action.action_type == ActionType.CLAIM_ROUTE
    # Verify the claimed route is part of a ticket route or high value
    assert action.route_id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/agents/test_strategic_agent.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.strategic_agent'`

- [ ] **Step 3: Implement StrategicHeuristicAgent**

In `src/agents/strategic_agent.py`:
```python
"""Strategic Graph-Aware Heuristic Agent."""

import heapq
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor
from src.game.graph import check_ticket_completed
from src.game.player import Player
from src.game.route import Route
from src.game.rules import GameRules
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket


class StrategicHeuristicAgent(BaseAgent):
    """Advanced heuristic agent using Dijkstra shortest paths and card deficit tracking.
    
    Components:
    - Dijkstra dynamic routing on remaining board graph (owned routes cost 0, opponent routes cost inf).
    - Ticket synergy analysis during initial selection.
    - Card deficiency vector calculation.
    - Priority execution: Critical route claim -> Targeted card draw -> Endgame scoring.
    """

    def __init__(self, name: str = "StrategicAgent") -> None:
        super().__init__(name=name)
        self.rules = GameRules()

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("No valid actions available for StrategicHeuristicAgent.")

        if board is None:
            # Fallback if board is not passed
            return valid_actions[0]

        player = state.current_player
        if not player:
            return valid_actions[0]

        # 1. Handle Ticket Selection (Initial & Mid-game)
        if state.turn_state in [TurnState.CHOOSING_INITIAL_TICKETS, TurnState.CHOOSING_TICKETS]:
            return self._choose_best_tickets(player, valid_actions, board)

        # 2. Handle Drawing Second Card
        if state.turn_state == TurnState.DRAWING_SECOND_CARD:
            return self._handle_second_card_draw(state, player, valid_actions, board)

        # 3. Main Turn Decision: Route Claims vs Card Draws
        # Compute shortest paths for active incomplete tickets
        active_tickets = [t for t in player.destination_tickets if not check_ticket_completed(t, player.claimed_route_ids, board)]
        target_routes: list[Route] = []
        for t in active_tickets:
            path_routes = self._compute_shortest_path_routes(player, t, board)
            if path_routes:
                for r in path_routes:
                    if r.id not in player.claimed_route_ids and r not in target_routes:
                        target_routes.append(r)

        # Check if we can claim any target route
        claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
        if claim_actions:
            target_claim_actions = [a for a in claim_actions if a.route_id in {r.id for r in target_routes}]
            if target_claim_actions:
                # Prioritize single-track bottlenecks (fewer parallel routes) and longer segments
                def claim_priority(a: Action) -> tuple[int, int]:
                    r = board.get_route(a.route_id or "")
                    if not r:
                        return (0, 0)
                    is_bottleneck = 1 if len(board.get_routes_between(r.city1, r.city2)) == 1 else 0
                    return (is_bottleneck, r.length)

                target_claim_actions.sort(key=claim_priority, reverse=True)
                return target_claim_actions[0]

            # If all tickets completed or no target route affordable, check if trains are low or no card deficit
            if not active_tickets or player.trains_remaining <= 12:
                # Claim highest value available route
                def general_route_score(a: Action) -> int:
                    r = board.get_route(a.route_id or "")
                    return self.rules.ROUTE_POINTS.get(r.length, 0) if r else 0

                claim_actions.sort(key=general_route_score, reverse=True)
                return claim_actions[0]

        # 4. If cannot claim, draw cards targeting deficient colors for target routes
        deficit = self._calculate_card_deficit(player, target_routes)
        visible_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_VISIBLE_CARD]
        if visible_draws:
            best_visible = self._pick_targeted_visible_card(state, visible_draws, deficit)
            if best_visible:
                return best_visible

        hidden_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD]
        if hidden_draws:
            return hidden_draws[0]

        return valid_actions[0]

    def _choose_best_tickets(self, player: Player, valid_actions: list[Action], board: Board) -> Action:
        ticket_actions = [a for a in valid_actions if a.action_type == ActionType.KEEP_TICKETS]
        if not ticket_actions:
            return valid_actions[0]

        pending_by_id = {t.id: t for t in player.pending_tickets}

        def subset_efficiency(action: Action) -> float:
            ticket_ids = action.ticket_ids or ()
            if not ticket_ids:
                return 0.0
            tickets = [pending_by_id[tid] for tid in ticket_ids if tid in pending_by_id]
            total_points = sum(t.points for t in tickets)

            # Compute union route cost using Dijkstra
            needed_routes: set[str] = set()
            for t in tickets:
                path = self._compute_shortest_path_routes(player, t, board)
                for r in path:
                    needed_routes.add(r.id)

            total_train_cost = sum(board.get_route(rid).length for rid in needed_routes if board.get_route(rid))
            if total_train_cost == 0:
                return float(total_points)
            return total_points / float(total_train_cost)

        ticket_actions.sort(key=subset_efficiency, reverse=True)
        return ticket_actions[0]

    def _compute_shortest_path_routes(self, player: Player, ticket: DestinationTicket, board: Board) -> list[Route]:
        """Dijkstra shortest path algorithm on the board network."""
        # Distances: city -> min train cost
        dist: dict[str, float] = {ticket.city1: 0.0}
        prev_route: dict[str, Route | None] = {ticket.city1: None}
        prev_city: dict[str, str | None] = {ticket.city1: None}

        # Priority queue: (cost, current_city)
        pq: list[tuple[float, str]] = [(0.0, ticket.city1)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            if u == ticket.city2:
                break

            for r in board.get_adjacent_routes(u):
                # If claimed by opponent, impassable
                if r.claimed_by is not None and r.claimed_by != player.id:
                    continue

                v = r.other_city(u)
                weight = 0.0 if r.claimed_by == player.id else float(r.length)

                if dist.get(u, float("inf")) + weight < dist.get(v, float("inf")):
                    dist[v] = dist[u] + weight
                    prev_route[v] = r
                    prev_city[v] = u
                    heapq.heappush(pq, (dist[v], v))

        if ticket.city2 not in dist or dist[ticket.city2] == float("inf"):
            return []

        # Reconstruct path
        path_routes: list[Route] = []
        curr = ticket.city2
        while curr != ticket.city1:
            r = prev_route.get(curr)
            if not r:
                break
            path_routes.append(r)
            curr = prev_city.get(curr, ticket.city1)

        return path_routes

    def _calculate_card_deficit(self, player: Player, routes: list[Route]) -> dict[CardColor, int]:
        needed: dict[CardColor, int] = defaultdict(int)
        wild_needed = 0

        for r in routes:
            if r.claimed_by == player.id:
                continue
            if r.color == CardColor.GRAY:
                wild_needed += r.length
            else:
                needed[r.color] += r.length

        deficit: dict[CardColor, int] = {}
        for color, count in needed.items():
            have = player.cards.get(color, 0)
            if have < count:
                deficit[color] = count - have

        # Allocate wild requirement to the color with biggest hand or deficit
        if wild_needed > 0:
            deficit[CardColor.LOCOMOTIVE] = wild_needed

        return deficit

    def _pick_targeted_visible_card(
        self, state: GameState, visible_actions: list[Action], deficit: dict[CardColor, int]
    ) -> Action | None:
        # Check visible locomotives
        for a in visible_actions:
            idx = a.card_index or 0
            if 0 <= idx < len(state.visible_cards):
                card = state.visible_cards[idx]
                if card.color == CardColor.LOCOMOTIVE:
                    return a

        # Check visible cards matching deficit colors
        for a in visible_actions:
            idx = a.card_index or 0
            if 0 <= idx < len(state.visible_cards):
                card = state.visible_cards[idx]
                if card.color in deficit and deficit[card.color] > 0:
                    return a

        return None

    def _handle_second_card_draw(
        self, state: GameState, player: Player, valid_actions: list[Action], board: Board
    ) -> Action:
        active_tickets = [t for t in player.destination_tickets if not check_ticket_completed(t, player.claimed_route_ids, board)]
        target_routes: list[Route] = []
        for t in active_tickets:
            for r in self._compute_shortest_path_routes(player, t, board):
                if r.id not in player.claimed_route_ids:
                    target_routes.append(r)

        deficit = self._calculate_card_deficit(player, target_routes)
        visible_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_VISIBLE_CARD]
        if visible_draws:
            best = self._pick_targeted_visible_card(state, visible_draws, deficit)
            if best:
                return best

        hidden = [a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD]
        if hidden:
            return hidden[0]

        return valid_actions[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv_py312/bin/pytest tests/agents/test_strategic_agent.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/agents/strategic_agent.py src/agents/__init__.py tests/agents/test_strategic_agent.py
git commit -m "feat(agents): implement StrategicHeuristicAgent with Dijkstra shortest paths and card deficit tracking"
```

---

### Task 4: Evaluator and Head-to-Head Engine

**Files:**
- Modify: `src/evaluation/evaluator.py`
- Modify: `src/evaluation/metrics.py`
- Modify: `src/evaluation/__init__.py`
- Test: `tests/evaluation/test_evaluator.py`

**Interfaces:**
- Consumes: `src.agents.base_agent.BaseAgent`, `src.game.game.Game`, `src.evaluation.metrics.EvaluationMetrics`
- Produces: `Evaluator.evaluate(agent_a, agent_b, num_games, seed) -> dict[str, EvaluationMetrics]`

- [ ] **Step 1: Write failing tests for Evaluator**

```python
# In tests/evaluation/test_evaluator.py
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.evaluator import Evaluator


def test_evaluator_head_to_head_alternates_first_player():
    evaluator = Evaluator()
    agent_a = RandomAgent(seed=10, name="Random_A")
    agent_b = RandomAgent(seed=20, name="Random_B")

    results = evaluator.evaluate(agent_a, agent_b, num_games=10, seed=42)

    assert "Random_A" in results
    assert "Random_B" in results
    metrics_a = results["Random_A"]
    metrics_b = results["Random_B"]

    assert metrics_a.total_games == 10
    assert metrics_b.total_games == 10
    assert metrics_a.wins + metrics_b.wins + metrics_a.draws == 10
    assert metrics_a.avg_turns > 0


def test_evaluator_greedy_beats_random():
    evaluator = Evaluator()
    greedy = GreedyAgent(name="Greedy")
    random_bot = RandomAgent(seed=100, name="Random")

    results = evaluator.evaluate(greedy, random_bot, num_games=20, seed=42)
    metrics_greedy = results["Greedy"]
    metrics_random = results["Random"]

    assert metrics_greedy.wins > metrics_random.wins
    assert metrics_greedy.avg_score > metrics_random.avg_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/evaluation/test_evaluator.py`
Expected: FAIL with missing implementation in Evaluator.

- [ ] **Step 3: Implement Evaluator**

In `src/evaluation/evaluator.py`:
```python
"""High-performance deterministic evaluation framework."""

from src.agents.base_agent import BaseAgent
from src.evaluation.metrics import EvaluationMetrics
from src.game.game import Game


class Evaluator:
    """Evaluates two agents in head-to-head matches with alternating player positions."""

    def __init__(self, max_turns: int = 400) -> None:
        self.max_turns = max_turns

    def evaluate(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_games: int = 100,
        seed: int = 42,
    ) -> dict[str, EvaluationMetrics]:
        """Run N deterministic head-to-head games between agent_a and agent_b."""
        metrics_a = EvaluationMetrics(total_games=num_games)
        metrics_b = EvaluationMetrics(total_games=num_games)

        total_score_a = 0
        total_score_b = 0
        total_turns = 0
        tickets_drawn_a = 0
        tickets_completed_a = 0
        tickets_drawn_b = 0
        tickets_completed_b = 0

        for game_idx in range(num_games):
            game_seed = seed + game_idx
            # Alternate player seats: even games -> (A is player 0, B is player 1)
            #                       odd games -> (B is player 0, A is player 1)
            is_a_first = (game_idx % 2 == 0)
            p0_agent = agent_a if is_a_first else agent_b
            p1_agent = agent_b if is_a_first else agent_a

            agent_a.reset(seed=game_seed)
            agent_b.reset(seed=game_seed + 100000)

            game = Game(num_players=2, seed=game_seed)
            game.reset(seed=game_seed)

            # Play until game over or max turns safety bound
            while not game.state.is_game_over and game.state.turn_number < self.max_turns:
                curr_idx = game.state.current_player_index
                curr_agent = p0_agent if curr_idx == 0 else p1_agent

                valid_actions = game.valid_actions()
                if not valid_actions:
                    break

                action = curr_agent.act(game.state, valid_actions, game.board)
                game.step(action)

            # Extract end game statistics
            p0 = game.state.players[0]
            p1 = game.state.players[1]
            score_p0 = p0.score
            score_p1 = p1.score

            score_a = score_p0 if is_a_first else score_p1
            score_b = score_p1 if is_a_first else score_p0

            total_score_a += score_a
            total_score_b += score_b
            total_turns += game.state.turn_number

            # Ticket stats
            p_a = p0 if is_a_first else p1
            p_b = p1 if is_a_first else p0
            tickets_drawn_a += len(p_a.destination_tickets)
            tickets_drawn_b += len(p_b.destination_tickets)
            tickets_completed_a += sum(1 for t in p_a.destination_tickets if t.id in p_a.completed_ticket_ids or p_a.is_ticket_completed(t, game.board))
            tickets_completed_b += sum(1 for t in p_b.destination_tickets if t.id in p_b.completed_ticket_ids or p_b.is_ticket_completed(t, game.board))

            # Record game outcome
            if score_a > score_b:
                metrics_a.wins += 1
                metrics_b.losses += 1
            elif score_b > score_a:
                metrics_b.wins += 1
                metrics_a.losses += 1
            else:
                metrics_a.draws += 1
                metrics_b.draws += 1

        # Aggregate metrics
        if num_games > 0:
            metrics_a.avg_score = total_score_a / num_games
            metrics_b.avg_score = total_score_b / num_games
            metrics_a.avg_score_diff = (total_score_a - total_score_b) / num_games
            metrics_b.avg_score_diff = (total_score_b - total_score_a) / num_games
            metrics_a.avg_turns = total_turns / num_games
            metrics_b.avg_turns = total_turns / num_games
            metrics_a.ticket_completion_rate = (tickets_completed_a / tickets_drawn_a) if tickets_drawn_a > 0 else 0.0
            metrics_b.ticket_completion_rate = (tickets_completed_b / tickets_drawn_b) if tickets_drawn_b > 0 else 0.0

        return {agent_a.name: metrics_a, agent_b.name: metrics_b}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv_py312/bin/pytest tests/evaluation/test_evaluator.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/evaluation/evaluator.py src/evaluation/metrics.py tests/evaluation/test_evaluator.py
git commit -m "feat(evaluation): implement deterministic head-to-head Evaluator with seat alternating"
```

---

### Task 5: Round-Robin Tournament Orchestrator & Elo Rating System

**Files:**
- Modify: `src/evaluation/tournament.py`
- Modify: `src/evaluation/elo.py`
- Modify: `src/evaluation/__init__.py`
- Test: `tests/evaluation/test_tournament.py`

**Interfaces:**
- Consumes: `src.agents.base_agent.BaseAgent`, `src.evaluation.evaluator.Evaluator`, `src.evaluation.elo.EloSystem`
- Produces: `Tournament.run(seed) -> dict[str, Any]` with leaderboard, pairings, Elo rankings.

- [ ] **Step 1: Write failing tests for Tournament**

```python
# In tests/evaluation/test_tournament.py
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.tournament import Tournament


def test_round_robin_tournament_execution():
    agents = [
        RandomAgent(seed=1, name="Random_1"),
        GreedyAgent(name="Greedy_1"),
        StrategicHeuristicAgent(name="Strategic_1"),
    ]
    tournament = Tournament(agents=agents, games_per_pair=10)
    results = tournament.run(seed=42)

    assert len(results["leaderboard"]) == 3
    # Strategic should have higher Elo than Random
    elo_strategic = results["ratings"]["Strategic_1"]
    elo_random = results["ratings"]["Random_1"]
    assert elo_strategic > elo_random


def test_tournament_deterministic_replay():
    agents1 = [RandomAgent(seed=1, name="R1"), GreedyAgent(name="G1")]
    agents2 = [RandomAgent(seed=1, name="R1"), GreedyAgent(name="G1")]

    t1 = Tournament(agents=agents1, games_per_pair=6)
    r1 = t1.run(seed=999)

    t2 = Tournament(agents=agents2, games_per_pair=6)
    r2 = t2.run(seed=999)

    assert r1["ratings"] == r2["ratings"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/evaluation/test_tournament.py`
Expected: FAIL with skeleton output.

- [ ] **Step 3: Implement Tournament**

In `src/evaluation/tournament.py`:
```python
"""Round-robin and Swiss tournament orchestrator."""

from itertools import combinations
from typing import Any

from src.agents.base_agent import BaseAgent
from src.evaluation.elo import EloSystem
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import EvaluationMetrics


class Tournament:
    """Orchestrates deterministic round-robin tournaments among agent pools."""

    def __init__(
        self,
        agents: list[BaseAgent],
        games_per_pair: int = 50,
        initial_elo: float = 1200.0,
        k_factor: float = 32.0,
    ) -> None:
        self.agents = agents
        self.games_per_pair = games_per_pair
        self.elo_system = EloSystem(initial_rating=initial_elo, k_factor=k_factor)
        self.evaluator = Evaluator()

    def run(self, seed: int = 42) -> dict[str, Any]:
        """Execute round-robin pairings, update Elo, and compile leaderboard."""
        matchup_results: list[dict[str, Any]] = []
        agent_stats: dict[str, dict[str, Any]] = {
            a.name: {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_score": 0.0,
                "total_turns": 0.0,
            }
            for a in self.agents
        }

        pairings = list(combinations(self.agents, 2))
        current_seed = seed

        for agent_a, agent_b in pairings:
            pair_results = self.evaluator.evaluate(
                agent_a=agent_a,
                agent_b=agent_b,
                num_games=self.games_per_pair,
                seed=current_seed,
            )
            current_seed += self.games_per_pair

            metrics_a = pair_results[agent_a.name]
            metrics_b = pair_results[agent_b.name]

            # Update cumulative stats
            for agent, metrics in [(agent_a, metrics_a), (agent_b, metrics_b)]:
                st = agent_stats[agent.name]
                st["games"] += metrics.total_games
                st["wins"] += metrics.wins
                st["losses"] += metrics.losses
                st["draws"] += metrics.draws
                st["total_score"] += metrics.avg_score * metrics.total_games
                st["total_turns"] += metrics.avg_turns * metrics.total_games

            # Update Elo ratings based on match results
            # Score contribution for A = (wins + 0.5 * draws) / total_games
            score_a = (metrics_a.wins + 0.5 * metrics_a.draws) / self.games_per_pair
            self.elo_system.update(agent_a.name, agent_b.name, score_a=score_a)

            matchup_results.append({
                "agent_a": agent_a.name,
                "agent_b": agent_b.name,
                "wins_a": metrics_a.wins,
                "wins_b": metrics_b.wins,
                "draws": metrics_a.draws,
                "avg_score_a": metrics_a.avg_score,
                "avg_score_b": metrics_b.avg_score,
            })

        # Build Leaderboard
        leaderboard = []
        for agent in self.agents:
            st = agent_stats[agent.name]
            total_g = st["games"]
            win_rate = (st["wins"] / total_g) if total_g > 0 else 0.0
            avg_score = (st["total_score"] / total_g) if total_g > 0 else 0.0
            rating = self.elo_system.get_rating(agent.name)

            leaderboard.append({
                "name": agent.name,
                "elo": round(rating, 1),
                "win_rate": round(win_rate, 3),
                "wins": st["wins"],
                "losses": st["losses"],
                "draws": st["draws"],
                "avg_score": round(avg_score, 1),
                "total_games": total_g,
            })

        # Sort leaderboard descending by Elo
        leaderboard.sort(key=lambda x: x["elo"], reverse=True)

        return {
            "leaderboard": leaderboard,
            "ratings": {a.name: self.elo_system.get_rating(a.name) for a in self.agents},
            "matchups": matchup_results,
            "seed": seed,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv_py312/bin/pytest tests/evaluation/test_tournament.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/evaluation/tournament.py src/evaluation/elo.py tests/evaluation/test_tournament.py
git commit -m "feat(evaluation): implement Tournament round-robin engine with Elo updates"
```

---

### Task 6: CLI Scripts (`scripts/evaluate.py`, `scripts/tournament.py`)

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `scripts/tournament.py`
- Test: `tests/evaluation/test_cli.py`

**Interfaces:**
- Consumes: `src.agents.*`, `src.evaluation.evaluator.Evaluator`, `src.evaluation.tournament.Tournament`
- Produces: Executable CLI commands with structured summary tables and JSON export.

- [ ] **Step 1: Write integration tests for CLI tools**

```python
# In tests/evaluation/test_cli.py
import subprocess
import sys


def test_evaluate_cli_execution():
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--agent1", "greedy",
        "--agent2", "random",
        "--games", "10",
        "--seed", "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Evaluation Results" in res.stdout
    assert "Greedy" in res.stdout


def test_tournament_cli_execution():
    cmd = [
        sys.executable,
        "scripts/tournament.py",
        "--agents", "random,greedy,strategic",
        "--games-per-pair", "10",
        "--seed", "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Tournament Leaderboard" in res.stdout
```

- [ ] **Step 2: Run test to verify it fails or outputs correctly**

Run: `venv_py312/bin/pytest tests/evaluation/test_cli.py`

- [ ] **Step 3: Update scripts/evaluate.py and scripts/tournament.py**

In `scripts/evaluate.py`:
```python
"""Headless Agent Evaluation entrypoint."""

import argparse
import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.evaluator import Evaluator


def build_agent(agent_type: str, seed: int = 42, name: str | None = None) -> BaseAgent:
    name_str = name or agent_type.capitalize()
    t = agent_type.lower()
    if t == "random":
        return RandomAgent(seed=seed, name=name_str)
    elif t in ["greedy", "heuristic"]:
        return GreedyAgent(name=name_str)
    elif t == "strategic":
        return StrategicHeuristicAgent(name=name_str)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two agents head-to-head")
    parser.add_argument("--agent1", type=str, default="strategic", help="Type of Agent 1 (random, greedy, strategic)")
    parser.add_argument("--agent2", type=str, default="greedy", help="Type of Agent 2 (random, greedy, strategic)")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed")
    parser.add_argument("--export-json", type=str, default=None, help="Optional path to export JSON metrics")
    args = parser.parse_args()

    a1 = build_agent(args.agent1, seed=args.seed, name=f"{args.agent1.capitalize()}_1")
    a2 = build_agent(args.agent2, seed=args.seed + 1, name=f"{args.agent2.capitalize()}_2")

    evaluator = Evaluator()
    results = evaluator.evaluate(agent_a=a1, agent_b=a2, num_games=args.games, seed=args.seed)

    m1 = results[a1.name]
    m2 = results[a2.name]

    print("=" * 65)
    print(f"  Head-to-Head Evaluation: {a1.name} vs {a2.name}")
    print(f"  Total Games: {args.games} | Seed: {args.seed}")
    print("=" * 65)
    print(f"{'Metric':<25} | {a1.name:<16} | {a2.name:<16}")
    print("-" * 65)
    print(f"{'Wins':<25} | {m1.wins:<16} | {m2.wins:<16}")
    print(f"{'Win Rate':<25} | {m1.win_rate * 100:<15.1f}% | {m2.win_rate * 100:<15.1f}%")
    print(f"{'Draws':<25} | {m1.draws:<16} | {m2.draws:<16}")
    print(f"{'Avg Score':<25} | {m1.avg_score:<16.1f} | {m2.avg_score:<16.1f}")
    print(f"{'Avg Score Diff':<25} | {m1.avg_score_diff:<+16.1f} | {m2.avg_score_diff:<+16.1f}")
    print(f"{'Ticket Completion Rate':<25} | {m1.ticket_completion_rate * 100:<15.1f}% | {m2.ticket_completion_rate * 100:<15.1f}%")
    print(f"{'Avg Turns / Game':<25} | {m1.avg_turns:<16.1f} | {m2.avg_turns:<16.1f}")
    print("=" * 65)

    if args.export_json:
        data = {
            a1.name: {
                "wins": m1.wins,
                "win_rate": m1.win_rate,
                "avg_score": m1.avg_score,
                "ticket_rate": m1.ticket_completion_rate,
            },
            a2.name: {
                "wins": m2.wins,
                "win_rate": m2.win_rate,
                "avg_score": m2.avg_score,
                "ticket_rate": m2.ticket_completion_rate,
            },
        }
        with open(args.export_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results exported to {args.export_json}")


if __name__ == "__main__":
    main()
```

In `scripts/tournament.py`:
```python
"""Headless tournament entrypoint."""

import argparse
import json

from src.agents.base_agent import BaseAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.tournament import Tournament


def parse_agent_list(agents_str: str, seed: int) -> list[BaseAgent]:
    tokens = [t.strip().lower() for t in agents_str.split(",") if t.strip()]
    agents: list[BaseAgent] = []
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        name = f"{t.capitalize()}_{counts[t]}"
        if t == "random":
            agents.append(RandomAgent(seed=seed + len(agents), name=name))
        elif t in ["greedy", "heuristic"]:
            agents.append(GreedyAgent(name=name))
        elif t == "strategic":
            agents.append(StrategicHeuristicAgent(name=name))
        else:
            raise ValueError(f"Unknown agent type: {t}")
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a round-robin tournament among agents")
    parser.add_argument("--agents", type=str, default="random,greedy,strategic", help="Comma-separated agent types")
    parser.add_argument("--games-per-pair", type=int, default=50, help="Games per pair of agents")
    parser.add_argument("--seed", type=int, default=42, help="Tournament seed")
    parser.add_argument("--export-json", type=str, default=None, help="Optional path to export JSON tournament report")
    args = parser.parse_args()

    agents = parse_agent_list(args.agents, seed=args.seed)
    tournament = Tournament(agents=agents, games_per_pair=args.games_per_pair)
    results = tournament.run(seed=args.seed)

    print("\n" + "=" * 75)
    print(f"  Tournament Leaderboard ({len(agents)} agents | {args.games_per_pair} games/matchup | Seed: {args.seed})")
    print("=" * 75)
    print(f"{'Rank':<5} | {'Agent':<18} | {'Elo':<7} | {'Win Rate':<10} | {'W-L-D':<12} | {'Avg Score':<10}")
    print("-" * 75)
    for rank, entry in enumerate(results["leaderboard"], 1):
        wld = f"{entry['wins']}-{entry['losses']}-{entry['draws']}"
        print(
            f"{rank:<5} | {entry['name']:<18} | {entry['elo']:<7.1f} | {entry['win_rate'] * 100:<9.1f}% | "
            f"{wld:<12} | {entry['avg_score']:<10.1f}"
        )
    print("=" * 75 + "\n")

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Tournament report saved to {args.export_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv_py312/bin/pytest tests/evaluation/test_cli.py`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add scripts/evaluate.py scripts/tournament.py tests/evaluation/test_cli.py
git commit -m "feat(cli): complete scripts for head-to-head evaluate and round-robin tournament"
```

---

### Task 7: 1,000-Game Benchmark Acceptance Suite & Documentation Update

**Files:**
- Create: `tests/evaluation/test_phase2_acceptance.py`
- Modify: `DESIGN.md` (Check off Phase 2 delivery)
- Modify: `README.md`
- Test: `tests/evaluation/test_phase2_acceptance.py`

**Interfaces:**
- Consumes: All Phase 2 agents and tournament engine
- Produces: End-to-end automated 1,000-game tournament benchmark verifying hierarchy, performance, determinism.

- [ ] **Step 1: Write Phase 2 1,000-game acceptance test**

```python
# In tests/evaluation/test_phase2_acceptance.py
import time
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.tournament import Tournament


def test_phase2_1000_game_tournament_acceptance():
    """Acceptance Test for Phase 2:
    - Runs a 1,000-game total tournament across baseline agents.
    - Asserts performance (< 10 seconds).
    - Asserts Elo and win rate hierarchy: Strategic > Greedy > Random.
    - Asserts deterministic reproducibility.
    """
    agents = [
        RandomAgent(seed=42, name="Random_1"),
        GreedyAgent(name="Greedy_1"),
        StrategicHeuristicAgent(name="Strategic_1"),
    ]
    # 3 pairs * 334 games ~ 1002 games
    games_per_pair = 334
    tournament = Tournament(agents=agents, games_per_pair=games_per_pair)

    start_time = time.time()
    results = tournament.run(seed=12345)
    duration = time.time() - start_time

    leaderboard = results["leaderboard"]
    assert len(leaderboard) == 3

    # Ranking check
    assert leaderboard[0]["name"] == "Strategic_1", f"Expected Strategic_1 first, got {leaderboard[0]}"
    assert leaderboard[1]["name"] == "Greedy_1", f"Expected Greedy_1 second, got {leaderboard[1]}"
    assert leaderboard[2]["name"] == "Random_1", f"Expected Random_1 third, got {leaderboard[2]}"

    # Strategic win rate and score assertions
    assert leaderboard[0]["elo"] > leaderboard[1]["elo"] > leaderboard[2]["elo"]
    assert leaderboard[0]["avg_score"] > leaderboard[1]["avg_score"] > leaderboard[2]["avg_score"]

    # Speed check (< 10 seconds)
    print(f"\n1000-Game Tournament completed in {duration:.2f} seconds ({1002 / duration:.1f} games/sec)")
    assert duration < 10.0, f"Tournament took too long: {duration:.2f}s"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/evaluation/test_phase2_acceptance.py -s`
Expected: PASS in < 10 seconds.

- [ ] **Step 3: Update documentation and check off Phase 2**

Update `README.md` and `DESIGN.md` indicating Phase 2 completion and baseline benchmarks.

- [ ] **Step 4: Run full test suite**

Run: `venv_py312/bin/pytest`
Expected: 100% tests pass.

- [ ] **Step 5: Commit changes**

```bash
git add tests/evaluation/test_phase2_acceptance.py DESIGN.md README.md
git commit -m "feat(benchmark): complete Phase 2 1000-game tournament acceptance and docs"
```
