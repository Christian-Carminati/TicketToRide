# Phase 1: Game Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete, 100% deterministic Game Core engine for Ticket to Ride, with zero ML/Web dependencies, supporting the official USA map, a fast synthetic mini-map, sub-step turn state machine, official rules (including 3-locomotive flush and 2-3 player double routes), and graph-based ticket/longest path scoring.

**Architecture:** A modular pure-Python engine consisting of isolated models for cards, tickets, routes, boards, players, a deterministic `SeededRNG`, a serializable `GameState` with explicit `TurnState` sub-states, a rules validation engine, and graph algorithms for ticket connectivity and longest continuous path calculation.

**Tech Stack:** Python 3.12, standard library (`dataclasses`, `enum`, `json`, `collections`, `random`), Pytest for unit & property testing. Zero external ML/web dependencies in `src/game/`.

**Spec:** [docs/superpowers/specs/2026-08-18-phase-1-game-core-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-18-phase-1-game-core-design.md)

## Global Constraints
- `src/game/` must never import `torch`, `gymnasium`, `fastapi`, `pydantic`, or any web/ML libraries.
- Execution given `(seed, state, action_sequence)` must be strictly deterministic and reproducible.
- All states and actions must support full `.to_dict()` and `.from_dict()` serialization.
- Follow TDD: write the failing test first, verify failure, implement minimal code, verify pass, commit.

---

### Task 1: Cards, Decks & Deterministic RNG

**Files:**
- Create: `tests/game/test_cards.py`
- Modify: `src/game/card.py`, `src/game/ticket.py`, `src/game/random.py`

**Interfaces:**
- Produces:
  - `CardColor(str, Enum)`: 8 colors + `LOCOMOTIVE`
  - `TrainCard(color: CardColor)` with `.is_locomotive()`
  - `create_standard_train_deck() -> list[TrainCard]` (110 cards: 12 per color + 14 locomotives)
  - `DestinationTicket(id: str, city_a: str, city_b: str, points: int)`
  - `SeededRNG(seed: int)` with `.seed()`, `.shuffle()`, `.choice()`, `.randint()`, `.sample()`

- [ ] **Step 1: Write the failing test for standard deck and RNG**

```python
# tests/game/test_cards.py
from src.game.card import CardColor, TrainCard, create_standard_train_deck
from src.game.random import SeededRNG
from src.game.ticket import DestinationTicket


def test_standard_train_deck_composition():
    deck = create_standard_train_deck()
    assert len(deck) == 110
    locomotives = [c for c in deck if c.is_locomotive()]
    assert len(locomotives) == 14
    for color in CardColor:
        if color != CardColor.LOCOMOTIVE:
            color_cards = [c for c in deck if c.color == color]
            assert len(color_cards) == 12


def test_seeded_rng_reproducibility():
    rng1 = SeededRNG(seed=12345)
    rng2 = SeededRNG(seed=12345)
    deck1 = create_standard_train_deck()
    deck2 = create_standard_train_deck()
    rng1.shuffle(deck1)
    rng2.shuffle(deck2)
    assert [c.color.value for c in deck1] == [c.color.value for c in deck2]


def test_destination_ticket_creation():
    ticket = DestinationTicket(id="t_nyc_mia", city_a="New York", city_b="Miami", points=10)
    assert ticket.points == 10
    assert ticket.city_a == "New York"
    assert ticket.city_b == "Miami"
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv_py312/bin/pytest tests/game/test_cards.py -v`
Expected: FAIL (missing `create_standard_train_deck`)

- [ ] **Step 3: Implement minimal code**

Update `src/game/card.py`, `src/game/random.py`, `src/game/ticket.py`.

```python
# src/game/card.py
from dataclasses import dataclass
from enum import Enum


class CardColor(str, Enum):
    PURPLE = "purple"
    WHITE = "white"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    LOCOMOTIVE = "locomotive"


@dataclass(frozen=True)
class TrainCard:
    color: CardColor

    def is_locomotive(self) -> bool:
        return self.color == CardColor.LOCOMOTIVE


def create_standard_train_deck() -> list[TrainCard]:
    deck: list[TrainCard] = []
    for color in CardColor:
        if color != CardColor.LOCOMOTIVE:
            deck.extend([TrainCard(color=color) for _ in range(12)])
    deck.extend([TrainCard(color=CardColor.LOCOMOTIVE) for _ in range(14)])
    return deck
```

```python
# src/game/random.py
import random
from typing import TypeVar

T = TypeVar("T")


class SeededRNG:
    """Isolated, deterministic RNG to ensure reproducible games."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def shuffle(self, items: list[T]) -> None:
        self.rng.shuffle(items)

    def choice(self, items: list[T]) -> T:
        return self.rng.choice(items)

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def sample(self, items: list[T], k: int) -> list[T]:
        return self.rng.sample(items, k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_cards.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/card.py src/game/ticket.py src/game/random.py tests/game/test_cards.py
git commit -m "feat(game): implement standard train card deck and deterministic RNG"
```

---

### Task 2: Board, Maps & Map Loaders

**Files:**
- Create: `src/game/maps.py`, `tests/game/test_board.py`
- Modify: `src/game/board.py`, `src/game/route.py`

**Interfaces:**
- Produces:
  - `Route(id, city_a, city_b, length, color, double_route_pair_id, claimed_by)` with `.is_claimed`, `.is_double_route`
  - `Board(cities, routes)` with `.add_city()`, `.get_city()`, `.get_routes_between()`, `.get_adjacent_cities()`
  - `load_usa_board() -> tuple[Board, list[DestinationTicket]]`
  - `create_synthetic_mini_board() -> tuple[Board, list[DestinationTicket]]`

- [ ] **Step 1: Write the failing test for maps and board queries**

```python
# tests/game/test_board.py
from src.game.board import Board, City
from src.game.card import CardColor
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.game.route import Route


def test_synthetic_mini_board():
    board, tickets = create_synthetic_mini_board()
    assert len(board.cities) >= 5
    assert len(board.routes) >= 6
    assert len(tickets) >= 3


def test_usa_board_completeness():
    board, tickets = load_usa_board()
    assert len(board.cities) == 36
    assert len(board.routes) == 100
    assert len(tickets) == 30

    # Test double route query
    routes_bos_nyc = board.get_routes_between("Boston", "New York")
    assert len(routes_bos_nyc) == 2
    assert routes_bos_nyc[0].double_route_pair_id == routes_bos_nyc[1].id


def test_board_adjacency():
    board, _ = create_synthetic_mini_board()
    cities = list(board.cities.keys())
    adj = board.get_adjacent_cities(cities[0])
    assert len(adj) > 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv_py312/bin/pytest tests/game/test_board.py -v`
Expected: FAIL (missing `maps.py`, missing methods on `Board`)

- [ ] **Step 3: Implement minimal code**

Update `src/game/route.py`, `src/game/board.py` and create `src/game/maps.py` with complete official USA data and synthetic mini-map.

```python
# src/game/route.py
from dataclasses import dataclass
from src.game.card import CardColor


@dataclass
class Route:
    id: str
    city_a: str
    city_b: str
    length: int
    color: CardColor | None = None  # None indicates Gray
    double_route_pair_id: str | None = None
    claimed_by: str | None = None

    @property
    def is_claimed(self) -> bool:
        return self.claimed_by is not None

    @property
    def is_double_route(self) -> bool:
        return self.double_route_pair_id is not None
```

```python
# src/game/board.py
from dataclasses import dataclass, field
from src.game.route import Route


@dataclass(frozen=True)
class City:
    id: str
    name: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class Board:
    cities: dict[str, City] = field(default_factory=dict)
    routes: list[Route] = field(default_factory=list)

    def add_city(self, city: City) -> None:
        self.cities[city.name] = city
        self.cities[city.id] = city

    def get_city(self, name_or_id: str) -> City | None:
        return self.cities.get(name_or_id)

    def get_route(self, route_id: str) -> Route | None:
        for r in self.routes:
            if r.id == route_id:
                return r
        return None

    def get_routes_between(self, city_a: str, city_b: str) -> list[Route]:
        results = []
        for r in self.routes:
            if (r.city_a == city_a and r.city_b == city_b) or (r.city_a == city_b and r.city_b == city_a):
                results.append(r)
        return results

    def get_adjacent_cities(self, city_name: str) -> list[str]:
        adj = set()
        for r in self.routes:
            if r.city_a == city_name:
                adj.add(r.city_b)
            elif r.city_b == city_name:
                adj.add(r.city_a)
        return sorted(list(adj))
```

```python
# src/game/maps.py
from src.game.board import Board, City
from src.game.card import CardColor
from src.game.route import Route
from src.game.ticket import DestinationTicket

# Define full USA cities, routes, tickets and synthetic mini-map preset
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_board.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/board.py src/game/route.py src/game/maps.py tests/game/test_board.py
git commit -m "feat(game): implement board queries, USA map loader, and mini-map preset"
```

---

### Task 3: Player State & Hand Management

**Files:**
- Create: `tests/game/test_player.py`
- Modify: `src/game/player.py`

**Interfaces:**
- Produces:
  - `Player(id, name, trains_remaining, score, cards, tickets, claimed_route_ids, pending_tickets)`
  - `.add_card()`, `.remove_cards()`, `.can_afford_route()`, `.has_trains_for_route()`, `.total_cards()`
  - `.to_dict()`, `.from_dict()`

- [ ] **Step 1: Write the failing test for Player**

```python
# tests/game/test_player.py
from src.game.card import CardColor, TrainCard
from src.game.player import Player
from src.game.route import Route
from src.game.ticket import DestinationTicket


def test_player_card_management():
    player = Player(id="p0", name="Player 1")
    assert player.total_cards() == 0
    player.add_card(TrainCard(color=CardColor.RED))
    player.add_card(TrainCard(color=CardColor.LOCOMOTIVE))
    assert player.total_cards() == 2
    assert player.cards[CardColor.RED] == 1
    assert player.cards[CardColor.LOCOMOTIVE] == 1

    # Remove cards
    success = player.remove_cards({CardColor.RED: 1})
    assert success is True
    assert player.cards[CardColor.RED] == 0


def test_player_afford_route():
    player = Player(id="p0", name="Player 1", trains_remaining=4)
    route = Route(id="r1", city_a="A", city_b="B", length=3, color=CardColor.BLUE)

    # Cannot afford with 0 cards
    assert player.can_afford_route(route) == []

    # Add 2 Blue and 1 Locomotive
    player.add_card(TrainCard(color=CardColor.BLUE))
    player.add_card(TrainCard(color=CardColor.BLUE))
    player.add_card(TrainCard(color=CardColor.LOCOMOTIVE))

    options = player.can_afford_route(route)
    assert len(options) == 1
    assert options[0] == {CardColor.BLUE: 2, CardColor.LOCOMOTIVE: 1}


def test_player_serialization():
    player = Player(id="p0", name="Player 1", score=15)
    player.tickets.append(DestinationTicket(id="t1", city_a="A", city_b="B", points=5))
    data = player.to_dict()
    restored = Player.from_dict(data)
    assert restored.id == "p0"
    assert restored.score == 15
    assert len(restored.tickets) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv_py312/bin/pytest tests/game/test_player.py -v`
Expected: FAIL (missing `.remove_cards`, `.can_afford_route`, serialization)

- [ ] **Step 3: Implement minimal code in `src/game/player.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_player.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/player.py tests/game/test_player.py
git commit -m "feat(game): implement Player state, route affordability calculation, and serialization"
```

---

### Task 4: State Machine, Action Model & Game State

**Files:**
- Create: `tests/game/test_state.py`
- Modify: `src/game/action.py`, `src/game/state.py`

**Interfaces:**
- Produces:
  - `TurnState(str, Enum)`: `CHOOSING_INITIAL_TICKETS`, `NORMAL`, `DRAWING_SECOND_CARD`, `CHOOSING_TICKETS`
  - `ActionType(str, Enum)`: `DRAW_HIDDEN_CARD`, `DRAW_VISIBLE_CARD`, `CLAIM_ROUTE`, `DRAW_TICKETS`, `KEEP_TICKETS`
  - `Action(action_type, card_index, route_id, color_chosen, locomotives_count, ticket_ids)`
  - `GameState`: Complete state with full roundtrip `.to_dict()`, `.to_json()`, `.from_dict()`

- [ ] **Step 1: Write the failing test for GameState and Actions**

```python
# tests/game/test_state.py
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.player import Player
from src.game.state import GameState, TurnState


def test_action_roundtrip_serialization():
    act = Action(
        action_type=ActionType.CLAIM_ROUTE,
        route_id="r_nyc_bos",
        color_chosen=CardColor.RED,
        locomotives_count=1,
    )
    d = act.to_dict()
    act2 = Action.from_dict(d)
    assert act2 == act


def test_gamestate_roundtrip_serialization():
    state = GameState(
        players=[Player(id="p0", name="Player 1"), Player(id="p1", name="Player 2")],
        current_player_index=0,
        turn_state=TurnState.NORMAL,
        visible_cards=[TrainCard(color=CardColor.RED), TrainCard(color=CardColor.LOCOMOTIVE)],
        train_deck=[TrainCard(color=CardColor.BLUE)],
        discard_pile=[TrainCard(color=CardColor.GREEN)],
        turn_number=3,
    )
    json_str = state.to_json()
    restored = GameState.from_json(json_str)
    assert restored.current_player_index == 0
    assert restored.turn_state == TurnState.NORMAL
    assert len(restored.visible_cards) == 2
    assert restored.visible_cards[1].color == CardColor.LOCOMOTIVE
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv_py312/bin/pytest tests/game/test_state.py -v`
Expected: FAIL (missing `TurnState`, `GameState.from_json`, etc.)

- [ ] **Step 3: Implement minimal code in `src/game/action.py` and `src/game/state.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/action.py src/game/state.py tests/game/test_state.py
git commit -m "feat(game): implement TurnState enum, Action model, and GameState serialization"
```

---

### Task 5: Rules Engine & Legal Action Generation

**Files:**
- Create: `tests/game/test_rules.py`
- Modify: `src/game/rules.py`

**Interfaces:**
- Produces:
  - `GameRules`:
    - `.points_for_route_length(length: int) -> int`
    - `.can_claim_route(route, player, state, board, num_players) -> bool`
    - `.get_valid_route_claim_actions(player, state, board, num_players) -> list[Action]`
    - `.get_valid_actions(player, state, board, num_players) -> list[Action]`
    - `.should_flush_visible_cards(visible_cards: list[TrainCard]) -> bool`

- [ ] **Step 1: Write the failing test for Rules and Legal Actions**

```python
# tests/game/test_rules.py
from src.game.action import ActionType
from src.game.card import CardColor, TrainCard
from src.game.maps import create_synthetic_mini_board
from src.game.player import Player
from src.game.rules import GameRules
from src.game.state import GameState, TurnState


def test_3_locomotives_flush_rule():
    visible = [
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.LOCOMOTIVE),
        TrainCard(color=CardColor.RED),
        TrainCard(color=CardColor.BLUE),
    ]
    assert GameRules.should_flush_visible_cards(visible) is True


def test_double_route_blocking_2_players():
    board, _ = create_synthetic_mini_board()
    # Find a double route
    double_r = [r for r in board.routes if r.is_double_route]
    if double_r:
        r1 = double_r[0]
        r2 = board.get_route(r1.double_route_pair_id)
        r1.claimed_by = "p0"

        player = Player(id="p1", name="P2", trains_remaining=10)
        state = GameState(num_players=2)

        # In 2-player game, r2 should not be claimable
        assert GameRules.can_claim_route(r2, player, state, board, num_players=2) is False


def test_valid_actions_in_drawing_second_card_state():
    state = GameState(
        turn_state=TurnState.DRAWING_SECOND_CARD,
        visible_cards=[
            TrainCard(color=CardColor.RED),
            TrainCard(color=CardColor.LOCOMOTIVE),
        ],
        train_deck=[TrainCard(color=CardColor.BLUE)],
    )
    player = Player(id="p0", name="P1")
    board, _ = create_synthetic_mini_board()
    actions = GameRules.get_valid_actions(player, state, board, num_players=2)

    # In DRAWING_SECOND_CARD, drawing visible locomotive is illegal
    assert any(a.action_type == ActionType.DRAW_HIDDEN_CARD for a in actions)
    assert any(a.action_type == ActionType.DRAW_VISIBLE_CARD and a.card_index == 0 for a in actions)
    assert not any(a.action_type == ActionType.DRAW_VISIBLE_CARD and a.card_index == 1 for a in actions)
    assert not any(a.action_type == ActionType.CLAIM_ROUTE for a in actions)
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv_py312/bin/pytest tests/game/test_rules.py -v`
Expected: FAIL

- [ ] **Step 3: Implement minimal code in `src/game/rules.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/rules.py tests/game/test_rules.py
git commit -m "feat(game): implement GameRules with double-route restriction and legal action generation"
```

---

### Task 6: Graph Algorithms, Final Scoring & Game Orchestrator

**Files:**
- Create: `src/game/graph.py`, `tests/game/test_game_engine.py`
- Modify: `src/game/game.py`

**Interfaces:**
- Produces:
  - `check_ticket_completed(player_routes, ticket) -> bool` (BFS connectivity)
  - `compute_longest_continuous_path(player_routes) -> int` (DFS trail search)
  - `Game(board, tickets_deck, num_players, seed)`:
    - `.reset(seed) -> GameState`
    - `.valid_actions() -> list[Action]`
    - `.step(action: Action) -> GameState`
    - `.compute_final_scores() -> dict[str, int]`

- [ ] **Step 1: Write the failing test for Game execution and scoring**

```python
# tests/game/test_game_engine.py
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.graph import check_ticket_completed, compute_longest_continuous_path
from src.game.maps import create_synthetic_mini_board
from src.game.route import Route
from src.game.ticket import DestinationTicket


def test_graph_ticket_connectivity():
    routes = [
        Route(id="r1", city_a="A", city_b="B", length=2, claimed_by="p0"),
        Route(id="r2", city_a="B", city_b="C", length=3, claimed_by="p0"),
    ]
    t_connected = DestinationTicket(id="t1", city_a="A", city_b="C", points=5)
    t_not_connected = DestinationTicket(id="t2", city_a="A", city_b="D", points=7)

    assert check_ticket_completed(routes, t_connected) is True
    assert check_ticket_completed(routes, t_not_connected) is False


def test_longest_continuous_path():
    routes = [
        Route(id="r1", city_a="A", city_b="B", length=2),
        Route(id="r2", city_a="B", city_b="C", length=4),
        Route(id="r3", city_a="C", city_b="D", length=1),
    ]
    assert compute_longest_continuous_path(routes) == 7


def test_full_game_reset_and_initial_tickets():
    game = Game(num_players=2, seed=42)
    state = game.reset()
    assert len(state.players) == 2
    assert len(state.visible_cards) == 5
    assert state.players[0].total_cards() == 4
    assert state.players[1].total_cards() == 4
    assert len(state.players[0].pending_tickets) == 3


def test_game_step_draw_cards():
    game = Game(num_players=2, seed=42)
    game.reset()
    # Player 0 keeps 2 initial tickets
    act1 = Action(
        action_type=ActionType.KEEP_TICKETS,
        ticket_ids=[game.state.players[0].pending_tickets[0].id, game.state.players[0].pending_tickets[1].id],
    )
    game.step(act1)
    # Player 1 keeps 2 initial tickets
    act2 = Action(
        action_type=ActionType.KEEP_TICKETS,
        ticket_ids=[game.state.players[1].pending_tickets[0].id, game.state.players[1].pending_tickets[1].id],
    )
    game.step(act2)

    # Now turn is NORMAL for Player 0
    p0_cards_before = game.state.players[0].total_cards()
    game.step(Action(action_type=ActionType.DRAW_HIDDEN_CARD))
    game.step(Action(action_type=ActionType.DRAW_HIDDEN_CARD))
    assert game.state.players[0].total_cards() == p0_cards_before + 2
    # Turn advanced to Player 1
    assert game.state.current_player_index == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv_py312/bin/pytest tests/game/test_game_engine.py -v`
Expected: FAIL (missing `graph.py`, full step methods)

- [ ] **Step 3: Implement `src/game/graph.py` and complete `src/game/game.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_game_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/game/graph.py src/game/game.py tests/game/test_game_engine.py
git commit -m "feat(game): implement graph connectivity, longest path, and Game orchestration engine"
```

---

### Task 7: Property Tests & Deterministic Replay Suite

**Files:**
- Create: `tests/game/test_invariants_and_determinism.py`
- Modify: `src/game/__init__.py`

**Interfaces:**
- Verifies:
  - 100% Deterministic Replay (Bit-exact matching across runs with fixed seed)
  - Card Conservation Invariant ($\sum cards == 110$)
  - Train Conservation Invariant ($trains\_remaining + \sum route\_lengths == 45$)
  - Full simulated games terminating cleanly without exception or illegal state

- [ ] **Step 1: Write property and replay tests**

```python
# tests/game/test_invariants_and_determinism.py
import random
from src.game.card import CardColor
from src.game.game import Game


def test_deterministic_full_game_replay():
    def play_game(seed: int):
        game = Game(num_players=2, seed=seed)
        game.reset(seed=seed)
        action_history = []

        # Play up to 200 turns with deterministic random agent
        rng = random.Random(seed)
        for _ in range(200):
            if game.state.is_game_over:
                break
            valid = game.valid_actions()
            if not valid:
                break
            action = rng.choice(valid)
            action_history.append(action.to_dict())
            game.step(action)

        return game.state.to_dict(), action_history

    state1, history1 = play_game(seed=999)
    state2, history2 = play_game(seed=999)

    assert history1 == history2
    assert state1 == state2


def test_card_conservation_invariant():
    game = Game(num_players=2, seed=777)
    game.reset(seed=777)
    rng = random.Random(777)

    for _ in range(150):
        if game.state.is_game_over:
            break
        # Verify card conservation
        total_in_hands = sum(p.total_cards() for p in game.state.players)
        total_visible = len(game.state.visible_cards)
        total_deck = len(game.state.train_deck)
        total_discard = len(game.state.discard_pile)
        assert total_in_hands + total_visible + total_deck + total_discard == 110

        valid = game.valid_actions()
        if not valid:
            break
        game.step(rng.choice(valid))


def test_train_conservation_invariant():
    game = Game(num_players=2, seed=555)
    game.reset(seed=555)
    rng = random.Random(555)

    for _ in range(150):
        if game.state.is_game_over:
            break
        for p in game.state.players:
            claimed_routes = [game.board.get_route(r_id) for r_id in p.claimed_route_ids]
            spent_trains = sum(r.length for r in claimed_routes if r)
            assert p.trains_remaining + spent_trains == 45

        valid = game.valid_actions()
        if not valid:
            break
        game.step(rng.choice(valid))
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./venv_py312/bin/pytest tests/game/test_invariants_and_determinism.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite and linter**

Run: `./venv_py312/bin/pytest && ./venv_py312/bin/ruff check src tests scripts`
Expected: 100% tests PASS, 0 linter errors

- [ ] **Step 4: Commit**

```bash
git add src/game/__init__.py tests/game/test_invariants_and_determinism.py
git commit -m "test(game): add property tests and deterministic replay verification suite"
```
