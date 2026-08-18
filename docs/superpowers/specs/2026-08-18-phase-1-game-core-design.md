# Specification: Phase 1 — Game Core Engine

## 1. Overview & Objective

The **Game Core Engine** is the foundational subsystem of the TicketToRide RL Lab. It implements the pure, deterministic game rules and data models of Ticket to Ride.

### Core Principles
- **100% Pure Python**: Zero dependencies on ML frameworks (`torch`, `gymnasium`, `stable_baselines3`), APIs (`fastapi`), or frontend code.
- **Strict Determinism**: Given a fixed seed, identical initial state, and sequence of actions, the game state progression and termination are 100% identical and reproducible.
- **Explicit Action & State Model**: Every state transition and micro-action is serializable to/from dictionary and JSON for replays, logging, and web streaming.
- **Sub-step State Machine**: Atomic representation of multi-step turns (drawing 2 cards, picking initial/mid-game tickets) enabling precise step-by-step observation and action masking in later RL phases.

---

## 2. Architecture & Components

```text
src/game/
├── __init__.py           # Public exports
├── card.py               # CardColor enum, TrainCard model, standard deck definitions
├── ticket.py             # DestinationTicket model
├── route.py              # Route model (single/double, color requirements, claimed status)
├── board.py              # City and Board models, USA map loader, synthetic mini-map generator
├── player.py             # Player state (trains, cards inventory, tickets, claimed routes, score)
├── state.py              # GameState (decks, visible cards, discard pile, turn state, turn counter)
├── action.py             # Action dataclass, ActionType enum, serializable schema
├── rules.py              # Scoring tables, double-route legality, 3-locomotive discard rule
├── random.py             # SeededRNG isolated deterministic random generator
└── game.py               # Game class: reset, valid_actions, step, check_connectivity, final scoring
```

---

## 3. Data Models & Game Components

### 3.1 Cards & Decks (`card.py`, `ticket.py`)
- **CardColor**: 8 standard colors (`PURPLE`, `WHITE`, `BLUE`, `YELLOW`, `ORANGE`, `BLACK`, `RED`, `GREEN`) + `LOCOMOTIVE` (Wild).
- **TrainCard Deck**: 110 cards total:
  - 12 cards per each of the 8 colors ($12 \times 8 = 96$).
  - 14 Locomotive cards.
- **DestinationTicket**: `id`, `city_a`, `city_b`, `points`.
- **Deck Management**:
  - `deck`: List of remaining cards.
  - `visible_cards`: Exactly 5 face-up cards on the table.
  - `discard_pile`: Cards discarded when routes are claimed or when 3 locomotives appear face-up.
  - When `deck` is empty, `discard_pile` is shuffled using the game's `SeededRNG` to form a new `deck`. If both are empty, drawing from deck is unavailable.
  - **3-Locomotive Rule**: If at any point 3 or more of the 5 visible cards are Locomotives, all 5 visible cards are moved to `discard_pile` and 5 new cards are dealt from the deck (repeated if necessary).

### 3.2 Board & Maps (`board.py`, `route.py`)
- **City**: `id`, `name`, `x`, `y` (coordinates for visualization).
- **Route**: `id`, `city_a`, `city_b`, `length` ($1 \dots 6$), `color` (`CardColor` or `None` for Gray/Any), `double_route_pair_id` (optional ID of the parallel route), `claimed_by` (Player ID or None).
- **USA Map Loader**:
  - 36 North American cities.
  - 100 route segments.
  - 30 destination tickets (points ranging from 4 to 22).
- **Synthetic Mini-Map Preset**:
  - 5 cities (e.g. A, B, C, D, E).
  - 6 routes.
  - 4 destination tickets.
  - Designed for fast property tests and micro-benchmarks.

### 3.3 Player State (`player.py`)
- `id`: str (e.g. `"player_0"`, `"player_1"`).
- `name`: str.
- `trains_remaining`: int (starts at 45).
- `score`: int (starts at 0).
- `cards`: Dict[CardColor, int] (inventory of train cards in hand).
- `tickets`: List[DestinationTicket] (secret destination tickets).
- `claimed_route_ids`: List[str].
- `pending_tickets`: List[DestinationTicket] (drawn tickets awaiting keep/discard decision).

---

## 4. Turn State Machine & Action Model

### 4.1 Turn Sub-States (`TurnState` in `state.py`)
1. `CHOOSING_INITIAL_TICKETS`: Initial setup phase. Each player receives 3 destination tickets and must choose to keep $\ge 2$.
2. `NORMAL`: Start of player's turn. Available action types:
   - `DRAW_HIDDEN_CARD`: Draw 1 card from top of face-down train deck. Transition $\rightarrow$ `DRAWING_SECOND_CARD`.
   - `DRAW_VISIBLE_CARD(card_index)`:
     - If non-locomotive: Draw card into hand, immediately replenish visible slot from deck. Transition $\rightarrow$ `DRAWING_SECOND_CARD`.
     - If locomotive: Draw locomotive into hand, immediately replenish visible slot. Turn ends immediately $\rightarrow$ Next player.
   - `CLAIM_ROUTE(route_id, color_chosen, locomotives_count)`:
     - Verify player has enough trains and matching cards + locomotives.
     - Discard cards, deduct trains, assign immediate route points, update board status. Turn ends $\rightarrow$ Next player.
   - `DRAW_TICKETS`:
     - Draw up to 3 tickets from ticket deck into `pending_tickets`. Transition $\rightarrow$ `CHOOSING_TICKETS`.
3. `DRAWING_SECOND_CARD`: Player must draw their second train card. Available actions:
   - `DRAW_HIDDEN_CARD`: Draw 1 card. Turn ends $\rightarrow$ Next player.
   - `DRAW_VISIBLE_CARD(card_index)`: Only non-locomotive visible cards are legal. Draw card, replenish slot. Turn ends $\rightarrow$ Next player.
4. `CHOOSING_TICKETS`: Player selected `DRAW_TICKETS` mid-game and must keep $\ge 1$ of the drawn tickets.
   - `KEEP_TICKETS(ticket_ids)`: Add chosen tickets to player's tickets, return unchosen tickets to bottom of ticket deck. Turn ends $\rightarrow$ Next player.

### 4.2 Legal Action Computation (`valid_actions()` in `game.py`)
- Returns a strictly validated list of `Action` objects based on the current player, hand, board state, and `turn_state`.
- **Double Route Constraint**:
  - If `num_players <= 3`: Once a route between City A and City B is claimed, any parallel route between City A and City B is permanently blocked/disabled.
  - If `num_players >= 4`: Both parallel routes may be claimed, but never by the same player.
- **Route Payment Validation**:
  - For colored routes: Requires $L$ cards of that color (where $k$ can be locomotives and $L-k$ matching color).
  - For gray routes: Player must explicitly declare which valid single color (plus locomotives) is being spent.

---

## 5. End Game & Final Scoring

### 5.1 End Game Trigger
- When any player has $\le 2$ trains remaining at the end of their turn:
  - `is_last_round` is set to `True`.
  - `final_turn_player_id` is set to the player who triggered it.
  - All other players (and the triggering player) receive exactly one final turn.
  - When the turn returns past the triggering player, `is_game_over` is set to `True`.

### 5.2 Graph Connectivity & Final Scoring
- **Connected Path Algorithm**:
  - Breadth-First Search (BFS) / Connected Component graph on the player's claimed routes.
  - For each `DestinationTicket(city_a, city_b, points)`:
    - If `city_a` and `city_b` are connected: `player.score += points`.
    - If not connected: `player.score -= points`.
- **Longest Continuous Path Bonus (10 points)**:
  - Depth-First Search with backtracking to find the longest simple trail (visiting each route at most once) across the player's claimed subgraph.
  - $+10$ points awarded to the player(s) with the longest path.
- **Winner Determination**:
  - Player with highest final score wins (`winner_id`). Ties broken by most completed tickets.

---

## 6. Testing & Validation Strategy

1. **Unit Tests (`tests/game/`)**:
   - `test_deck_reshuffle`: Card conservation when deck exhausts and discard pile is recycled.
   - `test_3_locomotives_flush`: Automatic flush and redrawing of 5 face-up cards.
   - `test_sub_step_state_machine`: Verifying all state transitions (`NORMAL` $\rightarrow$ `DRAWING_SECOND_CARD` $\rightarrow$ Next player).
   - `test_claim_route_gray_and_colored`: Correct card deduction, train deduction, and immediate points.
   - `test_double_routes_rules`: Blocking parallel routes in 2-player mode.
   - `test_graph_connectivity_and_tickets`: Accurate BFS route completion scoring.
   - `test_longest_path`: Accurate calculation of longest continuous train trail.
2. **Determinism & Property Tests**:
   - `test_deterministic_replay`: 2 full game simulations with identical seed and action sequence produce bit-exact identical game states and scores.
   - `test_card_conservation_invariant`: $\sum \text{cards in hands} + \text{deck} + \text{discard} + \text{visible} == 110$ at every step.
   - `test_train_conservation_invariant`: $\text{trains\_remaining} + \sum \text{claimed route lengths} == 45$ per player.
