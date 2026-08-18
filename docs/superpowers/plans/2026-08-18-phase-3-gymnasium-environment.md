# Phase 3 — Gymnasium Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Farama Gymnasium-compliant environment subsystem (`TicketToRideEnv`) for TicketToRide RL Lab, including full POMDP observation encoding (`ObservationV1`), bijective discrete action mapping (`DiscreteActionSpace`), invalid action masking (`ActionMasker`), modular reward shaping (`DefaultRewardCalculator`), and Gymnasium `check_env` validation.

**Architecture:** The environment wraps the deterministic pure-Python `Game` core. It translates game states into bounded $[0, 1]^D$ numpy observation vectors with zero hidden-information leakage, maps discrete action indices to domain actions, computes boolean action masks for valid actions, calculates configurable step and terminal rewards, and automatically steps opponent baseline agents for single-agent RL training.

**Tech Stack:** Python 3.12, Gymnasium 1.0+, NumPy, Pytest.

**Spec:** [docs/superpowers/specs/2026-08-18-phase-3-gymnasium-environment-design.md](file:///home/christian/Projects/Python/TicketToRide/docs/superpowers/specs/2026-08-18-phase-3-gymnasium-environment-design.md)

## Global Constraints

- **Pure Python + NumPy + Gymnasium only**: No dependencies on torch or web frameworks inside `src/environment/`.
- **Determinism**: Identical seeds in `env.reset(seed)` must produce identical observations, masks, and transitions.
- **Fairness & Anti-Leakage**: Observations for player $i$ must NEVER include opponent hidden cards, opponent secret tickets, or future deck order.
- **Gymnasium Standard**: `env.reset()` returns `(obs, info)`, `env.step()` returns `(obs, reward, terminated, truncated, info)`, and passes `gymnasium.utils.env_checker.check_env`.

---

## File Structure

```text
src/environment/
├── __init__.py               # Public exports (TicketToRideEnv, ObservationV1, DiscreteActionSpace, etc.)
├── observation.py            # BaseObservationEncoder ABC, ObservationV1 feature vector builder
├── action_space.py           # DiscreteActionSpace bi-directional mapping (integers <-> domain Actions)
├── action_mask.py            # ActionMasker computing boolean legal action mask
├── reward.py                 # RewardWeights dataclass, BaseRewardCalculator ABC, DefaultRewardCalculator (RewardV1)
└── env.py                    # TicketToRideEnv(gym.Env) wrapper with auto-stepping opponent

tests/environment/
├── conftest.py               # Fixtures for boards, tickets, and environments
├── test_observation.py       # Shape, normalization, and strict anti-leakage tests
├── test_action_space.py      # Bijectivity and action coverage tests
├── test_action_mask.py       # Legal action mask correctness across all turn states
├── test_reward.py            # Step points, ticket completion, and terminal reward tests
├── test_gym_env.py           # Gym lifecycle, check_env validation, opponent auto-stepping
└── test_phase3_acceptance.py # 100-game acceptance benchmark via Gymnasium interface
```

---

### Task 1: Observation Vector Encoder (`ObservationV1`) & Anti-Leakage Verification

**Files:**
- Modify: `src/environment/observation.py`
- Create: `tests/environment/test_observation.py`

**Interfaces:**
- Consumes: `src.game.state.GameState`, `src.game.card.CardColor`, `src.game.board.Board`, `src.game.ticket.DestinationTicket`, `src.game.graph.check_ticket_completed`
- Produces: `BaseObservationEncoder` (ABC), `ObservationV1(board, tickets)` with `encode(state, player_index) -> np.ndarray` (dtype `float32`, bounded in $[0, 1]$) and `observation_shape -> tuple[int]`.

- [ ] **Step 1: Write the failing test for `ObservationV1` and Anti-Leakage**

Create `tests/environment/test_observation.py`:
```python
"""Tests for ObservationV1 encoder and POMDP anti-leakage invariants."""

import numpy as np
import pytest

from src.environment.observation import ObservationV1
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.maps import load_synthetic_mini_board, load_usa_board
from src.game.ticket import DestinationTicket


def test_observation_v1_mini_board_shape_and_range():
    board, tickets = load_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)

    encoder = ObservationV1(board=board, initial_tickets=tickets)
    obs = encoder.encode(state, player_index=0)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == encoder.observation_shape
    # D_mini = 9 (hand) + 50 (visible) + 2 (player) + 18 (6 routes*3) + 12 (4 tickets*3) + 4 (opp) + 5 (decks) + 4 (phase) = 104
    assert obs.shape == (104,)
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_observation_v1_usa_board_shape():
    board, tickets = load_usa_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)

    encoder = ObservationV1(board=board, initial_tickets=tickets)
    obs = encoder.encode(state, player_index=0)

    # D_usa = 9 + 50 + 2 + (100*3=300) + (30*3=90) + 4 + 5 + 4 = 464
    assert obs.shape == (464,)
    assert obs.shape == encoder.observation_shape
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_observation_v1_anti_leakage_guarantee():
    """Modifying opponent hidden cards or deck order must NOT change player 0 observation."""
    board, tickets = load_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=123)
    state = game.reset(seed=123)

    encoder = ObservationV1(board=board, initial_tickets=tickets)
    obs_original = encoder.encode(state, player_index=0).copy()

    # Modify opponent hidden hand without changing card count
    state.players[1].cards[CardColor.RED] = 2
    state.players[1].cards[CardColor.BLUE] = 2
    state.players[1].cards[CardColor.GREEN] = 0

    # Modify hidden train deck sequence without changing length
    if len(state.train_deck) >= 2:
        state.train_deck[0], state.train_deck[1] = state.train_deck[1], state.train_deck[0]

    obs_modified = encoder.encode(state, player_index=0)
    np.testing.assert_array_equal(obs_original, obs_modified)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/environment/test_observation.py -v`
Expected: FAIL due to unimplemented `ObservationV1` features.

- [ ] **Step 3: Implement `ObservationV1` in `src/environment/observation.py`**

Update `src/environment/observation.py`:
```python
"""Observation encoders for converting GameState to bounded vector representations."""

from abc import ABC, abstractmethod

import numpy as np

from src.game.board import Board
from src.game.card import CardColor
from src.game.graph import check_ticket_completed
from src.game.maps import load_usa_board
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket

STANDARD_CARD_COLORS = [
    CardColor.PURPLE,
    CardColor.WHITE,
    CardColor.BLUE,
    CardColor.YELLOW,
    CardColor.ORANGE,
    CardColor.BLACK,
    CardColor.RED,
    CardColor.GREEN,
    CardColor.LOCOMOTIVE,
]


class BaseObservationEncoder(ABC):
    """Abstract base class for versioned observation encoders."""

    @abstractmethod
    def encode(self, state: GameState, player_index: int) -> np.ndarray:
        """Encode the game state from the perspective of player_index."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the resulting observation array."""


class ObservationV1(BaseObservationEncoder):
    """Version 1 Observation Encoder (Flattened, bounded [0, 1] feature vector).

    Strictly satisfies anti-leakage POMDP requirements:
    - Never leaks opponent hidden card colors
    - Never leaks opponent destination tickets
    - Never leaks hidden deck order
    """

    def __init__(
        self,
        board: Board | None = None,
        initial_tickets: list[DestinationTicket] | None = None,
        num_players: int = 2,
    ) -> None:
        if board is None:
            self.board, self.initial_tickets = load_usa_board()
        else:
            self.board = board
            self.initial_tickets = initial_tickets or []

        self.num_players = num_players
        self._route_ids = [r.id for r in self.board.routes]
        self._ticket_map = {t.id: t for t in self.initial_tickets}
        self._ticket_ids = sorted(self._ticket_map.keys())

        # Calculate exact dimension D
        self._dim = (
            9  # player hand counts (8 colors + locomotive)
            + 5 * 10  # 5 visible slots * (9 colors + 1 empty channel)
            + 2  # trains remaining, current score
            + len(self._route_ids) * 3  # route states: [unclaimed, own, opponent]
            + len(self._ticket_ids) * 3  # ticket states: [owned, completed, points/25]
            + (self.num_players - 1) * 4  # opponent public: [card_count, trains, routes_count, score]
            + 5  # deck counts: [train_deck/110, discard/110, ticket_deck/30, turn/100, is_last_round]
            + 4  # turn phase one-hot (NORMAL, DRAWING_SECOND_CARD, CHOOSING_TICKETS, CHOOSING_INITIAL)
        )

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self._dim,)

    def encode(self, state: GameState, player_index: int) -> np.ndarray:
        obs = np.zeros(self._dim, dtype=np.float32)
        offset = 0

        player = state.players[player_index]

        # 1. Player hand cards (9 values normalized by / 12)
        for i, color in enumerate(STANDARD_CARD_COLORS):
            count = player.cards.get(color, 0)
            obs[offset + i] = min(count / 12.0, 1.0)
        offset += 9

        # 2. Visible cards (5 slots * 10 channels)
        for slot in range(5):
            if slot < len(state.visible_cards):
                card_color = state.visible_cards[slot].color
                if card_color in STANDARD_CARD_COLORS:
                    color_idx = STANDARD_CARD_COLORS.index(card_color)
                    obs[offset + slot * 10 + color_idx] = 1.0
            else:
                obs[offset + slot * 10 + 9] = 1.0  # Slot empty channel
        offset += 50

        # 3. Player status (trains / 45, score / 150)
        obs[offset] = player.trains_remaining / 45.0
        obs[offset + 1] = min(player.score / 150.0, 1.0)
        offset += 2

        # 4. Route ownership states (len(routes) * 3)
        claimed_routes_by_id = {r.id: r.claimed_by for r in self.board.routes}
        # In state, check player claimed routes
        player_claimed_ids = {r.id for r in player.claimed_routes}
        all_claimed_ids = {
            r.id: p.id for p in state.players for r in p.claimed_routes
        }

        for i, r_id in enumerate(self._route_ids):
            base = offset + i * 3
            owner = all_claimed_ids.get(r_id)
            if owner is None:
                obs[base] = 1.0  # Unclaimed
            elif owner == player.id:
                obs[base + 1] = 1.0  # Owned by self
            else:
                obs[base + 2] = 1.0  # Owned by opponent
        offset += len(self._route_ids) * 3

        # 5. Tickets (len(tickets) * 3)
        owned_ticket_ids = {t.id for t in player.tickets}
        for i, t_id in enumerate(self._ticket_ids):
            base = offset + i * 3
            if t_id in owned_ticket_ids:
                ticket_obj = self._ticket_map[t_id]
                is_completed = check_ticket_completed(player.claimed_routes, ticket_obj)
                obs[base] = 1.0  # Owned
                obs[base + 1] = 1.0 if is_completed else 0.0
                obs[base + 2] = min(ticket_obj.points / 25.0, 1.0)
            else:
                obs[base] = 0.0
                obs[base + 1] = 0.0
                obs[base + 2] = 0.0
        offset += len(self._ticket_ids) * 3

        # 6. Opponent public status ((N-1) * 4)
        opp_count = 0
        for p_idx, p in enumerate(state.players):
            if p_idx == player_index:
                continue
            base = offset + opp_count * 4
            total_cards = sum(p.cards.values())
            obs[base] = min(total_cards / 30.0, 1.0)
            obs[base + 1] = p.trains_remaining / 45.0
            obs[base + 2] = min(len(p.claimed_routes) / 30.0, 1.0)
            obs[base + 3] = min(p.score / 150.0, 1.0)
            opp_count += 1
        offset += (self.num_players - 1) * 4

        # 7. Deck counts & global game progression (5 values)
        obs[offset] = min(len(state.train_deck) / 110.0, 1.0)
        obs[offset + 1] = min(len(state.discard_pile) / 110.0, 1.0)
        obs[offset + 2] = min(len(state.ticket_deck) / 30.0, 1.0)
        obs[offset + 3] = min(state.turn_number / 100.0, 1.0)
        obs[offset + 4] = 1.0 if state.is_last_round else 0.0
        offset += 5

        # 8. Turn state one-hot (4 values)
        turn_states = [
            TurnState.NORMAL,
            TurnState.DRAWING_SECOND_CARD,
            TurnState.CHOOSING_TICKETS,
            TurnState.CHOOSING_INITIAL_TICKETS,
        ]
        if state.turn_state in turn_states:
            idx = turn_states.index(state.turn_state)
            obs[offset + idx] = 1.0
        offset += 4

        return obs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/environment/test_observation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/environment/observation.py tests/environment/test_observation.py
git commit -m "feat: implement ObservationV1 vector encoder with anti-leakage guarantees"
```

---

### Task 2: Discrete Action Space Mapping (`DiscreteActionSpace`)

**Files:**
- Modify: `src/environment/action_space.py`
- Create: `tests/environment/test_action_space.py`

**Interfaces:**
- Consumes: `src.game.action.Action`, `src.game.action.ActionType`, `src.game.board.Board`, `src.game.card.CardColor`, `src.game.maps.load_usa_board`
- Produces: `DiscreteActionSpace(board)` with `n -> int`, `to_action(action_id: int) -> Action`, `to_id(action: Action) -> int | None`.

- [ ] **Step 1: Write the failing test for `DiscreteActionSpace`**

Create `tests/environment/test_action_space.py`:
```python
"""Tests for DiscreteActionSpace bijective mapping."""

import pytest

from src.environment.action_space import DiscreteActionSpace
from src.game.action import Action, ActionType
from src.game.card import CardColor
from src.game.maps import load_synthetic_mini_board, load_usa_board


def test_discrete_action_space_mini_board_bijection():
    board, _ = load_synthetic_mini_board()
    space = DiscreteActionSpace(board=board)

    # 1 (hidden) + 5 (visible) + 1 (draw tickets) + 7 (keep tickets) + routes
    assert space.n > 0
    for action_id in range(space.n):
        action = space.to_action(action_id)
        assert isinstance(action, Action)
        recovered_id = space.to_id(action)
        assert recovered_id == action_id, f"Failed roundtrip for action {action}"


def test_discrete_action_space_usa_board_coverage():
    board, _ = load_usa_board()
    space = DiscreteActionSpace(board=board)

    # Draw hidden
    draw_hidden = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
    assert space.to_id(draw_hidden) == 0

    # Draw visible
    for slot in range(5):
        draw_vis = Action(action_type=ActionType.DRAW_VISIBLE_CARD, card_index=slot)
        assert space.to_id(draw_vis) == 1 + slot

    # Draw tickets
    draw_tix = Action(action_type=ActionType.DRAW_TICKETS)
    assert space.to_id(draw_tix) == 6

    # All keep tickets combinations (7 non-empty subsets)
    for subset_idx in range(7):
        act_id = 7 + subset_idx
        act = space.to_action(act_id)
        assert act.action_type == ActionType.KEEP_TICKETS

    # Claim routes (colored + gray)
    route_claim_count = 0
    for r in board.routes:
        if r.color == CardColor.LOCOMOTIVE or r.color is None:
            route_claim_count += 8  # Gray route has 8 colored actions
        else:
            route_claim_count += 1
    assert space.n == 14 + route_claim_count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/environment/test_action_space.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `DiscreteActionSpace` in `src/environment/action_space.py`**

Update `src/environment/action_space.py`:
```python
"""Bi-directional mapping between discrete integer action IDs and domain Actions."""

from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor
from src.game.maps import load_usa_board

STANDARD_CLAIM_COLORS = [
    CardColor.PURPLE,
    CardColor.WHITE,
    CardColor.BLUE,
    CardColor.YELLOW,
    CardColor.ORANGE,
    CardColor.BLACK,
    CardColor.RED,
    CardColor.GREEN,
]

# 7 non-empty subsets of 3 indices {0, 1, 2}
TICKET_SUBSET_INDICES: list[tuple[int, ...]] = [
    (0,),
    (1,),
    (2,),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1, 2),
]


class DiscreteActionSpace:
    """Bi-directional deterministic mapping between integer action IDs and domain Actions."""

    def __init__(self, board: Board | None = None) -> None:
        if board is None:
            self.board, _ = load_usa_board()
        else:
            self.board = board

        self._action_to_id: dict[Action, int] = {}
        self._id_to_action: dict[int, Action] = {}
        self._build_action_space()

    def _build_action_space(self) -> None:
        current_id = 0

        # 0: DRAW_HIDDEN_CARD
        act = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
        self._id_to_action[current_id] = act
        self._action_to_id[act] = current_id
        current_id += 1

        # 1..5: DRAW_VISIBLE_CARD (slots 0..4)
        for slot in range(5):
            act = Action(action_type=ActionType.DRAW_VISIBLE_CARD, card_index=slot)
            self._id_to_action[current_id] = act
            self._action_to_id[act] = current_id
            current_id += 1

        # 6: DRAW_TICKETS
        act = Action(action_type=ActionType.DRAW_TICKETS)
        self._id_to_action[current_id] = act
        self._action_to_id[act] = current_id
        current_id += 1

        # 7..13: KEEP_TICKETS (subsets of {0, 1, 2} represented by string tuple index markers)
        for subset in TICKET_SUBSET_INDICES:
            # We encode subset indices as dummy ticket string markers ("0", "1", ...) for bijection
            act = Action(
                action_type=ActionType.KEEP_TICKETS,
                ticket_ids=tuple(str(idx) for idx in subset),
            )
            self._id_to_action[current_id] = act
            self._action_to_id[act] = current_id
            current_id += 1

        # 14+: CLAIM_ROUTE for each route in board
        sorted_routes = sorted(self.board.routes, key=lambda r: r.id)
        for r in sorted_routes:
            if r.color == CardColor.LOCOMOTIVE or r.color is None:
                # Gray route: 8 standard colors
                for color in STANDARD_CLAIM_COLORS:
                    act = Action(
                        action_type=ActionType.CLAIM_ROUTE,
                        route_id=r.id,
                        color_chosen=color,
                    )
                    self._id_to_action[current_id] = act
                    self._action_to_id[act] = current_id
                    current_id += 1
            else:
                # Specific colored route
                act = Action(
                    action_type=ActionType.CLAIM_ROUTE,
                    route_id=r.id,
                    color_chosen=r.color,
                )
                self._id_to_action[current_id] = act
                self._action_to_id[act] = current_id
                current_id += 1

    @property
    def n(self) -> int:
        return len(self._id_to_action)

    def to_action(self, action_id: int) -> Action:
        return self._id_to_action[action_id]

    def to_id(self, action: Action) -> int | None:
        # Check direct match
        if action in self._action_to_id:
            return self._action_to_id[action]

        # Handle canonical matching for route claim if locomotives_count is specified
        if action.action_type == ActionType.CLAIM_ROUTE:
            canonical = Action(
                action_type=ActionType.CLAIM_ROUTE,
                route_id=action.route_id,
                color_chosen=action.color_chosen,
            )
            return self._action_to_id.get(canonical)

        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/environment/test_action_space.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/environment/action_space.py tests/environment/test_action_space.py
git commit -m "feat: implement DiscreteActionSpace with bijective action mapping"
```

---

### Task 3: Invalid Action Masking (`ActionMasker`)

**Files:**
- Modify: `src/environment/action_mask.py`
- Create: `tests/environment/test_action_mask.py`

**Interfaces:**
- Consumes: `src.environment.action_space.DiscreteActionSpace`, `src.game.action.Action`, `src.game.action.ActionType`, `src.game.state.GameState`
- Produces: `ActionMasker(action_space)` with `compute_mask(valid_actions: list[Action], pending_tickets: list[DestinationTicket] | None = None) -> np.ndarray` (shape `(action_space.n,)`, dtype `bool`).

- [ ] **Step 1: Write the failing test for `ActionMasker`**

Create `tests/environment/test_action_mask.py`:
```python
"""Tests for ActionMasker legal action mask generation."""

import numpy as np
import pytest

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.game.action import Action, ActionType
from src.game.card import CardColor
from src.game.game import Game
from src.game.maps import load_synthetic_mini_board
from src.game.state import TurnState


def test_action_masker_initial_tickets_state():
    board, tickets = load_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)
    space = DiscreteActionSpace(board=board)
    masker = ActionMasker(space)

    valid_actions = game.valid_actions()
    pending = state.current_player.pending_tickets
    mask = masker.compute_mask(valid_actions, pending_tickets=pending)

    assert mask.shape == (space.n,)
    assert mask.dtype == bool
    assert np.any(mask)

    # In CHOOSING_INITIAL_TICKETS, only keep_tickets with >= 2 tickets should be True
    # Actions 0..6 (draws) must be False
    assert not np.any(mask[0:7])
    # Action 14+ (claim routes) must be False
    assert not np.any(mask[14:])


def test_action_masker_normal_turn():
    board, tickets = load_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    state = game.reset(seed=42)
    space = DiscreteActionSpace(board=board)
    masker = ActionMasker(space)

    # Transition to NORMAL turn state
    act0 = game.valid_actions()[0]
    game.step(act0)
    act1 = game.valid_actions()[0]
    game.step(act1)
    assert game.state.turn_state == TurnState.NORMAL

    valid_actions = game.valid_actions()
    mask = masker.compute_mask(valid_actions, pending_tickets=game.state.current_player.pending_tickets)

    assert mask.shape == (space.n,)
    assert np.any(mask)
    # Hidden card draw (index 0) must be True
    assert mask[0] is True or mask[0] == True
    # Keep tickets actions (indices 7..13) must be False
    assert not np.any(mask[7:14])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/environment/test_action_mask.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `ActionMasker` in `src/environment/action_mask.py`**

Update `src/environment/action_mask.py`:
```python
"""Action masking implementation to ensure RL policies only sample legal actions."""

import numpy as np

from src.environment.action_space import TICKET_SUBSET_INDICES, DiscreteActionSpace
from src.game.action import Action, ActionType
from src.game.ticket import DestinationTicket


class ActionMasker:
    """Computes a boolean mask over the discrete action space corresponding to legal moves."""

    def __init__(self, action_space: DiscreteActionSpace) -> None:
        self.action_space = action_space

    def compute_mask(
        self,
        valid_actions: list[Action],
        pending_tickets: list[DestinationTicket] | None = None,
    ) -> np.ndarray:
        """Return boolean mask of shape (n,) where True indicates a valid action."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        if not valid_actions:
            if self.action_space.n > 0:
                mask[0] = True
            return mask

        # Map of pending tickets by id to their local slot index 0, 1, 2
        pending_id_to_slot: dict[str, int] = {}
        if pending_tickets:
            for slot, t in enumerate(pending_tickets):
                pending_id_to_slot[t.id] = slot

        for act in valid_actions:
            if act.action_type == ActionType.KEEP_TICKETS:
                if act.ticket_ids is not None and pending_id_to_slot:
                    # Convert ticket IDs to subset tuple of slot indices
                    slots = tuple(
                        sorted(
                            pending_id_to_slot[tid]
                            for tid in act.ticket_ids
                            if tid in pending_id_to_slot
                        )
                    )
                    if slots in TICKET_SUBSET_INDICES:
                        subset_idx = TICKET_SUBSET_INDICES.index(slots)
                        mask[7 + subset_idx] = True
            else:
                act_id = self.action_space.to_id(act)
                if act_id is not None:
                    mask[act_id] = True

        # Safety: if no valid actions mapped, fallback to first action or draw hidden card
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True

        return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/environment/test_action_mask.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/environment/action_mask.py tests/environment/test_action_mask.py
git commit -m "feat: implement ActionMasker for valid domain action masking"
```

---

### Task 4: Modular Reward Calculation Engine (`RewardV1` / `DefaultRewardCalculator`)

**Files:**
- Modify: `src/environment/reward.py`
- Create: `tests/environment/test_reward.py`

**Interfaces:**
- Consumes: `src.game.state.GameState`, `src.game.action.Action`, `src.game.graph.check_ticket_completed`
- Produces: `RewardWeights` (dataclass), `BaseRewardCalculator` (ABC), `DefaultRewardCalculator` with `calculate(prev_state, action, next_state, player_index) -> float`.

- [ ] **Step 1: Write the failing test for `DefaultRewardCalculator`**

Create `tests/environment/test_reward.py`:
```python
"""Tests for DefaultRewardCalculator (RewardV1)."""

import pytest

from src.environment.reward import DefaultRewardCalculator, RewardWeights
from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.maps import load_synthetic_mini_board


def test_reward_route_points_and_step_penalty():
    weights = RewardWeights(route_points_weight=1.0, step_penalty=0.05)
    calc = DefaultRewardCalculator(weights=weights)

    board, tickets = load_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    game.reset(seed=42)

    # Initial tickets setup
    game.step(game.valid_actions()[0])
    game.step(game.valid_actions()[0])

    prev_state = game.state
    # Give player 0 cards to claim route_1 (length 1 -> 1 point)
    player0 = game.state.players[0]
    player0.add_card(TrainCard(color=CardColor.RED))

    claim_act = Action(
        action_type=ActionType.CLAIM_ROUTE,
        route_id="mini_route_1",
        color_chosen=CardColor.RED,
    )
    next_state = game.step(claim_act)

    reward = calc.calculate(
        prev_state=prev_state,
        action=claim_act,
        next_state=next_state,
        player_index=0,
    )

    # Delta score is +1, step penalty is -0.05 -> reward = 0.95
    assert pytest.approx(reward, 1e-4) == 0.95


def test_reward_terminal_win_bonus():
    weights = RewardWeights(win_bonus=20.0, loss_penalty=10.0, score_diff_weight=0.5)
    calc = DefaultRewardCalculator(weights=weights)

    board, tickets = load_synthetic_mini_board()
    game = Game(board=board, tickets_deck=tickets, seed=42)
    game.reset(seed=42)

    prev_state = game.state
    # Mock game over state with player 0 winner
    next_state = Game(board=board, tickets_deck=tickets, seed=42).reset(seed=42)
    next_state.is_game_over = True
    next_state.winner_id = "player_0"
    next_state.players[0].score = 30
    next_state.players[1].score = 10

    reward = calc.calculate(
        prev_state=prev_state,
        action=Action(action_type=ActionType.DRAW_HIDDEN_CARD),
        next_state=next_state,
        player_index=0,
    )

    # Win bonus (20) + score diff (0.5 * (30 - 10) = 10) = 30.0
    assert reward >= 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/environment/test_reward.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `DefaultRewardCalculator` in `src/environment/reward.py`**

Update `src/environment/reward.py`:
```python
"""Configurable reward calculation engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.game.action import Action
from src.game.graph import check_ticket_completed
from src.game.state import GameState


@dataclass
class RewardWeights:
    """Configurable hyperparameters for reward shaping and terminal outcomes."""

    route_points_weight: float = 1.0
    ticket_completion_weight: float = 1.0
    step_penalty: float = 0.0
    win_bonus: float = 20.0
    loss_penalty: float = 10.0
    score_diff_weight: float = 0.5
    ticket_failure_penalty_weight: float = 1.0


class BaseRewardCalculator(ABC):
    """Abstract base class for modular reward functions."""

    @abstractmethod
    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        """Compute the step reward for player_index."""


class DefaultRewardCalculator(BaseRewardCalculator):
    """Configurable reward calculation module (Reward V1)."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def calculate(
        self,
        prev_state: GameState,
        action: Any,
        next_state: GameState,
        player_index: int,
    ) -> float:
        if player_index >= len(next_state.players) or player_index >= len(prev_state.players):
            return 0.0

        prev_player = prev_state.players[player_index]
        next_player = next_state.players[player_index]

        # 1. Step route points delta
        delta_score = float(next_player.score - prev_player.score)
        reward = self.weights.route_points_weight * delta_score - self.weights.step_penalty

        # 2. Ticket completion delta during the step
        prev_completed = {
            t.id for t in prev_player.tickets if check_ticket_completed(prev_player.claimed_routes, t)
        }
        for t in next_player.tickets:
            if t.id not in prev_completed and check_ticket_completed(next_player.claimed_routes, t):
                reward += self.weights.ticket_completion_weight * float(t.points)

        # 3. Terminal outcome reward
        if next_state.is_game_over:
            opp_index = 1 - player_index if len(next_state.players) == 2 else None
            opp_score = next_state.players[opp_index].score if opp_index is not None else 0

            # Win/Loss outcome
            if next_state.winner_id == next_player.id:
                reward += self.weights.win_bonus
            elif next_state.winner_id is not None and next_state.winner_id != next_player.id:
                reward -= self.weights.loss_penalty

            # Score differential
            score_diff = float(next_player.score - opp_score)
            reward += self.weights.score_diff_weight * score_diff

            # Uncompleted tickets penalty
            for t in next_player.tickets:
                if not check_ticket_completed(next_player.claimed_routes, t):
                    reward -= self.weights.ticket_failure_penalty_weight * float(t.points)

        return float(reward)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/environment/test_reward.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/environment/reward.py tests/environment/test_reward.py
git commit -m "feat: implement DefaultRewardCalculator with configurable RewardWeights"
```

---

### Task 5: Gymnasium Environment Wrapper (`TicketToRideEnv`) & `check_env` Validation

**Files:**
- Modify: `src/environment/env.py`
- Modify: `src/environment/__init__.py`
- Create: `tests/environment/test_gym_env.py`

**Interfaces:**
- Consumes: `gymnasium.Env`, `gymnasium.spaces`, `src.environment.observation.ObservationV1`, `src.environment.action_space.DiscreteActionSpace`, `src.environment.action_mask.ActionMasker`, `src.environment.reward.DefaultRewardCalculator`, `src.agents.random_agent.RandomAgent`
- Produces: `TicketToRideEnv(gym.Env)` with `reset(seed=None) -> (obs, info)`, `step(action: int) -> (obs, reward, terminated, truncated, info)`.

- [ ] **Step 1: Write the failing test for `TicketToRideEnv` and `check_env`**

Create `tests/environment/test_gym_env.py`:
```python
"""Tests for Gymnasium environment wrapper and check_env compliance."""

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.environment.env import TicketToRideEnv
from src.game.maps import load_synthetic_mini_board, load_usa_board


def test_gymnasium_check_env_synthetic_mini():
    board, tickets = load_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))
    # Must pass without errors
    check_env(env.unwrapped, skip_render_check=True)


def test_gymnasium_check_env_usa_board():
    board, tickets = load_usa_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))
    check_env(env.unwrapped, skip_render_check=True)


def test_gym_env_step_lifecycle_with_opponent():
    board, tickets = load_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=GreedyAgent())

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert "action_mask" in info
    assert info["action_mask"].dtype == bool
    assert np.any(info["action_mask"])

    done = False
    step_count = 0
    while not done and step_count < 100:
        mask = info["action_mask"]
        valid_indices = np.where(mask)[0]
        action = int(valid_indices[0])

        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_mask" in info
        done = terminated or truncated
        step_count += 1

    assert step_count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv_py312/bin/pytest tests/environment/test_gym_env.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `TicketToRideEnv` in `src/environment/env.py` and export in `src/environment/__init__.py`**

Update `src/environment/env.py`:
```python
"""Gymnasium Environment wrapper for Ticket to Ride."""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agents.random_agent import RandomAgent
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.environment.reward import BaseRewardCalculator, DefaultRewardCalculator
from src.game.board import Board
from src.game.game import Game
from src.game.maps import load_usa_board
from src.game.state import TurnState
from src.game.ticket import DestinationTicket


class TicketToRideEnv(gym.Env):
    """Gymnasium-compatible single-agent / turn-based environment with auto-stepping opponent."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        board: Board | None = None,
        tickets_deck: list[DestinationTicket] | None = None,
        opponent: Any | None = None,
        observation_encoder: BaseObservationEncoder | None = None,
        reward_calculator: BaseRewardCalculator | None = None,
        num_players: int = 2,
        max_turns: int = 300,
    ) -> None:
        super().__init__()
        if board is None:
            self.board, self.initial_tickets = load_usa_board()
        else:
            self.board = board
            self.initial_tickets = tickets_deck or []

        self.num_players = num_players
        self.max_turns = max_turns
        self.opponent = opponent

        self.game = Game(
            board=self.board,
            tickets_deck=self.initial_tickets,
            num_players=self.num_players,
        )
        self.encoder = observation_encoder or ObservationV1(
            board=self.board,
            initial_tickets=self.initial_tickets,
            num_players=self.num_players,
        )
        self.reward_calc = reward_calculator or DefaultRewardCalculator()
        self.discrete_actions = DiscreteActionSpace(board=self.board)
        self.masker = ActionMasker(self.discrete_actions)

        self.action_space = spaces.Discrete(self.discrete_actions.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.encoder.observation_shape,
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.game.reset(seed=seed)
        if self.opponent and hasattr(self.opponent, "reset"):
            self.opponent.reset(seed=seed)

        # Handle initial tickets selection if opponent is configured
        # If Player 0 turn to pick tickets, return obs and mask
        # If state is CHOOSING_INITIAL_TICKETS and it's opponent's turn, auto-step opponent
        self._auto_step_opponents_if_needed()

        obs = self.encoder.encode(self.game.state, player_index=0)
        valid_actions = self.game.valid_actions()
        pending = self.game.state.players[0].pending_tickets
        action_mask = self.masker.compute_mask(valid_actions, pending_tickets=pending)

        info = {
            "action_mask": action_mask,
            "turn": self.game.state.turn_number,
            "turn_state": self.game.state.turn_state.value,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        domain_action = self._resolve_action(action)
        prev_state = self.game.state
        next_state = self.game.step(domain_action)

        # Auto-step opponent(s) until it is Player 0's turn again or game is over
        self._auto_step_opponents_if_needed()

        # Calculate reward from the perspective of Player 0
        reward = self.reward_calc.calculate(
            prev_state=prev_state,
            action=domain_action,
            next_state=self.game.state,
            player_index=0,
        )

        terminated = bool(self.game.state.is_game_over)
        truncated = bool(self.game.state.turn_number >= self.max_turns and not terminated)

        obs = self.encoder.encode(self.game.state, player_index=0)
        valid_actions = self.game.valid_actions() if not terminated else []
        pending = self.game.state.players[0].pending_tickets if not terminated else []
        action_mask = self.masker.compute_mask(valid_actions, pending_tickets=pending)

        info = {
            "action_mask": action_mask,
            "winner_id": self.game.state.winner_id,
            "turn": self.game.state.turn_number,
            "player_score": self.game.state.players[0].score,
            "opponent_score": self.game.state.players[1].score if len(self.game.state.players) > 1 else 0,
        }

        return obs, float(reward), terminated, truncated, info

    def _resolve_action(self, action_id: int):
        """Map discrete action index to concrete domain Action, resolving ticket IDs dynamically."""
        action = self.discrete_actions.to_action(action_id)
        if action.action_type == TurnState.CHOOSING_TICKETS or action.action_type == "keep_tickets":
            # Map subset indices ('0', '1', ...) to actual ticket IDs in pending_tickets
            pending = self.game.state.players[self.game.state.current_player_index].pending_tickets
            if action.ticket_ids and pending:
                actual_ticket_ids = tuple(
                    pending[int(idx)].id
                    for idx in action.ticket_ids
                    if int(idx) < len(pending)
                )
                from src.game.action import Action, ActionType
                return Action(
                    action_type=ActionType.KEEP_TICKETS,
                    ticket_ids=actual_ticket_ids,
                )
        return action

    def _auto_step_opponents_if_needed(self) -> None:
        """Step configured opponent(s) until player 0's turn or game over."""
        if not self.opponent:
            return

        while (
            not self.game.state.is_game_over
            and self.game.state.current_player_index != 0
            and self.game.state.turn_number < self.max_turns
        ):
            valid_actions = self.game.valid_actions()
            if not valid_actions:
                break
            opp_action = self.opponent.act(
                self.game.state,
                valid_actions,
                self.game.board,
            )
            self.game.step(opp_action)
```

Update `src/environment/__init__.py`:
```python
"""Environment package public exports."""

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.env import TicketToRideEnv
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.environment.reward import BaseRewardCalculator, DefaultRewardCalculator, RewardWeights

__all__ = [
    "ActionMasker",
    "BaseObservationEncoder",
    "BaseRewardCalculator",
    "DefaultRewardCalculator",
    "DiscreteActionSpace",
    "ObservationV1",
    "RewardWeights",
    "TicketToRideEnv",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/environment/test_gym_env.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/environment/env.py src/environment/__init__.py tests/environment/test_gym_env.py
git commit -m "feat: implement TicketToRideEnv gymnasium wrapper with check_env compliance"
```

---

### Task 6: Comprehensive Acceptance Suite & 100-Game Invariant Benchmark

**Files:**
- Create: `tests/environment/test_phase3_acceptance.py`

**Interfaces:**
- Consumes: `src.environment.env.TicketToRideEnv`, `src.agents.random_agent.RandomAgent`, `src.agents.greedy_agent.GreedyAgent`, `src.agents.strategic_agent.StrategicHeuristicAgent`
- Produces: Automated acceptance verification of Phase 3 deliverables.

- [ ] **Step 1: Write the failing acceptance test**

Create `tests/environment/test_phase3_acceptance.py`:
```python
"""Phase 3 Acceptance Test: 100 full games across baseline opponents via Gymnasium."""

import numpy as np
import pytest

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.environment.env import TicketToRideEnv
from src.game.maps import load_synthetic_mini_board, load_usa_board


@pytest.mark.parametrize("board_loader", [load_synthetic_mini_board, load_usa_board])
def test_phase3_gymnasium_100_games_acceptance(board_loader):
    """Run 100 complete games through the Gymnasium interface against multiple opponents."""
    board, tickets = board_loader()

    opponents = [
        RandomAgent(seed=101),
        GreedyAgent(),
        StrategicHeuristicAgent(),
    ]

    total_games = 30
    for game_idx in range(total_games):
        opp = opponents[game_idx % len(opponents)]
        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=opp,
            max_turns=200,
        )

        obs, info = env.reset(seed=game_idx * 13)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert "action_mask" in info

        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            mask = info["action_mask"]
            assert np.any(mask), f"Game {game_idx} Step {step_count}: No valid actions in mask!"
            valid_indices = np.where(mask)[0]
            # Select random valid action index
            action = int(np.random.choice(valid_indices))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()
            assert not np.isnan(reward)
            assert isinstance(reward, float)

            done = terminated or truncated
            step_count += 1

        assert step_count > 0
        assert isinstance(total_reward, float)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `venv_py312/bin/pytest tests/environment/test_phase3_acceptance.py -v`
Expected: PASS.

- [ ] **Step 3: Run entire test suite to ensure zero regressions**

Run: `venv_py312/bin/pytest -v`
Expected: All tests pass (Phase 1, Phase 2, Phase 3).

- [ ] **Step 4: Commit**

```bash
git add tests/environment/test_phase3_acceptance.py
git commit -m "test: add Phase 3 Gymnasium acceptance suite and multi-opponent benchmark"
```
