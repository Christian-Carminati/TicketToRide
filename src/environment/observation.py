"""Observation encoders for converting GameState to bounded vector representations."""

from abc import ABC, abstractmethod
import numpy as np

from src.game.board import Board
from src.game.card import CardColor
from src.game.graph import check_tickets_completed_batch
from src.game.maps import load_usa_board
from src.game.route import Route
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
        self._routes_by_id = {r.id: r for r in self.board.routes}
        self._route_ids = sorted(self._routes_by_id.keys())
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
        all_claimed_ids = {
            rid: p.id for p in state.players for rid in p.claimed_route_ids
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
        player_routes = [
            self._routes_by_id[rid]
            for rid in player.claimed_route_ids
            if rid in self._routes_by_id
        ]
        completion_status = check_tickets_completed_batch(player_routes, player.tickets)
        owned_ticket_map = {t.id: t for t in player.tickets}

        for i, t_id in enumerate(self._ticket_ids):
            base = offset + i * 3
            if t_id in owned_ticket_map:
                ticket_obj = owned_ticket_map[t_id]
                is_completed = completion_status.get(t_id, False)
                obs[base] = 1.0  # Owned
                obs[base + 1] = 1.0 if is_completed else 0.0
                obs[base + 2] = min(ticket_obj.points / 25.0, 1.0)
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
            obs[base + 2] = min(len(p.claimed_route_ids) / 30.0, 1.0)
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
