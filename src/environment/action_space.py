"""Bi-directional mapping between discrete integer action IDs and domain Actions."""

from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor
from src.game.maps import load_usa_board
from src.game.state import GameState

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
        self._key_to_id: dict[tuple, int] = {}
        self._build_action_space()

    def _build_action_space(self) -> None:
        current_id = 0

        # 0: DRAW_HIDDEN_CARD
        act = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
        self._id_to_action[current_id] = act
        self._action_to_id[act] = current_id
        self._key_to_id[(act.action_type, act.card_index, act.route_id, act.color_chosen)] = current_id
        current_id += 1

        # 1..5: DRAW_VISIBLE_CARD (slots 0..4)
        for slot in range(5):
            act = Action(action_type=ActionType.DRAW_VISIBLE_CARD, card_index=slot)
            self._id_to_action[current_id] = act
            self._action_to_id[act] = current_id
            self._key_to_id[(act.action_type, slot, act.route_id, act.color_chosen)] = current_id
            current_id += 1

        # 6: DRAW_TICKETS
        act = Action(action_type=ActionType.DRAW_TICKETS)
        self._id_to_action[current_id] = act
        self._action_to_id[act] = current_id
        self._key_to_id[(act.action_type, act.card_index, act.route_id, act.color_chosen)] = current_id
        current_id += 1

        # 7..13: KEEP_TICKETS (subsets of {0, 1, 2} represented by string tuple index markers)
        for subset in TICKET_SUBSET_INDICES:
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
                    self._key_to_id[(act.action_type, None, r.id, color)] = current_id
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
                self._key_to_id[(act.action_type, None, r.id, r.color)] = current_id
                current_id += 1

    @property
    def n(self) -> int:
        return len(self._id_to_action)

    @property
    def size(self) -> int:
        return self.n

    def to_action(self, action_id: int) -> Action:
        return self._id_to_action[action_id]

    def to_id(self, action: Action) -> int | None:
        key = (action.action_type, action.card_index, action.route_id, action.color_chosen)
        if key in self._key_to_id:
            return self._key_to_id[key]
        return self._action_to_id.get(action)

    def encode_action(self, action: Action) -> int:
        act_id = self.to_id(action)
        return 0 if act_id is None else act_id

    def decode(self, action_id: int, state: GameState | None = None) -> Action:
        """Decode integer action ID to domain Action, resolving pending tickets if state provided."""
        action = self.to_action(action_id)
        if state is not None and action.action_type == ActionType.KEEP_TICKETS:
            player = state.current_player
            if player and player.pending_tickets and action.ticket_ids:
                slots = set(int(idx) for idx in action.ticket_ids if str(idx).isdigit())
                chosen_ids = tuple(t.id for slot_i, t in enumerate(player.pending_tickets) if slot_i in slots)
                return Action(action_type=ActionType.KEEP_TICKETS, ticket_ids=chosen_ids)
        return action


ActionSpaceV1 = DiscreteActionSpace

__all__ = [
    "ActionSpaceV1",
    "DiscreteActionSpace",
    "STANDARD_CLAIM_COLORS",
    "TICKET_SUBSET_INDICES",
]
