"""Official rules, invariant validations, and legal action generation for Ticket to Ride."""

from itertools import combinations

from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor, TrainCard
from src.game.player import Player
from src.game.route import Route
from src.game.state import GameState, TurnState

ROUTE_POINTS_BY_LENGTH: dict[int, int] = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    5: 10,
    6: 15,
}


class GameRules:
    """Rules and invariant validations for Ticket to Ride."""

    INITIAL_TRAINS_PER_PLAYER = 45
    INITIAL_CARDS_PER_PLAYER = 4
    VISIBLE_CARDS_COUNT = 5
    INITIAL_TICKETS_DRAW_COUNT = 3
    MIN_TICKETS_KEEP_START = 2
    MIDGAME_TICKETS_DRAW_COUNT = 3
    MIN_TICKETS_KEEP_INGAME = 1
    LONGEST_PATH_BONUS_POINTS = 10

    @staticmethod
    def points_for_route_length(length: int) -> int:
        return ROUTE_POINTS_BY_LENGTH.get(length, 0)

    @staticmethod
    def should_flush_visible_cards(visible_cards: list[TrainCard]) -> bool:
        """Official rule: If 3 or more of the 5 visible cards are Locomotives, flush all 5."""
        if len(visible_cards) < 3:
            return False
        loco_count = sum(1 for c in visible_cards if c.is_locomotive())
        return loco_count >= 3

    @staticmethod
    def can_claim_route(
        route: Route,
        player: Player,
        state: GameState,
        board: Board,
        num_players: int = 2,
    ) -> bool:
        """Check if player can legally claim route given current state and player count rules."""
        if route.is_claimed:
            return False

        if not player.has_trains_for_route(route):
            return False

        # Double route rules
        if route.is_double_route and route.double_route_pair_id:
            pair = board.get_route(route.double_route_pair_id)
            if pair and pair.is_claimed:
                if num_players <= 3:
                    # In 2 or 3 player game, if either double route is claimed, the other is blocked
                    return False
                if pair.claimed_by == player.id:
                    # In 4 or 5 player game, same player cannot claim both routes
                    return False

        # Must have at least one payment option
        options = player.can_afford_route(route)
        return len(options) > 0

    @classmethod
    def get_valid_actions(
        cls,
        player: Player,
        state: GameState,
        board: Board,
        num_players: int = 2,
    ) -> list[Action]:
        """Compute all strictly legal actions for the player in the current state."""
        actions: list[Action] = []

        if state.turn_state == TurnState.CHOOSING_INITIAL_TICKETS:
            pending = player.pending_tickets
            n_pending = len(pending)
            for k in range(cls.MIN_TICKETS_KEEP_START, n_pending + 1):
                for comb in combinations(pending, k):
                    actions.append(
                        Action(
                            action_type=ActionType.KEEP_TICKETS,
                            ticket_ids=tuple(t.id for t in comb),
                        )
                    )
            return actions

        if state.turn_state == TurnState.CHOOSING_TICKETS:
            pending = player.pending_tickets
            n_pending = len(pending)
            min_keep = min(cls.MIN_TICKETS_KEEP_INGAME, n_pending)
            for k in range(min_keep, n_pending + 1):
                for comb in combinations(pending, k):
                    actions.append(
                        Action(
                            action_type=ActionType.KEEP_TICKETS,
                            ticket_ids=tuple(t.id for t in comb),
                        )
                    )
            return actions

        if state.turn_state == TurnState.DRAWING_SECOND_CARD:
            # Can draw hidden card if deck (or discard) has cards
            if state.train_deck or state.discard_pile:
                actions.append(Action(action_type=ActionType.DRAW_HIDDEN_CARD))

            # Can draw any visible card EXCEPT locomotives
            for idx, card in enumerate(state.visible_cards):
                if not card.is_locomotive():
                    actions.append(
                        Action(
                            action_type=ActionType.DRAW_VISIBLE_CARD,
                            card_index=idx,
                        )
                    )
            return actions

        if state.turn_state == TurnState.NORMAL:
            # Option 1: Draw hidden card
            if state.train_deck or state.discard_pile:
                actions.append(Action(action_type=ActionType.DRAW_HIDDEN_CARD))

            # Option 2: Draw visible card (any card, including locomotives)
            for idx in range(len(state.visible_cards)):
                actions.append(
                    Action(
                        action_type=ActionType.DRAW_VISIBLE_CARD,
                        card_index=idx,
                    )
                )

            # Option 3: Claim route (single-pass check)
            for route in board.routes:
                if route.is_claimed or not player.has_trains_for_route(route):
                    continue

                if route.is_double_route and route.double_route_pair_id:
                    pair = board.get_route(route.double_route_pair_id)
                    if pair and pair.is_claimed:
                        if num_players <= 3:
                            continue
                        if pair.claimed_by == player.id:
                            continue

                options = player.can_afford_route(route)
                for option in options:
                    locos = option.get(CardColor.LOCOMOTIVE, 0)
                    color_chosen = None
                    for c in option:
                        if c != CardColor.LOCOMOTIVE:
                            color_chosen = c
                            break
                    if color_chosen is None and locos > 0:
                        color_chosen = CardColor.LOCOMOTIVE

                    actions.append(
                        Action(
                            action_type=ActionType.CLAIM_ROUTE,
                            route_id=route.id,
                            color_chosen=color_chosen,
                            locomotives_count=locos,
                        )
                    )

            # Option 4: Draw destination tickets
            if state.ticket_deck:
                actions.append(Action(action_type=ActionType.DRAW_TICKETS))

            return actions

        return actions
