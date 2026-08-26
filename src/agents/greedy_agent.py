"""Greedy baseline agent maximizing immediate scoring opportunities."""

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
                all_pending = {t.id: t.points for t in player.pending_tickets} if player else {}

                def ticket_combo_value(a: Action) -> int:
                    return sum(all_pending.get(tid, 0) for tid in (a.ticket_ids or ()))

                # Sort descending by total points, then by number of tickets kept
                ticket_actions.sort(
                    key=lambda a: (ticket_combo_value(a), len(a.ticket_ids or ())),
                    reverse=True,
                )
                return ticket_actions[0]

        # 2. Handle Drawing Second Card
        if state.turn_state == TurnState.DRAWING_SECOND_CARD:
            visible_draws = [
                a for a in valid_actions if a.action_type == ActionType.DRAW_VISIBLE_CARD
            ]
            if visible_draws:
                best_visible = self._pick_best_visible_card(state, visible_draws)
                if best_visible:
                    return best_visible
            hidden_draws = [
                a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD
            ]
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
                points = self.rules.points_for_route_length(r.length)
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

    def _pick_best_visible_card(
        self, state: GameState, visible_actions: list[Action]
    ) -> Action | None:
        player = state.current_player
        if not player:
            return visible_actions[0] if visible_actions else None

        # Find non-locomotive color with highest count in player's hand
        hand_counts = {
            c: count for c, count in player.cards.items() if c != CardColor.LOCOMOTIVE and count > 0
        }
        target_color: CardColor | None = None
        if hand_counts:
            target_color = max(hand_counts, key=lambda k: hand_counts[k])

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

        # If nothing matches, return first visible card if available
        return visible_actions[0] if visible_actions else None
