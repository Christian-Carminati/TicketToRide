"""Strategic Graph-Aware Heuristic Agent."""

import heapq
from collections import defaultdict
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
        player_routes = [
            board.get_route(rid)
            for rid in player.claimed_route_ids
            if board.get_route(rid) is not None
        ]
        active_tickets = [
            t
            for t in player.tickets
            if not check_ticket_completed(player_routes, t)
        ]
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
            target_claim_actions = [
                a for a in claim_actions if a.route_id in {r.id for r in target_routes}
            ]
            if target_claim_actions:
                # Prioritize single-track bottlenecks (fewer parallel routes) and longer segments
                def claim_priority(a: Action) -> tuple[int, int]:
                    r = board.get_route(a.route_id or "")
                    if not r:
                        return (0, 0)
                    is_bottleneck = 1 if len(board.get_routes_between(r.city_a, r.city_b)) == 1 else 0
                    return (is_bottleneck, r.length)

                target_claim_actions.sort(key=claim_priority, reverse=True)
                return target_claim_actions[0]

            # If all tickets completed or trains are low, claim highest value available route
            if not active_tickets or player.trains_remaining <= 12:

                def general_route_score(a: Action) -> int:
                    r = board.get_route(a.route_id or "")
                    return self.rules.points_for_route_length(r.length) if r else 0

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

    def _choose_best_tickets(
        self, player: Player, valid_actions: list[Action], board: Board
    ) -> Action:
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

            total_train_cost = sum(
                board.get_route(rid).length for rid in needed_routes if board.get_route(rid)
            )
            if total_train_cost == 0:
                return float(total_points)
            return total_points / float(total_train_cost)

        ticket_actions.sort(key=subset_efficiency, reverse=True)
        return ticket_actions[0]

    def _compute_shortest_path_routes(
        self, player: Player, ticket: DestinationTicket, board: Board
    ) -> list[Route]:
        """Dijkstra shortest path algorithm on the board network."""
        city_start = ticket.city_a
        city_target = ticket.city_b

        dist: dict[str, float] = {city_start: 0.0}
        prev_route: dict[str, Route | None] = {city_start: None}
        prev_city: dict[str, str | None] = {city_start: None}

        pq: list[tuple[float, str]] = [(0.0, city_start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            if u == city_target:
                break

            for r in board.routes:
                if r.city_a != u and r.city_b != u:
                    continue

                # If claimed by opponent, impassable
                if r.claimed_by is not None and r.claimed_by != player.id:
                    continue

                v = r.city_b if r.city_a == u else r.city_a
                weight = 0.0 if r.claimed_by == player.id else float(r.length)

                if dist.get(u, float("inf")) + weight < dist.get(v, float("inf")):
                    dist[v] = dist[u] + weight
                    prev_route[v] = r
                    prev_city[v] = u
                    heapq.heappush(pq, (dist[v], v))

        if city_target not in dist or dist[city_target] == float("inf"):
            return []

        # Reconstruct path
        path_routes: list[Route] = []
        curr = city_target
        while curr != city_start:
            r = prev_route.get(curr)
            if not r:
                break
            path_routes.append(r)
            curr = prev_city.get(curr, city_start)

        return path_routes

    def _calculate_card_deficit(
        self, player: Player, routes: list[Route]
    ) -> dict[CardColor, int]:
        needed: dict[CardColor, int] = defaultdict(int)
        wild_needed = 0

        for r in routes:
            if r.claimed_by == player.id:
                continue
            if r.color is None or r.color == CardColor.GRAY:
                wild_needed += r.length
            else:
                needed[r.color] += r.length

        deficit: dict[CardColor, int] = {}
        for color, count in needed.items():
            have = player.cards.get(color, 0)
            if have < count:
                deficit[color] = count - have

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
        player_routes = [
            board.get_route(rid)
            for rid in player.claimed_route_ids
            if board.get_route(rid) is not None
        ]
        active_tickets = [
            t
            for t in player.tickets
            if not check_ticket_completed(player_routes, t)
        ]
        target_routes: list[Route] = []
        for t in active_tickets:
            for r in self._compute_shortest_path_routes(player, t, board):
                if r.id not in player.claimed_route_ids and r not in target_routes:
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
