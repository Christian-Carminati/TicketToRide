"""Steiner Minimal Tree (SMT) Heuristic Agent.

Implements the Takahashi-Matsuyama (1980) approximation algorithm for Steiner Minimal Trees
on graphs to optimize multi-commodity destination ticket networks in Ticket to Ride.
Guarantees a (2 - 2/|T|)-approximation factor over the NP-hard Steiner tree problem.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

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


class SteinerTreeAgent(BaseAgent):
    """
    Advanced heuristic agent utilizing Takahashi-Matsuyama Steiner Tree approximation.

    Key Advantages over Independent Dijkstra:
    1. Multi-terminal graph reuse: Connects multiple destination tickets via minimal shared subgraphs.
    2. Dynamic tree expansion: Iteratively attaches unreached terminals to the closest point of the existing network.
    3. Global train economy: Slashes redundant parallel branches, conserving train cars for endgame scoring.
    """

    def __init__(self, name: str = "SteinerTreeAgent") -> None:
        super().__init__(name=name)
        self.rules = GameRules()

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("No valid actions available for SteinerTreeAgent.")

        if board is None:
            return valid_actions[0]

        player = state.current_player
        if not player:
            return valid_actions[0]

        # 1. Handle Ticket Selection (Initial & Mid-game)
        if state.turn_state in [TurnState.CHOOSING_INITIAL_TICKETS, TurnState.CHOOSING_TICKETS]:
            return self._choose_best_tickets_steiner(player, valid_actions, board)

        # 2. Handle Drawing Second Card
        if state.turn_state == TurnState.DRAWING_SECOND_CARD:
            return self._handle_second_card_draw(state, player, valid_actions)

        # 3. Main Decision: Compute Steiner Tree for Active Incomplete Tickets
        player_routes = [
            r for rid in player.claimed_route_ids if (r := board.get_route(rid)) is not None
        ]
        active_tickets = [t for t in player.tickets if not check_ticket_completed(player_routes, t)]

        # Collect terminal cities from all incomplete tickets
        terminals: set[str] = set()
        for t in active_tickets:
            terminals.add(t.city_a)
            terminals.add(t.city_b)

        target_routes = self._compute_steiner_tree_routes(player, terminals, board)

        # Check for Claiming Target Routes in the Steiner Tree
        claim_actions = [a for a in valid_actions if a.action_type == ActionType.CLAIM_ROUTE]
        if claim_actions:
            target_claim_actions = [
                a for a in claim_actions if a.route_id in {r.id for r in target_routes}
            ]
            if target_claim_actions:
                # Prioritize single-track bottlenecks and longer segments
                def claim_priority(a: Action) -> tuple[int, int]:
                    r = board.get_route(a.route_id or "")
                    if not r:
                        return (0, 0)
                    is_bottleneck = 1 if len(board.get_routes_between(r.city_a, r.city_b)) == 1 else 0
                    return (is_bottleneck, r.length)

                target_claim_actions.sort(key=claim_priority, reverse=True)
                return target_claim_actions[0]

            # Endgame: if all tickets completed or trains running low, claim highest-value routes
            if not active_tickets or player.trains_remaining <= 12:
                def general_route_score(a: Action) -> int:
                    r = board.get_route(a.route_id or "")
                    return self.rules.points_for_route_length(r.length) if r else 0

                claim_actions.sort(key=general_route_score, reverse=True)
                return claim_actions[0]

        # 4. If cannot claim, draw cards targeting deficient colors for Steiner tree
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

    def _compute_steiner_tree_routes(
        self, player: Player, terminals: set[str], board: Board
    ) -> list[Route]:
        """
        Takahashi-Matsuyama Steiner Minimal Tree approximation.
        Iteratively finds the closest terminal to the current tree and attaches the connecting path.
        """
        if not terminals or len(terminals) <= 1:
            return []

        # Find starting terminal with highest degree or first terminal
        term_list = sorted(terminals)
        current_tree_nodes: set[str] = {term_list[0]}
        unconnected_terminals: set[str] = set(term_list[1:])
        tree_route_ids: set[str] = set()

        opp_claimed = set()
        for r_obj in board.routes:
            if r_obj.claimed_by is not None and r_obj.claimed_by != player.id:
                opp_claimed.add(r_obj.id)

        while unconnected_terminals:
            # Multi-source Dijkstra starting simultaneously from all nodes in current_tree_nodes
            dist: dict[str, float] = {node: 0.0 for node in current_tree_nodes}
            prev_route: dict[str, Route | None] = {node: None for node in current_tree_nodes}
            prev_city: dict[str, str | None] = {node: None for node in current_tree_nodes}

            pq: list[tuple[float, str]] = [(0.0, node) for node in current_tree_nodes]
            heapq.heapify(pq)

            closest_terminal: str | None = None
            closest_dist = float("inf")

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float("inf")):
                    continue

                if u in unconnected_terminals and d < closest_dist:
                    closest_terminal = u
                    closest_dist = d
                    break  # Found closest terminal to current tree!

                for r in board.get_adjacent_routes(u):
                    if r.id in opp_claimed:
                        continue

                    v = r.city_b if r.city_a == u else r.city_a
                    # Routes already owned by player cost 0 weight
                    weight = 0.0 if r.claimed_by == player.id else float(r.length)

                    if dist.get(u, float("inf")) + weight < dist.get(v, float("inf")):
                        dist[v] = dist[u] + weight
                        prev_route[v] = r
                        prev_city[v] = u
                        heapq.heappush(pq, (dist[v], v))

            if not closest_terminal or closest_dist == float("inf"):
                # Disconnected terminal due to opponent blockage
                break

            # Backtrack path from closest_terminal to current_tree_nodes
            curr = closest_terminal
            while curr not in current_tree_nodes:
                r_obj = prev_route.get(curr)
                if not r_obj:
                    break
                tree_route_ids.add(r_obj.id)
                next_city = prev_city.get(curr)
                current_tree_nodes.add(curr)
                if not next_city:
                    break
                curr = next_city

            unconnected_terminals.remove(closest_terminal)

        # Return needed routes that are not yet claimed by player
        needed_routes = []
        for rid in tree_route_ids:
            r = board.get_route(rid)
            if r and r.claimed_by != player.id:
                needed_routes.append(r)

        return needed_routes

    def _choose_best_tickets_steiner(
        self, player: Player, valid_actions: list[Action], board: Board
    ) -> Action:
        ticket_actions = [a for a in valid_actions if a.action_type == ActionType.KEEP_TICKETS]
        if not ticket_actions:
            return valid_actions[0]

        pending_by_id = {t.id: t for t in player.pending_tickets}

        def steiner_efficiency(action: Action) -> float:
            ticket_ids = action.ticket_ids or ()
            if not ticket_ids:
                return 0.0

            tickets = [pending_by_id[tid] for tid in ticket_ids if tid in pending_by_id]
            total_points = sum(t.points for t in tickets)

            terminals: set[str] = set()
            for t in tickets:
                terminals.add(t.city_a)
                terminals.add(t.city_b)

            needed_routes = self._compute_steiner_tree_routes(player, terminals, board)
            train_cost = sum(r.length for r in needed_routes)

            if train_cost == 0:
                return float(total_points)
            return total_points / float(train_cost)

        ticket_actions.sort(key=steiner_efficiency, reverse=True)
        return ticket_actions[0]

    def _calculate_card_deficit(self, player: Player, routes: list[Route]) -> dict[CardColor, int]:
        needed: dict[CardColor, int] = defaultdict(int)
        wild_needed = 0

        for r in routes:
            if r.claimed_by == player.id:
                continue
            if r.color is None:
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
        # 1. Take Locomotive if available
        for a in visible_actions:
            idx = a.card_index or 0
            if 0 <= idx < len(state.visible_cards):
                card = state.visible_cards[idx]
                if card.color == CardColor.LOCOMOTIVE:
                    return a

        # 2. Take deficient color
        for a in visible_actions:
            idx = a.card_index or 0
            if 0 <= idx < len(state.visible_cards):
                card = state.visible_cards[idx]
                if card.color in deficit and deficit[card.color] > 0:
                    return a

        return None

    def _handle_second_card_draw(
        self, state: GameState, player: Player, valid_actions: list[Action]
    ) -> Action:
        visible_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_VISIBLE_CARD]
        if not visible_draws:
            hidden = [a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD]
            return hidden[0] if hidden else valid_actions[0]

        # In second draw, cannot take locomotive from visible pool
        hidden_draws = [a for a in valid_actions if a.action_type == ActionType.DRAW_HIDDEN_CARD]
        return hidden_draws[0] if hidden_draws else visible_draws[0]
