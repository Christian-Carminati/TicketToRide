"""Monte Carlo Tree Search (MCTS) core implementation.

Includes MCTSNode, MCTSConfig, UCT action selection, heuristic rollout policies,
continuous leaf evaluation, and the complete MCTSSearchEngine with prioritized expansion.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
from typing import Any

import numpy as np

from src.agents.greedy_agent import GreedyAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.game.action import Action, ActionType
from src.game.board import Board
from src.game.card import CardColor
from src.game.game import Game
from src.game.graph import check_ticket_completed, compute_longest_continuous_path
from src.game.player import Player
from src.game.random import SeededRNG
from src.game.route import Route
from src.game.rules import GameRules
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket
from src.rl.mcts_determinization import determinize_game


class RolloutPolicyType(str, Enum):
    """Supported rollout simulation policies."""

    RANDOM = "random"
    GREEDY = "greedy"
    STRATEGIC = "strategic"


@dataclass
class MCTSConfig:
    """Hyperparameters and configuration for MCTS search engine."""

    num_simulations: int = 100
    c_puct: float = 1.41421356  # sqrt(2)
    max_rollout_depth: int = 10
    rollout_policy: RolloutPolicyType = RolloutPolicyType.STRATEGIC
    use_determinization: bool = True
    seed: int = 42
    heuristic_weights: tuple[float, float, float, float] = (0.60, 0.25, 0.10, 0.05)


def _prioritize_actions(
    valid_actions: list[Action],
    state: GameState,
    board: Board,
    player_id: str,
) -> list[Action]:
    """Sort untried actions by heuristic domain priority to accelerate MCTS convergence."""
    if len(valid_actions) <= 1:
        return list(valid_actions)

    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return list(valid_actions)

    # 1. Handle Ticket Selection (Synergy & Path Efficiency)
    if state.turn_state in [TurnState.CHOOSING_INITIAL_TICKETS, TurnState.CHOOSING_TICKETS]:
        ticket_actions = [a for a in valid_actions if a.action_type == ActionType.KEEP_TICKETS]
        if ticket_actions:
            pending_by_id = {t.id: t for t in player.pending_tickets}
            ticket_paths: dict[str, list[Route]] = {}

            for t in player.pending_tickets:
                dist: dict[str, int] = {t.city_a: 0}
                prev_route: dict[str, Route] = {}
                pq: list[tuple[int, str]] = [(0, t.city_a)]
                visited: set[str] = set()

                while pq:
                    d, u = heapq.heappop(pq)
                    if u in visited:
                        continue
                    visited.add(u)
                    if u == t.city_b:
                        break
                    for r in board.get_adjacent_routes(u):
                        if r.is_claimed and r.claimed_by != player.id:
                            continue
                        v = r.city_b if r.city_a == u else r.city_a
                        w = 0 if r.claimed_by == player.id else r.length
                        if v not in dist or d + w < dist[v]:
                            dist[v] = d + w
                            prev_route[v] = r
                            heapq.heappush(pq, (d + w, v))

                curr_c = t.city_b
                p_routes: list[Route] = []
                while curr_c in prev_route:
                    r = prev_route[curr_c]
                    p_routes.append(r)
                    curr_c = r.city_a if r.city_b == curr_c else r.city_b
                ticket_paths[t.id] = p_routes

            def ticket_subset_efficiency(act: Action) -> float:
                t_ids = act.ticket_ids or ()
                if not t_ids:
                    return 0.0
                tickets = [pending_by_id[tid] for tid in t_ids if tid in pending_by_id]
                total_pts = sum(t.points for t in tickets)
                needed_r_ids: set[str] = set()
                for tid in t_ids:
                    for r in ticket_paths.get(tid, []):
                        if r.claimed_by != player.id:
                            needed_r_ids.add(r.id)
                train_cost = sum(
                    board.get_route(rid).length for rid in needed_r_ids if board.get_route(rid)
                )
                if train_cost == 0:
                    return float(total_pts) * 10.0
                return (total_pts / float(train_cost)) * 10.0

            return sorted(ticket_actions, key=ticket_subset_efficiency, reverse=True)

    # 2. Main Turn Decision Analysis
    player_routes = [
        board.get_route(rid)
        for rid in player.claimed_route_ids
        if board.get_route(rid) is not None
    ]
    active_tickets = [t for t in player.tickets if not check_ticket_completed(player_routes, t)]

    target_routes: list[Route] = []
    for ticket in active_tickets:
        dist = {ticket.city_a: 0}
        prev_route = {}
        pq = [(0, ticket.city_a)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == ticket.city_b:
                break
            for r in board.get_adjacent_routes(u):
                if r.is_claimed and r.claimed_by != player.id:
                    continue
                v = r.city_b if r.city_a == u else r.city_a
                w = 0 if r.claimed_by == player.id else r.length
                if v not in dist or d + w < dist[v]:
                    dist[v] = d + w
                    prev_route[v] = r
                    heapq.heappush(pq, (d + w, v))
        curr_c = ticket.city_b
        while curr_c in prev_route:
            r = prev_route[curr_c]
            if r.claimed_by != player.id and r not in target_routes:
                target_routes.append(r)
            curr_c = r.city_a if r.city_b == curr_c else r.city_b

    # Calculate exact color deficit for target routes
    needed_counts: dict[CardColor, int] = defaultdict(int)
    wild_needed = 0
    for r in target_routes:
        if r.color is None:
            wild_needed += r.length
        else:
            needed_counts[r.color] += r.length

    deficit: dict[CardColor, int] = {}
    for color, cnt in needed_counts.items():
        have = player.cards.get(color, 0)
        if have < cnt:
            deficit[color] = cnt - have
    if wild_needed > 0:
        deficit[CardColor.LOCOMOTIVE] = wild_needed

    target_route_ids = {r.id for r in target_routes}

    def action_priority(act: Action) -> float:
        if act.action_type == ActionType.CLAIM_ROUTE:
            r = board.get_route(act.route_id or "")
            if not r:
                return 10.0
            if r.id in target_route_ids:
                is_bottleneck = (
                    50.0 if len(board.get_routes_between(r.city_a, r.city_b)) == 1 else 0.0
                )
                return 200.0 + is_bottleneck + r.length * 10.0
            if not active_tickets or player.trains_remaining <= 12:
                return 100.0 + r.length * 10.0
            return 10.0

        if act.action_type == ActionType.DRAW_VISIBLE_CARD:
            if act.card_index is not None and 0 <= act.card_index < len(state.visible_cards):
                card = state.visible_cards[act.card_index]
                if card.color in deficit:
                    return 120.0 + deficit[card.color] * 10.0
                if card.is_locomotive() and not deficit:
                    return 80.0
                return 20.0
            return 10.0

        if act.action_type == ActionType.DRAW_HIDDEN_CARD:
            return 60.0

        if act.action_type == ActionType.DRAW_TICKETS:
            if len(active_tickets) == 0 and player.trains_remaining >= 20:
                return 40.0
            return -100.0

        return 0.0

    return sorted(valid_actions, key=action_priority, reverse=True)


class MCTSNode:
    """Represents a single state/decision node in the Monte Carlo search tree."""

    def __init__(
        self,
        state: GameState,
        parent: "MCTSNode | None" = None,
        action: Action | None = None,
        player_id: str = "player_0",
        untried_actions: list[Action] | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.player_id = player_id
        self.untried_actions = list(untried_actions) if untried_actions is not None else []
        self.children: dict[Action, MCTSNode] = {}
        self.visits: int = 0
        self.total_value: float = 0.0

    @property
    def value(self) -> float:
        """Expected Q-value estimate Q(s) = W / N from root perspective."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        """Check if all legal actions from this node have been branched into children."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this node represents a game over state."""
        return self.state.is_game_over

    def select_best_child(
        self, c_puct: float = 1.41421356, root_player_id: str = "player_0"
    ) -> tuple[Action, "MCTSNode"]:
        """Select child maximizing UCT score from perspective of acting player."""
        best_uct = -float("inf")
        best_pair: tuple[Action, MCTSNode] | None = None
        log_n = math.log(max(1, self.visits))
        is_root_player = self.player_id == root_player_id

        for action, child in self.children.items():
            q_val = child.value if is_root_player else -child.value
            exploration = c_puct * math.sqrt(log_n / (child.visits + 1e-6))
            uct_score = q_val + exploration

            if uct_score > best_uct:
                best_uct = uct_score
                best_pair = (action, child)

        if best_pair is None:
            raise RuntimeError("Cannot select best child from empty children dictionary.")
        return best_pair

    def expand(
        self,
        action: Action,
        next_state: GameState,
        next_player_id: str,
        untried_actions: list[Action],
    ) -> "MCTSNode":
        """Add a new child node by applying an untried legal action."""
        if action in self.untried_actions:
            self.untried_actions.remove(action)

        child_node = MCTSNode(
            state=next_state,
            parent=self,
            action=action,
            player_id=next_player_id,
            untried_actions=untried_actions,
        )
        self.children[action] = child_node
        return child_node

    def update(self, reward: float) -> None:
        """Backpropagate simulation reward (from root player perspective)."""
        self.visits += 1
        self.total_value += reward


def _estimate_ticket_progress(player: Player, board: Board) -> float:
    """Compute estimated points from destination tickets taking graph progress into account."""
    if not player.tickets:
        return 0.0

    player_routes = [
        board.get_route(rid)
        for rid in player.claimed_route_ids
        if board.get_route(rid) is not None
    ]
    net_points = 0.0

    for ticket in player.tickets:
        if check_ticket_completed(player_routes, ticket):
            net_points += ticket.points
            continue

        # Compute shortest path using Dijkstra
        dist: dict[str, int] = {ticket.city_a: 0}
        pq: list[tuple[int, str]] = [(0, ticket.city_a)]
        visited: set[str] = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == ticket.city_b:
                break

            for r in board.get_adjacent_routes(u):
                if r.is_claimed and r.claimed_by != player.id:
                    continue

                v = r.city_b if r.city_a == u else r.city_a
                weight = 0 if r.claimed_by == player.id else r.length
                new_d = d + weight

                if v not in dist or new_d < dist[v]:
                    dist[v] = new_d
                    heapq.heappush(pq, (new_d, v))

        needed_trains = dist.get(ticket.city_b, None)
        if needed_trains is None or player.trains_remaining < needed_trains:
            # Unreachable or insufficient trains remaining: full negative penalty
            net_points -= ticket.points
        else:
            # Unfinished ticket: represents risk/obligation; penalty shrinks as we get closer to completion
            total_ticket_len = max(1, ticket.points)
            completed_fraction = max(0.0, 1.0 - (needed_trains / total_ticket_len))
            net_points += ticket.points * (1.8 * completed_fraction - 0.8)

    return net_points


def evaluate_leaf_state(
    game: Game,
    root_player_id: str,
    weights: tuple[float, float, float, float] = (0.60, 0.25, 0.10, 0.05),
) -> float:
    """Evaluate leaf state using a normalized continuous multi-feature score in [-1.0, 1.0]."""
    state = game.state
    root_p = next((p for p in state.players if p.id == root_player_id), state.players[0])
    opp_p = next((p for p in state.players if p.id != root_player_id), state.players[-1])

    if state.is_game_over:
        diff = root_p.score - opp_p.score
        if diff > 0:
            return 1.0
        elif diff < 0:
            return -1.0
        return 0.0

    w_score, w_tickets, w_routes, w_trains = weights

    # 1. Score Differential (Current Score + Estimated Ticket Points)
    root_ticket_est = _estimate_ticket_progress(root_p, game.board)
    opp_ticket_est = _estimate_ticket_progress(opp_p, game.board)

    root_total_est = root_p.score + root_ticket_est
    opp_total_est = opp_p.score + opp_ticket_est

    delta_score = math.tanh((root_total_est - opp_total_est) / 30.0)

    # 2. Direct Ticket Status Differential
    root_routes = [
        game.board.get_route(rid)
        for rid in root_p.claimed_route_ids
        if game.board.get_route(rid) is not None
    ]
    opp_routes = [
        game.board.get_route(rid)
        for rid in opp_p.claimed_route_ids
        if game.board.get_route(rid) is not None
    ]
    root_done = sum(1 for t in root_p.tickets if check_ticket_completed(root_routes, t))
    opp_done = sum(1 for t in opp_p.tickets if check_ticket_completed(opp_routes, t))
    total_tickets = max(1, len(root_p.tickets) + len(opp_p.tickets))
    delta_tickets = (root_done - opp_done) / total_tickets

    # 3. Route Length Differential
    root_len = sum(r.length for r in root_routes)
    opp_len = sum(r.length for r in opp_routes)
    delta_routes = (root_len - opp_len) / 45.0

    # 4. Longest Path Differential
    longest_root = compute_longest_continuous_path(root_routes)
    longest_opp = compute_longest_continuous_path(opp_routes)
    delta_longest = (longest_root - longest_opp) / 30.0

    val = (
        w_score * delta_score
        + w_tickets * delta_tickets
        + w_routes * delta_routes
        + w_trains * delta_longest
    )
    return max(-1.0, min(1.0, val))


def simulate_rollout(
    game: Game,
    root_player_id: str,
    max_depth: int,
    policy_type: RolloutPolicyType,
    rng: SeededRNG,
    weights: tuple[float, float, float, float] = (0.60, 0.25, 0.10, 0.05),
) -> float:
    """Execute a simulated rollout from the leaf state up to max_depth using chosen policy."""
    sim_game = game.clone(rng_seed=rng.randint(0, 1_000_000_000))
    greedy_agent = GreedyAgent() if policy_type == RolloutPolicyType.GREEDY else None
    strategic_agent = (
        StrategicHeuristicAgent() if policy_type == RolloutPolicyType.STRATEGIC else None
    )

    depth = 0
    while not sim_game.state.is_game_over and depth < max_depth:
        valid_actions = sim_game.valid_actions()
        if not valid_actions:
            break

        if policy_type == RolloutPolicyType.RANDOM:
            action = valid_actions[rng.randint(0, len(valid_actions) - 1)]
        elif policy_type == RolloutPolicyType.GREEDY and greedy_agent:
            action = greedy_agent.act(sim_game.state, valid_actions, sim_game.board)
        elif policy_type == RolloutPolicyType.STRATEGIC and strategic_agent:
            action = strategic_agent.act(sim_game.state, valid_actions, sim_game.board)
        else:
            action = valid_actions[0]

        sim_game.step(action)
        depth += 1

    return evaluate_leaf_state(sim_game, root_player_id=root_player_id, weights=weights)


class MCTSSearchEngine:
    """Monte Carlo Tree Search engine orchestrating selection, expansion, simulation, and backprop."""

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self.config = config or MCTSConfig()
        self.rng = SeededRNG(self.config.seed)

    def search(self, game: Game, root_player_id: str) -> Action:
        """Execute MCTS search from the current game state and return the most visited action."""
        valid_actions = game.valid_actions()
        if not valid_actions:
            raise ValueError("Cannot search from state with no valid actions.")
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Prioritize actions at root
        prioritized_valid = _prioritize_actions(
            valid_actions, game.state, game.board, root_player_id
        )

        root = MCTSNode(
            state=game.state,
            parent=None,
            action=None,
            player_id=root_player_id,
            untried_actions=list(prioritized_valid),
        )

        for _ in range(self.config.num_simulations):
            # 1. Determinization: Sample plausible world consistent with public observations
            if self.config.use_determinization:
                sim_game = determinize_game(game, root_player_id=root_player_id, rng=self.rng)
            else:
                sim_game = game.clone(rng_seed=self.rng.randint(0, 1_000_000_000))

            # 2. Selection: Descend tree via UCT
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                action, next_node = node.select_best_child(
                    self.config.c_puct, root_player_id=root_player_id
                )
                sim_game.step(action)
                node = next_node

            # 3. Expansion: If node is not terminal and has untried actions, expand one
            if not node.is_terminal() and not node.is_fully_expanded():
                untried_act = node.untried_actions.pop(0)
                sim_game.step(untried_act)
                next_player_id = (
                    sim_game.state.current_player.id
                    if sim_game.state.current_player
                    else root_player_id
                )
                child_valid = _prioritize_actions(
                    sim_game.valid_actions(),
                    sim_game.state,
                    sim_game.board,
                    next_player_id,
                )
                node = node.expand(
                    action=untried_act,
                    next_state=sim_game.state,
                    next_player_id=next_player_id,
                    untried_actions=child_valid,
                )

            # 4. Simulation (Rollout)
            reward = simulate_rollout(
                game=sim_game,
                root_player_id=root_player_id,
                max_depth=self.config.max_rollout_depth,
                policy_type=self.config.rollout_policy,
                rng=self.rng,
                weights=self.config.heuristic_weights,
            )

            # 5. Backpropagation: Update path from leaf back to root
            curr: MCTSNode | None = node
            while curr is not None:
                curr.update(reward)
                curr = curr.parent

        # Decision Rule: Robust Child (most visited action)
        if not root.children:
            return valid_actions[0]

        best_act = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_act
