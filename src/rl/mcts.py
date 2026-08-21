"""Monte Carlo Tree Search (MCTS) core implementation.

Includes MCTSNode, MCTSConfig, UCT action selection, heuristic rollout policies,
and continuous leaf evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from src.agents.greedy_agent import GreedyAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.game.action import Action
from src.game.game import Game
from src.game.graph import check_ticket_completed
from src.game.random import SeededRNG
from src.game.state import GameState


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
    heuristic_weights: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.05)


class MCTSNode:
    """Represents a single state/decision node in the Monte Carlo search tree.

    Mathematical Representation:
        - N(s): Visit count of node s (`visits`).
        - W(s): Cumulative backpropagated return (`total_value`).
        - Q(s) = W(s) / N(s): Empirical expected value estimate (`value`).
        - UCT(s, a) = Q(s, a) + c_puct * sqrt(ln(N(s)) / (N(s, a) + epsilon)).
    """

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
        """Expected Q-value estimate Q(s) = W / N."""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        """Check if all legal actions from this node have been branched into children."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this node represents a game over state."""
        return self.state.is_game_over

    def select_best_child(self, c_puct: float = 1.41421356) -> tuple[Action, "MCTSNode"]:
        """Select child maximizing the Upper Confidence Bound applied to Trees (UCT).

        Formula:
            a* = argmax_{a} [ Q(s, a) + c_puct * sqrt(ln(N(s)) / (N(s, a) + 1e-6)) ]
        """
        best_uct = -float("inf")
        best_pair: tuple[Action, MCTSNode] | None = None
        log_n = math.log(max(1, self.visits))

        for action, child in self.children.items():
            # Q-value from perspective of the node's acting player
            q_val = child.value
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
        """Backpropagate simulation reward, updating visit count and cumulative value."""
        self.visits += 1
        self.total_value += reward


def evaluate_leaf_state(
    game: Game,
    root_player_id: str,
    weights: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.05),
) -> float:
    """Evaluate leaf state using a normalized continuous multi-feature score in [-1.0, 1.0].

    Features:
        1. Delta Score: tanh((score_root - score_opp) / 30.0)
        2. Delta Tickets: (tickets_done_root - tickets_done_opp) / total_tickets
        3. Delta Route Length: (route_len_root - route_len_opp) / 45.0
        4. Delta Trains Remaining: (trains_root - trains_opp) / 45.0
    """
    state = game.state
    if state.is_game_over:
        root_p = next((p for p in state.players if p.id == root_player_id), state.players[0])
        opp_p = next((p for p in state.players if p.id != root_player_id), state.players[-1])
        if root_p.score > opp_p.score:
            return 1.0
        elif root_p.score < opp_p.score:
            return -1.0
        return 0.0

    w_score, w_tickets, w_routes, w_trains = weights
    root_p = next((p for p in state.players if p.id == root_player_id), state.players[0])
    opp_p = next((p for p in state.players if p.id != root_player_id), state.players[-1])

    # 1. Score Differential
    delta_score = math.tanh((root_p.score - opp_p.score) / 30.0)

    # 2. Ticket Completion Differential
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

    root_tickets_done = sum(1 for t in root_p.tickets if check_ticket_completed(root_routes, t))
    opp_tickets_done = sum(1 for t in opp_p.tickets if check_ticket_completed(opp_routes, t))
    total_tickets = max(1, len(root_p.tickets) + len(opp_p.tickets))
    delta_tickets = (root_tickets_done - opp_tickets_done) / total_tickets

    # 3. Route Length Differential
    root_len = sum(r.length for r in root_routes)
    opp_len = sum(r.length for r in opp_routes)
    delta_routes = (root_len - opp_len) / 45.0

    # 4. Trains Remaining Differential (preserving trains gives endgame flexibility)
    delta_trains = (root_p.trains_remaining - opp_p.trains_remaining) / 45.0

    val = (
        w_score * delta_score
        + w_tickets * delta_tickets
        + w_routes * delta_routes
        + w_trains * delta_trains
    )
    return max(-1.0, min(1.0, val))


def simulate_rollout(
    game: Game,
    root_player_id: str,
    max_depth: int,
    policy_type: RolloutPolicyType,
    rng: SeededRNG,
    weights: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.05),
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
