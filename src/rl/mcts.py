"""Monte Carlo Tree Search (MCTS) core implementation.

Includes MCTSNode, MCTSConfig, UCT action selection, heuristic rollout policies,
and continuous leaf evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from src.game.action import Action
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
