"""Monte Carlo Tree Search (MCTS) Agent implementing BaseAgent interface."""

from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.game.action import Action
from src.game.board import Board
from src.game.game import Game
from src.game.random import SeededRNG
from src.game.rules import GameRules
from src.game.state import GameState
from src.rl.mcts import MCTSConfig, MCTSSearchEngine


class MCTSAgent(BaseAgent):
    """MCTS Agent capable of planning via determinized tree search simulations."""

    def __init__(
        self,
        config: MCTSConfig | None = None,
        name: str = "MCTSAgent",
        num_simulations: int | None = None,
        seed: int = 42,
        board: Board | None = None,
        tickets: list[Any] | None = None,
    ) -> None:
        super().__init__(name=name)
        if config is not None:
            self.config = config
        else:
            sim_count = num_simulations if num_simulations is not None else 100
            self.config = MCTSConfig(num_simulations=sim_count, seed=seed)
        self.board = board
        self.tickets = tickets
        self.engine = MCTSSearchEngine(self.config)

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Board | None = None,
    ) -> Action:
        """Select domain action by running MCTS simulations from the current state."""
        if not valid_actions:
            raise ValueError("No valid actions available for MCTSAgent.")
        if len(valid_actions) == 1:
            return valid_actions[0]

        curr_p = state.current_player
        root_player_id = curr_p.id if curr_p else "player_0"

        # Reconstruct Game context for search engine
        active_board = (
            board if board is not None else (self.board if self.board is not None else Board())
        )
        active_tickets = self.tickets if self.tickets is not None else []
        game = Game.__new__(Game)
        game.board = active_board
        game.initial_tickets = list(active_tickets)
        game.num_players = state.num_players
        game.rules = GameRules()
        game.rng = SeededRNG(self.config.seed)
        game.state = state.clone()

        try:
            action = self.engine.search(game, root_player_id=root_player_id)
            if action in valid_actions:
                return action
            for a in valid_actions:
                if a.action_type == action.action_type:
                    return a
            return valid_actions[0]
        except Exception:
            return valid_actions[0]

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool = True,
        info: dict[str, Any] | None = None,
    ) -> int:
        """Select action index for Gymnasium environments."""
        if action_mask is not None:
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return int(valid_indices[0])
        return 0

    def reset(self, seed: int | None = None) -> None:
        """Reset internal RNG seed."""
        if seed is not None:
            self.config.seed = seed
            self.engine = MCTSSearchEngine(self.config)
