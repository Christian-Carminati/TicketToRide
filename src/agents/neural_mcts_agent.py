"""
Neural MCTS and Opponent-Aware Bayesian MCTS Agent Implementations.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

from src.game.action import Action
from src.game.board import Board
from src.game.maps import load_usa_board
from src.game.state import GameState
from src.game.game import Game
from src.game.random import SeededRNG
from src.game.rules import GameRules
from src.agents.base_agent import BaseAgent
from src.environment.observation import ObservationV1
from src.environment.action_space import DiscreteActionSpace
from src.environment.action_mask import ActionMasker
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine
from src.rl.opponent_model import BayesianTicketBeliefTracker, belief_weighted_determinization

class NeuralMCTSAgent(BaseAgent):
    """
    AlphaZero-style search agent using a PolicyValueNetwork.
    """
    def __init__(
        self,
        net: Optional[PolicyValueNetwork] = None,
        num_simulations: int = 40,
        c_puct: float = 1.5,
        name: str = "NeuralMCTSAgent",
        seed: int = 42,
        board: Optional[Board] = None,
        tickets: Optional[list[Any]] = None,
    ):
        super().__init__(name=name)
        default_board, default_tickets = load_usa_board()
        self.board = board if board is not None else default_board
        self.tickets = tickets if tickets is not None else list(default_tickets)
        self.encoder = ObservationV1(board=self.board, initial_tickets=self.tickets)
        self.action_space = DiscreteActionSpace(board=self.board)
        self.masker = ActionMasker(self.action_space)
        self.seed = seed
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
        obs_dim = self.encoder.observation_shape[0]
        action_dim = self.action_space.n
        
        self.net = net or PolicyValueNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64,
            num_res_blocks=1,
        )
        self.engine = NeuralMCTSEngine(
            net=self.net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            seed=seed,
            board=self.board,
            tickets=self.tickets,
        )

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> int:
        """Gymnasium interface: select action using direct neural policy evaluation."""
        probs, _ = self.net.evaluate_state(observation, action_mask)
        return int(np.argmax(probs))

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Optional[Board] = None,
    ) -> Action:
        """Select domain Action by running Neural MCTS tree search."""
        if not valid_actions:
            raise ValueError("No valid actions available for NeuralMCTSAgent.")
        if len(valid_actions) == 1:
            return valid_actions[0]

        curr_p = state.current_player
        root_player_id = curr_p.id if curr_p else "player_0"

        # Reconstruct Game context for search engine
        active_board = board if board is not None else self.board
        active_tickets = self.tickets if self.tickets else list(load_usa_board()[1])
        if active_board != self.board:
            self.board = active_board
            self.encoder = ObservationV1(board=self.board, initial_tickets=self.tickets)
            self.action_space = DiscreteActionSpace(board=self.board)
            self.masker = ActionMasker(self.action_space)
        
        game = Game.__new__(Game)
        game.board = active_board
        game.initial_tickets = list(active_tickets)
        game.num_players = state.num_players
        game.rules = GameRules()
        game.rng = SeededRNG(self.seed)
        game.state = state.clone()

        best_action_idx, _, _ = self.engine.search(
            game, player_id=root_player_id, is_root_exploration=False
        )
        chosen_action = self.action_space.to_action(best_action_idx)
        
        # Verify action is strictly valid or fallback to most similar
        if chosen_action in valid_actions:
            return chosen_action
            
        for a in valid_actions:
            if a.action_type == chosen_action.action_type:
                return a
        return valid_actions[0]

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
            self.engine = NeuralMCTSEngine(
                net=self.net,
                num_simulations=self.num_simulations,
                c_puct=self.c_puct,
                seed=seed,
                board=self.board,
                tickets=self.tickets,
            )

class OpponentAwareMCTSAgent(NeuralMCTSAgent):
    """
    Neural MCTS agent enhanced with Bayesian Ticket Belief Tracking and Belief-Weighted Determinization.
    """
    def __init__(
        self,
        net: Optional[PolicyValueNetwork] = None,
        num_simulations: int = 40,
        c_puct: float = 1.5,
        name: str = "OpponentAwareMCTSAgent",
        seed: int = 42,
        board: Optional[Board] = None,
        tickets: Optional[list[Any]] = None,
    ):
        super().__init__(
            net=net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            name=name,
            seed=seed,
            board=board,
            tickets=tickets,
        )
        self.tracker: Optional[BayesianTicketBeliefTracker] = None
        self._observed_routes: set[str] = set()

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Optional[Board] = None,
    ) -> Action:
        if not valid_actions:
            raise ValueError("No valid actions available for OpponentAwareMCTSAgent.")
        if len(valid_actions) == 1:
            return valid_actions[0]

        active_board = board if board is not None else self.board
        active_tickets = self.tickets if self.tickets else list(load_usa_board()[1])
        if active_board != self.board:
            self.board = active_board
            self.encoder = ObservationV1(board=self.board, initial_tickets=self.tickets)
            self.action_space = DiscreteActionSpace(board=self.board)
            self.masker = ActionMasker(self.action_space)
        
        if self.tracker is None or self.tracker.board != active_board:
            self.tracker = BayesianTicketBeliefTracker(
                board=active_board,
                all_tickets=active_tickets,
            )
            self._observed_routes.clear()

        # Update tracker with newly claimed routes
        for r in active_board.routes:
            if r.claimed_by is not None and r.id not in self._observed_routes:
                self.tracker.observe_route_claim(r.claimed_by, r.id)
                self._observed_routes.add(r.id)

        curr_p = state.current_player
        root_player_id = curr_p.id if curr_p else "player_0"

        game = Game.__new__(Game)
        game.board = active_board
        game.initial_tickets = list(active_tickets)
        game.num_players = state.num_players
        game.rules = GameRules()
        game.rng = SeededRNG(self.seed)
        game.state = state.clone()

        def custom_det(g: Game, pid: str, rng: SeededRNG) -> Game:
            return belief_weighted_determinization(g, pid, self.tracker, rng)

        best_action_idx, _, _ = self.engine.search(
            game,
            player_id=root_player_id,
            is_root_exploration=False,
            custom_determinizer=custom_det,
        )
        chosen_action = self.action_space.to_action(best_action_idx)
        
        if chosen_action in valid_actions:
            return chosen_action
            
        for a in valid_actions:
            if a.action_type == chosen_action.action_type:
                return a
        return valid_actions[0]

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        self.tracker = None
        self._observed_routes.clear()


# Backward compatible / analytical naming alias
BayesianOpponentMCTSAgent = OpponentAwareMCTSAgent

