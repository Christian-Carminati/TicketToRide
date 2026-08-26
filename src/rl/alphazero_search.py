"""
Neural Monte Carlo Tree Search (AlphaZero Search Engine).

Uses Polynomial Upper Confidence Trees (PUCT) guided by PolicyValueNetwork
with Dirichlet exploration noise at the root, completely eliminating random rollouts.
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from src.game.game import Game
from src.game.random import SeededRNG
from src.environment.observation import ObservationV1
from src.environment.action_space import DiscreteActionSpace
from src.environment.action_mask import ActionMasker
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.mcts_determinization import determinize_game

class NeuralMCTSNode:
    """Node in the Neural MCTS search tree."""
    def __init__(
        self,
        state_player: str,
        parent: Optional[NeuralMCTSNode] = None,
        action_from_parent: Optional[int] = None,
    ):
        self.state_player = state_player
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[int, NeuralMCTSNode] = {}
        
        self.legal_actions: List[int] = []
        self.priors: Dict[int, float] = {}
        self.visits: Dict[int, int] = {}
        self.values: Dict[int, float] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0

    @property
    def total_visits(self) -> int:
        return sum(self.visits.values())

    def get_q_value(self, action: int) -> float:
        n = self.visits.get(action, 0)
        return self.values.get(action, 0.0) / n if n > 0 else 0.0

    def select_puct_action(self, c_puct: float = 1.5) -> int:
        """Select action maximizing Q(s, a) + U(s, a)."""
        tot_v = self.total_visits
        sqrt_tot = math.sqrt(tot_v) if tot_v > 0 else 1.0
        
        best_score = -float("inf")
        best_action = self.legal_actions[0] if self.legal_actions else 0
        
        for a in self.legal_actions:
            q = self.get_q_value(a)
            p = self.priors.get(a, 0.0)
            n = self.visits.get(a, 0)
            u = c_puct * p * (sqrt_tot / (1.0 + n))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

class NeuralMCTSEngine:
    """
    AlphaZero-style Search Engine using Neural Policy Priors and State Values.
    """
    def __init__(
        self,
        net: PolicyValueNetwork,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        seed: int = 42,
        board: Any = None,
        tickets: Any = None,
    ):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = SeededRNG(seed=seed)
        
        self.board = board
        self.tickets = tickets
        self.encoder = ObservationV1(board=board, initial_tickets=tickets) if board is not None else ObservationV1()
        self.action_space = DiscreteActionSpace(board=board) if board is not None else DiscreteActionSpace()
        self.masker = ActionMasker(self.action_space)

    def _get_obs_and_mask(self, game: Game, player_id: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        player_idx = int(player_id.split("_")[-1]) if "_" in player_id else 0
        if game.board != self.board and game.board is not None:
            self.board = game.board
            self.tickets = game.initial_tickets if hasattr(game, "initial_tickets") and game.initial_tickets else None
            self.encoder = ObservationV1(board=self.board, initial_tickets=self.tickets)
            self.action_space = DiscreteActionSpace(board=self.board)
            self.masker = ActionMasker(self.action_space)

        obs = self.encoder.encode(game.state, player_idx)
        player = next((p for p in game.state.players if p.id == player_id), game.state.players[0])
        valid_actions = game.rules.get_valid_actions(player, game.state, game.board, game.num_players)
        mask = self.masker.compute_mask(valid_actions, player.pending_tickets)
        legal_actions = [i for i, v in enumerate(mask) if v]
        if not legal_actions:
            legal_actions = [0]
            mask[0] = True
        return obs, mask.astype(np.float32), legal_actions

    def search(
        self,
        game: Game,
        player_id: str,
        is_root_exploration: bool = True,
        custom_determinizer: Optional[Callable[[Game, str, SeededRNG], Game]] = None,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Runs MCTS search from root game state.
        Returns: (best_action, target_policy_probs, estimated_root_value)
        """
        curr_p = game.state.current_player
        curr_player_id = curr_p.id if curr_p is not None else player_id
        root = NeuralMCTSNode(state_player=curr_player_id)
        
        # Initial expansion of root
        obs, mask, legal_actions = self._get_obs_and_mask(game, curr_player_id)
        priors, root_v = self.net.evaluate_state(obs, mask)
        
        # Apply Dirichlet noise at root for self-play exploration
        if is_root_exploration and len(legal_actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for idx, a in enumerate(legal_actions):
                root.priors[a] = float((1.0 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * noise[idx])
        else:
            for a in legal_actions:
                root.priors[a] = float(priors[a])
                
        root.legal_actions = legal_actions
        for a in legal_actions:
            root.visits[a] = 0
            root.values[a] = 0.0
        root.is_expanded = True
        
        # Transposition table cache for evaluation (s_hash -> (priors, val))
        tt_cache: Dict[int, Tuple[np.ndarray, float]] = {}

        # Iterative simulations
        for _ in range(self.num_simulations):
            if custom_determinizer is not None:
                sim_game = custom_determinizer(game, player_id, self.rng)
            else:
                sim_game = determinize_game(game, player_id, self.rng)
                
            node = root
            search_path: List[Tuple[NeuralMCTSNode, int]] = []
            
            # 1. Selection
            while node.is_expanded and not sim_game.state.is_game_over:
                action_idx = node.select_puct_action(self.c_puct)
                search_path.append((node, action_idx))
                game_action = self.action_space.to_action(action_idx)
                sim_game.step(game_action)
                
                if action_idx not in node.children:
                    next_p = sim_game.state.current_player
                    next_player_id = next_p.id if next_p is not None else curr_player_id
                    node.children[action_idx] = NeuralMCTSNode(
                        state_player=next_player_id, parent=node, action_from_parent=action_idx
                    )
                node = node.children[action_idx]
                
            # 2. Evaluation & Expansion
            if sim_game.state.is_game_over:
                scores = {p.id: p.score for p in sim_game.state.players}
                p0_score = scores.get(player_id, 0)
                opp_ids = [pid for pid in scores if pid != player_id]
                opp_score = scores.get(opp_ids[0], 0) if opp_ids else 0
                if p0_score > opp_score:
                    leaf_val = 1.0
                elif p0_score < opp_score:
                    leaf_val = -1.0
                else:
                    leaf_val = 0.0
            else:
                s_p = sim_game.state.current_player
                s_player_id = s_p.id if s_p is not None else player_id
                obs_leaf, mask_leaf, legal_leaf = self._get_obs_and_mask(sim_game, s_player_id)

                # Compute fast 64-bit state hash for transposition caching
                s_hash = hash(obs_leaf.tobytes())
                if s_hash in tt_cache:
                    priors_leaf, val_leaf = tt_cache[s_hash]
                else:
                    priors_leaf, val_leaf = self.net.evaluate_state(obs_leaf, mask_leaf)
                    tt_cache[s_hash] = (priors_leaf, val_leaf)
                
                node.legal_actions = legal_leaf
                for a in legal_leaf:
                    node.priors[a] = float(priors_leaf[a])
                    node.visits[a] = 0
                    node.values[a] = 0.0
                node.is_expanded = True
                
                # Convert leaf value to root player's perspective
                leaf_val = val_leaf if s_player_id == player_id else -val_leaf
                
            # 3. Backpropagation
            for parent_node, action_taken in reversed(search_path):
                val_for_parent = leaf_val if parent_node.state_player == player_id else -leaf_val
                parent_node.visits[action_taken] += 1
                parent_node.values[action_taken] += val_for_parent
                
        # Construct target visit distribution pi
        pi = np.zeros(self.net.action_dim, dtype=np.float32)
        for a in root.legal_actions:
            pi[a] = root.visits[a]
            
        tot_visits = pi.sum()
        if tot_visits > 0:
            pi /= tot_visits
        else:
            for a in root.legal_actions:
                pi[a] = 1.0 / len(root.legal_actions)
                
        best_action = int(np.argmax(pi))
        root_val = sum(root.values.values()) / max(1, root.total_visits)
        return best_action, pi, root_val
