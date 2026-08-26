"""
AlphaZero Self-Play Data Generation & Joint Optimization Trainer.

Optimizes joint loss:
L(theta) = MSE(z, v_theta(s)) - pi^T * log(p_theta(s)) + c_reg * ||theta||_2^2
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.game.game import Game
from src.environment.observation import ObservationV1
from src.environment.action_space import DiscreteActionSpace
from src.environment.action_mask import ActionMasker
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine

class SelfPlayReplayBuffer:
    """Experience replay storing (obs, action_mask, pi_mcts, outcome_z) with O(1) circular indexing."""
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.observations: List[np.ndarray] = []
        self.action_masks: List[np.ndarray] = []
        self.target_policies: List[np.ndarray] = []
        self.target_values: List[float] = []
        self.ptr: int = 0

    def __len__(self) -> int:
        return len(self.observations)

    def add(self, obs: np.ndarray, mask: np.ndarray, pi: np.ndarray, z: float):
        if len(self.observations) < self.capacity:
            self.observations.append(obs)
            self.action_masks.append(mask)
            self.target_policies.append(pi)
            self.target_values.append(z)
        else:
            self.observations[self.ptr] = obs
            self.action_masks[self.ptr] = mask
            self.target_policies[self.ptr] = pi
            self.target_values[self.ptr] = z
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self.observations)
        b_size = min(batch_size, n)
        indices = random.sample(range(n), b_size)
        obs_b = torch.as_tensor(np.array([self.observations[i] for i in indices]), dtype=torch.float32)
        mask_b = torch.as_tensor(np.array([self.action_masks[i] for i in indices]), dtype=torch.float32)
        pi_b = torch.as_tensor(np.array([self.target_policies[i] for i in indices]), dtype=torch.float32)
        z_b = torch.as_tensor(np.array([self.target_values[i] for i in indices]), dtype=torch.float32).unsqueeze(1)
        return obs_b, mask_b, pi_b, z_b

class AlphaZeroTrainer:
    """
    Self-Play Collector & AlphaZero Optimization Pipeline.
    """
    def __init__(
        self,
        net: PolicyValueNetwork,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_simulations: int = 30,
        c_puct: float = 1.5,
        batch_size: int = 32,
        buffer_capacity: int = 50000,
        device: str = "cpu",
    ):
        self.net = net.to(device)
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.buffer = SelfPlayReplayBuffer(capacity=buffer_capacity)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.search_engine = NeuralMCTSEngine(net=self.net, num_simulations=num_simulations, c_puct=c_puct)
        self.encoder = ObservationV1()
        self.action_space = DiscreteActionSpace()
        self.masker = ActionMasker(self.action_space)

    def _get_obs_and_mask(self, game: Game, player_id: str) -> Tuple[np.ndarray, np.ndarray]:
        player_idx = int(player_id.split("_")[-1]) if "_" in player_id else 0
        obs = self.encoder.encode(game.state, player_idx)
        player = next((p for p in game.state.players if p.id == player_id), game.state.players[0])
        valid_actions = game.rules.get_valid_actions(player, game.state, game.board, game.num_players)
        mask = self.masker.compute_mask(valid_actions, player.pending_tickets)
        return obs, mask.astype(np.float32)

    def collect_self_play_games(self, num_games: int = 1, max_turns: int = 150) -> int:
        """
        Plays self-play games where Neural MCTS controls both players.
        Stores trajectories in replay buffer.
        """
        total_steps = 0
        for _ in range(num_games):
            game = Game(num_players=2)
            game.reset()
            
            history: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            turns = 0
            
            while not game.state.is_game_over and turns < max_turns:
                curr_p = game.state.current_player
                curr_player_id = curr_p.id if curr_p is not None else "player_0"
                obs, mask = self._get_obs_and_mask(game, curr_player_id)
                
                best_action, pi_target, _ = self.search_engine.search(
                    game, curr_player_id, is_root_exploration=True
                )
                
                history.append((curr_player_id, obs, mask, pi_target))
                game_action = self.action_space.to_action(best_action)
                game.step(game_action)
                turns += 1
                
            # Assign game outcomes
            scores = {p.id: p.score for p in game.state.players}
            p0_score = scores.get("player_0", 0)
            p1_score = scores.get("player_1", 0)
            
            for player_step, obs_step, mask_step, pi_step in history:
                if player_step == "player_0":
                    z = 1.0 if p0_score > p1_score else (-1.0 if p0_score < p1_score else 0.0)
                else:
                    z = 1.0 if p1_score > p0_score else (-1.0 if p1_score < p0_score else 0.0)
                self.buffer.add(obs_step, mask_step, pi_step, z)
                total_steps += 1
                
        return total_steps

    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Performs one gradient descent step on sampled batch from self-play buffer.
        """
        b_size = batch_size or self.batch_size
        if len(self.buffer) < b_size:
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
            
        self.net.train()
        obs_b, mask_b, pi_b, z_b = self.buffer.sample(b_size)
        obs_b = obs_b.to(self.device)
        mask_b = mask_b.to(self.device)
        pi_b = pi_b.to(self.device)
        z_b = z_b.to(self.device)
        
        pred_probs, pred_vals = self.net(obs_b, mask_b)
        
        # Value MSE loss
        value_loss = F.mse_loss(pred_vals, z_b)
        
        # Policy Cross Entropy: - sum(pi * log(p + 1e-8))
        log_probs = torch.log(pred_probs + 1e-8)
        policy_loss = -torch.mean(torch.sum(pi_b * log_probs, dim=-1))
        
        total_loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "loss": float(total_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        self.net.save(path)
