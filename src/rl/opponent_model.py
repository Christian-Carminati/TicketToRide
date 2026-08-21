"""
Bayesian Opponent Modeling, Detour Distance Engine, and Belief-Weighted Determinization.

Computes:
P(Ticket_k | E_claimed) proportional to P(Ticket_k) * prod( L(e | Ticket_k) )
where L(e | Ticket_k) decays with graph detour routing cost.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

from src.game.board import Board
from src.game.route import Route
from src.game.ticket import DestinationTicket
from src.game.game import Game
from src.game.random import SeededRNG
from src.rl.mcts_determinization import determinize_game

class GraphDetourEngine:
    """
    Computes shortest path distances and detour routing penalties for tickets.
    """
    def __init__(self, board: Board, tickets: List[DestinationTicket]):
        self.board = board
        self.tickets = tickets
        self.graph = nx.Graph()
        
        # Build undirected graph
        for r in board.routes:
            self.graph.add_edge(r.city_a, r.city_b, weight=r.length, id=r.id)
            
        self.distances: Dict[str, Dict[str, float]] = dict(nx.all_pairs_dijkstra_path_length(self.graph))

    def get_dist(self, c1: str, c2: str) -> float:
        if c1 == c2:
            return 0.0
        return self.distances.get(c1, {}).get(c2, 999.0)

    def compute_detour(self, route: Route, ticket: DestinationTicket) -> float:
        """
        Computes detour cost of routing through edge `route` between ticket endpoints.
        Detour = min(d(u, a) + len(e) + d(b, v), d(u, b) + len(e) + d(a, v)) - d(u, v)
        """
        u, v = ticket.city_a, ticket.city_b
        a, b = route.city_a, route.city_b
        w = route.length
        base_d = self.get_dist(u, v)
        
        d1 = self.get_dist(u, a) + w + self.get_dist(b, v)
        d2 = self.get_dist(u, b) + w + self.get_dist(a, v)
        detour = min(d1, d2) - base_d
        return max(0.0, detour)

class BayesianTicketBeliefTracker:
    """
    Maintains Bayesian posterior probability distribution over opponent destination tickets.
    """
    def __init__(
        self,
        board: Board,
        all_tickets: List[DestinationTicket],
        beta: float = 0.5,
        gamma: float = 0.85,
        noise_floor: float = 0.05,
    ):
        self.board = board
        self.all_tickets = all_tickets
        self.beta = beta
        self.gamma = gamma
        self.noise_floor = noise_floor
        
        self.detour_engine = GraphDetourEngine(board, all_tickets)
        self.player_beliefs: Dict[str, Dict[DestinationTicket, float]] = {}

    def _init_player(self, player_id: str):
        if player_id not in self.player_beliefs:
            uniform_p = 1.0 / max(1, len(self.all_tickets))
            self.player_beliefs[player_id] = {t: uniform_p for t in self.all_tickets}

    def observe_route_claim(self, player_id: str, route_id: str):
        """
        Updates Bayesian posterior given newly claimed route.
        """
        self._init_player(player_id)
        route = self.board.get_route(route_id)
        if route is None:
            return
            
        current_beliefs = self.player_beliefs[player_id]
        unnormalized = {}
        for t, prior in current_beliefs.items():
            detour = self.detour_engine.compute_detour(route, t)
            # Likelihood: higher when detour == 0
            likelihood = self.gamma * math.exp(-self.beta * detour) + (1.0 - self.gamma) * self.noise_floor
            unnormalized[t] = prior * likelihood
            
        tot = sum(unnormalized.values())
        if tot > 0:
            self.player_beliefs[player_id] = {t: val / tot for t, val in unnormalized.items()}
        else:
            self._init_player(player_id)

    def get_ticket_probabilities(self, player_id: str) -> Dict[DestinationTicket, float]:
        self._init_player(player_id)
        return dict(self.player_beliefs[player_id])

    def get_top_k_tickets(self, player_id: str, k: int = 3) -> List[Tuple[DestinationTicket, float]]:
        beliefs = self.get_ticket_probabilities(player_id)
        sorted_tickets = sorted(beliefs.items(), key=lambda item: item[1], reverse=True)
        return sorted_tickets[:k]

def belief_weighted_determinization(
    game: Game,
    root_player_id: str,
    tracker: BayesianTicketBeliefTracker,
    rng: Optional[SeededRNG] = None,
) -> Game:
    """
    Determinizes hidden opponent state weighted by Bayesian ticket probabilities.
    """
    active_rng = rng or SeededRNG(seed=np.random.randint(0, 1_000_000_000))
    # First apply standard card determinization
    cloned_game = determinize_game(game, root_player_id=root_player_id, rng=active_rng)
    
    # For opponents, re-sample their destination tickets proportional to belief probabilities
    for p in cloned_game.state.players:
        if p.id != root_player_id:
            beliefs = tracker.get_ticket_probabilities(p.id)
            tickets = list(beliefs.keys())
            probs = np.array([beliefs[t] for t in tickets], dtype=np.float64)
            sum_probs = probs.sum()
            if sum_probs > 0:
                probs /= sum_probs
            else:
                probs = np.ones(len(tickets)) / len(tickets)
            
            num_tickets_to_sample = len(p.tickets)
            if len(tickets) >= num_tickets_to_sample and num_tickets_to_sample > 0:
                sampled_indices = np.random.choice(
                    len(tickets), size=num_tickets_to_sample, replace=False, p=probs
                )
                p.tickets = [tickets[idx] for idx in sampled_indices]
                
    return cloned_game

class TacticalBlocker:
    """
    Identifies high-value bottleneck routes to intercept opponent tickets.
    """
    def __init__(self, board: Board, tracker: BayesianTicketBeliefTracker):
        self.board = board
        self.tracker = tracker
        self.detour_engine = GraphDetourEngine(board, tracker.all_tickets)

    def compute_unclaimed_route_threats(self, opponent_id: str) -> Dict[str, float]:
        """
        Computes threat score Threat(e) = sum_k P(T_k) * Points(T_k) * I(e on path).
        """
        beliefs = self.tracker.get_ticket_probabilities(opponent_id)
        threats: Dict[str, float] = {}
        
        for r in self.board.routes:
            if r.claimed_by is not None:
                continue
            threat_score = 0.0
            for t, prob in beliefs.items():
                if self.detour_engine.compute_detour(r, t) == 0.0:
                    threat_score += prob * t.points
            threats[r.id] = threat_score
            
        return threats
