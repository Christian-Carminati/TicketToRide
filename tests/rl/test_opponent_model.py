import pytest
import numpy as np
from src.game.game import Game
from src.game.board import Board
from src.rl.opponent_model import (
    GraphDetourEngine,
    BayesianTicketBeliefTracker,
    belief_weighted_determinization,
    TacticalBlocker,
)

def test_graph_detour_engine():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    board = game.board
    tickets = game.initial_tickets
    engine = GraphDetourEngine(board=board, tickets=tickets)
    
    t0 = tickets[0]
    # Find route with city_a or city_b touching ticket endpoints
    routes_on_path = [r for r in board.routes if (r.city_a == t0.city_a or r.city_b == t0.city_b)]
    assert len(routes_on_path) > 0
    r0 = routes_on_path[0]
    
    detour = engine.compute_detour(r0, t0)
    assert detour >= 0.0

def test_bayesian_ticket_belief_tracker_learning():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    board = game.board
    all_tickets = game.initial_tickets
    tracker = BayesianTicketBeliefTracker(board=board, all_tickets=all_tickets)
    
    # Opponent (player_1) true ticket
    p1 = game.state.players[1]
    player_tickets = p1.tickets if p1.tickets else p1.pending_tickets
    assert len(player_tickets) > 0
    true_ticket = player_tickets[0]
    
    # Initial probabilities should be uniform
    p_init = tracker.get_ticket_probabilities("player_1")
    assert pytest.approx(sum(p_init.values()), abs=1e-5) == 1.0
    
    # Find routes that have 0 detour for true_ticket
    engine = GraphDetourEngine(board=board, tickets=all_tickets)
    candidate_routes = [r for r in board.routes if engine.compute_detour(r, true_ticket) == 0.0]
    
    # Simulate player_1 claiming these routes
    for r in candidate_routes[:2]:
        tracker.observe_route_claim("player_1", r.id)
        
    p_updated = tracker.get_ticket_probabilities("player_1")
    assert p_updated[true_ticket] > p_init[true_ticket]
    
    # Top-K recall
    top_3 = tracker.get_top_k_tickets("player_1", k=3)
    assert len(top_3) == min(3, len(all_tickets))
    assert sum(prob for _, prob in top_3) > 0.0

def test_belief_weighted_determinization_no_leak():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    
    tracker = BayesianTicketBeliefTracker(board=game.board, all_tickets=game.initial_tickets)
    
    # Determinize for player_0
    cloned_game = belief_weighted_determinization(game, root_player_id="player_0", tracker=tracker)
    
    assert len(cloned_game.state.players) == 2
    assert cloned_game.state.players[0].tickets == game.state.players[0].tickets
    # player_1 tickets sampled consistently
    assert len(cloned_game.state.players[1].tickets) == len(game.state.players[1].tickets)

def test_tactical_blocker_evaluates_threats():
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    tracker = BayesianTicketBeliefTracker(board=game.board, all_tickets=game.initial_tickets)
    blocker = TacticalBlocker(board=game.board, tracker=tracker)
    
    threats = blocker.compute_unclaimed_route_threats("player_1")
    assert isinstance(threats, dict)
    assert len(threats) > 0
    for r_id, threat_val in threats.items():
        assert threat_val >= 0.0
