"""Unit tests for BehavioralEvaluator and BehavioralProfile."""

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.behavioral import BehavioralEvaluator, BehavioralProfile
from src.game.maps import create_synthetic_mini_board


def test_behavioral_evaluator_metrics_structure():
    board, tickets = create_synthetic_mini_board()
    evaluator = BehavioralEvaluator(board=board, tickets_deck=tickets, seed=42)
    agent = GreedyAgent(name="GreedyTest")
    opp = RandomAgent(name="RandomTest")

    profile = evaluator.profile_agent(agent=agent, opponent=opp, num_games=6)

    assert isinstance(profile, BehavioralProfile)
    assert 0.0 <= profile.win_rate <= 1.0
    assert profile.avg_routes_claimed >= 0.0
    assert profile.avg_route_length >= 0.0
    assert profile.route_efficiency >= 0.0
    assert 0.0 <= profile.ticket_completion_rate <= 1.0
    assert profile.avg_game_turns > 0.0
    assert 0.0 <= profile.cards_drawn_ratio <= 1.0

    d = profile.to_dict()
    assert "win_rate" in d
    assert "route_efficiency" in d
    assert "ticket_completion_rate" in d
    assert "avg_route_length" in d
    assert "cards_drawn_ratio" in d


def test_behavioral_evaluator_multi_opponent():
    board, tickets = create_synthetic_mini_board()
    evaluator = BehavioralEvaluator(board=board, tickets_deck=tickets, seed=100)
    agent = GreedyAgent(name="GreedyMulti")

    profiles = evaluator.profile_multi_opponent(
        agent=agent,
        opponents=["random", "greedy"],
        games_per_opponent=4,
    )

    assert "random" in profiles
    assert "greedy" in profiles
    assert isinstance(profiles["random"], BehavioralProfile)
    assert isinstance(profiles["greedy"], BehavioralProfile)
