"""Unit tests for Tournament round-robin orchestrator."""

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.tournament import Tournament


def test_round_robin_tournament_execution():
    agents = [
        RandomAgent(seed=1, name="Random_1"),
        GreedyAgent(name="Greedy_1"),
        StrategicHeuristicAgent(name="Strategic_1"),
    ]
    tournament = Tournament(agents=agents, games_per_pair=10)
    results = tournament.run(seed=42)

    assert len(results["leaderboard"]) == 3
    # Strategic should have higher Elo than Random
    elo_strategic = results["ratings"]["Strategic_1"]
    elo_random = results["ratings"]["Random_1"]
    assert elo_strategic > elo_random


def test_tournament_deterministic_replay():
    agents1 = [RandomAgent(seed=1, name="R1"), GreedyAgent(name="G1")]
    agents2 = [RandomAgent(seed=1, name="R1"), GreedyAgent(name="G1")]

    t1 = Tournament(agents=agents1, games_per_pair=6)
    r1 = t1.run(seed=999)

    t2 = Tournament(agents=agents2, games_per_pair=6)
    r2 = t2.run(seed=999)

    assert r1["ratings"] == r2["ratings"]
    assert r1["leaderboard"] == r2["leaderboard"]
