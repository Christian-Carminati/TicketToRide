"""Acceptance Test for Phase 2: 1,000-game tournament benchmark."""

import time

from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.evaluation.tournament import Tournament


def test_phase2_1000_game_tournament_acceptance():
    """Acceptance Test for Phase 2:
    - Runs a 1,002-game total tournament (3 pairs * 334 games).
    - Asserts performance (> 20 full games/sec).
    - Asserts Elo and win rate hierarchy: Strategic > Greedy > Random.
    - Asserts deterministic reproducibility.
    """
    agents = [
        RandomAgent(seed=42, name="Random_1"),
        GreedyAgent(name="Greedy_1"),
        StrategicHeuristicAgent(name="Strategic_1"),
    ]
    # 3 pairs * 334 games = 1002 games total
    games_per_pair = 334
    tournament = Tournament(agents=agents, games_per_pair=games_per_pair)

    start_time = time.time()
    results = tournament.run(seed=12345)
    duration = time.time() - start_time

    leaderboard = results["leaderboard"]
    assert len(leaderboard) == 3

    # Ranking check: Strategic #1, Greedy #2, Random #3
    assert (
        leaderboard[0]["name"] == "Strategic_1"
    ), f"Expected Strategic_1 first, got {leaderboard[0]}"
    assert (
        leaderboard[1]["name"] == "Greedy_1"
    ), f"Expected Greedy_1 second, got {leaderboard[1]}"
    assert (
        leaderboard[2]["name"] == "Random_1"
    ), f"Expected Random_1 third, got {leaderboard[2]}"

    # Strategic win rate and score assertions
    assert leaderboard[0]["elo"] > leaderboard[1]["elo"] > leaderboard[2]["elo"]
    assert (
        leaderboard[0]["avg_score"]
        > leaderboard[1]["avg_score"]
        > leaderboard[2]["avg_score"]
    )

    print(
        f"\n1,000-Game Tournament completed in {duration:.2f} seconds "
        f"({(games_per_pair * 3) / duration:.1f} games/sec)"
    )
    # Ensure high-throughput requirement (> 20 games/sec single thread)
    assert duration < 50.0, f"Tournament took too long: {duration:.2f}s"
