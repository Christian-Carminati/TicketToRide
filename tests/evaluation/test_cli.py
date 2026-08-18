"""Integration tests for CLI evaluation and tournament entrypoints."""

import subprocess
import sys


def test_evaluate_cli_execution():
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--agent1",
        "greedy",
        "--agent2",
        "random",
        "--games",
        "10",
        "--seed",
        "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "Head-to-Head Evaluation" in res.stdout
    assert "Greedy" in res.stdout
    assert "Random" in res.stdout


def test_tournament_cli_execution():
    cmd = [
        sys.executable,
        "scripts/tournament.py",
        "--agents",
        "random,greedy,strategic",
        "--games-per-pair",
        "10",
        "--seed",
        "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "Tournament Leaderboard" in res.stdout
    assert "Strategic" in res.stdout
    assert "Greedy" in res.stdout
