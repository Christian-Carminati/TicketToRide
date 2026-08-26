"""Integration tests for CLI evaluation, tournament, and thesis study entrypoints."""

import subprocess
import sys
from pathlib import Path


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
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
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
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "Tournament Leaderboard" in res.stdout
    assert "Strategic" in res.stdout
    assert "Greedy" in res.stdout


def test_run_thesis_study_cli(tmp_path: Path):
    cmd = [
        sys.executable,
        "scripts/run_thesis_study.py",
        "--fast-mode",
        "--output-dir",
        str(tmp_path),
        "--games-per-pair",
        "2",
        "--seeds",
        "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert (tmp_path / "thesis_study_results.json").exists()
    assert (tmp_path / "table1_main_results.tex").exists()
    assert (tmp_path / "table3_computational_profile.tex").exists()
