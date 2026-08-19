import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_tournament_leaderboard_endpoint():
    res = client.get("/api/tournament/leaderboard")
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
    assert "matchups" in data
    assert "total_games" in data
    assert len(data["leaderboard"]) >= 3
    assert len(data["matchups"]) >= 1

    # Verify agent fields
    first_agent = data["leaderboard"][0]
    assert "name" in first_agent
    assert "elo" in first_agent
    assert "win_rate" in first_agent
    assert "wins" in first_agent
    assert "losses" in first_agent
    assert "avg_score" in first_agent


def test_tournament_run_endpoint():
    res = client.post("/api/tournament/run", json={"games_per_pair": 2, "seed": 999})
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
    assert len(data["leaderboard"]) >= 3
    assert data["total_games"] > 0
