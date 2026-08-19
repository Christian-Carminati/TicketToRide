import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_tournament_available_participants():
    res = client.get("/api/tournament/participants/available")
    assert res.status_code == 200
    participants = res.json()
    assert isinstance(participants, list)
    assert len(participants) >= 3

    # Check baseline presence
    ids = [p["id"] for p in participants]
    assert "baseline_strategic" in ids
    assert "baseline_greedy" in ids
    assert "baseline_random" in ids


def test_tournament_leaderboard_endpoint():
    res = client.get("/api/tournament/leaderboard")
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
    assert "matchups" in data
    assert "total_games" in data
    assert "available_participants" in data
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


def test_tournament_custom_participants_run():
    # Run with only 2 specific baseline participants
    req = {
        "participant_ids": ["baseline_strategic", "baseline_greedy"],
        "games_per_pair": 2,
        "map_name": "mini",
        "seed": 777,
    }
    res = client.post("/api/tournament/run", json=req)
    assert res.status_code == 200
    data = res.json()
    assert len(data["leaderboard"]) == 2
    assert len(data["matchups"]) == 1
    assert data["total_games"] == 2
