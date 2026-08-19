"""End-to-end Phase 5 Acceptance Test Suite for Ticket to Ride RL Web Lab."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_phase5_e2e_game_and_introspection():
    # 1. Start a 2-player bot match (Mini map)
    res = client.post(
        "/api/game/new",
        json={"map_name": "mini", "player_types": ["ppo", "greedy"], "seed": 123},
    )
    assert res.status_code == 200
    game = res.json()
    session_id = game["session_id"]
    assert len(game["players"]) == 2
    assert game["map_name"] == "mini"

    # 2. Inspect the PPO agent brain on this live session
    res_brain = client.post("/api/brain/inspect", json={"session_id": session_id})
    assert res_brain.status_code == 200
    brain_dto = res_brain.json()
    assert brain_dto["model_type"] in ["ppo", "dqn"]
    assert len(brain_dto["action_probabilities"]) == len(brain_dto["action_mask"])
    assert len(brain_dto["layer_activations"]) >= 1

    # 3. Step the game multiple times
    for _ in range(5):
        step_res = client.post("/api/game/step", json={"session_id": session_id})
        assert step_res.status_code == 200
        step_data = step_res.json()
        if step_data["is_game_over"]:
            break

    # 4. Clean up session
    del_res = client.delete(f"/api/game/{session_id}")
    assert del_res.status_code == 200


def test_phase5_e2e_training_telemetry_flow():
    # Start training with minimal timesteps
    start_res = client.post(
        "/api/training/start",
        json={"config_name": "ppo_mini.yaml", "override_timesteps": 30, "seed": 42},
    )
    assert start_res.status_code == 200
    status = start_res.json()
    assert status["is_training"] is True

    # Check status endpoint
    get_status = client.get("/api/training/status")
    assert get_status.status_code == 200

    # Stop training
    stop_res = client.post("/api/training/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["is_training"] is False
