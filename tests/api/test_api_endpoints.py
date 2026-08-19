"""Integration tests for all FastAPI REST endpoints and WebSocket telemetry in TicketToRide RL Lab."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_game_endpoints_crud():
    # 1. Create session
    create_res = client.post(
        "/api/game/new",
        json={"map_name": "mini", "player_types": ["human", "random"], "seed": 42},
    )
    assert create_res.status_code == 200
    data = create_res.json()
    session_id = data["session_id"]
    assert session_id.startswith("sess_")
    assert data["turn_number"] == 1
    assert len(data["players"]) == 2

    # 2. Get state
    get_res = client.get(f"/api/game/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["session_id"] == session_id

    # 3. Step session (bot/first valid action)
    step_res = client.post("/api/game/step", json={"session_id": session_id})
    assert step_res.status_code == 200
    assert step_res.json()["session_id"] == session_id

    # 4. Delete session
    del_res = client.delete(f"/api/game/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 5. Verify deleted
    get_after = client.get(f"/api/game/{session_id}")
    assert get_after.status_code == 404


def test_training_endpoints():
    # 1. Status
    status_res = client.get("/api/training/status")
    assert status_res.status_code == 200

    # 2. Start
    start_res = client.post(
        "/api/training/start",
        json={"config_name": "ppo_usa.yaml", "override_timesteps": 50, "seed": 42},
    )
    assert start_res.status_code == 200
    assert start_res.json()["is_training"] is True

    # 3. Stop
    stop_res = client.post("/api/training/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["is_training"] is False


def test_replays_and_experiments_endpoints():
    # Replays list
    replays_res = client.get("/api/replays/list")
    assert replays_res.status_code == 200
    assert isinstance(replays_res.json(), list)

    # Experiments list
    exp_res = client.get("/api/experiments/list")
    assert exp_res.status_code == 200
    assert isinstance(exp_res.json(), list)


def test_brain_inspect_endpoint():
    # Inspect a synthetic observation
    obs = [0.1] * 128
    mask = [True] * 56
    res = client.post(
        "/api/brain/inspect",
        json={"model_type": "ppo", "observation": obs, "action_mask": mask},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_type"] == "ppo"
    assert len(data["action_probabilities"]) == 56


def test_websocket_telemetry_hub():
    with client.websocket_connect("/ws/telemetry") as websocket:
        websocket.send_text("ping")
        msg = websocket.receive_json()
        assert msg["type"] in ["ack", "training_started", "training_step", "training_finished"]
