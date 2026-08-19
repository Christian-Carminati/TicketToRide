"""End-to-end integration test verifying the full telemetry and workbench flow for TicketToRide RL Lab."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import GameSessionCreateRequest, TrainingStartRequest

client = TestClient(app)


def test_workbench_full_lifecycle_and_telemetry():
    """Verify session creation -> bot step -> neural brain inspection -> training telemetry status."""
    # 1. Create interactive match
    res = client.post(
        "/api/game/new",
        json={"map_name": "mini", "player_types": ["ppo", "greedy"], "seed": 42},
    )
    assert res.status_code == 200
    sess = res.json()
    session_id = sess["session_id"]
    assert sess["turn_number"] == 1
    assert len(sess["players"]) == 2
    assert "visible_cards" in sess

    # 2. Step bot
    step_res = client.post(
        "/api/game/step",
        json={"session_id": session_id},
    )
    assert step_res.status_code == 200
    stepped_state = step_res.json()
    assert stepped_state["session_id"] == session_id
    assert stepped_state["turn_number"] >= 1

    # 3. Inspect Neural Brain
    brain_res = client.post(
        "/api/brain/inspect",
        json={"session_id": session_id, "model_type": "ppo"},
    )
    assert brain_res.status_code == 200
    brain = brain_res.json()
    assert brain["model_type"] == "ppo"
    assert len(brain["action_probabilities"]) > 0
    assert len(brain["action_mask"]) > 0
    assert "estimated_value" in brain

    # 4. Check Training Endpoints
    status_res = client.get("/api/training/status")
    assert status_res.status_code == 200
    status = status_res.json()
    assert "is_training" in status

    # 5. Check Replay list
    replays_res = client.get("/api/replays/list")
    assert replays_res.status_code == 200
    assert isinstance(replays_res.json(), list)

    # 6. Check Experiments list
    exp_res = client.get("/api/experiments/list")
    assert exp_res.status_code == 200
    assert isinstance(exp_res.json(), list)
