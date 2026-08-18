"""Unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_get_state():
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert data["turn_number"] == 1
    assert data["deck_size"] == 110
