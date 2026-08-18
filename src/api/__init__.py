"""API package: FastAPI app, WebSockets, and Pydantic schemas for Web Lab."""

from src.api.main import app
from src.api.schemas import ActionRequest, GameStateResponse, TrainingTelemetry
from src.api.websocket import ConnectionManager

__all__ = [
    "ActionRequest",
    "ConnectionManager",
    "GameStateResponse",
    "TrainingTelemetry",
    "app",
]
