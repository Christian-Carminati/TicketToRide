"""Pydantic schemas for REST and WebSocket messages."""

from pydantic import BaseModel


class ActionRequest(BaseModel):
    action_type: str
    card_index: int | None = None
    route_id: str | None = None
    card_color: str | None = None
    ticket_ids: list[str] | None = None


class GameStateResponse(BaseModel):
    turn_number: int
    current_player_index: int
    deck_size: int
    discard_pile_size: int
    is_game_over: bool
    winner_id: str | None = None


class TrainingTelemetry(BaseModel):
    step: int
    episode: int
    reward: float
    mean_reward: float
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
