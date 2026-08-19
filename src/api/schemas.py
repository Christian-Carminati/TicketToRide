"""Pydantic v2 schemas and DTO models for REST and WebSocket messages in TicketToRide RL Lab."""

from typing import Any, Literal
from pydantic import BaseModel, Field


# --- Game DTOs ---
class ActionDTO(BaseModel):
    action_type: str
    card_index: int | None = None
    route_id: str | None = None
    card_color: str | None = None
    ticket_ids: list[str] | None = None


class PlayerStateDTO(BaseModel):
    player_id: str
    name: str
    score: int
    trains_remaining: int
    cards_in_hand: dict[str, int]
    tickets: list[dict[str, Any]]
    claimed_route_ids: list[str]
    color: str


class GameSessionCreateRequest(BaseModel):
    map_name: Literal["mini", "usa"] = "mini"
    player_types: list[str] = Field(default_factory=lambda: ["human", "random"])
    seed: int = 42
    model_checkpoint: str | None = None


class GameStateDTO(BaseModel):
    session_id: str
    turn_number: int
    current_player_index: int
    map_name: str
    players: list[PlayerStateDTO]
    visible_cards: list[str]
    deck_size: int
    discard_pile_size: int
    tickets_deck_size: int
    claimed_routes: dict[str, str] = Field(default_factory=dict)  # route_id -> player_id
    valid_actions: list[ActionDTO] = Field(default_factory=list)
    action_mask: list[bool] = Field(default_factory=list)
    is_game_over: bool = False
    winner_id: str | None = None
    last_action: ActionDTO | None = None
    last_reward: float | None = None


class GameStepRequest(BaseModel):
    session_id: str
    action: ActionDTO | None = None


# --- Brain & Introspection DTOs ---
class LayerActivationDTO(BaseModel):
    layer_name: str
    shape: list[int]
    mean: float
    std: float
    min: float
    max: float
    values: list[float] = Field(default_factory=list)


class BrainInspectionDTO(BaseModel):
    model_type: Literal["dqn", "ppo"]
    observation_vector: list[float]
    action_mask: list[bool]
    layer_activations: list[LayerActivationDTO]
    raw_logits_or_q: list[float]
    masked_logits_or_q: list[float]
    action_probabilities: list[float]
    estimated_value: float | None = None
    greedy_action_index: int
    action_labels: list[str] = Field(default_factory=list)


class BrainInspectRequest(BaseModel):
    session_id: str | None = None
    model_type: Literal["dqn", "ppo"] = "ppo"
    checkpoint_path: str | None = None
    observation: list[float] | None = None
    action_mask: list[bool] | None = None


# --- Training Telemetry DTOs ---
class TrainingStartRequest(BaseModel):
    config_name: str
    override_timesteps: int | None = None
    seed: int = 42


class TrainingStatusDTO(BaseModel):
    is_training: bool
    experiment_id: str | None = None
    algorithm: str | None = None
    current_step: int = 0
    total_timesteps: int = 0
    episodes: int = 0
    mean_reward: float = 0.0


class TelemetryEventDTO(BaseModel):
    type: Literal["training_started", "training_step", "checkpoint_saved", "training_finished", "error"]
    experiment_id: str
    step: int
    episode: int
    reward: float
    mean_reward: float
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    approx_kl: float | None = None
    win_rate: float | None = None
    fps: float | None = None


# --- Replay DTOs ---
class ReplayFrameDTO(BaseModel):
    step_index: int
    turn_number: int
    player_index: int
    action: ActionDTO
    reward: float
    state_snapshot: dict[str, Any]
    observation: list[float] | None = None
    action_mask: list[bool] | None = None
    action_probabilities: list[float] | None = None


class ReplayDetailDTO(BaseModel):
    replay_id: str
    map_name: str
    seed: int
    date: str
    player_names: list[str]
    total_steps: int
    winner_index: int
    final_scores: list[int]
    frames: list[ReplayFrameDTO]


# Aliases for backward compatibility
GameStateResponse = GameStateDTO
ActionRequest = ActionDTO
TrainingTelemetry = TelemetryEventDTO
