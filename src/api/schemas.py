"""Pydantic Data Transfer Objects (DTOs) for API serialization."""

from typing import Any, Literal
from pydantic import BaseModel


# --- Game State DTOs ---
class PlayerStateDTO(BaseModel):
    player_id: str
    name: str
    score: int
    trains_remaining: int
    cards_in_hand: dict[str, int]
    tickets: list[dict[str, Any]]
    claimed_route_ids: list[str]
    color: str


class ActionDTO(BaseModel):
    action_type: str
    card_index: int | None = None
    route_id: str | None = None
    card_color: str | None = None
    ticket_ids: list[str] | None = None


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
    claimed_routes: dict[str, str]  # route_id -> player_id
    valid_actions: list[ActionDTO]
    action_mask: list[bool] | None = None
    is_game_over: bool
    winner_id: str | None = None
    last_action: ActionDTO | None = None
    last_reward: float | None = None


class GameSessionCreateRequest(BaseModel):
    map_name: str = "usa"
    player_types: list[str] = ["human", "random"]  # "human", "random", "greedy", "strategic", "dqn", "ppo"
    seed: int = 42
    model_checkpoint: str | None = None


class GameStepRequest(BaseModel):
    session_id: str
    action: ActionDTO | None = None  # None for bot step


# --- Neural Brain Inspection DTOs ---
class LayerActivationDTO(BaseModel):
    layer_name: str
    shape: list[int]
    mean: float
    std: float
    min: float
    max: float
    values: list[float]


class BrainInspectionDTO(BaseModel):
    model_type: Literal["dqn", "ppo"]
    estimated_value: float
    action_probabilities: list[float]  # Policy distribution (PPO) or Q-values (DQN)
    action_mask: list[bool]
    action_labels: list[str]
    greedy_action_index: int
    raw_logits_or_q: list[float] | None = None
    masked_logits_or_q: list[float]
    observation_vector: list[float]
    layer_activations: list[LayerActivationDTO] | None = None


class BrainInspectRequest(BaseModel):
    session_id: str | None = None
    model_type: Literal["dqn", "ppo"] = "ppo"
    checkpoint_path: str | None = None
    observation: list[float] | None = None
    action_mask: list[bool] | None = None


# --- Training Telemetry DTOs ---
class TrainingStartRequest(BaseModel):
    config_name: str = "ppo_usa.yaml"
    override_timesteps: int | None = None
    seed: int = 42
    opponent_type: str = "random"  # "random", "greedy", "strategic"


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
    clip_fraction: float | None = None
    explained_var: float | None = None
    win_rate: float | None = None
    fps: float | None = None



# --- Tournament & Leaderboard DTOs ---
class TournamentParticipantOptionDTO(BaseModel):
    id: str
    name: str
    category: Literal["baseline", "checkpoint"]
    algorithm: str  # "strategic", "greedy", "random", "ppo", "dqn"
    checkpoint_path: str | None = None
    description: str | None = None


class TournamentAgentDTO(BaseModel):
    agent_id: str
    name: str
    elo: float
    win_rate: float
    wins: int
    losses: int
    draws: int
    avg_score: float
    total_games: int


class TournamentMatchupDTO(BaseModel):
    agent_a: str
    agent_b: str
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    avg_score_a: float
    avg_score_b: float
    games_played: int


class TournamentLeaderboardDTO(BaseModel):
    leaderboard: list[TournamentAgentDTO]
    matchups: list[TournamentMatchupDTO]
    total_games: int
    updated_at: str
    map_name: str = "usa"
    available_participants: list[TournamentParticipantOptionDTO] | None = None


class TournamentRunRequest(BaseModel):
    participant_ids: list[str] | None = None  # Specific list of IDs to include
    games_per_pair: int = 15
    map_name: str = "usa"
    seed: int = 42


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


# --- Checkpoint DTOs ---
class CheckpointDTO(BaseModel):
    checkpoint_id: str
    name: str
    algorithm: str
    path: str
    size_mb: float
    modified_at: str
    total_timesteps: int | None = None


# --- Scientific Reports & Benchmarks DTOs ---
class ReportItemDTO(BaseModel):
    id: str
    name: str
    filename: str
    file_type: str  # "markdown", "json", "text"
    size_kb: float
    modified_at: str
    phase: str | None = None


class ReportDetailDTO(BaseModel):
    id: str
    name: str
    filename: str
    file_type: str
    raw_content: str
    json_data: dict[str, Any] | None = None
    size_kb: float
    modified_at: str


# Aliases for backward compatibility
GameStateResponse = GameStateDTO
ActionRequest = ActionDTO
TrainingTelemetry = TelemetryEventDTO

