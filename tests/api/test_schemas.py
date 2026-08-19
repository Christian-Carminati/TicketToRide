"""Unit tests for Phase 5 Pydantic API schemas and DTO models."""

from src.api.schemas import (
    ActionDTO,
    BrainInspectionDTO,
    GameSessionCreateRequest,
    GameStateDTO,
    GameStepRequest,
    LayerActivationDTO,
    PlayerStateDTO,
    ReplayDetailDTO,
    ReplayFrameDTO,
    TelemetryEventDTO,
    TrainingStartRequest,
    TrainingStatusDTO,
)


def test_game_dto_serialization():
    action = ActionDTO(action_type="CLAIM_ROUTE", route_id="r_0_bos_ny", card_color="RED")
    assert action.action_type == "CLAIM_ROUTE"
    assert action.route_id == "r_0_bos_ny"

    player = PlayerStateDTO(
        player_id="p0",
        name="Agent 1",
        score=15,
        trains_remaining=35,
        cards_in_hand={"RED": 4, "BLUE": 2},
        tickets=[{"id": "t1", "points": 10, "completed": True}],
        claimed_route_ids=["r_0_bos_ny"],
        color="#3B82F6",
    )
    assert player.score == 15

    state = GameStateDTO(
        session_id="sess_123",
        turn_number=5,
        current_player_index=0,
        map_name="mini",
        players=[player],
        visible_cards=["RED", "BLUE", "LOCOMOTIVE", "GREEN", "YELLOW"],
        deck_size=90,
        discard_pile_size=5,
        tickets_deck_size=20,
        claimed_routes={"r_0_bos_ny": "p0"},
        valid_actions=[action],
        action_mask=[True, False, True],
        is_game_over=False,
    )
    data = state.model_dump()
    assert data["session_id"] == "sess_123"
    assert len(data["players"]) == 1


def test_brain_dto_serialization():
    layer = LayerActivationDTO(
        layer_name="fc1",
        shape=[1, 256],
        mean=0.45,
        std=0.12,
        min=0.0,
        max=1.85,
        values=[0.1, 0.4, 0.9],
    )
    brain = BrainInspectionDTO(
        model_type="ppo",
        observation_vector=[0.1] * 128,
        action_mask=[True] * 56,
        layer_activations=[layer],
        raw_logits_or_q=[1.5, 3.2, 0.1],
        masked_logits_or_q=[1.5, 3.2, 0.1],
        action_probabilities=[0.15, 0.80, 0.05],
        estimated_value=4.2,
        greedy_action_index=1,
        action_labels=["Draw Deck", "Draw Card 1", "Claim R1"],
    )
    assert brain.greedy_action_index == 1
    assert brain.estimated_value == 4.2


def test_training_and_replay_dtos():
    train_req = TrainingStartRequest(config_name="ppo_mini.yaml", override_timesteps=1000, seed=42)
    assert train_req.config_name == "ppo_mini.yaml"

    train_status = TrainingStatusDTO(
        is_training=True,
        experiment_id="exp_01",
        algorithm="ppo",
        current_step=500,
        total_timesteps=1000,
        episodes=5,
        mean_reward=12.5,
    )
    assert train_status.is_training is True

    telemetry = TelemetryEventDTO(
        type="training_step",
        experiment_id="exp_01",
        step=500,
        episode=5,
        reward=12.5,
        mean_reward=12.5,
        policy_loss=0.02,
        value_loss=1.1,
        entropy=1.5,
        approx_kl=0.008,
        win_rate=0.75,
        fps=120.0,
    )
    assert telemetry.type == "training_step"

    frame = ReplayFrameDTO(
        step_index=0,
        turn_number=1,
        player_index=0,
        action=ActionDTO(action_type="DRAW_TRAIN_CARDS"),
        reward=0.0,
        state_snapshot={"turn": 1},
    )
    replay = ReplayDetailDTO(
        replay_id="rep_01",
        map_name="mini",
        seed=42,
        date="2026-08-19",
        player_names=["Agent 1", "Agent 2"],
        total_steps=1,
        winner_index=0,
        final_scores=[20, 10],
        frames=[frame],
    )
    assert replay.replay_id == "rep_01"
    assert len(replay.frames) == 1
