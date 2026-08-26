"""Unit tests for Gymnasium Environment wrapper and components."""

import numpy as np
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.env import TicketToRideEnv
from src.environment.observation import ObservationV1
from src.game.action import ActionType


def test_observation_encoder():
    encoder = ObservationV1()
    assert encoder.observation_shape == (464,)


def test_action_space_mapping():
    space = DiscreteActionSpace()
    assert space.n >= 1
    act = space.to_action(0)
    assert act.action_type == ActionType.DRAW_HIDDEN_CARD
    assert space.to_id(act) == 0


def test_action_masker():
    space = DiscreteActionSpace()
    masker = ActionMasker(space)
    mask = masker.compute_mask([])
    assert len(mask) == space.n
    assert mask.dtype == bool


def test_environment_lifecycle(sample_game):
    env = TicketToRideEnv(board=sample_game.board, tickets_deck=sample_game.initial_tickets)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert "action_mask" in info
    assert len(info["action_mask"]) == env.action_space.n

    # Sample valid action from mask
    valid_indices = np.where(info["action_mask"])[0]
    action = int(valid_indices[0])

    next_obs, reward, terminated, _truncated, next_info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert "reward_components" in next_info
    assert isinstance(next_info["reward_components"], dict)


def test_environment_reward_version_initialization(sample_game):
    env_v1 = TicketToRideEnv(
        board=sample_game.board,
        tickets_deck=sample_game.initial_tickets,
        reward_calculator="sparse",
    )
    assert env_v1.reward_calc.__class__.__name__ == "RewardV1_Sparse"

    env_v2 = TicketToRideEnv(
        board=sample_game.board,
        tickets_deck=sample_game.initial_tickets,
        reward_calculator="dense_routes",
    )
    assert env_v2.reward_calc.__class__.__name__ == "RewardV2_DenseRoutes"

    env_v3 = TicketToRideEnv(
        board=sample_game.board,
        tickets_deck=sample_game.initial_tickets,
        reward_calculator=3,
    )
    assert env_v3.reward_calc.__class__.__name__ == "RewardV3_TicketMilestones"
