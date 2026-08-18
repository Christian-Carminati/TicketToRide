"""Unit tests for Gymnasium Environment wrapper and components."""

import numpy as np
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.env import TicketToRideEnv
from src.environment.observation import ObservationV1
from src.game.action import ActionType


def test_observation_encoder():
    encoder = ObservationV1(feature_dim=64)
    assert encoder.observation_shape == (64,)


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
    env = TicketToRideEnv(game=sample_game)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert "action_mask" in info
    assert len(info["action_mask"]) == env.action_space.n

    next_obs, reward, terminated, _truncated, _next_info = env.step(0)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
