"""Tests for MultiMapTicketToRideEnv."""

import numpy as np
import pytest
from src.environment.multi_map_env import MultiMapTicketToRideEnv
from src.game.procedural import ProceduralMapGenerator


def test_multi_map_env_lifecycle():
    gen = ProceduralMapGenerator()
    maps = [gen.generate(seed=s) for s in [10, 20, 30]]

    env = MultiMapTicketToRideEnv(maps=maps, max_turns=50, seed=42)
    assert env.action_space.n > 0
    assert env.observation_space.shape[0] > 0

    obs1, info1 = env.reset(seed=100)
    assert isinstance(obs1, np.ndarray)
    assert "action_mask" in info1
    assert obs1.shape == env.observation_space.shape

    # Step in environment
    mask = info1["action_mask"]
    valid_act = int(np.where(mask)[0][0])
    obs2, reward, terminated, truncated, info2 = env.step(valid_act)

    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_multi_map_env_cycles_maps():
    gen = ProceduralMapGenerator()
    maps = [gen.generate(seed=s) for s in [1, 2, 3]]
    env = MultiMapTicketToRideEnv(maps=maps, sampling="round_robin", seed=42)

    seen_boards = []
    for _ in range(3):
        env.reset()
        seen_boards.append([c.name for c in env.current_board.cities.values()])

    assert len(seen_boards) == 3
