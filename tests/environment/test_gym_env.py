"""Tests for Gymnasium environment wrapper and check_env compliance."""

import numpy as np
from gymnasium.utils.env_checker import check_env
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board, load_usa_board


def test_gymnasium_check_env_synthetic_mini():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))
    # Must pass without errors
    check_env(env.unwrapped, skip_render_check=True)


def test_gymnasium_check_env_usa_board():
    board, tickets = load_usa_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=RandomAgent(seed=42))
    check_env(env.unwrapped, skip_render_check=True)


def test_gym_env_step_lifecycle_with_opponent():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets, opponent=GreedyAgent())

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert "action_mask" in info
    assert info["action_mask"].dtype == bool
    assert np.any(info["action_mask"])

    done = False
    step_count = 0
    while not done and step_count < 100:
        mask = info["action_mask"]
        valid_indices = np.where(mask)[0]
        action = int(valid_indices[0])

        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_mask" in info
        done = terminated or truncated
        step_count += 1

    assert step_count > 0
