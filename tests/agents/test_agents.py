"""Unit tests for Baseline and RL Agent skeletons."""

import numpy as np
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent


def test_random_agent_selection():
    agent = RandomAgent(seed=42)
    obs = np.zeros(64)
    mask = np.array([True, False, False])
    action = agent.select_action(obs, action_mask=mask)
    assert action == 0


def test_heuristic_agent_selection():
    agent = HeuristicAgent()
    obs = np.zeros(64)
    mask = np.array([False, True, False])
    action = agent.select_action(obs, action_mask=mask)
    assert action == 1
