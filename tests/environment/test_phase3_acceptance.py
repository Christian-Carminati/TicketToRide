"""Phase 3 Acceptance Test: Full games across baseline opponents via Gymnasium."""

import numpy as np
import pytest
from src.agents.greedy_agent import GreedyAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicHeuristicAgent
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board, load_usa_board


@pytest.mark.parametrize("board_loader", [create_synthetic_mini_board, load_usa_board])
def test_phase3_gymnasium_games_acceptance(board_loader):
    """Run full games through the Gymnasium interface across Random, Greedy, and Strategic opponents."""
    board, tickets = board_loader()

    opponents = [
        RandomAgent(seed=101),
        GreedyAgent(),
        StrategicHeuristicAgent(),
    ]

    total_games = 6  # 2 full games per opponent type per board
    for game_idx in range(total_games):
        opp = opponents[game_idx % len(opponents)]
        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=opp,
            max_turns=80,
        )

        obs, info = env.reset(seed=game_idx * 17 + 3)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert "action_mask" in info

        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            mask = info["action_mask"]
            assert np.any(mask), f"Game {game_idx} Step {step_count}: No valid actions in mask!"
            valid_indices = np.where(mask)[0]
            # Select random valid action index among masked actions
            rng = np.random.default_rng(seed=game_idx * 1000 + step_count)
            action = int(rng.choice(valid_indices))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()
            assert not np.isnan(reward)
            assert isinstance(reward, float)

            done = terminated or truncated
            step_count += 1

        assert step_count > 0
        assert isinstance(total_reward, float)


def test_phase3_gymnasium_deterministic_reproducibility():
    """Identical seeds must produce identical observation and reward trajectories."""
    board, tickets = load_usa_board()

    def run_trajectory(seed: int):
        env = TicketToRideEnv(
            board=board,
            tickets_deck=tickets,
            opponent=GreedyAgent(),
            max_turns=60,
        )
        obs, info = env.reset(seed=seed)
        trajectory = []
        done = False
        step = 0
        while not done and step < 20:
            mask = info["action_mask"]
            valid_indices = np.where(mask)[0]
            # Deterministic selection: always lowest valid index
            action = int(valid_indices[0])
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs.copy(), reward, terminated, truncated))
            done = terminated or truncated
            step += 1
        return trajectory

    traj1 = run_trajectory(seed=777)
    traj2 = run_trajectory(seed=777)

    assert len(traj1) == len(traj2)
    for (o1, r1, t1, tr1), (o2, r2, t2, tr2) in zip(traj1, traj2):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2
