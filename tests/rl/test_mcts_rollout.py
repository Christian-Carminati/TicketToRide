"""Unit tests for leaf heuristic evaluation and rollout simulation."""

from src.game.game import Game
from src.game.random import SeededRNG
from src.rl.mcts import RolloutPolicyType, evaluate_leaf_state, simulate_rollout


def test_evaluate_leaf_state_bounds():
    """Verify that heuristic evaluation values stay within [-1.0, 1.0]."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    val = evaluate_leaf_state(game, root_player_id="player_0")
    assert -1.0 <= val <= 1.0


def test_evaluate_leaf_terminal_win_loss():
    """Verify terminal win/loss gives exact +/-1.0 returns."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    game.state.is_game_over = True

    # Root wins
    game.state.players[0].score = 100
    game.state.players[1].score = 50
    assert evaluate_leaf_state(game, root_player_id="player_0") == 1.0

    # Opponent wins
    game.state.players[0].score = 40
    game.state.players[1].score = 80
    assert evaluate_leaf_state(game, root_player_id="player_0") == -1.0

    # Draw
    game.state.players[0].score = 50
    game.state.players[1].score = 50
    assert evaluate_leaf_state(game, root_player_id="player_0") == 0.0


def test_simulate_rollout_policies():
    """Verify rollout simulation under different policies produces valid returns."""
    rng = SeededRNG(42)

    for pol in [
        RolloutPolicyType.RANDOM,
        RolloutPolicyType.GREEDY,
        RolloutPolicyType.STRATEGIC,
    ]:
        game = Game(num_players=2, seed=42)
        game.reset(seed=42)

        ret = simulate_rollout(
            game=game,
            root_player_id="player_0",
            max_depth=5,
            policy_type=pol,
            rng=rng,
        )
        assert -1.0 <= ret <= 1.0
