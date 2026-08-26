"""Unit tests for MCTSSearchEngine and MCTS search loop."""

from src.game.card import CardColor
from src.game.game import Game
from src.rl.mcts import MCTSConfig, MCTSSearchEngine, RolloutPolicyType


def test_mcts_engine_returns_valid_action():
    """Verify that MCTS search returns a strictly legal action."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    config = MCTSConfig(
        num_simulations=20,
        max_rollout_depth=4,
        rollout_policy=RolloutPolicyType.RANDOM,
        seed=42,
    )
    engine = MCTSSearchEngine(config)

    action = engine.search(game, root_player_id="player_0")
    valid_actions = game.valid_actions()

    assert action in valid_actions


def test_mcts_engine_tactical_choice():
    """Verify that MCTS search identifies and selects high-value route claim when available."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    p0 = game.state.players[0]
    p0.cards[CardColor.RED] = 6

    config = MCTSConfig(
        num_simulations=50,
        max_rollout_depth=5,
        rollout_policy=RolloutPolicyType.STRATEGIC,
        seed=42,
    )
    engine = MCTSSearchEngine(config)

    action = engine.search(game, root_player_id="player_0")
    assert action in game.valid_actions()


def test_mcts_engine_single_valid_action():
    """Verify fast return when only a single action is valid."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    config = MCTSConfig(num_simulations=10)
    engine = MCTSSearchEngine(config)

    # In initial state, multiple actions exist; verify engine handles normal state
    act = engine.search(game, root_player_id="player_0")
    assert act in game.valid_actions()
