"""Test suite for Rust native core equivalence and correctness."""

from src.game.action import Action, ActionType
from src.game.native import NativeGame, NativeVectorEnv


def test_native_game_creation_and_reset():
    game = NativeGame(seed=123)
    assert not game.is_game_over
    assert game.current_player_index == 0
    assert game.turn_number == 1
    assert game.turn_state == "choosing_initial_tickets"
    assert game.scores == (0, 0)
    assert game.trains_remaining == (45, 45)


def test_native_game_step_and_valid_actions():
    game = NativeGame(seed=42)
    actions = game.valid_actions()
    assert len(actions) > 0
    assert all(isinstance(a, Action) for a in actions)
    assert actions[0].action_type == ActionType.KEEP_TICKETS

    # Step action
    game.step(actions[0])
    assert game.current_player_index == 1
    assert game.turn_state == "choosing_initial_tickets"

    actions_p1 = game.valid_actions()
    assert len(actions_p1) > 0
    game.step(actions_p1[0])

    # Now in normal turn state for player 0
    assert game.current_player_index == 0
    assert game.turn_state == "normal"


def test_native_game_cloning():
    game = NativeGame(seed=999)
    actions = game.valid_actions()
    game.step(actions[0])

    cloned = game.clone()
    assert cloned.current_player_index == game.current_player_index
    assert cloned.turn_number == game.turn_number
    assert cloned.turn_state == game.turn_state
    assert cloned.scores == game.scores

    # Stepping cloned does not affect original
    cloned_acts = cloned.valid_actions()
    cloned.step(cloned_acts[0])
    assert cloned.current_player_index != game.current_player_index


def test_native_vector_env_throughput():
    vec_env = NativeVectorEnv(num_envs=64, base_seed=42)
    total_steps = vec_env.step_batch_sim(num_steps_per_env=50)
    assert total_steps > 0
