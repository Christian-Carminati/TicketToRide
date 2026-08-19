"""Unit tests for GameService interactive and bot session management."""

from src.api.game_service import GameService
from src.api.schemas import ActionDTO, GameSessionCreateRequest


def test_game_service_create_get_delete():
    service = GameService()
    req = GameSessionCreateRequest(map_name="mini", player_types=["human", "random"], seed=42)
    state = service.create_session(req)

    assert state.session_id is not None
    assert state.turn_number == 1
    assert state.map_name == "mini"
    assert len(state.players) == 2
    assert len(state.valid_actions) > 0
    assert len(state.action_mask) > 0

    fetched = service.get_session_state(state.session_id)
    assert fetched is not None
    assert fetched.session_id == state.session_id

    deleted = service.delete_session(state.session_id)
    assert deleted is True
    assert service.get_session_state(state.session_id) is None


def test_game_service_bot_and_human_step():
    service = GameService()
    req = GameSessionCreateRequest(map_name="mini", player_types=["random", "greedy"], seed=42)
    state = service.create_session(req)

    # Initial turn is choosing tickets or normal turn
    # Perform a bot step (action=None)
    next_state = service.step_session(state.session_id, action=None)
    assert next_state.session_id == state.session_id
    assert next_state.last_action is not None


def test_game_service_human_step():
    service = GameService()
    req = GameSessionCreateRequest(map_name="mini", player_types=["human", "random"], seed=100)
    state = service.create_session(req)

    # Take the first valid action
    first_valid = state.valid_actions[0]
    next_state = service.step_session(state.session_id, action=first_valid)
    assert next_state.session_id == state.session_id
    assert next_state.last_action is not None
