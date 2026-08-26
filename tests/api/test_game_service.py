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


def test_game_service_safe_action_dto_conversion():
    service = GameService()
    # Test valid action
    act1 = service._from_action_dto(ActionDTO(action_type="DRAW_HIDDEN_CARD"))
    assert act1.action_type.name == "DRAW_HIDDEN_CARD"

    # Test gray or invalid color string gracefully handled
    act2 = service._from_action_dto(
        ActionDTO(action_type="CLAIM_ROUTE", route_id="r1", card_color="gray")
    )
    assert act2.color_chosen is None

    # Test lowercase color string
    act3 = service._from_action_dto(
        ActionDTO(action_type="CLAIM_ROUTE", route_id="r1", card_color="blue")
    )
    assert act3.color_chosen is not None
    assert act3.color_chosen.name == "BLUE"


def test_game_service_all_agent_types_instantiation():
    service = GameService()
    types = [
        "human",
        "random",
        "greedy",
        "strategic",
        "mcts",
        "bayesian_mcts",
        "alphazero",
        "recurrent_ppo",
        "ppo",
        "dqn",
        "random_bot",
    ]
    for p_type in types:
        req = GameSessionCreateRequest(map_name="mini", player_types=["human", p_type], seed=42)
        state = service.create_session(req)
        assert state.session_id is not None
        assert len(state.players) == 2
        # Verify bot can step
        step1 = service.step_session(state.session_id, action=state.valid_actions[0])
        assert step1 is not None


def test_game_service_none_seed_creates_session():
    service = GameService()
    req = GameSessionCreateRequest(map_name="mini", player_types=["random_bot", "random_bot"], seed=None)
    state = service.create_session(req)
    assert state.session_id is not None
    assert len(state.players) == 2

