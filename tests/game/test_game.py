"""Unit tests for Game Core."""

from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.game import Game
from src.game.rules import GameRules
from src.game.state import GameState


def test_card_creation():
    card = TrainCard(color=CardColor.LOCOMOTIVE)
    assert card.is_locomotive() is True
    assert card.color == CardColor.LOCOMOTIVE


def test_rules_route_scoring():
    assert GameRules.points_for_route_length(1) == 1
    assert GameRules.points_for_route_length(6) == 15
    assert GameRules.points_for_route_length(10) == 0


def test_game_deterministic_reset(sample_board):
    game1 = Game(board=sample_board, num_players=2, seed=123)
    game2 = Game(board=sample_board, num_players=2, seed=123)

    state1 = game1.reset()
    state2 = game2.reset()

    assert state1.turn_number == state2.turn_number
    assert len(state1.players) == len(state2.players) == 2


def test_action_serialization():
    act = Action(action_type=ActionType.CLAIM_ROUTE, route_id="r_nyc_bos")
    data = act.to_dict()
    restored = Action.from_dict(data)
    assert restored.action_type == ActionType.CLAIM_ROUTE
    assert restored.route_id == "r_nyc_bos"


def test_state_serialization():
    state = GameState(turn_number=5, current_player_index=1)
    json_str = state.to_json()
    assert '"turn_number": 5' in json_str
    assert '"current_player_index": 1' in json_str
