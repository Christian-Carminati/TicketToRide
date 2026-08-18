"""Unit tests for TurnState, Action, and GameState serialization."""

from src.game.action import Action, ActionType
from src.game.card import CardColor, TrainCard
from src.game.player import Player
from src.game.state import GameState, TurnState
from src.game.ticket import DestinationTicket


def test_action_roundtrip_serialization():
    act = Action(
        action_type=ActionType.CLAIM_ROUTE,
        route_id="r_nyc_bos",
        color_chosen=CardColor.RED,
        locomotives_count=1,
    )
    d = act.to_dict()
    act2 = Action.from_dict(d)
    assert act2 == act


def test_gamestate_roundtrip_serialization():
    state = GameState(
        players=[Player(id="p0", name="Player 1"), Player(id="p1", name="Player 2")],
        current_player_index=0,
        turn_state=TurnState.NORMAL,
        visible_cards=[
            TrainCard(color=CardColor.RED),
            TrainCard(color=CardColor.LOCOMOTIVE),
        ],
        train_deck=[TrainCard(color=CardColor.BLUE)],
        discard_pile=[TrainCard(color=CardColor.GREEN)],
        ticket_deck=[DestinationTicket(id="t1", city_a="A", city_b="B", points=5)],
        turn_number=3,
        is_last_round=False,
        is_game_over=False,
    )
    json_str = state.to_json()
    restored = GameState.from_json(json_str)

    assert restored.current_player_index == 0
    assert restored.turn_state == TurnState.NORMAL
    assert len(restored.visible_cards) == 2
    assert restored.visible_cards[1].color == CardColor.LOCOMOTIVE
    assert len(restored.train_deck) == 1
    assert len(restored.discard_pile) == 1
    assert len(restored.ticket_deck) == 1
    assert restored.turn_number == 3
