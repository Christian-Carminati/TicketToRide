"""Explicit, serializable Game State representation."""

import json
from enum import Enum
from typing import Any

from src.game.card import CardColor, TrainCard
from src.game.player import Player
from src.game.ticket import DestinationTicket


class TurnState(str, Enum):
    CHOOSING_INITIAL_TICKETS = "choosing_initial_tickets"
    NORMAL = "normal"
    DRAWING_SECOND_CARD = "drawing_second_card"
    CHOOSING_TICKETS = "choosing_tickets"


class GameState:
    def __init__(
        self,
        players: list[Player] | None = None,
        current_player_index: int = 0,
        turn_state: TurnState = TurnState.NORMAL,
        visible_cards: list[TrainCard] | None = None,
        train_deck: list[TrainCard] | None = None,
        discard_pile: list[TrainCard] | None = None,
        ticket_deck: list[DestinationTicket] | None = None,
        turn_number: int = 1,
        is_last_round: bool = False,
        final_turn_player_id: str | None = None,
        is_game_over: bool = False,
        winner_id: str | None = None,
        num_players: int = 2,
    ) -> None:
        self.players = players or []
        self.current_player_index = current_player_index
        self.turn_state = turn_state
        self.visible_cards = visible_cards or []
        self.train_deck = train_deck or []
        self.discard_pile = discard_pile or []
        self.ticket_deck = ticket_deck or []
        self.turn_number = turn_number
        self.is_last_round = is_last_round
        self.final_turn_player_id = final_turn_player_id
        self.is_game_over = is_game_over
        self.winner_id = winner_id
        self.num_players = num_players

    @property
    def current_player(self) -> Player | None:
        if self.players and 0 <= self.current_player_index < len(self.players):
            return self.players[self.current_player_index]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": [p.to_dict() for p in self.players],
            "current_player_index": self.current_player_index,
            "turn_state": self.turn_state.value,
            "visible_cards": [c.color.value for c in self.visible_cards],
            "train_deck": [c.color.value for c in self.train_deck],
            "discard_pile": [c.color.value for c in self.discard_pile],
            "ticket_deck": [
                {"id": t.id, "city_a": t.city_a, "city_b": t.city_b, "points": t.points}
                for t in self.ticket_deck
            ],
            "turn_number": self.turn_number,
            "is_last_round": self.is_last_round,
            "final_turn_player_id": self.final_turn_player_id,
            "is_game_over": self.is_game_over,
            "winner_id": self.winner_id,
            "num_players": self.num_players,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        players = [Player.from_dict(p) for p in data.get("players", [])]
        visible_cards = [TrainCard(color=CardColor(c)) for c in data.get("visible_cards", [])]
        train_deck = [TrainCard(color=CardColor(c)) for c in data.get("train_deck", [])]
        discard_pile = [TrainCard(color=CardColor(c)) for c in data.get("discard_pile", [])]
        ticket_deck = [
            DestinationTicket(
                id=t["id"],
                city_a=t["city_a"],
                city_b=t["city_b"],
                points=t["points"],
            )
            for t in data.get("ticket_deck", [])
        ]

        return cls(
            players=players,
            current_player_index=data.get("current_player_index", 0),
            turn_state=TurnState(data.get("turn_state", TurnState.NORMAL.value)),
            visible_cards=visible_cards,
            train_deck=train_deck,
            discard_pile=discard_pile,
            ticket_deck=ticket_deck,
            turn_number=data.get("turn_number", 1),
            is_last_round=data.get("is_last_round", False),
            final_turn_player_id=data.get("final_turn_player_id"),
            is_game_over=data.get("is_game_over", False),
            winner_id=data.get("winner_id"),
            num_players=data.get("num_players", len(players)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "GameState":
        return cls.from_dict(json.loads(json_str))

    def clone(self) -> "GameState":
        """Fast in-memory clone of GameState."""
        return GameState(
            players=[p.clone() for p in self.players],
            current_player_index=self.current_player_index,
            turn_state=self.turn_state,
            visible_cards=[TrainCard(color=c.color) for c in self.visible_cards],
            train_deck=[TrainCard(color=c.color) for c in self.train_deck],
            discard_pile=[TrainCard(color=c.color) for c in self.discard_pile],
            ticket_deck=list(self.ticket_deck),
            turn_number=self.turn_number,
            is_last_round=self.is_last_round,
            final_turn_player_id=self.final_turn_player_id,
            is_game_over=self.is_game_over,
            winner_id=self.winner_id,
            num_players=self.num_players,
        )

