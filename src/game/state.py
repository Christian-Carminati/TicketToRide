"""Explicit, serializable Game State representation."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.game.card import TrainCard
from src.game.player import Player


@dataclass
class GameState:
    players: list[Player] = field(default_factory=list)
    current_player_index: int = 0
    visible_cards: list[TrainCard] = field(default_factory=list)
    deck_size: int = 0
    discard_pile_size: int = 0
    tickets_deck_size: int = 0
    claimed_routes: dict[str, str] = field(default_factory=dict)  # route_id -> player_id
    turn_number: int = 0
    is_last_round: bool = False
    is_game_over: bool = False
    winner_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        # Deserialization helper - will be enhanced in Phase 1
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
