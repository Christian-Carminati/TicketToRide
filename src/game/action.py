"""Explicit game actions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.game.card import CardColor


class ActionType(str, Enum):
    DRAW_VISIBLE_CARD = "draw_visible_card"
    DRAW_HIDDEN_CARD = "draw_hidden_card"
    CLAIM_ROUTE = "claim_route"
    DRAW_TICKETS = "draw_tickets"
    KEEP_TICKETS = "keep_tickets"


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    card_index: int | None = None
    route_id: str | None = None
    card_color: CardColor | None = None
    ticket_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "card_index": self.card_index,
            "route_id": self.route_id,
            "card_color": self.card_color.value if self.card_color else None,
            "ticket_ids": self.ticket_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            action_type=ActionType(data["action_type"]),
            card_index=data.get("card_index"),
            route_id=data.get("route_id"),
            card_color=CardColor(data["card_color"]) if data.get("card_color") else None,
            ticket_ids=data.get("ticket_ids"),
        )
