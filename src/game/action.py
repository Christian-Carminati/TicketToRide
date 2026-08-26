"""Explicit game actions."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.game.card import CardColor


class ActionType(StrEnum):
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
    color_chosen: CardColor | None = None
    locomotives_count: int = 0
    ticket_ids: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "card_index": self.card_index,
            "route_id": self.route_id,
            "color_chosen": self.color_chosen.value if self.color_chosen else None,
            "locomotives_count": self.locomotives_count,
            "ticket_ids": list(self.ticket_ids) if self.ticket_ids else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        ticket_ids = data.get("ticket_ids")
        return cls(
            action_type=ActionType(data["action_type"]),
            card_index=data.get("card_index"),
            route_id=data.get("route_id"),
            color_chosen=(CardColor(data["color_chosen"]) if data.get("color_chosen") else None),
            locomotives_count=data.get("locomotives_count", 0),
            ticket_ids=tuple(ticket_ids) if ticket_ids is not None else None,
        )
