"""Player state and inventory representation."""

from typing import Any

from src.game.card import CardColor, TrainCard
from src.game.route import Route
from src.game.ticket import DestinationTicket


class Player:
    """Represents a player's private inventory and public scoring state."""

    def __init__(
        self,
        id: str,
        name: str,
        trains_remaining: int = 45,
        score: int = 0,
        cards: dict[CardColor, int] | None = None,
        tickets: list[DestinationTicket] | None = None,
        claimed_route_ids: list[str] | None = None,
        pending_tickets: list[DestinationTicket] | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.trains_remaining = trains_remaining
        self.score = score
        self.cards: dict[CardColor, int] = (
            cards.copy() if cards is not None else {c: 0 for c in CardColor}
        )
        self.tickets: list[DestinationTicket] = (
            tickets.copy() if tickets is not None else []
        )
        self.claimed_route_ids: list[str] = (
            claimed_route_ids.copy() if claimed_route_ids is not None else []
        )
        self.pending_tickets: list[DestinationTicket] = (
            pending_tickets.copy() if pending_tickets is not None else []
        )

    def total_cards(self) -> int:
        return sum(self.cards.values())

    def add_card(self, card: TrainCard) -> None:
        self.cards[card.color] = self.cards.get(card.color, 0) + 1

    def remove_cards(self, cards_to_remove: dict[CardColor, int]) -> bool:
        """Remove specified cards if player has sufficient inventory.

        Returns True if successful, False without changes if insufficient.
        """
        for color, count in cards_to_remove.items():
            if self.cards.get(color, 0) < count:
                return False

        for color, count in cards_to_remove.items():
            self.cards[color] -= count
        return True

    def has_trains_for_route(self, route: Route) -> bool:
        return self.trains_remaining >= route.length

    def can_afford_route(self, route: Route) -> list[dict[CardColor, int]]:
        """Compute all valid combinations of cards the player can use to claim this route.

        Returns a list of dicts: {CardColor: count_to_spend}.
        """
        if not self.has_trains_for_route(route):
            return []

        required_len = route.length
        locomotives = self.cards.get(CardColor.LOCOMOTIVE, 0)
        options: list[dict[CardColor, int]] = []

        if route.color is not None:
            # Colored route: must use route.color and/or LOCOMOTIVE
            color_count = self.cards.get(route.color, 0)
            if color_count + locomotives >= required_len:
                min_color_needed = max(0, required_len - locomotives)
                max_color_possible = min(required_len, color_count)
                for c_used in range(min_color_needed, max_color_possible + 1):
                    locos_used = required_len - c_used
                    option: dict[CardColor, int] = {}
                    if c_used > 0:
                        option[route.color] = c_used
                    if locos_used > 0:
                        option[CardColor.LOCOMOTIVE] = locos_used
                    if option:
                        options.append(option)
        else:
            # Gray route: can use any single color (plus locomotives) or all locomotives
            for color, color_count in self.cards.items():
                if color == CardColor.LOCOMOTIVE or color_count == 0:
                    continue
                if color_count + locomotives >= required_len:
                    min_color_needed = max(1, required_len - locomotives)
                    max_color_possible = min(required_len, color_count)
                    for c_used in range(min_color_needed, max_color_possible + 1):
                        locos_used = required_len - c_used
                        option = {color: c_used}
                        if locos_used > 0:
                            option[CardColor.LOCOMOTIVE] = locos_used
                        options.append(option)

            # Option to pay purely with locomotives if enough
            if locomotives >= required_len and {CardColor.LOCOMOTIVE: required_len} not in options:
                options.append({CardColor.LOCOMOTIVE: required_len})

        return options

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trains_remaining": self.trains_remaining,
            "score": self.score,
            "cards": {k.value: v for k, v in self.cards.items()},
            "tickets": [
                {"id": t.id, "city_a": t.city_a, "city_b": t.city_b, "points": t.points}
                for t in self.tickets
            ],
            "claimed_route_ids": list(self.claimed_route_ids),
            "pending_tickets": [
                {"id": t.id, "city_a": t.city_a, "city_b": t.city_b, "points": t.points}
                for t in self.pending_tickets
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Player":
        cards = {CardColor(k): v for k, v in data.get("cards", {}).items()}
        for color in CardColor:
            if color not in cards:
                cards[color] = 0

        tickets = [
            DestinationTicket(
                id=t["id"],
                city_a=t["city_a"],
                city_b=t["city_b"],
                points=t["points"],
            )
            for t in data.get("tickets", [])
        ]
        pending_tickets = [
            DestinationTicket(
                id=t["id"],
                city_a=t["city_a"],
                city_b=t["city_b"],
                points=t["points"],
            )
            for t in data.get("pending_tickets", [])
        ]

        return cls(
            id=data["id"],
            name=data["name"],
            trains_remaining=data.get("trains_remaining", 45),
            score=data.get("score", 0),
            cards=cards,
            tickets=tickets,
            claimed_route_ids=data.get("claimed_route_ids", []),
            pending_tickets=pending_tickets,
        )
