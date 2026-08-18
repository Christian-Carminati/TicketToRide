"""Player state representation."""

from dataclasses import dataclass, field

from src.game.card import CardColor, TrainCard
from src.game.ticket import DestinationTicket


@dataclass
class Player:
    id: str
    name: str
    trains_remaining: int = 45
    score: int = 0
    cards: dict[CardColor, int] = field(
        default_factory=lambda: {color: 0 for color in CardColor}
    )
    tickets: list[DestinationTicket] = field(default_factory=list)
    claimed_route_ids: list[str] = field(default_factory=list)

    def total_cards(self) -> int:
        return sum(self.cards.values())

    def add_card(self, card: TrainCard) -> None:
        self.cards[card.color] = self.cards.get(card.color, 0) + 1
