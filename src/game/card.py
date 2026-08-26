"""Card representations and standard deck generation for Ticket to Ride."""

from dataclasses import dataclass
from enum import StrEnum


class CardColor(StrEnum):
    PURPLE = "purple"
    WHITE = "white"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    LOCOMOTIVE = "locomotive"  # Wild card


@dataclass(frozen=True)
class TrainCard:
    color: CardColor

    def is_locomotive(self) -> bool:
        return self.color == CardColor.LOCOMOTIVE


def create_standard_train_deck() -> list[TrainCard]:
    """Create a standard 110-card Ticket to Ride train deck.

    - 12 cards for each of the 8 standard colors (96 cards)
    - 14 Locomotive (Wild) cards
    """
    deck: list[TrainCard] = []
    for color in CardColor:
        if color != CardColor.LOCOMOTIVE:
            deck.extend([TrainCard(color=color) for _ in range(12)])
    deck.extend([TrainCard(color=CardColor.LOCOMOTIVE) for _ in range(14)])
    return deck
