"""Card representations for Ticket to Ride."""

from dataclasses import dataclass
from enum import Enum


class CardColor(str, Enum):
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
