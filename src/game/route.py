"""Route representation between two cities."""

from dataclasses import dataclass

from src.game.card import CardColor


@dataclass
class Route:
    id: str
    city_a: str
    city_b: str
    length: int
    color: CardColor | None = None  # None indicates Gray / any color
    claimed_by: str | None = None  # Player ID if claimed

    @property
    def is_claimed(self) -> bool:
        return self.claimed_by is not None
