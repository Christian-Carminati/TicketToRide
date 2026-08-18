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
    double_route_pair_id: str | None = None
    claimed_by: str | None = None  # Player ID if claimed

    @property
    def is_claimed(self) -> bool:
        return self.claimed_by is not None

    @property
    def is_double_route(self) -> bool:
        return self.double_route_pair_id is not None
