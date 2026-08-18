"""Board and City representations."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class City:
    id: str
    name: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class Board:
    cities: dict[str, City] = field(default_factory=dict)
    routes: list["Route"] = field(default_factory=list)  # type: ignore # noqa: F821

    def add_city(self, city: City) -> None:
        self.cities[city.id] = city

    def get_city(self, city_id: str) -> City | None:
        return self.cities.get(city_id)
