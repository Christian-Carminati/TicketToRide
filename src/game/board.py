"""Board and City representations."""

from dataclasses import dataclass, field

from src.game.route import Route


@dataclass(frozen=True)
class City:
    id: str
    name: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class Board:
    cities: dict[str, City] = field(default_factory=dict)
    routes: list[Route] = field(default_factory=list)

    def add_city(self, city: City) -> None:
        self.cities[city.name] = city

    def get_city(self, name_or_id: str) -> City | None:
        if name_or_id in self.cities:
            return self.cities[name_or_id]
        for c in self.cities.values():
            if c.id == name_or_id:
                return c
        return None

    def get_route(self, route_id: str) -> Route | None:
        for r in self.routes:
            if r.id == route_id:
                return r
        return None

    def get_routes_between(self, city_a: str, city_b: str) -> list[Route]:
        results = []
        for r in self.routes:
            if (r.city_a == city_a and r.city_b == city_b) or (
                r.city_a == city_b and r.city_b == city_a
            ):
                results.append(r)
        return results

    def get_adjacent_cities(self, city_name_or_id: str) -> list[str]:
        city = self.get_city(city_name_or_id)
        name = city.name if city else city_name_or_id
        adj = set()
        for r in self.routes:
            if r.city_a == name:
                adj.add(r.city_b)
            elif r.city_b == name:
                adj.add(r.city_a)
        return sorted(adj)
