"""Board and City representations."""

from collections import defaultdict
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
    _route_map: dict[str, Route] = field(default_factory=dict, init=False, repr=False)
    _adjacent_routes: dict[str, list[Route]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _routes_between: dict[tuple[str, str], list[Route]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.rebuild_indexes()

    def rebuild_indexes(self) -> None:
        """Rebuild O(1) route lookup caches."""
        self._route_map = {r.id: r for r in self.routes}
        self._adjacent_routes = defaultdict(list)
        self._routes_between = defaultdict(list)

        for r in self.routes:
            self._adjacent_routes[r.city_a].append(r)
            self._adjacent_routes[r.city_b].append(r)
            pair_key = (min(r.city_a, r.city_b), max(r.city_a, r.city_b))
            self._routes_between[pair_key].append(r)

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
        if not self._route_map and self.routes:
            self.rebuild_indexes()
        return self._route_map.get(route_id)

    def get_routes_between(self, city_a: str, city_b: str) -> list[Route]:
        if not self._routes_between and self.routes:
            self.rebuild_indexes()
        pair_key = (min(city_a, city_b), max(city_a, city_b))
        return self._routes_between.get(pair_key, [])

    def get_adjacent_routes(self, city_name_or_id: str) -> list[Route]:
        city = self.get_city(city_name_or_id)
        name = city.name if city else city_name_or_id
        if not self._adjacent_routes and self.routes:
            self.rebuild_indexes()
        return self._adjacent_routes.get(name, [])

    def get_adjacent_cities(self, city_name_or_id: str) -> list[str]:
        routes = self.get_adjacent_routes(city_name_or_id)
        city = self.get_city(city_name_or_id)
        name = city.name if city else city_name_or_id
        adj = set()
        for r in routes:
            adj.add(r.city_b if r.city_a == name else r.city_a)
        return sorted(adj)

    def clone(self) -> "Board":
        """Fast in-memory clone of Board with deeply copied routes while reusing immutable City objects."""
        cloned_routes = [
            Route(
                id=r.id,
                city_a=r.city_a,
                city_b=r.city_b,
                length=r.length,
                color=r.color,
                double_route_pair_id=r.double_route_pair_id,
                claimed_by=r.claimed_by,
            )
            for r in self.routes
        ]
        b = Board.__new__(Board)
        b.cities = self.cities
        b.routes = cloned_routes
        b._route_map = {r.id: r for r in cloned_routes}
        b._adjacent_routes = self._adjacent_routes
        b._routes_between = self._routes_between
        return b
