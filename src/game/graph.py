"""Graph connectivity and longest continuous trail algorithms for Ticket to Ride."""

from collections import defaultdict
from src.game.route import Route
from src.game.ticket import DestinationTicket


class DisjointSet:
    """Fast Disjoint Set Union (DSU) with path compression and union by rank."""

    __slots__ = ("parent", "rank")

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        root = i
        while root != self.parent[root]:
            root = self.parent[root]
        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

    def connected(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)


def check_ticket_completed(player_routes: list[Route], ticket: DestinationTicket) -> bool:
    """Check if city_a and city_b are connected by player_routes using fast DSU."""
    if not player_routes:
        return False

    city_to_idx: dict[str, int] = {}
    idx = 0
    for r in player_routes:
        if r.city_a not in city_to_idx:
            city_to_idx[r.city_a] = idx
            idx += 1
        if r.city_b not in city_to_idx:
            city_to_idx[r.city_b] = idx
            idx += 1

    if ticket.city_a not in city_to_idx or ticket.city_b not in city_to_idx:
        return False

    dsu = DisjointSet(idx)
    for r in player_routes:
        dsu.union(city_to_idx[r.city_a], city_to_idx[r.city_b])

    return dsu.connected(city_to_idx[ticket.city_a], city_to_idx[ticket.city_b])


def check_tickets_completed_batch(
    player_routes: list[Route], tickets: list[DestinationTicket]
) -> dict[str, bool]:
    """Check completion for a batch of tickets in a single DSU pass."""
    if not tickets:
        return {}
    if not player_routes:
        return {t.id: False for t in tickets}

    city_to_idx: dict[str, int] = {}
    idx = 0
    for r in player_routes:
        if r.city_a not in city_to_idx:
            city_to_idx[r.city_a] = idx
            idx += 1
        if r.city_b not in city_to_idx:
            city_to_idx[r.city_b] = idx
            idx += 1

    dsu = DisjointSet(idx)
    for r in player_routes:
        dsu.union(city_to_idx[r.city_a], city_to_idx[r.city_b])

    result = {}
    for t in tickets:
        if t.city_a not in city_to_idx or t.city_b not in city_to_idx:
            result[t.id] = False
        else:
            result[t.id] = dsu.connected(city_to_idx[t.city_a], city_to_idx[t.city_b])
    return result


def compute_longest_continuous_path(player_routes: list[Route]) -> int:
    """Compute the length of the longest continuous train trail (each route visited at most once)."""
    if not player_routes:
        return 0

    # Build multigraph adjacency: city -> list of (neighbor_city, route_id, route_length)
    adj: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    cities = set()
    for r in player_routes:
        adj[r.city_a].append((r.city_b, r.id, r.length))
        adj[r.city_b].append((r.city_a, r.id, r.length))
        cities.add(r.city_a)
        cities.add(r.city_b)

    max_length = 0

    def dfs(current_city: str, current_len: int, visited_edges: set[str]) -> None:
        nonlocal max_length
        if current_len > max_length:
            max_length = current_len

        for neighbor, edge_id, length in adj[current_city]:
            if edge_id not in visited_edges:
                visited_edges.add(edge_id)
                dfs(neighbor, current_len + length, visited_edges)
                visited_edges.remove(edge_id)

    # Explore trails starting from every node in the player's network
    for start_city in cities:
        dfs(start_city, 0, set())

    return max_length
