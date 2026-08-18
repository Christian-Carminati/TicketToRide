"""Graph connectivity and longest continuous trail algorithms for Ticket to Ride."""

from collections import defaultdict, deque

from src.game.route import Route
from src.game.ticket import DestinationTicket


def check_ticket_completed(player_routes: list[Route], ticket: DestinationTicket) -> bool:
    """Check if city_a and city_b are connected by player_routes using BFS."""
    if not player_routes:
        return False

    adj: dict[str, set[str]] = defaultdict(set)
    for r in player_routes:
        adj[r.city_a].add(r.city_b)
        adj[r.city_b].add(r.city_a)

    if ticket.city_a not in adj or ticket.city_b not in adj:
        return False

    visited: set[str] = set()
    queue: deque[str] = deque([ticket.city_a])
    visited.add(ticket.city_a)

    while queue:
        current = queue.popleft()
        if current == ticket.city_b:
            return True
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


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
        max_length = max(max_length, current_len)

        for neighbor, edge_id, length in adj[current_city]:
            if edge_id not in visited_edges:
                visited_edges.add(edge_id)
                dfs(neighbor, current_len + length, visited_edges)
                visited_edges.remove(edge_id)

    # Explore trails starting from every node in the player's network
    for start_city in cities:
        dfs(start_city, 0, set())

    return max_length
