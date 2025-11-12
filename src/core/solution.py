"""Solution dataclass shared across modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.routes import RouteTuple


@dataclass
class Solution:
    routes: List[RouteTuple]
    route_points: int
    longest_path_length: int
    longest_bonus: int
    ticket_points: int
    total_score: int
    trains_used: int
    computation_time: float
    algorithm: str = ""
    episodes: int = 0
