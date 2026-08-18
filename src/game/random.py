"""Deterministic random number generation utilities for Game Core."""

import random
from typing import TypeVar

T = TypeVar("T")


class SeededRNG:
    """Isolated, deterministic RNG to ensure reproducible games."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def shuffle(self, items: list[T]) -> None:
        self.rng.shuffle(items)

    def choice(self, items: list[T]) -> T:
        return self.rng.choice(items)

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)
