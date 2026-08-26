"""Elo rating calculations for multi-agent tournament ranking."""


class EloSystem:
    """Computes and updates Elo ratings for agents."""

    def __init__(self, initial_rating: float = 1200.0, k_factor: float = 32.0) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: dict[str, float] = {}

    def get_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)))

    def update(self, agent_a: str, agent_b: str, score_a: float) -> None:
        """Update Elo after a match. score_a is 1.0 (win), 0.5 (draw), or 0.0 (loss)."""
        ra = self.get_rating(agent_a)
        rb = self.get_rating(agent_b)

        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        score_b = 1.0 - score_a

        self.ratings[agent_a] = ra + self.k_factor * (score_a - ea)
        self.ratings[agent_b] = rb + self.k_factor * (score_b - eb)
