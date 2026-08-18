"""Official rules and route scoring tables."""


ROUTE_POINTS_BY_LENGTH: dict[int, int] = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    5: 10,
    6: 15,
}


class GameRules:
    """Rules and invariant validations for Ticket to Ride."""

    INITIAL_TRAINS_PER_PLAYER = 45
    INITIAL_CARDS_PER_PLAYER = 4
    VISIBLE_CARDS_COUNT = 5
    MIN_TICKETS_KEEP_START = 2
    MIN_TICKETS_KEEP_INGAME = 1

    @staticmethod
    def points_for_route_length(length: int) -> int:
        return ROUTE_POINTS_BY_LENGTH.get(length, 0)
