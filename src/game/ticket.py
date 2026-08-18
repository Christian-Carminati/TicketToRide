"""Destination ticket representation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationTicket:
    id: str
    city_a: str
    city_b: str
    points: int
