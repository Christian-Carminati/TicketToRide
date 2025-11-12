"""
Common constants for Ticket to Ride game logic.

This module defines shared constants used across the codebase,
including the total number of trains available to each player
and the points table for route lengths.
"""
from __future__ import annotations

from dataclasses import dataclass

# Total number of train pieces each player starts with
TOTAL_TRAINS: int = 45


@dataclass(frozen=True)
class PointsTable:
    """
    Points awarded for claiming routes of specific lengths.
    
    According to Ticket to Ride rules:
    - 1 train: 1 point
    - 2 trains: 2 points
    - 3 trains: 4 points
    - 4 trains: 7 points
    - 5 trains: 10 points
    - 6 trains: 15 points
    
    Routes longer than 6 trains are not standard in Ticket to Ride USA.
    """

    one: int = 1
    two: int = 2
    three: int = 4
    four: int = 7
    five: int = 10
    six: int = 15

    def __getitem__(self, length: int) -> int:
        """
        Get points for a route of given length.
        
        Args:
            length: Number of trains in the route (1-6)
            
        Returns:
            Points awarded for that route length, or 0 if invalid
        """
        mapping = {
            1: self.one,
            2: self.two,
            3: self.three,
            4: self.four,
            5: self.five,
            6: self.six,
        }
        return mapping.get(length, 0)


POINTS_TABLE = PointsTable()
