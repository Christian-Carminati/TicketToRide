"""Observation encoders for converting GameState to vector representations."""

from abc import ABC, abstractmethod

import numpy as np

from src.game.state import GameState


class BaseObservationEncoder(ABC):
    """Abstract base class for versioned observation encoders."""

    @abstractmethod
    def encode(self, state: GameState, player_index: int) -> np.ndarray:
        """Encode the game state from the perspective of player_index."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        """Shape of the resulting observation array."""


class ObservationV1(BaseObservationEncoder):
    """Version 1 Observation Encoder (Flattened feature vector)."""

    def __init__(self, feature_dim: int = 128) -> None:
        self._feature_dim = feature_dim

    @property
    def observation_shape(self) -> tuple:
        return (self._feature_dim,)

    def encode(self, state: GameState, player_index: int) -> np.ndarray:
        # Skeleton vector - implemented in Phase 3
        return np.zeros(self._feature_dim, dtype=np.float32)
