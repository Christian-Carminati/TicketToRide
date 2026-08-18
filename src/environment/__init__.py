"""Environment package: Gymnasium wrapper, observation encoding, action masking, and reward shaping."""

from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.env import TicketToRideEnv
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.environment.reward import BaseRewardCalculator, DefaultRewardCalculator

__all__ = [
    "ActionMasker",
    "BaseObservationEncoder",
    "BaseRewardCalculator",
    "DefaultRewardCalculator",
    "DiscreteActionSpace",
    "ObservationV1",
    "TicketToRideEnv",
]
