"""Action masking implementation to ensure RL policies only sample legal actions."""


import numpy as np

from src.environment.action_space import DiscreteActionSpace
from src.game.action import Action


class ActionMasker:
    """Computes a boolean mask over the discrete action space."""

    def __init__(self, action_space: DiscreteActionSpace) -> None:
        self.action_space = action_space

    def compute_mask(self, valid_actions: list[Action]) -> np.ndarray:
        """Return boolean mask of shape (n,) where True indicates a valid action."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        for act in valid_actions:
            act_id = self.action_space.to_id(act)
            if act_id is not None:
                mask[act_id] = True
        # Safety: if no valid actions mapped, allow no-op or first action
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True
        return mask
