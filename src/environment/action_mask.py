"""Action masking implementation to ensure RL policies only sample legal actions."""

import numpy as np

from src.environment.action_space import TICKET_SUBSET_INDICES, DiscreteActionSpace
from src.game.action import Action, ActionType
from src.game.ticket import DestinationTicket


class ActionMasker:
    """Computes a boolean mask over the discrete action space corresponding to legal moves."""

    def __init__(self, action_space: DiscreteActionSpace) -> None:
        self.action_space = action_space

    def compute_mask(
        self,
        valid_actions: list[Action],
        pending_tickets: list[DestinationTicket] | None = None,
    ) -> np.ndarray:
        """Return boolean mask of shape (n,) where True indicates a valid action."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        if not valid_actions:
            if self.action_space.n > 0:
                mask[0] = True
            return mask

        # Map of pending tickets by id to their local slot index 0, 1, 2
        pending_id_to_slot: dict[str, int] = {}
        if pending_tickets:
            for slot, t in enumerate(pending_tickets):
                pending_id_to_slot[t.id] = slot

        for act in valid_actions:
            if act.action_type == ActionType.KEEP_TICKETS:
                if act.ticket_ids is not None and pending_id_to_slot:
                    # Convert ticket IDs to subset tuple of slot indices
                    slots = tuple(
                        sorted(
                            pending_id_to_slot[tid]
                            for tid in act.ticket_ids
                            if tid in pending_id_to_slot
                        )
                    )
                    if slots in TICKET_SUBSET_INDICES:
                        subset_idx = TICKET_SUBSET_INDICES.index(slots)
                        mask[7 + subset_idx] = True
            else:
                act_id = self.action_space.to_id(act)
                if act_id is not None:
                    mask[act_id] = True

        # Safety: if no valid actions mapped, fallback to first action or draw hidden card
        if not np.any(mask) and self.action_space.n > 0:
            mask[0] = True

        return mask
