"""Action space mapping between discrete integer actions and explicit game Actions."""


from src.game.action import Action, ActionType


class DiscreteActionSpace:
    """Bi-directional mapping between integer action IDs and domain Actions."""

    def __init__(self) -> None:
        self._action_to_id: dict[Action, int] = {}
        self._id_to_action: dict[int, Action] = {}
        self._build_default_space()

    def _build_default_space(self) -> None:
        # Skeleton default action mappings
        self._id_to_action[0] = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
        self._action_to_id[self._id_to_action[0]] = 0

    @property
    def n(self) -> int:
        return len(self._id_to_action)

    def to_action(self, action_id: int) -> Action:
        return self._id_to_action[action_id]

    def to_id(self, action: Action) -> int | None:
        return self._action_to_id.get(action)
