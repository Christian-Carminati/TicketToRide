"""Historical policy pool management for self-play training."""

import random
from dataclasses import dataclass, field


@dataclass
class PolicyPool:
    """Maintains a historical pool of policy checkpoints for matchmaking."""

    checkpoints: list[str] = field(default_factory=list)
    max_size: int = 50

    def add_checkpoint(self, checkpoint_path: str) -> None:
        if len(self.checkpoints) >= self.max_size:
            self.checkpoints.pop(0)
        self.checkpoints.append(checkpoint_path)

    def sample_opponent(self) -> str | None:
        if not self.checkpoints:
            return None
        return random.choice(self.checkpoints)
