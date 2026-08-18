"""Game engine orchestration class."""


from src.game.action import Action
from src.game.board import Board
from src.game.player import Player
from src.game.random import SeededRNG
from src.game.rules import GameRules
from src.game.state import GameState


class Game:
    """Core game engine representing Ticket to Ride.

    Independent of RL, UI, or frameworks.
    """

    def __init__(
        self,
        board: Board | None = None,
        num_players: int = 2,
        seed: int = 42,
    ) -> None:
        self.board = board or Board()
        self.num_players = num_players
        self.rng = SeededRNG(seed)
        self.rules = GameRules()
        self.state = GameState()

    def reset(self, seed: int | None = None) -> GameState:
        """Reset the game state deterministically."""
        if seed is not None:
            self.rng.seed(seed)

        self.state = GameState(
            players=[
                Player(id=f"player_{i}", name=f"Player {i+1}")
                for i in range(self.num_players)
            ],
            current_player_index=0,
            turn_number=1,
        )
        return self.state

    def valid_actions(self) -> list[Action]:
        """Compute the list of valid actions for the current player."""
        # Skeleton implementation - extended in Phase 1
        return []

    def step(self, action: Action) -> GameState:
        """Execute an action, update game state, and advance turns."""
        # Skeleton implementation - extended in Phase 1
        return self.state
