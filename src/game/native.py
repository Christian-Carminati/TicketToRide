"""Native Rust Game Engine bridge and high-speed simulation interfaces."""

from typing import Any, cast

import ttr_core

from src.game.action import Action


class NativeGame:
    """High-speed native Rust game engine wrapper for Ticket to Ride."""

    def __init__(self, seed: int = 42) -> None:
        self.py_game = ttr_core.PyGame(seed)
        self.seed = seed

    def reset(self, seed: int | None = None) -> None:
        s = seed if seed is not None else self.seed
        self.py_game.reset(s)

    @property
    def is_game_over(self) -> bool:
        return cast(bool, self.py_game.is_game_over())

    @property
    def current_player_index(self) -> int:
        return cast(int, self.py_game.current_player_index())

    @property
    def turn_number(self) -> int:
        return cast(int, self.py_game.turn_number())

    @property
    def turn_state(self) -> str:
        return cast(str, self.py_game.turn_state())

    @property
    def scores(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.py_game.scores())

    @property
    def winner_id(self) -> int:
        return cast(int, self.py_game.winner_id())

    @property
    def trains_remaining(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.py_game.trains_remaining())

    def player_hand(self, player_idx: int) -> list[int]:
        return cast(list[int], self.py_game.player_hand(player_idx))

    def player_tickets(self, player_idx: int) -> list[int]:
        return cast(list[int], self.py_game.player_tickets(player_idx))

    def is_ticket_completed(self, player_idx: int, ticket_id: int) -> bool:
        return cast(bool, self.py_game.is_ticket_completed(player_idx, ticket_id))

    def valid_actions_raw(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.py_game.valid_actions())

    def valid_actions(self) -> list[Action]:
        raw = self.valid_actions_raw()
        return [Action.from_dict(d) for d in raw]

    def step(self, action: Action | dict[str, Any]) -> None:
        if isinstance(action, Action):
            self.py_game.step_dict(action.to_dict())
        else:
            self.py_game.step_dict(action)

    def clone(self) -> "NativeGame":
        cloned = NativeGame.__new__(NativeGame)
        cloned.py_game = self.py_game.clone_game()
        cloned.seed = self.seed
        return cloned


class NativeVectorEnv:
    """Multi-threaded native batch environment manager running at millions of steps/sec."""

    def __init__(self, num_envs: int = 64, base_seed: int = 42) -> None:
        self.py_vec = ttr_core.PyVectorEnv(num_envs, base_seed)
        self.num_envs = num_envs

    def reset_all(self, base_seed: int = 42) -> None:
        self.py_vec.reset_all(base_seed)

    def step_batch_sim(self, num_steps_per_env: int = 100) -> int:
        return cast(int, self.py_vec.step_batch_sim(num_steps_per_env))
