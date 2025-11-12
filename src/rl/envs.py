"""Gymnasium and PettingZoo environments for Ticket to Ride RL training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rl.scoring import (
    POINTS_TABLE,
    calculate_longest_path,
    calculate_route_points,
    calculate_ticket_points,
    final_score,
    trains_used,
    unique_routes_from_graph,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "gymnasium is required to use the Ticket to Ride RL environments"
    ) from exc

try:
    from pettingzoo.utils import agent_selector
    from pettingzoo.utils.env import AECEnv
except ImportError:
    AECEnv = None  # type: ignore[assignment]


RouteTuple = Tuple[str, str, int, str]


@dataclass
class RewardConfig:
    """Configuration values used to compute intermediate rewards."""

    invalid_action_penalty: float = 10.0
    efficiency_weight: float = 0.5
    connectivity_bonus: float = 5.0
    ticket_weight: float = 0.05
    final_score_scale: float = 0.1


class OpponentPolicyProtocol:
    """Protocol implemented by opponent policies."""

    def reset(self, env: "SingleAgentTicketToRideEnv") -> None:  # pragma: no cover - interface
        """Called whenever the environment is reset."""

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        env: "SingleAgentTicketToRideEnv",
    ) -> Optional[int]:  # pragma: no cover - interface
        """Return the action index, or ``None`` to skip the move."""


class SingleAgentTicketToRideEnv(gym.Env[np.ndarray, int]):
    """Ticket to Ride environment where the learning agent faces a scripted opponent."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        graph,
        tickets_file: str,
        opponent_policy: Optional[OpponentPolicyProtocol] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.tickets_df = pd.read_csv(tickets_file)
        self.routes: List[RouteTuple] = unique_routes_from_graph(graph)
        self.num_routes = len(self.routes)
        self.route_pairs: List[Tuple[str, str]] = [tuple(sorted((r[0], r[1]))) for r in self.routes]
        self.pair_to_idx: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(self.route_pairs)
        }

        self.total_trains = 45
        self.opponent_policy = opponent_policy
        self.reward_config = reward_config or RewardConfig()

        # Observation: [agent_routes, opponent_routes, agent_trains, opponent_trains,
        #               agent_claims_norm, opponent_claims_norm]
        obs_len = 2 * self.num_routes + 4
        self._obs_shape = (obs_len,)

        self.action_space = spaces.Discrete(self.num_routes)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=np.float32,
        )

        self._reset_internal_state()

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_internal_state()
        if self.opponent_policy:
            self.opponent_policy.reset(self)
        return self._get_state(), {}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Cannot call step() on a finished episode. Reset the environment.")

        info = {}
        terminated = False
        truncated = False

        if not self._is_valid_action(action, self.agent_routes, self.trains_agent):
            reward = -self.reward_config.invalid_action_penalty
            terminated = True
            info["invalid_action"] = True
            self._done = True
            self._last_scores = (self._estimate_score(self.agent_routes), self._estimate_score(self.opponent_routes))
            return self._get_state(), reward, terminated, truncated, info

        reward = self._apply_agent_action(action)

        if self._episode_should_end():
            terminated = True
        else:
            opponent_reward = self._opponent_turn()
            info["opponent_reward"] = opponent_reward
            if self._episode_should_end():
                terminated = True

        if terminated:
            self._done = True
            agent_score = final_score(self.agent_routes, self.tickets_df)
            opponent_score = final_score(self.opponent_routes, self.tickets_df)
            self._last_scores = (agent_score, opponent_score)
            reward += self.reward_config.final_score_scale * (agent_score - opponent_score)
            info["agent_score"] = agent_score
            info["opponent_score"] = opponent_score

        return self._get_state(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_internal_state(self) -> None:
        self.agent_routes: List[RouteTuple] = []
        self.opponent_routes: List[RouteTuple] = []
        self.claimed_pairs: Dict[Tuple[str, str], str] = {}
        self.trains_agent = self.total_trains
        self.trains_opponent = self.total_trains
        self._done = False
        self._last_scores = (0, 0)

    def _episode_should_end(self) -> bool:
        no_actions_agent = len(self.get_valid_actions(for_agent=True)) == 0
        no_actions_opponent = len(self.get_valid_actions(for_agent=False)) == 0
        agent_out_of_trains = self.trains_agent <= 0
        opponent_out_of_trains = self.trains_opponent <= 0
        return no_actions_agent or no_actions_opponent or agent_out_of_trains or opponent_out_of_trains

    def _apply_agent_action(self, action: int) -> float:
        route = self.routes[action]
        base_points = float(POINTS_TABLE[route[2]])
        efficiency = POINTS_TABLE[route[2]] / route[2]
        self._claim_route(action, self.agent_routes, owner="agent")
        ticket_points = calculate_ticket_points(self.agent_routes, self.tickets_df)
        connectivity = calculate_longest_path(self.agent_routes)
        return (
            base_points
            + self.reward_config.efficiency_weight * efficiency
            + self.reward_config.ticket_weight * ticket_points
            + self.reward_config.connectivity_bonus * (connectivity / (self.total_trains or 1))
        )

    def _opponent_turn(self) -> float:
        if not self.opponent_policy:
            return 0.0

        opponent_state = self._get_state(perspective="opponent")
        valid_actions = self.get_valid_actions(for_agent=False)
        if not valid_actions:
            return 0.0

        action = self.opponent_policy.select_action(opponent_state, valid_actions, self)
        if action is None or action not in valid_actions:
            return 0.0

        self._claim_route(action, self.opponent_routes, owner="opponent")

        route = self.routes[action]
        reward = float(POINTS_TABLE[route[2]])
        efficiency = POINTS_TABLE[route[2]] / route[2]
        ticket_points = calculate_ticket_points(self.opponent_routes, self.tickets_df)
        connectivity = calculate_longest_path(self.opponent_routes)
        return (
            reward
            + self.reward_config.efficiency_weight * efficiency
            + self.reward_config.ticket_weight * ticket_points
            + self.reward_config.connectivity_bonus * (connectivity / (self.total_trains or 1))
        )

    def _claim_route(
        self,
        action: int,
        target_routes: List[RouteTuple],
        *,
        owner: str,
    ) -> float:
        route = self.routes[action]
        pair = self.route_pairs[action]
        self.claimed_pairs[pair] = owner
        target_routes.append(route)
        if owner == "agent":
            self.trains_agent -= route[2]
        else:
            self.trains_opponent -= route[2]

        return 0.0

    def _is_valid_action(
        self,
        action: int,
        player_routes: List[RouteTuple],
        trains_remaining: int,
    ) -> bool:
        if action < 0 or action >= self.num_routes:
            return False
        route = self.routes[action]
        pair = self.route_pairs[action]
        if pair in self.claimed_pairs:
            return False
        if trains_remaining < route[2]:
            return False
        return True

    def get_valid_actions(self, *, for_agent: bool) -> List[int]:
        trains = self.trains_agent if for_agent else self.trains_opponent
        return [
            idx
            for idx in range(self.num_routes)
            if self._is_valid_action(idx, self.agent_routes if for_agent else self.opponent_routes, trains)
        ]

    def _get_state(self, perspective: str = "agent") -> np.ndarray:
        agent_binary = np.zeros(self.num_routes, dtype=np.float32)
        opponent_binary = np.zeros(self.num_routes, dtype=np.float32)
        for idx, pair in enumerate(self.route_pairs):
            owner = self.claimed_pairs.get(pair)
            if owner == "agent":
                agent_binary[idx] = 1.0
            elif owner == "opponent":
                opponent_binary[idx] = 1.0

        if perspective == "opponent":
            agent_binary, opponent_binary = opponent_binary, agent_binary
            trains_current = self.trains_opponent
            trains_other = self.trains_agent
        else:
            trains_current = self.trains_agent
            trains_other = self.trains_opponent

        obs = np.concatenate(
            [
                agent_binary,
                opponent_binary,
                np.array(
                    [
                        trains_current / self.total_trains,
                        trains_other / self.total_trains,
                        agent_binary.mean() if self.num_routes else 0.0,
                        opponent_binary.mean() if self.num_routes else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def _estimate_score(self, routes: Iterable[RouteTuple]) -> int:
        return final_score(list(routes), self.tickets_df)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render(self):  # pragma: no cover - rendering is optional
        agent_score = final_score(self.agent_routes, self.tickets_df)
        opponent_score = final_score(self.opponent_routes, self.tickets_df)
        print("Agent routes:")
        for route in self.agent_routes:
            print(f"  {route[0]} -> {route[1]} (len={route[2]})")
        print(f"Agent score: {agent_score}\n")
        print("Opponent routes:")
        for route in self.opponent_routes:
            print(f"  {route[0]} -> {route[1]} (len={route[2]})")
        print(f"Opponent score: {opponent_score}\n")

    def close(self):  # pragma: no cover - nothing to close
        pass


# ----------------------------------------------------------------------
# PettingZoo multi-agent environment
# ----------------------------------------------------------------------
class TicketToRideSelfPlayEnv(AECEnv):
    """PettingZoo environment for self-play training.

    Two agents alternate in claiming routes. Episodes end when no valid moves
    remain or a player runs out of trains.
    """

    metadata = {"name": "TicketToRideSelfPlay-v0", "render_modes": ["human"], "is_parallelizable": False}

    def __init__(self, graph, tickets_file: str, reward_config: Optional[RewardConfig] = None) -> None:
        if AECEnv is None:  # pragma: no cover - optional dependency guard
            raise ImportError("pettingzoo is required to use TicketToRideSelfPlayEnv")

        super().__init__()
        self.graph = graph
        self.tickets_df = pd.read_csv(tickets_file)
        self.routes: List[RouteTuple] = unique_routes_from_graph(graph)
        self.num_routes = len(self.routes)
        self.route_pairs: List[Tuple[str, str]] = [tuple(sorted((r[0], r[1]))) for r in self.routes]
        self.pair_to_idx: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(self.route_pairs)
        }
        self.total_trains = 45
        self.reward_config = reward_config or RewardConfig()

        obs_len = 2 * self.num_routes + 4
        self._obs_shape = (obs_len,)

        self.agents: List[str] = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.action_spaces = {
            agent: spaces.Discrete(self.num_routes) for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=np.float32)
            for agent in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self._reset_state()

    # ------------------------------------------------------------------
    def observation_space(self, agent: str):  # pragma: no cover - API compatibility
        return self.observation_spaces[agent]

    def action_space(self, agent: str):  # pragma: no cover - API compatibility
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self._reset_state()
        self.agent_selection = self._agent_selector.reset()
        return

    def _reset_state(self) -> None:
        self.claimed_pairs: Dict[Tuple[str, str], str] = {}
        self.routes_by_agent: Dict[str, List[RouteTuple]] = {agent: [] for agent in self.possible_agents}
        self.trains_remaining: Dict[str, int] = {agent: self.total_trains for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

    def observe(self, agent: str) -> np.ndarray:
        opponent = self._opponent_of(agent)
        agent_binary = np.zeros(self.num_routes, dtype=np.float32)
        opponent_binary = np.zeros(self.num_routes, dtype=np.float32)
        for idx, pair in enumerate(self.route_pairs):
            owner = self.claimed_pairs.get(pair)
            if owner == agent:
                agent_binary[idx] = 1.0
            elif owner == opponent:
                opponent_binary[idx] = 1.0

        obs = np.concatenate(
            [
                agent_binary,
                opponent_binary,
                np.array(
                    [
                        self.trains_remaining[agent] / self.total_trains,
                        self.trains_remaining[opponent] / self.total_trains,
                        agent_binary.mean() if self.num_routes else 0.0,
                        opponent_binary.mean() if self.num_routes else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def step(self, action: Optional[int]):
        if self.agents is None or len(self.agents) == 0:
            return

        agent = self.agent_selection
        opponent = self._opponent_of(agent)

        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step(action)
            return

        valid_actions = self._valid_actions(agent)
        if action is None or action not in valid_actions:
            self.rewards[agent] = -self.reward_config.invalid_action_penalty
            for ag in self.agents:
                self.terminations[ag] = True
            self._apply_final_scores()
            self._accumulate_rewards()
            self.agents = []
            return

        self._claim_route(agent, action)
        self.rewards[agent] = self._step_reward(agent, action)

        if self._game_should_stop():
            self._apply_final_scores()
            for ag in self.agents:
                self.terminations[ag] = True
            self._accumulate_rewards()
            self.agents = []
            return

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def _step_reward(self, agent: str, action: int) -> float:
        route = self.routes[action]
        reward = float(POINTS_TABLE[route[2]])
        efficiency = POINTS_TABLE[route[2]] / route[2]
        reward += self.reward_config.efficiency_weight * efficiency
        ticket_points = calculate_ticket_points(self.routes_by_agent[agent], self.tickets_df)
        reward += self.reward_config.ticket_weight * ticket_points
        longest = calculate_longest_path(self.routes_by_agent[agent])
        reward += self.reward_config.connectivity_bonus * (longest / (self.total_trains or 1))
        return reward

    def _claim_route(self, agent: str, action: int) -> None:
        route = self.routes[action]
        pair = self.route_pairs[action]
        self.claimed_pairs[pair] = agent
        self.routes_by_agent[agent].append(route)
        self.trains_remaining[agent] -= route[2]

    def _valid_actions(self, agent: str) -> List[int]:
        trains_left = self.trains_remaining[agent]
        return [
            idx
            for idx in range(self.num_routes)
            if self._is_claimable(idx, trains_left)
        ]

    def _is_claimable(self, idx: int, trains_left: int) -> bool:
        pair = self.route_pairs[idx]
        route = self.routes[idx]
        return pair not in self.claimed_pairs and trains_left >= route[2]

    def _game_should_stop(self) -> bool:
        if any(trains <= 0 for trains in self.trains_remaining.values()):
            return True
        return all(len(self._valid_actions(agent)) == 0 for agent in self.agents)

    def _apply_final_scores(self) -> None:
        if not self.agents:
            return
        score_agent0 = final_score(self.routes_by_agent.get("player_0", []), self.tickets_df)
        score_agent1 = final_score(self.routes_by_agent.get("player_1", []), self.tickets_df)
        diff = score_agent0 - score_agent1
        self.rewards.setdefault("player_0", 0.0)
        self.rewards.setdefault("player_1", 0.0)
        self.rewards["player_0"] += self.reward_config.final_score_scale * diff
        self.rewards["player_1"] -= self.reward_config.final_score_scale * diff
        self.infos["player_0"]["score"] = score_agent0
        self.infos["player_1"]["score"] = score_agent1

    def render(self):  # pragma: no cover - optional
        if not self.agents:
            print("Game over")
            return
        for agent in self.possible_agents:
            score = final_score(self.routes_by_agent[agent], self.tickets_df)
            print(f"{agent} score: {score}")

    def close(self):  # pragma: no cover - optional
        pass

    def _opponent_of(self, agent: str) -> str:
        return "player_1" if agent == "player_0" else "player_0"


__all__ = [
    "RewardConfig",
    "SingleAgentTicketToRideEnv",
    "TicketToRideSelfPlayEnv",
]
