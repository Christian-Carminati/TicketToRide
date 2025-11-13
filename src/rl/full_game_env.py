"""
Full game RL environment with complete card mechanics.

This module provides RL environments that use the complete Game class,
including card management, deck, face-up cards, and proper route claiming validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from game import Color, Game, Player, Route
from rl.scoring import (
    POINTS_TABLE,
    calculate_longest_path,
    calculate_route_points,
    calculate_ticket_points,
    final_score,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required to use the Ticket to Ride RL environments"
    ) from exc

try:
    from pettingzoo.utils import agent_selector
    from pettingzoo.utils.env import AECEnv
except ImportError:
    AECEnv = None  # type: ignore[assignment]


@dataclass
class RewardConfig:
    """Configuration values used to compute intermediate rewards."""

    invalid_action_penalty: float = 10.0
    efficiency_weight: float = 0.5
    connectivity_bonus: float = 5.0
    ticket_weight: float = 0.05
    final_score_scale: float = 0.1
    card_draw_bonus: float = 0.1  # Small bonus when draws increase options
    card_draw_penalty: float = 0.05  # Penalty when draws do not improve options


def get_color_mapping() -> Dict[Color, int]:
    """Shared color to index mapping for observations."""
    return {
        Color.RED: 0,
        Color.BLUE: 1,
        Color.GREEN: 2,
        Color.YELLOW: 3,
        Color.ORANGE: 4,
        Color.BLACK: 5,
        Color.SILVER: 6,
        Color.PURPLE: 7,
        Color.JOLLY: 8,
    }


def find_claim_color_for_player(route: Route, player: Player) -> Optional[Color]:
    """
    Determine a valid color a player can use to claim a route.
    Returns None if the player cannot claim the route with any valid color.
    """
    if route.color == Color.GREY:
        for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.BLACK]:
            if player.hand.can_claim_route(route, color):
                return color
        return None
    return route.color if player.hand.can_claim_route(route, route.color) else None


def compute_valid_actions(
    *,
    routes: List[Route],
    player: Player,
    num_routes: int,
    num_card_actions: int,
) -> List[int]:
    """Shared valid action computation for both envs."""
    valid_actions: List[int] = []

    # Valid route claims
    for idx, route in enumerate(routes):
        if route.owner is not None:
            continue
        if player.trains_left < route.length:
            continue
        if find_claim_color_for_player(route, player) is not None:
            valid_actions.append(idx)

    # Card drawing actions (always available)
    valid_actions.extend(range(num_routes, num_routes + num_card_actions))

    # Ticket drawing (always available)
    valid_actions.append(num_routes + num_card_actions)

    return valid_actions


def count_claimable_routes(routes: List[Route], player: Player) -> int:
    """Count how many routes the given player can currently claim."""
    claimable = 0
    for route in routes:
        if route.owner is not None:
            continue
        if player.trains_left < route.length:
            continue
        if find_claim_color_for_player(route, player) is not None:
            claimable += 1
    return claimable


def build_observation(
    *,
    game: Game,
    routes: List[Route],
    route_to_idx: Dict[Tuple[str, str, Color], int],
    color_to_idx: Dict[Color, int],
    num_colors: int,
    agent_idx: int,
) -> np.ndarray:
    """Create the flat observation vector from a player perspective."""
    opponent_idx = 1 - agent_idx
    player = game.players[agent_idx]
    opponent = game.players[opponent_idx]

    # Agent routes (binary vector)
    agent_routes = np.zeros(len(routes), dtype=np.float32)
    for route in player.routes:
        key = (route.city1, route.city2, route.color)
        if key in route_to_idx:
            agent_routes[route_to_idx[key]] = 1.0

    # Opponent routes
    opponent_routes = np.zeros(len(routes), dtype=np.float32)
    for route in opponent.routes:
        key = (route.city1, route.city2, route.color)
        if key in route_to_idx:
            opponent_routes[route_to_idx[key]] = 1.0

    # Trains
    agent_trains = player.trains_left / 45.0
    opponent_trains = opponent.trains_left / 45.0

    # Player cards (normalized)
    agent_cards = np.zeros(num_colors, dtype=np.float32)
    for color, count in player.hand.cards.items():
        if color in color_to_idx:
            idx = color_to_idx[color]
            agent_cards[idx] = min(count / 12.0, 1.0)

    # Opponent cards (partial info: only total, uniformly distributed)
    opponent_total_cards = sum(opponent.hand.cards.values())
    opponent_cards = np.full(num_colors, opponent_total_cards / (12.0 * num_colors), dtype=np.float32)

    # Face-up cards (one-hot per position)
    face_up_encoded = np.zeros(5 * num_colors, dtype=np.float32)
    for i, card in enumerate(game.game_state.face_up_cards[:5]):
        if card and card in color_to_idx:
            color_idx = color_to_idx[card]
            face_up_encoded[i * num_colors + color_idx] = 1.0

    # Deck size (normalized by initial deck size)
    total_deck_cards = sum(game.game_state.deck.cards.values())
    deck_size = min(total_deck_cards / 110.0, 1.0)

    obs = np.concatenate(
        [
            agent_routes,
            opponent_routes,
            np.array([agent_trains, opponent_trains], dtype=np.float32),
            agent_cards,
            opponent_cards,
            face_up_encoded,
            np.array([deck_size], dtype=np.float32),
        ]
    )
    return obs.astype(np.float32)


class FullGameSingleAgentEnv(gym.Env[np.ndarray, int]):
    """
    Full game RL environment with complete card mechanics.
    
    Uses the complete Game class, including:
    - Card hands for each player
    - Deck and face-up cards
    - Proper route claiming validation
    - Card drawing actions
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Action encoding:
    # 0 to num_routes-1: Claim route (if valid)
    # num_routes: Draw 2 cards from deck
    # num_routes+1 to num_routes+5: Draw face-up card at index (action - num_routes - 1), then deck
    # num_routes+6 to num_routes+12: Reserved (currently unused)
    # num_routes+6 effectively maps to invalid card draws

    def __init__(
        self,
        graph,
        tickets_file: str,
        opponent_policy: Optional[object] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.tickets_df = pd.read_csv(tickets_file)
        self.opponent_policy = opponent_policy
        self.reward_config = reward_config or RewardConfig()

        # Create game instance
        self.game = Game(number_of_players=2, graph=graph)
        self.game.setup()

        # Get route information
        self.routes: List[Route] = self.game.route_manager.routes
        self.num_routes = len(self.routes)
        self.route_to_idx: Dict[Tuple[str, str, Color], int] = {
            (r.city1, r.city2, r.color): idx for idx, r in enumerate(self.routes)
        }

        # Color mapping for observation
        self.color_to_idx = get_color_mapping()
        self.num_colors = len(self.color_to_idx)

        # Action space: routes + card actions + ticket action
        # Routes: 0 to num_routes-1
        # Draw cards: num_routes to num_routes+5
        #   0 -> draw two from deck
        #   1-5 -> draw face-up (index-1) then deck
        # Draw tickets: num_routes+6
        self.num_card_actions = 6  # Deck+deck plus five face-up combinations
        self.num_actions = self.num_routes + self.num_card_actions + 1  # +1 for tickets

        # Observation space includes:
        # - Agent routes (num_routes)
        # - Opponent routes (num_routes)
        # - Agent trains (1)
        # - Opponent trains (1)
        # - Agent cards (num_colors)
        # - Opponent cards (num_colors) - partial info
        # - Face-up cards (5 * num_colors one-hot)
        # - Deck size (1, normalized)
        obs_len = (
            2 * self.num_routes  # Routes
            + 2  # Trains
            + 2 * self.num_colors  # Cards
            + 5 * self.num_colors  # Face-up cards (one-hot per position)
            + 1  # Deck size
        )
        self._obs_shape = (obs_len,)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=np.float32,
        )

        self._done = False
        self._last_scores = (0, 0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.game = Game(number_of_players=2, graph=self.graph)
        self.game.setup()
        self._done = False
        self._last_scores = (0, 0)
        if self.opponent_policy:
            self.opponent_policy.reset(self)
        return self._get_state(), {}

    def step(self, action: int):
        """Execute one step in the environment."""
        if self._done:
            raise RuntimeError("Cannot call step() on a finished episode. Reset the environment.")

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        # Decode action
        if action < self.num_routes:
            # Claim route
            reward = self._claim_route_action(action)
        elif action < self.num_routes + self.num_card_actions:
            # Draw cards
            reward = self._draw_cards_action(action - self.num_routes)
        elif action == self.num_routes + self.num_card_actions:
            # Draw destination tickets
            reward = self._draw_tickets_action()
        else:
            # Invalid action
            reward = -self.reward_config.invalid_action_penalty
            terminated = True
            info["invalid_action"] = True

        # Check if episode should end
        if self.game.is_game_over or self._episode_should_end():
            terminated = True
            self._done = True
            agent_score = self._calculate_final_score(0)
            opponent_score = self._calculate_final_score(1)
            self._last_scores = (agent_score, opponent_score)
            reward += self.reward_config.final_score_scale * (agent_score - opponent_score)
            info["agent_score"] = agent_score
            info["opponent_score"] = opponent_score
        else:
            # Opponent turn
            opponent_reward = self._opponent_turn()
            info["opponent_reward"] = opponent_reward
            if self.game.is_game_over or self._episode_should_end():
                terminated = True
                self._done = True
                agent_score = self._calculate_final_score(0)
                opponent_score = self._calculate_final_score(1)
                self._last_scores = (agent_score, opponent_score)
                reward += self.reward_config.final_score_scale * (agent_score - opponent_score)
                info["agent_score"] = agent_score
                info["opponent_score"] = opponent_score

        return self._get_state(), reward, terminated, truncated, info

    def _claim_route_action(self, route_idx: int) -> float:
        """Attempt to claim a route. Returns reward."""
        if route_idx < 0 or route_idx >= len(self.routes):
            return -self.reward_config.invalid_action_penalty

        route = self.routes[route_idx]
        agent = self.game.players[0]  # Agent is player 0

        # Check if route is available
        if route.owner is not None:
            return -self.reward_config.invalid_action_penalty

        # Find valid color to use
        color_to_use = find_claim_color_for_player(route, agent)

        if color_to_use is None:
            # Cannot claim - not enough cards
            return -self.reward_config.invalid_action_penalty

        # Claim the route
        success = self.game.claim_route_action(
            route.city1, route.city2, route.color, color_to_use
        )

        if not success:
            return -self.reward_config.invalid_action_penalty

        # Calculate reward
        reward = float(POINTS_TABLE[route.length])
        efficiency = POINTS_TABLE[route.length] / route.length
        reward += self.reward_config.efficiency_weight * efficiency

        # Ticket and connectivity bonuses
        agent_routes = [r for r in agent.routes]
        ticket_points = self._calculate_ticket_points(agent_routes)
        reward += self.reward_config.ticket_weight * ticket_points

        longest = self._calculate_longest_path(agent_routes)
        reward += self.reward_config.connectivity_bonus * (longest / 45.0)

        return reward

    def _draw_cards_action(self, card_action: int) -> float:
        """
        Draw cards based on action encoding.

        Action encoding:
        0: Draw 2 from deck
        1-5: Draw face-up at index (action-1), then deck
        """
        agent = self.game.players[0]
        claimable_before = count_claimable_routes(self.routes, agent)
        hand_size_before = sum(agent.hand.cards.values())

        draw_success = False
        if card_action == 0:
            draw_success = self.game.draw_card_action(None)
        elif 1 <= card_action <= 5:
            face_up_idx = card_action - 1
            if face_up_idx >= len(self.game.game_state.face_up_cards):
                return -self.reward_config.invalid_action_penalty
            draw_success = self.game.draw_card_action(face_up_idx)
        else:
            # Unsupported card action
            return -self.reward_config.invalid_action_penalty

        if not draw_success:
            return -self.reward_config.invalid_action_penalty

        claimable_after = count_claimable_routes(self.routes, agent)
        hand_size_after = sum(agent.hand.cards.values())

        delta_claimable = claimable_after - claimable_before
        reward = 0.0
        if delta_claimable > 0:
            reward += self.reward_config.card_draw_bonus * delta_claimable
        else:
            reward -= self.reward_config.card_draw_penalty

        # discourage hoarding too many cards without using them
        if hand_size_after - hand_size_before >= 2 and delta_claimable <= 0:
            reward -= 0.5 * self.reward_config.card_draw_penalty

        return reward

    def _draw_tickets_action(self) -> float:
        """Draw destination tickets. Must keep at least 1."""
        # Simplified: keep first ticket
        success = self.game.draw_destination_ticket_action([0])
        if success:
            return 0.0  # Neutral reward for tickets
        return -self.reward_config.invalid_action_penalty

    def _opponent_turn(self) -> float:
        """Execute opponent's turn."""
        if not self.opponent_policy:
            return 0.0

        # Get valid actions for opponent
        valid_actions = self.get_valid_actions(for_agent=False)
        if not valid_actions:
            return 0.0

        # Get opponent state (simplified - would need proper observation)
        opponent_state = self._get_state(perspective="opponent")
        action = self.opponent_policy.select_action(opponent_state, valid_actions, self)

        if action is None or action not in valid_actions:
            return 0.0

        # Execute opponent action
        if action < self.num_routes:
            return self._claim_route_action_opponent(action)
        elif action < self.num_routes + self.num_card_actions:
            return self._draw_cards_action_opponent(action - self.num_routes)
        else:
            return 0.0

    def _claim_route_action_opponent(self, route_idx: int) -> float:
        """Opponent claims a route."""
        if route_idx < 0 or route_idx >= len(self.routes):
            return 0.0

        route = self.routes[route_idx]
        opponent = self.game.players[1]

        if route.owner is not None:
            return 0.0

        # Find valid color
        color_to_use = None
        if route.color == Color.GREY:
            for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]:
                if opponent.hand.can_claim_route(route, color):
                    color_to_use = color
                    break
        else:
            if opponent.hand.can_claim_route(route, route.color):
                color_to_use = route.color

        if color_to_use is None:
            return 0.0

        success = self.game.claim_route_action(
            route.city1, route.city2, route.color, color_to_use
        )

        if success:
            return float(POINTS_TABLE[route.length])
        return 0.0

    def _draw_cards_action_opponent(self, card_action: int) -> float:
        """Opponent draws cards."""
        if card_action == 0:
            self.game.draw_card_action(None)
        elif 1 <= card_action <= 5:
            face_up_idx = card_action - 1
            if face_up_idx < len(self.game.game_state.face_up_cards):
                self.game.draw_card_action(face_up_idx)
        return 0.0

    def _episode_should_end(self) -> bool:
        """Check if episode should end."""
        # Game ends when a player has <= 2 trains left (triggers final round)
        # or when game.is_game_over is True
        return any(player.trains_left <= 2 for player in self.game.players)

    def _get_state(self, perspective: str = "agent") -> np.ndarray:
        """Get current observation state."""
        agent_idx = 0 if perspective == "agent" else 1
        return build_observation(
            game=self.game,
            routes=self.routes,
            route_to_idx=self.route_to_idx,
            color_to_idx=self.color_to_idx,
            num_colors=self.num_colors,
            agent_idx=agent_idx,
        )

    def get_valid_actions(self, *, for_agent: bool) -> List[int]:
        """Get list of valid actions."""
        player_idx = 0 if for_agent else 1
        player = self.game.players[player_idx]
        return compute_valid_actions(
            routes=self.routes,
            player=player,
            num_routes=self.num_routes,
            num_card_actions=self.num_card_actions,
        )

    def _calculate_final_score(self, player_idx: int) -> int:
        """Calculate final score for a player."""
        player = self.game.players[player_idx]
        routes = [(r.city1, r.city2, r.length, 'X') for r in player.routes]  # Convert to RouteTuple format
        return final_score(routes, self.tickets_df)

    def _calculate_ticket_points(self, routes: List[Route]) -> int:
        """Calculate ticket points for routes."""
        route_tuples = [(r.city1, r.city2, r.length, 'X') for r in routes]
        return calculate_ticket_points(route_tuples, self.tickets_df)

    def _calculate_longest_path(self, routes: List[Route]) -> int:
        """Calculate longest path for routes."""
        route_tuples = [(r.city1, r.city2, r.length, 'X') for r in routes]
        return calculate_longest_path(route_tuples)

    def render(self):
        """Render the current state."""
        agent = self.game.players[0]
        opponent = self.game.players[1]
        agent_score = self._calculate_final_score(0)
        opponent_score = self._calculate_final_score(1)

        print("=" * 60)
        print("AGENT (Player 0):")
        print(f"  Score: {agent_score}")
        print(f"  Trains: {agent.trains_left}/45")
        print(f"  Routes: {len(agent.routes)}")
        print(f"  Cards: {sum(agent.hand.cards.values())}")
        print(f"  Cards breakdown: {dict((k.value, v) for k, v in agent.hand.cards.items() if v > 0)}")
        print("\nOPPONENT (Player 1):")
        print(f"  Score: {opponent_score}")
        print(f"  Trains: {opponent.trains_left}/45")
        print(f"  Routes: {len(opponent.routes)}")
        print(f"  Cards: {sum(opponent.hand.cards.values())}")
        print("=" * 60)

    def close(self):
        """Close the environment."""
        pass


class FullGameSelfPlayEnv(AECEnv):
    """
    PettingZoo environment for self-play with complete card mechanics.
    
    Two agents play against each other with full game rules including cards.
    """

    metadata = {"name": "TicketToRideFullGame-v0", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, graph, tickets_file: str, reward_config: Optional[RewardConfig] = None) -> None:
        if AECEnv is None:
            raise ImportError("pettingzoo is required to use FullGameSelfPlayEnv")

        super().__init__()
        self.graph = graph
        self.tickets_df = pd.read_csv(tickets_file)
        self.reward_config = reward_config or RewardConfig()

        # Create game instance
        self.game = Game(number_of_players=2, graph=graph)
        self.game.setup()

        # Get route information
        self.routes: List[Route] = self.game.route_manager.routes
        self.num_routes = len(self.routes)
        self.route_to_idx: Dict[Tuple[str, str, Color], int] = {
            (r.city1, r.city2, r.color): idx for idx, r in enumerate(self.routes)
        }

        # Color mapping
        self.color_to_idx = get_color_mapping()
        self.num_colors = len(self.color_to_idx)
        self.num_card_actions = 6
        self.num_actions = self.num_routes + self.num_card_actions + 1

        # Observation space (same as single agent)
        obs_len = (
            2 * self.num_routes + 2 + 2 * self.num_colors +
            5 * self.num_colors + 1
        )
        self._obs_shape = (obs_len,)

        self.agents: List[str] = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.action_spaces = {
            agent: spaces.Discrete(self.num_actions) for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=np.float32)
            for agent in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self._reset_state()

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game = Game(number_of_players=2, graph=self.graph)
        self.game.setup()
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self._reset_state()
        self.agent_selection = self._agent_selector.reset()
        return

    def _reset_state(self) -> None:
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

    def observe(self, agent: str) -> np.ndarray:
        """Get observation for an agent."""
        agent_idx = int(agent.split("_")[1])
        return build_observation(
            game=self.game,
            routes=self.routes,
            route_to_idx=self.route_to_idx,
            color_to_idx=self.color_to_idx,
            num_colors=self.num_colors,
            agent_idx=agent_idx,
        )

    def step(self, action: Optional[int]):
        if self.agents is None or len(self.agents) == 0:
            return

        agent = self.agent_selection
        agent_idx = int(agent.split("_")[1])

        if self.terminations[agent] or self.truncations[agent]:
            self._was_done_step(action)
            return

        valid_actions = self.get_valid_actions(agent)
        if action is None or action not in valid_actions:
            self.rewards[agent] = -self.reward_config.invalid_action_penalty
            for ag in self.agents:
                self.terminations[ag] = True
            self._apply_final_scores()
            self._accumulate_rewards()
            self.agents = []
            return

        # Execute action
        reward = 0.0
        if action < self.num_routes:
            reward = self._claim_route(agent_idx, action)
        elif action < self.num_routes + self.num_card_actions:
            reward = self._draw_cards(agent_idx, action - self.num_routes)
        else:
            reward = self._draw_tickets(agent_idx)

        self.rewards[agent] = reward

        # Check if game should end
        if self.game.is_game_over or self._game_should_stop():
            self._apply_final_scores()
            for ag in self.agents:
                self.terminations[ag] = True
            self._accumulate_rewards()
            self.agents = []
            return

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def _claim_route(self, player_idx: int, route_idx: int) -> float:
        """Claim a route for a player."""
        if route_idx < 0 or route_idx >= len(self.routes):
            return -self.reward_config.invalid_action_penalty

        route = self.routes[route_idx]
        player = self.game.players[player_idx]

        if route.owner is not None or player.trains_left < route.length:
            return -self.reward_config.invalid_action_penalty

        # Find valid color
        color_to_use = find_claim_color_for_player(route, player)

        if color_to_use is None:
            return -self.reward_config.invalid_action_penalty

        success = self.game.claim_route_action(
            route.city1, route.city2, route.color, color_to_use
        )

        if not success:
            return -self.reward_config.invalid_action_penalty

        # Reward
        reward = float(POINTS_TABLE[route.length])
        efficiency = POINTS_TABLE[route.length] / route.length
        reward += self.reward_config.efficiency_weight * efficiency

        routes = [(r.city1, r.city2, r.length, 'X') for r in player.routes]
        ticket_points = calculate_ticket_points(routes, self.tickets_df)
        reward += self.reward_config.ticket_weight * ticket_points

        longest = calculate_longest_path(routes)
        reward += self.reward_config.connectivity_bonus * (longest / 45.0)

        return reward

    def _draw_cards(self, player_idx: int, card_action: int) -> float:
        """Draw cards for a player."""
        # Ensure it's the player's turn
        if self.game.current_player_index != player_idx:
            return -self.reward_config.invalid_action_penalty

        player = self.game.players[player_idx]
        claimable_before = count_claimable_routes(self.routes, player)

        if card_action == 0:
            self.game.draw_card_action(None)
        elif 1 <= card_action <= 5:
            face_up_idx = card_action - 1
            if face_up_idx >= len(self.game.game_state.face_up_cards):
                return -self.reward_config.invalid_action_penalty
            self.game.draw_card_action(face_up_idx)
        else:
            return -self.reward_config.invalid_action_penalty

        claimable_after = count_claimable_routes(self.routes, player)
        delta_claimable = claimable_after - claimable_before

        if delta_claimable > 0:
            return self.reward_config.card_draw_bonus * delta_claimable
        return -self.reward_config.card_draw_penalty

    def _draw_tickets(self, player_idx: int) -> float:
        """Draw destination tickets."""
        if self.game.current_player_index != player_idx:
            return -self.reward_config.invalid_action_penalty
        success = self.game.draw_destination_ticket_action([0])
        return 0.0 if success else -self.reward_config.invalid_action_penalty

    def get_valid_actions(self, agent: str) -> List[int]:
        """Get valid actions for an agent."""
        agent_idx = int(agent.split("_")[1])
        player = self.game.players[agent_idx]
        return compute_valid_actions(
            routes=self.routes,
            player=player,
            num_routes=self.num_routes,
            num_card_actions=self.num_card_actions,
        )

    def _game_should_stop(self) -> bool:
        """Check if game should stop."""
        return any(p.trains_left <= 2 for p in self.game.players)

    def _apply_final_scores(self) -> None:
        """Apply final scores as rewards."""
        if not self.agents:
            return

        routes_0 = [(r.city1, r.city2, r.length, 'X') for r in self.game.players[0].routes]
        routes_1 = [(r.city1, r.city2, r.length, 'X') for r in self.game.players[1].routes]

        score_0 = final_score(routes_0, self.tickets_df)
        score_1 = final_score(routes_1, self.tickets_df)

        diff = score_0 - score_1
        self.rewards.setdefault("player_0", 0.0)
        self.rewards.setdefault("player_1", 0.0)
        self.rewards["player_0"] += self.reward_config.final_score_scale * diff
        self.rewards["player_1"] -= self.reward_config.final_score_scale * diff

        self.infos["player_0"]["score"] = score_0
        self.infos["player_1"]["score"] = score_1

    def render(self):
        """Render the game state."""
        if not self.agents:
            print("Game over")
            return
        for i, agent in enumerate(self.possible_agents):
            player = self.game.players[i]
            routes = [(r.city1, r.city2, r.length, 'X') for r in player.routes]
            score = final_score(routes, self.tickets_df)
            print(f"{agent}: {score} points, {len(player.routes)} routes, {player.trains_left} trains")

    def close(self):
        """Close the environment."""
        pass

    def _opponent_of(self, agent: str) -> str:
        return "player_1" if agent == "player_0" else "player_0"


__all__ = ["FullGameSingleAgentEnv", "FullGameSelfPlayEnv", "RewardConfig"]

