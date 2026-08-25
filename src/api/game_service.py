import os
import uuid
from typing import Any
import torch

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.neural_mcts_agent import BayesianOpponentMCTSAgent, NeuralMCTSAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.agents.strategic_agent import StrategicAgent
from src.api.schemas import ActionDTO, GameSessionCreateRequest, GameStateDTO, PlayerStateDTO
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.game.action import Action, ActionType
from src.game.card import CardColor
from src.game.game import Game
from src.game.graph import check_ticket_completed
from src.game.maps import create_synthetic_mini_board, load_usa_board
from src.game.state import GameState


class HumanAgent(BaseAgent):
    """Placeholder agent representing a human player making manual moves."""

    def __init__(self, name: str = "Human Player", seed: int = 42) -> None:
        super().__init__(name=name)

    def act(
        self,
        state: GameState,
        valid_actions: list[Action],
        board: Any = None,
    ) -> Action:
        # Fallback to first valid action if called automatically
        return valid_actions[0] if valid_actions else Action(action_type=ActionType.DRAW_HIDDEN_CARD)


class ActiveGameSession:
    """Encapsulates a live game session and its associated environment encoders."""

    def __init__(
        self,
        session_id: str,
        game: Game,
        map_name: str,
        agents: list[BaseAgent],
        action_space: DiscreteActionSpace,
        masker: ActionMasker,
        encoder: BaseObservationEncoder,
    ) -> None:
        self.session_id = session_id
        self.game = game
        self.map_name = map_name
        self.agents = agents
        self.action_space = action_space
        self.masker = masker
        self.encoder = encoder
        self.last_action: ActionDTO | None = None
        self.last_reward: float | None = None


PLAYER_COLORS = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]


class GameService:
    """Thread-safe manager of active game sessions with LRU cache eviction."""

    def __init__(self, max_sessions: int = 20) -> None:
        self._sessions: dict[str, ActiveGameSession] = {}
        self._max_sessions = max_sessions

    def create_session(self, request: GameSessionCreateRequest) -> GameStateDTO:
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

        if request.map_name == "mini":
            board, tickets = create_synthetic_mini_board()
        else:
            board, tickets = load_usa_board()

        action_space = DiscreteActionSpace(board=board)
        masker = ActionMasker(action_space=action_space)
        encoder = ObservationV1(board=board, initial_tickets=tickets, num_players=len(request.player_types))

        agents: list[BaseAgent] = []
        for idx, p_type in enumerate(request.player_types):
            agent_seed = request.seed + idx * 100
            p_type_clean = p_type.lower()
            
            # Checkpoint resolution per player
            ckpt_path = None
            if request.player_checkpoints and idx < len(request.player_checkpoints) and request.player_checkpoints[idx]:
                ckpt_path = request.player_checkpoints[idx]
            elif request.model_checkpoint:
                ckpt_path = request.model_checkpoint

            if p_type_clean == "human":
                agents.append(HumanAgent(name=f"Human Conductor ({idx+1})"))
            elif p_type_clean == "random":
                agents.append(RandomAgent(name=f"RandomBot ({idx+1})", seed=agent_seed))
            elif p_type_clean == "greedy":
                agents.append(GreedyAgent(name=f"GreedyBot ({idx+1})"))
            elif p_type_clean == "strategic":
                agents.append(StrategicAgent(name=f"StrategicBot ({idx+1})"))
            elif p_type_clean == "mcts":
                agents.append(MCTSAgent(name=f"IS-MCTS Bot ({idx+1})", num_simulations=20, seed=agent_seed, board=board, tickets=tickets))
            elif p_type_clean == "bayesian_mcts":
                agents.append(BayesianOpponentMCTSAgent(name=f"Bayesian MCTS ({idx+1})", num_simulations=20, seed=agent_seed, board=board, tickets=tickets))
            elif p_type_clean == "alphazero":
                agent_az = NeuralMCTSAgent(name=f"AlphaZero ({idx+1})", num_simulations=20, seed=agent_seed, board=board, tickets=tickets)
                if ckpt_path and os.path.exists(ckpt_path):
                    try:
                        agent_az.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    except Exception as e:
                        print(f"Failed to load AlphaZero checkpoint {ckpt_path}: {e}")
                agents.append(agent_az)
            elif p_type_clean in ("recurrent_ppo", "lstm_ppo"):
                agent_rec = RecurrentPPOAgent(
                    name=f"Recurrent PPO ({idx+1})",
                    board=board,
                    tickets=tickets,
                    num_players=len(request.player_types),
                    model_or_path=ckpt_path if ckpt_path and os.path.exists(ckpt_path) else None,
                )
                agents.append(agent_rec)
            elif p_type_clean == "dqn":
                agent_dqn = DQNAgent(
                    name=f"DQN Trained ({idx+1})",
                    input_dim=encoder.observation_shape[0],
                    action_dim=action_space.n,
                    encoder=encoder,
                    discrete_actions=action_space,
                )
                if not ckpt_path and os.path.exists("experiments/checkpoints"):
                    ckpts = sorted(
                        [os.path.join("experiments/checkpoints", f) for f in os.listdir("experiments/checkpoints") if "dqn" in f.lower() and f.endswith(".pt")],
                        key=os.path.getmtime,
                        reverse=True,
                    )
                    if ckpts:
                        ckpt_path = ckpts[0]

                if ckpt_path and os.path.exists(ckpt_path):
                    try:
                        agent_dqn.load(ckpt_path)
                    except Exception as e:
                        print(f"Failed to load DQN checkpoint {ckpt_path}: {e}")
                agents.append(agent_dqn)
            elif p_type_clean == "ppo":
                agent_ppo = PPOAgent(
                    name=f"PPO Trained ({idx+1})",
                    input_dim=encoder.observation_shape[0],
                    action_dim=action_space.n,
                    encoder=encoder,
                    discrete_actions=action_space,
                )
                if not ckpt_path and os.path.exists("experiments/checkpoints"):
                    ckpts = sorted(
                        [os.path.join("experiments/checkpoints", f) for f in os.listdir("experiments/checkpoints") if "ppo" in f.lower() and f.endswith(".pt")],
                        key=os.path.getmtime,
                        reverse=True,
                    )
                    if ckpts:
                        ckpt_path = ckpts[0]

                if ckpt_path and os.path.exists(ckpt_path):
                    try:
                        agent_ppo.load(ckpt_path)
                    except Exception as e:
                        print(f"Failed to load PPO checkpoint {ckpt_path}: {e}")
                agents.append(agent_ppo)
            else:
                agents.append(RandomAgent(name=f"RandomBot ({idx+1})", seed=agent_seed))

        game = Game(
            board=board,
            tickets_deck=tickets,
            num_players=len(request.player_types),
            seed=request.seed,
        )
        game.reset(seed=request.seed)

        session = ActiveGameSession(
            session_id=session_id,
            game=game,
            map_name=request.map_name,
            agents=agents,
            action_space=action_space,
            masker=masker,
            encoder=encoder,
        )
        self._sessions[session_id] = session

        # Evict oldest session if limit reached
        if len(self._sessions) > self._max_sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]

        return self._to_state_dto(session)

    def get_session(self, session_id: str) -> ActiveGameSession | None:
        return self._sessions.get(session_id)

    def get_session_state(self, session_id: str) -> GameStateDTO | None:
        session = self.get_session(session_id)
        if not session:
            return None
        return self._to_state_dto(session)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def step_session(
        self,
        session_id: str,
        action: ActionDTO | None = None,
        action_dto: ActionDTO | None = None,
    ) -> GameStateDTO:
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found.")

        game = session.game
        if game.state.is_game_over:
            return self._to_state_dto(session)

        current_player_idx = game.state.current_player_index
        active_agent = session.agents[current_player_idx]
        current_player = game.state.players[current_player_idx]
        valid_actions = game.rules.get_valid_actions(current_player, game.state, game.board, game.num_players)

        if not valid_actions:
            return self._to_state_dto(session)

        # Determine action
        act_input = action if action is not None else action_dto
        if act_input is not None:
            domain_action = self._from_action_dto(act_input)
        elif isinstance(active_agent, HumanAgent):
            domain_action = valid_actions[0]
        else:
            domain_action = active_agent.act(game.state, valid_actions, board=game.board)

        # Execute in Game engine
        prev_score = current_player.score
        game.step(domain_action)
        reward = float(current_player.score - prev_score)

        session.last_action = self._to_action_dto(domain_action)
        session.last_reward = reward

        return self._to_state_dto(session)

    def _to_state_dto(self, session: ActiveGameSession) -> GameStateDTO:
        game = session.game
        state = game.state

        players_dto: list[PlayerStateDTO] = []
        for idx, p in enumerate(state.players):
            cards_dict = {}
            for c in CardColor:
                cards_dict[c.name.lower()] = p.cards.get(c, 0)

            p_routes = [game.board.get_route(r_id) for r_id in p.claimed_route_ids]
            valid_p_routes = [r for r in p_routes if r is not None]
            tickets_list = [
                {
                    "ticket_id": t.id,
                    "city_a": t.city_a,
                    "city_b": t.city_b,
                    "points": t.points,
                    "completed": check_ticket_completed(valid_p_routes, t),
                }
                for t in p.tickets
            ]

            players_dto.append(
                PlayerStateDTO(
                    player_id=p.id,
                    name=p.name,
                    score=p.score,
                    trains_remaining=p.trains_remaining,
                    cards_in_hand=cards_dict,
                    tickets=tickets_list,
                    claimed_route_ids=list(p.claimed_route_ids),
                    color=PLAYER_COLORS[idx % len(PLAYER_COLORS)],
                )
            )

        curr_p = state.current_player
        valid_actions = game.rules.get_valid_actions(curr_p, state, game.board, game.num_players) if curr_p else []
        valid_dtos = [self._to_action_dto(a) for a in valid_actions]

        action_mask = session.masker.compute_mask(valid_actions, curr_p.pending_tickets if curr_p else None)

        claimed_dict = {}
        for p in state.players:
            for r_id in p.claimed_route_ids:
                claimed_dict[r_id] = p.id

        return GameStateDTO(
            session_id=session.session_id,
            turn_number=state.turn_number,
            current_player_index=state.current_player_index,
            map_name=session.map_name,
            players=players_dto,
            visible_cards=[c.color.name.lower() for c in state.visible_cards],
            deck_size=len(state.train_deck),
            discard_pile_size=len(state.discard_pile),
            tickets_deck_size=len(state.ticket_deck),
            claimed_routes=claimed_dict,
            valid_actions=valid_dtos,
            action_mask=action_mask.tolist(),
            is_game_over=state.is_game_over,
            winner_id=state.winner_id,
            last_action=session.last_action,
            last_reward=session.last_reward,
        )

    def _to_action_dto(self, a: Action) -> ActionDTO:
        return ActionDTO(
            action_type=a.action_type.name,
            card_index=a.card_index,
            route_id=a.route_id,
            card_color=a.color_chosen.name.lower() if a.color_chosen else None,
            ticket_ids=list(a.ticket_ids) if a.ticket_ids else None,
        )

    def _from_action_dto(self, dto: ActionDTO) -> Action:
        try:
            action_type = ActionType[dto.action_type]
        except (KeyError, ValueError):
            action_type = ActionType.DRAW_HIDDEN_CARD

        card_color = None
        if dto.card_color:
            try:
                card_color = CardColor[dto.card_color.upper()]
            except (KeyError, ValueError):
                for c in CardColor:
                    if c.value == dto.card_color.lower() or c.name.lower() == dto.card_color.lower():
                        card_color = c
                        break

        ticket_tuple = tuple(dto.ticket_ids) if dto.ticket_ids else None
        return Action(
            action_type=action_type,
            card_index=dto.card_index,
            route_id=dto.route_id,
            color_chosen=card_color,
            ticket_ids=ticket_tuple,
        )
