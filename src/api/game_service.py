import os
import uuid
from typing import Any
import torch

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.strategic_agent import StrategicAgent
from src.api.schemas import ActionDTO, GameSessionCreateRequest, GameStateDTO, PlayerStateDTO
from src.environment.action_mask import ActionMasker
from src.environment.action_space import DiscreteActionSpace
from src.environment.observation import BaseObservationEncoder, ObservationV1
from src.game.action import Action, ActionType
from src.game.card import CardColor
from src.game.game import Game
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
        self.max_sessions = max_sessions

    def create_session(self, request: GameSessionCreateRequest) -> GameStateDTO:
        # Evict oldest sessions if exceeding capacity to prevent memory bloat
        if len(self._sessions) >= self.max_sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]

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
            if p_type_clean == "human":
                agents.append(HumanAgent(name=f"Human ({idx+1})"))
            elif p_type_clean == "random":
                agents.append(RandomAgent(name=f"RandomBot ({idx+1})", seed=agent_seed))
            elif p_type_clean == "greedy":
                agents.append(GreedyAgent(name=f"GreedyBot ({idx+1})"))
            elif p_type_clean == "strategic":
                agents.append(StrategicAgent(name=f"StrategicBot ({idx+1})"))
            elif p_type_clean == "dqn":
                agent = DQNAgent(
                    name=f"DQN Trained ({idx+1})",
                    input_dim=encoder.observation_shape[0],
                    action_dim=action_space.n,
                    encoder=encoder,
                    discrete_actions=action_space,
                )
                ckpt_path = request.model_checkpoint
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
                        agent.load(ckpt_path)
                    except Exception as e:
                        print(f"Failed to load DQN checkpoint {ckpt_path}: {e}")
                agents.append(agent)
            elif p_type_clean == "ppo":
                agent = PPOAgent(
                    name=f"PPO Trained ({idx+1})",
                    input_dim=encoder.observation_shape[0],
                    action_dim=action_space.n,
                    encoder=encoder,
                    discrete_actions=action_space,
                )
                ckpt_path = request.model_checkpoint
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
                        agent.load(ckpt_path)
                    except Exception as e:
                        print(f"Failed to load PPO checkpoint {ckpt_path}: {e}")
                agents.append(agent)
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
        return self._to_dto(session)

    def get_session_state(self, session_id: str) -> GameStateDTO | None:
        session = self._sessions.get(session_id)
        if not session:
            return None
        return self._to_dto(session)

    def step_session(self, session_id: str, action: ActionDTO | None = None) -> GameStateDTO:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        game = session.game
        if game.state.is_game_over:
            return self._to_dto(session)

        current_player_idx = game.state.current_player_index
        current_agent = session.agents[current_player_idx]
        valid_actions = game.valid_actions()

        domain_action: Action | None = None

        if action is not None:
            # Explicit action provided (human or client-driven)
            domain_action = self._dto_to_action(action)
        else:
            # Bot chooses action via standard act() method
            domain_action = current_agent.act(game.state, valid_actions, board=game.board)

        # Fallback if somehow no action or invalid
        if domain_action is None:
            domain_action = valid_actions[0] if valid_actions else Action(action_type=ActionType.DRAW_HIDDEN_CARD)

        score_before = game.state.players[current_player_idx].score
        game.step(domain_action)
        score_after = game.state.players[current_player_idx].score

        session.last_action = self._action_to_dto(domain_action)
        session.last_reward = float(score_after - score_before)

        return self._to_dto(session)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_session(self, session_id: str) -> ActiveGameSession | None:
        return self._sessions.get(session_id)

    def _to_dto(self, session: ActiveGameSession) -> GameStateDTO:
        game = session.game
        state = game.state

        players_dto: list[PlayerStateDTO] = []
        for idx, p in enumerate(state.players):
            cards_dict: dict[str, int] = {
                color.name: count for color, count in p.cards.items() if count > 0
            }

            claimed_ids = [r.id for r in game.board.routes if r.claimed_by == p.id]
            tickets_list = [
                {
                    "id": t.id,
                    "city_a": t.city_a,
                    "city_b": t.city_b,
                    "points": t.points,
                    "completed": False,
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
                    claimed_route_ids=claimed_ids,
                    color=PLAYER_COLORS[idx % len(PLAYER_COLORS)],
                )
            )

        claimed_routes_map: dict[str, str] = {
            r.id: r.claimed_by for r in game.board.routes if r.claimed_by is not None
        }

        valid_acts = game.valid_actions()
        valid_actions_dto = [self._action_to_dto(a) for a in valid_acts]

        # Compute boolean action mask
        pending = state.current_player.pending_tickets if state.current_player else None
        mask_np = session.masker.compute_mask(valid_acts, pending_tickets=pending)
        action_mask = [bool(m) for m in mask_np]

        visible_cards = [c.color.name for c in state.visible_cards]

        return GameStateDTO(
            session_id=session.session_id,
            turn_number=state.turn_number,
            current_player_index=state.current_player_index,
            map_name=session.map_name,
            players=players_dto,
            visible_cards=visible_cards,
            deck_size=len(state.train_deck),
            discard_pile_size=len(state.discard_pile),
            tickets_deck_size=len(state.ticket_deck),
            claimed_routes=claimed_routes_map,
            valid_actions=valid_actions_dto,
            action_mask=action_mask,
            is_game_over=state.is_game_over,
            winner_id=None,
            last_action=session.last_action,
            last_reward=session.last_reward,
        )

    def _action_to_dto(self, act: Action) -> ActionDTO:
        color_str = act.color_chosen.name if act.color_chosen else None
        return ActionDTO(
            action_type=act.action_type.name,
            card_index=act.card_index,
            route_id=act.route_id,
            card_color=color_str,
            ticket_ids=list(act.ticket_ids) if act.ticket_ids else None,
        )

    def _dto_to_action(self, dto: ActionDTO) -> Action:
        act_type = ActionType[dto.action_type]
        color_chosen = CardColor[dto.card_color] if dto.card_color else None
        ticket_tuple = tuple(dto.ticket_ids) if dto.ticket_ids else None
        return Action(
            action_type=act_type,
            card_index=dto.card_index,
            route_id=dto.route_id,
            color_chosen=color_chosen,
            ticket_ids=ticket_tuple,
        )
