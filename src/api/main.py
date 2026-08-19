import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.brain_service import BrainService
from src.api.game_service import GameService
from src.api.replay_service import ReplayService
from src.api.schemas import (
    BrainInspectionDTO,
    BrainInspectRequest,
    GameSessionCreateRequest,
    GameStateDTO,
    GameStepRequest,
    ReplayDetailDTO,
    TrainingStartRequest,
    TrainingStatusDTO,
)
from src.api.trainer_service import TrainerService
from src.api.websocket import ConnectionManager
from src.experiments.registry import ExperimentRegistry
from src.rl.networks import MaskedActorCritic, MaskedQNetwork

connection_manager = ConnectionManager()
game_service = GameService()
brain_service = BrainService()
replay_service = ReplayService()
trainer_service = TrainerService(connection_manager=connection_manager)
registry = ExperimentRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set the running event loop for threadsafe WebSocket broadcasting
    connection_manager.set_loop(asyncio.get_running_loop())
    yield


app = FastAPI(
    title="TicketToRide RL Lab API",
    version="1.0.0",
    description="Backend API and WebSocket streaming hub for TicketToRide RL Lab",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- System Health & Legacy Endpoints ---
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "app": "TicketToRide RL Lab", "version": "1.0.0"}


@app.get("/state")
def legacy_state() -> dict[str, Any]:
    req = GameSessionCreateRequest(map_name="usa", player_types=["human", "greedy"], seed=42)
    dto = game_service.create_session(req)
    data = dto.model_dump()
    data["deck_size"] = 110
    return data


# --- Game Endpoints ---
@app.post("/api/game/new", response_model=GameStateDTO)
def create_game_session(request: GameSessionCreateRequest) -> GameStateDTO:
    return game_service.create_session(request)


@app.get("/api/game/{session_id}", response_model=GameStateDTO)
def get_game_state(session_id: str) -> GameStateDTO:
    state = game_service.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return state


@app.post("/api/game/step", response_model=GameStateDTO)
def step_game_session(request: GameStepRequest) -> GameStateDTO:
    try:
        return game_service.step_session(request.session_id, action=request.action)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/game/{session_id}")
def delete_game_session(session_id: str) -> dict[str, bool]:
    success = game_service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"success": True}


# --- Training Endpoints ---
@app.get("/api/training/status", response_model=TrainingStatusDTO)
def get_training_status() -> TrainingStatusDTO:
    return trainer_service.get_status()


@app.post("/api/training/start", response_model=TrainingStatusDTO)
def start_training(request: TrainingStartRequest) -> TrainingStatusDTO:
    return trainer_service.start_training(request)


@app.post("/api/training/stop", response_model=TrainingStatusDTO)
def stop_training() -> TrainingStatusDTO:
    return trainer_service.stop_training()


# --- Replays & Experiments Endpoints ---
@app.get("/api/replays/list")
def list_replays() -> list[dict[str, Any]]:
    return replay_service.list_replays()


@app.get("/api/replays/{replay_id}", response_model=ReplayDetailDTO)
def get_replay(replay_id: str) -> ReplayDetailDTO:
    replay = replay_service.load_replay(replay_id)
    if not replay:
        raise HTTPException(status_code=404, detail=f"Replay '{replay_id}' not found.")
    return replay


@app.get("/api/experiments/list")
def list_experiments() -> list[dict[str, Any]]:
    records = registry.list_experiments()
    return [r if isinstance(r, dict) else r.model_dump() for r in records]


# --- Brain Introspection Endpoint ---
@app.post("/api/brain/inspect", response_model=BrainInspectionDTO)
def inspect_brain(request: BrainInspectRequest) -> BrainInspectionDTO:
    if request.session_id:
        session = game_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

        current_idx = session.game.state.current_player_index
        obs = session.encoder.encode(session.game.state, current_idx).tolist()
        pending = session.game.state.current_player.pending_tickets if session.game.state.current_player else None
        valid_acts = session.game.valid_actions()
        mask = [bool(m) for m in session.masker.compute_mask(valid_acts, pending_tickets=pending)]

        action_labels = [f"Act {i}: {session.action_space.to_action(i).action_type.name}" for i in range(session.action_space.n)]

        agent = session.agents[current_idx]
        if hasattr(agent, "q_net"):
            return brain_service.inspect_q_network(agent.q_net, obs, mask, action_labels=action_labels)
        elif hasattr(agent, "actor_critic"):
            return brain_service.inspect_actor_critic(agent.actor_critic, obs, mask, action_labels=action_labels)
        else:
            # Synthetic default model for inspection
            net = MaskedActorCritic(input_dim=len(obs), action_dim=len(mask), hidden_dim=128)
            return brain_service.inspect_actor_critic(net, obs, mask, action_labels=action_labels)

    # Standalone observation vector inspection
    obs_vec = request.observation or ([0.1] * 128)
    mask_vec = request.action_mask or ([True] * 56)
    labels = [f"Action {i}" for i in range(len(mask_vec))]

    if request.model_type == "dqn":
        net_dqn = MaskedQNetwork(input_dim=len(obs_vec), action_dim=len(mask_vec), hidden_dim=128)
        if os.path.exists("experiments/checkpoints"):
            ckpts = sorted(
                [os.path.join("experiments/checkpoints", f) for f in os.listdir("experiments/checkpoints") if "dqn" in f.lower() and f.endswith(".pt")],
                key=os.path.getmtime,
                reverse=True,
            )
            if ckpts:
                try:
                    net_dqn.load_state_dict(torch.load(ckpts[0], weights_only=True))
                except Exception:  # noqa: BLE001
                    pass
        return brain_service.inspect_q_network(net_dqn, obs_vec, mask_vec, action_labels=labels)
    else:
        net_ppo = MaskedActorCritic(input_dim=len(obs_vec), action_dim=len(mask_vec), hidden_dim=128)
        if os.path.exists("experiments/checkpoints"):
            ckpts = sorted(
                [os.path.join("experiments/checkpoints", f) for f in os.listdir("experiments/checkpoints") if "ppo" in f.lower() and f.endswith(".pt")],
                key=os.path.getmtime,
                reverse=True,
            )
            if ckpts:
                try:
                    net_ppo.load_state_dict(torch.load(ckpts[0], weights_only=True))
                except Exception:  # noqa: BLE001
                    pass
        return brain_service.inspect_actor_critic(net_ppo, obs_vec, mask_vec, action_labels=labels)


# --- WebSocket Telemetry Hub ---
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo heartbeat/ping
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
