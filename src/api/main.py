"""FastAPI Application entrypoint for TicketToRide RL Lab."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import GameStateResponse
from src.api.websocket import ConnectionManager

app = FastAPI(
    title="TicketToRide RL Lab API",
    version="0.1.0",
    description="Backend API and WebSocket streaming for RL visualization and lab controls",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "app": "TicketToRide RL Lab"}


@app.get("/state", response_model=GameStateResponse)
def get_state() -> GameStateResponse:
    # Skeleton state endpoint
    return GameStateResponse(
        turn_number=1,
        current_player_index=0,
        deck_size=110,
        discard_pile_size=0,
        is_game_over=False,
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or process client command
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
