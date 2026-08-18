"""WebSocket connection manager for real-time telemetry broadcast."""

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for live streaming."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to send message to websocket client: %s", exc)
