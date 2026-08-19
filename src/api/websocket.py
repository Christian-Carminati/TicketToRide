"""WebSocket connection manager for real-time telemetry broadcast."""

import asyncio
import logging
from typing import Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and thread-safe event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to send message to websocket client: %s", exc)
                self.disconnect(connection)

    def broadcast_sync(self, data: dict[str, Any]) -> None:
        """Thread-safe synchronous broadcast method for background workers."""
        if not self.active_connections:
            return

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_json(data), self._loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.broadcast_json(data))
            except Exception:  # noqa: BLE001
                pass
