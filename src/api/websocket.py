"""High-performance non-blocking WebSocket Hub with conflated client queues and orjson."""

import asyncio
import logging
from typing import Any
from fastapi import WebSocket
import orjson

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections with non-blocking conflated ring buffers."""

    def __init__(self, queue_maxsize: int = 5) -> None:
        self.queue_maxsize = queue_maxsize
        self._queues: dict[WebSocket, asyncio.Queue[str]] = {}
        self._tasks: dict[WebSocket, asyncio.Task] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def active_connections(self) -> list[WebSocket]:
        return list(self._queues.keys())

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_maxsize)
        self._queues[websocket] = q
        task = asyncio.create_task(self._client_sender(websocket, q))
        self._tasks[websocket] = task

        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._tasks:
            self._tasks[websocket].cancel()
            del self._tasks[websocket]
        if websocket in self._queues:
            del self._queues[websocket]

    async def _client_sender(self, websocket: WebSocket, queue: asyncio.Queue[str]) -> None:
        """Isolated per-client sender worker preventing head-of-line blocking."""
        try:
            while True:
                payload = await queue.get()
                await websocket.send_text(payload)
                queue.task_done()
        except (asyncio.CancelledError, Exception) as exc:
            logger.debug("Client sender disconnected: %s", exc)
        finally:
            self.disconnect(websocket)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """Asynchronous broadcast using orjson serialization and lossy conflation."""
        if not self._queues:
            return

        payload: str = orjson.dumps(data).decode("utf-8")
        for ws, q in list(self._queues.items()):
            if q.full():
                try:
                    q.get_nowait()
                    q.task_done()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def broadcast_sync(self, data: dict[str, Any]) -> None:
        """Thread-safe, non-blocking broadcast with automatic frame dropping for slow clients."""
        if not self._queues or not self._loop or not self._loop.is_running():
            return

        payload: str = orjson.dumps(data).decode("utf-8")

        def _enqueue() -> None:
            for ws, q in list(self._queues.items()):
                if q.full():
                    try:
                        q.get_nowait()  # Drop oldest frame (conflation)
                        q.task_done()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

        self._loop.call_soon_threadsafe(_enqueue)
