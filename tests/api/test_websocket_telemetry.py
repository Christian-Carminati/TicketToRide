"""Unit tests for WebSocket telemetry hub."""

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from src.api.websocket import ConnectionManager

app = FastAPI()
manager = ConnectionManager()


@app.websocket("/ws/test")
async def ws_test_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.broadcast_json({"echo": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def test_websocket_connection_and_broadcast():
    client = TestClient(app)
    with client.websocket_connect("/ws/test") as websocket:
        websocket.send_json({"message": "hello"})
        data = websocket.receive_json()
        assert data == {"echo": {"message": "hello"}}
