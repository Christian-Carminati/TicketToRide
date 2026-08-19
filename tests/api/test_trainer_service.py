"""Unit tests for TrainerService background runner and telemetry streaming."""

import time
from src.api.schemas import TrainingStartRequest
from src.api.trainer_service import TrainerService
from src.api.websocket import ConnectionManager


def test_trainer_service_start_status_stop():
    manager = ConnectionManager()
    service = TrainerService(connection_manager=manager)

    req = TrainingStartRequest(config_name="ppo_usa.yaml", override_timesteps=100, seed=42)
    status = service.start_training(req)
    assert status.is_training is True
    assert status.algorithm == "ppo"

    time.sleep(0.2)
    current_status = service.get_status()
    assert current_status.experiment_id is not None

    service.stop_training()
    time.sleep(0.1)
    status_after = service.get_status()
    assert status_after.is_training is False
