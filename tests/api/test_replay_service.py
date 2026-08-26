"""Unit tests for ReplayService trajectory serialization and storage."""

import tempfile
from pathlib import Path

from src.api.replay_service import ReplayService
from src.api.schemas import ActionDTO, ReplayDetailDTO, ReplayFrameDTO


def test_replay_service_save_list_load_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        service = ReplayService(replays_dir=Path(tmpdir))

        frame = ReplayFrameDTO(
            step_index=0,
            turn_number=1,
            player_index=0,
            action=ActionDTO(action_type="DRAW_TRAIN_CARDS"),
            reward=0.0,
            state_snapshot={"turn": 1, "scores": [0, 0]},
        )
        replay = ReplayDetailDTO(
            replay_id="rep_test_01",
            map_name="mini",
            seed=42,
            date="2026-08-19",
            player_names=["P1", "P2"],
            total_steps=1,
            winner_index=0,
            final_scores=[10, 5],
            frames=[frame],
        )

        # Save
        saved_path = service.save_replay(replay)
        assert Path(saved_path).exists()

        # List
        replays = service.list_replays()
        assert len(replays) == 1
        assert replays[0]["replay_id"] == "rep_test_01"

        # Load
        loaded = service.load_replay("rep_test_01")
        assert loaded is not None
        assert loaded.replay_id == "rep_test_01"
        assert len(loaded.frames) == 1
        assert loaded.frames[0].action.action_type == "DRAW_TRAIN_CARDS"

        # Delete
        assert service.delete_replay("rep_test_01") is True
        assert service.load_replay("rep_test_01") is None
