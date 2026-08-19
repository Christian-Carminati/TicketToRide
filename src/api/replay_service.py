"""ReplayService: Manages storage, retrieval, and trajectory parsing of game replays."""

import json
from pathlib import Path
from typing import Any

from src.api.schemas import ReplayDetailDTO, ReplayFrameDTO


class ReplayService:
    """Provides file-based storage and listing of match replay recordings."""

    def __init__(self, replays_dir: Path | str | None = None) -> None:
        if replays_dir is None:
            self.replays_dir = Path("experiments/replays")
        else:
            self.replays_dir = Path(replays_dir)
        self.replays_dir.mkdir(parents=True, exist_ok=True)

    def save_replay(self, replay: ReplayDetailDTO) -> str:
        filepath = self.replays_dir / f"{replay.replay_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(replay.model_dump(), f, indent=2)
        return str(filepath)

    def list_replays(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for file in sorted(self.replays_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(
                        {
                            "replay_id": data.get("replay_id", file.stem),
                            "map_name": data.get("map_name", "unknown"),
                            "seed": data.get("seed", 0),
                            "date": data.get("date", ""),
                            "player_names": data.get("player_names", []),
                            "total_steps": data.get("total_steps", len(data.get("frames", []))),
                            "winner_index": data.get("winner_index", 0),
                            "final_scores": data.get("final_scores", []),
                            "filename": file.name,
                        }
                    )
            except Exception:  # noqa: BLE001
                continue
        return results

    def load_replay(self, replay_id: str) -> ReplayDetailDTO | None:
        filepath = self.replays_dir / f"{replay_id}.json"
        if not filepath.exists():
            # Try searching with exact stem
            matches = list(self.replays_dir.glob(f"{replay_id}*"))
            if matches:
                filepath = matches[0]
            else:
                return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ReplayDetailDTO.model_validate(data)
        except Exception:  # noqa: BLE001
            return None

    def delete_replay(self, replay_id: str) -> bool:
        filepath = self.replays_dir / f"{replay_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
