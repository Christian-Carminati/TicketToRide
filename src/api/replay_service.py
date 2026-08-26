"""ReplayService: High-performance SQLite WAL indexed storage and compressed match replays."""

import json
from pathlib import Path
import sqlite3
from typing import Any
import orjson
import zstandard as zstd

from src.api.schemas import ReplayDetailDTO


class ReplayService:
    """Provides fast SQLite-indexed storage and zstandard compressed match recordings."""

    def __init__(self, replays_dir: Path | str | None = None) -> None:
        if replays_dir is None:
            self.replays_dir = Path("experiments/replays")
        else:
            self.replays_dir = Path(replays_dir)
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.replays_dir / "replays_index.db"
        self._cctx = zstd.ZstdCompressor(level=3)
        self._dctx = zstd.ZstdDecompressor()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replays_index (
                    replay_id TEXT PRIMARY KEY,
                    map_name TEXT,
                    seed INTEGER,
                    date TEXT,
                    player_names TEXT,
                    total_steps INTEGER,
                    winner_index INTEGER,
                    final_scores TEXT,
                    filename TEXT,
                    mtime REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mtime ON replays_index(mtime DESC);")

    def save_replay(self, replay: ReplayDetailDTO) -> str:
        filepath_zst = self.replays_dir / f"{replay.replay_id}.zst"
        raw_bytes = orjson.dumps(replay.model_dump())
        compressed_bytes = self._cctx.compress(raw_bytes)

        with open(filepath_zst, "wb") as f:
            f.write(compressed_bytes)

        # Also write lightweight uncompressed json for legacy clients if needed
        filepath_json = self.replays_dir / f"{replay.replay_id}.json"
        with open(filepath_json, "wb") as f:
            f.write(raw_bytes)

        # Index in SQLite WAL
        mtime = filepath_zst.stat().st_mtime
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO replays_index
                (replay_id, map_name, seed, date, player_names, total_steps, winner_index, final_scores, filename, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    replay.replay_id,
                    replay.map_name,
                    replay.seed,
                    replay.date,
                    json.dumps(replay.player_names),
                    replay.total_steps,
                    replay.winner_index,
                    json.dumps(replay.final_scores),
                    filepath_zst.name,
                    mtime,
                ),
            )
        return str(filepath_zst)

    def list_replays(self) -> list[dict[str, Any]]:
        # Fast query from SQLite WAL
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT replay_id, map_name, seed, date, player_names, total_steps, winner_index, final_scores, filename
                FROM replays_index
                ORDER BY mtime DESC
                """
            )
            rows = cursor.fetchall()

        if rows:
            return [
                {
                    "replay_id": r[0],
                    "map_name": r[1],
                    "seed": r[2],
                    "date": r[3],
                    "player_names": json.loads(r[4]) if r[4] else [],
                    "total_steps": r[5],
                    "winner_index": r[6],
                    "final_scores": json.loads(r[7]) if r[7] else [],
                    "filename": r[8],
                }
                for r in rows
            ]

        # Fallback disk scan if database was newly initialized with existing JSON files
        results: list[dict[str, Any]] = []
        for file in sorted(self.replays_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(file, "rb") as f:
                    data = orjson.loads(f.read())
                    entry = {
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
                    results.append(entry)
                    # Cache to SQLite
                    with self._get_connection() as conn:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO replays_index
                            (replay_id, map_name, seed, date, player_names, total_steps, winner_index, final_scores, filename, mtime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entry["replay_id"],
                                entry["map_name"],
                                entry["seed"],
                                entry["date"],
                                json.dumps(entry["player_names"]),
                                entry["total_steps"],
                                entry["winner_index"],
                                json.dumps(entry["final_scores"]),
                                file.name,
                                file.stat().st_mtime,
                            ),
                        )
            except Exception:  # noqa: BLE001
                continue
        return results

    def load_replay(self, replay_id: str) -> ReplayDetailDTO | None:
        # Check .zst first
        filepath_zst = self.replays_dir / f"{replay_id}.zst"
        if filepath_zst.exists():
            try:
                with open(filepath_zst, "rb") as f:
                    decompressed = self._dctx.decompress(f.read())
                    data = orjson.loads(decompressed)
                    return ReplayDetailDTO.model_validate(data)
            except Exception:  # noqa: BLE001
                pass

        # Fallback to .json
        filepath_json = self.replays_dir / f"{replay_id}.json"
        if filepath_json.exists():
            try:
                with open(filepath_json, "rb") as f:
                    data = orjson.loads(f.read())
                    return ReplayDetailDTO.model_validate(data)
            except Exception:  # noqa: BLE001
                pass

        return None

    def delete_replay(self, replay_id: str) -> bool:
        deleted = False
        for ext in [".zst", ".json"]:
            fp = self.replays_dir / f"{replay_id}{ext}"
            if fp.exists():
                fp.unlink()
                deleted = True

        with self._get_connection() as conn:
            conn.execute("DELETE FROM replays_index WHERE replay_id = ?", (replay_id,))

        return deleted
