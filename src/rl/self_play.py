"""Historical policy pool management and matchmaking for self-play training."""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.agents.base_agent import BaseAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.recurrent_ppo_agent import RecurrentPPOAgent
from src.rl.lstm_ppo import RecurrentMaskedActorCritic
from src.rl.networks import MaskedActorCritic


@dataclass
class PolicySnapshot:
    """Frozen snapshot of an agent policy checkpoint."""

    generation: int
    step: int
    name: str
    state_dict: dict[str, torch.Tensor]
    is_recurrent: bool = False
    hidden_dim: int = 128
    lstm_hidden_dim: int = 128
    input_dim: int = 50
    action_dim: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyPool:
    """Historical pool of frozen policy checkpoints with capacity limits and retention."""

    max_size: int = 50
    _snapshots: list[PolicySnapshot] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[PolicySnapshot]:
        return list(self._snapshots)

    def add_policy(
        self,
        model: nn.Module,
        step: int,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot:
        gen = len(self._snapshots)
        snap_name = name or f"gen_{gen:03d}_step_{step}"
        is_recurrent = isinstance(model, RecurrentMaskedActorCritic) or hasattr(model, "lstm")

        # Clone state_dict to CPU
        cpu_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        hidden_dim = getattr(model, "hidden_dim", 128)
        lstm_hidden_dim = getattr(model, "lstm_hidden_dim", 128)
        input_dim = getattr(model, "input_dim", 50)
        action_dim = getattr(model, "action_dim", 10)

        if is_recurrent:
            if "encoder.0.weight" in cpu_state_dict:
                input_dim = cpu_state_dict["encoder.0.weight"].shape[1]
                hidden_dim = cpu_state_dict["encoder.0.weight"].shape[0]
            if "critic.weight" in cpu_state_dict:
                lstm_hidden_dim = cpu_state_dict["critic.weight"].shape[1]
            if "actor.weight" in cpu_state_dict:
                action_dim = cpu_state_dict["actor.weight"].shape[0]
        else:
            if "actor.0.weight" in cpu_state_dict:
                input_dim = cpu_state_dict["actor.0.weight"].shape[1]
                hidden_dim = cpu_state_dict["actor.0.weight"].shape[0]
            if "actor.4.weight" in cpu_state_dict:
                action_dim = cpu_state_dict["actor.4.weight"].shape[0]

        snapshot = PolicySnapshot(
            generation=gen,
            step=step,
            name=snap_name,
            state_dict=cpu_state_dict,
            is_recurrent=is_recurrent,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            input_dim=input_dim,
            action_dim=action_dim,
            metadata=metadata or {},
        )

        if len(self._snapshots) >= self.max_size:
            # Preserve generation 0 as the anchor baseline, remove second oldest
            if len(self._snapshots) > 1:
                self._snapshots.pop(1)
            else:
                self._snapshots.pop(0)

        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshot(self, index_or_name: int | str) -> PolicySnapshot:
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self._snapshots):
                return self._snapshots[index_or_name]
            raise IndexError(f"Snapshot index {index_or_name} out of bounds (size {len(self._snapshots)})")

        for snap in self._snapshots:
            if snap.name == index_or_name:
                return snap
        raise KeyError(f"Snapshot '{index_or_name}' not found in policy pool")

    def create_agent(
        self,
        index_or_name: int | str,
        board: Any = None,
        tickets: Any = None,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> BaseAgent:
        snapshot = self.get_snapshot(index_or_name)
        if snapshot.is_recurrent:
            model = RecurrentMaskedActorCritic(
                input_dim=snapshot.input_dim,
                action_dim=snapshot.action_dim,
                hidden_dim=snapshot.hidden_dim,
                lstm_hidden_dim=snapshot.lstm_hidden_dim,
            )
            model.load_state_dict(snapshot.state_dict)
            model.eval()
            return RecurrentPPOAgent(
                model=model,
                board=board,
                tickets=tickets,
                deterministic=deterministic,
                device=device,
                name=snapshot.name,
            )
        else:
            model = MaskedActorCritic(
                input_dim=snapshot.input_dim,
                action_dim=snapshot.action_dim,
                hidden_dim=snapshot.hidden_dim,
            )
            model.load_state_dict(snapshot.state_dict)
            model.eval()
            return PPOAgent(
                model=model,
                board=board,
                tickets=tickets,
                device=device,
                name=snapshot.name,
            )

    def save_pool(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        manifest = []
        for snap in self._snapshots:
            snap_file = f"{snap.name}.pt"
            snap_path = os.path.join(directory, snap_file)
            torch.save(
                {
                    "generation": snap.generation,
                    "step": snap.step,
                    "name": snap.name,
                    "state_dict": snap.state_dict,
                    "is_recurrent": snap.is_recurrent,
                    "hidden_dim": snap.hidden_dim,
                    "lstm_hidden_dim": snap.lstm_hidden_dim,
                    "input_dim": snap.input_dim,
                    "action_dim": snap.action_dim,
                    "metadata": snap.metadata,
                },
                snap_path,
            )
            manifest.append({
                "name": snap.name,
                "generation": snap.generation,
                "step": snap.step,
                "file": snap_file,
            })
        with open(os.path.join(directory, "pool_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def load_pool(self, directory: str) -> None:
        manifest_path = os.path.join(directory, "pool_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self._snapshots.clear()
        for item in manifest:
            snap_path = os.path.join(directory, item["file"])
            ckpt = torch.load(snap_path, map_location="cpu")
            snapshot = PolicySnapshot(
                generation=ckpt["generation"],
                step=ckpt["step"],
                name=ckpt["name"],
                state_dict=ckpt["state_dict"],
                is_recurrent=ckpt.get("is_recurrent", False),
                hidden_dim=ckpt.get("hidden_dim", 128),
                lstm_hidden_dim=ckpt.get("lstm_hidden_dim", 128),
                input_dim=ckpt.get("input_dim", 50),
                action_dim=ckpt.get("action_dim", 10),
                metadata=ckpt.get("metadata", {}),
            )
            self._snapshots.append(snapshot)
