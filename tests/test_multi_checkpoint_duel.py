import os
import torch
import pytest
from src.api.game_service import GameService
from src.api.schemas import GameSessionCreateRequest
from src.api.tournament_service import TournamentService
from src.rl.networks import MaskedActorCritic


def test_game_service_multi_checkpoint_names(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    
    net1 = MaskedActorCritic(input_dim=180, action_dim=150)
    net2 = MaskedActorCritic(input_dim=180, action_dim=150)
    p1_path = str(ckpt_dir / "ppo_gen_10.pt")
    p2_path = str(ckpt_dir / "ppo_gen_50.pt")
    torch.save({"actor_critic_state_dict": net1.state_dict(), "total_timesteps": 1000}, p1_path)
    torch.save({"actor_critic_state_dict": net2.state_dict(), "total_timesteps": 5000}, p2_path)
    
    service = GameService()
    req = GameSessionCreateRequest(
        map_name="mini",
        player_types=["ppo", "ppo"],
        player_checkpoints=[p1_path, p2_path],
        seed=42,
    )
    state = service.create_session(req)
    assert len(state.players) == 2
    assert "ppo_gen_10" in state.players[0].name
    assert "ppo_gen_50" in state.players[1].name


def test_tournament_service_discovers_checkpoints(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    p1_path = str(ckpt_dir / "ppo_experiment_test.pt")
    net = MaskedActorCritic(input_dim=180, action_dim=150)
    torch.save({"actor_critic_state_dict": net.state_dict(), "total_timesteps": 1000}, p1_path)
    
    service = TournamentService()
    participants = service.get_available_participants(ckpt_dir=str(ckpt_dir))
    checkpoint_participants = [p for p in participants if p.category == "checkpoint"]
    assert len(checkpoint_participants) >= 1
    assert any("PPO" in p.name for p in checkpoint_participants)
