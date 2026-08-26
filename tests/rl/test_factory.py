import pytest
from src.environment.env import TicketToRideEnv
from src.game.maps import create_synthetic_mini_board
from src.rl.alphazero_trainer import AlphaZeroTrainer
from src.rl.dqn import MaskedDQNTrainer
from src.rl.factory import TrainerFactory
from src.rl.lstm_ppo import MaskedRecurrentPPOTrainer
from src.rl.ppo import MaskedPPOTrainer
from src.rl.self_play import SelfPlayPPOTrainer


def test_factory_creates_dqn():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets)
    trainer = TrainerFactory.create("dqn", env, config={"batch_size": 16})
    assert isinstance(trainer, MaskedDQNTrainer)


def test_factory_creates_ppo():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets)
    trainer = TrainerFactory.create("ppo", env, config={"rollout_steps": 32})
    assert isinstance(trainer, MaskedPPOTrainer)


def test_factory_creates_self_play_ppo():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets)
    trainer = TrainerFactory.create("ppo", env, config={"rollout_steps": 32, "self_play": True})
    assert isinstance(trainer, SelfPlayPPOTrainer)


def test_factory_creates_recurrent_ppo():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets)
    trainer = TrainerFactory.create("recurrent_ppo", env, config={"rollout_steps": 32})
    assert isinstance(trainer, MaskedRecurrentPPOTrainer)


def test_factory_creates_alphazero():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets)
    trainer = TrainerFactory.create("alphazero", env, config={"num_simulations": 5})
    assert isinstance(trainer, AlphaZeroTrainer)


def test_factory_invalid_algo_raises():
    board, tickets = create_synthetic_mini_board()
    env = TicketToRideEnv(board=board, tickets_deck=tickets)
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        TrainerFactory.create("unknown_xyz", env)
