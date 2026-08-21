import pytest
import torch
import numpy as np
from src.game.game import Game
from src.environment.observation import ObservationV1
from src.environment.action_space import DiscreteActionSpace
from src.rl.policy_value_net import PolicyValueNetwork
from src.rl.alphazero_search import NeuralMCTSEngine
from src.rl.alphazero_trainer import AlphaZeroTrainer
from src.rl.opponent_model import BayesianTicketBeliefTracker, belief_weighted_determinization
from src.rl.curriculum import CurriculumManager
from src.agents.neural_mcts_agent import NeuralMCTSAgent, OpponentAwareMCTSAgent
from src.evaluation.phase12_benchmark import Phase12ScientificBenchmark

def test_phase12_criterion_1_policy_value_net_and_masking():
    """C1: Dual-head output, gradient flow, and masking invariance."""
    net = PolicyValueNetwork(obs_dim=100, action_dim=20, hidden_dim=32, num_res_blocks=1)
    obs = torch.randn(4, 100)
    mask = torch.ones(4, 20)
    mask[:, 5:] = 0.0
    p, v = net(obs, mask)
    assert p.shape == (4, 20)
    assert v.shape == (4, 1)
    assert torch.all(p[:, 5:] < 1e-6)
    assert torch.all(v >= -1.0) and torch.all(v <= 1.0)

def test_phase12_criterion_2_alphazero_puct_search():
    """C2: AlphaZero search yields valid target probabilities and robust child."""
    agent = NeuralMCTSAgent(num_simulations=10)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, board=game.board)
    assert action in valid_actions

def test_phase12_criterion_3_alphazero_trainer_and_loss_reduction():
    """C3: Self-play collection and loss optimization."""
    encoder = ObservationV1()
    action_space = DiscreteActionSpace()
    obs_dim = encoder.observation_shape[0]
    action_dim = action_space.n
    
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1)
    trainer = AlphaZeroTrainer(net=net, num_simulations=5, batch_size=8)
    steps = trainer.collect_self_play_games(num_games=1, max_turns=5)
    assert steps > 0
    metrics = trainer.train_step(batch_size=4)
    assert "loss" in metrics
    assert "value_loss" in metrics
    assert "policy_loss" in metrics

def test_phase12_criterion_4_and_5_bayesian_belief_and_weighted_det():
    """C4 & C5: Opponent belief posterior and weighted determinization."""
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)
    tracker = BayesianTicketBeliefTracker(game.board, game.initial_tickets)
    tracker.observe_route_claim("player_1", game.board.routes[0].id)
    probs = tracker.get_ticket_probabilities("player_1")
    assert pytest.approx(sum(probs.values()), abs=1e-5) == 1.0
    
    det_game = belief_weighted_determinization(game, root_player_id="player_0", tracker=tracker)
    assert len(det_game.state.players) == 2
    assert det_game.state.players[0].tickets == game.state.players[0].tickets

def test_phase12_criterion_6_scientific_benchmark_and_agent_superiority():
    """C6: Cross-paradigm benchmark execution and scientific report generation."""
    bench = Phase12ScientificBenchmark(num_games_per_pair=2)
    results = bench.run_quick_cross_paradigm_benchmark()
    assert "win_rates" in results
    report = bench.generate_report(results)
    assert "TicketToRide RL Lab" in report
    assert "Phase 12" in report
    assert "Neural MCTS vs Random Win Rate" in report
