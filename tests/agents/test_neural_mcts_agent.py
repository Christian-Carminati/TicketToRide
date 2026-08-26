from src.agents.neural_mcts_agent import NeuralMCTSAgent, OpponentAwareMCTSAgent
from src.game.game import Game


def test_neural_mcts_agent_acts_valid():
    agent = NeuralMCTSAgent(num_simulations=15)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, board=game.board)
    assert action in valid_actions

    # Execute action
    game.step(action)
    assert len(game.state.players) == 2


def test_opponent_aware_mcts_agent_tracks_and_acts():
    agent = OpponentAwareMCTSAgent(num_simulations=15)
    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    valid_actions = game.valid_actions()
    action = agent.act(game.state, valid_actions, board=game.board)
    assert action in valid_actions

    game.step(action)
    assert len(game.state.players) == 2
