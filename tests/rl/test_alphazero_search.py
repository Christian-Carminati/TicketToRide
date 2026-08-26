import numpy as np
import pytest
from src.environment.env import TicketToRideEnv
from src.game.game import Game
from src.rl.alphazero_search import NeuralMCTSEngine, NeuralMCTSNode
from src.rl.policy_value_net import PolicyValueNetwork


def test_neural_mcts_node_puct_selection():
    root = NeuralMCTSNode(state_player="player_0")
    # Action 0: high prior 0.8, 0 visits
    # Action 1: low prior 0.2, 0 visits
    root.priors = {0: 0.8, 1: 0.2}
    root.legal_actions = [0, 1]

    # Selection should pick action 0 (highest UCT due to prior)
    best_a = root.select_puct_action(c_puct=1.5)
    assert best_a == 0

    # Simulate visiting action 0 multiple times with lower Q
    root.visits[0] = 10
    root.values[0] = -5.0  # Q = -0.5
    root.visits[1] = 1
    root.values[1] = 0.5  # Q = 0.5

    # Selection should now favor action 1
    best_a2 = root.select_puct_action(c_puct=1.5)
    assert best_a2 == 1


def test_neural_mcts_engine_search_execution():
    env = TicketToRideEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = PolicyValueNetwork(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1
    )
    engine = NeuralMCTSEngine(net=net, num_simulations=20, c_puct=1.5)

    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    best_action, visit_probs, root_value = engine.search(
        game, player_id="player_0", is_root_exploration=True
    )

    assert isinstance(best_action, int)
    assert 0 <= best_action < action_dim
    assert isinstance(visit_probs, np.ndarray)
    assert visit_probs.shape == (action_dim,)
    assert pytest.approx(visit_probs.sum(), abs=1e-5) == 1.0
    assert -1.0 <= root_value <= 1.0

    # Check that chosen action has visit_probs > 0
    assert visit_probs[best_action] > 0.0


def test_neural_mcts_engine_deterministic_reproducibility():
    env = TicketToRideEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = PolicyValueNetwork(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1
    )
    NeuralMCTSEngine(net=net, num_simulations=20, c_puct=1.5)

    game1 = Game(num_players=2, seed=123)
    game1.reset(seed=123)

    game2 = Game(num_players=2, seed=123)
    game2.reset(seed=123)

    engine1 = NeuralMCTSEngine(net=net, num_simulations=20, c_puct=1.5, seed=42)
    engine2 = NeuralMCTSEngine(net=net, num_simulations=20, c_puct=1.5, seed=42)

    np.random.seed(42)
    a1, pi1, v1 = engine1.search(game1, player_id="player_0", is_root_exploration=False)

    np.random.seed(42)
    a2, pi2, v2 = engine2.search(game2, player_id="player_0", is_root_exploration=False)

    assert a1 == a2
    np.testing.assert_allclose(pi1, pi2, atol=1e-5)
    assert pytest.approx(v1, abs=1e-5) == v2


def test_subtree_reuse_preserves_accumulated_visits():
    env = TicketToRideEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, num_res_blocks=1)
    engine = NeuralMCTSEngine(net=net, num_simulations=15, c_puct=1.5)

    game = Game(num_players=2, seed=42)
    game.reset(seed=42)

    best_act, pi, val, root_node = engine.search_with_root(game, "player_0", is_root_exploration=False)
    assert root_node.total_visits >= 15
    assert best_act in root_node.children

    child_node = root_node.children[best_act]
    child_visits_before = child_node.total_visits
    game.step(engine.action_space.to_action(best_act))

    best_act_2, pi_2, val_2, root_node_2 = engine.search_with_root(
        game, "player_1", is_root_exploration=False, previous_root=root_node, action_taken=best_act
    )
    assert root_node_2.total_visits >= child_visits_before

