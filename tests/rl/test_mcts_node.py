"""Unit tests for MCTSNode, MCTSConfig, and UCT selection."""

import math
from src.game.action import Action, ActionType
from src.game.game import Game
from src.rl.mcts import MCTSConfig, MCTSNode, RolloutPolicyType


def test_mcts_node_initialization():
    """Verify default attributes of a new MCTSNode."""
    game = Game(num_players=2, seed=42)
    state = game.reset(seed=42)
    valid_actions = game.valid_actions()

    node = MCTSNode(
        state=state,
        parent=None,
        action=None,
        player_id="player_0",
        untried_actions=valid_actions,
    )

    assert node.visits == 0
    assert node.total_value == 0.0
    assert node.value == 0.0
    assert not node.is_fully_expanded()
    assert not node.is_terminal()
    assert len(node.untried_actions) == len(valid_actions)
    assert len(node.children) == 0


def test_mcts_node_expansion_and_update():
    """Verify node expansion removes action from untried and creates child."""
    game = Game(num_players=2, seed=42)
    state = game.reset(seed=42)
    valid_actions = game.valid_actions()

    root = MCTSNode(
        state=state,
        player_id="player_0",
        untried_actions=list(valid_actions),
    )

    act_to_try = root.untried_actions[0]
    next_game = game.clone()
    next_state = next_game.step(act_to_try)
    next_actions = next_game.valid_actions()

    child = root.expand(
        action=act_to_try,
        next_state=next_state,
        next_player_id="player_1",
        untried_actions=next_actions,
    )

    assert child.parent is root
    assert child.action == act_to_try
    assert act_to_try not in root.untried_actions
    assert act_to_try in root.children

    # Test update
    child.update(0.8)
    assert child.visits == 1
    assert child.total_value == 0.8
    assert child.value == 0.8

    root.update(0.8)
    assert root.visits == 1
    assert root.total_value == 0.8


def test_mcts_node_uct_selection():
    """Verify UCT selection selects child with optimal exploration/exploitation tradeoff."""
    game = Game(num_players=2, seed=42)
    state = game.reset(seed=42)

    root = MCTSNode(state=state, player_id="player_0", untried_actions=[])
    root.visits = 10

    # Create two children
    a1 = Action(action_type=ActionType.DRAW_HIDDEN_CARD)
    a2 = Action(action_type=ActionType.DRAW_VISIBLE_CARD, card_index=0)

    c1 = MCTSNode(state=state, parent=root, action=a1, player_id="player_0", untried_actions=[])
    c1.visits = 6
    c1.total_value = 4.0  # Q = 4/6 = 0.667

    c2 = MCTSNode(state=state, parent=root, action=a2, player_id="player_0", untried_actions=[])
    c2.visits = 4
    c2.total_value = 1.0  # Q = 1/4 = 0.25

    root.children[a1] = c1
    root.children[a2] = c2

    best_act, best_child = root.select_best_child(c_puct=1.414)
    assert best_act == a1
    assert best_child is c1
