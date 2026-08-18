import numpy as np
import torch

from src.rl.networks import MaskedActorCritic, MaskedQNetwork


def test_masked_q_network_forward_and_action_selection() -> None:
    input_dim = 20
    action_dim = 5
    q_net = MaskedQNetwork(input_dim=input_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn(input_dim)
    mask = np.array([1, 0, 1, 0, 0], dtype=np.int8)

    # Deterministic greedy action selection (epsilon=0.0)
    action = q_net.select_action(obs, mask, epsilon=0.0)
    assert action in [0, 2]

    # Random exploration with mask (epsilon=1.0)
    actions = [q_net.select_action(obs, mask, epsilon=1.0) for _ in range(50)]
    assert all(a in [0, 2] for a in actions)
    assert 0 in actions and 2 in actions


def test_masked_actor_critic_distribution_and_masking() -> None:
    input_dim = 20
    action_dim = 6
    ac = MaskedActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn(2, input_dim)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]], dtype=torch.bool)

    actions, log_probs, entropy, values = ac.get_action_and_value(obs, action_mask=mask)

    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert entropy.shape == (2,)
    assert values.shape == (2, 1)

    assert actions[0].item() in [0, 2]
    assert actions[1].item() in [1, 3]

    # Test evaluated log_prob on given action
    given_actions = torch.tensor([0, 1])
    _, eval_log_probs, _, _ = ac.get_action_and_value(obs, action_mask=mask, action=given_actions)
    assert torch.all(torch.isfinite(eval_log_probs))


def test_masked_actor_critic_deterministic() -> None:
    input_dim = 15
    action_dim = 4
    ac = MaskedActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=32)

    obs = torch.randn(1, input_dim)
    mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.bool)

    # Deterministic mode chooses argmax of valid logits
    action, _, _, _ = ac.get_action_and_value(obs, action_mask=mask, deterministic=True)
    assert action.item() in [1, 3]
