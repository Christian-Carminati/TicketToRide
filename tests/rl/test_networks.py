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


def test_masked_actor_critic_orthogonal_initialization() -> None:
    input_dim = 64
    action_dim = 10
    hidden_dim = 128
    model = MaskedActorCritic(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        orthogonal_init=True,
    )

    # Test hidden layer weights are orthogonal and biases are zero
    # For layer 0 (128, 64), rows > cols => W^T W = gain^2 * I_cols
    w0 = model.actor[0].weight.data
    gram0 = torch.mm(w0.t(), w0)
    assert torch.allclose(gram0, 2.0 * torch.eye(64), atol=1e-3)
    assert torch.allclose(model.actor[0].bias.data, torch.zeros_like(model.actor[0].bias.data))

    # For layer 2 (128, 128), square => W W^T = gain^2 * I
    w2 = model.actor[2].weight.data
    gram2 = torch.mm(w2, w2.t())
    assert torch.allclose(gram2, 2.0 * torch.eye(128), atol=1e-3)
    assert torch.allclose(model.actor[2].bias.data, torch.zeros_like(model.actor[2].bias.data))

    # Test actor output head (10, 128) has gain 0.01 => W W^T = 0.01^2 * I_10
    actor_out = model.actor[4]
    w_act = actor_out.weight.data
    gram_act = torch.mm(w_act, w_act.t())
    assert torch.allclose(gram_act, (0.01**2) * torch.eye(10), atol=1e-5)
    assert torch.allclose(actor_out.bias.data, torch.zeros_like(actor_out.bias.data))

    # Test critic output head (1, 128) has gain 1.0 => W W^T = 1.0^2 * I_1
    critic_out = model.critic[4]
    w_crit = critic_out.weight.data
    gram_crit = torch.mm(w_crit, w_crit.t())
    assert torch.allclose(gram_crit, (1.0**2) * torch.eye(1), atol=1e-3)
    assert torch.allclose(critic_out.bias.data, torch.zeros_like(critic_out.bias.data))


