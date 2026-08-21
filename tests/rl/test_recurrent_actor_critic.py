import torch

from src.rl.lstm_ppo import RecurrentMaskedActorCritic


def test_recurrent_actor_critic_shapes_single_step():
    model = RecurrentMaskedActorCritic(input_dim=50, action_dim=20, hidden_dim=64, lstm_hidden_dim=64)
    obs = torch.randn(1, 50)
    hidden = model.get_initial_hidden(batch_size=1)

    action, log_prob, entropy, value, new_hidden = model.get_action_and_value(obs, hidden)
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1, 1)
    assert new_hidden[0].shape == (1, 1, 64)
    assert new_hidden[1].shape == (1, 1, 64)


def test_recurrent_actor_critic_action_masking_enforcement():
    model = RecurrentMaskedActorCritic(input_dim=50, action_dim=5, hidden_dim=64, lstm_hidden_dim=64)
    obs = torch.randn(1, 50)
    hidden = model.get_initial_hidden(batch_size=1)

    # Only action index 3 is allowed
    mask = torch.tensor([[False, False, False, True, False]], dtype=torch.bool)
    action, log_prob, entropy, value, _ = model.get_action_and_value(obs, hidden, action_mask=mask)
    assert action.item() == 3


def test_recurrent_actor_critic_sequence_batch_forward():
    model = RecurrentMaskedActorCritic(input_dim=50, action_dim=20, hidden_dim=64, lstm_hidden_dim=64)
    batch_size = 4
    seq_len = 8
    obs_seq = torch.randn(batch_size, seq_len, 50)
    hidden = model.get_initial_hidden(batch_size=batch_size)

    logits, values, new_hidden = model.forward(obs_seq, hidden)
    assert logits.shape == (batch_size, seq_len, 20)
    assert values.shape == (batch_size, seq_len, 1)
    assert new_hidden[0].shape == (1, batch_size, 64)
    assert new_hidden[1].shape == (1, batch_size, 64)


def test_recurrent_actor_critic_deterministic_action():
    model = RecurrentMaskedActorCritic(input_dim=10, action_dim=4, hidden_dim=32, lstm_hidden_dim=32)
    obs = torch.randn(2, 10)
    hidden = model.get_initial_hidden(batch_size=2)
    mask = torch.tensor([
        [True, True, False, False],
        [False, False, True, True],
    ], dtype=torch.bool)

    action, log_prob, entropy, value, _ = model.get_action_and_value(
        obs, hidden, action_mask=mask, deterministic=True
    )
    assert action[0].item() in (0, 1)
    assert action[1].item() in (2, 3)


def test_recurrent_actor_critic_get_value():
    model = RecurrentMaskedActorCritic(input_dim=10, action_dim=4, hidden_dim=32, lstm_hidden_dim=32)
    obs = torch.randn(2, 10)
    hidden = model.get_initial_hidden(batch_size=2)

    val, new_hidden = model.get_value(obs, hidden)
    assert val.shape == (2, 1)
    assert new_hidden[0].shape == (1, 2, 32)
    assert new_hidden[1].shape == (1, 2, 32)


def test_recurrent_actor_critic_provided_action_evaluation():
    model = RecurrentMaskedActorCritic(input_dim=10, action_dim=4, hidden_dim=32, lstm_hidden_dim=32)
    obs = torch.randn(2, 10)
    hidden = model.get_initial_hidden(batch_size=2)
    provided_action = torch.tensor([1, 2], dtype=torch.int64)

    action, log_prob, entropy, value, _ = model.get_action_and_value(
        obs, hidden, action=provided_action
    )
    assert torch.equal(action, provided_action)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)


def test_recurrent_actor_critic_non_orthogonal_init():
    model = RecurrentMaskedActorCritic(
        input_dim=10, action_dim=4, hidden_dim=32, lstm_hidden_dim=32, orthogonal_init=False
    )
    obs = torch.randn(1, 10)
    hidden = model.get_initial_hidden(batch_size=1)
    action, log_prob, entropy, value, new_hidden = model.get_action_and_value(obs, hidden)
    assert action.shape == (1,)
