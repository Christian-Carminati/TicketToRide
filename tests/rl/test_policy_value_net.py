import pytest
import torch
import numpy as np
from pathlib import Path
from src.rl.policy_value_net import PolicyValueNetwork

def test_policy_value_net_forward_shapes():
    obs_dim = 150
    action_dim = 45
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64, num_res_blocks=1)
    
    batch_size = 8
    obs = torch.randn(batch_size, obs_dim)
    mask = torch.ones(batch_size, action_dim)
    mask[:, 10:] = 0.0  # only first 10 actions valid
    
    policy_probs, values = net(obs, mask)
    
    assert policy_probs.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, 1)
    
    # Check probabilities sum to 1
    sums = policy_probs.sum(dim=-1)
    np.testing.assert_allclose(sums.detach().numpy(), np.ones(batch_size), atol=1e-5)
    
    # Check masked actions have prob == 0.0
    assert torch.all(policy_probs[:, 10:] < 1e-6)
    
    # Value range in [-1, 1] due to tanh
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)

def test_policy_value_net_evaluate_state_numpy():
    obs_dim = 50
    action_dim = 12
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    mask = np.zeros(action_dim, dtype=np.float32)
    mask[0] = 1.0
    mask[3] = 1.0
    
    probs, val = net.evaluate_state(obs, mask)
    assert isinstance(probs, np.ndarray)
    assert isinstance(val, float)
    assert probs.shape == (action_dim,)
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0
    assert probs[1] == 0.0
    assert probs[0] > 0.0
    assert -1.0 <= val <= 1.0

def test_policy_value_net_save_load(tmp_path: Path):
    obs_dim = 40
    action_dim = 10
    net = PolicyValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)
    
    save_path = tmp_path / "pv_net.pt"
    net.save(save_path)
    
    loaded_net = PolicyValueNetwork.load(save_path)
    assert loaded_net.obs_dim == obs_dim
    assert loaded_net.action_dim == action_dim
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    p1, v1 = net.evaluate_state(obs)
    p2, v2 = loaded_net.evaluate_state(obs)
    np.testing.assert_allclose(p1, p2, atol=1e-6)
    assert pytest.approx(v1, abs=1e-6) == v2
