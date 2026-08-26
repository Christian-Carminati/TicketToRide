"""Unit tests for BrainService neural network introspection."""

from src.api.brain_service import BrainService
from src.rl.networks import MaskedActorCritic, MaskedQNetwork


def test_brain_service_q_network():
    service = BrainService()
    net = MaskedQNetwork(input_dim=128, action_dim=56, hidden_dim=64)
    obs = [0.1] * 128
    mask = [True] * 56
    mask[2] = False  # Action 2 is masked

    inspection = service.inspect_q_network(net, obs, mask)
    assert inspection.model_type == "dqn"
    assert len(inspection.layer_activations) > 0
    assert inspection.action_probabilities[2] == 0.0
    assert len(inspection.raw_logits_or_q) == 56
    assert len(inspection.masked_logits_or_q) == 56
    assert inspection.greedy_action_index != 2


def test_brain_service_actor_critic():
    service = BrainService()
    net = MaskedActorCritic(input_dim=128, action_dim=56, hidden_dim=64)
    obs = [0.1] * 128
    mask = [True] * 56
    mask[0] = False  # Action 0 is masked

    inspection = service.inspect_actor_critic(net, obs, mask)
    assert inspection.model_type == "ppo"
    assert inspection.estimated_value is not None
    assert inspection.action_probabilities[0] == 0.0
    assert abs(sum(inspection.action_probabilities) - 1.0) < 1e-4
    assert len(inspection.layer_activations) >= 2  # Actor and Critic layers
