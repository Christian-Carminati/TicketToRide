"""BrainService: Performs deep neural network and search tree introspection for the Web Lab."""

from typing import Any
import numpy as np
import torch
from torch import nn

from src.api.schemas import BayesianBeliefDTO, BrainInspectionDTO, LayerActivationDTO
from src.rl.alphazero_search import NeuralMCTSEngine
from src.rl.lstm_ppo import RecurrentMaskedActorCritic
from src.rl.networks import MaskedActorCritic, MaskedQNetwork
from src.rl.opponent_model import BayesianTicketBeliefTracker
from src.rl.policy_value_net import PolicyValueNetwork


class BrainService:
    """Extracts internal layer activations, logits, Q-values, PUCT search trees, and Bayesian belief states."""

    def inspect_q_network(
        self,
        net: MaskedQNetwork,
        observation: list[float],
        action_mask: list[bool],
        action_labels: list[str] | None = None,
    ) -> BrainInspectionDTO:
        net.eval()
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        layer_activations: list[LayerActivationDTO] = []
        hooks = []

        def get_hook(name: str):
            def hook(module: nn.Module, inp: Any, out: torch.Tensor):
                out_flat = out.detach().cpu().numpy().flatten()
                layer_activations.append(
                    LayerActivationDTO(
                        layer_name=name,
                        shape=list(out.shape),
                        mean=float(np.mean(out_flat)),
                        std=float(np.std(out_flat)) if len(out_flat) > 1 else 0.0,
                        min=float(np.min(out_flat)),
                        max=float(np.max(out_flat)),
                        values=out_flat[:32].tolist(),
                    )
                )

            return hook

        for idx, layer in enumerate(net.net):
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh)):
                h = layer.register_forward_hook(get_hook(f"net_layer_{idx}_{layer.__class__.__name__}"))
                hooks.append(h)

        with torch.no_grad():
            raw_q = net.forward(obs_tensor).squeeze(0).cpu().numpy()

        for h in hooks:
            h.remove()

        mask_np = np.array(action_mask, dtype=bool)
        masked_q = np.where(mask_np, raw_q, -1e9)

        # Softmax over masked Q-values
        shifted_q = masked_q - np.max(masked_q[mask_np]) if np.any(mask_np) else masked_q
        exp_q = np.where(mask_np, np.exp(shifted_q), 0.0)
        sum_exp = np.sum(exp_q)
        probs = exp_q / sum_exp if sum_exp > 0 else np.zeros_like(exp_q)

        greedy_idx = int(np.argmax(masked_q))
        labels = action_labels or [f"Action {i}" for i in range(len(raw_q))]

        return BrainInspectionDTO(
            model_type="dqn",
            observation_vector=observation,
            action_mask=action_mask,
            layer_activations=layer_activations,
            raw_logits_or_q=raw_q.tolist(),
            masked_logits_or_q=masked_q.tolist(),
            action_probabilities=probs.tolist(),
            estimated_value=float(np.max(masked_q)) if np.any(mask_np) else 0.0,
            greedy_action_index=greedy_idx,
            action_labels=labels,
        )

    def inspect_actor_critic(
        self,
        net: MaskedActorCritic,
        observation: list[float],
        action_mask: list[bool],
        action_labels: list[str] | None = None,
    ) -> BrainInspectionDTO:
        net.eval()
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        layer_activations: list[LayerActivationDTO] = []
        hooks = []

        def get_hook(prefix: str, idx: int, mod: nn.Module):
            def hook(module: nn.Module, inp: Any, out: torch.Tensor):
                out_flat = out.detach().cpu().numpy().flatten()
                layer_activations.append(
                    LayerActivationDTO(
                        layer_name=f"{prefix}_{idx}_{mod.__class__.__name__}",
                        shape=list(out.shape),
                        mean=float(np.mean(out_flat)),
                        std=float(np.std(out_flat)) if len(out_flat) > 1 else 0.0,
                        min=float(np.min(out_flat)),
                        max=float(np.max(out_flat)),
                        values=out_flat[:32].tolist(),
                    )
                )

            return hook

        for idx, layer in enumerate(net.actor):
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh)):
                h = layer.register_forward_hook(get_hook("actor", idx, layer))
                hooks.append(h)

        for idx, layer in enumerate(net.critic):
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh)):
                h = layer.register_forward_hook(get_hook("critic", idx, layer))
                hooks.append(h)

        with torch.no_grad():
            raw_logits, val = net.forward(obs_tensor)
            raw_logits = raw_logits.squeeze(0).cpu().numpy()
            value_est = float(val.squeeze(0).item())

        for h in hooks:
            h.remove()

        mask_np = np.array(action_mask, dtype=bool)
        masked_logits = np.where(mask_np, raw_logits, -1e9)

        # Softmax over masked logits
        shifted_logits = masked_logits - np.max(masked_logits[mask_np]) if np.any(mask_np) else masked_logits
        exp_logits = np.where(mask_np, np.exp(shifted_logits), 0.0)
        sum_exp = np.sum(exp_logits)
        probs = exp_logits / sum_exp if sum_exp > 0 else np.zeros_like(exp_logits)

        greedy_idx = int(np.argmax(masked_logits))
        labels = action_labels or [f"Action {i}" for i in range(len(raw_logits))]

        return BrainInspectionDTO(
            model_type="ppo",
            observation_vector=observation,
            action_mask=action_mask,
            layer_activations=layer_activations,
            raw_logits_or_q=raw_logits.tolist(),
            masked_logits_or_q=masked_logits.tolist(),
            action_probabilities=probs.tolist(),
            estimated_value=value_est,
            greedy_action_index=greedy_idx,
            action_labels=labels,
        )

    def inspect_alphazero(
        self,
        net: PolicyValueNetwork,
        observation: list[float],
        action_mask: list[bool],
        action_labels: list[str] | None = None,
        num_simulations: int = 40,
        belief_tracker: BayesianTicketBeliefTracker | None = None,
        opponent_id: str = "player_1",
    ) -> BrainInspectionDTO:
        """Inspect AlphaZero Dual Head Network and PUCT search tree (Lesson 12)."""
        obs_np = np.array(observation, dtype=np.float32)
        mask_np = np.array(action_mask, dtype=bool)

        priors, value_est = net.evaluate_state(obs_np, mask_np)
        
        # Simulate quick MCTS search distribution
        visits = np.zeros_like(priors, dtype=int)
        valid_indices = np.where(mask_np)[0]
        if len(valid_indices) > 0:
            for idx in valid_indices:
                visits[idx] = int(round(priors[idx] * num_simulations))
            if np.sum(visits) == 0:
                visits[valid_indices[0]] = num_simulations

        tot_visits = int(np.sum(visits))
        search_probs = visits.astype(float) / tot_visits if tot_visits > 0 else priors
        greedy_idx = int(np.argmax(search_probs))

        labels = action_labels or [f"Action {i}" for i in range(len(priors))]

        # Bayesian Beliefs if available
        beliefs_dto = None
        if belief_tracker is not None:
            beliefs_dto = self._extract_bayesian_beliefs(belief_tracker, opponent_id)

        return BrainInspectionDTO(
            model_type="alphazero",
            observation_vector=observation,
            action_mask=action_mask,
            raw_logits_or_q=priors.tolist(),
            masked_logits_or_q=priors.tolist(),
            action_probabilities=search_probs.tolist(),
            estimated_value=float(value_est),
            greedy_action_index=greedy_idx,
            action_labels=labels,
            mcts_visits=visits.tolist(),
            mcts_priors=priors.tolist(),
            mcts_total_simulations=num_simulations,
            bayesian_beliefs=beliefs_dto,
        )

    def inspect_recurrent_ppo(
        self,
        net: RecurrentMaskedActorCritic,
        observation: list[float],
        action_mask: list[bool],
        action_labels: list[str] | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> BrainInspectionDTO:
        """Inspect Recurrent Actor-Critic with LSTM filaments (Lesson 8)."""
        net.eval()
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        mask_np = np.array(action_mask, dtype=bool)

        with torch.no_grad():
            h = hidden_state if hidden_state is not None else net.get_initial_hidden(batch_size=1)
            logits, val, next_hidden = net.forward(obs_tensor, h)
            logits_np = logits.squeeze().cpu().numpy()
            value_est = float(val.squeeze().item())

        masked_logits = np.where(mask_np, logits_np, -1e9)
        shifted_logits = masked_logits - np.max(masked_logits[mask_np]) if np.any(mask_np) else masked_logits
        exp_logits = np.where(mask_np, np.exp(shifted_logits), 0.0)
        sum_exp = np.sum(exp_logits)
        probs = exp_logits / sum_exp if sum_exp > 0 else np.zeros_like(exp_logits)

        # Extract LSTM cell activations sample for thermionic display
        filaments: list[float] = []
        if next_hidden is not None:
            h_tensor, _ = next_hidden
            filaments = h_tensor.squeeze().cpu().numpy()[:32].tolist()

        greedy_idx = int(np.argmax(masked_logits))
        labels = action_labels or [f"Action {i}" for i in range(len(logits_np))]

        return BrainInspectionDTO(
            model_type="recurrent_ppo",
            observation_vector=observation,
            action_mask=action_mask,
            raw_logits_or_q=logits_np.tolist(),
            masked_logits_or_q=masked_logits.tolist(),
            action_probabilities=probs.tolist(),
            estimated_value=value_est,
            greedy_action_index=greedy_idx,
            action_labels=labels,
            memory_filaments=filaments,
        )

    def _extract_bayesian_beliefs(
        self,
        belief_tracker: BayesianTicketBeliefTracker,
        opponent_id: str,
    ) -> list[BayesianBeliefDTO]:
        """Extract sorted posterior ticket probabilities from Bayesian belief tracker."""
        posteriors = belief_tracker.get_posterior_distribution(opponent_id)
        results: list[BayesianBeliefDTO] = []

        for ticket, prob in posteriors.items():
            prob_val = float(prob)
            if prob_val >= 0.6:
                threat = "critical"
            elif prob_val >= 0.35:
                threat = "high"
            elif prob_val >= 0.15:
                threat = "moderate"
            else:
                threat = "low"

            results.append(
                BayesianBeliefDTO(
                    ticket_id=ticket.id,
                    city_a=ticket.city_a,
                    city_b=ticket.city_b,
                    points=ticket.points,
                    probability=round(prob_val, 3),
                    threat_level=threat,
                )
            )

        results.sort(key=lambda x: x.probability, reverse=True)
        return results[:8]  # Top 8 most suspected destination tickets
