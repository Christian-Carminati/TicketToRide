"""BrainService: Performs deep neural network introspection for the Web Lab."""

from typing import Any
import numpy as np
import torch
from torch import nn

from src.api.schemas import BrainInspectionDTO, LayerActivationDTO
from src.rl.networks import MaskedActorCritic, MaskedQNetwork


class BrainService:
    """Extracts internal layer activations, logits, Q-values, and action distributions."""

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
                        values=out_flat[:32].tolist(),  # First 32 elements sample
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

        # Softmax over masked Q-values (scaled with temperature 1.0 for probability visualization)
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
