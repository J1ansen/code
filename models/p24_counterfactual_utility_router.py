"""Counterfactual utility routing for P24 prompt topology adaptation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class P24CounterfactualUtilityRouter(nn.Module):
    """Predict whether a node benefits from candidate prompt-edge weights.

    All inputs are available without labels at inference time. Training labels
    are only used by the runner to supervise the predicted utility on held-out
    support-query folds.
    """

    feature_dim = 12

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        hidden_dim = int(self.config.get("hidden_dim", 32))
        dropout = float(self.config.get("dropout", 0.10))
        self.route_start_epoch = int(self.config.get("route_start_epoch", 100))
        self.deployment_warmup_epochs = int(self.config.get("deployment_warmup_epochs", 50))
        self.max_adapt = float(self.config.get("max_adapt", 0.30))
        self.gain_margin = float(self.config.get("gain_margin", 0.0005))
        self.gain_temperature = float(self.config.get("gain_temperature", 0.005))
        self.gain_init = float(self.config.get("gain_init", -0.002))
        self.detach_candidate_weight = bool(self.config.get("detach_candidate_weight", True))
        self.hub_quantile = float(self.config.get("hub_quantile", 0.80))

        if hidden_dim <= 0 or self.deployment_warmup_epochs < 0:
            raise ValueError("counterfactual router dimensions and warmup must be non-negative")
        if not 0.0 <= self.max_adapt <= 1.0:
            raise ValueError("counterfactual router max_adapt must be in [0, 1]")
        if self.gain_temperature <= 0.0:
            raise ValueError("counterfactual router gain_temperature must be positive")
        if not 0.0 <= self.hub_quantile <= 1.0:
            raise ValueError("counterfactual router hub_quantile must be in [0, 1]")

        self.predictor = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.constant_(self.predictor[-1].bias, self.gain_init)

    @staticmethod
    def _entropy(probability: torch.Tensor) -> torch.Tensor:
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
        normalizer = probability.new_tensor(float(max(2, probability.size(-1)))).log()
        return entropy / normalizer.clamp_min(1e-12)

    @staticmethod
    def _margin(probability: torch.Tensor) -> torch.Tensor:
        if probability.size(-1) <= 1:
            return probability.squeeze(-1)
        top = probability.topk(k=2, dim=-1).values
        return top[:, 0] - top[:, 1]

    def _topology_features(
        self,
        *,
        static_weight: torch.Tensor,
        candidate_weight: torch.Tensor,
        edge_node: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_direction: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = static_weight.device
        dtype = static_weight.dtype
        node = edge_node.to(device=device, dtype=torch.long)
        feature = edge_feature.to(device=device, dtype=torch.long)
        direction = edge_direction.to(device=device, dtype=torch.long)
        active = direction == 1
        active_node = node[active]
        base = static_weight[active].detach().clamp_min(0.0)
        candidate = candidate_weight[active].detach()

        base_mass = static_weight.new_zeros(num_nodes)
        abs_delta = static_weight.new_zeros(num_nodes)
        signed_delta = static_weight.new_zeros(num_nodes)
        base_mass.index_add_(0, active_node, base)
        abs_delta.index_add_(0, active_node, (candidate - base).abs())
        signed_delta.index_add_(0, active_node, candidate - base)
        topology_abs_delta = abs_delta / base_mass.clamp_min(1e-12)
        topology_signed_delta = signed_delta / base_mass.clamp_min(1e-12)

        feature_count = int(feature.max().item()) + 1 if feature.numel() else 0
        usage = static_weight.new_zeros(feature_count)
        if bool(active.any()) and feature_count > 0:
            usage.index_add_(0, feature[active], torch.ones_like(base, dtype=dtype))
            used = usage > 0.0
            threshold = torch.quantile(usage[used], self.hub_quantile) if bool(used.any()) else usage.new_tensor(0.0)
            hub_feature = usage >= threshold
            hub_mass = static_weight.new_zeros(num_nodes)
            active_hub = hub_feature[feature[active]]
            if bool(active_hub.any()):
                hub_mass.index_add_(0, active_node[active_hub], base[active_hub])
            hub_exposure = hub_mass / base_mass.clamp_min(1e-12)
        else:
            hub_exposure = static_weight.new_zeros(num_nodes)
        return topology_abs_delta, topology_signed_delta, hub_exposure

    def forward(
        self,
        *,
        static_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        static_weight: torch.Tensor,
        candidate_weight: torch.Tensor,
        edge_node: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_direction: torch.Tensor,
        epoch: int | None,
    ) -> dict[str, torch.Tensor]:
        static_probability = F.softmax(static_logits.detach(), dim=-1)
        candidate_probability = F.softmax(candidate_logits.detach(), dim=-1)
        midpoint = 0.5 * (static_probability + candidate_probability)
        js_divergence = 0.5 * (
            (static_probability * (static_probability.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=-1)
            + (candidate_probability * (candidate_probability.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=-1)
        )
        topology_abs_delta, topology_signed_delta, hub_exposure = self._topology_features(
            static_weight=static_weight,
            candidate_weight=candidate_weight,
            edge_node=edge_node,
            edge_feature=edge_feature,
            edge_direction=edge_direction,
            num_nodes=int(static_logits.size(0)),
        )
        features = torch.stack(
            [
                self._entropy(static_probability),
                self._entropy(candidate_probability),
                self._margin(static_probability),
                self._margin(candidate_probability),
                static_probability.max(dim=-1).values,
                candidate_probability.max(dim=-1).values,
                js_divergence,
                (static_probability - candidate_probability).abs().sum(dim=-1),
                (static_probability.argmax(dim=-1) != candidate_probability.argmax(dim=-1)).to(static_probability.dtype),
                topology_abs_delta,
                topology_signed_delta,
                hub_exposure,
            ],
            dim=-1,
        )
        predicted_gain = self.predictor(features).squeeze(-1)
        current_epoch = 0 if epoch is None else int(epoch)
        route_stage_active = self.enabled and current_epoch > self.route_start_epoch
        if not route_stage_active:
            deployment_mix = predicted_gain.new_tensor(0.0)
            adapt_ratio = torch.zeros_like(predicted_gain)
        else:
            if self.deployment_warmup_epochs <= 0:
                deployment_mix = predicted_gain.new_tensor(1.0)
            else:
                deployment_mix = predicted_gain.new_tensor(
                    min(1.0, max(0.0, (current_epoch - self.route_start_epoch) / self.deployment_warmup_epochs))
                )
            raw_ratio = torch.sigmoid(
                (predicted_gain - self.gain_margin) / self.gain_temperature
            )
            adapt_ratio = self.max_adapt * deployment_mix * raw_ratio
        return {
            "predicted_gain": predicted_gain,
            "adapt_ratio": adapt_ratio,
            "p23_ratio": 1.0 - adapt_ratio,
            "features": features,
            "deployment_mix": deployment_mix,
            "route_stage_active": predicted_gain.new_tensor(float(route_stage_active)),
            "topology_abs_delta": topology_abs_delta,
            "hub_exposure": hub_exposure,
        }

    def blend_weights(
        self,
        *,
        static_weight: torch.Tensor,
        candidate_weight: torch.Tensor,
        edge_node: torch.Tensor,
        edge_direction: torch.Tensor,
        adapt_ratio: torch.Tensor,
    ) -> torch.Tensor:
        node = edge_node.to(device=static_weight.device, dtype=torch.long)
        direction = edge_direction.to(device=static_weight.device, dtype=torch.long)
        active = direction == 1
        candidate = candidate_weight.detach() if self.detach_candidate_weight else candidate_weight
        output = static_weight.clone()
        ratio = adapt_ratio[node[active]]
        output[active] = (1.0 - ratio) * static_weight[active] + ratio * candidate[active]
        return output
