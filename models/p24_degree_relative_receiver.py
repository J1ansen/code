"""Degree-relative prompt message receiver for P24 candidate topologies."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class P24DegreeRelativeReceiver(nn.Module):
    """Apply a bounded, residual node-wise budget to prompt-to-node edges.

    The receiver is an identity at epoch zero. It operates after the P23 graph
    has been constructed, so it cannot alter P23 feature selection or edges.
    """

    feature_dim = 6

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        hidden_dim = int(self.config.get("hidden_dim", 16))
        dropout = float(self.config.get("dropout", 0.10))
        self.alpha_max = float(self.config.get("alpha_max", 0.20))
        self.warmup_epochs = int(self.config.get("warmup_epochs", 50))
        self.scale_min = float(self.config.get("scale_min", 0.80))
        self.scale_max = float(self.config.get("scale_max", 1.20))
        self.degree_prior_strength = float(self.config.get("degree_prior_strength", 0.10))
        self.detach_features = bool(self.config.get("detach_features", True))

        if hidden_dim <= 0:
            raise ValueError("degree_relative_receiver.hidden_dim must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("degree_relative_receiver.warmup_epochs must be non-negative")
        if not 0.0 <= self.alpha_max <= 1.0:
            raise ValueError("degree_relative_receiver.alpha_max must be in [0, 1]")
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min:
            raise ValueError("degree_relative_receiver scale bounds are invalid")

        self.budget_predictor = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start exactly at the degree prior; epoch-zero warmup still makes the
        # complete receiver an identity.
        nn.init.zeros_(self.budget_predictor[-1].weight)
        nn.init.zeros_(self.budget_predictor[-1].bias)

    @staticmethod
    def _percentile_rank(value: torch.Tensor) -> torch.Tensor:
        if value.numel() <= 1:
            return torch.full_like(value, 0.5)
        order = torch.argsort(value.detach(), stable=True)
        rank = torch.empty_like(value)
        rank[order] = torch.linspace(0.0, 1.0, value.numel(), device=value.device, dtype=value.dtype)
        return rank

    @staticmethod
    def _robust_unit(value: torch.Tensor) -> torch.Tensor:
        positive = value[value > 0]
        if positive.numel() == 0:
            return torch.zeros_like(value)
        median = positive.median().clamp_min(1e-12)
        return (value / median).clamp(0.0, 4.0) / 4.0

    def _mix(self, epoch: int | None, reference: torch.Tensor) -> torch.Tensor:
        current_epoch = 0 if epoch is None else max(0, int(epoch))
        if self.warmup_epochs <= 0:
            value = 1.0 if current_epoch > 0 else 0.0
        else:
            value = min(1.0, current_epoch / float(self.warmup_epochs))
        return reference.new_tensor(value)

    def forward(
        self,
        *,
        static_weight: torch.Tensor,
        candidate_weight: torch.Tensor,
        edge_node: torch.Tensor,
        edge_direction: torch.Tensor,
        original_edge_index: torch.Tensor,
        epoch: int | None = None,
    ) -> dict[str, torch.Tensor]:
        num_nodes = (
            int(original_edge_index.max().item()) + 1
            if original_edge_index.numel() > 0
            else (int(edge_node.max().item()) + 1 if edge_node.numel() > 0 else 0)
        )
        one = candidate_weight.new_ones(num_nodes)
        zero = candidate_weight.new_tensor(0.0)
        if (
            not self.enabled
            or candidate_weight.numel() == 0
            or num_nodes == 0
            or original_edge_index.numel() == 0
        ):
            return {
                "edge_weight": candidate_weight,
                "node_scale": one,
                "mix_ratio": zero,
                "budget_loss": zero,
                "enabled": zero,
            }

        node = edge_node.to(device=candidate_weight.device, dtype=torch.long)
        direction = edge_direction.to(device=candidate_weight.device, dtype=torch.long)
        active = direction == 1
        if not bool(active.any()):
            return {
                "edge_weight": candidate_weight,
                "node_scale": one,
                "mix_ratio": zero,
                "budget_loss": zero,
                "enabled": candidate_weight.new_tensor(1.0),
            }

        original = original_edge_index.to(device=candidate_weight.device, dtype=torch.long)
        degree = candidate_weight.new_zeros(num_nodes)
        degree.index_add_(0, original[1], torch.ones(original.size(1), device=degree.device, dtype=degree.dtype))
        degree_rank = self._percentile_rank(torch.log1p(degree))

        static_mass = candidate_weight.new_zeros(num_nodes)
        candidate_mass = candidate_weight.new_zeros(num_nodes)
        edge_count = candidate_weight.new_zeros(num_nodes)
        active_node = node[active]
        static_active = static_weight[active].detach().clamp_min(0.0)
        candidate_active = candidate_weight[active]
        static_mass.index_add_(0, active_node, static_active)
        candidate_mass.index_add_(0, active_node, candidate_active.detach().clamp_min(0.0))
        edge_count.index_add_(0, active_node, torch.ones_like(candidate_active))

        mass_ratio = (candidate_mass / static_mass.clamp_min(1e-12)).clamp(0.0, 2.0) / 2.0
        topology_delta = ((candidate_mass - static_mass).abs() / static_mass.clamp_min(1e-12)).clamp(0.0, 1.0)
        features = torch.stack(
            [
                degree_rank,
                degree_rank - 0.5,
                self._robust_unit(static_mass),
                mass_ratio,
                topology_delta,
                self._percentile_rank(edge_count),
            ],
            dim=-1,
        )
        if self.detach_features:
            features = features.detach()
        learned_adjustment = torch.tanh(self.budget_predictor(features).squeeze(-1))
        mix = self._mix(epoch, candidate_weight)
        log_scale = mix * (
            self.degree_prior_strength * (degree_rank - 0.5)
            + self.alpha_max * learned_adjustment
        )
        node_scale = torch.exp(log_scale).clamp(self.scale_min, self.scale_max)
        output = candidate_weight.clone()
        output[active] = candidate_weight[active] * node_scale[active_node]
        active_nodes = edge_count > 0
        budget_loss = (
            (node_scale[active_nodes] - 1.0).pow(2).mean()
            if bool(active_nodes.any())
            else zero
        )
        return {
            "edge_weight": output,
            "node_scale": node_scale,
            "mix_ratio": mix,
            "budget_loss": budget_loss,
            "enabled": candidate_weight.new_tensor(1.0),
            "degree_rank": degree_rank,
            "static_mass": static_mass,
            "candidate_mass_before_receiver": candidate_mass,
            "active_node_mask": active_nodes,
        }
