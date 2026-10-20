"""Conservative receiver for P23 topology under a trainable input projector."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class P23ProjectorPromptReceiver(nn.Module):
    """Inject leave-one-out feature-hub context as a bounded residual.

    The receiver deliberately has only one learned signed scale.  P23 decides
    which feature hubs and edges exist; this module only decides whether their
    aggregate context should be added to or subtracted from the no-prompt
    adapted representation.
    """

    def __init__(self, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.max_update_ratio = max(0.0, float(cfg.get("max_update_ratio", 0.10)))
        self.scale_cap = max(0.0, float(cfg.get("scale_cap", 1.0)))
        self.min_context_count = max(1, int(cfg.get("min_context_count", 1)))
        self.detach_context = bool(cfg.get("detach_context", False))
        self.base_cls_weight = max(0.0, float(cfg.get("base_cls_weight", 0.0)))
        scale_init = float(cfg.get("scale_init", 0.0))
        if self.scale_cap <= 0.0:
            raw_init = 0.0
        else:
            ratio = min(max(scale_init / self.scale_cap, -0.999999), 0.999999)
            raw_init = float(torch.atanh(torch.tensor(ratio)).item())
        if self.enabled:
            self.raw_scale = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.register_parameter("raw_scale", None)
        self.hidden_dim = int(hidden_dim)

    @property
    def signed_scale(self) -> torch.Tensor:
        if self.raw_scale is None:
            return torch.tensor(0.0)
        return self.raw_scale.tanh() * float(self.scale_cap)

    def forward(
        self,
        *,
        h_base: torch.Tensor,
        graph: dict[str, Any],
        prompt_edge_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if h_base.dim() != 2 or int(h_base.size(-1)) != self.hidden_dim:
            raise ValueError(
                f"Expected h_base [N, {self.hidden_dim}], got {tuple(h_base.shape)}"
            )
        num_nodes = int(h_base.size(0))
        node_index = graph["prompt_aggregation_node_index"].to(
            device=h_base.device, dtype=torch.long
        )
        feature_local = graph["prompt_aggregation_feature_local"].to(
            device=h_base.device, dtype=torch.long
        )
        aggregation_weight = graph["prompt_aggregation_weight"].to(
            device=h_base.device, dtype=h_base.dtype
        )
        num_prompts = int(graph["prompt_aggregation_denom"].numel())

        zero_update = h_base.new_zeros(h_base.shape)
        if not self.enabled or node_index.numel() == 0 or num_prompts == 0:
            return self._output(
                h_base=h_base,
                update=zero_update,
                context=h_base,
                connected=h_base.new_zeros(num_nodes, dtype=torch.bool),
                edge_context_norm=h_base.new_tensor(0.0),
            )

        hub_sum = h_base.new_zeros((num_prompts, self.hidden_dim))
        hub_sum.index_add_(0, feature_local, h_base[node_index] * aggregation_weight.unsqueeze(-1))
        hub_denom = h_base.new_zeros(num_prompts)
        hub_denom.index_add_(0, feature_local, aggregation_weight)
        hub_count = torch.bincount(feature_local, minlength=num_prompts)

        loo_denom = hub_denom[feature_local] - aggregation_weight
        loo_sum = hub_sum[feature_local] - h_base[node_index] * aggregation_weight.unsqueeze(-1)
        valid_edge = (loo_denom > 1e-6) & (
            (hub_count[feature_local] - 1) >= int(self.min_context_count)
        )
        edge_context = loo_sum / loo_denom.clamp_min(1e-6).unsqueeze(-1)

        edge_weight = self._receiver_edge_weight(
            graph=graph,
            prompt_edge_weight=prompt_edge_weight,
            feature_local=feature_local,
            node_index=node_index,
            ref=h_base,
        )
        edge_weight = edge_weight * valid_edge.to(dtype=h_base.dtype)
        context_sum = h_base.new_zeros(h_base.shape)
        context_sum.index_add_(0, node_index, edge_context * edge_weight.unsqueeze(-1))
        context_denom = h_base.new_zeros(num_nodes)
        context_denom.index_add_(0, node_index, edge_weight)
        connected = context_denom > 1e-12
        context = torch.where(
            connected.unsqueeze(-1),
            context_sum / context_denom.clamp_min(1e-12).unsqueeze(-1),
            h_base,
        )
        if self.detach_context:
            context = context.detach()

        direction = context - h_base
        direction_norm = direction.norm(dim=-1, keepdim=True)
        base_norm = h_base.norm(dim=-1, keepdim=True)
        max_norm = float(self.max_update_ratio) * base_norm
        bound = (max_norm / direction_norm.clamp_min(1e-12)).clamp(max=1.0)
        bounded_direction = direction * bound
        update = self.signed_scale.to(device=h_base.device, dtype=h_base.dtype) * bounded_direction
        update = update * connected.to(dtype=h_base.dtype).unsqueeze(-1)
        return self._output(
            h_base=h_base,
            update=update,
            context=context,
            connected=connected,
            edge_context_norm=edge_context.norm(dim=-1)[valid_edge].mean()
            if bool(valid_edge.any())
            else h_base.new_tensor(0.0),
        )

    def _receiver_edge_weight(
        self,
        *,
        graph: dict[str, Any],
        prompt_edge_weight: torch.Tensor | None,
        feature_local: torch.Tensor,
        node_index: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(prompt_edge_weight, torch.Tensor):
            direction = graph["edge_direction"].to(device=ref.device, dtype=torch.long)
            graph_node = graph["edge_node_index"].to(device=ref.device, dtype=torch.long)
            graph_feature = graph["edge_feature_local"].to(device=ref.device, dtype=torch.long)
            receive = direction == 1
            candidate = prompt_edge_weight.to(device=ref.device, dtype=ref.dtype)[receive]
            if (
                candidate.numel() == node_index.numel()
                and torch.equal(graph_node[receive], node_index)
                and torch.equal(graph_feature[receive], feature_local)
            ):
                return candidate.clamp_min(0.0)
        feature_stats = graph["feature_stats"].to(device=ref.device, dtype=ref.dtype)
        if feature_stats.numel() == 0:
            return ref.new_ones(node_index.numel())
        purity = feature_stats[:, 1].clamp(0.0, 1.0)
        entropy = feature_stats[:, 2].clamp(0.0, 1.0)
        return (purity * (1.0 - entropy)).clamp_min(1e-6)[feature_local]

    def _output(
        self,
        *,
        h_base: torch.Tensor,
        update: torch.Tensor,
        context: torch.Tensor,
        connected: torch.Tensor,
        edge_context_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        update_norm = update.norm(dim=-1)
        base_norm = h_base.norm(dim=-1)
        connected_ratio = connected.to(dtype=h_base.dtype).mean() if connected.numel() else h_base.new_tensor(0.0)
        return {
            "h_adp": h_base + update,
            "prompt_update": update,
            "prompt_context": context,
            "node_receive_gate": connected.to(dtype=h_base.dtype) * self.signed_scale.abs().to(
                device=h_base.device, dtype=h_base.dtype
            ),
            "signed_scale": self.signed_scale.to(device=h_base.device, dtype=h_base.dtype),
            "connected_ratio": connected_ratio,
            "context_delta_norm_mean": (context - h_base).norm(dim=-1).mean(),
            "edge_context_norm_mean": edge_context_norm,
            "update_norm_mean": update_norm.mean(),
            "update_norm_max": update_norm.max() if update_norm.numel() else h_base.new_tensor(0.0),
            "update_to_base_ratio": update_norm.mean() / base_norm.mean().clamp_min(1e-12),
        }
