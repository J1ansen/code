"""Prompt-aware GP2F variant for P2 adapted-branch message passing."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.dual_branch_framework import DualBranchFramework


def _gate_logit(value: float) -> torch.Tensor:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(value, dtype=torch.float32))


def _safe_mean_norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.norm(dim=-1).mean()


def _prompt_gate_init_tensor(
    *,
    num_layers: int,
    default_value: float,
    node_to_prompt_value: float | None,
    prompt_to_node_value: float | None,
) -> torch.Tensor:
    node_value = default_value if node_to_prompt_value is None else node_to_prompt_value
    prompt_value = default_value if prompt_to_node_value is None else prompt_to_node_value
    values = torch.tensor([float(node_value), float(prompt_value)], dtype=torch.float32)
    return torch.stack([_gate_logit(float(item)) for item in values]).repeat(num_layers, 1)


class PromptAwareGP2F(DualBranchFramework):
    """GP2F with direction-aware prompt-edge messages in the adapted branch.

    Original graph edges still use the frozen backbone convolutions plus
    residual adapters. Prompt edges are consumed by separate per-layer message
    transforms and small learnable gates.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        num_classes: int,
        adapter_bottleneck_dim: int = 16,
        adapter_beta_init: float = 0.01,
        adapter_alpha_init: float = 0.1,
        adapter_style: str = "stable_zero_init",
        alpha_init: float = 0.5,
        fusion_alpha_style: str = "sigmoid",
        prompt_aware_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            backbone,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_beta_init=adapter_beta_init,
            adapter_alpha_init=adapter_alpha_init,
            adapter_style=adapter_style,
            alpha_init=alpha_init,
            fusion_alpha_style=fusion_alpha_style,
        )
        cfg = dict(prompt_aware_config or {})
        self.prompt_aware_dropout = float(cfg.get("dropout", 0.2))
        self.use_node_to_prompt = bool(cfg.get("use_node_to_prompt", True))
        self.use_prompt_to_node = bool(cfg.get("use_prompt_to_node", True))
        self.receiver_version = str(cfg.get("receiver_version", "v1_linear"))
        if self.receiver_version not in {
            "v1_linear",
            "v2_conditioned",
            "v3_node_residual",
            "v4_multi_expert_residual",
            "v5_prototype_directional",
            "v6_classifier_directional",
        }:
            raise ValueError(
                "prompt_aware.receiver_version must be 'v1_linear', 'v2_conditioned', "
                "'v3_node_residual', 'v4_multi_expert_residual', 'v5_prototype_directional', "
                "or 'v6_classifier_directional'"
            )
        self.prompt_fusion = str(cfg.get("prompt_fusion", "residual_gate"))
        gate_init = float(cfg.get("gate_init", 0.01))
        gate_init = float(cfg.get("receiver_gate_init", gate_init))
        node_to_prompt_gate_init = cfg.get("node_to_prompt_gate_init")
        prompt_to_node_gate_init = cfg.get("prompt_to_node_gate_init")
        self.prompt_message_scale = float(cfg.get("message_scale", 1.0))
        self.prompt_message_norm = str(cfg.get("prompt_message_norm", cfg.get("message_norm", "weighted_mean")))
        self.pool_only_prompt_update = bool(cfg.get("pool_only_prompt_update", False))
        self.zero_init_prompt_messages = bool(cfg.get("zero_init_prompt_messages", False))
        self.use_bounded_prompt_update = bool(
            cfg.get("use_bounded_prompt_update", cfg.get("bounded_prompt_update", False))
        )
        self.max_prompt_update_norm = float(cfg.get("max_prompt_update_norm", 0.05))
        self.prompt_update_bound_mode = str(cfg.get("prompt_update_bound_mode", "norm_clip"))
        self.prompt_slot_head_count = int(cfg.get("prompt_slot_head_count", cfg.get("num_prompt_nodes", 16)))
        if self.prompt_slot_head_count <= 0:
            raise ValueError("prompt_aware.prompt_slot_head_count must be positive")
        self.prototype_direction_normalize = bool(cfg.get("prototype_direction_normalize", True))
        self.prototype_direction_init = float(
            cfg.get("prototype_direction_init", 0.0 if self.zero_init_prompt_messages else 0.05)
        )
        valid_norms = {"weighted_mean", "weighted_sum", "degree_mean", "layernorm"}
        if self.prompt_message_norm not in valid_norms:
            raise ValueError(
                f"Unsupported prompt_aware.message_norm={self.prompt_message_norm!r}; "
                f"expected one of {sorted(valid_norms)}"
            )
        valid_bound_modes = {"norm_clip", "tanh"}
        if self.prompt_update_bound_mode not in valid_bound_modes:
            raise ValueError(
                f"Unsupported prompt_aware.prompt_update_bound_mode={self.prompt_update_bound_mode!r}; "
                f"expected one of {sorted(valid_bound_modes)}"
            )
        if self.max_prompt_update_norm <= 0:
            raise ValueError("prompt_aware.max_prompt_update_norm must be positive")

        in_dim = int(self.backbone.convs[0].lin.weight.shape[1])
        layer_input_dims = [in_dim] + [self.hidden_dim for _ in range(len(self.backbone.convs) - 1)]
        self.node_to_prompt_msgs = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in layer_input_dims]
        )
        self.prompt_to_node_msgs = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in layer_input_dims]
        )
        self.node_to_prompt_conditioned_msgs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4 * dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.prompt_aware_dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for dim in layer_input_dims
            ]
        )
        self.prompt_to_node_conditioned_msgs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4 * dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.prompt_aware_dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for dim in layer_input_dims
            ]
        )
        self.prompt_node_residual_corrections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4 * dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.prompt_aware_dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for dim in layer_input_dims
            ]
        )
        self.prompt_slot_residual_corrections = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(4 * dim, self.hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(self.prompt_aware_dropout),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                        )
                        for _ in range(self.prompt_slot_head_count)
                    ]
                )
                for dim in layer_input_dims
            ]
        )
        self.prototype_direction_projections = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim, bias=False) for dim in layer_input_dims]
        )
        self.prototype_direction_strength = nn.Parameter(
            torch.full((len(layer_input_dims),), self.prototype_direction_init, dtype=torch.float32)
        )
        self.prompt_update_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in layer_input_dims])
        self._init_prototype_direction_projections(layer_input_dims)
        if self.zero_init_prompt_messages:
            for module in [*self.node_to_prompt_msgs, *self.prompt_to_node_msgs]:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
            for module in [
                *self.node_to_prompt_conditioned_msgs,
                *self.prompt_to_node_conditioned_msgs,
                *self.prompt_node_residual_corrections,
            ]:
                final = module[-1]
                if isinstance(final, nn.Linear):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
            for layer_modules in self.prompt_slot_residual_corrections:
                for module in layer_modules:
                    final = module[-1]
                    if isinstance(final, nn.Linear):
                        nn.init.zeros_(final.weight)
                        nn.init.zeros_(final.bias)
        self.prompt_gate_logit = nn.Parameter(
            _prompt_gate_init_tensor(
                num_layers=len(self.backbone.convs),
                default_value=gate_init,
                node_to_prompt_value=(
                    None if node_to_prompt_gate_init is None else float(node_to_prompt_gate_init)
                ),
                prompt_to_node_value=(
                    None if prompt_to_node_gate_init is None else float(prompt_to_node_gate_init)
                ),
            )
        )
        self._last_prompt_aware_aux: dict[str, Any] = {}

    def _init_prototype_direction_projections(self, layer_input_dims: list[int]) -> None:
        """Initialize prototype-direction projections conservatively.

        The P9 receiver should not learn arbitrary residual patches. For hidden
        layers we start from an identity projection; for the input layer we copy
        the corresponding frozen backbone linear map when the shape matches.
        """

        with torch.no_grad():
            for layer_idx, (dim, projection) in enumerate(
                zip(layer_input_dims, self.prototype_direction_projections)
            ):
                projection.weight.zero_()
                if dim == self.hidden_dim:
                    projection.weight.copy_(torch.eye(self.hidden_dim, dtype=projection.weight.dtype))
                    continue
                conv = self.backbone.convs[layer_idx]
                lin = getattr(conv, "lin", None)
                weight = getattr(lin, "weight", None)
                if weight is not None and tuple(weight.shape) == tuple(projection.weight.shape):
                    projection.weight.copy_(weight.detach().to(dtype=projection.weight.dtype))
                else:
                    nn.init.xavier_uniform_(projection.weight)

    @property
    def prompt_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.prompt_gate_logit)

    def _aggregate_prompt_message(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        type_id: int,
        transform: nn.Linear,
    ) -> torch.Tensor:
        mask = edge_type == int(type_id)
        out = h.new_zeros((h.size(0), self.hidden_dim))
        if not bool(mask.any()):
            return out
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        msg = transform(h[src])
        msg = F.dropout(msg, p=self.prompt_aware_dropout, training=self.training)
        if edge_weight is None:
            weight = h.new_ones(src.numel())
        else:
            weight = edge_weight[mask].to(dtype=h.dtype, device=h.device)
        weighted_msg = msg * weight.unsqueeze(-1)
        out.index_add_(0, dst, weighted_msg)
        if self.prompt_message_norm == "weighted_sum":
            return out
        denom = h.new_zeros(h.size(0))
        if self.prompt_message_norm == "degree_mean":
            denom.index_add_(0, dst, torch.ones_like(weight))
        else:
            denom.index_add_(0, dst, weight)
        return out / denom.clamp_min(1e-12).unsqueeze(-1)

    def _aggregate_prompt_summary(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        type_id: int,
    ) -> torch.Tensor:
        mask = edge_type == int(type_id)
        out = h.new_zeros(h.shape)
        if not bool(mask.any()):
            return out
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        if edge_weight is None:
            weight = h.new_ones(src.numel())
        else:
            weight = edge_weight[mask].to(dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, h[src] * weight.unsqueeze(-1))
        if self.prompt_message_norm == "weighted_sum":
            return out
        denom = h.new_zeros(h.size(0))
        if self.prompt_message_norm == "degree_mean":
            denom.index_add_(0, dst, torch.ones_like(weight))
        else:
            denom.index_add_(0, dst, weight)
        return out / denom.clamp_min(1e-12).unsqueeze(-1)

    def _aggregate_prompt_summary_with_mass(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        type_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = edge_type == int(type_id)
        out = h.new_zeros(h.shape)
        mass = h.new_zeros(h.size(0))
        if not bool(mask.any()):
            return out, mass
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        if edge_weight is None:
            weight = h.new_ones(src.numel())
        else:
            weight = edge_weight[mask].to(dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, h[src] * weight.unsqueeze(-1))
        if self.prompt_message_norm == "weighted_sum":
            mass.index_add_(0, dst, torch.ones_like(weight))
            return out, mass
        if self.prompt_message_norm == "degree_mean":
            mass.index_add_(0, dst, torch.ones_like(weight))
        else:
            mass.index_add_(0, dst, weight)
        return out / mass.clamp_min(1e-12).unsqueeze(-1), mass

    def _aggregate_prototype_direction_prompt_message(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        type_id: int,
        projection: nn.Linear,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_summary, prompt_mass = self._aggregate_prompt_summary_with_mass(
            h=h,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_type=edge_type,
            type_id=type_id,
        )
        direction = prompt_summary - h
        has_prompt = (prompt_mass > 0).to(dtype=h.dtype).unsqueeze(-1)
        if self.prototype_direction_normalize:
            direction = F.normalize(direction, dim=-1, eps=1e-12)
        direction = direction * has_prompt
        projected = projection(direction)
        strength = torch.tanh(self.prototype_direction_strength[layer_idx]).to(dtype=h.dtype, device=h.device)
        return strength * projected, prompt_mass, strength

    def _aggregate_classifier_direction_prompt_message(
        self,
        *,
        h_base: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        original_node_count: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = edge_type == 2
        out = h_base.new_zeros((h_base.size(0), self.hidden_dim))
        mass = h_base.new_zeros(h_base.size(0))
        if not bool(mask.any()):
            return out, mass, torch.tanh(self.prototype_direction_strength[layer_idx]).to(
                dtype=h_base.dtype, device=h_base.device
            )

        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        slot = (src - int(original_node_count)).clamp(min=0, max=self.prompt_slot_head_count - 1)
        target = h_base[src].clone()
        class_count = int(self.classifier.out_features)
        class_slot_mask = slot < class_count
        if bool(class_slot_mask.any()):
            class_ids = slot[class_slot_mask].clamp(max=class_count - 1)
            class_targets = self.classifier.weight[class_ids].to(dtype=h_base.dtype, device=h_base.device)
            target[class_slot_mask] = class_targets

        direction = target - h_base[dst]
        if self.prototype_direction_normalize:
            direction = F.normalize(direction, dim=-1, eps=1e-12)
        if edge_weight is None:
            weight = h_base.new_ones(src.numel())
        else:
            weight = edge_weight[mask].to(dtype=h_base.dtype, device=h_base.device)
        out.index_add_(0, dst, direction * weight.unsqueeze(-1))
        if self.prompt_message_norm == "weighted_sum":
            mass.index_add_(0, dst, torch.ones_like(weight))
        elif self.prompt_message_norm == "degree_mean":
            mass.index_add_(0, dst, torch.ones_like(weight))
            out = out / mass.clamp_min(1e-12).unsqueeze(-1)
        else:
            mass.index_add_(0, dst, weight)
            out = out / mass.clamp_min(1e-12).unsqueeze(-1)

        strength = torch.tanh(self.prototype_direction_strength[layer_idx]).to(
            dtype=h_base.dtype, device=h_base.device
        )
        return strength * out, mass, strength

    def _aggregate_conditioned_prompt_message(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        type_id: int,
        transform: nn.Module,
    ) -> torch.Tensor:
        mask = edge_type == int(type_id)
        out = h.new_zeros((h.size(0), self.hidden_dim))
        if not bool(mask.any()):
            return out
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        src_h = h[src]
        dst_h = h[dst]
        msg_input = torch.cat([dst_h, src_h, dst_h - src_h, dst_h * src_h], dim=-1)
        msg = transform(msg_input)
        if edge_weight is None:
            weight = h.new_ones(src.numel())
        else:
            weight = edge_weight[mask].to(dtype=h.dtype, device=h.device)
        weighted_msg = msg * weight.unsqueeze(-1)
        out.index_add_(0, dst, weighted_msg)
        if self.prompt_message_norm == "weighted_sum":
            return out
        denom = h.new_zeros(h.size(0))
        if self.prompt_message_norm == "degree_mean":
            denom.index_add_(0, dst, torch.ones_like(weight))
        else:
            denom.index_add_(0, dst, weight)
        return out / denom.clamp_min(1e-12).unsqueeze(-1)

    def _aggregate_slot_residual_prompt_message(
        self,
        *,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor,
        original_node_count: int,
        transforms: nn.ModuleList,
    ) -> torch.Tensor:
        mask = edge_type == 2
        out = h.new_zeros((h.size(0), self.hidden_dim))
        if not bool(mask.any()):
            return out
        src_all = edge_index[0, mask]
        dst_all = edge_index[1, mask]
        slot_all = (src_all - int(original_node_count)).clamp(min=0, max=self.prompt_slot_head_count - 1)
        if edge_weight is None:
            weight_all = h.new_ones(src_all.numel())
        else:
            weight_all = edge_weight[mask].to(dtype=h.dtype, device=h.device)

        denom = h.new_zeros(h.size(0))
        if self.prompt_message_norm == "degree_mean":
            denom.index_add_(0, dst_all, torch.ones_like(weight_all))
        elif self.prompt_message_norm != "weighted_sum":
            denom.index_add_(0, dst_all, weight_all)

        for slot_id, transform in enumerate(transforms):
            slot_mask = slot_all == int(slot_id)
            if not bool(slot_mask.any()):
                continue
            src = src_all[slot_mask]
            dst = dst_all[slot_mask]
            src_h = h[src]
            dst_h = h[dst]
            msg_input = torch.cat([dst_h, src_h, dst_h - src_h, dst_h * src_h], dim=-1)
            msg = transform(msg_input)
            weight = weight_all[slot_mask]
            out.index_add_(0, dst, msg * weight.unsqueeze(-1))

        if self.prompt_message_norm == "weighted_sum":
            return out
        return out / denom.clamp_min(1e-12).unsqueeze(-1)

    def _bound_prompt_update(self, update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep prompt residuals as small corrections instead of free branch rewrites."""
        if not self.use_bounded_prompt_update:
            scale = update.new_ones((update.size(0), 1))
            return update, scale, update.new_tensor(0.0)

        max_norm = float(self.max_prompt_update_norm)
        if self.prompt_update_bound_mode == "tanh":
            bounded = max_norm * torch.tanh(update / max_norm)
            scale = bounded.norm(dim=-1, keepdim=True) / update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            clipped_ratio = (scale.squeeze(-1) < 1.0 - 1e-6).to(update.dtype).mean()
            return bounded, scale.clamp(max=1.0), clipped_ratio

        norm = update.norm(dim=-1, keepdim=True)
        scale = (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
        clipped_ratio = (norm.squeeze(-1) > max_norm).to(update.dtype).mean()
        return update * scale, scale, clipped_ratio

    def _encode_adapted_prompt_aware(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        *,
        original_node_count: int,
        prompt_update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            self._last_prompt_aware_aux = {
                "prompt_msg_norm": x.new_tensor(0.0),
                "prompt_to_original_msg_norm": x.new_tensor(0.0),
                "node_to_prompt_msg_norm": x.new_tensor(0.0),
                "prompt_to_original_update_norm": x.new_tensor(0.0),
                "node_to_prompt_update_norm": x.new_tensor(0.0),
                "raw_prompt_update_norm": x.new_tensor(0.0),
                "unbounded_prompt_update_norm": x.new_tensor(0.0),
                "prompt_update_clip_ratio": x.new_tensor(0.0),
                "prompt_gate_mean": self.prompt_gates.mean(),
                "prompt_gate_node_to_prompt_mean": self.prompt_gates[:, 0].mean(),
                "prompt_gate_prompt_to_node_mean": self.prompt_gates[:, 1].mean(),
                "prompt_gate_by_layer": self.prompt_gates,
                "adapted_branch_delta_norm": x.new_tensor(0.0),
                "prompt_message_scale": x.new_tensor(float(self.prompt_message_scale)),
                "prompt_message_norm": self.prompt_message_norm,
                "bounded_prompt_update": x.new_tensor(float(self.use_bounded_prompt_update)),
                "max_prompt_update_norm": x.new_tensor(float(self.max_prompt_update_norm)),
                "prompt_update_bound_mode": self.prompt_update_bound_mode,
                "pool_only_prompt_update": x.new_tensor(float(self.pool_only_prompt_update)),
                "zero_init_prompt_messages": x.new_tensor(float(self.zero_init_prompt_messages)),
                "receiver_version": self.receiver_version,
                "prompt_fusion": self.prompt_fusion,
                "prompt_slot_head_count": x.new_tensor(float(self.prompt_slot_head_count)),
                "prompt_receiver_gate_mean": self.prompt_gates[:, 1].mean(),
                "prototype_direction_strength_mean": torch.tanh(self.prototype_direction_strength).mean(),
                "prototype_direction_normalize": x.new_tensor(float(self.prototype_direction_normalize)),
            }
            return self._encode_adapted(x, edge_index, edge_weight)

        if edge_type.numel() != edge_index.size(1):
            raise ValueError("edge_type length must match adapted_edge_index edge count")
        edge_type = edge_type.to(device=edge_index.device, dtype=torch.long)
        original_mask = edge_type == 0
        original_edge_index = edge_index[:, original_mask]
        original_edge_weight = edge_weight[original_mask] if edge_weight is not None else None

        h = x
        clean_h = x
        update_mask: torch.Tensor | None = None
        if self.pool_only_prompt_update and prompt_update_mask is not None:
            original_mask = prompt_update_mask.to(device=x.device, dtype=torch.bool)
            if original_mask.numel() != int(original_node_count):
                raise ValueError("prompt_update_mask length must match original node count")
            update_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            update_mask[:original_node_count] = original_mask
            update_mask[original_node_count:] = True
        gates = self.prompt_gates.to(dtype=x.dtype, device=x.device)
        prompt_norms: list[torch.Tensor] = []
        prompt_to_original_norms: list[torch.Tensor] = []
        node_to_prompt_norms: list[torch.Tensor] = []
        prompt_to_original_update_norms: list[torch.Tensor] = []
        node_to_prompt_update_norms: list[torch.Tensor] = []
        raw_prompt_update_norms: list[torch.Tensor] = []
        unbounded_prompt_update_norms: list[torch.Tensor] = []
        prompt_update_clip_ratios: list[torch.Tensor] = []
        branch_delta_norms: list[torch.Tensor] = []
        num_layers = len(self.backbone.convs)
        for layer_idx, (conv, adapter) in enumerate(zip(self.backbone.convs, self.adapters)):
            if original_edge_weight is None:
                h_conv = conv(h, original_edge_index)
            else:
                h_conv = conv(h, original_edge_index, original_edge_weight)
            h_base = adapter(h_conv) if self.use_adapters else h_conv
            if update_mask is not None:
                if original_edge_weight is None:
                    clean_h_conv = conv(clean_h, original_edge_index)
                else:
                    clean_h_conv = conv(clean_h, original_edge_index, original_edge_weight)
                clean_h_base = adapter(clean_h_conv) if self.use_adapters else clean_h_conv
            else:
                clean_h_base = h_base

            if self.receiver_version == "v4_multi_expert_residual":
                node_to_prompt = self._aggregate_conditioned_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=1,
                    transform=self.node_to_prompt_conditioned_msgs[layer_idx],
                )
                prompt_to_node = self._aggregate_slot_residual_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    original_node_count=original_node_count,
                    transforms=self.prompt_slot_residual_corrections[layer_idx],
                )
            elif self.receiver_version == "v5_prototype_directional":
                node_to_prompt = self._aggregate_conditioned_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=1,
                    transform=self.node_to_prompt_conditioned_msgs[layer_idx],
                )
                prompt_to_node, _, _ = self._aggregate_prototype_direction_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=2,
                    projection=self.prototype_direction_projections[layer_idx],
                    layer_idx=layer_idx,
                )
            elif self.receiver_version == "v6_classifier_directional":
                node_to_prompt = self._aggregate_conditioned_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=1,
                    transform=self.node_to_prompt_conditioned_msgs[layer_idx],
                )
                prompt_to_node, _, _ = self._aggregate_classifier_direction_prompt_message(
                    h_base=h_base,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    original_node_count=original_node_count,
                    layer_idx=layer_idx,
                )
            elif self.receiver_version == "v3_node_residual":
                node_to_prompt = self._aggregate_conditioned_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=1,
                    transform=self.node_to_prompt_conditioned_msgs[layer_idx],
                )
                prompt_summary = self._aggregate_prompt_summary(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=2,
                )
                correction_input = torch.cat(
                    [h, prompt_summary, h - prompt_summary, h * prompt_summary],
                    dim=-1,
                )
                prompt_to_node = self.prompt_node_residual_corrections[layer_idx](correction_input)
            elif self.receiver_version == "v2_conditioned":
                node_to_prompt = self._aggregate_conditioned_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=1,
                    transform=self.node_to_prompt_conditioned_msgs[layer_idx],
                )
                prompt_to_node = self._aggregate_conditioned_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=2,
                    transform=self.prompt_to_node_conditioned_msgs[layer_idx],
                )
            else:
                node_to_prompt = self._aggregate_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=1,
                    transform=self.node_to_prompt_msgs[layer_idx],
                )
                prompt_to_node = self._aggregate_prompt_message(
                    h=h,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_type=edge_type,
                    type_id=2,
                    transform=self.prompt_to_node_msgs[layer_idx],
                )
            if not self.use_node_to_prompt:
                node_to_prompt = torch.zeros_like(node_to_prompt)
            if not self.use_prompt_to_node:
                prompt_to_node = torch.zeros_like(prompt_to_node)
            gated_node_to_prompt = gates[layer_idx, 0] * node_to_prompt
            gated_prompt_to_node = gates[layer_idx, 1] * prompt_to_node
            raw_prompt_update = gated_node_to_prompt + gated_prompt_to_node
            if self.prompt_message_norm == "layernorm":
                raw_prompt_update = self.prompt_update_norms[layer_idx](raw_prompt_update)
            unbounded_prompt_update = float(self.prompt_message_scale) * raw_prompt_update
            prompt_update, bound_scale, clip_ratio = self._bound_prompt_update(unbounded_prompt_update)
            prompted_h = h_base + prompt_update
            if update_mask is not None:
                h = torch.where(update_mask.unsqueeze(-1), prompted_h, clean_h_base)
            else:
                h = prompted_h

            prompt_norms.append(_safe_mean_norm(prompt_update))
            prompt_to_original_norms.append(_safe_mean_norm(prompt_to_node[:original_node_count]))
            node_to_prompt_norms.append(_safe_mean_norm(node_to_prompt[original_node_count:]))
            prompt_to_original_update_norms.append(
                _safe_mean_norm((bound_scale * float(self.prompt_message_scale) * gated_prompt_to_node)[:original_node_count])
            )
            node_to_prompt_update_norms.append(
                _safe_mean_norm((bound_scale * float(self.prompt_message_scale) * gated_node_to_prompt)[original_node_count:])
            )
            raw_prompt_update_norms.append(_safe_mean_norm(raw_prompt_update))
            unbounded_prompt_update_norms.append(_safe_mean_norm(unbounded_prompt_update))
            prompt_update_clip_ratios.append(clip_ratio)
            branch_delta_norms.append(_safe_mean_norm(prompt_update[:original_node_count]))

            if layer_idx < num_layers - 1:
                act = getattr(self.backbone, "act", None)
                h = act(h) if act is not None else F.relu(h)
                if update_mask is not None:
                    clean_h = act(clean_h_base) if act is not None else F.relu(clean_h_base)
                dropout = float(getattr(self.backbone, "dropout", 0.0))
                h = F.dropout(h, p=dropout, training=self.training)
                if update_mask is not None:
                    clean_h = F.dropout(clean_h, p=dropout, training=self.training)

        self._last_prompt_aware_aux = {
            "prompt_msg_norm": torch.stack(prompt_norms).mean() if prompt_norms else x.new_tensor(0.0),
            "prompt_to_original_msg_norm": (
                torch.stack(prompt_to_original_norms).mean() if prompt_to_original_norms else x.new_tensor(0.0)
            ),
            "node_to_prompt_msg_norm": (
                torch.stack(node_to_prompt_norms).mean() if node_to_prompt_norms else x.new_tensor(0.0)
            ),
            "prompt_to_original_update_norm": (
                torch.stack(prompt_to_original_update_norms).mean()
                if prompt_to_original_update_norms
                else x.new_tensor(0.0)
            ),
            "node_to_prompt_update_norm": (
                torch.stack(node_to_prompt_update_norms).mean()
                if node_to_prompt_update_norms
                else x.new_tensor(0.0)
            ),
            "raw_prompt_update_norm": (
                torch.stack(raw_prompt_update_norms).mean()
                if raw_prompt_update_norms
                else x.new_tensor(0.0)
            ),
            "unbounded_prompt_update_norm": (
                torch.stack(unbounded_prompt_update_norms).mean()
                if unbounded_prompt_update_norms
                else x.new_tensor(0.0)
            ),
            "prompt_update_clip_ratio": (
                torch.stack(prompt_update_clip_ratios).mean()
                if prompt_update_clip_ratios
                else x.new_tensor(0.0)
            ),
            "prompt_gate_mean": gates.mean(),
            "prompt_gate_node_to_prompt_mean": gates[:, 0].mean(),
            "prompt_gate_prompt_to_node_mean": gates[:, 1].mean(),
            "prompt_gate_by_layer": gates,
            "adapted_branch_delta_norm": (
                torch.stack(branch_delta_norms).mean() if branch_delta_norms else x.new_tensor(0.0)
            ),
            "prompt_message_scale": x.new_tensor(float(self.prompt_message_scale)),
            "prompt_message_norm": self.prompt_message_norm,
            "bounded_prompt_update": x.new_tensor(float(self.use_bounded_prompt_update)),
            "max_prompt_update_norm": x.new_tensor(float(self.max_prompt_update_norm)),
            "prompt_update_bound_mode": self.prompt_update_bound_mode,
            "pool_only_prompt_update": x.new_tensor(float(self.pool_only_prompt_update)),
            "zero_init_prompt_messages": x.new_tensor(float(self.zero_init_prompt_messages)),
            "receiver_version": self.receiver_version,
            "prompt_fusion": self.prompt_fusion,
            "prompt_slot_head_count": x.new_tensor(float(self.prompt_slot_head_count)),
            "prompt_receiver_gate_mean": gates[:, 1].mean(),
            "prototype_direction_strength_mean": torch.tanh(self.prototype_direction_strength).mean(),
            "prototype_direction_normalize": x.new_tensor(float(self.prototype_direction_normalize)),
        }
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        adapted_x: torch.Tensor | None = None,
        adapted_edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        adapted_edge_weight: torch.Tensor | None = None,
        adapted_edge_type: torch.Tensor | None = None,
        prompt_update_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, Any]:
        if adapted_x is None:
            adapted_x = x
        if adapted_edge_index is None:
            adapted_edge_index = edge_index
        if adapted_edge_weight is None:
            adapted_edge_weight = edge_weight

        h_pre = self._encode_backbone(x, edge_index, edge_weight)
        return self.forward_with_h_pre(
            x,
            edge_index,
            h_pre=h_pre,
            adapted_x=adapted_x,
            adapted_edge_index=adapted_edge_index,
            adapted_edge_weight=adapted_edge_weight,
            adapted_edge_type=adapted_edge_type,
            prompt_update_mask=prompt_update_mask,
            return_aux=return_aux,
        )

    def forward_with_h_pre(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        h_pre: torch.Tensor,
        adapted_x: torch.Tensor | None = None,
        adapted_edge_index: torch.Tensor | None = None,
        adapted_edge_weight: torch.Tensor | None = None,
        adapted_edge_type: torch.Tensor | None = None,
        prompt_update_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, Any]:
        if adapted_x is None:
            adapted_x = x
        if adapted_edge_index is None:
            adapted_edge_index = edge_index

        h_adp_full = self._encode_adapted_prompt_aware(
            adapted_x,
            adapted_edge_index,
            adapted_edge_weight,
            adapted_edge_type,
            original_node_count=h_pre.size(0),
            prompt_update_mask=prompt_update_mask,
        )
        h_adp = h_adp_full[: h_pre.size(0)]

        alpha = self.alpha
        h_mix = alpha * h_pre + (1.0 - alpha) * h_adp
        logits = self.classifier(h_mix)

        if return_aux:
            return {
                "logits": logits,
                "h_pre": h_pre,
                "h_adp": h_adp,
                "h_mix": h_mix,
                "alpha": alpha,
                "h_adp_full": h_adp_full,
                "prompt_aware": self._last_prompt_aware_aux,
            }
        return logits, h_pre, h_adp, h_mix, alpha
