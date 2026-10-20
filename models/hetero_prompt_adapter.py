"""Heterophily-aware node-level prompt adapter.

This module is intentionally independent from prompt graph construction. It
does not create prompt nodes or prompt edges; it produces a bounded residual
correction for the adapted GP2F branch from ego/low/high-frequency context.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary, mean_neighbor_variance


def _as_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _init_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


class HeterophilyAwarePromptAdapter(nn.Module):
    """Generate node-level residual corrections for the adapted branch."""

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = _as_config(config)

        self.context_base = str(self.config.get("context_base", "z_detached"))
        if self.context_base not in {"z", "z_detached"}:
            raise ValueError("prompt_adapter.context_base must be 'z' or 'z_detached'")
        self.use_ego = bool(self.config.get("use_ego", True))
        self.use_low_frequency = bool(self.config.get("use_low_frequency", True))
        self.use_two_step = bool(self.config.get("use_two_step", True))
        self.use_high_frequency = bool(self.config.get("use_high_frequency", True))
        self.use_role_features = bool(self.config.get("use_role_features", True))
        self.hidden = int(self.config.get("hidden_dim", self.hidden_dim))
        self.dropout = float(self.config.get("dropout", 0.2))
        self.zero_init_delta = bool(self.config.get("zero_init_delta", True))
        self.gate_init = float(self.config.get("gate_init", 0.05))
        self.max_update_norm = float(self.config.get("max_update_norm", 0.05))
        self.message_scale = float(self.config.get("message_scale", 1.0))
        self.num_classes = int(self.config.get("num_classes", 0))
        self.use_support_context = bool(self.config.get("use_support_context", False))
        self.support_tau = float(self.config.get("support_tau", 0.5))
        self.use_support_class_similarity = bool(self.config.get("use_support_class_similarity", True))
        self.use_support_proto_residual = bool(self.config.get("use_support_proto_residual", True))
        self.use_support_high_residual = bool(self.config.get("use_support_high_residual", True))
        self.support_context_mode = str(self.config.get("support_context_mode", "class_proto"))
        if self.support_context_mode not in {"class_proto", "topk_attention"}:
            raise ValueError("prompt_adapter.support_context_mode must be 'class_proto' or 'topk_attention'")
        self.support_topk = int(self.config.get("support_topk", 8))
        self.use_support_uncertainty_features = bool(self.config.get("use_support_uncertainty_features", False))
        self.use_support_reliability_gate = bool(self.config.get("use_support_reliability_gate", False))
        self.support_reliability_strength = float(self.config.get("support_reliability_strength", 1.0))
        self.support_reliability_floor = float(self.config.get("support_reliability_floor", 0.05))
        self.support_reliability_margin_weight = float(self.config.get("support_reliability_margin_weight", 0.5))

        context_dim = 0
        if self.use_ego:
            context_dim += self.source_dim
        if self.use_low_frequency:
            context_dim += self.source_dim
        if self.use_two_step:
            context_dim += self.source_dim
        if self.use_high_frequency:
            context_dim += self.source_dim
        if self.use_role_features:
            context_dim += 5
        if self.use_support_context:
            if self.num_classes <= 0:
                raise ValueError("prompt_adapter.num_classes must be positive when use_support_context=true")
            if self.use_support_class_similarity:
                context_dim += self.num_classes
            if self.use_support_proto_residual:
                context_dim += self.source_dim
            if self.use_support_high_residual:
                context_dim += self.source_dim
            if self.use_support_uncertainty_features:
                context_dim += 3
        if context_dim <= 0:
            raise ValueError("At least one prompt adapter context view must be enabled")
        self.context_dim = int(context_dim)

        self.delta_mlp = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )
        if self.zero_init_delta:
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, _init_logit(self.gate_init))

    def _base(self, z: torch.Tensor) -> torch.Tensor:
        return z.detach() if self.context_base == "z_detached" else z

    def _empty_support_context(self, base: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pieces: list[torch.Tensor] = []
        if self.use_support_class_similarity:
            pieces.append(base.new_zeros((base.size(0), self.num_classes)))
        if self.use_support_proto_residual:
            pieces.append(base.new_zeros(base.shape))
        if self.use_support_high_residual:
            pieces.append(base.new_zeros(base.shape))
        reliability_value = 1.0 if not self.use_support_reliability_gate else self.support_reliability_floor
        if self.use_support_uncertainty_features:
            uncertainty = torch.stack(
                [
                    base.new_zeros(base.size(0)),
                    base.new_ones(base.size(0)),
                    base.new_full((base.size(0),), float(reliability_value)),
                ],
                dim=-1,
            )
            pieces.append(uncertainty)
        support_context = torch.cat(pieces, dim=-1) if pieces else base.new_zeros((base.size(0), 0))
        stats = {
            "support_context_enabled": base.new_tensor(float(self.use_support_context)),
            "support_context_available": base.new_tensor(0.0),
            "support_context_coverage": base.new_tensor(0.0),
            "support_context_count": base.new_tensor(0.0),
            "support_similarity_margin": base.new_tensor(0.0),
            "support_similarity_entropy": base.new_tensor(0.0),
            "support_reliability": base.new_full((base.size(0),), float(reliability_value)),
            "support_reliability_mean": base.new_tensor(float(reliability_value)),
            "support_reliability_min": base.new_tensor(float(reliability_value)),
            "support_reliability_max": base.new_tensor(float(reliability_value)),
            "support_topk_mean_score": base.new_tensor(0.0),
        }
        return support_context, stats

    def _support_context(
        self,
        *,
        base: torch.Tensor,
        high: torch.Tensor,
        labels: torch.Tensor | None,
        support_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.use_support_context:
            return base.new_zeros((base.size(0), 0)), {
                "support_context_enabled": base.new_tensor(0.0),
                "support_context_available": base.new_tensor(0.0),
                "support_context_coverage": base.new_tensor(0.0),
                "support_context_count": base.new_tensor(0.0),
                "support_similarity_margin": base.new_tensor(0.0),
                "support_similarity_entropy": base.new_tensor(0.0),
                "support_reliability": base.new_ones(base.size(0)),
                "support_reliability_mean": base.new_tensor(1.0),
                "support_reliability_min": base.new_tensor(1.0),
                "support_reliability_max": base.new_tensor(1.0),
                "support_topk_mean_score": base.new_tensor(0.0),
            }
        if labels is None or support_mask is None:
            return self._empty_support_context(base)

        labels = labels.to(device=base.device, dtype=torch.long)
        support_mask = support_mask.to(device=base.device, dtype=torch.bool)
        valid = support_mask & (labels >= 0) & (labels < self.num_classes)
        if int(valid.sum().item()) == 0:
            return self._empty_support_context(base)

        class_ids = labels[valid]
        support_base = base[valid]
        support_high = high[valid]
        counts = base.new_zeros((self.num_classes, 1))
        counts.index_add_(0, class_ids, torch.ones((class_ids.numel(), 1), dtype=base.dtype, device=base.device))
        coverage = counts.squeeze(-1) > 0
        tau = max(float(self.support_tau), 1e-6)
        topk_mean_score = base.new_tensor(0.0)

        if self.support_context_mode == "topk_attention":
            support_logits = F.normalize(base, dim=-1, eps=1e-12) @ F.normalize(support_base, dim=-1, eps=1e-12).t()
            support_logits = support_logits / tau
            topk = min(max(1, int(self.support_topk)), int(support_logits.size(1)))
            top_scores, top_idx = torch.topk(support_logits, k=topk, dim=-1)
            top_weights = torch.softmax(top_scores, dim=-1)
            selected_classes = class_ids[top_idx]
            weights = base.new_zeros((base.size(0), self.num_classes))
            weights.scatter_add_(1, selected_classes, top_weights)
            weight_sums = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            weights = weights / weight_sums
            weighted_proto = (top_weights.unsqueeze(-1) * support_base[top_idx]).sum(dim=1)
            weighted_high_proto = (top_weights.unsqueeze(-1) * support_high[top_idx]).sum(dim=1)
            topk_mean_score = top_scores.mean()
        else:
            proto = base.new_zeros((self.num_classes, base.size(-1)))
            high_proto = base.new_zeros((self.num_classes, high.size(-1)))
            proto.index_add_(0, class_ids, support_base)
            high_proto.index_add_(0, class_ids, support_high)
            proto = proto / counts.clamp_min(1.0)
            high_proto = high_proto / counts.clamp_min(1.0)
            logits = F.normalize(base, dim=-1, eps=1e-12) @ F.normalize(proto, dim=-1, eps=1e-12).t()
            logits = logits / tau
            logits = logits.masked_fill(~coverage.view(1, -1), -1e9)
            weights = torch.softmax(logits, dim=-1)
            weights = torch.where(coverage.view(1, -1), weights, torch.zeros_like(weights))
            weight_sums = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            weights = weights / weight_sums
            weighted_proto = weights @ proto
            weighted_high_proto = weights @ high_proto

        pieces: list[torch.Tensor] = []
        if self.use_support_class_similarity:
            pieces.append(weights)
        if self.use_support_proto_residual:
            pieces.append(base - weighted_proto)
        if self.use_support_high_residual:
            pieces.append(high - weighted_high_proto)

        sorted_scores = torch.sort(weights, dim=-1, descending=True).values
        if sorted_scores.size(1) >= 2:
            margin = sorted_scores[:, 0] - sorted_scores[:, 1]
        else:
            margin = sorted_scores[:, 0]
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        if self.num_classes > 1:
            entropy = entropy / math.log(float(self.num_classes))
        margin_reliability = margin.clamp(0.0, 1.0)
        entropy_reliability = (1.0 - entropy).clamp(0.0, 1.0)
        reliability = (
            self.support_reliability_margin_weight * margin_reliability
            + (1.0 - self.support_reliability_margin_weight) * entropy_reliability
        ).clamp(0.0, 1.0)
        floor = min(max(float(self.support_reliability_floor), 0.0), 1.0)
        reliability = floor + (1.0 - floor) * reliability
        if self.use_support_uncertainty_features:
            pieces.append(torch.stack([margin, entropy, reliability], dim=-1))
        support_context = torch.cat(pieces, dim=-1) if pieces else base.new_zeros((base.size(0), 0))
        stats = {
            "support_context_enabled": base.new_tensor(1.0),
            "support_context_available": base.new_tensor(1.0),
            "support_context_coverage": coverage.to(dtype=base.dtype).mean(),
            "support_context_count": valid.to(dtype=base.dtype).sum(),
            "support_similarity_margin": margin.mean(),
            "support_similarity_entropy": entropy.mean(),
            "support_reliability": reliability,
            "support_reliability_mean": reliability.mean(),
            "support_reliability_min": reliability.min(),
            "support_reliability_max": reliability.max(),
            "support_topk_mean_score": topk_mean_score,
        }
        return support_context, stats

    def context(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self._base(z)
        low = mean_neighbor_summary(base, edge_index, num_nodes=base.size(0))
        two_step = mean_neighbor_summary(low, edge_index, num_nodes=base.size(0))
        high = base - low
        degree = torch.zeros(base.size(0), dtype=base.dtype, device=base.device)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=base.dtype, device=base.device))
        neighbor_var = mean_neighbor_variance(base, edge_index, num_nodes=base.size(0)).mean(dim=-1)
        cos_low = F.cosine_similarity(base, low, dim=-1, eps=1e-12)
        cos_two = F.cosine_similarity(low, two_step, dim=-1, eps=1e-12)
        role = torch.stack(
            [
                _minmax(degree),
                _minmax(torch.log1p(degree)),
                _minmax(neighbor_var),
                cos_low,
                cos_two,
            ],
            dim=-1,
        )
        pieces: list[torch.Tensor] = []
        if self.use_ego:
            pieces.append(base)
        if self.use_low_frequency:
            pieces.append(low)
        if self.use_two_step:
            pieces.append(two_step)
        if self.use_high_frequency:
            pieces.append(high)
        if self.use_role_features:
            pieces.append(role)
        support_context, support_stats = self._support_context(
            base=base,
            high=high,
            labels=labels,
            support_mask=support_mask,
        )
        if support_context.numel() > 0:
            pieces.append(support_context)
        return {
            "context": torch.cat(pieces, dim=-1),
            "ego": base,
            "low": low,
            "two_step": two_step,
            "high": high,
            "role": role,
            "degree": degree,
            "neighbor_variance": neighbor_var,
            **support_stats,
        }

    def forward(
        self,
        *,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        h_adp: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        message_scale: float | None = None,
        support_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(z, edge_index, labels=labels, support_mask=support_mask)
        context = ctx["context"]
        raw_delta = self.delta_mlp(context)
        raw_norm = raw_delta.norm(dim=-1, keepdim=True)
        max_norm = max(float(self.max_update_norm), 0.0)
        if max_norm > 0.0:
            scale = (max_norm / raw_norm.clamp_min(1e-12)).clamp_max(1.0)
            bounded_delta = raw_delta * scale
            clip_ratio = (raw_norm.squeeze(-1) > max_norm).to(dtype=h_adp.dtype).mean()
        else:
            bounded_delta = raw_delta
            clip_ratio = raw_delta.new_tensor(0.0)
        raw_gate = torch.sigmoid(self.gate_mlp(context)).squeeze(-1)
        reliability = ctx.get("support_reliability")
        if self.use_support_reliability_gate and isinstance(reliability, torch.Tensor):
            strength = min(max(float(self.support_reliability_strength), 0.0), 1.0)
            gate_factor = (1.0 - strength) + strength * reliability.to(device=raw_gate.device, dtype=raw_gate.dtype)
            gate = raw_gate * gate_factor
        else:
            gate = raw_gate
        effective_mask = torch.ones_like(gate, dtype=torch.bool) if update_mask is None else update_mask.bool().to(gate.device)
        scale_value = self.message_scale if message_scale is None else float(message_scale)
        update = float(scale_value) * gate.unsqueeze(-1) * bounded_delta
        update = update * effective_mask.to(dtype=update.dtype).unsqueeze(-1)
        h_prompted = h_adp + update
        update_norm = update.norm(dim=-1)
        return {
            "h_adp": h_prompted,
            "delta": bounded_delta,
            "raw_delta": raw_delta,
            "update": update,
            "gate": gate,
            "raw_gate": raw_gate,
            "update_mask": effective_mask,
            "context": context,
            "ego": ctx["ego"],
            "low": ctx["low"],
            "two_step": ctx["two_step"],
            "high": ctx["high"],
            "role": ctx["role"],
            "prompt_update_norm": update_norm.mean(),
            "prompt_update_max_norm": update_norm.max() if update_norm.numel() > 0 else update_norm.new_tensor(0.0),
            "prompt_raw_delta_norm": raw_delta.norm(dim=-1).mean(),
            "prompt_delta_norm": bounded_delta.norm(dim=-1).mean(),
            "prompt_gate_mean": gate.mean(),
            "prompt_gate_min": gate.min() if gate.numel() > 0 else gate.new_tensor(0.0),
            "prompt_gate_max": gate.max() if gate.numel() > 0 else gate.new_tensor(0.0),
            "prompt_raw_gate_mean": raw_gate.mean(),
            "prompt_raw_gate_min": raw_gate.min() if raw_gate.numel() > 0 else raw_gate.new_tensor(0.0),
            "prompt_raw_gate_max": raw_gate.max() if raw_gate.numel() > 0 else raw_gate.new_tensor(0.0),
            "prompt_update_mask_ratio": effective_mask.to(dtype=h_adp.dtype).mean(),
            "prompt_update_clip_ratio": clip_ratio,
            "high_frequency_norm": ctx["high"].norm(dim=-1).mean(),
            "low_frequency_norm": ctx["low"].norm(dim=-1).mean(),
            "support_context_enabled": ctx["support_context_enabled"],
            "support_context_available": ctx["support_context_available"],
            "support_context_coverage": ctx["support_context_coverage"],
            "support_context_count": ctx["support_context_count"],
            "support_similarity_margin": ctx["support_similarity_margin"],
            "support_similarity_entropy": ctx["support_similarity_entropy"],
            "support_reliability_mean": ctx["support_reliability_mean"],
            "support_reliability_min": ctx["support_reliability_min"],
            "support_reliability_max": ctx["support_reliability_max"],
            "support_topk_mean_score": ctx["support_topk_mean_score"],
            "message_scale": h_adp.new_tensor(float(scale_value)),
            "max_update_norm": h_adp.new_tensor(float(max_norm)),
        }


def prompt_adapter_update_norm_loss(adapter_out: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
    update = adapter_out.get("update")
    if not isinstance(update, torch.Tensor):
        return torch.tensor(0.0)
    if mask is not None:
        mask = mask.to(device=update.device, dtype=torch.bool)
        if int(mask.sum().item()) == 0:
            return update.new_tensor(0.0)
        update = update[mask]
    return update.pow(2).sum(dim=-1).mean()


def prompt_adapter_gate_budget_loss(
    adapter_out: dict[str, torch.Tensor],
    *,
    max_gate: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    gate = adapter_out.get("gate")
    if not isinstance(gate, torch.Tensor) or gate.numel() == 0:
        return torch.tensor(0.0)
    if mask is not None:
        mask = mask.to(device=gate.device, dtype=torch.bool)
        if int(mask.sum().item()) == 0:
            return gate.new_tensor(0.0)
        gate = gate[mask]
    return F.relu(gate.mean() - float(max_gate)).pow(2)


def prompt_adapter_message_help_loss(
    *,
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
    anti_harm_weight: float = 0.0,
    anti_harm_margin: float = 0.0,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_message_help_loss": 0.0,
            "prompt_adapter_message_help_mean_delta_ce": 0.0,
            "prompt_adapter_message_help_positive_ratio": 0.0,
            "prompt_adapter_message_help_count": 0.0,
            "prompt_adapter_message_help_anti_harm_loss": 0.0,
        }
    y = labels.to(device=logits_prompt.device)[idx]
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt[idx], y, reduction="none")
    delta = ce_no - ce_prompt
    losses = F.relu(float(margin) - delta)
    anti_harm_losses = F.relu(float(anti_harm_margin) - delta).pow(2)
    if class_balanced:
        per_class = []
        per_class_anti_harm = []
        for class_id in torch.unique(y.detach()).tolist():
            class_mask = y == int(class_id)
            if bool(class_mask.any()):
                per_class.append(losses[class_mask].mean())
                per_class_anti_harm.append(anti_harm_losses[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else losses.mean()
        anti_harm_loss = torch.stack(per_class_anti_harm).mean() if per_class_anti_harm else anti_harm_losses.mean()
    else:
        loss = losses.mean()
        anti_harm_loss = anti_harm_losses.mean()
    if float(anti_harm_weight) > 0.0:
        loss = loss + float(anti_harm_weight) * anti_harm_loss
    stats = {
        "prompt_adapter_message_help_loss": float(loss.detach().item()),
        "prompt_adapter_message_help_mean_delta_ce": float(delta.detach().mean().item()),
        "prompt_adapter_message_help_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
        "prompt_adapter_message_help_count": float(idx.numel()),
        "prompt_adapter_message_help_anti_harm_loss": float(anti_harm_loss.detach().item()),
    }
    return loss, stats


def prompt_adapter_utility_gate_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.02,
    margin: float = 0.0,
    class_balanced: bool = True,
    gate_source: str = "raw_gate",
) -> tuple[torch.Tensor, dict[str, float]]:
    if gate_source == "effective_gate":
        gate = adapter_out.get("gate", adapter_out.get("raw_gate"))
    elif gate_source == "raw_gate":
        gate = adapter_out.get("raw_gate", adapter_out.get("gate"))
    else:
        raise ValueError("gate_source must be 'raw_gate' or 'effective_gate'")
    if not isinstance(gate, torch.Tensor) or gate.numel() == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": 0.0,
            "prompt_adapter_utility_gate_positive_ratio": 0.0,
        }
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": 0.0,
            "prompt_adapter_utility_gate_positive_ratio": 0.0,
        }
    y = labels.to(device=logits_prompt.device)[idx]
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt.detach()[idx], y, reduction="none")
    delta = ce_no - ce_prompt
    temp = max(float(temperature), 1e-6)
    if margin > 0.0:
        supervised = delta.abs() >= float(margin)
    else:
        supervised = torch.ones_like(delta, dtype=torch.bool)
    if int(supervised.sum().item()) == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": float(delta.detach().mean().item()),
            "prompt_adapter_utility_gate_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
        }
    target = torch.sigmoid(delta.detach() / temp)
    gate_values = gate.to(device=logits_prompt.device)[idx].clamp(1e-6, 1.0 - 1e-6)
    losses = F.binary_cross_entropy(gate_values[supervised], target[supervised], reduction="none")
    y_supervised = y[supervised]
    if class_balanced:
        per_class = []
        for class_id in torch.unique(y_supervised.detach()).tolist():
            class_mask = y_supervised == int(class_id)
            if bool(class_mask.any()):
                per_class.append(losses[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else losses.mean()
    else:
        loss = losses.mean()
    stats = {
        "prompt_adapter_utility_gate_loss": float(loss.detach().item()),
        "prompt_adapter_utility_gate_target_mean": float(target[supervised].detach().mean().item()),
        "prompt_adapter_utility_gate_count": float(supervised.sum().item()),
        "prompt_adapter_utility_gate_delta_mean": float(delta.detach().mean().item()),
        "prompt_adapter_utility_gate_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
    }
    return loss, stats
