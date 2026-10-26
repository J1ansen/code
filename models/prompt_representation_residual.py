"""Prompt residual vector learner for PromptGraph selective feature prompts."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class PromptRepresentationResidual(nn.Module):
    """Low-rank, reliability-calibrated residuals for feature prompt vectors."""

    def __init__(self, source_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.source_dim = int(source_dim)
        self.enabled = bool(cfg.get("enabled", False))
        self.rank = int(cfg.get("rank", 4))
        self.hidden_dim = int(cfg.get("hidden_dim", min(64, max(16, self.source_dim // 4))))
        self.dropout = float(cfg.get("dropout", 0.10))
        self.gate_init_bias = float(cfg.get("gate_init_bias", -3.5))
        self.max_delta_ratio = float(cfg.get("max_delta_ratio", 0.05))
        self.delta_scale = float(cfg.get("delta_scale", 1.0))
        self.basis_init_std = float(cfg.get("basis_init_std", 0.01))
        self.coefficient_mode = str(cfg.get("coefficient_mode", "softmax")).lower()
        self.gate_floor = float(cfg.get("gate_floor", 0.0))
        self.use_reliability_prior = bool(cfg.get("use_reliability_prior", True))
        self.reliability_prior_strength = float(cfg.get("reliability_prior_strength", 1.0))

        self.stat_dim = 9
        if not self.enabled:
            return
        if self.rank <= 0:
            raise ValueError("prompt_residual.rank must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("prompt_residual.hidden_dim must be positive")
        if self.max_delta_ratio < 0.0:
            raise ValueError("prompt_residual.max_delta_ratio must be non-negative")
        if self.coefficient_mode not in {"softmax", "signed"}:
            raise ValueError("prompt_residual.coefficient_mode must be 'softmax' or 'signed'")
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("prompt_residual.gate_floor must be in [0, 1)")

        self.prompt_norm = nn.LayerNorm(self.source_dim)
        self.stat_norm = nn.LayerNorm(self.stat_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.source_dim + self.stat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.coeff_mlp = nn.Linear(self.hidden_dim, self.rank)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.stat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.basis = nn.Parameter(torch.empty(self.rank, self.source_dim))
        nn.init.normal_(self.basis, mean=0.0, std=self.basis_init_std)
        nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)

    def _zero_out(self, prompt_x: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = prompt_x.new_tensor(0.0)
        return {
            "prompt_x": prompt_x,
            "delta": torch.zeros_like(prompt_x),
            "raw_delta": torch.zeros_like(prompt_x),
            "gate": prompt_x.new_zeros(prompt_x.size(0)),
            "delta_norm_loss": zero,
            "delta_align_loss": zero,
            "gate_budget_loss": zero,
            "delta_norm_mean": zero,
            "delta_norm_max": zero,
            "delta_to_init_ratio": zero,
            "delta_cosine_to_init": zero,
            "gate_mean": zero,
            "gate_max": zero,
            "gate_reliability_corr": zero,
            "coeff_abs_mean": zero,
            "coeff_entropy": zero,
            "clip_fraction": zero,
            "reliability_mean": zero,
            "reliability_active_ratio": zero,
        }

    def _feature_stats(
        self,
        *,
        graph: dict[str, Any],
        ref: torch.Tensor,
        feature_gate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_stats = graph.get("feature_stats", ref.new_zeros((0, 6)))
        if not isinstance(feature_stats, torch.Tensor):
            feature_stats = ref.new_zeros((0, 6))
        feature_stats = feature_stats.to(device=ref.device, dtype=ref.dtype)
        num_prompts = int(feature_stats.size(0))
        if num_prompts == 0:
            return ref.new_zeros((0, self.stat_dim)), ref.new_zeros(0)

        usage = graph.get("usage", ref.new_zeros(num_prompts))
        if not isinstance(usage, torch.Tensor):
            usage = ref.new_zeros(num_prompts)
        usage = usage.to(device=ref.device, dtype=ref.dtype)
        usage_norm = usage / usage.mean().clamp_min(1e-6)

        if isinstance(feature_gate, torch.Tensor) and feature_gate.numel() == num_prompts:
            gate = feature_gate.to(device=ref.device, dtype=ref.dtype)
        else:
            gate = ref.new_ones(num_prompts)

        base_edge_weight = graph.get("base_edge_weight")
        edge_feature = graph.get("edge_feature_local")
        base_feature = ref.new_zeros(num_prompts)
        if (
            isinstance(base_edge_weight, torch.Tensor)
            and isinstance(edge_feature, torch.Tensor)
            and base_edge_weight.numel() == edge_feature.numel()
            and edge_feature.numel() > 0
        ):
            edge_feature = edge_feature.to(device=ref.device, dtype=torch.long)
            base_edge_weight = base_edge_weight.to(device=ref.device, dtype=ref.dtype)
            counts = ref.new_zeros(num_prompts)
            base_feature.index_add_(0, edge_feature, base_edge_weight)
            counts.index_add_(0, edge_feature, torch.ones_like(base_edge_weight))
            base_feature = base_feature / counts.clamp_min(1.0)

        stats = torch.cat(
            [
                feature_stats,
                usage_norm.unsqueeze(-1),
                gate.unsqueeze(-1),
                base_feature.unsqueeze(-1),
            ],
            dim=-1,
        )
        purity = feature_stats[:, 1] if feature_stats.size(1) > 1 else ref.new_zeros(num_prompts)
        entropy = feature_stats[:, 2] if feature_stats.size(1) > 2 else ref.new_zeros(num_prompts)
        margin = feature_stats[:, 3] if feature_stats.size(1) > 3 else ref.new_zeros(num_prompts)
        evidence = feature_stats[:, 4] if feature_stats.size(1) > 4 else ref.new_zeros(num_prompts)
        reliability = (purity * (1.0 - entropy).clamp_min(0.0) * margin.clamp_min(0.0) * evidence.clamp_min(0.0)).clamp(0.0, 1.0)
        return stats, reliability

    def forward(
        self,
        *,
        prompt_x: torch.Tensor,
        graph: dict[str, Any],
        ref: torch.Tensor,
        feature_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (not self.enabled) or prompt_x.numel() == 0:
            return self._zero_out(prompt_x)

        raw_stats, reliability = self._feature_stats(graph=graph, ref=ref, feature_gate=feature_gate)
        stats = self.stat_norm(raw_stats)
        p0 = prompt_x.to(device=ref.device, dtype=ref.dtype)
        query = self.query_mlp(torch.cat([self.prompt_norm(p0), stats], dim=-1))
        coeff_logits = self.coeff_mlp(query)
        if self.coefficient_mode == "signed":
            coeff = torch.tanh(coeff_logits)
            coeff = coeff / coeff.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            coeff_entropy = coeff.abs().mul(coeff.abs().clamp_min(1e-12).log()).sum(dim=-1).neg().mean()
        else:
            coeff = F.softmax(coeff_logits, dim=-1)
            coeff_entropy = coeff.mul(coeff.clamp_min(1e-12).log()).sum(dim=-1).neg().mean()
        basis = F.normalize(self.basis.to(device=ref.device, dtype=ref.dtype), dim=-1, eps=1e-12)
        raw_delta = coeff @ basis

        gate_logit = self.gate_mlp(torch.cat([query, stats], dim=-1)).squeeze(-1)
        if self.use_reliability_prior:
            prior = reliability.to(device=ref.device, dtype=ref.dtype).clamp(1e-4, 1.0 - 1e-4)
            gate_logit = gate_logit + float(self.reliability_prior_strength) * torch.logit(prior)
        gate = torch.sigmoid(gate_logit)
        if self.gate_floor > 0.0:
            gate = float(self.gate_floor) + (1.0 - float(self.gate_floor)) * gate
        p0_norm_col = p0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = float(self.delta_scale) * gate.unsqueeze(-1) * raw_delta

        clip_fraction = prompt_x.new_tensor(0.0)
        if self.max_delta_ratio > 0.0:
            max_norm = float(self.max_delta_ratio) * p0_norm_col
            delta_norm = delta.norm(dim=-1, keepdim=True)
            clip_fraction = (delta_norm > max_norm).to(dtype=ref.dtype).mean()
            delta = delta * torch.clamp(max_norm / delta_norm.clamp_min(1e-12), max=1.0)

        prompt_eff = p0 + delta
        delta_norm_flat = delta.norm(dim=-1)
        p0_norm = p0.norm(dim=-1).clamp_min(1e-12)
        delta_norm_ratio = delta_norm_flat / p0_norm
        cosine = F.cosine_similarity(p0, prompt_eff, dim=-1, eps=1e-12)
        gate_centered = gate - gate.mean()
        rel_centered = reliability - reliability.mean()
        corr = (gate_centered * rel_centered).mean() / (
            gate_centered.pow(2).mean().sqrt() * rel_centered.pow(2).mean().sqrt()
        ).clamp_min(1e-12)

        return {
            "prompt_x": prompt_eff,
            "delta": delta,
            "raw_delta": raw_delta,
            "gate": gate,
            "delta_norm_loss": delta_norm_flat.pow(2).mean(),
            "delta_align_loss": 1.0 - cosine.mean(),
            "gate_budget_loss": gate.mean(),
            "delta_norm_mean": delta_norm_flat.mean().detach(),
            "delta_norm_max": delta_norm_flat.max().detach(),
            "delta_to_init_ratio": delta_norm_ratio.mean().detach(),
            "delta_cosine_to_init": cosine.mean().detach(),
            "gate_mean": gate.mean().detach(),
            "gate_max": gate.max().detach(),
            "gate_reliability_corr": corr.detach(),
            "coeff_abs_mean": coeff.abs().mean().detach(),
            "coeff_entropy": coeff_entropy.detach(),
            "clip_fraction": clip_fraction.detach(),
            "reliability_mean": reliability.mean().detach(),
            "reliability_active_ratio": (reliability > 0.0).to(dtype=ref.dtype).mean().detach(),
        }
