"""P21-lite adaptive heterophily filter.

This module applies a prompt-controlled structural filter on the adapted GP2F
branch without adding prompt nodes or changing graph edges.  It keeps a strict
reject channel plus three structural residual channels (low, two-hop and
high-pass), then uses a conservative prior plus a node-wise router and gate.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary


CHANNEL_NAMES = ("reject", "low", "two", "high")
STRUCTURAL_CHANNEL_NAMES = ("low", "two", "high")


def _init_logit_from_probability(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


class P21LiteAdaptiveFilter(nn.Module):
    """Adaptive structural-channel filter for heterophilous graphs."""

    consumes_base_logits = True

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        self.context_detach = bool(self.config.get("context_detach", True))
        self.residual_scale = float(self.config.get("residual_scale", 0.1))
        self.use_node_wise_gate = bool(self.config.get("use_node_wise_gate", True))
        self.beta_max = float(self.config.get("beta_max", self.config.get("gate_max", 0.20)))
        self.gate_max = float(self.config.get("gate_max", self.beta_max))
        self.max_update_norm = float(self.config.get("max_update_norm", 0.05))
        beta_init = float(self.config.get("beta_init", 0.05))
        beta_prob = beta_init / max(self.beta_max, 1e-12)
        self.beta_logit = nn.Parameter(torch.tensor(_init_logit_from_probability(beta_prob)))

        prior = self.config.get("channel_prior", [0.50, 0.15, 0.20, 0.15])
        if not isinstance(prior, (list, tuple)) or len(prior) != len(CHANNEL_NAMES):
            raise ValueError("p21.channel_prior must contain four values: reject, low, two, high")
        prior_t = torch.tensor([float(v) for v in prior], dtype=torch.float32).clamp_min(1e-8)
        prior_t = prior_t / prior_t.sum().clamp_min(1e-12)
        self.theta_global = nn.Parameter(prior_t.log())

        descriptor_dim = 4 * self.hidden_dim + 9
        router_hidden = int(self.config.get("router_hidden_dim", self.hidden_dim))
        dropout = float(self.config.get("dropout", 0.1))
        self.router = nn.Sequential(
            nn.Linear(descriptor_dim, router_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, len(CHANNEL_NAMES)),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.gate_router = nn.Sequential(
            nn.Linear(descriptor_dim, router_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 1),
        )
        nn.init.zeros_(self.gate_router[-1].weight)
        gate_prob = beta_init / max(self.gate_max, 1e-12)
        nn.init.constant_(self.gate_router[-1].bias, _init_logit_from_probability(gate_prob))
        self.channel_projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in STRUCTURAL_CHANNEL_NAMES
            ]
        )
        for project in self.channel_projects:
            nn.init.normal_(project[-1].weight, std=0.02)
            nn.init.zeros_(project[-1].bias)
        # Backward-compatible alias for tests/checkpoints that inspect the
        # old shared projector name. This is the low-channel projector; reject
        # itself never passes through a projector.
        self.project = self.channel_projects[0]

    def _channels(self, h: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        num_nodes = int(h.size(0))
        low = mean_neighbor_summary(h, edge_index, num_nodes=num_nodes)
        two = mean_neighbor_summary(low, edge_index, num_nodes=num_nodes)
        high = h - low
        return {"ego": h, "low": low, "two": two, "high": high}

    def _logit_features(self, logits: torch.Tensor | None, num_nodes: int, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(logits, torch.Tensor):
            zero = ref.new_zeros(num_nodes)
            return zero, zero
        logits_d = logits.detach().to(device=ref.device, dtype=ref.dtype)
        prob = F.softmax(logits_d, dim=-1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        if logits_d.size(-1) > 1:
            entropy = entropy / math.log(float(logits_d.size(-1)))
        top2 = torch.topk(prob, k=min(2, prob.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]
        return entropy, margin

    def _degree_norm(self, edge_index: torch.Tensor, num_nodes: int, ref: torch.Tensor) -> torch.Tensor:
        degree = ref.new_zeros(num_nodes)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1].to(ref.device), ref.new_ones(edge_index.size(1)))
        return _minmax(torch.log1p(degree))

    def forward(
        self,
        *,
        z: torch.Tensor | None = None,
        edge_index: torch.Tensor,
        h_adp: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        h_pre: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        base = h_adp.detach() if self.context_detach else h_adp
        channels = self._channels(base, edge_index)
        ego = channels["ego"]
        low = channels["low"]
        two = channels["two"]
        high = channels["high"]
        num_nodes = int(h_adp.size(0))

        ego_low_norm = (ego - low).norm(dim=-1)
        low_two_norm = (low - two).norm(dim=-1)
        cos_ego_low = F.cosine_similarity(ego, low, dim=-1, eps=1e-12)
        cos_ego_two = F.cosine_similarity(ego, two, dim=-1, eps=1e-12)
        cos_low_two = F.cosine_similarity(low, two, dim=-1, eps=1e-12)
        degree_norm = self._degree_norm(edge_index, num_nodes, ego)
        entropy, margin = self._logit_features(base_logits, num_nodes, ego)
        if isinstance(h_pre, torch.Tensor) and h_pre.shape == ego.shape:
            disagreement = 1.0 - F.cosine_similarity(ego, h_pre.detach().to(ego), dim=-1, eps=1e-12)
        else:
            disagreement = ego.new_zeros(num_nodes)

        descriptor = torch.cat(
            [
                ego,
                low,
                two,
                high,
                ego_low_norm.unsqueeze(-1),
                low_two_norm.unsqueeze(-1),
                cos_ego_low.unsqueeze(-1),
                cos_ego_two.unsqueeze(-1),
                cos_low_two.unsqueeze(-1),
                degree_norm.unsqueeze(-1),
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                disagreement.unsqueeze(-1),
            ],
            dim=-1,
        )
        alpha_global = torch.softmax(self.theta_global, dim=-1)
        residual = self.router(descriptor)
        alpha_logits = alpha_global.clamp_min(1e-8).log().unsqueeze(0) + self.residual_scale * residual
        alpha = torch.softmax(alpha_logits, dim=-1)

        stacked = torch.stack([ego, low, two, high], dim=1)
        h_filter = torch.einsum("nk,nkh->nh", alpha, stacked)
        structural_residuals = torch.stack([low - ego, two - ego, high - ego], dim=1)
        zero_delta = ego.new_zeros(num_nodes, 1, self.hidden_dim)
        raw_structural_deltas = torch.stack(
            [project(structural_residuals[:, idx, :]) for idx, project in enumerate(self.channel_projects)],
            dim=1,
        )
        raw_channel_deltas = torch.stack(
            [zero_delta.squeeze(1), raw_structural_deltas[:, 0, :], raw_structural_deltas[:, 1, :], raw_structural_deltas[:, 2, :]],
            dim=1,
        )
        raw_channel_norm = raw_channel_deltas.norm(dim=-1, keepdim=True)
        if self.max_update_norm > 0.0:
            channel_clip_scale = (self.max_update_norm / raw_channel_norm.clamp_min(1e-12)).clamp_max(1.0)
            channel_deltas = raw_channel_deltas * channel_clip_scale
            clip_ratio = (raw_channel_norm.squeeze(-1) > self.max_update_norm).to(dtype=h_adp.dtype).mean()
        else:
            channel_deltas = raw_channel_deltas
            clip_ratio = h_adp.new_tensor(0.0)
        raw_delta = torch.einsum("nk,nkh->nh", alpha, raw_channel_deltas)
        delta = torch.einsum("nk,nkh->nh", alpha, channel_deltas)

        beta = self.beta_max * torch.sigmoid(self.beta_logit)
        if self.use_node_wise_gate:
            raw_gate = self.gate_router(descriptor).squeeze(-1)
            gate = self.gate_max * torch.sigmoid(raw_gate)
        else:
            raw_gate = self.beta_logit.expand(num_nodes)
            gate = beta.expand(num_nodes)
        effective_mask = (
            torch.ones(num_nodes, dtype=torch.bool, device=h_adp.device)
            if update_mask is None
            else update_mask.to(device=h_adp.device, dtype=torch.bool)
        )
        update = gate.unsqueeze(-1) * delta * effective_mask.to(dtype=h_adp.dtype).unsqueeze(-1)
        h_prompted = h_adp + update
        update_norm = update.norm(dim=-1)
        delta_norm = delta.norm(dim=-1)
        alpha_entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(dim=-1)
        alpha_entropy = alpha_entropy / math.log(float(len(CHANNEL_NAMES)))

        return {
            "h_adp": h_prompted,
            "h_filter": h_filter,
            "filter_update": update,
            "delta": delta,
            "raw_delta": raw_delta,
            "channel_deltas": channel_deltas,
            "raw_channel_deltas": raw_channel_deltas,
            "update": update,
            "gate": gate,
            "raw_gate": raw_gate,
            "gate_max": h_adp.new_tensor(float(self.gate_max)),
            "beta": beta,
            "alpha": alpha,
            "alpha_global": alpha_global,
            "alpha_mean": alpha.mean(dim=0),
            "alpha_entropy": alpha_entropy.mean(),
            "update_mask": effective_mask,
            "prompt_update_norm": update_norm.mean(),
            "prompt_update_max_norm": update_norm.max() if update_norm.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_raw_delta_norm": raw_delta.norm(dim=-1).mean(),
            "prompt_delta_norm": delta_norm.mean(),
            "prompt_gate_mean": gate.mean(),
            "prompt_gate_min": gate.min() if gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_gate_max": gate.max() if gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_raw_gate_mean": raw_gate.mean(),
            "prompt_raw_gate_min": raw_gate.min() if raw_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_raw_gate_max": raw_gate.max() if raw_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_update_mask_ratio": effective_mask.to(dtype=h_adp.dtype).mean(),
            "prompt_update_clip_ratio": clip_ratio,
            "high_frequency_norm": high.norm(dim=-1).mean(),
            "low_frequency_norm": low.norm(dim=-1).mean(),
            "p21_filter_enabled": h_adp.new_tensor(1.0),
            "p21_beta": beta.detach(),
            "p21_gate_mean": gate.mean(),
            "p21_gate_min": gate.min() if gate.numel() > 0 else h_adp.new_tensor(0.0),
            "p21_gate_max": gate.max() if gate.numel() > 0 else h_adp.new_tensor(0.0),
            "p21_channel_delta_reject_norm": channel_deltas[:, 0, :].norm(dim=-1).mean(),
            "p21_channel_delta_ego_norm": channel_deltas[:, 0, :].norm(dim=-1).mean(),
            "p21_channel_delta_low_norm": channel_deltas[:, 1, :].norm(dim=-1).mean(),
            "p21_channel_delta_two_norm": channel_deltas[:, 2, :].norm(dim=-1).mean(),
            "p21_channel_delta_high_norm": channel_deltas[:, 3, :].norm(dim=-1).mean(),
            "p21_alpha_entropy": alpha_entropy.mean(),
            "p21_alpha_reject_mean": alpha[:, 0].mean(),
            "p21_alpha_ego_mean": alpha[:, 0].mean(),
            "p21_alpha_low_mean": alpha[:, 1].mean(),
            "p21_alpha_two_mean": alpha[:, 2].mean(),
            "p21_alpha_high_mean": alpha[:, 3].mean(),
            "p21_alpha_global_reject": alpha_global[0],
            "p21_alpha_global_ego": alpha_global[0],
            "p21_alpha_global_low": alpha_global[1],
            "p21_alpha_global_two": alpha_global[2],
            "p21_alpha_global_high": alpha_global[3],
            "p21_channel_reject_norm": ego.new_tensor(0.0),
            "p21_channel_ego_norm": ego.norm(dim=-1).mean(),
            "p21_channel_low_norm": low.norm(dim=-1).mean(),
            "p21_channel_two_norm": two.norm(dim=-1).mean(),
            "p21_channel_high_norm": high.norm(dim=-1).mean(),
            "p21_ego_low_discrepancy": ego_low_norm.mean(),
            "p21_low_two_discrepancy": low_two_norm.mean(),
            "p21_no_prompt_entropy": entropy.mean(),
            "p21_no_prompt_margin": margin.mean(),
            "support_context_enabled": h_adp.new_tensor(0.0),
            "support_context_available": h_adp.new_tensor(0.0),
            "support_context_coverage": h_adp.new_tensor(0.0),
            "support_context_count": h_adp.new_tensor(0.0),
            "support_similarity_margin": h_adp.new_tensor(0.0),
            "support_similarity_entropy": h_adp.new_tensor(0.0),
            "support_reliability_mean": h_adp.new_tensor(1.0),
            "support_reliability_min": h_adp.new_tensor(1.0),
            "support_reliability_max": h_adp.new_tensor(1.0),
            "support_topk_mean_score": h_adp.new_tensor(0.0),
        }
