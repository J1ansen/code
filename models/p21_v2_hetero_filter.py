"""P21-v2 reject-aware heterophily expert filter."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.p21_adaptive_filter import _init_logit_from_probability, _minmax
from models.prompt_module import mean_neighbor_summary


P21_V2_CHANNEL_NAMES = ("reject", "low", "two", "high", "compat", "role")
P21_V2_STRUCTURAL_CHANNEL_NAMES = ("low", "two", "high")


class P21V2HeteroChannelBank(nn.Module):
    """Builds reject, structural, compatibility, and role expert deltas."""

    def __init__(self, hidden_dim: int, num_classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.config = dict(config)
        dropout = float(self.config.get("dropout", 0.1))
        role_hidden = int(self.config.get("role_hidden_dim", 64))
        self.compat_smoothing = float(self.config.get("compat_smoothing", 0.10))
        self.compat_detach_logits = bool(self.config.get("compat_detach_logits", True))
        self.compat_detach_prototypes = bool(self.config.get("compat_detach_prototypes", True))
        self.use_compat_channel = bool(self.config.get("use_compat_channel", True))
        self.use_role_channel = bool(self.config.get("use_role_channel", True))

        self.channel_projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in P21_V2_STRUCTURAL_CHANNEL_NAMES
            ]
        )
        self.compat_project = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.role_project = nn.Sequential(
            nn.Linear(11, role_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(role_hidden, self.hidden_dim),
        )
        for module in [*self.channel_projects, self.compat_project, self.role_project]:
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, std=0.02)
                nn.init.zeros_(last.bias)

    def _class_prototypes(
        self,
        h: torch.Tensor,
        labels: torch.Tensor | None,
        support_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is None or support_mask is None:
            proto = h.new_zeros(self.num_classes, self.hidden_dim)
            return proto, h.new_tensor(0.0)
        labels = labels.to(device=h.device, dtype=torch.long)
        support = support_mask.to(device=h.device, dtype=torch.bool)
        valid = support & (labels >= 0) & (labels < self.num_classes)
        if not bool(valid.any()):
            proto = h.new_zeros(self.num_classes, self.hidden_dim)
            return proto, h.new_tensor(0.0)
        source = h.detach() if self.compat_detach_prototypes else h
        global_proto = source[valid].mean(dim=0)
        proto = source.new_zeros(self.num_classes, self.hidden_dim)
        covered = source.new_zeros(self.num_classes, dtype=torch.bool)
        for class_id in range(self.num_classes):
            class_mask = valid & (labels == class_id)
            if bool(class_mask.any()):
                proto[class_id] = source[class_mask].mean(dim=0)
                covered[class_id] = True
            else:
                proto[class_id] = global_proto
        return proto, covered.to(dtype=h.dtype).mean()

    def _compatibility_matrix(
        self,
        neigh_prob: torch.Tensor,
        labels: torch.Tensor | None,
        support_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uniform = neigh_prob.new_full((self.num_classes, self.num_classes), 1.0 / max(1, self.num_classes))
        if labels is None or support_mask is None:
            return uniform, neigh_prob.new_tensor(0.0)
        labels = labels.to(device=neigh_prob.device, dtype=torch.long)
        support = support_mask.to(device=neigh_prob.device, dtype=torch.bool)
        valid = support & (labels >= 0) & (labels < self.num_classes)
        if not bool(valid.any()):
            return uniform, neigh_prob.new_tensor(0.0)
        matrix = neigh_prob.new_full(
            (self.num_classes, self.num_classes),
            float(self.compat_smoothing) / max(1, self.num_classes),
        )
        for class_id in range(self.num_classes):
            class_mask = valid & (labels == class_id)
            if bool(class_mask.any()):
                matrix[:, class_id] = matrix[:, class_id] + neigh_prob[class_mask].mean(dim=0)
        matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
        coverage = torch.unique(labels[valid]).numel() / max(1, self.num_classes)
        return matrix, neigh_prob.new_tensor(float(coverage))

    def forward(
        self,
        *,
        ego: torch.Tensor,
        low: torch.Tensor,
        two: torch.Tensor,
        high: torch.Tensor,
        edge_index: torch.Tensor,
        base_logits: torch.Tensor | None,
        support_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        role_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_nodes = int(ego.size(0))
        zero_delta = ego.new_zeros(num_nodes, self.hidden_dim)
        structural_residuals = torch.stack([low - ego, two - ego, high - ego], dim=1)
        structural_deltas = torch.stack(
            [project(structural_residuals[:, idx, :]) for idx, project in enumerate(self.channel_projects)],
            dim=1,
        )

        if self.use_compat_channel and isinstance(base_logits, torch.Tensor):
            logits = base_logits.detach() if self.compat_detach_logits else base_logits
            prob = F.softmax(logits.to(device=ego.device, dtype=ego.dtype), dim=-1)
            neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=num_nodes)
            compat_matrix, compat_class_coverage = self._compatibility_matrix(neigh_prob.detach(), labels, support_mask)
            class_proto, compat_proto_coverage = self._class_prototypes(ego, labels, support_mask)
            compat_prob = neigh_prob @ compat_matrix
            compat_hidden = compat_prob @ class_proto
            compat_delta = self.compat_project(compat_hidden - ego)
            neigh_entropy = -(neigh_prob * neigh_prob.clamp_min(1e-12).log()).sum(dim=-1)
            if self.num_classes > 1:
                neigh_entropy = neigh_entropy / math.log(float(self.num_classes))
        else:
            compat_delta = zero_delta
            compat_class_coverage = ego.new_tensor(0.0)
            compat_proto_coverage = ego.new_tensor(0.0)
            neigh_entropy = ego.new_zeros(num_nodes)

        role_delta = self.role_project(role_feat) if self.use_role_channel else zero_delta
        channel_deltas = torch.stack(
            [
                zero_delta,
                structural_deltas[:, 0, :],
                structural_deltas[:, 1, :],
                structural_deltas[:, 2, :],
                compat_delta,
                role_delta,
            ],
            dim=1,
        )
        stats = {
            "p21_v2_compat_class_coverage": compat_class_coverage,
            "p21_v2_compat_proto_coverage": compat_proto_coverage,
            "p21_v2_neighbor_prediction_entropy": neigh_entropy.mean(),
        }
        return channel_deltas, stats


class P21V2HeteroFilter(nn.Module):
    """Reject-aware heterophily expert filter with compat and role channels."""

    consumes_base_logits = True

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        self.num_classes = int(self.config.get("num_classes", 1))
        self.context_detach = bool(self.config.get("context_detach", True))
        self.residual_scale = float(self.config.get("residual_scale", 0.30))
        self.use_node_wise_gate = bool(self.config.get("use_node_wise_gate", True))
        self.beta_max = float(self.config.get("beta_max", self.config.get("gate_max", 0.30)))
        self.gate_max = float(self.config.get("gate_max", self.beta_max))
        self.max_update_norm = float(self.config.get("max_update_norm", 0.08))
        beta_init = float(self.config.get("beta_init", 0.10))
        beta_prob = beta_init / max(self.beta_max, 1e-12)
        self.beta_logit = nn.Parameter(torch.tensor(_init_logit_from_probability(beta_prob)))

        prior = self.config.get("channel_prior", [0.50, 0.10, 0.12, 0.10, 0.10, 0.08])
        if not isinstance(prior, (list, tuple)) or len(prior) != len(P21_V2_CHANNEL_NAMES):
            raise ValueError("p21-v2.channel_prior must contain six values: reject, low, two, high, compat, role")
        prior_t = torch.tensor([float(v) for v in prior], dtype=torch.float32).clamp_min(1e-8)
        prior_t = prior_t / prior_t.sum().clamp_min(1e-12)
        self.theta_global = nn.Parameter(prior_t.log())

        descriptor_dim = 4 * self.hidden_dim + 11
        router_hidden = int(self.config.get("router_hidden_dim", self.hidden_dim))
        dropout = float(self.config.get("dropout", 0.1))
        self.router = nn.Sequential(
            nn.Linear(descriptor_dim, router_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, len(P21_V2_CHANNEL_NAMES)),
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
        nn.init.constant_(self.gate_router[-1].bias, _init_logit_from_probability(beta_prob))
        self.channel_bank = P21V2HeteroChannelBank(self.hidden_dim, self.num_classes, self.config)

    def _channels(self, h: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        num_nodes = int(h.size(0))
        low = mean_neighbor_summary(h, edge_index, num_nodes=num_nodes)
        two = mean_neighbor_summary(low, edge_index, num_nodes=num_nodes)
        high = h - low
        return {"ego": h, "low": low, "two": two, "high": high}

    def _degree_norm(self, edge_index: torch.Tensor, num_nodes: int, ref: torch.Tensor) -> torch.Tensor:
        degree = ref.new_zeros(num_nodes)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1].to(ref.device), ref.new_ones(edge_index.size(1)))
        return _minmax(torch.log1p(degree))

    def _logit_features(
        self,
        logits: torch.Tensor | None,
        edge_index: torch.Tensor,
        num_nodes: int,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(logits, torch.Tensor):
            zero = ref.new_zeros(num_nodes)
            return zero, zero, zero, zero
        logits_d = logits.detach().to(device=ref.device, dtype=ref.dtype)
        prob = F.softmax(logits_d, dim=-1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        if logits_d.size(-1) > 1:
            entropy = entropy / math.log(float(logits_d.size(-1)))
        top2 = torch.topk(prob, k=min(2, prob.size(-1)), dim=-1).values
        margin = top2[:, 0] if top2.size(-1) == 1 else top2[:, 0] - top2[:, 1]
        neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=num_nodes)
        neigh_entropy = -(neigh_prob * neigh_prob.clamp_min(1e-12).log()).sum(dim=-1)
        if logits_d.size(-1) > 1:
            neigh_entropy = neigh_entropy / math.log(float(logits_d.size(-1)))
        neigh_top2 = torch.topk(neigh_prob, k=min(2, neigh_prob.size(-1)), dim=-1).values
        neigh_margin = neigh_top2[:, 0] if neigh_top2.size(-1) == 1 else neigh_top2[:, 0] - neigh_top2[:, 1]
        return entropy, margin, neigh_entropy, neigh_margin

    def forward(
        self,
        *,
        z: torch.Tensor | None = None,
        edge_index: torch.Tensor,
        h_adp: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        h_pre: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        compat_support_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
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
        entropy, margin, neigh_entropy, neigh_margin = self._logit_features(base_logits, edge_index, num_nodes, ego)
        if isinstance(h_pre, torch.Tensor) and h_pre.shape == ego.shape:
            disagreement = 1.0 - F.cosine_similarity(ego, h_pre.detach().to(ego), dim=-1, eps=1e-12)
        else:
            disagreement = ego.new_zeros(num_nodes)

        role_feat = torch.cat(
            [
                degree_norm.unsqueeze(-1),
                ego_low_norm.unsqueeze(-1),
                low_two_norm.unsqueeze(-1),
                cos_ego_low.unsqueeze(-1),
                cos_ego_two.unsqueeze(-1),
                cos_low_two.unsqueeze(-1),
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                neigh_entropy.unsqueeze(-1),
                neigh_margin.unsqueeze(-1),
                disagreement.unsqueeze(-1),
            ],
            dim=-1,
        )
        descriptor = torch.cat([ego, low, two, high, role_feat], dim=-1)
        alpha_global = torch.softmax(self.theta_global, dim=-1)
        residual = self.router(descriptor)
        alpha_logits = alpha_global.clamp_min(1e-8).log().unsqueeze(0) + self.residual_scale * residual
        alpha = torch.softmax(alpha_logits, dim=-1)

        compat_source = str(self.config.get("compat_support_source", "episode_support"))
        support_for_compat = compat_support_mask if compat_source == "full_train" and compat_support_mask is not None else support_mask
        raw_channel_deltas, bank_stats = self.channel_bank(
            ego=ego,
            low=low,
            two=two,
            high=high,
            edge_index=edge_index,
            base_logits=base_logits,
            support_mask=support_for_compat,
            labels=labels,
            role_feat=role_feat,
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
        alpha_entropy = alpha_entropy / math.log(float(len(P21_V2_CHANNEL_NAMES)))

        stacked = torch.stack([ego, low, two, high, ego + channel_deltas[:, 4, :], ego + channel_deltas[:, 5, :]], dim=1)
        h_filter = torch.einsum("nk,nkh->nh", alpha, stacked)
        out: dict[str, torch.Tensor | tuple[str, ...]] = {
            "h_adp": h_prompted,
            "h_filter": h_filter,
            "filter_update": update,
            "delta": delta,
            "raw_delta": raw_delta,
            "channel_deltas": channel_deltas,
            "raw_channel_deltas": raw_channel_deltas,
            "channel_names": P21_V2_CHANNEL_NAMES,
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
            "p21_v2_filter_enabled": h_adp.new_tensor(1.0),
            "p21_beta": beta.detach(),
            "p21_gate_mean": gate.mean(),
            "p21_gate_min": gate.min() if gate.numel() > 0 else h_adp.new_tensor(0.0),
            "p21_gate_max": gate.max() if gate.numel() > 0 else h_adp.new_tensor(0.0),
            "p21_alpha_entropy": alpha_entropy.mean(),
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
        for idx, name in enumerate(P21_V2_CHANNEL_NAMES):
            out[f"p21_channel_delta_{name}_norm"] = channel_deltas[:, idx, :].norm(dim=-1).mean()
            out[f"p21_alpha_{name}_mean"] = alpha[:, idx].mean()
            out[f"p21_alpha_global_{name}"] = alpha_global[idx]
            out[f"p21_channel_{name}_norm"] = (
                ego.new_tensor(0.0) if name == "reject" else channel_deltas[:, idx, :].norm(dim=-1).mean()
            )
            out[f"p21_v2_channel_delta_{name}_norm"] = out[f"p21_channel_delta_{name}_norm"]
            out[f"p21_v2_alpha_{name}_mean"] = out[f"p21_alpha_{name}_mean"]
            out[f"p21_v2_alpha_global_{name}"] = out[f"p21_alpha_global_{name}"]
        out["p21_channel_delta_ego_norm"] = out["p21_channel_delta_reject_norm"]
        out["p21_alpha_ego_mean"] = out["p21_alpha_reject_mean"]
        out["p21_alpha_global_ego"] = out["p21_alpha_global_reject"]
        out["p21_channel_ego_norm"] = ego.norm(dim=-1).mean()
        out.update(bank_stats)
        return out
