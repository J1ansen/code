"""Residual prompt modules for adapted-branch conditioning."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _as_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _zero_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _make_zero_init_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    last = nn.Linear(int(hidden_dim), int(out_dim))
    _zero_init(last)
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        last,
    )


def _init_gamma_logit(gamma_init: float, gamma_max: float) -> torch.Tensor:
    gamma_max = float(gamma_max)
    if gamma_max <= 0:
        raise ValueError("gamma_max must be positive")
    ratio = min(max(float(gamma_init) / gamma_max, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def count_trainable_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def prompt_budget_loss(gate: torch.Tensor, route_budget: float) -> torch.Tensor:
    non_null = gate[:, 0] + gate[:, 1]
    excess = non_null.mean() - float(route_budget)
    return torch.clamp(excess, min=0.0).pow(2)


def prompt_message_norm_loss(
    prompt_out: dict[str, Any],
    z: torch.Tensor,
    *,
    target_ratio: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Penalize residual prompt messages that become large relative to the input."""

    u_prompt = prompt_out["u_prompt"]
    gamma = prompt_out["gamma"]
    message = gamma * u_prompt
    ratio = message.norm(dim=-1) / z.detach().norm(dim=-1).clamp_min(float(eps))
    excess = ratio - float(target_ratio)
    return torch.clamp(excess, min=0.0).pow(2).mean()


def mean_neighbor_summary(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """Mean aggregate source features into destination nodes without adding self loops."""

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    num_nodes = int(features.size(0) if num_nodes is None else num_nodes)
    src, dst = edge_index[0], edge_index[1]
    out = features.new_zeros((num_nodes, features.size(-1)))
    degree = features.new_zeros((num_nodes, 1))
    out.index_add_(0, dst, features[src])
    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=features.dtype, device=features.device))
    return out / degree.clamp_min(1.0)


def degree_summary(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return normalized log in/out degree features for each node."""

    src, dst = edge_index[0], edge_index[1]
    in_degree = torch.zeros((num_nodes, 1), dtype=dtype, device=device)
    out_degree = torch.zeros((num_nodes, 1), dtype=dtype, device=device)
    ones = torch.ones((src.numel(), 1), dtype=dtype, device=device)
    in_degree.index_add_(0, dst, ones)
    out_degree.index_add_(0, src, ones)
    degree = torch.cat([in_degree, out_degree], dim=-1)
    scale = torch.log1p(degree.max()).clamp_min(1.0)
    return torch.log1p(degree) / scale


def mean_neighbor_variance(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """Mean neighbor variance without self loops."""

    mean = mean_neighbor_summary(features, edge_index, num_nodes=num_nodes)
    mean_sq = mean_neighbor_summary(features.pow(2), edge_index, num_nodes=num_nodes)
    return (mean_sq - mean.pow(2)).clamp_min(0.0)


def _safe_cosine_column(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x, y, dim=-1, eps=1e-12).unsqueeze(-1)


def _cosine_scores(x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-12) @ F.normalize(prototypes, dim=-1, eps=1e-12).t()


class UnifiedMultiViewResidualPrompt(nn.Module):
    """Semantic + structural residual prompt with a soft null route."""

    def __init__(
        self,
        source_dim: int,
        hidden_dim: int,
        num_classes: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.config = _as_config(config)

        semantic_cfg = _as_config(self.config.get("semantic"))
        structural_cfg = _as_config(self.config.get("structural"))
        gate_cfg = _as_config(self.config.get("gate"))
        gradient_cfg = _as_config(self.config.get("gradient"))

        self.semantic_enabled = bool(semantic_cfg.get("enabled", True))
        self.structural_enabled = bool(structural_cfg.get("enabled", True))
        self.null_enabled = bool(gate_cfg.get("use_null", True))
        self.use_feature_proto = bool(semantic_cfg.get("use_feature_proto", True))
        self.use_hidden_proto = bool(semantic_cfg.get("use_hidden_proto", True))
        self.beta_z = float(semantic_cfg.get("beta_z", 0.5))
        self.beta_h = float(semantic_cfg.get("beta_h", 1.0))
        self.tau_sem = float(semantic_cfg.get("tau_sem", 0.5))
        self.use_leave_one_out = bool(semantic_cfg.get("use_leave_one_out_train_proto", True))
        self.one_shot_train_semantic = str(semantic_cfg.get("one_shot_train_semantic", "mask"))
        self.use_margin_mask = bool(semantic_cfg.get("use_margin_mask", False))
        self.margin_threshold = float(semantic_cfg.get("margin_threshold", 0.05))
        self.include_degree_features = bool(structural_cfg.get("include_degree_features", False))
        self.include_neighbor_variance = bool(structural_cfg.get("include_neighbor_variance", False))
        self.include_similarity_features = bool(structural_cfg.get("include_similarity_features", False))

        self.h_pre_for_prompt = str(gradient_cfg.get("h_pre_for_prompt", "detach"))
        self.z_for_prompt = str(gradient_cfg.get("z_for_prompt", "detach"))
        self.structural_base = str(structural_cfg.get("base", "h_pre_detached"))

        semantic_hidden = int(semantic_cfg.get("hidden_dim", max(1, min(self.hidden_dim, self.source_dim))))
        semantic_dropout = float(semantic_cfg.get("dropout", 0.0))
        structural_hidden = int(structural_cfg.get("hidden_dim", self.hidden_dim))
        structural_dropout = float(structural_cfg.get("dropout", 0.3))
        gate_hidden = int(gate_cfg.get("hidden_dim", 64))
        gate_dropout = float(gate_cfg.get("dropout", 0.3))

        self.semantic_projector = _make_zero_init_mlp(
            self.source_dim,
            semantic_hidden,
            self.source_dim,
            dropout=semantic_dropout,
        )
        self.structural_projector = _make_zero_init_mlp(
            self._structural_projector_input_dim(),
            structural_hidden,
            self.source_dim,
            dropout=structural_dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(self._gate_input_dim(), gate_hidden),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden, 3),
        )
        self._init_gate_bias(float(gate_cfg.get("null_bias_init", 1.0)))

        self.gamma_max = float(self.config.get("gamma_max", 0.5))
        gamma_init = float(self.config.get("gamma_init", 0.05))
        self.gamma_logit = nn.Parameter(_init_gamma_logit(gamma_init, self.gamma_max))

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.gamma_logit)

    def _base_dim(self) -> int:
        if self.structural_base in {"h_pre_detached", "h_pre"}:
            return self.hidden_dim
        if self.structural_base in {"z_detached", "z"}:
            return self.source_dim
        raise ValueError(f"Unsupported structural base: {self.structural_base}")

    def _degree_dim(self) -> int:
        return 2 if self.include_degree_features else 0

    def _similarity_dim(self) -> int:
        return 2 if self.include_similarity_features else 0

    def _structural_projector_input_dim(self) -> int:
        base_dim = self._base_dim()
        context_dim = 5 * base_dim
        if self.include_neighbor_variance:
            context_dim += base_dim
        context_dim += self._degree_dim()
        context_dim += self._similarity_dim()
        return context_dim

    def _gate_input_dim(self) -> int:
        base_dim = self._base_dim()
        context_dim = 4 * base_dim
        if self.include_neighbor_variance:
            context_dim += base_dim
        context_dim += self._degree_dim()
        context_dim += self._similarity_dim()
        return context_dim + 1

    def _init_gate_bias(self, null_bias_init: float) -> None:
        final = self.gate[-1]
        if isinstance(final, nn.Linear) and final.bias is not None:
            nn.init.zeros_(final.bias)
            final.bias.data[2] = float(null_bias_init)

    def _prompt_z(self, z: torch.Tensor) -> torch.Tensor:
        return z.detach() if self.z_for_prompt == "detach" else z

    def _prompt_h(self, h_pre: torch.Tensor) -> torch.Tensor:
        return h_pre.detach() if self.h_pre_for_prompt == "detach" else h_pre

    def _base_features(self, z: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        if self.structural_base == "h_pre_detached":
            return h_pre.detach()
        if self.structural_base == "h_pre":
            return h_pre
        if self.structural_base == "z_detached":
            return z.detach()
        if self.structural_base == "z":
            return z
        raise ValueError(f"Unsupported structural base: {self.structural_base}")

    def _class_sums_counts(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        train_idx = torch.where(train_mask)[0]
        train_labels = labels[train_idx].long()
        sums = values.new_zeros((self.num_classes, values.size(-1)))
        counts = values.new_zeros((self.num_classes, 1))
        if train_idx.numel() > 0:
            sums.index_add_(0, train_labels, values[train_idx])
            counts.index_add_(
                0,
                train_labels,
                torch.ones((train_idx.numel(), 1), dtype=values.dtype, device=values.device),
            )
        if bool((counts.squeeze(-1) <= 0).any()):
            missing = torch.where(counts.squeeze(-1) <= 0)[0].detach().cpu().tolist()
            raise ValueError(f"Every class needs at least one train label for semantic prototypes; missing={missing}")
        return sums, counts

    def _semantic_message(
        self,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        train_mask: torch.Tensor,
        labels: torch.Tensor,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self._prompt_z(z)
        h_e = self._prompt_h(h_pre)
        z_sums, counts = self._class_sums_counts(z_e, labels, train_mask)
        h_sums, _ = self._class_sums_counts(h_e, labels, train_mask)
        z_proto = z_sums / counts.clamp_min(1.0)
        h_proto = h_sums / counts.clamp_min(1.0)

        scores = z.new_zeros((z.size(0), self.num_classes))
        if self.use_feature_proto:
            scores = scores + self.beta_z * _cosine_scores(z_e, z_proto)
        if self.use_hidden_proto:
            scores = scores + self.beta_h * _cosine_scores(h_e, h_proto)

        semantic_allowed = torch.ones(z.size(0), dtype=torch.bool, device=z.device)
        loo_z_proto = None
        if split == "train" and self.use_leave_one_out:
            train_idx = torch.where(train_mask)[0]
            if train_idx.numel() > 0:
                train_labels = labels[train_idx].long()
                own_counts = counts[train_labels]
                valid_loo = own_counts.squeeze(-1) > 1
                invalid_loo_idx = train_idx[~valid_loo]
                if invalid_loo_idx.numel() > 0 and self.one_shot_train_semantic == "mask":
                    semantic_allowed[invalid_loo_idx] = False
                if bool(valid_loo.any()):
                    valid_idx = train_idx[valid_loo]
                    valid_labels = labels[valid_idx].long()
                    denom = (counts[valid_labels] - 1.0).clamp_min(1.0)
                    loo_z_proto = (z_sums[valid_labels] - z_e[valid_idx]) / denom
                    loo_h_proto = (h_sums[valid_labels] - h_e[valid_idx]) / denom
                    own_scores = scores[valid_idx, valid_labels]
                    if self.use_feature_proto:
                        own_scores = self.beta_z * F.cosine_similarity(z_e[valid_idx], loo_z_proto, dim=-1)
                    else:
                        own_scores = torch.zeros_like(own_scores)
                    if self.use_hidden_proto:
                        own_scores = own_scores + self.beta_h * F.cosine_similarity(
                            h_e[valid_idx], loo_h_proto, dim=-1
                        )
                    scores[valid_idx, valid_labels] = own_scores

        if self.num_classes > 1:
            top2 = torch.topk(scores, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        if self.use_margin_mask:
            semantic_allowed = semantic_allowed & (margin >= self.margin_threshold)

        attention = torch.softmax(scores / max(self.tau_sem, 1e-6), dim=1)
        support = attention @ z_proto
        if split == "train" and self.use_leave_one_out and loo_z_proto is not None:
            train_idx = torch.where(train_mask)[0]
            train_labels = labels[train_idx].long()
            valid_loo = (counts[train_labels].squeeze(-1) > 1)
            if bool(valid_loo.any()):
                valid_idx = train_idx[valid_loo]
                valid_labels = labels[valid_idx].long()
                delta = loo_z_proto - z_proto[valid_labels]
                support[valid_idx] = support[valid_idx] + attention[valid_idx, valid_labels].unsqueeze(-1) * delta
        u_sem = self.semantic_projector(support)
        return u_sem, scores, margin, semantic_allowed

    def _materialize_structural_context(self, aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = aux["base"]
        m1 = aux["m1"]
        m2 = aux["m2"]
        context_parts = [base, m1, m2, base - m1, m1 - m2]
        gate_parts = [base, m1, m2, base - m1]
        if self.include_neighbor_variance and "var1" in aux:
            var1 = aux["var1"]
            context_parts.append(var1)
            gate_parts.append(var1)
        if self.include_degree_features and "degree" in aux:
            degree = aux["degree"]
            context_parts.append(degree)
            gate_parts.append(degree)
        if self.include_similarity_features and "similarity" in aux:
            sim = aux["similarity"]
            context_parts.append(sim)
            gate_parts.append(sim)
        aux = dict(aux)
        aux["structural_context"] = torch.cat(context_parts, dim=-1)
        aux["gate_context"] = torch.cat(gate_parts, dim=-1)
        return aux

    def _build_structural_context(
        self,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = self._base_features(z, h_pre)
        m1 = mean_neighbor_summary(base, edge_index, num_nodes=base.size(0))
        m2 = mean_neighbor_summary(m1, edge_index, num_nodes=base.size(0))
        aux: dict[str, torch.Tensor] = {"base": base, "m1": m1, "m2": m2}
        if self.include_neighbor_variance:
            aux["var1"] = mean_neighbor_variance(base, edge_index, num_nodes=base.size(0))
        if self.include_degree_features:
            aux["degree"] = degree_summary(edge_index, num_nodes=base.size(0), dtype=base.dtype, device=base.device)
        if self.include_similarity_features:
            aux["similarity"] = torch.cat([_safe_cosine_column(base, m1), _safe_cosine_column(m1, m2)], dim=-1)
        return self._materialize_structural_context(aux)

    def _validate_structural_cache(self, cache: dict[str, torch.Tensor], num_nodes: int) -> None:
        required = {"base", "m1", "m2"}
        if self.include_neighbor_variance:
            required.add("var1")
        if self.include_degree_features:
            required.add("degree")
        if self.include_similarity_features:
            required.add("similarity")
        missing = sorted(required - set(cache))
        if missing:
            raise ValueError(f"structural_cache missing required keys: {missing}")
        if any(value.size(0) != num_nodes for key, value in cache.items() if isinstance(value, torch.Tensor)):
            raise ValueError("structural_cache node count does not match current graph")

    @torch.no_grad()
    def build_structural_cache(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Precompute label-free structural context for frozen-base prompt-only runs."""

        full = self._build_structural_context(z, h_pre, edge_index)
        cache_keys = ["base", "m1", "m2"]
        if self.include_neighbor_variance:
            cache_keys.append("var1")
        if self.include_degree_features:
            cache_keys.append("degree")
        if self.include_similarity_features:
            cache_keys.append("similarity")
        return {key: full[key].detach() for key in cache_keys}

    def _structural_message(
        self,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        structural_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if structural_cache is None:
            aux = self._build_structural_context(z, h_pre, edge_index)
            cache_hit = 0.0
        else:
            self._validate_structural_cache(structural_cache, z.size(0))
            aux = self._materialize_structural_context(dict(structural_cache))
            cache_hit = 1.0
        u_struct = self.structural_projector(aux["structural_context"])
        aux["structural_cache_hit"] = aux["structural_context"].new_tensor(cache_hit)
        return u_struct, aux

    def forward(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        y: torch.Tensor,
        split: str = "train",
        structural_cache: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        u_sem, sem_scores, sem_margin, sem_allowed = self._semantic_message(z, h_pre, train_mask, y, split)
        u_struct, struct_aux = self._structural_message(z, h_pre, edge_index, structural_cache=structural_cache)
        if not self.semantic_enabled:
            sem_allowed = torch.zeros_like(sem_allowed)
            u_sem = torch.zeros_like(u_sem)
        if not self.structural_enabled:
            u_struct = torch.zeros_like(u_struct)

        gate_input = torch.cat([struct_aux["gate_context"], sem_margin.unsqueeze(-1)], dim=-1)
        gate_logits = self.gate(gate_input)
        route_allowed = torch.ones_like(gate_logits, dtype=torch.bool)
        route_allowed[:, 0] = sem_allowed & self.semantic_enabled
        route_allowed[:, 1] = self.structural_enabled
        route_allowed[:, 2] = self.null_enabled
        if not bool(route_allowed.any(dim=1).all()):
            route_allowed[:, 2] = True
        gate_logits = gate_logits.masked_fill(~route_allowed, -1.0e9)
        gate = torch.softmax(gate_logits, dim=-1)

        u_prompt = gate[:, 0:1] * u_sem + gate[:, 1:2] * u_struct
        adapted_x = z + self.gamma * u_prompt
        aux = {
            "semantic_scores": sem_scores,
            "semantic_margin": sem_margin,
            "semantic_allowed": sem_allowed,
            "m1": struct_aux["m1"],
            "m2": struct_aux["m2"],
            "connected_edge_count": 0,
        }
        for key in ("var1", "degree", "similarity", "structural_context", "structural_cache_hit"):
            if key in struct_aux:
                aux[key] = struct_aux[key]
        return {
            "adapted_x": adapted_x,
            "u_sem": u_sem,
            "u_struct": u_struct,
            "u_prompt": u_prompt,
            "gate": gate,
            "gamma": self.gamma,
            "aux": aux,
        }


class ParameterMatchedResidualControl(nn.Module):
    """Label-free residual control with a comparable gate/gamma interface."""

    def __init__(
        self,
        source_dim: int,
        hidden_dim: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = _as_config(config)
        gate_cfg = _as_config(self.config.get("gate"))
        dropout = float(self.config.get("control_dropout", gate_cfg.get("dropout", 0.3)))
        self.control_sem = _make_zero_init_mlp(self.source_dim, self.hidden_dim, self.source_dim, dropout=dropout)
        self.control_struct = _make_zero_init_mlp(self.source_dim, self.hidden_dim, self.source_dim, dropout=dropout)
        gate_hidden = int(gate_cfg.get("hidden_dim", 64))
        self.gate = nn.Sequential(
            nn.Linear(self.source_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(float(gate_cfg.get("dropout", 0.3))),
            nn.Linear(gate_hidden, 3),
        )
        final = self.gate[-1]
        if isinstance(final, nn.Linear) and final.bias is not None:
            nn.init.zeros_(final.bias)
            final.bias.data[2] = float(gate_cfg.get("null_bias_init", 1.0))
        self.gamma_max = float(self.config.get("gamma_max", 0.5))
        self.gamma_logit = nn.Parameter(_init_gamma_logit(float(self.config.get("gamma_init", 0.05)), self.gamma_max))

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.gamma_logit)

    def forward(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        split: str = "train",
        structural_cache: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        del h_pre, edge_index, train_mask, y, split, structural_cache
        evidence = z.detach()
        u_sem = self.control_sem(evidence)
        u_struct = self.control_struct(evidence)
        gate = torch.softmax(self.gate(evidence), dim=-1)
        u_prompt = gate[:, 0:1] * u_sem + gate[:, 1:2] * u_struct
        adapted_x = z + self.gamma * u_prompt
        aux = {
            "semantic_margin": z.new_zeros(z.size(0)),
            "semantic_allowed": torch.ones(z.size(0), dtype=torch.bool, device=z.device),
            "connected_edge_count": 0,
        }
        return {
            "adapted_x": adapted_x,
            "u_sem": u_sem,
            "u_struct": u_struct,
            "u_prompt": u_prompt,
            "gate": gate,
            "gamma": self.gamma,
            "aux": aux,
        }
