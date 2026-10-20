"""Prompt graph construction for P1 adapted-branch graph prompting."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary, mean_neighbor_variance


def _as_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _init_logit(value: float, max_value: float) -> torch.Tensor:
    max_value = float(max_value)
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    ratio = min(max(float(value) / max_value, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def _safe_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x, y, dim=-1, eps=1e-12)


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


def _deterministic_random_scores(num_nodes: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(num_nodes, dtype=dtype, device=device)
    scores = torch.sin(idx * 12.9898 + float(seed) * 78.233) * 43758.5453
    return scores - torch.floor(scores)


def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    entropy = -(values * values.clamp_min(1e-12).log()).sum()
    if values.numel() > 1:
        entropy = entropy / math.log(float(values.numel()))
    return entropy


def _mean_route_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.numel() == 0 or logits.size(1) < 2:
        return logits.new_tensor(0.0)
    top2 = torch.topk(logits, k=2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).mean()


class PromptGraphModuleP1(nn.Module):
    """Build a prompt-augmented graph for the adapted GP2F branch.

    Prompt nodes are appended after original nodes. Prompt edges are
    bidirectional and only appear in the adapted branch graph.
    """

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

        self.num_prompt_nodes = int(self.config.get("num_prompt_nodes", 8))
        if self.num_prompt_nodes <= 0:
            raise ValueError("num_prompt_nodes must be positive")
        self.use_class_aware_routing = bool(self.config.get("use_class_aware_routing", False))
        self.num_classes = int(self.config.get("num_classes", 0))
        self.residual_prompt_count = int(self.config.get("residual_prompt_count", 2))
        if self.use_class_aware_routing:
            if self.num_classes <= 0:
                raise ValueError("num_classes must be positive when use_class_aware_routing=true")
            if self.residual_prompt_count < 0:
                raise ValueError("residual_prompt_count must be non-negative")
            min_prompt_nodes = self.num_classes + self.residual_prompt_count
            self.num_prompt_nodes = max(self.num_prompt_nodes, min_prompt_nodes)
        self.num_class_prompt_slots = self.num_classes if self.use_class_aware_routing else 0
        self.rho = float(self.config.get("rho", 0.2))
        self.topk_prompt_per_node = int(self.config.get("topk_prompt_per_node", 2))
        if self.topk_prompt_per_node <= 0:
            raise ValueError("topk_prompt_per_node must be positive")
        self.structural_base = str(self.config.get("structural_base", "z_detached"))
        self.pool_strategy = str(self.config.get("pool_strategy", "structural"))
        if self.pool_strategy not in {"structural", "random", "utility_structural"}:
            raise ValueError(f"Unsupported pool_strategy={self.pool_strategy!r}")
        self.utility_pool_structural_weight = float(self.config.get("utility_pool_structural_weight", 0.35))
        self.utility_pool_uncertainty_weight = float(self.config.get("utility_pool_uncertainty_weight", 0.35))
        self.utility_pool_disagreement_weight = float(self.config.get("utility_pool_disagreement_weight", 0.30))
        self.random_seed = int(self.config.get("random_seed", 0))
        self.tau = float(self.config.get("tau", 0.5))
        self.normalize_query_key = bool(self.config.get("normalize_query_key", True))
        self.use_capacity_routing = bool(self.config.get("use_capacity_routing", False))
        self.capacity_factor = float(self.config.get("capacity_factor", 1.25))
        self.query_dim = int(self.config.get("query_dim", self.hidden_dim))
        self.query_hidden_dim = int(self.config.get("query_hidden_dim", self.hidden_dim))
        self.query_dropout = float(self.config.get("query_dropout", 0.2))
        self.prompt_init_std = float(self.config.get("prompt_init_std", 0.02))
        self.use_rejection_gate = bool(self.config.get("use_rejection_gate", False))
        self.rejection_gate_hidden_dim = int(self.config.get("rejection_gate_hidden_dim", self.query_hidden_dim))
        self.rejection_gate_bias_init = float(self.config.get("rejection_gate_bias_init", -1.0))
        self.rejection_gate_use_role_context = bool(self.config.get("rejection_gate_use_role_context", False))
        self.rejection_gate_use_routing_features = bool(self.config.get("rejection_gate_use_routing_features", False))
        self.use_hard_acceptance = bool(self.config.get("use_hard_acceptance", False))
        self.hard_acceptance_ratio = float(self.config.get("hard_acceptance_ratio", 0.10))
        self.hard_acceptance_straight_through = bool(
            self.config.get("hard_acceptance_straight_through", True)
        )
        self.use_multiview_routing = bool(self.config.get("use_multiview_routing", False))
        self.use_attribute_view = bool(self.config.get("use_attribute_view", False))
        self.use_enhanced_role_view = bool(self.config.get("use_enhanced_role_view", False))
        self.view_count = 4 if self.use_attribute_view else 3
        self.view_gate_hidden_dim = int(self.config.get("view_gate_hidden_dim", self.query_hidden_dim))
        self.role_context_dim = 8 if self.use_enhanced_role_view else 6
        self.role_hidden_dim = int(self.config.get("role_hidden_dim", max(16, self.query_hidden_dim // 2)))
        self.use_pattern_prompt_bank = bool(self.config.get("use_pattern_prompt_bank", False))
        self.use_receiver_only_prompt = bool(self.config.get("use_receiver_only_prompt", False))
        self.use_benefit_gate = bool(self.config.get("use_benefit_gate", False))
        self.benefit_gate_hidden_dim = int(self.config.get("benefit_gate_hidden_dim", self.query_hidden_dim))
        self.benefit_gate_bias_init = float(self.config.get("benefit_gate_bias_init", -1.0))
        self.use_utility_receive_gate = bool(self.config.get("use_utility_receive_gate", False))
        self.utility_receive_gate_hidden_dim = int(
            self.config.get("utility_receive_gate_hidden_dim", self.query_hidden_dim)
        )
        self.utility_receive_gate_min = float(self.config.get("utility_receive_gate_min", 0.10))
        if not 0.0 <= self.utility_receive_gate_min < 1.0:
            raise ValueError("utility_receive_gate_min must be in [0, 1)")
        self.utility_receive_gate_init = float(self.config.get("utility_receive_gate_init", 0.50))
        self.utility_receive_gate_init = min(max(self.utility_receive_gate_init, 1e-6), 1.0 - 1e-6)
        self.use_hard_receive_gate = bool(self.config.get("use_hard_receive_gate", False))
        self.hard_receive_ratio = float(self.config.get("hard_receive_ratio", 0.15))
        self.hard_receive_straight_through = bool(self.config.get("hard_receive_straight_through", True))
        self.use_edge_utility = bool(self.config.get("use_edge_utility", False))
        self.edge_utility_hidden_dim = int(self.config.get("edge_utility_hidden_dim", self.query_hidden_dim))
        self.edge_utility_init = float(self.config.get("edge_utility_init", 0.50))
        self.edge_utility_init = min(max(self.edge_utility_init, 1e-6), 1.0 - 1e-6)
        self.class_key_init_coverage = 0.0
        self.class_key_init_missing_classes: list[int] = []
        self.pattern_key_init_coverage = 0.0
        self.pattern_key_init_selected_nodes: list[int] = []

        self.prompt_node_x = nn.Parameter(torch.empty(self.num_prompt_nodes, self.source_dim))
        nn.init.normal_(self.prompt_node_x, mean=0.0, std=self.prompt_init_std)
        self.prompt_keys = nn.Parameter(torch.empty(self.num_prompt_nodes, self.query_dim))
        nn.init.xavier_uniform_(self.prompt_keys)
        self.semantic_prompt_keys = nn.Parameter(torch.empty(self.num_prompt_nodes, self.query_dim))
        self.structural_prompt_keys = nn.Parameter(torch.empty(self.num_prompt_nodes, self.query_dim))
        self.role_prompt_keys = nn.Parameter(torch.empty(self.num_prompt_nodes, self.query_dim))
        nn.init.xavier_uniform_(self.semantic_prompt_keys)
        nn.init.xavier_uniform_(self.structural_prompt_keys)
        nn.init.xavier_uniform_(self.role_prompt_keys)

        base_dim = self._base_dim()
        self.query_mlp = nn.Sequential(
            nn.Linear(5 * base_dim, self.query_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.query_hidden_dim, self.query_dim),
        )
        self.semantic_query_mlp = nn.Sequential(
            nn.Linear(base_dim, self.query_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.query_hidden_dim, self.query_dim),
        )
        self.structural_query_mlp = nn.Sequential(
            nn.Linear(5 * base_dim, self.query_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.query_hidden_dim, self.query_dim),
        )
        self.role_query_mlp = nn.Sequential(
            nn.Linear(self.role_context_dim, self.role_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.role_hidden_dim, self.query_dim),
        )
        self.attribute_query_mlp = nn.Sequential(
            nn.Linear(self.source_dim, self.query_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.query_hidden_dim, self.query_dim),
        )
        self.view_gate_mlp = nn.Sequential(
            nn.Linear(5 * base_dim + self.role_context_dim, self.view_gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.view_gate_hidden_dim, self.view_count),
        )
        rejection_gate_input_dim = 5 * base_dim
        if self.rejection_gate_use_role_context:
            rejection_gate_input_dim += self.role_context_dim
        if self.rejection_gate_use_routing_features:
            rejection_gate_input_dim += 4
        self.rejection_gate_mlp = nn.Sequential(
            nn.Linear(rejection_gate_input_dim, self.rejection_gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.rejection_gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.rejection_gate_mlp[-1].weight)
        nn.init.constant_(self.rejection_gate_mlp[-1].bias, self.rejection_gate_bias_init)
        self.benefit_gate_mlp = nn.Sequential(
            nn.Linear(3 * self.query_dim + 3, self.benefit_gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.benefit_gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.benefit_gate_mlp[-1].weight)
        nn.init.constant_(self.benefit_gate_mlp[-1].bias, self.benefit_gate_bias_init)
        utility_gate_input_dim = 5 * base_dim + self.role_context_dim + 5
        self.utility_receive_gate_mlp = nn.Sequential(
            nn.Linear(utility_gate_input_dim, self.utility_receive_gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.utility_receive_gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.utility_receive_gate_mlp[-1].weight)
        nn.init.constant_(self.utility_receive_gate_mlp[-1].bias, torch.logit(torch.tensor(self.utility_receive_gate_init)).item())
        edge_utility_input_dim = 3 * self.query_dim + 4 + self.view_count
        self.edge_utility_mlp = nn.Sequential(
            nn.Linear(edge_utility_input_dim, self.edge_utility_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.query_dropout),
            nn.Linear(self.edge_utility_hidden_dim, 1),
        )
        nn.init.zeros_(self.edge_utility_mlp[-1].weight)
        nn.init.constant_(self.edge_utility_mlp[-1].bias, torch.logit(torch.tensor(self.edge_utility_init)).item())
        self.edge_scale_max = float(self.config.get("edge_scale_max", 0.2))
        edge_scale_init = float(self.config.get("edge_scale_init", 0.01))
        self.edge_scale_logit = nn.Parameter(_init_logit(edge_scale_init, self.edge_scale_max))

    @property
    def edge_scale(self) -> torch.Tensor:
        return self.edge_scale_max * torch.sigmoid(self.edge_scale_logit)

    def _base_dim(self) -> int:
        if self.structural_base in {"z_detached", "z"}:
            return self.source_dim
        if self.structural_base in {"h_pre_detached", "h_pre"}:
            return self.hidden_dim
        raise ValueError(f"Unsupported structural_base={self.structural_base!r}")

    def _base_features(self, z: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        if self.structural_base == "z_detached":
            return z.detach()
        if self.structural_base == "z":
            return z
        if self.structural_base == "h_pre_detached":
            return h_pre.detach()
        if self.structural_base == "h_pre":
            return h_pre
        raise ValueError(f"Unsupported structural_base={self.structural_base!r}")

    def structural_scores(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = self._base_features(z, h_pre)
        m1 = mean_neighbor_summary(base, edge_index, num_nodes=base.size(0))
        m2 = mean_neighbor_summary(m1, edge_index, num_nodes=base.size(0))
        var = mean_neighbor_variance(base, edge_index, num_nodes=base.size(0)).mean(dim=-1)
        sim_1 = 1.0 - _safe_cosine(base, m1)
        sim_2 = 1.0 - _safe_cosine(m1, m2)
        var_norm = _minmax_normalize(var)
        score = torch.stack([sim_1.clamp_min(0.0), sim_2.clamp_min(0.0), var_norm], dim=-1).mean(dim=-1)
        context = torch.cat([base, m1, m2, base - m1, m1 - m2], dim=-1)
        degree = torch.zeros(base.size(0), dtype=base.dtype, device=base.device)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=base.dtype, device=base.device))
        base_m1_norm = (base - m1).norm(dim=-1)
        m1_m2_norm = (m1 - m2).norm(dim=-1)
        role_features = [
            _minmax_normalize(degree),
            sim_1.clamp_min(0.0),
            sim_2.clamp_min(0.0),
            var_norm,
            base_m1_norm,
            m1_m2_norm,
        ]
        if self.use_enhanced_role_view:
            role_features = [
                _minmax_normalize(degree),
                _minmax_normalize(torch.log1p(degree)),
                sim_1.clamp_min(0.0),
                sim_2.clamp_min(0.0),
                base_m1_norm,
                m1_m2_norm,
                var_norm,
                score,
            ]
        role_context = torch.stack(role_features, dim=-1)
        return score, {
            "base": base,
            "attribute_base": z.detach(),
            "m1": m1,
            "m2": m2,
            "neighbor_variance": var,
            "neighbor_variance_norm": var_norm,
            "structural_score": score,
            "query_context": context,
            "role_context": role_context,
        }

    def _utility_pool_score(
        self,
        *,
        structural_score: torch.Tensor,
        h_pre: torch.Tensor,
        no_prompt_logits: torch.Tensor | None = None,
        h_adp_no_prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structural_component = _minmax_normalize(structural_score.detach()).to(dtype=structural_score.dtype)
        uncertainty = torch.zeros_like(structural_component)
        disagreement = torch.zeros_like(structural_component)

        if no_prompt_logits is not None:
            logits = no_prompt_logits.detach().to(device=structural_score.device, dtype=structural_score.dtype)
            prob = torch.softmax(logits, dim=-1)
            entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
            if logits.size(-1) > 1:
                entropy = entropy / math.log(float(logits.size(-1)))
            uncertainty = entropy.clamp_min(0.0)

        if h_adp_no_prompt is not None and h_pre is not None:
            h_left = h_pre.detach().to(device=structural_score.device, dtype=structural_score.dtype)
            h_right = h_adp_no_prompt.detach().to(device=structural_score.device, dtype=structural_score.dtype)
            if h_left.shape == h_right.shape:
                disagreement = (1.0 - _safe_cosine(h_left, h_right)).clamp_min(0.0)
                disagreement = _minmax_normalize(disagreement)

        score = (
            float(self.utility_pool_structural_weight) * structural_component
            + float(self.utility_pool_uncertainty_weight) * uncertainty
            + float(self.utility_pool_disagreement_weight) * disagreement
        )
        return score, {
            "pool_score": score,
            "pool_structural_component": structural_component,
            "pool_uncertainty_component": uncertainty,
            "pool_disagreement_component": disagreement,
        }

    def _routing_logits(
        self,
        aux: dict[str, torch.Tensor],
        pool_idx: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.use_multiview_routing:
            query = self.query_mlp(aux["query_context"][pool_idx])
            prompt_keys = self.prompt_keys.to(dtype=dtype, device=device)
            if self.normalize_query_key:
                query_for_logits = F.normalize(query, dim=-1, eps=1e-12)
                keys_for_logits = F.normalize(prompt_keys, dim=-1, eps=1e-12)
            else:
                query_for_logits = query
                keys_for_logits = prompt_keys
            logits = query_for_logits @ keys_for_logits.t()
            logits = logits / max(self.tau, 1e-6)
            view_gate = torch.zeros((pool_idx.numel(), self.view_count), dtype=dtype, device=device)
            if view_gate.numel() > 0:
                view_gate[:, 1] = 1.0
            return logits, {
                "query": query,
                "semantic_query": query,
                "structural_query": query,
                "role_query": query,
                "attribute_query": query,
                "semantic_logits": logits,
                "structural_logits": logits,
                "role_logits": logits,
                "attribute_logits": logits,
                "view_gate": view_gate,
            }

        semantic_query = self.semantic_query_mlp(aux["base"][pool_idx])
        structural_query = self.structural_query_mlp(aux["query_context"][pool_idx])
        role_query = self.role_query_mlp(aux["role_context"][pool_idx])
        attribute_query = self.attribute_query_mlp(aux["attribute_base"][pool_idx])
        attribute_keys = self.attribute_query_mlp(self.prompt_node_x.to(dtype=dtype, device=device))
        semantic_keys = self.semantic_prompt_keys.to(dtype=dtype, device=device)
        structural_keys = self.structural_prompt_keys.to(dtype=dtype, device=device)
        role_keys = self.role_prompt_keys.to(dtype=dtype, device=device)
        if self.normalize_query_key:
            semantic_query_for_logits = F.normalize(semantic_query, dim=-1, eps=1e-12)
            structural_query_for_logits = F.normalize(structural_query, dim=-1, eps=1e-12)
            role_query_for_logits = F.normalize(role_query, dim=-1, eps=1e-12)
            attribute_query_for_logits = F.normalize(attribute_query, dim=-1, eps=1e-12)
            semantic_keys_for_logits = F.normalize(semantic_keys, dim=-1, eps=1e-12)
            structural_keys_for_logits = F.normalize(structural_keys, dim=-1, eps=1e-12)
            role_keys_for_logits = F.normalize(role_keys, dim=-1, eps=1e-12)
            attribute_keys_for_logits = F.normalize(attribute_keys, dim=-1, eps=1e-12)
        else:
            semantic_query_for_logits = semantic_query
            structural_query_for_logits = structural_query
            role_query_for_logits = role_query
            attribute_query_for_logits = attribute_query
            semantic_keys_for_logits = semantic_keys
            structural_keys_for_logits = structural_keys
            role_keys_for_logits = role_keys
            attribute_keys_for_logits = attribute_keys
        semantic_logits = semantic_query_for_logits @ semantic_keys_for_logits.t()
        structural_logits = structural_query_for_logits @ structural_keys_for_logits.t()
        role_logits = role_query_for_logits @ role_keys_for_logits.t()
        attribute_logits = attribute_query_for_logits @ attribute_keys_for_logits.t()
        view_context = torch.cat([aux["query_context"][pool_idx], aux["role_context"][pool_idx]], dim=-1)
        view_gate = torch.softmax(self.view_gate_mlp(view_context), dim=-1)
        views = [semantic_logits, structural_logits, role_logits]
        if self.use_attribute_view:
            views.append(attribute_logits)
        stacked_logits = torch.stack(views, dim=1)
        logits = (view_gate.unsqueeze(-1) * stacked_logits).sum(dim=1)
        logits = logits / max(self.tau, 1e-6)
        return logits, {
            "query": structural_query,
            "semantic_query": semantic_query,
            "structural_query": structural_query,
            "role_query": role_query,
            "attribute_query": attribute_query,
            "semantic_logits": semantic_logits / max(self.tau, 1e-6),
            "structural_logits": structural_logits / max(self.tau, 1e-6),
            "role_logits": role_logits / max(self.tau, 1e-6),
            "attribute_logits": attribute_logits / max(self.tau, 1e-6),
            "view_gate": view_gate,
        }

    @torch.no_grad()
    def initialize_class_keys_from_train_prototypes(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
        source: str = "structural_query",
    ) -> dict[str, Any]:
        """Initialize class routing keys from train-only structural prototypes.

        This method must be called after a split is built. It only reads labels
        at ``train_mask`` nodes and never inspects validation/test labels.
        """

        if not self.use_class_aware_routing or self.num_class_prompt_slots <= 0:
            self.class_key_init_coverage = 0.0
            self.class_key_init_missing_classes = []
            return {"class_key_proto_init_coverage": 0.0, "class_key_proto_init_missing_classes": []}

        if source not in {"query", "structural_query"}:
            raise ValueError("class_key_init_source must be 'query' or 'structural_query'")

        self.eval()
        structural_score, aux = self.structural_scores(z=z, h_pre=h_pre, edge_index=edge_index)
        pool_mask = self._pool_mask(structural_score, train_mask)
        pool_idx = torch.where(pool_mask)[0]
        if pool_idx.numel() == 0:
            self.class_key_init_coverage = 0.0
            self.class_key_init_missing_classes = list(range(self.num_class_prompt_slots))
            return {
                "class_key_proto_init_coverage": 0.0,
                "class_key_proto_init_missing_classes": self.class_key_init_missing_classes,
            }
        _, routing_aux = self._routing_logits(aux, pool_idx, dtype=z.dtype, device=z.device)
        query = routing_aux[source]
        train_pool = train_mask.to(device=z.device, dtype=torch.bool)[pool_idx]
        y = labels.to(device=z.device)[pool_idx]

        if self.use_multiview_routing:
            key_targets = [
                self.semantic_prompt_keys,
                self.structural_prompt_keys,
                self.role_prompt_keys,
            ]
        else:
            key_targets = [self.prompt_keys]

        initialized = 0
        missing: list[int] = []
        for class_id in range(self.num_class_prompt_slots):
            rows = train_pool & (y == class_id)
            if not bool(rows.any()):
                missing.append(class_id)
                continue
            proto = query[rows].mean(dim=0)
            if normalize:
                proto = F.normalize(proto, dim=0, eps=1e-12)
            for key_param in key_targets:
                key_param[class_id].copy_(proto.to(dtype=key_param.dtype, device=key_param.device))
            initialized += 1

        coverage = float(initialized / max(1, self.num_class_prompt_slots))
        self.class_key_init_coverage = coverage
        self.class_key_init_missing_classes = missing
        return {
            "class_key_proto_init_coverage": coverage,
            "class_key_proto_init_missing_classes": missing,
        }

    @torch.no_grad()
    def initialize_pattern_keys_from_pool_medoids(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        normalize: bool = True,
    ) -> dict[str, Any]:
        """Initialize prompt keys from label-free pool pattern medoids.

        Medoids are actual pool-node queries selected by deterministic farthest
        traversal. This avoids smoothing multi-modal patterns into mean centers.
        """

        self.eval()
        structural_score, aux = self.structural_scores(z=z, h_pre=h_pre, edge_index=edge_index)
        pool_mask = self._pool_mask(structural_score, train_mask)
        pool_idx = torch.where(pool_mask)[0]
        if pool_idx.numel() == 0:
            self.pattern_key_init_coverage = 0.0
            self.pattern_key_init_selected_nodes = []
            return {"pattern_key_init_coverage": 0.0, "pattern_key_init_selected_nodes": []}

        _, routing_aux = self._routing_logits(aux, pool_idx, dtype=z.dtype, device=z.device)
        structural_query = routing_aux["structural_query"]
        query_for_select = F.normalize(structural_query, dim=-1, eps=1e-12)
        selected_rows: list[int] = []
        first = int(torch.argmax(structural_score[pool_idx]).item())
        selected_rows.append(first)
        min_dist = torch.cdist(query_for_select, query_for_select[first : first + 1], p=2).squeeze(1)
        target_count = min(self.num_prompt_nodes, int(pool_idx.numel()))
        while len(selected_rows) < target_count:
            next_row = int(torch.argmax(min_dist).item())
            if next_row in selected_rows:
                break
            selected_rows.append(next_row)
            next_dist = torch.cdist(query_for_select, query_for_select[next_row : next_row + 1], p=2).squeeze(1)
            min_dist = torch.minimum(min_dist, next_dist)

        selected = torch.tensor(selected_rows, dtype=torch.long, device=z.device)
        selected_nodes = pool_idx[selected]

        def _copy_keys(parameter: torch.nn.Parameter, values: torch.Tensor) -> None:
            count = min(parameter.size(0), values.size(0))
            copied = values[:count]
            if normalize:
                copied = F.normalize(copied, dim=-1, eps=1e-12)
            parameter[:count].copy_(copied.to(dtype=parameter.dtype, device=parameter.device))

        if self.use_multiview_routing:
            _copy_keys(self.semantic_prompt_keys, routing_aux["semantic_query"][selected])
            _copy_keys(self.structural_prompt_keys, routing_aux["structural_query"][selected])
            _copy_keys(self.role_prompt_keys, routing_aux["role_query"][selected])
        else:
            _copy_keys(self.prompt_keys, routing_aux["query"][selected])

        count = min(self.prompt_node_x.size(0), selected_nodes.numel())
        self.prompt_node_x[:count].copy_(z[selected_nodes[:count]].to(dtype=self.prompt_node_x.dtype))
        coverage = float(count / max(1, self.num_prompt_nodes))
        self.pattern_key_init_coverage = coverage
        self.pattern_key_init_selected_nodes = [int(item) for item in selected_nodes.detach().cpu().tolist()]
        return {
            "pattern_key_init_coverage": coverage,
            "pattern_key_init_selected_nodes": self.pattern_key_init_selected_nodes,
        }

    def _rejection_gate_context(
        self,
        aux: dict[str, torch.Tensor],
        pool_idx: torch.Tensor,
        logits: torch.Tensor,
        full_prob: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [aux["query_context"][pool_idx]]
        if self.rejection_gate_use_role_context:
            pieces.append(aux["role_context"][pool_idx])
        if self.rejection_gate_use_routing_features:
            top_prob = full_prob.max(dim=-1).values
            entropy = -(full_prob * full_prob.clamp_min(1e-12).log()).sum(dim=-1)
            if full_prob.size(1) > 1:
                entropy = entropy / math.log(float(full_prob.size(1)))
            if logits.size(1) > 1:
                top2 = torch.topk(logits, k=2, dim=-1).values
                logit_margin = top2[:, 0] - top2[:, 1]
            else:
                logit_margin = torch.ones_like(top_prob)
            structural_score = aux["structural_score"][pool_idx]
            pieces.append(
                torch.stack(
                    [
                        top_prob,
                        entropy,
                        _minmax_normalize(logit_margin),
                        _minmax_normalize(structural_score),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(pieces, dim=-1)

    def _hard_acceptance_mask(self, pool_acceptance: torch.Tensor) -> torch.Tensor:
        if pool_acceptance.numel() == 0:
            return pool_acceptance
        ratio = min(max(float(self.hard_acceptance_ratio), 0.0), 1.0)
        if ratio <= 0.0:
            return torch.zeros_like(pool_acceptance)
        keep = int(math.ceil(ratio * int(pool_acceptance.numel())))
        keep = min(max(keep, 1), int(pool_acceptance.numel()))
        top_idx = torch.topk(pool_acceptance, k=keep, largest=True).indices
        mask = torch.zeros_like(pool_acceptance)
        mask[top_idx] = 1.0
        return mask

    def _hard_top_ratio_mask(self, score: torch.Tensor, ratio: float) -> torch.Tensor:
        if score.numel() == 0:
            return score
        ratio = min(max(float(ratio), 0.0), 1.0)
        if ratio <= 0.0:
            return torch.zeros_like(score)
        keep = int(math.ceil(ratio * int(score.numel())))
        keep = min(max(keep, 1), int(score.numel()))
        top_idx = torch.topk(score, k=keep, largest=True).indices
        mask = torch.zeros_like(score)
        mask[top_idx] = 1.0
        return mask

    def _benefit_gate(
        self,
        *,
        routing_aux: dict[str, torch.Tensor],
        top_prompt_ids: torch.Tensor,
        assign_prob: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_benefit_gate:
            ones = assign_prob.new_ones(assign_prob.shape)
            logits_out = torch.full_like(assign_prob, 30.0)
            return ones, logits_out
        query = routing_aux["structural_query"]
        if self.use_multiview_routing:
            keys = self.structural_prompt_keys.to(dtype=query.dtype, device=query.device)
        else:
            keys = self.prompt_keys.to(dtype=query.dtype, device=query.device)
        selected_keys = keys[top_prompt_ids]
        query_expanded = query.unsqueeze(1).expand_as(selected_keys)
        route_entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        if logits.size(1) > 1:
            route_entropy = route_entropy / math.log(float(logits.size(1)))
            top2 = torch.topk(logits, k=2, dim=-1).values
            route_margin = top2[:, 0] - top2[:, 1]
        else:
            route_margin = torch.ones_like(route_entropy)
        route_margin = _minmax_normalize(route_margin)
        scalar_features = torch.stack(
            [
                assign_prob,
                route_entropy.unsqueeze(-1).expand_as(assign_prob),
                route_margin.unsqueeze(-1).expand_as(assign_prob),
            ],
            dim=-1,
        )
        gate_input = torch.cat(
            [
                query_expanded,
                selected_keys,
                query_expanded - selected_keys,
                scalar_features,
            ],
            dim=-1,
        )
        benefit_logit = self.benefit_gate_mlp(gate_input).squeeze(-1)
        return torch.sigmoid(benefit_logit), benefit_logit

    def _edge_utility(
        self,
        *,
        aux: dict[str, torch.Tensor],
        routing_aux: dict[str, torch.Tensor],
        pool_idx: torch.Tensor,
        top_prompt_ids: torch.Tensor,
        assign_prob: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_edge_utility:
            ones = assign_prob.new_ones(assign_prob.shape)
            logits_out = torch.full_like(assign_prob, 30.0)
            return ones, logits_out

        query = routing_aux["structural_query"]
        if self.use_multiview_routing:
            keys = self.structural_prompt_keys.to(dtype=query.dtype, device=query.device)
        else:
            keys = self.prompt_keys.to(dtype=query.dtype, device=query.device)
        selected_keys = keys[top_prompt_ids]
        query_expanded = query.unsqueeze(1).expand_as(selected_keys)

        full_prob = torch.softmax(logits, dim=-1)
        route_entropy = -(full_prob * full_prob.clamp_min(1e-12).log()).sum(dim=-1)
        if logits.size(1) > 1:
            route_entropy = route_entropy / math.log(float(logits.size(1)))
            top2 = torch.topk(logits, k=2, dim=-1).values
            route_margin = top2[:, 0] - top2[:, 1]
        else:
            route_margin = torch.ones_like(route_entropy)
        route_margin = _minmax_normalize(route_margin)
        pool_score = aux.get("pool_score", aux.get("structural_score"))[pool_idx]
        pool_score = _minmax_normalize(pool_score)

        scalar_features = torch.stack(
            [
                assign_prob,
                route_entropy.unsqueeze(-1).expand_as(assign_prob),
                route_margin.unsqueeze(-1).expand_as(assign_prob),
                pool_score.unsqueeze(-1).expand_as(assign_prob),
            ],
            dim=-1,
        )
        view_gate = routing_aux.get("view_gate")
        if not isinstance(view_gate, torch.Tensor) or view_gate.numel() == 0:
            view_gate = assign_prob.new_zeros((assign_prob.size(0), self.view_count))
        view_features = view_gate.unsqueeze(1).expand(-1, assign_prob.size(1), -1)
        gate_input = torch.cat(
            [
                query_expanded,
                selected_keys,
                query_expanded - selected_keys,
                scalar_features,
                view_features,
            ],
            dim=-1,
        )
        edge_logit = self.edge_utility_mlp(gate_input).squeeze(-1)
        return torch.sigmoid(edge_logit), edge_logit

    def _utility_receive_gate(
        self,
        *,
        aux: dict[str, torch.Tensor],
        pool_idx: torch.Tensor,
        logits: torch.Tensor,
        full_prob: torch.Tensor,
        benefit_node_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_utility_receive_gate:
            ones = benefit_node_score.new_ones(benefit_node_score.shape)
            logits_out = torch.full_like(benefit_node_score, 30.0)
            return ones, logits_out, ones

        top_prob = full_prob.max(dim=-1).values
        entropy = -(full_prob * full_prob.clamp_min(1e-12).log()).sum(dim=-1)
        if full_prob.size(1) > 1:
            entropy = entropy / math.log(float(full_prob.size(1)))
            top2 = torch.topk(logits, k=2, dim=-1).values
            route_margin = top2[:, 0] - top2[:, 1]
        else:
            route_margin = torch.ones_like(top_prob)
        structural_score = aux["structural_score"][pool_idx]
        scalar_features = torch.stack(
            [
                top_prob,
                entropy,
                _minmax_normalize(route_margin),
                _minmax_normalize(structural_score),
                benefit_node_score,
            ],
            dim=-1,
        )
        gate_input = torch.cat([aux["query_context"][pool_idx], aux["role_context"][pool_idx], scalar_features], dim=-1)
        receive_logit = self.utility_receive_gate_mlp(gate_input).squeeze(-1)
        receive_raw = torch.sigmoid(receive_logit)
        min_gate = float(self.utility_receive_gate_min)
        receive_gate = min_gate + (1.0 - min_gate) * receive_raw
        return receive_gate, receive_logit, receive_raw

    def _pool_mask(
        self,
        scores: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(scores.numel())
        pool_mask = train_mask.bool().clone()
        k = int(math.ceil(max(0.0, self.rho) * num_nodes))
        k = min(max(k, 0), num_nodes)
        if k > 0:
            if self.pool_strategy == "random":
                rank_scores = _deterministic_random_scores(
                    num_nodes,
                    seed=self.random_seed,
                    device=scores.device,
                    dtype=scores.dtype,
                )
            else:
                rank_scores = scores
            top_idx = torch.topk(rank_scores, k=k, largest=True).indices
            pool_mask[top_idx] = True
        return pool_mask

    def _capacity_aware_topk(
        self,
        logits: torch.Tensor,
        *,
        priority: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        if not self.use_capacity_routing:
            top_values, top_prompt_ids = torch.topk(logits, k=k, dim=-1)
            return top_values, top_prompt_ids, {
                "capacity_routing_enabled": 0,
                "prompt_capacity": 0,
                "capacity_overflow_count": 0,
            }

        num_pool, num_prompt = int(logits.size(0)), int(logits.size(1))
        total_assignments = num_pool * int(k)
        capacity = max(1, int(math.ceil(total_assignments / max(1, num_prompt) * max(1.0, self.capacity_factor))))
        counts = [0 for _ in range(num_prompt)]
        selected_rows: list[list[int]] = [[0 for _ in range(k)] for _ in range(num_pool)]
        overflow_count = 0
        order = torch.argsort(priority, descending=True).tolist()
        ranked_prompts = torch.argsort(logits.detach(), dim=-1, descending=True).tolist()

        for row in order:
            selected: list[int] = []
            selected_set: set[int] = set()
            for prompt_id in ranked_prompts[row]:
                if counts[prompt_id] < capacity:
                    selected.append(prompt_id)
                    selected_set.add(prompt_id)
                    counts[prompt_id] += 1
                    if len(selected) == k:
                        break
            if len(selected) < k:
                overflow_count += k - len(selected)
                for prompt_id in ranked_prompts[row]:
                    if prompt_id in selected_set:
                        continue
                    selected.append(prompt_id)
                    counts[prompt_id] += 1
                    if len(selected) == k:
                        break
            selected_rows[row] = selected

        top_prompt_ids = torch.tensor(selected_rows, dtype=torch.long, device=logits.device)
        top_values = logits.gather(1, top_prompt_ids)
        return top_values, top_prompt_ids, {
            "capacity_routing_enabled": 1,
            "prompt_capacity": capacity,
            "capacity_overflow_count": overflow_count,
        }

    def _empty_prompt_graph(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        aux: dict[str, torch.Tensor],
        pool_mask: torch.Tensor,
        effective_edge_scale: torch.Tensor,
        edge_scale_multiplier: torch.Tensor,
    ) -> dict[str, Any]:
        prompt_node_x = self.prompt_node_x.to(dtype=z.dtype, device=z.device)
        adapted_x = torch.cat([z, prompt_node_x], dim=0)
        usage = z.new_zeros(self.num_prompt_nodes)
        num_nodes = int(z.size(0))
        aux = {
            **aux,
            "prompt_edge_weight": z.new_zeros(0),
            "prompt_usage": usage,
            "prompt_usage_full": usage,
            "prompt_usage_entropy": z.new_tensor(0.0),
            "prompt_usage_full_entropy": z.new_tensor(0.0),
            "pool_acceptance_gate": z.new_zeros(0),
            "effective_acceptance_gate": z.new_zeros(0),
            "hard_acceptance_mask": z.new_zeros(0),
            "use_hard_acceptance": int(self.use_hard_acceptance),
            "hard_acceptance_ratio": z.new_tensor(float(self.hard_acceptance_ratio)),
            "hard_acceptance_selected_ratio": z.new_tensor(0.0),
            "pool_acceptance_mean": z.new_tensor(0.0),
            "pool_acceptance_min": z.new_tensor(0.0),
            "pool_acceptance_max": z.new_tensor(0.0),
            "benefit_gate": z.new_zeros((0, self.topk_prompt_per_node)),
            "benefit_gate_logit": z.new_zeros((0, self.topk_prompt_per_node)),
            "benefit_gate_mean": z.new_tensor(0.0),
            "benefit_gate_min": z.new_tensor(0.0),
            "benefit_gate_max": z.new_tensor(0.0),
            "edge_utility": z.new_zeros((0, self.topk_prompt_per_node)),
            "edge_utility_logit": z.new_zeros((0, self.topk_prompt_per_node)),
            "edge_utility_mean": z.new_tensor(0.0),
            "edge_utility_min": z.new_tensor(0.0),
            "edge_utility_max": z.new_tensor(0.0),
            "use_edge_utility": int(self.use_edge_utility),
            "utility_receive_gate": z.new_zeros(0),
            "utility_receive_gate_logit": z.new_zeros(0),
            "utility_receive_gate_raw": z.new_zeros(0),
            "utility_receive_gate_mean": z.new_tensor(0.0),
            "utility_receive_gate_min": z.new_tensor(0.0),
            "utility_receive_gate_max": z.new_tensor(0.0),
            "use_utility_receive_gate": int(self.use_utility_receive_gate),
            "utility_receive_gate_floor": z.new_tensor(float(self.utility_receive_gate_min)),
            "receive_gate_score": z.new_zeros(0),
            "effective_receive_gate": z.new_zeros(0),
            "hard_receive_mask": z.new_zeros(0),
            "use_hard_receive_gate": int(self.use_hard_receive_gate),
            "hard_receive_ratio": z.new_tensor(float(self.hard_receive_ratio)),
            "hard_receive_selected_ratio": z.new_tensor(0.0),
            "receive_gate_mean": z.new_tensor(0.0),
            "receive_gate_min": z.new_tensor(0.0),
            "receive_gate_max": z.new_tensor(0.0),
            "raw_edge_scale": self.edge_scale,
            "edge_scale_multiplier": edge_scale_multiplier,
            "pool_score": aux.get("pool_score", aux.get("structural_score", z.new_zeros(num_nodes))),
            "pool_structural_component": aux.get("pool_structural_component", aux.get("structural_score", z.new_zeros(num_nodes))),
            "pool_uncertainty_component": aux.get("pool_uncertainty_component", z.new_zeros(num_nodes)),
            "pool_disagreement_component": aux.get("pool_disagreement_component", z.new_zeros(num_nodes)),
            "pool_strategy_id": {"structural": 0, "random": 1, "utility_structural": 2}.get(self.pool_strategy, -1),
            "pool_selected_ratio": z.new_tensor(float(pool_mask.float().mean().item()) if pool_mask.numel() > 0 else 0.0),
            "pool_score_mean": z.new_tensor(0.0),
            "pool_score_selected_mean": z.new_tensor(0.0),
            "capacity_routing_enabled": int(self.use_capacity_routing),
            "prompt_capacity": 0,
            "capacity_overflow_count": 0,
            "connected_edge_count": 0,
            "edge_type_counts": [int(edge_index.size(1)), 0, 0],
            "use_multiview_routing": int(self.use_multiview_routing),
            "view_gate_mean": torch.tensor(
                [0.0, 1.0, 0.0, 0.0] if self.use_attribute_view else [0.0, 1.0, 0.0],
                dtype=z.dtype,
                device=z.device,
            ),
            "view_gate_entropy": z.new_tensor(0.0),
            "semantic_route_margin": z.new_tensor(0.0),
            "structural_route_margin": z.new_tensor(0.0),
            "role_route_margin": z.new_tensor(0.0),
            "attribute_route_margin": z.new_tensor(0.0),
            "routing_full_prob": z.new_zeros((0, self.num_prompt_nodes)),
            "pool_idx": torch.empty(0, dtype=torch.long, device=z.device),
            "use_class_aware_routing": int(self.use_class_aware_routing),
            "use_attribute_view": int(self.use_attribute_view),
            "use_enhanced_role_view": int(self.use_enhanced_role_view),
            "use_pattern_prompt_bank": int(self.use_pattern_prompt_bank),
            "use_receiver_only_prompt": int(self.use_receiver_only_prompt),
            "use_benefit_gate": int(self.use_benefit_gate),
            "num_class_prompt_slots": int(self.num_class_prompt_slots),
            "residual_prompt_count": int(self.residual_prompt_count if self.use_class_aware_routing else self.num_prompt_nodes),
            "class_key_proto_init_coverage": float(self.class_key_init_coverage),
            "class_key_proto_init_missing_classes": list(self.class_key_init_missing_classes),
            "pattern_key_init_coverage": float(self.pattern_key_init_coverage),
            "pattern_key_init_selected_nodes": list(self.pattern_key_init_selected_nodes),
        }
        return {
            "adapted_x": adapted_x,
            "adapted_edge_index": edge_index,
            "adapted_edge_weight": edge_weight,
            "adapted_edge_type": torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
            "prompt_node_x": prompt_node_x,
            "pool_mask": pool_mask,
            "prompt_edge_count": 0,
            "edge_scale": effective_edge_scale,
            "aux": aux,
        }

    def forward(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        edge_scale_multiplier: float | torch.Tensor = 1.0,
        no_prompt_logits: torch.Tensor | None = None,
        h_adp_no_prompt: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        num_nodes = int(z.size(0))
        device = z.device
        dtype = z.dtype
        prompt_node_x = self.prompt_node_x.to(dtype=dtype, device=device)
        prompt_offset = num_nodes
        if isinstance(edge_scale_multiplier, torch.Tensor):
            multiplier = edge_scale_multiplier.to(dtype=dtype, device=device)
        else:
            multiplier = z.new_tensor(float(edge_scale_multiplier))
        multiplier = multiplier.clamp_min(0.0)
        effective_edge_scale = self.edge_scale * multiplier

        structural_score, aux = self.structural_scores(z=z, h_pre=h_pre, edge_index=edge_index)
        if self.pool_strategy == "utility_structural":
            pool_score, pool_aux = self._utility_pool_score(
                structural_score=structural_score,
                h_pre=h_pre,
                no_prompt_logits=no_prompt_logits,
                h_adp_no_prompt=h_adp_no_prompt,
            )
        else:
            pool_score, pool_aux = structural_score, {
                "pool_score": structural_score,
                "pool_structural_component": _minmax_normalize(structural_score.detach()).to(dtype=dtype),
                "pool_uncertainty_component": torch.zeros_like(structural_score),
                "pool_disagreement_component": torch.zeros_like(structural_score),
            }
        aux = {**aux, **pool_aux}
        pool_mask = self._pool_mask(pool_score, train_mask)
        pool_idx = torch.where(pool_mask)[0]
        original_edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=device)
        if pool_idx.numel() == 0:
            return self._empty_prompt_graph(
                z,
                edge_index,
                original_edge_weight,
                aux,
                pool_mask,
                effective_edge_scale,
                multiplier,
            )

        logits, routing_aux = self._routing_logits(aux, pool_idx, dtype=dtype, device=device)
        full_prob = torch.softmax(logits, dim=-1)
        k = min(self.topk_prompt_per_node, self.num_prompt_nodes)
        top_values, top_prompt_ids, capacity_info = self._capacity_aware_topk(
            logits,
            priority=pool_score[pool_idx].detach(),
            k=k,
        )
        assign_prob = torch.softmax(top_values, dim=-1)
        if self.use_rejection_gate:
            rejection_context = self._rejection_gate_context(aux, pool_idx, logits, full_prob)
            pool_acceptance_logit = self.rejection_gate_mlp(rejection_context).squeeze(-1)
            pool_acceptance = torch.sigmoid(pool_acceptance_logit)
        else:
            rejection_context = aux["query_context"][pool_idx]
            pool_acceptance_logit = torch.full(
                (pool_idx.numel(),),
                30.0,
                dtype=dtype,
                device=device,
            )
            pool_acceptance = torch.ones(pool_idx.numel(), dtype=dtype, device=device)
        hard_acceptance_mask = self._hard_acceptance_mask(pool_acceptance) if self.use_hard_acceptance else pool_acceptance
        if self.use_hard_acceptance and self.hard_acceptance_straight_through:
            effective_acceptance = hard_acceptance_mask.detach() + pool_acceptance - pool_acceptance.detach()
        else:
            effective_acceptance = hard_acceptance_mask
        benefit_weight, benefit_logit = self._benefit_gate(
            routing_aux=routing_aux,
            top_prompt_ids=top_prompt_ids,
            assign_prob=assign_prob,
            logits=logits,
        )
        edge_utility, edge_utility_logit = self._edge_utility(
            aux=aux,
            routing_aux=routing_aux,
            pool_idx=pool_idx,
            top_prompt_ids=top_prompt_ids,
            assign_prob=assign_prob,
            logits=logits,
        )
        benefit_node_score = (assign_prob * benefit_weight).sum(dim=-1)
        utility_receive_gate, utility_receive_logit, utility_receive_raw = self._utility_receive_gate(
            aux=aux,
            pool_idx=pool_idx,
            logits=logits,
            full_prob=full_prob,
            benefit_node_score=benefit_node_score,
        )
        receive_score = effective_acceptance * benefit_node_score * utility_receive_gate
        if self.use_hard_receive_gate:
            hard_receive_mask = self._hard_top_ratio_mask(receive_score, self.hard_receive_ratio)
            if self.hard_receive_straight_through:
                effective_receive = hard_receive_mask.detach() + receive_score - receive_score.detach()
            else:
                effective_receive = hard_receive_mask
            prompt_weights = effective_edge_scale * effective_receive.unsqueeze(-1) * assign_prob * benefit_weight
        else:
            hard_receive_mask = receive_score
            effective_receive = receive_score
            prompt_weights = (
                effective_edge_scale
                * effective_acceptance.unsqueeze(-1)
                * utility_receive_gate.unsqueeze(-1)
                * assign_prob
                * benefit_weight
            )
        prompt_weights = prompt_weights * edge_utility

        src_node = pool_idx.repeat_interleave(k)
        dst_prompt = (prompt_offset + top_prompt_ids.reshape(-1)).long()
        flat_weights = prompt_weights.reshape(-1)
        prompt_edges_forward = torch.stack([src_node, dst_prompt], dim=0)
        prompt_edges_backward = torch.stack([dst_prompt, src_node], dim=0)
        original_edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        if self.use_receiver_only_prompt:
            prompt_edges = prompt_edges_backward
            prompt_edge_weight = flat_weights
            forward_edge_type = torch.zeros(0, dtype=torch.long, device=device)
            backward_edge_type = torch.full((prompt_edges_backward.size(1),), 2, dtype=torch.long, device=device)
        else:
            prompt_edges = torch.cat([prompt_edges_forward, prompt_edges_backward], dim=1)
            prompt_edge_weight = torch.cat([flat_weights, flat_weights], dim=0)
            forward_edge_type = torch.ones(prompt_edges_forward.size(1), dtype=torch.long, device=device)
            backward_edge_type = torch.full(
                (prompt_edges_backward.size(1),),
                2,
                dtype=torch.long,
                device=device,
            )
        adapted_edge_type = torch.cat([original_edge_type, forward_edge_type, backward_edge_type], dim=0)

        adapted_x = torch.cat([z, prompt_node_x], dim=0)
        adapted_edge_index = torch.cat([edge_index, prompt_edges], dim=1)
        adapted_edge_weight = torch.cat([original_edge_weight, prompt_edge_weight], dim=0)

        usage_raw = torch.zeros(self.num_prompt_nodes, dtype=dtype, device=device)
        usage_raw.index_add_(0, top_prompt_ids.reshape(-1), assign_prob.reshape(-1))
        usage = usage_raw / usage_raw.sum().clamp_min(1e-12)
        usage_full = full_prob.mean(dim=0)
        usage_entropy = _normalized_entropy(usage)
        usage_full_entropy = _normalized_entropy(usage_full)

        aux = {
            **aux,
            **routing_aux,
            "routing_logits": logits,
            "pool_idx": pool_idx,
            "rejection_gate_context": rejection_context,
            "routing_full_prob": full_prob,
            "top_prompt_ids": top_prompt_ids,
            "assignment_prob": assign_prob,
            "prompt_edge_weight": prompt_edge_weight,
            "pool_acceptance_logit": pool_acceptance_logit,
            "pool_acceptance_gate": pool_acceptance,
            "effective_acceptance_gate": effective_acceptance,
            "benefit_gate": benefit_weight,
            "benefit_gate_logit": benefit_logit,
            "benefit_gate_mean": benefit_weight.mean(),
            "benefit_gate_min": benefit_weight.min(),
            "benefit_gate_max": benefit_weight.max(),
            "edge_utility": edge_utility,
            "edge_utility_logit": edge_utility_logit,
            "edge_utility_mean": edge_utility.mean(),
            "edge_utility_min": edge_utility.min(),
            "edge_utility_max": edge_utility.max(),
            "use_edge_utility": int(self.use_edge_utility),
            "utility_receive_gate": utility_receive_gate,
            "utility_receive_gate_logit": utility_receive_logit,
            "utility_receive_gate_raw": utility_receive_raw,
            "utility_receive_gate_mean": utility_receive_gate.mean(),
            "utility_receive_gate_min": utility_receive_gate.min(),
            "utility_receive_gate_max": utility_receive_gate.max(),
            "use_utility_receive_gate": int(self.use_utility_receive_gate),
            "utility_receive_gate_floor": receive_score.new_tensor(float(self.utility_receive_gate_min)),
            "receive_gate_score": receive_score,
            "effective_receive_gate": effective_receive,
            "hard_receive_mask": hard_receive_mask,
            "use_hard_receive_gate": int(self.use_hard_receive_gate),
            "hard_receive_ratio": receive_score.new_tensor(float(self.hard_receive_ratio)),
            "hard_receive_selected_ratio": (
                hard_receive_mask.mean() if self.use_hard_receive_gate else receive_score.new_tensor(1.0)
            ),
            "receive_gate_mean": receive_score.mean(),
            "receive_gate_min": receive_score.min(),
            "receive_gate_max": receive_score.max(),
            "hard_acceptance_mask": hard_acceptance_mask,
            "use_hard_acceptance": int(self.use_hard_acceptance),
            "hard_acceptance_ratio": pool_acceptance.new_tensor(float(self.hard_acceptance_ratio)),
            "hard_acceptance_selected_ratio": hard_acceptance_mask.mean(),
            "pool_acceptance_mean": pool_acceptance.mean(),
            "pool_acceptance_min": pool_acceptance.min(),
            "pool_acceptance_max": pool_acceptance.max(),
            "prompt_usage": usage,
            "prompt_usage_full": usage_full,
            "prompt_usage_entropy": usage_entropy,
            "prompt_usage_full_entropy": usage_full_entropy,
            "raw_edge_scale": self.edge_scale,
            "edge_scale_multiplier": multiplier,
            "pool_score": pool_score,
            "pool_structural_component": aux["pool_structural_component"],
            "pool_uncertainty_component": aux["pool_uncertainty_component"],
            "pool_disagreement_component": aux["pool_disagreement_component"],
            "pool_strategy_id": {"structural": 0, "random": 1, "utility_structural": 2}.get(self.pool_strategy, -1),
            "pool_selected_ratio": pool_mask.float().mean(),
            "pool_score_mean": pool_score.mean(),
            "pool_score_selected_mean": pool_score[pool_idx].mean(),
            **capacity_info,
            "connected_edge_count": int(prompt_edges.size(1)),
            "edge_type_counts": [
                int(original_edge_type.numel()),
                int(forward_edge_type.numel()),
                int(backward_edge_type.numel()),
            ],
            "use_multiview_routing": int(self.use_multiview_routing),
            "use_attribute_view": int(self.use_attribute_view),
            "use_enhanced_role_view": int(self.use_enhanced_role_view),
            "view_gate_mean": routing_aux["view_gate"].mean(dim=0),
            "view_gate_entropy": (
                -(routing_aux["view_gate"] * routing_aux["view_gate"].clamp_min(1e-12).log()).sum(dim=-1).mean()
                / math.log(float(routing_aux["view_gate"].size(1)))
            ),
            "semantic_route_margin": _mean_route_margin(routing_aux["semantic_logits"]),
            "structural_route_margin": _mean_route_margin(routing_aux["structural_logits"]),
            "role_route_margin": _mean_route_margin(routing_aux["role_logits"]),
            "attribute_route_margin": _mean_route_margin(routing_aux["attribute_logits"]),
            "use_class_aware_routing": int(self.use_class_aware_routing),
            "use_pattern_prompt_bank": int(self.use_pattern_prompt_bank),
            "use_receiver_only_prompt": int(self.use_receiver_only_prompt),
            "use_benefit_gate": int(self.use_benefit_gate),
            "num_class_prompt_slots": int(self.num_class_prompt_slots),
            "residual_prompt_count": int(self.residual_prompt_count if self.use_class_aware_routing else self.num_prompt_nodes),
            "class_key_proto_init_coverage": float(self.class_key_init_coverage),
            "class_key_proto_init_missing_classes": list(self.class_key_init_missing_classes),
            "pattern_key_init_coverage": float(self.pattern_key_init_coverage),
            "pattern_key_init_selected_nodes": list(self.pattern_key_init_selected_nodes),
        }
        return {
            "adapted_x": adapted_x,
            "adapted_edge_index": adapted_edge_index,
            "adapted_edge_weight": adapted_edge_weight,
            "adapted_edge_type": adapted_edge_type,
            "prompt_node_x": prompt_node_x,
            "pool_mask": pool_mask,
            "prompt_edge_count": int(prompt_edges.size(1)),
            "edge_scale": effective_edge_scale,
            "aux": aux,
        }


def prompt_edge_l1_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    weights = prompt_out.get("aux", {}).get("prompt_edge_weight")
    if isinstance(weights, torch.Tensor) and weights.numel() > 0:
        return weights.mean()
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_balance_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    aux = prompt_out.get("aux", {})
    usage = aux.get("prompt_usage_full", aux.get("prompt_usage"))
    if isinstance(usage, torch.Tensor) and usage.numel() > 0:
        target = usage.new_full(usage.shape, 1.0 / float(usage.numel()))
        return (usage - target).pow(2).sum()
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_role_diversity_loss(module: PromptGraphModuleP1 | None) -> torch.Tensor:
    """Penalize highly similar prompt keys to encourage role separation."""

    if module is None:
        return torch.tensor(0.0)
    if bool(getattr(module, "use_multiview_routing", False)):
        keys = torch.cat(
            [
                module.semantic_prompt_keys,
                module.structural_prompt_keys,
                module.role_prompt_keys,
            ],
            dim=0,
        )
    else:
        keys = module.prompt_keys
    if keys.size(0) < 2:
        return keys.new_tensor(0.0)
    norm_keys = F.normalize(keys, dim=-1, eps=1e-12)
    sim = norm_keys @ norm_keys.t()
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    off_diag = sim[~eye]
    return off_diag.pow(2).mean()


def prompt_acceptance_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    """Small regularizer that discourages accepting every pool node by default."""

    aux = prompt_out.get("aux", {})
    gates = aux.get("pool_acceptance_gate")
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(gates, torch.Tensor) and gates.numel() > 0:
        return gates.mean()
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_acceptance_budget_loss(
    prompt_out: dict[str, Any],
    *,
    min_acceptance: float | None = None,
    max_acceptance: float | None = None,
) -> torch.Tensor:
    """Soft safety budget over the mean acceptance gate.

    This is a null-route style constraint for prompt graphs. It does not decide
    which nodes are correct; it only prevents the acceptance gate from drifting
    into always-off or always-on behavior when the available supervision is weak.
    """

    aux = prompt_out.get("aux", {})
    gates = aux.get("pool_acceptance_gate")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(gates, torch.Tensor) and gates.numel() > 0):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    mean_gate = gates.mean()
    losses: list[torch.Tensor] = []
    if min_acceptance is not None:
        losses.append(F.relu(float(min_acceptance) - mean_gate).pow(2))
    if max_acceptance is not None:
        losses.append(F.relu(mean_gate - float(max_acceptance)).pow(2))
    if not losses:
        return mean_gate.new_tensor(0.0)
    return torch.stack(losses).sum()


def prompt_usage_consistency_loss(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    margin: float = 0.25,
    negative_weight: float = 0.05,
) -> torch.Tensor:
    """Train-only class consistency over prompt usage distributions.

    This does not assign a class to any prompt node. It only nudges labeled
    train-pool nodes from the same class to use similar prompt distributions.
    Validation/test labels are ignored because the mask is always restricted
    to ``train_mask``.
    """

    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != full_prob.size(0):
        raise ValueError("routing_full_prob rows must match pool_mask true count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) < 2:
        return full_prob.new_tensor(0.0)
    probs = full_prob[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx][train_pool]
    dist = torch.cdist(probs, probs, p=2).pow(2)
    same = y[:, None] == y[None, :]
    eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    same = same & ~eye
    diff = ~same & ~eye
    losses: list[torch.Tensor] = []
    if bool(same.any()):
        losses.append(dist[same].mean())
    if negative_weight > 0 and bool(diff.any()):
        diff_dist = torch.sqrt(dist[diff].clamp_min(1e-12))
        losses.append(float(negative_weight) * F.relu(float(margin) - diff_dist).pow(2).mean())
    if not losses:
        return full_prob.new_tensor(0.0)
    return torch.stack(losses).sum()


def _train_pool_rows(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        return None
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != full_prob.size(0):
        raise ValueError("routing rows must match pool index count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return None
    y = labels.to(device=pool_mask.device)[pool_idx][train_pool]
    return train_pool, y, edge_scale if isinstance(edge_scale, torch.Tensor) else full_prob.new_tensor(0.0)


def prompt_class_route_loss(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    """Train-only class-slot routing loss for labeled pool nodes.

    Only labeled training nodes inside the prompt pool are supervised. The first
    ``num_class_prompt_slots`` prompt slots are treated as routing anchors; all
    remaining prompt slots stay residual and are never class targets.
    """

    aux = prompt_out.get("aux", {})
    logits = aux.get("routing_logits")
    class_slots = int(aux.get("num_class_prompt_slots", 0))
    if not (isinstance(logits, torch.Tensor) and class_slots > 0):
        edge_scale = prompt_out.get("edge_scale")
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    rows = _train_pool_rows(prompt_out, labels, train_mask)
    if rows is None:
        return logits.new_tensor(0.0)
    train_pool, y, _ = rows
    valid = (y >= 0) & (y < class_slots)
    if int(valid.sum().item()) == 0:
        return logits.new_tensor(0.0)
    class_logits = logits[train_pool][:, :class_slots][valid]
    targets = y[valid].long()
    return F.cross_entropy(class_logits, targets)


def prompt_key_proto_loss(
    module: PromptGraphModuleP1 | None,
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    """Align class prompt keys to train-only structural query prototypes."""

    if module is None:
        return torch.tensor(0.0)
    class_slots = int(getattr(module, "num_class_prompt_slots", 0))
    if class_slots <= 0:
        return module.prompt_keys.new_tensor(0.0)
    aux = prompt_out.get("aux", {})
    query = aux.get("structural_query", aux.get("query"))
    if not isinstance(query, torch.Tensor):
        return module.prompt_keys.new_tensor(0.0)
    rows = _train_pool_rows(prompt_out, labels, train_mask)
    if rows is None:
        return query.new_tensor(0.0)
    train_pool, y, _ = rows
    train_query = query[train_pool]
    if train_query.numel() == 0:
        return query.new_tensor(0.0)
    if bool(getattr(module, "use_multiview_routing", False)):
        class_keys = module.structural_prompt_keys[:class_slots]
    else:
        class_keys = module.prompt_keys[:class_slots]
    losses: list[torch.Tensor] = []
    for class_id in range(class_slots):
        mask = y == class_id
        if not bool(mask.any()):
            continue
        proto = train_query[mask].detach().mean(dim=0)
        key = class_keys[class_id].to(dtype=query.dtype, device=query.device)
        losses.append(1.0 - F.cosine_similarity(key.unsqueeze(0), proto.unsqueeze(0), dim=-1, eps=1e-12).squeeze(0))
    if not losses:
        return query.new_tensor(0.0)
    return torch.stack(losses).mean()


def prompt_view_entropy_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    """Optional small loss encouraging node-wise view gates to make choices."""

    aux = prompt_out.get("aux", {})
    entropy = aux.get("view_gate_entropy")
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(entropy, torch.Tensor):
        return entropy
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_view_prior_loss(prompt_out: dict[str, Any], prior: list[float] | tuple[float, ...]) -> torch.Tensor:
    """Match the average view gate to a configured soft prior."""

    aux = prompt_out.get("aux", {})
    view_gate = aux.get("view_gate")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(view_gate, torch.Tensor) and view_gate.numel() > 0):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    target = view_gate.new_tensor(list(prior), dtype=view_gate.dtype)
    if target.numel() != view_gate.size(1):
        raise ValueError("view_prior length must match the number of routing views")
    target = target.clamp_min(0.0)
    target = target / target.sum().clamp_min(1e-12)
    return F.mse_loss(view_gate.mean(dim=0), target)


def utility_receive_gate_budget_loss(
    prompt_out: dict[str, Any],
    *,
    min_receive: float | None = None,
    max_receive: float | None = None,
) -> torch.Tensor:
    """Soft budget over the node-level effective receive gate."""

    aux = prompt_out.get("aux", {})
    gates = aux.get("effective_receive_gate", aux.get("receive_gate_score"))
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(gates, torch.Tensor) and gates.numel() > 0):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    mean_gate = gates.mean()
    losses: list[torch.Tensor] = []
    if min_receive is not None:
        losses.append(F.relu(float(min_receive) - mean_gate).pow(2))
    if max_receive is not None:
        losses.append(F.relu(mean_gate - float(max_receive)).pow(2))
    if not losses:
        return mean_gate.new_tensor(0.0)
    return torch.stack(losses).sum()
