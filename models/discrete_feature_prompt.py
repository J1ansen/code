"""Selective discrete feature prompt graph construction for P23."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary, mean_neighbor_variance


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


class SelectiveDiscreteFeaturePromptGraph(nn.Module):
    """Build GRAPHITE-style feature prompt nodes for selected hard nodes.

    The module returns a prompt-graph output dictionary compatible with the
    runner's ``adapted_x/adapted_edge_index`` path. The frozen branch keeps the
    original graph; only the adapted branch sees the added feature prompt nodes.
    """

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        self.tokenizer = str(self.config.get("tokenizer", "binary_nonzero"))
        if self.tokenizer not in {"binary_nonzero", "topk_activation"}:
            raise ValueError("P23 tokenizer must be 'binary_nonzero' or 'topk_activation' in the first implementation")
        self.binary_threshold = float(self.config.get("binary_threshold", 0.0))
        self.topk_feature_dims = int(self.config.get("topk_feature_dims", 32))
        self.min_df = int(self.config.get("min_df", 2))
        self.max_df_ratio = float(self.config.get("max_df_ratio", 0.20))
        self.rho = float(self.config.get("rho", 0.15))
        self.include_train_in_pool = bool(self.config.get("include_train_in_pool", True))
        self.topk_per_node = int(self.config.get("topk_feature_prompt_per_node", 3))
        self.direction = str(self.config.get("direction", "prompt_to_node"))
        if self.direction not in {"prompt_to_node", "bidirectional"}:
            raise ValueError("P23 direction must be 'prompt_to_node' or 'bidirectional'")
        self.feature_edge_weight = float(self.config.get("feature_edge_weight", self.config.get("message_scale", 0.10)))
        self.difficulty_edge_weight = float(self.config.get("difficulty_edge_weight", 0.50))
        self.structural_weight = float(self.config.get("utility_pool_structural_weight", 0.40))
        self.uncertainty_weight = float(self.config.get("utility_pool_uncertainty_weight", 0.30))
        self.disagreement_weight = float(self.config.get("utility_pool_disagreement_weight", 0.30))
        self.use_idf = bool(self.config.get("use_idf", True))
        self.use_feature_reliability = bool(self.config.get("use_feature_reliability", True))
        self.detach_feature_prompt = str(self.config.get("feature_node_init", "avg_z_detached")) == "avg_z_detached"
        self.static_graph = bool(self.config.get("static_graph", True))
        self._cached_graph: dict[str, Any] | None = None
        self._cache_signature: tuple[int, int, int, str] | None = None

    def clear_cache(self) -> None:
        self._cached_graph = None
        self._cache_signature = None

    def _token_membership(self, z: torch.Tensor) -> torch.Tensor:
        if self.tokenizer == "binary_nonzero":
            membership = z.abs() > self.binary_threshold
            if self.topk_feature_dims > 0 and self.topk_feature_dims < z.size(1):
                scores = z.abs().masked_fill(~membership, -torch.inf)
                topk = torch.topk(scores, k=self.topk_feature_dims, dim=-1).indices
                limited = torch.zeros_like(membership)
                limited.scatter_(1, topk, True)
                membership = membership & limited
            return membership
        k = min(max(1, self.topk_feature_dims), int(z.size(1)))
        topk = torch.topk(z.abs(), k=k, dim=-1).indices
        membership = torch.zeros(z.shape, dtype=torch.bool, device=z.device)
        membership.scatter_(1, topk, True)
        return membership

    def _pool_score(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        no_prompt_logits: torch.Tensor | None,
        h_adp_no_prompt: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        low = mean_neighbor_summary(z.detach(), edge_index, num_nodes=int(z.size(0)))
        two = mean_neighbor_summary(low, edge_index, num_nodes=int(z.size(0)))
        variance = mean_neighbor_variance(z.detach(), edge_index, num_nodes=int(z.size(0))).mean(dim=-1)
        structural = (
            1.0
            - F.cosine_similarity(z.detach(), low, dim=-1, eps=1e-12)
            + 1.0
            - F.cosine_similarity(low, two, dim=-1, eps=1e-12)
            + _minmax(variance)
        ) / 3.0
        structural = _minmax(structural)

        if no_prompt_logits is None:
            uncertainty = z.new_zeros(z.size(0))
        else:
            prob = F.softmax(no_prompt_logits.detach(), dim=-1)
            uncertainty = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
            if prob.size(-1) > 1:
                uncertainty = uncertainty / math.log(float(prob.size(-1)))
            uncertainty = _minmax(uncertainty)

        if h_adp_no_prompt is None:
            disagreement = z.new_zeros(z.size(0))
        else:
            disagreement = 1.0 - F.cosine_similarity(
                h_pre.detach(),
                h_adp_no_prompt.detach().to(device=h_pre.device, dtype=h_pre.dtype),
                dim=-1,
                eps=1e-12,
            )
            disagreement = _minmax(disagreement)

        score = (
            self.structural_weight * structural
            + self.uncertainty_weight * uncertainty
            + self.disagreement_weight * disagreement
        )
        return _minmax(score), {
            "p23_pool_structural_score": structural,
            "p23_pool_uncertainty_score": uncertainty,
            "p23_pool_disagreement_score": disagreement,
        }

    def _select_pool(self, score: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
        n = int(score.numel())
        pool = torch.zeros(n, dtype=torch.bool, device=score.device)
        if self.include_train_in_pool:
            pool |= train_mask.to(device=score.device, dtype=torch.bool)
        extra = int(round(max(0.0, min(1.0, self.rho)) * float(n)))
        if extra > 0:
            top = torch.topk(score, k=min(extra, n), largest=True).indices
            pool[top] = True
        return pool

    def _feature_stats(self, z: torch.Tensor, membership: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = membership.to(dtype=z.dtype).sum(dim=0)
        sums = membership.to(dtype=z.dtype).t() @ z
        feature_x = sums / counts.clamp_min(1.0).unsqueeze(-1)
        if self.detach_feature_prompt:
            feature_x = feature_x.detach()
        idf = torch.log((z.new_tensor(float(z.size(0))) + 1.0) / (counts + 1.0))
        rows, cols = torch.nonzero(membership, as_tuple=True)
        if rows.numel() > 0:
            z_norm = F.normalize(z.detach(), dim=-1, eps=1e-12)
            feature_norm = F.normalize(feature_x.detach(), dim=-1, eps=1e-12)
            cosine = (z_norm[rows] * feature_norm[cols]).sum(dim=-1)
            cohesion_sum = z.new_zeros(z.size(1))
            cohesion_sum.index_add_(0, cols, cosine.to(dtype=z.dtype))
            cohesion_mean = cohesion_sum / counts.clamp_min(1.0)
        else:
            cohesion_mean = z.new_zeros(z.size(1))
        reliability = idf if self.use_idf else z.new_zeros(idf.shape)
        if self.use_feature_reliability:
            reliability = reliability + cohesion_mean - torch.log1p(counts)
        reliability = _minmax(reliability)
        max_df = max(1.0, self.max_df_ratio * float(z.size(0)))
        valid = (counts >= float(self.min_df)) & (counts <= max_df)
        return feature_x, reliability.masked_fill(~valid, -torch.inf), counts

    def _signature(self, z: torch.Tensor, edge_index: torch.Tensor) -> tuple[int, int, int, str]:
        return (int(z.size(0)), int(z.size(1)), int(edge_index.size(1)), str(z.device))

    def _output_from_cache(
        self,
        *,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scale_multiplier: float | torch.Tensor,
        cache_hit: bool,
    ) -> dict[str, Any]:
        if self._cached_graph is None:
            raise RuntimeError("P23 cache is empty")
        cached = self._cached_graph
        scale = torch.as_tensor(edge_scale_multiplier, dtype=z.dtype, device=z.device)
        prompt_x = cached["prompt_x"].to(device=z.device, dtype=z.dtype)
        prompt_edges = cached["prompt_edges"].to(device=edge_index.device)
        prompt_edge_weight = cached["prompt_edge_weight"].to(device=z.device, dtype=z.dtype) * scale
        adapted_x = torch.cat([z, prompt_x], dim=0)
        base_weight = z.new_ones(edge_index.size(1))
        adapted_edge_index = torch.cat([edge_index, prompt_edges], dim=1)
        adapted_edge_weight = torch.cat([base_weight, prompt_edge_weight], dim=0)
        adapted_edge_type = torch.cat(
            [
                torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
                torch.ones(prompt_edges.size(1), dtype=torch.long, device=edge_index.device),
            ],
            dim=0,
        )
        aux = {
            key: (value.to(device=z.device, dtype=z.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
            for key, value in cached["aux"].items()
        }
        aux["prompt_edge_weight"] = prompt_edge_weight
        aux["p23_static_graph_enabled"] = z.new_tensor(float(self.static_graph))
        aux["p23_static_cache_hit"] = z.new_tensor(float(cache_hit))
        return {
            "adapted_x": adapted_x,
            "adapted_edge_index": adapted_edge_index,
            "adapted_edge_weight": adapted_edge_weight,
            "adapted_edge_type": adapted_edge_type,
            "prompt_node_x": prompt_x,
            "pool_mask": cached["pool_mask"].to(device=z.device),
            "prompt_edge_count": int(prompt_edges.size(1)),
            "edge_scale": scale,
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
        signature = self._signature(z, edge_index)
        if self.static_graph and self._cached_graph is not None and self._cache_signature == signature:
            return self._output_from_cache(
                z=z,
                edge_index=edge_index,
                edge_scale_multiplier=edge_scale_multiplier,
                cache_hit=True,
            )
        membership = self._token_membership(z)
        pool_score, score_parts = self._pool_score(
            z=z,
            h_pre=h_pre,
            edge_index=edge_index,
            no_prompt_logits=no_prompt_logits,
            h_adp_no_prompt=h_adp_no_prompt,
        )
        pool_mask = self._select_pool(pool_score, train_mask)
        feature_x_all, reliability, counts = self._feature_stats(z, membership)

        edge_src: list[torch.Tensor] = []
        edge_dst: list[torch.Tensor] = []
        edge_weight: list[torch.Tensor] = []
        selected_feature_ids: list[torch.Tensor] = []
        selected_feature_pos: dict[int, int] = {}
        scale = torch.as_tensor(edge_scale_multiplier, dtype=z.dtype, device=z.device)
        pool_nodes = torch.nonzero(pool_mask, as_tuple=False).view(-1)
        for node in pool_nodes.tolist():
            candidates = torch.nonzero(membership[node] & torch.isfinite(reliability), as_tuple=False).view(-1)
            if candidates.numel() == 0:
                continue
            k = min(self.topk_per_node, int(candidates.numel()))
            local_scores = reliability[candidates]
            top_local = torch.topk(local_scores, k=k, largest=True).indices
            chosen = candidates[top_local]
            weights = F.softmax(local_scores[top_local], dim=0)
            difficulty = 1.0 + self.difficulty_edge_weight * pool_score[node]
            for feature_id, weight in zip(chosen.tolist(), weights):
                if feature_id not in selected_feature_pos:
                    selected_feature_pos[feature_id] = len(selected_feature_ids)
                    selected_feature_ids.append(torch.tensor(feature_id, device=z.device, dtype=torch.long))
                prompt_idx = z.size(0) + selected_feature_pos[feature_id]
                edge_src.append(torch.tensor(prompt_idx, device=z.device, dtype=torch.long))
                edge_dst.append(torch.tensor(node, device=z.device, dtype=torch.long))
                edge_weight.append((self.feature_edge_weight * difficulty * weight).to(dtype=z.dtype))
                if self.direction == "bidirectional":
                    edge_src.append(torch.tensor(node, device=z.device, dtype=torch.long))
                    edge_dst.append(torch.tensor(prompt_idx, device=z.device, dtype=torch.long))
                    edge_weight.append((self.feature_edge_weight * difficulty * weight).to(dtype=z.dtype))

        if selected_feature_ids:
            feature_ids = torch.stack(selected_feature_ids)
            prompt_x = feature_x_all[feature_ids]
            prompt_edges = torch.stack([torch.stack(edge_src), torch.stack(edge_dst)], dim=0)
            prompt_edge_weight_base = torch.stack(edge_weight).to(device=z.device, dtype=z.dtype)
            prompt_edge_weight = prompt_edge_weight_base * scale
        else:
            feature_ids = torch.empty(0, dtype=torch.long, device=z.device)
            prompt_x = z.new_zeros((0, z.size(1)))
            prompt_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            prompt_edge_weight_base = z.new_zeros(0)
            prompt_edge_weight = z.new_zeros(0)

        adapted_x = torch.cat([z, prompt_x], dim=0)
        base_weight = z.new_ones(edge_index.size(1))
        adapted_edge_index = torch.cat([edge_index, prompt_edges], dim=1)
        adapted_edge_weight = torch.cat([base_weight, prompt_edge_weight], dim=0)
        adapted_edge_type = torch.cat(
            [
                torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
                torch.ones(prompt_edges.size(1), dtype=torch.long, device=edge_index.device),
            ],
            dim=0,
        )
        valid_feature_count = torch.isfinite(reliability).to(dtype=z.dtype).sum()
        usage = torch.zeros(feature_ids.numel(), dtype=z.dtype, device=z.device)
        if prompt_edges.numel() > 0 and feature_ids.numel() > 0:
            prompt_src = prompt_edges[0] - z.size(0)
            usage.index_add_(0, prompt_src, torch.ones_like(prompt_src, dtype=z.dtype))
            usage_prob = usage / usage.sum().clamp_min(1.0)
            usage_entropy = -(usage_prob * usage_prob.clamp_min(1e-12).log()).sum()
            if usage_prob.numel() > 1:
                usage_entropy = usage_entropy / math.log(float(usage_prob.numel()))
        else:
            usage_entropy = z.new_tensor(0.0)
        aux = {
                "prompt_edge_weight": prompt_edge_weight,
                "prompt_usage": usage,
                "prompt_usage_entropy": usage_entropy,
                "connected_edge_count": int(prompt_edges.size(1)),
                "edge_type_counts": [int(edge_index.size(1)), int(prompt_edges.size(1)), 0],
                "p23_pool_score": pool_score.detach(),
                "p23_pool_ratio": pool_mask.to(dtype=z.dtype).mean(),
                "p23_pool_score_mean": pool_score.mean(),
                "p23_feature_prompt_count": z.new_tensor(float(feature_ids.numel())),
                "p23_valid_feature_count": valid_feature_count,
                "p23_prompt_edge_count": z.new_tensor(float(prompt_edges.size(1))),
                "p23_avg_prompt_degree": usage.mean() if usage.numel() > 0 else z.new_tensor(0.0),
                "p23_feature_df_mean": counts[torch.isfinite(reliability)].mean() if bool(torch.isfinite(reliability).any()) else z.new_tensor(0.0),
                "p23_feature_reliability_mean": reliability[torch.isfinite(reliability)].mean() if bool(torch.isfinite(reliability).any()) else z.new_tensor(0.0),
                "p23_static_graph_enabled": z.new_tensor(float(self.static_graph)),
                "p23_static_cache_hit": z.new_tensor(0.0),
                **{key: value.detach() for key, value in score_parts.items()},
        }
        if self.static_graph:
            self._cached_graph = {
                "prompt_x": prompt_x.detach(),
                "prompt_edges": prompt_edges.detach(),
                "prompt_edge_weight": prompt_edge_weight_base.detach(),
                "pool_mask": pool_mask.detach(),
                "aux": {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in aux.items()
                    if key != "prompt_edge_weight"
                },
            }
            self._cache_signature = signature
        return {
            "adapted_x": adapted_x,
            "adapted_edge_index": adapted_edge_index,
            "adapted_edge_weight": adapted_edge_weight,
            "adapted_edge_type": adapted_edge_type,
            "prompt_node_x": prompt_x,
            "pool_mask": pool_mask,
            "prompt_edge_count": int(prompt_edges.size(1)),
            "edge_scale": scale,
            "aux": aux,
        }


class GraphiteStylePromptGraphAdapter(nn.Module):
    """GRAPHITE-style feature-node adapter for GP2F's adapted branch.

    The frozen GP2F branch still reads the original graph. The adapted branch
    receives an expanded graph with raw-discrete feature nodes and bidirectional
    node-feature edges, so the adapter GNN performs message passing through the
    prompt nodes instead of receiving a post-hoc residual patch.
    """

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        tokenizer_cfg = dict(self.config.get("feature_tokenizer", {}))
        self.tokenizer = str(tokenizer_cfg.get("mode", self.config.get("tokenizer", "binary_nonzero")))
        if self.tokenizer not in {"binary_nonzero", "topk_activation"}:
            raise ValueError("Graphite-style prompt graph supports binary_nonzero and topk_activation tokenizers")
        self.binary_threshold = float(tokenizer_cfg.get("binary_threshold", self.config.get("binary_threshold", 0.0)))
        self.binary_topk = int(tokenizer_cfg.get("binary_topk", tokenizer_cfg.get("max_nonzero_per_node", 0)))
        self.topk = int(tokenizer_cfg.get("topk", self.config.get("topk_feature_dims", 0)))
        filter_cfg = dict(self.config.get("feature_filter", {}))
        self.min_df = float(filter_cfg.get("min_df_global", filter_cfg.get("min_df_pool", self.config.get("min_df", 2))))
        self.max_df_ratio = float(
            filter_cfg.get("max_df_global_ratio", filter_cfg.get("max_df_ratio", self.config.get("max_df_ratio", 1.0)))
        )
        self.feature_edge_weight = float(self.config.get("feature_edge_weight", self.config.get("graphite_feature_edge_weight", 1.0)))
        self.original_edge_weight = float(self.config.get("original_edge_weight", 1.0))
        self.static_graph = bool(self.config.get("static_graph", True))
        self.detach_feature_prompt = bool(self.config.get("detach_feature_prompt", True))
        self.use_feature_filter = bool(self.config.get("use_feature_filter", True))
        self.learn_feature_edge_weight = bool(self.config.get("learn_feature_edge_weight", False))
        if self.learn_feature_edge_weight:
            init = max(self.feature_edge_weight, 1e-6)
            self.feature_edge_log_scale = nn.Parameter(torch.log(torch.tensor(init, dtype=torch.float32)))
        else:
            self.register_parameter("feature_edge_log_scale", None)
        self._cached_graph: dict[str, Any] | None = None
        self._cache_signature: tuple[int, int, int, int, str] | None = None

    def clear_cache(self) -> None:
        self._cached_graph = None
        self._cache_signature = None

    def _token_membership(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.tokenizer == "binary_nonzero":
            membership = x_raw.abs() > self.binary_threshold
            if self.binary_topk > 0 and self.binary_topk < int(x_raw.size(1)):
                capped = torch.zeros_like(membership)
                scores = x_raw.abs().masked_fill(~membership, float("-inf"))
                topk = torch.topk(scores, k=self.binary_topk, dim=-1).indices
                capped.scatter_(1, topk, True)
                membership = capped & membership
            return membership
        k = self.topk if self.topk > 0 else int(x_raw.size(1))
        k = min(max(1, k), int(x_raw.size(1)))
        topk = torch.topk(x_raw.abs(), k=k, dim=-1).indices
        membership = torch.zeros(x_raw.shape, dtype=torch.bool, device=x_raw.device)
        membership.scatter_(1, topk, True)
        return membership

    def _signature(self, z: torch.Tensor, x_raw: torch.Tensor, edge_index: torch.Tensor) -> tuple[int, int, int, int, str]:
        return (int(z.size(0)), int(z.size(1)), int(x_raw.size(1)), int(edge_index.size(1)), str(z.device))

    def _feature_edge_scale(self, ref: torch.Tensor, edge_scale_multiplier: float | torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(edge_scale_multiplier, dtype=ref.dtype, device=ref.device)
        if self.feature_edge_log_scale is None:
            weight = ref.new_tensor(float(self.feature_edge_weight))
        else:
            weight = self.feature_edge_log_scale.to(device=ref.device, dtype=ref.dtype).exp()
        return weight * scale

    def _output_from_cache(
        self,
        *,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scale_multiplier: float | torch.Tensor,
        cache_hit: bool,
    ) -> dict[str, Any]:
        if self._cached_graph is None:
            raise RuntimeError("Graphite-style prompt graph cache is empty")
        cached = self._cached_graph
        prompt_x = cached["prompt_x"].to(device=z.device, dtype=z.dtype)
        prompt_edges = cached["prompt_edges"].to(device=edge_index.device)
        feature_weight = self._feature_edge_scale(z, edge_scale_multiplier)
        prompt_edge_weight = torch.ones(prompt_edges.size(1), dtype=z.dtype, device=z.device) * feature_weight
        adapted_x = torch.cat([z, prompt_x], dim=0)
        base_weight = torch.ones(edge_index.size(1), dtype=z.dtype, device=z.device) * float(self.original_edge_weight)
        adapted_edge_index = torch.cat([edge_index, prompt_edges], dim=1)
        adapted_edge_weight = torch.cat([base_weight, prompt_edge_weight], dim=0)
        adapted_edge_type = torch.cat(
            [
                torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
                torch.ones(prompt_edges.size(1), dtype=torch.long, device=edge_index.device),
            ],
            dim=0,
        )
        aux = {
            key: (value.to(device=z.device, dtype=z.dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
            for key, value in cached["aux"].items()
        }
        aux["prompt_edge_weight"] = prompt_edge_weight
        aux["p23_graphite_static_cache_hit"] = z.new_tensor(float(cache_hit))
        aux["p23_graphite_feature_edge_weight"] = feature_weight.detach()
        return {
            "adapted_x": adapted_x,
            "adapted_edge_index": adapted_edge_index,
            "adapted_edge_weight": adapted_edge_weight,
            "adapted_edge_type": adapted_edge_type,
            "prompt_node_x": prompt_x,
            "pool_mask": cached["pool_mask"].to(device=z.device),
            "prompt_edge_count": int(prompt_edges.size(1)),
            "edge_scale": torch.as_tensor(edge_scale_multiplier, dtype=z.dtype, device=z.device),
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
        x_raw: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del h_pre, train_mask, no_prompt_logits, h_adp_no_prompt
        if x_raw is None:
            x_raw = z
        x_raw = x_raw.to(device=z.device)
        signature = self._signature(z, x_raw, edge_index)
        if self.static_graph and self._cached_graph is not None and self._cache_signature == signature:
            return self._output_from_cache(
                z=z,
                edge_index=edge_index,
                edge_scale_multiplier=edge_scale_multiplier,
                cache_hit=True,
            )

        membership = self._token_membership(x_raw)
        counts = membership.to(dtype=z.dtype).sum(dim=0)
        if self.use_feature_filter:
            max_df = max(1.0, float(self.max_df_ratio) * float(z.size(0)))
            valid = (counts >= float(self.min_df)) & (counts <= max_df)
        else:
            valid = counts > 0
        feature_ids = torch.where(valid)[0]
        selected_membership = membership[:, feature_ids] if feature_ids.numel() > 0 else membership[:, :0]
        if feature_ids.numel() > 0:
            sums = selected_membership.to(dtype=z.dtype).t() @ z
            denom = selected_membership.to(dtype=z.dtype).sum(dim=0).clamp_min(1.0)
            prompt_x = sums / denom.unsqueeze(-1)
            if self.detach_feature_prompt:
                prompt_x = prompt_x.detach()
            edge_node, edge_feature_local = torch.nonzero(selected_membership, as_tuple=True)
            if edge_node.numel() > 0:
                prompt_node = edge_feature_local + int(z.size(0))
                node_to_prompt = torch.stack([edge_node, prompt_node], dim=0)
                prompt_to_node = torch.stack([prompt_node, edge_node], dim=0)
                prompt_edges = torch.cat([node_to_prompt, prompt_to_node], dim=1).to(device=edge_index.device)
            else:
                prompt_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        else:
            prompt_x = z.new_zeros((0, z.size(1)))
            edge_node = torch.empty(0, dtype=torch.long, device=z.device)
            edge_feature_local = torch.empty(0, dtype=torch.long, device=z.device)
            prompt_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        usage = torch.zeros(feature_ids.numel(), dtype=z.dtype, device=z.device)
        if edge_feature_local.numel() > 0:
            usage.index_add_(0, edge_feature_local, torch.ones_like(edge_feature_local, dtype=z.dtype))
            usage_prob = usage / usage.sum().clamp_min(1.0)
            usage_entropy = -(usage_prob * usage_prob.clamp_min(1e-12).log()).sum()
            if usage_prob.numel() > 1:
                usage_entropy = usage_entropy / math.log(float(usage_prob.numel()))
        else:
            usage_entropy = z.new_tensor(0.0)

        pool_mask = torch.ones(z.size(0), dtype=torch.bool, device=z.device)
        aux = {
            "prompt_edge_weight": torch.ones(prompt_edges.size(1), dtype=z.dtype, device=z.device)
            * self._feature_edge_scale(z, edge_scale_multiplier),
            "prompt_usage": usage,
            "prompt_usage_entropy": usage_entropy,
            "connected_edge_count": int(prompt_edges.size(1)),
            "edge_type_counts": [int(edge_index.size(1)), int(prompt_edges.size(1)), 0],
            "p23_pool_ratio": z.new_tensor(1.0),
            "p23_graphite_enabled": z.new_tensor(1.0),
            "p23_graphite_feature_prompt_count": z.new_tensor(float(feature_ids.numel())),
            "p23_graphite_feature_edge_count": z.new_tensor(float(prompt_edges.size(1))),
            "p23_graphite_raw_feature_count": z.new_tensor(float(x_raw.size(1))),
            "p23_graphite_valid_feature_ratio": valid.to(dtype=z.dtype).mean() if valid.numel() > 0 else z.new_tensor(0.0),
            "p23_graphite_avg_feature_degree": usage.mean() if usage.numel() > 0 else z.new_tensor(0.0),
            "p23_graphite_max_feature_degree": usage.max() if usage.numel() > 0 else z.new_tensor(0.0),
            "p23_graphite_static_graph_enabled": z.new_tensor(float(self.static_graph)),
            "p23_graphite_static_cache_hit": z.new_tensor(0.0),
            "p23_feature_prompt_count": z.new_tensor(float(feature_ids.numel())),
            "p23_prompt_edge_count": z.new_tensor(float(prompt_edges.size(1))),
        }
        self._cached_graph = {
            "prompt_x": prompt_x.detach(),
            "prompt_edges": prompt_edges.detach(),
            "pool_mask": pool_mask.detach(),
            "aux": {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in aux.items()
                if key != "prompt_edge_weight"
            },
        }
        self._cache_signature = signature if self.static_graph else None
        return self._output_from_cache(
            z=z,
            edge_index=edge_index,
            edge_scale_multiplier=edge_scale_multiplier,
            cache_hit=False,
        )
