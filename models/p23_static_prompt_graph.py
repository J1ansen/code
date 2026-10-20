"""Run-level static prompt graph builder for P23 v0.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary, mean_neighbor_variance


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


@dataclass
class P23PromptGraphState:
    pool_mask: torch.Tensor
    heterophily_risk: torch.Tensor
    graph_risk: torch.Tensor
    pool_ratio: torch.Tensor
    feature_ids: torch.Tensor
    prompt_x: torch.Tensor
    prompt_edge_index: torch.Tensor
    prompt_edge_type: torch.Tensor
    prompt_edge_prior: torch.Tensor
    feature_static_stats: dict[str, torch.Tensor]
    node_uncertainty: torch.Tensor
    branch_disagreement: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "P23PromptGraphState":
        tensor_dtype = dtype

        def move(value: torch.Tensor) -> torch.Tensor:
            if value.is_floating_point() and tensor_dtype is not None:
                return value.to(device=device, dtype=tensor_dtype)
            return value.to(device=device)

        return P23PromptGraphState(
            pool_mask=self.pool_mask.to(device=device),
            heterophily_risk=move(self.heterophily_risk),
            graph_risk=move(self.graph_risk),
            pool_ratio=move(self.pool_ratio),
            feature_ids=self.feature_ids.to(device=device),
            prompt_x=move(self.prompt_x),
            prompt_edge_index=self.prompt_edge_index.to(device=device),
            prompt_edge_type=self.prompt_edge_type.to(device=device),
            prompt_edge_prior=move(self.prompt_edge_prior),
            feature_static_stats={key: move(value) for key, value in self.feature_static_stats.items()},
            node_uncertainty=move(self.node_uncertainty),
            branch_disagreement=move(self.branch_disagreement),
        )


class RawFeatureTokenizer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.mode = str(cfg.get("mode", "binary_nonzero"))
        if self.mode not in {"binary_nonzero", "topk_activation"}:
            raise ValueError("P23 v0.1 supports raw feature tokenizer modes: binary_nonzero, topk_activation")
        self.topk = int(cfg.get("topk", 8))
        self.binary_threshold = float(cfg.get("binary_threshold", 0.0))
        self.binary_topk = int(cfg.get("binary_topk", cfg.get("max_nonzero_per_node", 0)))

    def __call__(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary_nonzero":
            membership = x_raw.abs() > self.binary_threshold
            if self.binary_topk > 0 and self.binary_topk < int(x_raw.size(1)):
                capped = torch.zeros_like(membership)
                scores = x_raw.abs().masked_fill(~membership, float("-inf"))
                k = min(self.binary_topk, int(x_raw.size(1)))
                topk = torch.topk(scores, k=k, dim=-1).indices
                capped.scatter_(1, topk, True)
                membership = capped & membership
            return membership
        k = min(max(1, self.topk), int(x_raw.size(1)))
        topk = torch.topk(x_raw.abs(), k=k, dim=-1).indices
        membership = torch.zeros(x_raw.shape, dtype=torch.bool, device=x_raw.device)
        membership.scatter_(1, topk, True)
        return membership


class P23StaticPromptGraphBuilder:
    """Build a static, run-level P23 prompt graph from raw features."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.tokenizer = RawFeatureTokenizer(self.config.get("feature_tokenizer", {}))

    def _risk_components(
        self,
        *,
        z_snapshot: torch.Tensor,
        edge_index: torch.Tensor,
        no_prompt_logits: torch.Tensor | None,
        h_pre_snapshot: torch.Tensor | None,
        h_adp0_snapshot: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        z = z_snapshot.detach()
        low = mean_neighbor_summary(z, edge_index, num_nodes=int(z.size(0)))
        two = mean_neighbor_summary(low, edge_index, num_nodes=int(z.size(0)))
        ego_neighbor = _minmax((1.0 - F.cosine_similarity(z, low, dim=-1, eps=1e-12)).clamp_min(0.0))
        onehop_twohop = _minmax((1.0 - F.cosine_similarity(low, two, dim=-1, eps=1e-12)).clamp_min(0.0))
        neighbor_variance = _minmax(mean_neighbor_variance(z, edge_index, num_nodes=int(z.size(0))).mean(dim=-1))

        if no_prompt_logits is None:
            uncertainty = z.new_zeros(z.size(0))
        else:
            prob = F.softmax(no_prompt_logits.detach().to(device=z.device, dtype=z.dtype), dim=-1)
            uncertainty = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
            if prob.size(-1) > 1:
                uncertainty = uncertainty / math.log(float(prob.size(-1)))
            uncertainty = _minmax(uncertainty)

        if h_pre_snapshot is None or h_adp0_snapshot is None:
            branch = z.new_zeros(z.size(0))
        else:
            branch = 1.0 - F.cosine_similarity(
                h_pre_snapshot.detach().to(device=z.device, dtype=z.dtype),
                h_adp0_snapshot.detach().to(device=z.device, dtype=z.dtype),
                dim=-1,
                eps=1e-12,
            )
            branch = _minmax(branch.clamp_min(0.0))

        return {
            "ego_neighbor": ego_neighbor,
            "onehop_twohop": onehop_twohop,
            "neighbor_variance": neighbor_variance,
            "uncertainty": uncertainty,
            "branch_disagreement": branch,
        }

    def _weighted_sum(self, components: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        if len(weights) != len(components):
            raise ValueError("P23 risk weight count must match component count")
        out = components[0].new_zeros(components[0].shape)
        total = max(sum(float(weight) for weight in weights), 1e-12)
        for component, weight in zip(components, weights):
            out = out + (float(weight) / total) * component
        return out

    def build(
        self,
        *,
        x_raw: torch.Tensor,
        z_snapshot: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        no_prompt_logits: torch.Tensor | None = None,
        h_pre_snapshot: torch.Tensor | None = None,
        h_adp0_snapshot: torch.Tensor | None = None,
    ) -> P23PromptGraphState:
        device = z_snapshot.device
        dtype = z_snapshot.dtype
        x_raw = x_raw.to(device=device)
        train = train_mask.to(device=device, dtype=torch.bool)
        membership = self.tokenizer(x_raw)
        components = self._risk_components(
            z_snapshot=z_snapshot,
            edge_index=edge_index,
            no_prompt_logits=no_prompt_logits,
            h_pre_snapshot=h_pre_snapshot,
            h_adp0_snapshot=h_adp0_snapshot,
        )
        graph_weights = self.config.get("graph_risk_weights", {})
        graph_risk_score = self._weighted_sum(
            [
                components["ego_neighbor"],
                components["onehop_twohop"],
                components["neighbor_variance"],
            ],
            [
                float(graph_weights.get("ego_neighbor", 0.40)),
                float(graph_weights.get("onehop_twohop", 0.30)),
                float(graph_weights.get("neighbor_variance", 0.30)),
            ],
        ).mean().clamp(0.0, 1.0)
        risk_cfg = self.config.get("risk", {})
        lambda_hetero = [float(item) for item in risk_cfg.get("lambda_hetero", [0.25, 0.20, 0.20, 0.15, 0.20])]
        lambda_homo = [float(item) for item in risk_cfg.get("lambda_homo", lambda_hetero)]
        if len(lambda_homo) != len(lambda_hetero):
            raise ValueError("P23 risk.lambda_homo and risk.lambda_hetero must have the same length")
        gamma = float(graph_risk_score.detach().item())
        risk_weights = [(1.0 - gamma) * h + gamma * g for h, g in zip(lambda_homo, lambda_hetero)]
        heterophily_risk = _minmax(
            self._weighted_sum(
                [
                    components["ego_neighbor"],
                    components["onehop_twohop"],
                    components["neighbor_variance"],
                    components["uncertainty"],
                    components["branch_disagreement"],
                ],
                risk_weights,
            )
        )

        pool_cfg = self.config.get("pool", {})
        rho_min = float(pool_cfg.get("rho_min", 0.05))
        rho_max = float(pool_cfg.get("rho_max", 0.40))
        rho_power = float(pool_cfg.get("rho_power", 1.0))
        adaptive = bool(pool_cfg.get("adaptive_ratio", True))
        pool_ratio = (
            rho_min + (rho_max - rho_min) * graph_risk_score.pow(rho_power)
            if adaptive
            else z_snapshot.new_tensor(float(pool_cfg.get("rho", rho_max)))
        ).clamp(0.0, 1.0)

        num_nodes = int(z_snapshot.size(0))
        pool_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        non_train = torch.where(~train)[0]
        topk_count = min(int(math.ceil(float(pool_ratio.item()) * float(non_train.numel()))), int(non_train.numel()))
        if topk_count > 0:
            selected = non_train[torch.topk(heterophily_risk[non_train], k=topk_count, largest=True).indices]
            pool_mask[selected] = True
        if bool(pool_cfg.get("force_train_nodes", True)):
            pool_mask[train] = True

        filter_cfg = self.config.get("feature_filter", {})
        pool_membership = membership[pool_mask]
        df_pool = pool_membership.to(dtype=dtype).sum(dim=0) if pool_membership.numel() > 0 else z_snapshot.new_zeros(x_raw.size(1))
        df_global = membership.to(dtype=dtype).sum(dim=0)
        min_df_pool = float(filter_cfg.get("min_df_pool", 2))
        max_df_pool = max(1.0, float(filter_cfg.get("max_df_pool_ratio", 0.50)) * max(1, int(pool_mask.sum().item())))
        max_df_global = max(1.0, float(filter_cfg.get("max_df_global_ratio", 0.80)) * float(num_nodes))
        candidate = (df_pool >= min_df_pool) & (df_pool <= max_df_pool) & (df_global <= max_df_global)
        feature_ids = torch.where(candidate)[0]

        if feature_ids.numel() > 0:
            init_scope = str(self.config.get("prompt_graph", {}).get("init_scope", "global_same_feature"))
            init_membership = membership[:, feature_ids]
            if init_scope == "pool_only":
                init_membership = init_membership & pool_mask.unsqueeze(-1)
            sums = init_membership.to(dtype=dtype).t() @ z_snapshot.detach()
            denom = init_membership.to(dtype=dtype).sum(dim=0).clamp_min(1.0)
            prompt_x = sums / denom.unsqueeze(-1)
        else:
            prompt_x = z_snapshot.new_zeros((0, z_snapshot.size(1)))

        df_pool_sel = df_pool[feature_ids] if feature_ids.numel() > 0 else z_snapshot.new_zeros(0)
        df_global_sel = df_global[feature_ids] if feature_ids.numel() > 0 else z_snapshot.new_zeros(0)
        idf = torch.log((z_snapshot.new_tensor(float(num_nodes)) + 1.0) / (df_global_sel + 1.0))
        pool_freq = df_pool_sel / max(1.0, float(pool_mask.sum().item()))
        pool_rel = df_pool_sel / (df_global_sel + 1.0)
        if feature_ids.numel() > 0:
            rows, local_cols = torch.nonzero(membership[:, feature_ids], as_tuple=True)
            z_norm = F.normalize(z_snapshot.detach(), dim=-1, eps=1e-12)
            prompt_norm = F.normalize(prompt_x.detach(), dim=-1, eps=1e-12)
            cosine = (z_norm[rows] * prompt_norm[local_cols]).sum(dim=-1)
            cohesion_sum = z_snapshot.new_zeros(feature_ids.numel())
            cohesion_sum.index_add_(0, local_cols, cosine.to(dtype=dtype))
            cohesion = cohesion_sum / df_global_sel.clamp_min(1.0)
        else:
            cohesion = z_snapshot.new_zeros(0)
        hub_score = df_global_sel / max(1.0, float(num_nodes))
        static_reliability = _minmax(idf + cohesion + pool_rel - hub_score) if feature_ids.numel() > 0 else z_snapshot.new_zeros(0)

        if feature_ids.numel() > 0:
            pool_feature_membership = membership[:, feature_ids] & pool_mask.unsqueeze(-1)
            edge_node, edge_feature_local = torch.nonzero(pool_feature_membership, as_tuple=True)
            topk_per_node = int(
                self.config.get("prompt_graph", {}).get(
                    "topk_feature_prompt_per_node",
                    self.config.get("topk_feature_prompt_per_node", 0),
                )
            )
            if topk_per_node > 0 and edge_node.numel() > 0:
                keep = torch.zeros(edge_node.size(0), dtype=torch.bool, device=device)
                edge_score = static_reliability[edge_feature_local] * heterophily_risk[edge_node].clamp_min(1e-6)
                for node in torch.unique(edge_node).tolist():
                    mask = edge_node == int(node)
                    local_idx = torch.where(mask)[0]
                    k = min(topk_per_node, int(local_idx.numel()))
                    chosen = local_idx[torch.topk(edge_score[local_idx], k=k, largest=True).indices]
                    keep[chosen] = True
                edge_node = edge_node[keep]
                edge_feature_local = edge_feature_local[keep]
            prompt_edge_index = torch.stack([edge_feature_local, edge_node], dim=0) if edge_node.numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
            prompt_edge_type = torch.full(
                (prompt_edge_index.size(1),),
                int(self.config.get("prompt_graph", {}).get("edge_type_prompt_to_node", 2)),
                dtype=torch.long,
                device=device,
            )
        else:
            prompt_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            prompt_edge_type = torch.empty(0, dtype=torch.long, device=device)
        if prompt_edge_index.numel() > 0:
            prompt_edge_prior = static_reliability[prompt_edge_index[0]].clamp_min(1e-6)
        else:
            prompt_edge_prior = z_snapshot.new_zeros(0)

        feature_stats = {
            "feature_ids": feature_ids.to(dtype=torch.float32),
            "df_pool": df_pool_sel,
            "df_global": df_global_sel,
            "idf": idf,
            "pool_rel": pool_rel,
            "pool_freq": pool_freq,
            "cohesion": cohesion,
            "hub_score": hub_score,
            "static_reliability": static_reliability,
        }
        return P23PromptGraphState(
            pool_mask=pool_mask,
            heterophily_risk=heterophily_risk,
            graph_risk=graph_risk_score.reshape(()),
            pool_ratio=pool_ratio.reshape(()),
            feature_ids=feature_ids,
            prompt_x=prompt_x.detach(),
            prompt_edge_index=prompt_edge_index,
            prompt_edge_type=prompt_edge_type,
            prompt_edge_prior=prompt_edge_prior.detach(),
            feature_static_stats=feature_stats,
            node_uncertainty=components["uncertainty"],
            branch_disagreement=components["branch_disagreement"],
        )
