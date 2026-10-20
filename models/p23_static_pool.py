"""Static candidate pools for P23 prompt receivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class P23StaticPoolResult:
    mask: torch.Tensor
    core_mask: torch.Tensor
    expand_mask: torch.Tensor
    receive_scale: torch.Tensor
    score: torch.Tensor
    ratio: float
    core_ratio: float
    expand_ratio: float
    strategy: str
    mode: str


def _minmax(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    lo = score.min()
    hi = score.max()
    return (score - lo) / (hi - lo).clamp_min(1e-12)


def _token_membership(
    x: torch.Tensor,
    *,
    mode: str,
    topk: int,
    binary_threshold: float,
    binary_topk: int,
) -> torch.Tensor:
    if mode == "binary_nonzero":
        membership = x.abs() > float(binary_threshold)
        if binary_topk > 0 and binary_topk < int(x.size(1)):
            capped = torch.zeros_like(membership)
            scores = x.abs().masked_fill(~membership, float("-inf"))
            indices = torch.topk(scores, k=int(binary_topk), dim=-1).indices
            capped.scatter_(1, indices, True)
            membership = capped & membership
        return membership
    if mode != "topk_activation":
        raise ValueError(f"Unsupported static pool feature tokenizer: {mode!r}")
    k = min(max(1, int(topk)), int(x.size(1)))
    membership = torch.zeros_like(x, dtype=torch.bool)
    membership.scatter_(1, torch.topk(x.abs(), k=k, dim=-1).indices, True)
    return membership


class P23StaticPoolSelector:
    """Build a static pool before P23 prompt-to-node message passing.

    The selector is deliberately parameter-free so it can be used as an
    ablation switch without adding trainable capacity.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "hard"))
        if self.mode not in {"hard", "soft"}:
            raise ValueError("static_pool.mode must be 'hard' or 'soft'")
        self.strategy = str(cfg.get("strategy", "feature_label_ambiguity"))
        self.ratio = float(cfg.get("ratio", cfg.get("pool_ratio", 1.0)))
        self.core_ratio = float(cfg.get("core_ratio", min(self.ratio, 0.15)))
        self.expand_ratio = float(cfg.get("expand_ratio", max(self.ratio, self.core_ratio)))
        self.core_scale = float(cfg.get("core_scale", 1.2))
        self.expand_scale = float(cfg.get("expand_scale", 1.1))
        self.non_pool_scale = float(cfg.get("non_pool_scale", 1.0))
        self.min_nodes = int(cfg.get("min_nodes", 1))
        self.force_train_nodes = bool(cfg.get("force_train_nodes", False))
        self.feature_topk = int(cfg.get("feature_topk", cfg.get("topk", 16)))
        self.tokenizer = str(cfg.get("tokenizer", cfg.get("mode", "topk_activation")))
        self.binary_threshold = float(cfg.get("binary_threshold", 0.0))
        self.binary_topk = int(cfg.get("binary_topk", 0))
        self.support_weight = float(cfg.get("support_weight", 0.5))
        self.feature_weight = float(cfg.get("feature_weight", 0.5))
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError("static_pool.ratio must be in (0, 1]")
        if not 0.0 < self.core_ratio <= 1.0:
            raise ValueError("static_pool.core_ratio must be in (0, 1]")
        if not 0.0 < self.expand_ratio <= 1.0:
            raise ValueError("static_pool.expand_ratio must be in (0, 1]")
        if self.expand_ratio < self.core_ratio:
            raise ValueError("static_pool.expand_ratio must be >= core_ratio")

    def select(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor | None,
        train_mask: torch.Tensor,
        num_classes: int,
    ) -> P23StaticPoolResult:
        num_nodes = int(x.size(0))
        device = x.device
        if not self.enabled:
            mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
            score = torch.ones(num_nodes, dtype=x.dtype, device=device)
            return P23StaticPoolResult(
                mask=mask,
                core_mask=mask,
                expand_mask=mask,
                receive_scale=score,
                score=score,
                ratio=1.0,
                core_ratio=1.0,
                expand_ratio=1.0,
                strategy="disabled",
                mode="disabled",
            )
        if labels is None:
            raise ValueError("static_pool requires labels for feature-label ambiguity scoring")

        if self.strategy == "feature_label_ambiguity":
            score = self._feature_label_ambiguity_score(
                x=x,
                labels=labels,
                train_mask=train_mask,
                num_classes=num_classes,
            )
        elif self.strategy == "support_neighbor_entropy":
            score = self._support_neighbor_entropy_score(
                x=x,
                edge_index=edge_index,
                labels=labels,
                train_mask=train_mask,
                num_classes=num_classes,
            )
        elif self.strategy == "hybrid_support_feature":
            support = self._support_neighbor_entropy_score(
                x=x,
                edge_index=edge_index,
                labels=labels,
                train_mask=train_mask,
                num_classes=num_classes,
            )
            feature = self._feature_label_ambiguity_score(
                x=x,
                labels=labels,
                train_mask=train_mask,
                num_classes=num_classes,
            )
            score = max(0.0, self.support_weight) * _minmax(support) + max(0.0, self.feature_weight) * _minmax(feature)
        else:
            raise ValueError(f"Unsupported static_pool.strategy={self.strategy!r}")

        core_mask = self._top_ratio_mask(score, ratio=self.core_ratio)
        expand_mask = self._top_ratio_mask(score, ratio=self.expand_ratio)
        mask = self._top_ratio_mask(score, ratio=self.ratio)
        if self.force_train_nodes:
            mask = mask | train_mask.to(device=device, dtype=torch.bool)
            expand_mask = expand_mask | train_mask.to(device=device, dtype=torch.bool)
            core_mask = core_mask | train_mask.to(device=device, dtype=torch.bool)
        if self.mode == "soft":
            receive_scale = torch.full((num_nodes,), float(self.non_pool_scale), dtype=x.dtype, device=device)
            receive_scale = torch.where(expand_mask, receive_scale.new_tensor(float(self.expand_scale)), receive_scale)
            receive_scale = torch.where(core_mask, receive_scale.new_tensor(float(self.core_scale)), receive_scale)
            mask = expand_mask
        else:
            receive_scale = mask.to(dtype=x.dtype)
        return P23StaticPoolResult(
            mask=mask,
            core_mask=core_mask,
            expand_mask=expand_mask,
            receive_scale=receive_scale,
            score=score,
            ratio=float(mask.to(dtype=torch.float32).mean().detach().item()) if mask.numel() else 0.0,
            core_ratio=float(core_mask.to(dtype=torch.float32).mean().detach().item()) if core_mask.numel() else 0.0,
            expand_ratio=float(expand_mask.to(dtype=torch.float32).mean().detach().item()) if expand_mask.numel() else 0.0,
            strategy=self.strategy,
            mode=self.mode,
        )

    def _top_ratio_mask(self, score: torch.Tensor, *, ratio: float | None = None) -> torch.Tensor:
        num_nodes = int(score.numel())
        selected_ratio = self.ratio if ratio is None else float(ratio)
        k = max(int(self.min_nodes), int(round(selected_ratio * float(num_nodes))))
        k = max(1, min(num_nodes, k))
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=score.device)
        if k > 0:
            idx = torch.topk(score, k=k, largest=True).indices
            mask[idx] = True
        return mask

    def _feature_label_ambiguity_score(
        self,
        *,
        x: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros(x.size(0))
        train = train_mask.to(device=x.device, dtype=torch.bool)
        if not bool(train.any()):
            return x.new_zeros(x.size(0))
        y = labels.to(device=x.device, dtype=torch.long)
        membership = _token_membership(
            x,
            mode=self.tokenizer,
            topk=self.feature_topk,
            binary_threshold=self.binary_threshold,
            binary_topk=self.binary_topk,
        )
        train_membership = membership[train].to(dtype=x.dtype)
        one_hot = F.one_hot(y[train].clamp_min(0), num_classes=max(1, int(num_classes))).to(dtype=x.dtype)
        label_counts = train_membership.t() @ one_hot
        totals = label_counts.sum(dim=-1)
        prob = label_counts / totals.clamp_min(1.0).unsqueeze(-1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        if int(num_classes) > 1:
            entropy = entropy / torch.log(entropy.new_tensor(float(num_classes))).clamp_min(1e-12)
        ambiguity = torch.where(totals > 0, entropy, torch.zeros_like(entropy))
        per_node_count = membership.to(dtype=x.dtype).sum(dim=-1).clamp_min(1.0)
        return (membership.to(dtype=x.dtype) @ ambiguity) / per_node_count

    def _support_neighbor_entropy_score(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        num_nodes = int(x.size(0))
        counts = x.new_zeros((num_nodes, max(1, int(num_classes))))
        if edge_index.numel() == 0:
            return x.new_zeros(num_nodes)
        src, dst = edge_index.to(device=x.device, dtype=torch.long)
        y = labels.to(device=x.device, dtype=torch.long)
        train = train_mask.to(device=x.device, dtype=torch.bool)
        for center, neigh in ((src, dst), (dst, src)):
            labeled = train[neigh]
            if bool(labeled.any()):
                cls = y[neigh[labeled]].clamp_min(0)
                counts.index_add_(0, center[labeled], F.one_hot(cls, num_classes=max(1, int(num_classes))).to(dtype=x.dtype))
        total = counts.sum(dim=-1)
        prob = counts / total.clamp_min(1.0).unsqueeze(-1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        if int(num_classes) > 1:
            entropy = entropy / torch.log(entropy.new_tensor(float(num_classes))).clamp_min(1e-12)
        confidence = 1.0 - prob.max(dim=-1).values
        return (total > 0).to(dtype=x.dtype) * (0.5 * entropy + 0.5 * confidence)
