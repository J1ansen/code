"""Loss functions for classification and optional graph adaptation objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


@dataclass(frozen=True)
class GP2FLossConfig:
    use_cls: bool = True
    use_original_contrastive: bool = False
    use_original_topology_fusion: bool = False
    lambda_ctr: float = 0.0
    lambda_fus: float = 0.0
    tau_ctr: float = 0.5
    tau_fus: float = 0.05
    topology_percentile: float = 70.0
    topology_threshold_mode: str = "exact"
    topology_threshold_sample_size: int = 5000
    topology_threshold_large_graph_nodes: int = 1000
    dense_contrastive_max_nodes: int = 5000
    contrastive_sample_size: int = 2048
    dense_topology_max_nodes: int = 5000
    topology_sample_size: int = 20000


@dataclass(frozen=True)
class GP2FLossOutput:
    total: torch.Tensor
    cls: torch.Tensor
    contrastive: torch.Tensor
    topology_fusion: torch.Tensor
    contrastive_mode: str = "disabled"
    topology_mode: str = "disabled"

    def to_log_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "cls": float(self.cls.detach().item()),
            "contrastive": float(self.contrastive.detach().item()),
            "topology_fusion": float(self.topology_fusion.detach().item()),
        }

    def to_metadata_dict(self) -> dict[str, str]:
        return {
            "contrastive_mode": self.contrastive_mode,
            "topology_mode": self.topology_mode,
        }


def _similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1, eps=1e-12)
    z2 = F.normalize(z2, dim=-1, eps=1e-12)
    return z1 @ z2.t()


def _positive_masks(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    refl_mask = (adj - eye > 0).float()
    between_mask = (adj > 0).float()
    between_mask.fill_diagonal_(1.0)
    return refl_mask, between_mask


def original_contrastive_loss(
    h_pre: torch.Tensor,
    h_adp: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    tau: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official GP2F-style original-adjacency cross-view contrastive loss."""

    num_nodes = h_pre.size(0)
    refl_mask, between_mask = _positive_masks(edge_index, num_nodes)
    scale = max(float(tau), 1e-6)
    exp_sim = lambda x: torch.exp(x / scale)

    pre_self = _similarity(h_pre, h_pre)
    adp_self = _similarity(h_adp, h_adp)
    pre_to_adp = _similarity(h_pre, h_adp)
    adp_to_pre = _similarity(h_adp, h_pre)

    pre_num = (exp_sim(pre_self) * refl_mask).sum(dim=1) + (exp_sim(pre_to_adp) * between_mask).sum(dim=1)
    pre_den = exp_sim(pre_self).sum(dim=1) + exp_sim(pre_to_adp).sum(dim=1) - exp_sim(pre_self).diag() + 1e-8
    adp_num = (exp_sim(adp_self) * refl_mask).sum(dim=1) + (exp_sim(adp_to_pre) * between_mask).sum(dim=1)
    adp_den = exp_sim(adp_self).sum(dim=1) + exp_sim(adp_to_pre).sum(dim=1) - exp_sim(adp_self).diag() + 1e-8

    loss = -0.5 * (torch.log(pre_num / pre_den) + torch.log(adp_num / adp_den))
    return loss.mean(), pre_self, adp_self


def sampled_original_contrastive_loss(
    h_pre: torch.Tensor,
    h_adp: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    tau: float = 0.5,
    sample_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Memory-safe contrastive loss on an induced random node subset."""

    num_nodes = h_pre.size(0)
    subset_size = min(max(2, int(sample_size)), num_nodes)
    if subset_size >= num_nodes:
        return original_contrastive_loss(h_pre, h_adp, edge_index, tau=tau)

    device = h_pre.device
    idx = torch.randperm(num_nodes, device=device)[:subset_size]
    inverse = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    inverse[idx] = torch.arange(subset_size, device=device)

    src, dst = edge_index[0], edge_index[1]
    keep = (inverse[src] >= 0) & (inverse[dst] >= 0)
    sub_edge_index = torch.stack([inverse[src[keep]], inverse[dst[keep]]], dim=0)
    loss, _, _ = original_contrastive_loss(h_pre[idx], h_adp[idx], sub_edge_index, tau=tau)
    return loss, None, None


def _similarity_threshold(fused_similarity: torch.Tensor, percentile: float) -> torch.Tensor:
    num_nodes = fused_similarity.size(0)
    if num_nodes <= 1:
        return fused_similarity.new_tensor(0.0)
    mask = ~torch.eye(num_nodes, device=fused_similarity.device, dtype=torch.bool)
    return torch.quantile(fused_similarity[mask], float(percentile) / 100.0)


def _as_alpha_tensor(alpha: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(alpha, torch.Tensor):
        return alpha.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(float(alpha), device=reference.device, dtype=reference.dtype)


def original_topology_fusion_loss(
    fused_similarity: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    tau: float = 0.05,
    percentile: float = 70.0,
    threshold_mode: str = "exact",
    threshold_sample_size: int = 5000,
    threshold_large_graph_nodes: int = 1000,
) -> torch.Tensor:
    """Official GP2F-style topology consistency against original adjacency."""

    num_nodes = fused_similarity.size(0)
    target_adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    with torch.no_grad():
        if threshold_mode == "exact":
            threshold = _similarity_threshold(fused_similarity, percentile)
        elif threshold_mode == "gp2f_sampled_pairs":
            if num_nodes > int(threshold_large_graph_nodes):
                sample_size = min(
                    max(1, int(threshold_sample_size)),
                    num_nodes * (num_nodes - 1) // 2,
                )
                idx1 = torch.randint(0, num_nodes, (sample_size,), device=fused_similarity.device)
                idx2 = torch.randint(0, num_nodes, (sample_size,), device=fused_similarity.device)
                valid_mask = idx1 != idx2
                if int(valid_mask.sum().item()) > 100:
                    threshold = torch.quantile(
                        fused_similarity[idx1[valid_mask], idx2[valid_mask]],
                        float(percentile) / 100.0,
                    )
                else:
                    threshold = fused_similarity.new_tensor(0.0)
            else:
                threshold = _similarity_threshold(fused_similarity, percentile)
        else:
            raise ValueError(
                f"Unsupported topology threshold mode: {threshold_mode!r}; "
                "expected 'exact' or 'gp2f_sampled_pairs'"
            )
        constraint_mask = ((fused_similarity > threshold) & (target_adj > 0)) | (
            (fused_similarity <= threshold) & (target_adj <= 0)
        )

    prob = torch.sigmoid(fused_similarity / max(float(tau), 1e-6))
    loss_map = F.binary_cross_entropy(prob, target_adj * constraint_mask, reduction="none")
    return (loss_map * constraint_mask).sum() / (constraint_mask.sum() + 1e-8)


def _alpha_fused_similarity(
    h_pre: torch.Tensor,
    h_adp: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    alpha_t = _as_alpha_tensor(alpha, h_pre)
    pre_sim = _similarity(h_pre, h_pre)
    adp_sim = _similarity(h_adp, h_adp)
    return alpha_t * pre_sim + (1.0 - alpha_t) * adp_sim


def _sample_negative_pairs(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    edge_hash = edge_index[0] * num_nodes + edge_index[1]
    neg_src_parts: list[torch.Tensor] = []
    neg_dst_parts: list[torch.Tensor] = []
    remaining = int(count)
    attempts = 0
    while remaining > 0 and attempts < 10:
        attempts += 1
        draw = max(remaining * 2, 16)
        src = torch.randint(0, num_nodes, (draw,), device=device)
        dst = torch.randint(0, num_nodes, (draw,), device=device)
        candidate_hash = src * num_nodes + dst
        keep = (src != dst) & ~torch.isin(candidate_hash, edge_hash)
        src = src[keep][:remaining]
        dst = dst[keep][:remaining]
        if src.numel() > 0:
            neg_src_parts.append(src)
            neg_dst_parts.append(dst)
            remaining -= int(src.numel())

    if not neg_src_parts:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    neg_src = torch.cat(neg_src_parts, dim=0)[:count]
    neg_dst = torch.cat(neg_dst_parts, dim=0)[:count]
    return neg_src, neg_dst


def sampled_original_topology_fusion_loss(
    h_pre: torch.Tensor,
    h_adp: torch.Tensor,
    alpha: torch.Tensor | float,
    edge_index: torch.Tensor,
    *,
    tau: float = 0.05,
    percentile: float = 70.0,
    sample_size: int = 20000,
) -> torch.Tensor:
    """Sampled approximation of GP2F topology consistency against original topology."""

    num_nodes = h_pre.size(0)
    device = h_pre.device
    if num_nodes <= 1 or edge_index.numel() == 0:
        return h_pre.new_tensor(0.0)

    edge_src, edge_dst = edge_index[0], edge_index[1]
    keep = edge_src != edge_dst
    edge_src, edge_dst = edge_src[keep], edge_dst[keep]
    if edge_src.numel() == 0:
        return h_pre.new_tensor(0.0)

    half = max(1, int(sample_size) // 2)
    pos_count = min(half, int(edge_src.numel()))
    pos_perm = torch.randperm(edge_src.numel(), device=device)[:pos_count]
    pos_src = edge_src[pos_perm]
    pos_dst = edge_dst[pos_perm]

    neg_count = max(1, int(sample_size) - pos_count)
    neg_src, neg_dst = _sample_negative_pairs(
        num_nodes=num_nodes,
        edge_index=edge_index,
        count=neg_count,
        device=device,
    )
    if neg_src.numel() == 0:
        return h_pre.new_tensor(0.0)

    src = torch.cat([pos_src, neg_src], dim=0)
    dst = torch.cat([pos_dst, neg_dst], dim=0)
    target = torch.cat(
        [
            torch.ones(pos_src.size(0), device=device, dtype=h_pre.dtype),
            torch.zeros(neg_src.size(0), device=device, dtype=h_pre.dtype),
        ],
        dim=0,
    )
    alpha_t = _as_alpha_tensor(alpha, h_pre)
    pre_norm = F.normalize(h_pre, dim=-1, eps=1e-12)
    adp_norm = F.normalize(h_adp, dim=-1, eps=1e-12)
    pre_sim = (pre_norm[src] * pre_norm[dst]).sum(dim=-1)
    adp_sim = (adp_norm[src] * adp_norm[dst]).sum(dim=-1)
    fused_similarity = alpha_t * pre_sim + (1.0 - alpha_t) * adp_sim

    with torch.no_grad():
        threshold = torch.quantile(fused_similarity.detach(), float(percentile) / 100.0)
        constraint_mask = ((fused_similarity > threshold) & (target > 0)) | (
            (fused_similarity <= threshold) & (target <= 0)
        )
    if not bool(constraint_mask.any()):
        return h_pre.new_tensor(0.0)

    prob = torch.sigmoid(fused_similarity / max(float(tau), 1e-6))
    loss_map = F.binary_cross_entropy(prob, target, reduction="none")
    return (loss_map * constraint_mask.float()).sum() / (constraint_mask.float().sum() + 1e-8)


def compute_gp2f_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    h_pre: torch.Tensor,
    h_adp: torch.Tensor,
    h_mix: torch.Tensor,
    alpha: torch.Tensor | float | None = None,
    edge_index: torch.Tensor,
    cfg: GP2FLossConfig,
) -> GP2FLossOutput:
    if cfg.use_cls:
        cls = F.cross_entropy(logits[train_mask], labels[train_mask])
    else:
        cls = logits.new_tensor(0.0)

    contrastive = logits.new_tensor(0.0)
    contrastive_mode = "disabled"
    pre_sim = adp_sim = None
    if cfg.use_original_contrastive and float(cfg.lambda_ctr) > 0:
        if h_pre.size(0) > int(cfg.dense_contrastive_max_nodes):
            contrastive_mode = "sampled_induced"
            contrastive, pre_sim, adp_sim = sampled_original_contrastive_loss(
                h_pre,
                h_adp,
                edge_index,
                tau=cfg.tau_ctr,
                sample_size=cfg.contrastive_sample_size,
            )
        else:
            contrastive_mode = "dense_original"
            contrastive, pre_sim, adp_sim = original_contrastive_loss(
                h_pre,
                h_adp,
                edge_index,
                tau=cfg.tau_ctr,
            )

    topology = logits.new_tensor(0.0)
    topology_mode = "disabled"
    if cfg.use_original_topology_fusion and float(cfg.lambda_fus) > 0:
        if alpha is None:
            raise ValueError("alpha must be provided when original topology fusion loss is enabled.")
        if h_mix.size(0) > int(cfg.dense_topology_max_nodes):
            topology_mode = "sampled_alpha_consistency_approx"
            topology = sampled_original_topology_fusion_loss(
                h_pre,
                h_adp,
                alpha,
                edge_index,
                tau=cfg.tau_fus,
                percentile=cfg.topology_percentile,
                sample_size=cfg.topology_sample_size,
            )
        else:
            topology_mode = (
                "dense_alpha_consistency_gp2f_sampled_threshold"
                if cfg.topology_threshold_mode == "gp2f_sampled_pairs"
                else "dense_alpha_consistency"
            )
            if pre_sim is None or adp_sim is None:
                fused_similarity = _alpha_fused_similarity(h_pre, h_adp, alpha)
            else:
                alpha_t = _as_alpha_tensor(alpha, h_pre)
                fused_similarity = alpha_t * pre_sim + (1.0 - alpha_t) * adp_sim
            topology = original_topology_fusion_loss(
                fused_similarity,
                edge_index,
                tau=cfg.tau_fus,
                percentile=cfg.topology_percentile,
                threshold_mode=cfg.topology_threshold_mode,
                threshold_sample_size=cfg.topology_threshold_sample_size,
                threshold_large_graph_nodes=cfg.topology_threshold_large_graph_nodes,
            )

    total = cls + float(cfg.lambda_ctr) * contrastive + float(cfg.lambda_fus) * topology
    return GP2FLossOutput(
        total=total,
        cls=cls,
        contrastive=contrastive,
        topology_fusion=topology,
        contrastive_mode=contrastive_mode,
        topology_mode=topology_mode,
    )
