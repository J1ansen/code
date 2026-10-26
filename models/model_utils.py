"""Shared tensor and parameter utilities for the paper model."""
from __future__ import annotations
import torch
from torch import nn


def count_trainable_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


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
