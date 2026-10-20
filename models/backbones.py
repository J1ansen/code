"""Pretrained GNN backbone loading utilities.

This module intentionally does not implement pretraining. Baseline experiments
must load an existing checkpoint and fail fast if the checkpoint is missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from models.graphmae_gat_backbone import (
    GraphMAEGATBackbone,
    infer_graphmae_gat_dimensions,
    load_graphmae_gat_state,
)


class BaseGCN(nn.Module):
    """GP2F-compatible GCN encoder used for loading pretrained weights."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        act: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.jk_mode = "last"
        self.jk = None
        self.act = act if act is not None else nn.PReLU()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(int(in_channels), int(hidden_channels)))
        for _ in range(1, self.num_layers):
            self.convs.append(GCNConv(int(hidden_channels), int(hidden_channels)))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for layer_idx, conv in enumerate(self.convs):
            if edge_weight is None:
                h = conv(h, edge_index)
            else:
                h = conv(h, edge_index, edge_weight)

            if layer_idx < len(self.convs) - 1:
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def unwrap_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Return a state dict from common checkpoint payload formats."""

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, Mapping):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    return state_dict


def normalize_backbone_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize GP2F/GraphTOP-like GCN keys to this repo's `convs.{i}.*` form."""

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        stripped_prefix = True
        while stripped_prefix:
            stripped_prefix = False
            for prefix in ("module.", "gnn.", "encoder.", "backbone.", "gcn."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    stripped_prefix = True
                    break

        if key.startswith(("disc.", "read.", "sigm.")):
            continue

        if key.startswith("conv1."):
            key = key.replace("conv1.", "convs.0.", 1)
        elif key.startswith("conv2."):
            key = key.replace("conv2.", "convs.1.", 1)
        elif key.startswith("layers."):
            key = key.replace("layers.", "convs.", 1)

        normalized[key] = value
    return normalized


def infer_gcn_dimensions(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer `(in_channels, hidden_channels, num_layers)` from a GCN state dict."""

    normalized = normalize_backbone_keys(state_dict)
    first_weight = None
    layer_ids: set[int] = set()

    for key, value in normalized.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("convs.") and key.endswith("lin.weight"):
            parts = key.split(".")
            if len(parts) >= 4 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
            if key == "convs.0.lin.weight":
                first_weight = value
        elif key.startswith("convs.") and key.endswith("weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
            if key == "convs.0.weight":
                first_weight = value

    if first_weight is None or first_weight.ndim != 2:
        raise ValueError("Cannot infer GCN dimensions from checkpoint keys.")

    num_layers = max(layer_ids) + 1 if layer_ids else 2
    return int(first_weight.shape[1]), int(first_weight.shape[0]), int(num_layers)


def load_pretrained_gcn(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    freeze: bool = True,
    strict: bool = False,
) -> BaseGCN:
    """Load an existing pretrained GCN checkpoint without running pretraining."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}. "
            "Provide an existing .pkl/.pth file instead of triggering pretraining."
        )

    checkpoint = torch.load(path, map_location=device)
    state_dict = normalize_backbone_keys(unwrap_checkpoint(checkpoint))
    in_channels, hidden_channels, num_layers = infer_gcn_dimensions(state_dict)

    model = BaseGCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=0.0,
        act=nn.PReLU(),
    ).to(device)
    model.load_state_dict(state_dict, strict=strict)

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

    return model


def infer_backbone_dimensions(backbone: nn.Module) -> tuple[int, int]:
    """Return source input and output dimensions for a supported backbone."""

    if isinstance(backbone, GraphMAEGATBackbone):
        return int(backbone.in_channels), int(backbone.hidden_channels)
    convs = getattr(backbone, "convs", None)
    if convs is None or len(convs) == 0:
        raise TypeError("Backbone must expose at least one convolution layer")
    first_conv = convs[0]
    linear = getattr(first_conv, "lin", None)
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("Cannot infer dimensions from the pretrained backbone")
    return int(weight.shape[1]), int(weight.shape[0])


def load_pretrained_backbone(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    freeze: bool = True,
    strict: bool = False,
    graphmae_edge_weight_mode: str = "attention_prior",
    graphmae_edge_weight_strength: float = 1.0,
) -> nn.Module:
    """Auto-detect and load a GP2F GCN or GraphMAE-GAT encoder checkpoint."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}. "
            "Provide an existing .pkl/.pth file instead of triggering pretraining."
        )
    checkpoint = torch.load(path, map_location=device)
    state_dict = normalize_backbone_keys(unwrap_checkpoint(checkpoint))
    if "convs.0.att_src" not in state_dict:
        return load_pretrained_gcn(
            path,
            device=device,
            freeze=freeze,
            strict=strict,
        )

    in_channels, hidden_channels, num_layers, heads = infer_graphmae_gat_dimensions(state_dict)
    model = GraphMAEGATBackbone(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        attention_dropout=0.1,
        feature_dropout=0.2,
        edge_weight_mode=graphmae_edge_weight_mode,
        edge_weight_strength=graphmae_edge_weight_strength,
    ).to(device)
    load_graphmae_gat_state(model, state_dict)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model
