"""Identity-bound prompt-edge parameters, registered before optimizer creation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class IndependentPromptEdgeWeights(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = str(config.get("independent_activation", "sigmoid"))
        if self.activation not in {"sigmoid", "elu"}:
            raise ValueError("independent_activation must be sigmoid or elu")
        self.elu_scale = float(config.get("elu_scale", 1.0))
        if self.elu_scale <= 0:
            raise ValueError("elu_scale must be positive")
        path = Path(config["edge_manifest_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["edge_manifest_sha256"]:
            raise ValueError("Independent edge manifest hash mismatch")
        manifest = torch.load(path, map_location="cpu", weights_only=True)
        for name in ("node_index", "feature_local", "feature_ids"):
            value = manifest[name]
            if value.ndim != 1 or value.dtype != torch.long:
                raise ValueError(f"Invalid edge manifest tensor: {name}")
            self.register_buffer(name, value.clone())
        if self.node_index.shape != self.feature_local.shape:
            raise ValueError("Edge manifest lengths differ")
        if self.feature_local.numel() and (
            self.feature_local.min() < 0 or self.feature_local.max() >= self.feature_ids.numel()
            or self.node_index.min() < 0
        ):
            raise ValueError("Invalid edge manifest indices")
        self.delta = nn.Parameter(torch.zeros(self.node_index.numel()))
        self.register_buffer("gradient_seen", torch.zeros_like(self.delta, dtype=torch.bool))
        self.delta.register_hook(self._record_gradient)

    def _record_gradient(self, gradient):
        with torch.no_grad():
            self.gradient_seen.logical_or_(gradient != 0)
        return gradient

    def validate_features(self, feature_ids):
        if not torch.equal(feature_ids.to(self.feature_ids.device), self.feature_ids):
            raise ValueError("Independent edge feature identity mismatch")

    def forward(self, node_index, feature_local):
        if not torch.equal(node_index, self.node_index) or not torch.equal(feature_local, self.feature_local):
            raise ValueError("Independent edge identity/order mismatch")
        if self.activation == "sigmoid":
            return 0.5 + torch.sigmoid(self.delta)
        return F.elu(self.elu_scale * self.delta) + 1.0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("node_index", "feature_local", "feature_ids"):
            incoming = state_dict.get(prefix + name)
            expected = getattr(self, name)
            if incoming is not None and not torch.equal(incoming.to(expected.device), expected):
                raise ValueError("Checkpoint edge identity mismatch: " + name)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
