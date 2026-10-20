"""Layer-wise Representation Adaptation with GP2F-style residual adapters."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ResidualRepresentationAdapter(nn.Module):
    """Bottleneck residual adapter: `h + beta * up(relu(down(h)))`."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, *, beta_init: float = 0.01) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.down = nn.Linear(int(hidden_dim), int(bottleneck_dim), bias=False)
        self.up = nn.Linear(int(bottleneck_dim), int(hidden_dim), bias=False)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.beta * self.up(F.relu(self.down(h)))


class ScaledRepresentationAdapter(nn.Module):
    """Official GP2F-style adapter block with a learnable residual scale."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, *, alpha_init: float = 0.1) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.adapter = nn.Sequential(
            nn.Linear(int(hidden_dim), int(bottleneck_dim)),
            nn.ReLU(),
            nn.Linear(int(bottleneck_dim), int(hidden_dim)),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def beta(self) -> torch.Tensor:
        return self.alpha

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.alpha * self.adapter(h)


def build_adapter(
    *,
    style: str,
    hidden_dim: int,
    bottleneck_dim: int,
    beta_init: float = 0.01,
    alpha_init: float = 0.1,
) -> nn.Module:
    """Build a layer adapter by style name."""

    if style == "stable_zero_init":
        return ResidualRepresentationAdapter(hidden_dim, bottleneck_dim, beta_init=beta_init)
    if style == "official_gp2f":
        return ScaledRepresentationAdapter(hidden_dim, bottleneck_dim, alpha_init=alpha_init)
    raise ValueError(f"Unsupported adapter style: {style}")
