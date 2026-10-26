"""GP2F GraphMAE-GAT checkpoint compatibility for weighted prompt graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_remaining_self_loops, softmax


class GraphMAEWeightedGATConv(MessagePassing):
    """GP2F GraphMAE GAT layer with an optional scalar attention prior.

    The parameter layout matches GP2F's bundled GraphMAE implementation. When
    ``edge_weight_mode='ignore'`` or no weights are supplied, the attention is
    the original GP2F behavior. ``attention_prior`` adds
    ``edge_weight_strength * log(edge_weight)`` before neighborhood softmax so
    PromptGraph prompt-edge weights remain meaningful without forcing full-strength
    reweighting on every pretrained encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        *,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        activation: nn.Module | None = None,
        edge_weight_mode: str = "attention_prior",
        edge_weight_strength: float = 1.0,
    ) -> None:
        super().__init__(node_dim=0, aggr="add")
        if edge_weight_mode not in {"ignore", "attention_prior"}:
            raise ValueError("edge_weight_mode must be 'ignore' or 'attention_prior'")
        if float(edge_weight_strength) < 0.0:
            raise ValueError("edge_weight_strength must be non-negative")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = True
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)
        self.edge_weight_mode = str(edge_weight_mode)
        self.edge_weight_strength = float(edge_weight_strength)
        self.activation = activation

        self.lin_src = Linear(
            self.in_channels,
            self.heads * self.out_channels,
            bias=False,
            weight_initializer="glorot",
        )
        self.lin_dst = self.lin_src
        self.att_src = Parameter(torch.empty(1, self.heads, self.out_channels))
        self.att_dst = Parameter(torch.empty(1, self.heads, self.out_channels))
        self.bias = Parameter(torch.empty(self.heads * self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.lin_src.weight, gain=1.414)
        nn.init.xavier_normal_(self.att_src, gain=1.414)
        nn.init.xavier_normal_(self.att_dst, gain=1.414)
        nn.init.zeros_(self.bias)
        if self.activation is not None and hasattr(self.activation, "reset_parameters"):
            self.activation.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if not isinstance(edge_index, Tensor):
            raise TypeError("GraphMAE prompt compatibility currently requires tensor edge_index")
        if x.dim() != 2:
            raise ValueError("GraphMAE GAT expects a two-dimensional feature tensor")

        num_nodes = int(x.size(0))
        if edge_weight is None:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
            effective_weight = None
        else:
            edge_index, effective_weight = add_remaining_self_loops(
                edge_index,
                edge_weight.reshape(-1),
                fill_value=1.0,
                num_nodes=num_nodes,
            )

        transformed = self.lin_src(x).view(-1, self.heads, self.out_channels)
        alpha_src = (transformed * self.att_src).sum(dim=-1)
        alpha_dst = (transformed * self.att_dst).sum(dim=-1)
        alpha = self.edge_updater(
            edge_index,
            alpha=(alpha_src, alpha_dst),
            edge_weight=effective_weight,
        )
        out = self.propagate(
            edge_index,
            x=(transformed, transformed),
            alpha=alpha,
            size=size,
        )
        out = out.reshape(-1, self.heads * self.out_channels)
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

    def edge_update(
        self,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_weight: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if self.edge_weight_mode == "attention_prior" and edge_weight is not None:
            prior = edge_weight.to(dtype=alpha.dtype).clamp_min(1e-12).log().unsqueeze(-1)
            alpha = alpha + self.edge_weight_strength * prior
        alpha = softmax(alpha, index, ptr, size_i)
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


class GraphMAEGATBackbone(nn.Module):
    """Frozen two-layer GAT encoder emitted by GP2F GraphMAE pretraining."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        attention_dropout: float = 0.1,
        feature_dropout: float = 0.2,
        edge_weight_mode: str = "attention_prior",
        edge_weight_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_heads = int(heads)
        self.dropout = float(feature_dropout)
        self.edge_weight_mode = str(edge_weight_mode)
        self.edge_weight_strength = float(edge_weight_strength)
        self.jk_mode = "last"
        self.activation = nn.PReLU()
        # Each copied GraphMAE convolution already contains its own activation.
        self.act = nn.Identity()
        per_head = self.hidden_channels // self.num_heads
        self.convs = nn.ModuleList()
        layer_input = self.in_channels
        for _ in range(self.num_layers):
            self.convs.append(
                GraphMAEWeightedGATConv(
                    layer_input,
                    per_head,
                    self.num_heads,
                    dropout=float(attention_dropout),
                    activation=nn.PReLU(),
                    edge_weight_mode=self.edge_weight_mode,
                    edge_weight_strength=self.edge_weight_strength,
                )
            )
            layer_input = self.hidden_channels

    @property
    def uses_edge_weight(self) -> bool:
        return self.edge_weight_mode == "attention_prior"

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        h = x
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index, edge_weight)
        return h


def infer_graphmae_gat_dimensions(
    state_dict: Mapping[str, Tensor],
) -> tuple[int, int, int, int]:
    first_weight = state_dict.get("convs.0.lin_src.weight")
    first_attention = state_dict.get("convs.0.att_src")
    if first_weight is None or first_attention is None:
        raise ValueError("Checkpoint does not contain GP2F GraphMAE-GAT parameters")
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("convs.") and key.endswith(".att_src")
    }
    in_channels = int(first_weight.shape[1])
    heads = int(first_attention.shape[1])
    hidden_channels = int(first_attention.shape[1] * first_attention.shape[2])
    num_layers = max(layer_ids) + 1
    return in_channels, hidden_channels, num_layers, heads


def load_graphmae_gat_state(
    model: GraphMAEGATBackbone,
    state_dict: Mapping[str, Tensor],
) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            "GraphMAE-GAT checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(dict(state_dict), strict=True)
