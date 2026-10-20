"""GRACE objective and graph views following the GP2F implementation."""
import torch
from torch import nn
import torch.nn.functional as F


def normalized_edge_view(edge_index, num_nodes, drop_rate=0.0):
    if not 0 <= drop_rate < 1:
        raise ValueError("edge drop rate must be in [0, 1)")
    if drop_rate:
        keep = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_rate
        keep |= edge_index[0] == edge_index[1]
        edge_index = edge_index[:, keep]
    row, col = edge_index
    degree = torch.bincount(row, minlength=num_nodes).float()
    inverse = degree.pow(-0.5).masked_fill(degree == 0, 0)
    return edge_index, inverse[row] * inverse[col]


def drop_feature(x, probability):
    mask = torch.rand(x.size(1), device=x.device) < probability
    result = x.clone()
    result[:, mask] = 0
    return result


class GRACE(nn.Module):
    def __init__(self, encoder, hidden_dim, projection_dim, temperature=0.5):
        super().__init__()
        self.gnn = encoder
        self.fc1 = nn.Linear(hidden_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, hidden_dim)
        self.temperature = temperature

    def forward(self, x, edge_index, edge_weight):
        # The views already contain normalized weights; avoid normalizing twice.
        for layer, conv in enumerate(self.gnn.convs):
            x = conv.propagate(edge_index, x=conv.lin(x), edge_weight=edge_weight)
            if conv.bias is not None:
                x = x + conv.bias
            if layer < len(self.gnn.convs)-1:
                x = self.gnn.act(x)
        return x

    def projection(self, x):
        return self.fc2(F.elu(self.fc1(x)))

    def semi_loss(self, first, second):
        first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
        within = torch.exp((first @ first.t()) / self.temperature)
        between = torch.exp((first @ second.t()) / self.temperature)
        return -torch.log(between.diag() / (within.sum(1) + between.sum(1) - within.diag()))

    def loss(self, first, second):
        first, second = self.projection(first), self.projection(second)
        return ((self.semi_loss(first, second) + self.semi_loss(second, first)) * 0.5).mean()
