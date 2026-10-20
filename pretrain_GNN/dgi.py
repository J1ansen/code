"""Deep Graph Infomax with graph-wide readout and corrupted features."""
import torch
from torch import nn
import torch.nn.functional as F


def encode_weighted(encoder, x, edge_index, edge_weight):
    for layer, conv in enumerate(encoder.convs):
        x = conv.propagate(edge_index, x=conv.lin(x), edge_weight=edge_weight)
        if conv.bias is not None:
            x = x + conv.bias
        if layer < len(encoder.convs)-1:
            x = encoder.act(x)
            x = F.dropout(x,p=encoder.dropout,training=encoder.training)
    return x


class DGI(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.gnn = encoder
        self.discriminator = nn.Bilinear(hidden_dim,hidden_dim,1)
        nn.init.xavier_uniform_(self.discriminator.weight)
        nn.init.zeros_(self.discriminator.bias)

    def loss(self, x, corrupted_x, edge_index, edge_weight, corrupted_edges, corrupted_weight):
        positive = encode_weighted(self.gnn,x,edge_index,edge_weight)
        negative = encode_weighted(self.gnn,corrupted_x,corrupted_edges,corrupted_weight)
        summary = positive.mean(dim=0).sigmoid().expand_as(positive)
        logits = torch.cat([self.discriminator(positive,summary).squeeze(-1),
                            self.discriminator(negative,summary).squeeze(-1)])
        labels = torch.cat([torch.ones_like(logits[:x.size(0)]),torch.zeros_like(logits[:x.size(0)])])
        return F.binary_cross_entropy_with_logits(logits,labels)
