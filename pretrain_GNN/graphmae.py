"""Masked feature reconstruction with a GP2F-compatible GraphMAE-GAT encoder."""
import torch
from torch import nn
import torch.nn.functional as F

from models.graphmae_gat_backbone import GraphMAEGATBackbone, GraphMAEWeightedGATConv


class GraphMAE(nn.Module):
    def __init__(self,input_dim,spec):
        super().__init__()
        self.mask_rate = float(spec["mask_rate"])
        if not 0 < self.mask_rate < 1:
            raise ValueError("GraphMAE mask_rate must be in (0,1)")
        self.alpha = float(spec["sce_alpha"])
        self.feature_dropout = float(spec["feature_dropout"])
        self.encoder = GraphMAEGATBackbone(in_channels=input_dim,
            hidden_channels=spec["hidden_dim"],num_layers=spec["num_layers"],heads=spec["heads"],
            attention_dropout=spec["attention_dropout"],feature_dropout=self.feature_dropout,
            edge_weight_mode="ignore")
        self.decoder = GraphMAEWeightedGATConv(spec["hidden_dim"],input_dim,1,
            dropout=spec["attention_dropout"],edge_weight_mode="ignore")
        self.mask_token = nn.Parameter(torch.zeros(1,input_dim))
        self.encoder_to_decoder = nn.Linear(spec["hidden_dim"],spec["hidden_dim"],bias=False)

    def loss(self,x,edge_index):
        count = int(self.mask_rate*x.size(0))
        if not 0 < count < x.size(0):
            raise ValueError("GraphMAE requires at least one masked and one unmasked node")
        selected = torch.randperm(x.size(0),device=x.device)[:count]
        masked = x.clone()
        masked[selected] = 0
        masked[selected] += self.mask_token
        representation = self.encoder_to_decoder(self.encoder(masked,edge_index))
        representation = representation.clone()
        representation[selected] = 0
        reconstruction = self.decoder(F.dropout(representation,p=self.feature_dropout,training=self.training),edge_index)
        reconstructed = F.normalize(reconstruction[selected],p=2,dim=-1)
        original = F.normalize(x[selected],p=2,dim=-1)
        return (1-(reconstructed*original).sum(dim=-1)).pow(self.alpha).mean()
