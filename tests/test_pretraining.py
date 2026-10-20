from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.backbones import load_pretrained_backbone
from pretrain_GNN.grace import normalized_edge_view, drop_feature
from pretrain_GNN import pretrain


def test_normalized_views_preserve_self_loops():
    edges, _ = add_self_loops(torch.tensor([[0,1,1,2],[1,0,2,1]]),num_nodes=3)
    view, weights = normalized_edge_view(edges,3,.9)
    assert (view[0] == view[1]).sum() == 3
    degree = torch.bincount(view[0],minlength=3).float()
    assert torch.allclose(weights, degree[view[0]].rsqrt()*degree[view[1]].rsqrt())
    assert torch.isfinite(weights).all()
    x = torch.ones(3,8)
    torch.manual_seed(20)
    result = drop_feature(x,.5)
    assert torch.equal(result[0],result[1])
    assert torch.equal(x,torch.ones_like(x))


@pytest.mark.parametrize("strategy",["GRACE","DGI","GraphMAE"])
def test_missing_checkpoint_trains_and_existing_checkpoint_reuses(tmp_path,monkeypatch,strategy):
    spec = yaml.safe_load((ROOT/f"pretrain_GNN/cora_{strategy.lower()}.yaml").read_text())
    spec.update(hidden_dim=8,epochs=2,patience=0,device="cpu")
    if strategy == "GRACE":
        spec["projection_dim"] = 8
    path = tmp_path/"pretrain.yaml"
    path.write_text(yaml.safe_dump(spec))
    graph = Data(x=torch.arange(32).reshape(8,4).float()/32,
        edge_index=torch.stack([torch.arange(8),torch.arange(8).roll(1)]), y=torch.zeros(8,dtype=torch.long))
    monkeypatch.setattr(pretrain,"load_node_dataset",lambda *a,**kw: SimpleNamespace(data=graph,num_features=4))
    config = dict(data={"root":"datasets"},pretrained={"checkpoint_path":"checkpoints/test.pt", "auto_pretrain":True,"pretraining_config_path":str(path)})
    checkpoint = pretrain.ensure_pretrained(config,repo_root=tmp_path)
    assert checkpoint.is_file()
    backbone = load_pretrained_backbone(checkpoint,device=torch.device("cpu"),freeze=True,strict=True)
    output = backbone(graph.x,graph.edge_index)
    assert output.shape == (8,8) and torch.isfinite(output).all()
    original = checkpoint.read_bytes()
    monkeypatch.setattr(pretrain,"load_node_dataset",lambda *a,**kw: pytest.fail("Cache reuse must not train"))
    assert pretrain.ensure_pretrained(config,repo_root=tmp_path) == checkpoint
    assert checkpoint.read_bytes() == original
    config["pretrained"]["pretraining_overrides"] = {"epochs":3}
    with pytest.raises(ValueError,match="settings differ"):
        pretrain.ensure_pretrained(config,repo_root=tmp_path)
    assert not list(tmp_path.rglob("*.lock"))
