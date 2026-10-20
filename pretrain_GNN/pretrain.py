"""Train a source encoder automatically, then reuse its checked checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data import load_node_dataset
from models.backbones import BaseGCN
from pretrain_GNN.grace import GRACE, drop_feature, normalized_edge_view
from pretrain_GNN.dgi import DGI
from pretrain_GNN.graphmae import GraphMAE


def resolve(path, root):
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def ensure_pretrained(config, *, repo_root=ROOT):
    pretrained = config["pretrained"]
    checkpoint = resolve(pretrained["checkpoint_path"], repo_root)
    if not pretrained.get("auto_pretrain", True):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    spec = yaml.safe_load(resolve(pretrained.get("pretraining_config_path", "pretrain_GNN/cora_grace.yaml"), repo_root).read_text())
    overrides = pretrained.get("pretraining_overrides", {})
    unknown = set(overrides) - set(spec)
    if unknown:
        raise ValueError(f"Unknown pretraining overrides: {sorted(unknown)}")
    spec.update(overrides)
    strategy = spec.get("strategy","GRACE")
    if strategy not in ("GRACE","DGI","GraphMAE"):
        raise ValueError(f"Unsupported pretraining strategy: {strategy}")
    if pretrained.get("strategy",strategy) != strategy:
        raise ValueError("Requested strategy does not match pretraining_config_path")
    signature = {k:v for k,v in spec.items() if k not in ("device", "download_if_missing")}
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if checkpoint.is_file():
        if not metadata_path.is_file():
            raise ValueError(f"Checkpoint lacks pretraining provenance: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata["specification"] != signature:
            raise ValueError("Cached pretraining settings differ; use a new checkpoint path")
        if metadata["sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("Cached checkpoint checksum mismatch")
        print(f"[pretrain] Reusing {checkpoint}", flush=True)
        return checkpoint
    if int(spec["epochs"]) < 1 or int(spec["patience"]) < 0:
        raise ValueError("Pretraining epochs must be positive and patience nonnegative")
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec["seed"])
    device_name = spec["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    source = load_node_dataset(spec["dataset"], resolve(config["data"]["root"], repo_root),
        download_if_missing=spec["download_if_missing"], normalize_features=True)
    graph = source.data.to(device)
    edge_index, _ = remove_self_loops(graph.edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=graph.num_nodes)
    if strategy == "GraphMAE":
        model = GraphMAE(source.num_features,spec)
        encoder = model.encoder
    else:
        encoder = BaseGCN(source.num_features,spec["hidden_dim"],num_layers=spec["num_layers"],
                          dropout=spec.get("encoder_dropout",0.0))
        model = (GRACE(encoder,spec["hidden_dim"],spec["projection_dim"],spec["temperature"])
                 if strategy == "GRACE" else DGI(encoder,spec["hidden_dim"]))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    # An exclusive lock prevents concurrent dataset processes from training the
    # shared source checkpoint into the same destination.
    with lock.open("x") as handle:
        handle.write("pretraining in progress\n")
    best, best_epoch, waiting, history = float("inf"), 0, 0, []
    best_state = None
    try:
        print(f"[pretrain] Training {strategy} on {spec['dataset']} ({device})", flush=True)
        for epoch in range(1, spec["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if strategy == "GRACE":
                first_edges, first_weights = normalized_edge_view(edge_index,graph.num_nodes,spec["edge_drop_rate"])
                second_edges, second_weights = normalized_edge_view(edge_index,graph.num_nodes,spec["edge_drop_rate"])
                first_x = drop_feature(graph.x,spec["feature_drop_rate"])
                second_x = drop_feature(graph.x,spec["feature_drop_rate"])
                loss = model.loss(model(first_x,first_edges,first_weights),model(second_x,second_edges,second_weights))
            elif strategy == "DGI":
                corrupted = drop_feature(graph.x[torch.randperm(graph.num_nodes,device=device)],spec["feature_drop_rate"])
                positive_edges, positive_weights = normalized_edge_view(edge_index,graph.num_nodes)
                negative_edges, negative_weights = normalized_edge_view(edge_index,graph.num_nodes,spec["edge_drop_rate"])
                loss = model.loss(graph.x,corrupted,positive_edges,positive_weights,negative_edges,negative_weights)
            else:
                loss = model.loss(graph.x,edge_index)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {strategy} loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            history.append(dict(epoch=epoch, loss=value))
            if value < best:
                best, best_epoch, waiting = value, epoch, 0
                best_state = {k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
            else:
                waiting += 1
            if epoch == 1 or epoch % 20 == 0:
                print(f"[pretrain] epoch={epoch} loss={value:.6f} best={best:.6f}", flush=True)
            if spec["patience"] and waiting >= spec["patience"]:
                break
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(best_state, temporary)
        metadata = dict(strategy=strategy, specification=signature, best_epoch=best_epoch,
            best_loss=best, epochs_completed=epoch, num_nodes=graph.num_nodes,
            input_dim=source.num_features, selection="minimum_self_supervised_training_loss",
            target_labels_used=False, source_labels_used=False,
            sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(), history=history)
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2))
        temporary_metadata.replace(metadata_path)
        temporary.replace(checkpoint)
        print(f"[pretrain] Saved {checkpoint}; best epoch={best_epoch}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main/cora.yaml")
    parser.add_argument("--strategy",choices=["GRACE","DGI","GraphMAE"])
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config,ROOT).read_text())
    if args.strategy:
        config["pretrained"].update(strategy=args.strategy,
            checkpoint_path=f"checkpoints/cora_{args.strategy.lower()}.pt",
            pretraining_config_path=f"pretrain_GNN/cora_{args.strategy.lower()}.yaml")
    ensure_pretrained(config)
