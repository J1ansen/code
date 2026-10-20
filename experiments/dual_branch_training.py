"""Run the faithful GP2F baseline on one target node-classification dataset."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_geometric
from torch import nn
from tqdm.auto import tqdm

from data import build_few_shot_split, load_node_dataset
from losses import GP2FLossConfig, compute_gp2f_loss
from models import DualBranchFramework, load_pretrained_gcn
from utils.io import read_yaml, write_json, write_yaml
from utils.metrics import split_metrics


class InputAligner(nn.Module):
    """Trainable target-feature projection into the pretrained source feature space."""

    def __init__(self, in_dim: int, out_dim: int, *, style: str = "linear") -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.style = style
        if style == "identity_if_same_dim" and self.in_dim == self.out_dim:
            self.proj: nn.Module = nn.Identity()
        elif style == "official_projector":
            self.proj = nn.Sequential(nn.Linear(self.in_dim, self.out_dim), nn.PReLU())
        elif style in {"linear", "identity_if_same_dim"}:
            linear = nn.Linear(self.in_dim, self.out_dim, bias=False)
            if self.in_dim == self.out_dim and style == "identity_if_same_dim":
                nn.init.eye_(linear.weight)
            else:
                nn.init.orthogonal_(linear.weight)
            self.proj = linear
        else:
            raise ValueError(f"Unsupported input aligner style: {style}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _resolve_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _format_mean_std(values: list[float], *, scale: float = 100.0) -> str:
    mean, std = _mean_std(values)
    return f"{mean * scale:.2f}+-{std * scale:.2f}"


def _environment_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _split_counts(labels: torch.Tensor, masks: dict[str, torch.Tensor], num_classes: int) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for split_name, mask in masks.items():
        counts[split_name] = {
            "total": int(mask.sum().item()),
            "per_class": [
                int(((labels == class_id) & mask).sum().item())
                for class_id in range(int(num_classes))
            ],
        }
    return counts


def _adapter_stats(model: DualBranchFramework) -> dict[str, float]:
    beta_values = [float(adapter.beta.detach().item()) for adapter in model.adapters if hasattr(adapter, "beta")]
    return {
        "adapter_beta_mean": float(np.mean(beta_values)) if beta_values else 0.0,
        "adapter_beta_std": float(np.std(beta_values, ddof=1)) if len(beta_values) > 1 else 0.0,
    }


def _model_variant(model_cfg: dict[str, Any]) -> str:
    if (
        str(model_cfg.get("input_aligner", "")) == "official_projector"
        and str(model_cfg.get("adapter_style", "")) == "official_gp2f"
        and str(model_cfg.get("fusion_alpha_style", "")) == "raw"
    ):
        return "OfficialStyleGP2F"
    return "StableDualBranch"


def _save_checkpoint(
    path: Path,
    *,
    model: DualBranchFramework,
    input_aligner: InputAligner,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_aligner": input_aligner.state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def _load_checkpoint(path: Path, *, model: DualBranchFramework, input_aligner: InputAligner) -> dict[str, Any]:
    payload = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(payload["model"])
    input_aligner.load_state_dict(payload["input_aligner"])
    return dict(payload)


@torch.no_grad()
def evaluate(
    *,
    model: DualBranchFramework,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    input_aligner.eval()
    aligned_x = input_aligner(x)
    logits, h_pre, h_adp, _, alpha = model(aligned_x, edge_index)
    branch_cosine = torch.nn.functional.cosine_similarity(h_pre, h_adp, dim=-1).mean()

    train = split_metrics(logits, labels, train_mask, num_classes=num_classes)
    val = split_metrics(logits, labels, val_mask, num_classes=num_classes)
    test = split_metrics(logits, labels, test_mask, num_classes=num_classes)
    return {
        "train_acc": train["acc"],
        "train_macro_f1": train["macro_f1"],
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "alpha": float(alpha.detach().item()),
        "branch_cosine": float(branch_cosine.detach().item()),
    }


def run_single(
    config: dict[str, Any],
    *,
    repo_root: Path,
    run_index: int = 0,
    total_runs: int = 1,
    run_group_dir: Path | None = None,
) -> dict[str, Any]:
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    pretrained_cfg = config.get("pretrained", {})
    model_cfg = config.get("model", {})
    loss_cfg_raw = config.get("loss", {})
    training_cfg = config.get("training", {})

    seed = int(experiment_cfg.get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_dataset = str(experiment_cfg.get("target_dataset", "Cora"))
    data_root = _resolve_path(data_cfg.get("root", "datasets"), base_dir=repo_root)
    loaded = load_node_dataset(
        target_dataset,
        data_root,
        download_if_missing=bool(data_cfg.get("download_if_missing", False)),
    )
    graph = loaded.data.to(device)
    split = build_few_shot_split(
        graph.y,
        shots=int(data_cfg.get("shots", experiment_cfg.get("shots", 5))),
        val_per_class=int(data_cfg.get("val_per_class", experiment_cfg.get("val_per_class", 30))),
        seed=seed,
    )

    checkpoint_path = _resolve_path(
        pretrained_cfg.get("checkpoint_path", "checkpoints/cora_grace.pt"),
        base_dir=repo_root,
    )
    backbone = load_pretrained_gcn(
        checkpoint_path,
        device=device,
        freeze=bool(pretrained_cfg.get("freeze_backbone", True)),
    )
    source_dim = int(backbone.convs[0].lin.weight.shape[1])
    hidden_dim = int(backbone.convs[0].lin.weight.shape[0])

    aligner_style = str(model_cfg.get("input_aligner", "linear"))
    input_aligner = InputAligner(loaded.num_features, source_dim, style=aligner_style).to(device)
    model = DualBranchFramework(
        backbone,
        hidden_dim=hidden_dim,
        num_classes=loaded.num_classes,
        adapter_bottleneck_dim=int(model_cfg.get("adapter_bottleneck_dim", 16)),
        adapter_beta_init=float(model_cfg.get("adapter_beta_init", 0.01)),
        adapter_alpha_init=float(model_cfg.get("adapter_alpha_init", 0.1)),
        adapter_style=str(model_cfg.get("adapter_style", "stable_zero_init")),
        alpha_init=float(model_cfg.get("alpha_init", 0.5)),
        fusion_alpha_style=str(model_cfg.get("fusion_alpha_style", "sigmoid")),
    ).to(device)

    loss_cfg = GP2FLossConfig(
        use_cls=bool(loss_cfg_raw.get("use_cls", True)),
        use_original_contrastive=bool(loss_cfg_raw.get("use_original_contrastive", False)),
        use_original_topology_fusion=bool(loss_cfg_raw.get("use_original_topology_fusion", False)),
        lambda_ctr=float(loss_cfg_raw.get("lambda_ctr", 0.0)),
        lambda_fus=float(loss_cfg_raw.get("lambda_fus", 0.0)),
        tau_ctr=float(loss_cfg_raw.get("tau_ctr", 0.5)),
        tau_fus=float(loss_cfg_raw.get("tau_fus", 0.05)),
        topology_percentile=float(loss_cfg_raw.get("topology_percentile", 70.0)),
        dense_contrastive_max_nodes=int(loss_cfg_raw.get("dense_contrastive_max_nodes", 5000)),
        contrastive_sample_size=int(loss_cfg_raw.get("contrastive_sample_size", 2048)),
        dense_topology_max_nodes=int(loss_cfg_raw.get("dense_topology_max_nodes", 5000)),
        topology_sample_size=int(loss_cfg_raw.get("topology_sample_size", 20000)),
    )

    trainable_params = list(input_aligner.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
    )

    output_root = _resolve_path(training_cfg.get("output_dir", "outputs/gp2f_baseline"), base_dir=repo_root)
    if run_group_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_group_dir = output_root / loaded.name / timestamp
    run_dir = run_group_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config)

    epochs = int(training_cfg.get("epochs", 200))
    log_every = int(training_cfg.get("log_every", 20))
    best_val = -1.0
    best_metrics: dict[str, float] = {}
    loss_curve: list[dict[str, float]] = []
    monitor = str(training_cfg.get("early_stop_metric", "val_acc"))
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_epochs = int(training_cfg.get("early_stop_min_epochs", 0))
    keep_checkpoint = bool(training_cfg.get("keep_best_checkpoint", True))
    best_checkpoint_path = run_dir / "best_model.pt"
    epochs_no_improve = 0
    stopped_epoch = epochs
    early_stopped = False

    run_label = f"{loaded.name} seed={seed}"
    if total_runs > 1:
        run_label = f"{run_label} ({run_index + 1}/{total_runs})"
    progress = tqdm(
        range(1, epochs + 1),
        desc=run_label,
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in progress:
        model.train()
        input_aligner.train()
        optimizer.zero_grad()

        aligned_x = input_aligner(graph.x)
        logits, h_pre, h_adp, h_mix, alpha = model(aligned_x, graph.edge_index)
        loss_out = compute_gp2f_loss(
            logits=logits,
            labels=graph.y,
            train_mask=split.train_mask,
            h_pre=h_pre,
            h_adp=h_adp,
            h_mix=h_mix,
            alpha=alpha,
            edge_index=graph.edge_index,
            cfg=loss_cfg,
        )
        if not torch.isfinite(loss_out.total):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss_out.total.item()}")
        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, float(training_cfg.get("grad_clip", 1.0)))
        optimizer.step()

        log_item = {
            "epoch": float(epoch),
            **loss_out.to_log_dict(),
            **loss_out.to_metadata_dict(),
            "alpha": float(alpha.detach().item()),
        }
        loss_curve.append(log_item)

        metrics = evaluate(
            model=model,
            input_aligner=input_aligner,
            x=graph.x,
            edge_index=graph.edge_index,
            labels=graph.y,
            train_mask=split.train_mask,
            val_mask=split.val_mask,
            test_mask=split.test_mask,
            num_classes=loaded.num_classes,
        )
        monitor_lookup = {**metrics, **loss_out.to_log_dict(), "train_loss": float(loss_out.total.detach().item())}
        monitor_value = float(monitor_lookup.get(monitor, metrics["val_acc"]))
        is_improved = monitor_value < best_val if monitor in {"train_loss", "total", "cls"} else monitor_value > best_val
        if is_improved or not best_metrics:
            best_val = monitor_value
            best_metrics = {
                "best_epoch": float(epoch),
                "monitor_value": monitor_value,
                **metrics,
                **loss_out.to_log_dict(),
                **loss_out.to_metadata_dict(),
                **_adapter_stats(model),
            }
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                input_aligner=input_aligner,
                epoch=epoch,
                metrics=best_metrics,
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            progress.set_postfix(
                {
                    "loss": f"{loss_out.total.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "alpha": f"{metrics['alpha']:.3f}",
                    "best": f"{best_val:.4f}",
                }
            )
        if patience > 0 and epoch >= min_epochs and epochs_no_improve >= patience:
            stopped_epoch = epoch
            early_stopped = True
            progress.set_postfix(
                {
                    "loss": f"{loss_out.total.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "alpha": f"{metrics['alpha']:.3f}",
                    "stop": "early",
                }
            )
            break

    final_metrics = evaluate(
        model=model,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        labels=graph.y,
        train_mask=split.train_mask,
        val_mask=split.val_mask,
        test_mask=split.test_mask,
        num_classes=loaded.num_classes,
    )
    result = {
        "dataset": loaded.name,
        "seed": seed,
        "split_seed": split.seed,
        "num_nodes": int(graph.num_nodes),
        "num_features": loaded.num_features,
        "num_classes": loaded.num_classes,
        "checkpoint_path": str(checkpoint_path),
        "source_dim": source_dim,
        "hidden_dim": hidden_dim,
        "aligner_style": aligner_style,
        "adapter_style": str(model_cfg.get("adapter_style", "stable_zero_init")),
        "fusion_alpha_style": str(model_cfg.get("fusion_alpha_style", "sigmoid")),
        "model_variant": _model_variant(model_cfg),
        "final": final_metrics,
        "best": best_metrics,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "early_stop_metric": monitor,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() and keep_checkpoint else "",
        "split_counts": _split_counts(
            graph.y,
            {
                "train": split.train_mask,
                "val": split.val_mask,
                "test": split.test_mask,
            },
            loaded.num_classes,
        ),
        "environment": _environment_info(),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "loss_curve.json", {"loss_curve": loss_curve})
    if best_checkpoint_path.exists() and not keep_checkpoint:
        best_checkpoint_path.unlink()
    print(f"Saved results to {run_dir}")
    return result


def _resolve_run_seeds(config: dict[str, Any]) -> list[int]:
    experiment_cfg = config.get("experiment", {})
    if "seeds" in experiment_cfg and experiment_cfg["seeds"] is not None:
        seeds = experiment_cfg["seeds"]
        if not isinstance(seeds, list):
            raise ValueError("experiment.seeds must be a list of integers")
        return [int(seed) for seed in seeds]

    base_seed = int(experiment_cfg.get("seed", 0))
    runs = int(experiment_cfg.get("runs", 1))
    return [base_seed + offset for offset in range(max(1, runs))]


def run(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    seeds = _resolve_run_seeds(config)
    output_root = _resolve_path(
        config.get("training", {}).get("output_dir", "outputs/gp2f_baseline"),
        base_dir=repo_root,
    )
    target_dataset = str(config.get("experiment", {}).get("target_dataset", "Cora"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = output_root / target_dataset / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        seed_config = _deep_update(config, {"experiment": {"seed": seed}})
        run_result = run_single(
            seed_config,
            repo_root=repo_root,
            run_index=idx,
            total_runs=len(seeds),
            run_group_dir=summary_dir,
        )
        results.append(run_result)

    best_test = [float(result["best"].get("test_acc", 0.0)) for result in results]
    best_macro_f1 = [float(result["best"].get("test_macro_f1", 0.0)) for result in results]
    final_test = [float(result["final"].get("test_acc", 0.0)) for result in results]
    final_macro_f1 = [float(result["final"].get("test_macro_f1", 0.0)) for result in results]

    summary = {
        "dataset": target_dataset,
        "preset": str(config.get("experiment", {}).get("preset", "")),
        "model_variant": results[0].get("model_variant", "") if results else "",
        "seeds": seeds,
        "num_runs": len(seeds),
        "best_test_acc_mean_std": _format_mean_std(best_test),
        "best_test_macro_f1_mean_std": _format_mean_std(best_macro_f1),
        "final_test_acc_mean_std": _format_mean_std(final_test),
        "final_test_macro_f1_mean_std": _format_mean_std(final_macro_f1),
        "runs": results,
        "environment": _environment_info(),
    }
    write_json(summary_dir / "summary.json", summary)
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "best_epoch",
            "best_test_acc",
            "best_test_macro_f1",
            "final_test_acc",
            "final_test_macro_f1",
            "alpha",
            "branch_cosine",
            "contrastive_mode",
            "topology_mode",
            "model_variant",
            "early_stopped",
            "stopped_epoch",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "best_epoch": result["best"].get("best_epoch", 0),
                    "best_test_acc": result["best"].get("test_acc", 0.0),
                    "best_test_macro_f1": result["best"].get("test_macro_f1", 0.0),
                    "final_test_acc": result["final"].get("test_acc", 0.0),
                    "final_test_macro_f1": result["final"].get("test_macro_f1", 0.0),
                    "alpha": result["best"].get("alpha", 0.0),
                    "branch_cosine": result["best"].get("branch_cosine", 0.0),
                    "contrastive_mode": result["best"].get("contrastive_mode", "disabled"),
                    "topology_mode": result["best"].get("topology_mode", "disabled"),
                    "model_variant": result.get("model_variant", ""),
                    "early_stopped": result.get("early_stopped", False),
                    "stopped_epoch": result.get("stopped_epoch", 0),
                }
            )
    print("=" * 72)
    print(f"Summary | dataset={target_dataset} | runs={len(seeds)}")
    print(f"Best Test Acc:      {summary['best_test_acc_mean_std']}")
    print(f"Best Test Macro-F1: {summary['best_test_macro_f1_mean_std']}")
    print(f"Final Test Acc:     {summary['final_test_acc_mean_std']}")
    print(f"Final Test Macro-F1: {summary['final_test_macro_f1_mean_std']}")
    print(f"Saved summary to {summary_dir / 'summary.json'}")
    return summary


PRESETS: dict[str, dict[str, Any]] = {
    "B0": {
        "loss": {
            "use_original_contrastive": False,
            "use_original_topology_fusion": False,
            "lambda_ctr": 0.0,
            "lambda_fus": 0.0,
        }
    },
    "B1": {
        "loss": {
            "use_original_contrastive": True,
            "use_original_topology_fusion": False,
            "lambda_ctr": 0.05,
            "lambda_fus": 0.0,
        }
    },
    "B2": {
        "loss": {
            "use_original_contrastive": False,
            "use_original_topology_fusion": True,
            "lambda_ctr": 0.0,
            "lambda_fus": 0.1,
        }
    },
    "B3": {
        "loss": {
            "use_original_contrastive": True,
            "use_original_topology_fusion": True,
            "lambda_ctr": 0.05,
            "lambda_fus": 0.1,
        }
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful GP2F baseline runner")
    parser.add_argument("--config", type=str, default="configs/gp2f_baseline.yaml")
    parser.add_argument("--style_config", type=str, default=None)
    parser.add_argument("--target_dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 0,1,2,3,4")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--preset", type=str, choices=sorted(PRESETS), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(args.config, base_dir=repo_root)
    config = read_yaml(config_path)
    if args.style_config is not None:
        style_path = _resolve_path(args.style_config, base_dir=repo_root)
        config = _deep_update(config, read_yaml(style_path))
    if args.preset is not None:
        config = _deep_update(config, PRESETS[args.preset])
        config = _deep_update(config, {"experiment": {"preset": args.preset}})
    overrides: dict[str, Any] = {}
    if args.target_dataset is not None:
        overrides.setdefault("experiment", {})["target_dataset"] = args.target_dataset
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed
    if args.seeds is not None:
        overrides.setdefault("experiment", {})["seeds"] = [
            int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()
        ]
    if args.runs is not None:
        overrides.setdefault("experiment", {})["runs"] = args.runs
        overrides.setdefault("experiment", {})["seeds"] = None
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.weight_decay is not None:
        overrides.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.patience is not None:
        overrides.setdefault("training", {})["early_stop_patience"] = args.patience
    if args.output_dir is not None:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    config = _deep_update(config, overrides)
    run(config, repo_root=repo_root)


if __name__ == "__main__":
    main()
