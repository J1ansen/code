"""Few-shot split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class FewShotSplit:
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    seed: int


def load_gp2f_precomputed_split(
    labels: torch.Tensor,
    *,
    sample_data_root: str | Path,
    dataset: str,
    shots: int,
    trial_index: int,
    seed: int,
    unused_as_validation: bool = True,
) -> FewShotSplit:
    """Load one of GP2F's saved few-shot tasks without changing its train/test sets.

    GP2F numbers tasks from 1, whereas this repository numbers trials from 0.
    Nodes omitted by GP2F can optionally form a diagnostics-only validation mask;
    callers must use a non-validation checkpoint criterion for protocol parity.
    """

    y = labels.view(-1)
    device = y.device
    num_nodes = int(y.numel())
    task_dir = (
        Path(sample_data_root).expanduser()
        / str(dataset)
        / f"{int(shots)}_shot"
        / str(int(trial_index) + 1)
    )
    required = {
        "train_idx": task_dir / "train_idx.pt",
        "train_labels": task_dir / "train_labels.pt",
        "test_idx": task_dir / "test_idx.pt",
        "test_labels": task_dir / "test_labels.pt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing GP2F precomputed split files: " + ", ".join(missing)
        )

    train_idx = torch.load(required["train_idx"], map_location="cpu", weights_only=False).long().view(-1)
    test_idx = torch.load(required["test_idx"], map_location="cpu", weights_only=False).long().view(-1)
    saved_train_labels = torch.load(
        required["train_labels"], map_location="cpu", weights_only=False
    ).long().view(-1)
    saved_test_labels = torch.load(
        required["test_labels"], map_location="cpu", weights_only=False
    ).long().view(-1)

    for name, idx in (("train", train_idx), ("test", test_idx)):
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= num_nodes):
            raise ValueError(f"GP2F {name} indices are out of range for {num_nodes} nodes")
        if torch.unique(idx).numel() != idx.numel():
            raise ValueError(f"GP2F {name} indices contain duplicates in {task_dir}")
    if torch.isin(train_idx, test_idx).any():
        raise ValueError(f"GP2F train/test indices overlap in {task_dir}")

    labels_cpu = y.detach().cpu().long()
    if not torch.equal(labels_cpu[train_idx], saved_train_labels):
        raise ValueError(f"GP2F train labels do not match the loaded {dataset} graph")
    if not torch.equal(labels_cpu[test_idx], saved_test_labels):
        raise ValueError(f"GP2F test labels do not match the loaded {dataset} graph")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx.to(device)] = True
    test_mask[test_idx.to(device)] = True
    if unused_as_validation:
        val_mask = ~(train_mask | test_mask)
    else:
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    return FewShotSplit(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=int(seed),
    )


def build_few_shot_split(
    labels: torch.Tensor,
    *,
    shots: int,
    val_per_class: int,
    seed: int,
    shot_ratio: float | None = None,
    test_ratio: float | None = None,
) -> FewShotSplit:
    """Build class-balanced train/validation masks and use remaining nodes for test."""

    y = labels.view(-1)
    device = y.device
    num_nodes = int(y.numel())
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    rng = np.random.default_rng(int(seed))
    num_classes = int(y.max().item()) + 1
    if test_ratio is not None:
        ratio = float(test_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError("test_ratio must be in (0, 1)")
        permutation = rng.permutation(num_nodes)
        test_count = int(ratio * num_nodes)
        test_idx = permutation[:test_count]
        remaining_idx = permutation[test_count:]
        labels_cpu = y.detach().cpu().numpy()
        test_mask[torch.as_tensor(test_idx, dtype=torch.long, device=device)] = True

        for class_id in range(num_classes):
            idx = remaining_idx[labels_cpu[remaining_idx] == class_id]
            if shot_ratio is None:
                class_shots = int(shots)
            else:
                full_class_size = int((labels_cpu == class_id).sum())
                class_shots = max(1, int(np.ceil(float(shot_ratio) * full_class_size)))
            class_shots = min(class_shots, int(idx.size))
            train_idx = idx[:class_shots]
            val_idx = idx[class_shots:]
            if train_idx.size:
                train_mask[torch.as_tensor(train_idx, dtype=torch.long, device=device)] = True
            if val_idx.size:
                val_mask[torch.as_tensor(val_idx, dtype=torch.long, device=device)] = True
        return FewShotSplit(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            seed=int(seed),
        )

    for class_id in range(num_classes):
        idx = torch.where(y == class_id)[0].detach().cpu().numpy()
        rng.shuffle(idx)
        if shot_ratio is None:
            class_shots = int(shots)
        else:
            class_shots = int(np.ceil(float(shot_ratio) * int(idx.size)))
            class_shots = max(1, class_shots)
        class_shots = min(class_shots, int(idx.size))
        train_idx = idx[:class_shots]
        val_start = class_shots
        val_end = val_start + int(val_per_class)
        val_idx = idx[val_start:val_end]
        test_idx = idx[val_end:]

        if train_idx.size:
            train_mask[torch.as_tensor(train_idx, dtype=torch.long, device=device)] = True
        if val_idx.size:
            val_mask[torch.as_tensor(val_idx, dtype=torch.long, device=device)] = True
        if test_idx.size:
            test_mask[torch.as_tensor(test_idx, dtype=torch.long, device=device)] = True

    return FewShotSplit(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, seed=int(seed))
