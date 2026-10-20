"""Metrics for node classification experiments."""

from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((pred[mask] == labels[mask]).float().mean().item())


def macro_f1(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, num_classes: int) -> float:
    if not bool(mask.any()):
        return 0.0
    pred_m = pred[mask]
    label_m = labels[mask]
    scores: list[float] = []
    for class_id in range(int(num_classes)):
        true_c = label_m == class_id
        pred_c = pred_m == class_id
        support = true_c.sum()
        if int(support.item()) == 0:
            continue
        tp = (true_c & pred_c).sum().float()
        precision = tp / pred_c.sum().clamp_min(1).float()
        recall = tp / support.clamp_min(1).float()
        denom = precision + recall
        f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, denom)
        scores.append(float(f1.item()))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _binary_auroc(scores: torch.Tensor, target: torch.Tensor) -> float:
    positives = int(target.sum().item())
    negatives = int((~target).sum().item())
    if positives == 0 or negatives == 0:
        return 0.0
    order = torch.argsort(scores, descending=False)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores, dtype=torch.float32)
    start = 0
    rank_value = 1.0
    while start < int(scores.numel()):
        end = start + 1
        while end < int(scores.numel()) and bool(sorted_scores[end] == sorted_scores[start]):
            end += 1
        avg_rank = (rank_value + float(end)) / 2.0
        ranks[order[start:end]] = avg_rank
        rank_value = float(end + 1)
        start = end
    rank_sum_pos = ranks[target].sum()
    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / max(1, positives * negatives)
    return float(auc.clamp(0.0, 1.0).item())


def _binary_average_precision(scores: torch.Tensor, target: torch.Tensor) -> float:
    positives = int(target.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_target = target[order].float()
    tp = torch.cumsum(sorted_target, dim=0)
    rank = torch.arange(1, int(sorted_target.numel()) + 1, device=scores.device, dtype=torch.float32)
    precision = tp / rank
    ap = (precision * sorted_target).sum() / float(positives)
    return float(ap.clamp(0.0, 1.0).item())


def macro_auroc_auprc(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, num_classes: int) -> tuple[float, float]:
    if not bool(mask.any()):
        return 0.0, 0.0
    logits_m = logits[mask].detach()
    labels_m = labels[mask].detach()
    prob = torch.softmax(logits_m, dim=-1)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for class_id in range(int(num_classes)):
        target = labels_m == class_id
        if int(target.sum().item()) == 0 or int((~target).sum().item()) == 0:
            continue
        scores = prob[:, class_id]
        aurocs.append(_binary_auroc(scores, target))
        auprcs.append(_binary_average_precision(scores, target))
    auroc = float(sum(aurocs) / len(aurocs)) if aurocs else 0.0
    auprc = float(sum(auprcs) / len(auprcs)) if auprcs else 0.0
    return auroc, auprc


def split_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    auroc, auprc = macro_auroc_auprc(logits, labels, mask, num_classes)
    return {
        "acc": accuracy(pred, labels, mask),
        "macro_f1": macro_f1(pred, labels, mask, num_classes),
        "auroc": auroc,
        "auprc": auprc,
    }
