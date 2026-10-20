"""Out-of-fold counterfactual utility supervision for P24 routing."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class P24CrossFittedUtilityLearner(nn.Module):
    """Accumulate held-out utility targets and supervise the P24 router.

    The module owns no predictor. It trains the existing counterfactual router
    from rotating held-out folds and automatically disables label supervision
    when any class has too few examples.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.num_folds = max(2, int(self.config.get("num_folds", 5)))
        self.min_train_per_class = max(1, int(self.config.get("min_train_per_class", 20)))
        self.ema_decay = float(self.config.get("ema_decay", 0.90))
        self.gain_cap = float(self.config.get("gain_cap", 0.02))
        self.gain_margin = float(self.config.get("gain_margin", 0.0005))
        self.harm_weight = float(self.config.get("harm_weight", 2.0))
        self.rank_weight = float(self.config.get("rank_weight", 0.10))
        self.budget_weight = float(self.config.get("budget_weight", 0.02))
        self.positive_adapt_floor = float(self.config.get("positive_adapt_floor", 0.05))
        self.class_balanced = bool(self.config.get("class_balanced", True))
        self.fallback = str(self.config.get("fallback", "unsupervised_stability"))
        self.register_buffer("gain_ema", torch.empty(0), persistent=False)

        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("crossfit_utility.ema_decay must be in [0, 1)")
        if self.gain_cap <= 0.0:
            raise ValueError("crossfit_utility.gain_cap must be positive")

    def eligibility(
        self, labels: torch.Tensor, train_mask: torch.Tensor
    ) -> tuple[bool, int, int]:
        train = train_mask.to(device=labels.device, dtype=torch.bool)
        if not self.enabled or not bool(train.any()):
            return False, 0, 0
        expected_class_count = int(torch.unique(labels.long()).numel())
        counts = torch.bincount(labels[train].long())
        present_counts = counts[counts > 0]
        if present_counts.numel() == 0:
            return False, 0, 0
        all_classes_present = int(present_counts.numel()) == expected_class_count
        minimum = int(present_counts.min().item()) if all_classes_present else 0
        return (
            all_classes_present and minimum >= self.min_train_per_class,
            minimum,
            int(present_counts.numel()),
        )

    def _ensure_state(self, reference: torch.Tensor, num_nodes: int) -> None:
        if (
            self.gain_ema.numel() != num_nodes
            or self.gain_ema.device != reference.device
            or self.gain_ema.dtype != reference.dtype
        ):
            self.gain_ema = torch.full(
                (num_nodes,), float("nan"), device=reference.device, dtype=reference.dtype
            )

    @staticmethod
    def _zero_stats(*, eligible: bool, minimum: int, class_count: int) -> dict[str, float]:
        return {
            "p24_counterfactual_route_loss": 0.0,
            "p24_counterfactual_route_count": 0.0,
            "p24_counterfactual_true_gain_mean": 0.0,
            "p24_counterfactual_gain_corr": 0.0,
            "p24_counterfactual_harm_loss": 0.0,
            "p24_counterfactual_rank_loss": 0.0,
            "p24_counterfactual_budget_loss": 0.0,
            "p24_crossfit_enabled": 1.0,
            "p24_crossfit_eligible": float(eligible),
            "p24_crossfit_min_train_per_class": float(minimum),
            "p24_crossfit_class_count": float(class_count),
            "p24_crossfit_target_coverage": 0.0,
        }

    def loss(
        self,
        *,
        static_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        predicted_gain: torch.Tensor,
        adapt_ratio: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        eligible, minimum, class_count = self.eligibility(labels, train_mask)
        zero = predicted_gain.sum() * 0.0
        if not eligible:
            return zero, self._zero_stats(
                eligible=False, minimum=minimum, class_count=class_count
            )

        device = predicted_gain.device
        y = labels.to(device=device, dtype=torch.long)
        train = train_mask.to(device=device, dtype=torch.bool)
        query = query_mask.to(device=device, dtype=torch.bool) & train
        if not bool(query.any()):
            return zero, self._zero_stats(
                eligible=True, minimum=minimum, class_count=class_count
            )

        self._ensure_state(predicted_gain, int(predicted_gain.numel()))
        with torch.no_grad():
            static_ce = F.cross_entropy(static_logits.detach(), y, reduction="none")
            candidate_ce = F.cross_entropy(candidate_logits.detach(), y, reduction="none")
            current_gain = (static_ce - candidate_ce).clamp(-self.gain_cap, self.gain_cap)
            old = self.gain_ema[query]
            updated = torch.where(
                torch.isfinite(old),
                self.ema_decay * old + (1.0 - self.ema_decay) * current_gain[query],
                current_gain[query],
            )
            self.gain_ema[query] = updated
            target = self.gain_ema.detach()

        target_for_loss = torch.where(torch.isfinite(target), target, predicted_gain.detach())
        per_node = F.smooth_l1_loss(predicted_gain, target_for_loss, reduction="none")
        if self.class_balanced:
            parts = [
                per_node[query & (y == class_id)].mean()
                for class_id in torch.unique(y[query])
            ]
            regression = torch.stack(parts).mean() if parts else per_node[query].mean()
        else:
            regression = per_node[query].mean()

        positive = query & (target > self.gain_margin)
        harmful = query & (target < -self.gain_margin)
        harm_loss = F.relu(predicted_gain[harmful]).mean() if bool(harmful.any()) else zero
        if bool(positive.any()) and bool(harmful.any()):
            rank_loss = F.relu(
                predicted_gain.new_tensor(self.gain_margin)
                - (predicted_gain[positive].mean() - predicted_gain[harmful].mean())
            )
        else:
            rank_loss = zero
        budget_loss = (
            F.relu(adapt_ratio.new_tensor(self.positive_adapt_floor) - adapt_ratio[positive].mean())
            if bool(positive.any())
            else zero
        )
        total = (
            regression
            + self.harm_weight * harm_loss
            + self.rank_weight * rank_loss
            + self.budget_weight * budget_loss
        )
        with torch.no_grad():
            pred = predicted_gain[query]
            truth = target[query]
            pred_centered = pred - pred.mean()
            truth_centered = truth - truth.mean()
            corr = (pred_centered * truth_centered).sum() / (
                pred_centered.norm() * truth_centered.norm()
            ).clamp_min(1e-12)
            coverage = torch.isfinite(target[train]).float().mean()
        return total, {
            "p24_counterfactual_route_loss": float(total.detach().item()),
            "p24_counterfactual_route_count": float(query.sum().item()),
            "p24_counterfactual_true_gain_mean": float(target[query].mean().item()),
            "p24_counterfactual_true_gain_std": float(target[query].std(unbiased=False).item()),
            "p24_counterfactual_positive_ratio": float(positive[query].float().mean().item()),
            "p24_counterfactual_harmful_ratio": float(harmful[query].float().mean().item()),
            "p24_counterfactual_gain_corr": float(corr.item()),
            "p24_counterfactual_harm_loss": float(harm_loss.detach().item()),
            "p24_counterfactual_rank_loss": float(rank_loss.detach().item()),
            "p24_counterfactual_budget_loss": float(budget_loss.detach().item()),
            "p24_counterfactual_positive_adapt_mean": float(
                adapt_ratio[positive].detach().mean().item() if bool(positive.any()) else 0.0
            ),
            "p24_counterfactual_harmful_adapt_mean": float(
                adapt_ratio[harmful].detach().mean().item() if bool(harmful.any()) else 0.0
            ),
            "p24_crossfit_enabled": 1.0,
            "p24_crossfit_eligible": 1.0,
            "p24_crossfit_min_train_per_class": float(minimum),
            "p24_crossfit_class_count": float(class_count),
            "p24_crossfit_target_coverage": float(coverage.item()),
        }
