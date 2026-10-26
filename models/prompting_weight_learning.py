"""Prompting Weight Learning for candidate prompt-to-node connections."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(clipped)).item())


class PromptingWeightLearning(nn.Module):
    """Learn bounded propagation adjustments over candidate prompt edges.

    The module never creates new edges.  It only predicts multiplicative gates
    for the constructed prompt graph, with the default gate initialized near
    one so enabling the layer starts close to the static graph.
    """

    def __init__(self, source_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.descriptor_type = str(self.config.get("descriptor_type", "statistics"))
        if self.descriptor_type not in {"statistics", "representation_concat", "raw_representation_concat"}:
            raise ValueError("Unknown Prompting descriptor_type: " + self.descriptor_type)
        self.enabled = bool(self.config.get("enabled", False))
        self.allow_new_edges = bool(self.config.get("allow_new_edges", False))
        if self.allow_new_edges:
            raise ValueError("Prompting v1 only denoises existing PromptGraph edges; allow_new_edges must be false")
        self.apply_to = str(self.config.get("apply_to", "prompt_to_node"))
        if self.apply_to not in {"prompt_to_node", "both"}:
            raise ValueError("prompt_edge_denoising.apply_to must be 'prompt_to_node' or 'both'")
        self.gate_floor = float(self.config.get("gate_floor", 0.20))
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("prompt_edge_denoising.gate_floor must be in [0, 1)")
        self.gate_cap = float(self.config.get("gate_cap", 1.0))
        if self.gate_cap <= self.gate_floor:
            raise ValueError("prompt_edge_denoising.gate_cap must be greater than gate_floor")
        self.gate_init = float(self.config.get("gate_init", 0.95))
        if not self.gate_floor < self.gate_init <= self.gate_cap:
            raise ValueError("prompt_edge_denoising.gate_init must be in (gate_floor, gate_cap]")
        self.fusion = str(self.config.get("fusion", "direct"))
        if self.fusion not in {"direct", "bootstrapped"}:
            raise ValueError("prompt_edge_denoising.fusion must be 'direct' or 'bootstrapped'")
        self.mix_ratio_init = float(self.config.get("mix_ratio_init", 1.0 if self.fusion == "direct" else 0.0))
        self.mix_ratio_max = float(self.config.get("mix_ratio_max", 1.0 if self.fusion == "direct" else 0.20))
        self.mix_warmup_epochs = int(self.config.get("mix_warmup_epochs", 0 if self.fusion == "direct" else 50))
        if self.mix_ratio_init < 0.0 or self.mix_ratio_max < 0.0:
            raise ValueError("prompt_edge_denoising mix ratios must be non-negative")
        self.detach_node_repr = bool(self.config.get("detach_node_repr", True))
        self.detach_prompt_repr = bool(self.config.get("detach_prompt_repr", True))
        self.hub_quantile = float(self.config.get("hub_quantile", 0.80))
        self.homophily_keep_weight = float(self.config.get("homophily_keep_weight", 0.25))
        self.keep_threshold = float(self.config.get("keep_threshold", 0.80))

        input_dim = 14
        hidden_dim = int(self.config.get("hidden_dim", 16))
        dropout = float(self.config.get("dropout", 0.10))
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.edge_gate[-1].weight)
        normalized_init = (self.gate_init - self.gate_floor) / max(1e-6, self.gate_cap - self.gate_floor)
        nn.init.constant_(self.edge_gate[-1].bias, _logit(normalized_init))

        self.representation_projection: nn.Module | None = None
        self.representation_gate: nn.Module | None = None
        if self.descriptor_type == "representation_concat":
            if self.apply_to != "prompt_to_node":
                raise ValueError("representation_concat requires apply_to='prompt_to_node'")
            projection_dim = int(self.config.get("projection_dim", 16))
            if source_dim <= 0 or projection_dim <= 0:
                raise ValueError("source_dim and projection_dim must be positive")
            if not 0.0 <= self.mix_ratio_init <= self.mix_ratio_max <= 1.0:
                raise ValueError("representation_concat requires 0 <= mix_ratio_init <= mix_ratio_max <= 1")
            # PromptGraph constructs this module before its own gates. Keep its RNG stream
            # and legacy state-dict keys unchanged when adding the new networks.
            with torch.random.fork_rng(devices=[]):
                self.representation_projection = nn.Sequential(
                    nn.Linear(source_dim, projection_dim), nn.LayerNorm(projection_dim)
                )
                self.representation_gate = nn.Sequential(
                    nn.Linear(2 * projection_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.representation_gate[-1].weight)
                nn.init.constant_(self.representation_gate[-1].bias, _logit(normalized_init))
            self.edge_gate.requires_grad_(False)

        self.raw_representation_gate: nn.Module | None = None
        if self.descriptor_type == "raw_representation_concat":
            if self.apply_to != "prompt_to_node":
                raise ValueError("raw_representation_concat requires apply_to='prompt_to_node'")
            if source_dim <= 0 or hidden_dim <= 0:
                raise ValueError("source_dim and hidden_dim must be positive")
            if not 0.0 <= self.mix_ratio_init <= self.mix_ratio_max <= 1.0:
                raise ValueError("raw_representation_concat requires 0 <= mix_ratio_init <= mix_ratio_max <= 1")
            with torch.random.fork_rng(devices=[]):
                self.raw_representation_gate = nn.Sequential(
                    nn.Linear(2 * source_dim, hidden_dim),
                    nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.raw_representation_gate[-1].weight)
                nn.init.constant_(self.raw_representation_gate[-1].bias, _logit(normalized_init))
            self.edge_gate.requires_grad_(False)

        self.independent_edge_weights: nn.Module | None = None
        self.calibrator_grad_scale = float(self.config.get("calibrator_grad_scale", 1.0))
        if self.calibrator_grad_scale <= 0.0:
            raise ValueError("prompt_edge_denoising.calibrator_grad_scale must be positive")
        if self.calibrator_grad_scale != 1.0:
            if self.descriptor_type == "representation_concat":
                calibrator_modules = (self.representation_projection, self.representation_gate)
            elif self.descriptor_type == "raw_representation_concat":
                calibrator_modules = (self.raw_representation_gate,)
            elif self.descriptor_type == "statistics":
                calibrator_modules = (self.edge_gate,)
            else:
                calibrator_modules = (self.independent_edge_weights,)
            for module in calibrator_modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.register_hook(
                        lambda gradient, scale=self.calibrator_grad_scale: gradient * scale
                    )

    def _raw_representation_logits(
        self, node: torch.Tensor, prompt: torch.Tensor,
        node_index: torch.Tensor, prompt_index: torch.Tensor,
    ) -> torch.Tensor:
        first = self.raw_representation_gate[0]
        width = node.size(-1)
        # Exactly Linear([prompt, node]); gather only hidden-width tensors per edge.
        prompt_hidden = F.linear(prompt, first.weight[:, :width], first.bias)
        node_hidden = F.linear(node, first.weight[:, width:])
        hidden = prompt_hidden[prompt_index] + node_hidden[node_index]
        for layer in list(self.raw_representation_gate.children())[1:]:
            hidden = layer(hidden)
        return hidden.squeeze(-1)

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def _identity_output(self, ref: torch.Tensor, edge_count: int) -> dict[str, Any]:
        gate = ref.new_ones(edge_count)
        zero = self._zero(ref)
        one = ref.new_tensor(1.0)
        return {
            "edge_gate": gate,
            "effective_edge_gate": gate,
            "mix_ratio": zero,
            "losses": {
                "keep_static": zero,
                "hub_suppression": zero,
                "prompt_homophily": zero,
                "edge_budget": zero,
            },
            "stats": {
                "prompting_enabled": zero,
                "prompting_edge_gate_mean": gate.mean() if gate.numel() > 0 else zero,
                "prompting_edge_gate_min": gate.min() if gate.numel() > 0 else zero,
                "prompting_edge_gate_max": gate.max() if gate.numel() > 0 else zero,
                "prompting_prompt_to_node_gate_mean": gate.mean() if gate.numel() > 0 else zero,
                "prompting_node_to_prompt_gate_mean": gate.mean() if gate.numel() > 0 else zero,
                "prompting_hub_gate_mean": zero,
                "prompting_nonhub_gate_mean": zero,
                "prompting_edge_keep_ratio": gate.new_tensor(1.0) if gate.numel() > 0 else zero,
                "prompting_homophily_loss": zero,
                "prompting_learned_gate_mean": gate.mean() if gate.numel() > 0 else one,
                "prompting_effective_gate_mean": gate.mean() if gate.numel() > 0 else one,
                "prompting_mix_ratio": zero,
                "prompting_gate_above_one_ratio": zero,
                "prompting_effective_gate_below_0p3_ratio": zero,
                "prompting_effective_gate_below_0p5_ratio": zero,
                "prompting_effective_gate_above_one_ratio": zero,
                "prompting_within_prompt_gate_std_mean": zero,
                "prompting_weight_delta_mean": zero,
                "prompting_weight_delta_abs_mean": zero,
                "prompting_train_edge_homophily_raw": zero,
                "prompting_train_edge_homophily_weighted": zero,
                "prompting_train_same_class_gate_mean": zero,
                "prompting_train_offclass_gate_mean": zero,
            },
        }

    def _apply_mask(self, edge_direction: torch.Tensor) -> torch.Tensor:
        if self.apply_to == "both":
            return torch.ones_like(edge_direction, dtype=torch.bool)
        return edge_direction.to(dtype=torch.long) == 1

    def mix_ratio(self, ref: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        if self.fusion == "direct":
            return ref.new_tensor(1.0)
        current_epoch = 0 if epoch is None else max(0, int(epoch))
        if self.mix_warmup_epochs <= 0:
            progress = 1.0
        else:
            progress = min(1.0, float(current_epoch) / float(self.mix_warmup_epochs))
        value = self.mix_ratio_init + (self.mix_ratio_max - self.mix_ratio_init) * progress
        return ref.new_tensor(max(0.0, value))

    def forward(
        self,
        *,
        edge_node_index: torch.Tensor,
        edge_feature_local: torch.Tensor,
        edge_direction: torch.Tensor,
        base_edge_weight: torch.Tensor,
        feature_stats: torch.Tensor,
        feature_gate: torch.Tensor,
        node_repr: torch.Tensor,
        prompt_repr: torch.Tensor,
        feature_usage: torch.Tensor | None = None,
        feature_dominant_class: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        edge_count = int(edge_node_index.numel())
        if not self.enabled or edge_count == 0:
            return self._identity_output(base_edge_weight, edge_count)

        device = base_edge_weight.device
        dtype = base_edge_weight.dtype
        edge_node = edge_node_index.to(device=device, dtype=torch.long)
        edge_feature = edge_feature_local.to(device=device, dtype=torch.long)
        direction = edge_direction.to(device=device, dtype=torch.long)
        apply_mask = self._apply_mask(direction)
        if not bool(apply_mask.any()):
            return self._identity_output(base_edge_weight, edge_count)

        node = node_repr.to(device=device, dtype=dtype)
        prompt = prompt_repr.to(device=device, dtype=dtype)
        if self.detach_node_repr:
            node = node.detach()
        if self.detach_prompt_repr:
            prompt = prompt.detach()
        usage = (
            feature_usage.to(device=device, dtype=dtype)
            if isinstance(feature_usage, torch.Tensor) and feature_usage.numel() > 0
            else base_edge_weight.new_ones(max(1, int(edge_feature.max().item()) + 1))
        )
        usage_norm = usage / usage.mean().clamp_min(1e-6)
        usage_edge = usage_norm[edge_feature].unsqueeze(-1)
        if usage.numel() > 1:
            threshold = torch.quantile(usage.detach().to(dtype=torch.float32), self.hub_quantile).to(device=device, dtype=dtype)
        else:
            threshold = usage.new_tensor(float("inf"))
        hub_score = (usage / threshold.clamp_min(1.0) - 1.0).clamp_min(0.0)
        hub_score_edge = hub_score[edge_feature].unsqueeze(-1)

        if self.descriptor_type == "independent_edge":
            denoise_gate = self.independent_edge_weights(edge_node[apply_mask], edge_feature[apply_mask])
            edge_gate = base_edge_weight.new_ones(edge_count).masked_scatter(apply_mask, denoise_gate)
        elif self.descriptor_type == "raw_representation_concat":
            logits = self._raw_representation_logits(
                node, prompt, edge_node[apply_mask], edge_feature[apply_mask]
            )
            denoise_gate = self.gate_floor + (self.gate_cap - self.gate_floor) * torch.sigmoid(logits)
            edge_gate = base_edge_weight.new_ones(edge_count).masked_scatter(apply_mask, denoise_gate)
        elif self.descriptor_type == "representation_concat":
            # Project once per node/prompt; only gather compact active-edge inputs.
            node_projected = self.representation_projection(node)
            prompt_projected = self.representation_projection(prompt)
            gate_input = torch.cat(
                [prompt_projected[edge_feature[apply_mask]], node_projected[edge_node[apply_mask]]],
                dim=-1,
            )
            raw_gate = torch.sigmoid(self.representation_gate(gate_input).squeeze(-1))
            denoise_gate = self.gate_floor + (self.gate_cap - self.gate_floor) * raw_gate
            edge_gate = base_edge_weight.new_ones(edge_count).masked_scatter(apply_mask, denoise_gate)
        else:
            stats = feature_stats.to(device=device, dtype=dtype)
            edge_stats = stats[edge_feature] if stats.numel() > 0 else base_edge_weight.new_zeros((edge_count, 6))
            gate_by_feature = feature_gate.to(device=device, dtype=dtype)
            if gate_by_feature.numel() == 0:
                gate_by_feature = base_edge_weight.new_ones(max(1, int(edge_feature.max().item()) + 1))
            feature_gate_edge = gate_by_feature[edge_feature].unsqueeze(-1)
            node_norm = node.norm(dim=-1)
            prompt_norm = prompt.norm(dim=-1)
            node_norm_edge = (node_norm / node_norm.mean().clamp_min(1e-6))[edge_node].unsqueeze(-1)
            prompt_norm_edge = (prompt_norm / prompt_norm.mean().clamp_min(1e-6))[edge_feature].unsqueeze(-1)
            cosine = F.cosine_similarity(node[edge_node], prompt[edge_feature], dim=-1, eps=1e-8).unsqueeze(-1)
            gate_input = torch.cat(
                [edge_stats, feature_gate_edge, base_edge_weight.detach().unsqueeze(-1),
                 direction.to(dtype=dtype).unsqueeze(-1), node_norm_edge, prompt_norm_edge,
                 cosine, usage_edge, hub_score_edge],
                dim=-1,
            )
            raw_gate = torch.sigmoid(self.edge_gate(gate_input).squeeze(-1))
            denoise_gate = self.gate_floor + (self.gate_cap - self.gate_floor) * raw_gate
            edge_gate = torch.where(apply_mask, denoise_gate, base_edge_weight.new_ones(edge_count))
        mix_ratio = self.mix_ratio(base_edge_weight, epoch=epoch)
        effective_edge_gate = 1.0 + mix_ratio * (edge_gate - 1.0)

        active_gate = edge_gate[apply_mask]
        keep_static = (1.0 - active_gate).pow(2).mean() if active_gate.numel() > 0 else self._zero(base_edge_weight)
        active_hub_score = hub_score_edge.squeeze(-1)[apply_mask]
        hub_mask = active_hub_score > 0.0
        hub_suppression = (
            (active_gate[hub_mask] * active_hub_score[hub_mask]).mean()
            if bool(hub_mask.any())
            else self._zero(base_edge_weight)
        )
        edge_budget = active_gate.mean() if active_gate.numel() > 0 else self._zero(base_edge_weight)

        homophily_loss, homophily_stats = self._homophily_loss(
            edge_gate=edge_gate,
            apply_mask=apply_mask,
            edge_node=edge_node,
            edge_feature=edge_feature,
            feature_dominant_class=feature_dominant_class,
            labels=labels,
            train_mask=train_mask,
        )

        p2n_mask = direction == 1
        n2p_mask = direction == 0
        hub_edge_mask = p2n_mask & (hub_score[edge_feature] > 0.0)
        nonhub_edge_mask = p2n_mask & ~hub_edge_mask
        zero = self._zero(base_edge_weight)
        weight_delta = base_edge_weight * (effective_edge_gate - 1.0)
        active_effective_gate = effective_edge_gate[apply_mask]
        active_feature = edge_feature[apply_mask]
        within_prompt_std = self._within_prompt_std(
            gate=active_effective_gate,
            feature_index=active_feature,
            feature_count=int(usage.numel()),
        )
        stats_out = {
            "prompting_enabled": base_edge_weight.new_tensor(1.0),
            "prompting_edge_gate_mean": edge_gate.mean() if edge_gate.numel() > 0 else zero,
            "prompting_edge_gate_min": edge_gate.min() if edge_gate.numel() > 0 else zero,
            "prompting_edge_gate_max": edge_gate.max() if edge_gate.numel() > 0 else zero,
            "prompting_prompt_to_node_gate_mean": edge_gate[p2n_mask].mean() if bool(p2n_mask.any()) else zero,
            "prompting_node_to_prompt_gate_mean": edge_gate[n2p_mask].mean() if bool(n2p_mask.any()) else zero,
            "prompting_hub_gate_mean": edge_gate[hub_edge_mask].mean() if bool(hub_edge_mask.any()) else zero,
            "prompting_nonhub_gate_mean": edge_gate[nonhub_edge_mask].mean() if bool(nonhub_edge_mask.any()) else zero,
            "prompting_edge_keep_ratio": (active_gate >= self.keep_threshold).to(dtype=dtype).mean()
            if active_gate.numel() > 0
            else zero,
            "prompting_homophily_loss": homophily_loss,
            "prompting_learned_gate_mean": active_gate.mean() if active_gate.numel() > 0 else zero,
            "prompting_effective_gate_mean": effective_edge_gate[apply_mask].mean() if bool(apply_mask.any()) else zero,
            "prompting_mix_ratio": mix_ratio,
            "prompting_gate_above_one_ratio": (active_gate > 1.0).to(dtype=dtype).mean() if active_gate.numel() > 0 else zero,
            "prompting_effective_gate_below_0p3_ratio": (active_effective_gate < 0.3).to(dtype=dtype).mean()
            if active_effective_gate.numel() > 0 else zero,
            "prompting_effective_gate_below_0p5_ratio": (active_effective_gate < 0.5).to(dtype=dtype).mean()
            if active_effective_gate.numel() > 0 else zero,
            "prompting_effective_gate_above_one_ratio": (active_effective_gate > 1.0).to(dtype=dtype).mean()
            if active_effective_gate.numel() > 0 else zero,
            "prompting_within_prompt_gate_std_mean": within_prompt_std,
            "prompting_weight_delta_mean": weight_delta[apply_mask].mean() if bool(apply_mask.any()) else zero,
            "prompting_weight_delta_abs_mean": weight_delta[apply_mask].abs().mean() if bool(apply_mask.any()) else zero,
            **homophily_stats,
        }
        if self.independent_edge_weights is not None:
            seen = self.independent_edge_weights.gradient_seen
            stats_out["prompting_independent_gradient_seen_ratio"] = seen.to(dtype=dtype).mean() if seen.numel() else zero
        return {
            "edge_gate": edge_gate,
            "effective_edge_gate": effective_edge_gate,
            "mix_ratio": mix_ratio,
            "losses": {
                "keep_static": keep_static,
                "hub_suppression": hub_suppression,
                "prompt_homophily": homophily_loss,
                "edge_budget": edge_budget,
            },
            "stats": stats_out,
        }

    def _within_prompt_std(
        self,
        *,
        gate: torch.Tensor,
        feature_index: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        """Mean within-prompt gate std without materializing edge groups."""
        if gate.numel() == 0 or feature_count <= 0:
            return self._zero(gate)
        count = gate.new_zeros(feature_count)
        total = gate.new_zeros(feature_count)
        total_sq = gate.new_zeros(feature_count)
        count.index_add_(0, feature_index, torch.ones_like(gate))
        total.index_add_(0, feature_index, gate)
        total_sq.index_add_(0, feature_index, gate.square())
        valid = count > 1
        if not bool(valid.any()):
            return self._zero(gate)
        mean = total[valid] / count[valid]
        variance = total_sq[valid] / count[valid] - mean.square()
        return variance.clamp_min(0.0).sqrt().mean()

    def _homophily_loss(
        self,
        *,
        edge_gate: torch.Tensor,
        apply_mask: torch.Tensor,
        edge_node: torch.Tensor,
        edge_feature: torch.Tensor,
        feature_dominant_class: torch.Tensor | None,
        labels: torch.Tensor | None,
        train_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = self._zero(edge_gate)
        empty_stats = {
            "prompting_train_edge_homophily_raw": zero,
            "prompting_train_edge_homophily_weighted": zero,
            "prompting_train_same_class_gate_mean": zero,
            "prompting_train_offclass_gate_mean": zero,
        }
        if not (
            isinstance(feature_dominant_class, torch.Tensor)
            and isinstance(labels, torch.Tensor)
            and isinstance(train_mask, torch.Tensor)
            and feature_dominant_class.numel() > 0
        ):
            return zero, empty_stats
        dominant = feature_dominant_class.to(device=edge_gate.device, dtype=torch.long)
        y = labels.to(device=edge_gate.device, dtype=torch.long)
        train = train_mask.to(device=edge_gate.device, dtype=torch.bool)
        valid = apply_mask & train[edge_node] & (edge_feature < dominant.numel()) & (dominant[edge_feature] >= 0)
        if not bool(valid.any()):
            return zero, empty_stats
        same = y[edge_node[valid]] == dominant[edge_feature[valid]]
        valid_gate = edge_gate[valid]
        off_loss = valid_gate[~same].mean() if bool((~same).any()) else zero
        same_loss = (1.0 - valid_gate[same]).pow(2).mean() if bool(same.any()) else zero
        loss = off_loss + max(0.0, self.homophily_keep_weight) * same_loss
        weighted = (
            (valid_gate * same.to(dtype=valid_gate.dtype)).sum() / valid_gate.sum().clamp_min(1e-6)
            if valid_gate.numel() > 0
            else zero
        )
        stats = {
            "prompting_train_edge_homophily_raw": same.to(dtype=valid_gate.dtype).mean() if same.numel() > 0 else zero,
            "prompting_train_edge_homophily_weighted": weighted,
            "prompting_train_same_class_gate_mean": valid_gate[same].mean() if bool(same.any()) else zero,
            "prompting_train_offclass_gate_mean": valid_gate[~same].mean() if bool((~same).any()) else zero,
        }
        return loss, stats
