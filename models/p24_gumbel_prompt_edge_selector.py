"""P23-anchored hierarchical Gumbel selector for P24 prompt edges."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class P24GumbelPromptEdgeSelector(nn.Module):
    """Learn prompt-level and edge-level keep probabilities on P23 edges."""

    def __init__(self, repr_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.apply_to = str(self.config.get("apply_to", "prompt_to_node"))
        if self.apply_to != "prompt_to_node":
            raise ValueError("Gumbel P24 currently supports prompt_to_node only")
        if bool(self.config.get("allow_new_edges", False)):
            raise ValueError("Gumbel P24 only selects existing P23 prompt edges")

        self.repr_dim = int(repr_dim)
        self.projection_dim = int(self.config.get("projection_dim", 16))
        self.hidden_dim = int(self.config.get("hidden_dim", 32))
        self.dropout = float(self.config.get("dropout", 0.10))
        self.detach_context = bool(self.config.get("detach_context", True))
        self.keep_prob_init = float(self.config.get("keep_prob_init", 0.95))
        self.temperature_init = float(self.config.get("temperature_init", 1.0))
        self.temperature_min = float(self.config.get("temperature_min", 0.30))
        self.temperature_anneal_epochs = int(self.config.get("temperature_anneal_epochs", 100))
        self.pruning_strength_init = float(self.config.get("pruning_strength_init", 0.0))
        self.pruning_strength_max = float(self.config.get("pruning_strength_max", 0.40))
        self.pruning_warmup_epochs = int(self.config.get("pruning_warmup_epochs", 50))
        self.target_keep_ratio = float(self.config.get("target_keep_ratio", 0.80))
        self.keep_threshold = float(self.config.get("keep_threshold", 0.50))
        self.hub_quantile = float(self.config.get("hub_quantile", 0.80))
        self.straight_through = bool(self.config.get("straight_through", False))
        self.deterministic_eval = bool(self.config.get("deterministic_eval", True))
        self.hard_eval = bool(self.config.get("hard_eval", False))
        self.minimum_edges_per_node = max(1, int(self.config.get("minimum_edges_per_node", 1)))
        self.edge_selector_enabled = bool(self.config.get("edge_selector_enabled", True))

        if self.repr_dim <= 0 or self.projection_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("Gumbel selector dimensions must be positive")
        if not 0.0 < self.keep_prob_init < 1.0:
            raise ValueError("keep_prob_init must be in (0, 1)")
        if not 0.0 < self.temperature_min <= self.temperature_init:
            raise ValueError("temperature must satisfy 0 < min <= init")
        if not 0.0 <= self.pruning_strength_init <= self.pruning_strength_max <= 1.0:
            raise ValueError("pruning strengths must satisfy 0 <= init <= max <= 1")
        if not 0.0 <= self.target_keep_ratio <= 1.0:
            raise ValueError("target_keep_ratio must be in [0, 1]")
        if not 0.0 <= self.hub_quantile <= 1.0:
            raise ValueError("hub_quantile must be in [0, 1]")

        self.node_projection = nn.Sequential(
            nn.LayerNorm(self.repr_dim),
            nn.Linear(self.repr_dim, self.projection_dim),
            nn.GELU(),
        )
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(self.repr_dim),
            nn.Linear(self.repr_dim, self.projection_dim),
            nn.GELU(),
        )

        # P23 supplies six feature statistics, feature gate, usage, hub rank,
        # and mean static edge weight.
        self.prompt_static_dim = 10
        self.node_static_dim = 6
        self.prompt_scorer = nn.Sequential(
            nn.LayerNorm(self.projection_dim + self.prompt_static_dim),
            nn.Linear(self.projection_dim + self.prompt_static_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        edge_input_dim = 2 * self.projection_dim + self.prompt_static_dim + self.node_static_dim + 1
        self.edge_scorer = nn.Sequential(
            nn.LayerNorm(edge_input_dim),
            nn.Linear(edge_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        initial_logit = torch.logit(torch.tensor(self.keep_prob_init)).item()
        nn.init.zeros_(self.prompt_scorer[-1].weight)
        nn.init.zeros_(self.edge_scorer[-1].weight)
        nn.init.constant_(
            self.prompt_scorer[-1].bias,
            0.5 * initial_logit if self.edge_selector_enabled else initial_logit,
        )
        nn.init.constant_(
            self.edge_scorer[-1].bias,
            0.5 * initial_logit if self.edge_selector_enabled else 0.0,
        )

    @staticmethod
    def _fit_feature_stats(stats: torch.Tensor, count: int, ref: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 0:
            return ref.new_zeros((count, 6))
        stats = stats.to(device=ref.device, dtype=ref.dtype)
        if stats.size(0) != count:
            raise ValueError("feature_stats must match prompt count")
        if stats.size(1) < 6:
            stats = torch.cat([stats, ref.new_zeros((count, 6 - stats.size(1)))], dim=-1)
        return stats[:, :6]

    @staticmethod
    def _normalized_log(values: torch.Tensor) -> torch.Tensor:
        logged = torch.log1p(values.clamp_min(0.0))
        return logged / logged.mean().clamp_min(1e-6)

    def _schedule(self, epoch: int | None, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = 0 if epoch is None else max(0, int(epoch))
        temp_progress = min(1.0, current / max(1, self.temperature_anneal_epochs))
        temperature = self.temperature_init + temp_progress * (
            self.temperature_min - self.temperature_init
        )
        prune_progress = min(1.0, current / max(1, self.pruning_warmup_epochs))
        strength = self.pruning_strength_init + prune_progress * (
            self.pruning_strength_max - self.pruning_strength_init
        )
        return ref.new_tensor(temperature), ref.new_tensor(strength)

    def _node_static(
        self,
        *,
        node_repr: torch.Tensor,
        active_node: torch.Tensor,
        original_edge_index: torch.Tensor | None,
        no_prompt_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        num_nodes = int(node_repr.size(0))
        dtype = node_repr.dtype
        repr_norm = node_repr.norm(dim=-1)
        repr_norm = repr_norm / repr_norm.mean().clamp_min(1e-6)
        prompt_degree = node_repr.new_zeros(num_nodes)
        prompt_degree.index_add_(0, active_node, torch.ones_like(active_node, dtype=dtype))
        prompt_degree = self._normalized_log(prompt_degree)
        graph_degree = node_repr.new_zeros(num_nodes)
        if isinstance(original_edge_index, torch.Tensor) and original_edge_index.numel() > 0:
            target = original_edge_index[1].to(device=node_repr.device, dtype=torch.long)
            graph_degree.index_add_(0, target, torch.ones_like(target, dtype=dtype))
        graph_degree = self._normalized_log(graph_degree)
        entropy = node_repr.new_zeros(num_nodes)
        margin = node_repr.new_zeros(num_nodes)
        confidence = node_repr.new_zeros(num_nodes)
        if isinstance(no_prompt_logits, torch.Tensor) and no_prompt_logits.size(0) == num_nodes:
            probability = F.softmax(no_prompt_logits.detach().to(dtype=dtype), dim=-1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            if probability.size(-1) > 1:
                entropy = entropy / probability.new_tensor(float(probability.size(-1))).log()
                top = probability.topk(k=2, dim=-1).values
                margin = top[:, 0] - top[:, 1]
            confidence = probability.max(dim=-1).values
        return torch.stack(
            [repr_norm, prompt_degree, graph_degree, entropy, margin, confidence], dim=-1
        )

    def _ensure_minimum_hard_edges(
        self,
        *,
        hard_mask: torch.Tensor,
        probability: torch.Tensor,
        active_node: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        output = hard_mask.clone()
        for node_index in torch.unique(active_node).tolist():
            positions = torch.where(active_node == int(node_index))[0]
            current = int(output[positions].sum().item())
            needed = min(self.minimum_edges_per_node, int(positions.numel())) - current
            if needed > 0:
                chosen = positions[torch.topk(probability[positions], k=needed).indices]
                output[chosen] = 1.0
        return output

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
        original_edge_index: torch.Tensor | None = None,
        no_prompt_logits: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        edge_node = edge_node_index.to(device=base_edge_weight.device, dtype=torch.long)
        edge_feature = edge_feature_local.to(device=base_edge_weight.device, dtype=torch.long)
        direction = edge_direction.to(device=base_edge_weight.device, dtype=torch.long)
        active = direction == 1
        if not self.enabled or base_edge_weight.numel() == 0 or not bool(active.any()):
            one = torch.ones_like(base_edge_weight)
            zero = base_edge_weight.new_tensor(0.0)
            return {
                "edge_weight": base_edge_weight,
                "edge_gate": one,
                "candidate_edge_weight": base_edge_weight,
                "prompt_keep_probability": base_edge_weight.new_ones(prompt_repr.size(0)),
                "prompt_keep_logit": base_edge_weight.new_zeros(prompt_repr.size(0)),
                "losses": {"keep_static": zero, "keep_ratio": zero, "entropy": zero},
                "stats": {"p24_gumbel_enabled": zero},
            }

        node = node_repr.to(device=base_edge_weight.device, dtype=base_edge_weight.dtype)
        if self.detach_context:
            node = node.detach()
        feature_count = int(edge_feature.max().item()) + 1
        prompt = node.new_zeros((feature_count, node.size(-1)))
        prompt_mass = node.new_zeros(feature_count)
        prototype_weight = base_edge_weight[active].detach().clamp_min(0.0)
        prompt.index_add_(
            0,
            edge_feature[active],
            node[edge_node[active]] * prototype_weight.unsqueeze(-1),
        )
        prompt_mass.index_add_(0, edge_feature[active], prototype_weight)
        prompt = prompt / prompt_mass.clamp_min(1e-6).unsqueeze(-1)
        stats = self._fit_feature_stats(feature_stats, feature_count, base_edge_weight)
        feature_gate = feature_gate.to(device=base_edge_weight.device, dtype=base_edge_weight.dtype)
        if feature_gate.numel() != feature_count:
            feature_gate = base_edge_weight.new_ones(feature_count)
        usage = (
            feature_usage.to(device=base_edge_weight.device, dtype=base_edge_weight.dtype)
            if isinstance(feature_usage, torch.Tensor) and feature_usage.numel() == feature_count
            else base_edge_weight.new_ones(feature_count)
        )
        usage_norm = self._normalized_log(usage)
        order = torch.argsort(usage.detach())
        rank = torch.empty_like(usage)
        rank[order] = torch.linspace(0.0, 1.0, feature_count, device=usage.device, dtype=usage.dtype)
        hub_rank = ((rank - self.hub_quantile) / max(1e-6, 1.0 - self.hub_quantile)).clamp(0.0, 1.0)
        active_node = edge_node[active]
        active_feature = edge_feature[active]
        mean_static = base_edge_weight.new_zeros(feature_count)
        count = base_edge_weight.new_zeros(feature_count)
        mean_static.index_add_(0, active_feature, base_edge_weight[active].detach())
        count.index_add_(0, active_feature, torch.ones_like(base_edge_weight[active]))
        mean_static = mean_static / count.clamp_min(1.0)
        mean_static = mean_static / mean_static.mean().clamp_min(1e-6)
        prompt_static = torch.cat(
            [
                stats,
                feature_gate.detach().unsqueeze(-1),
                usage_norm.unsqueeze(-1),
                hub_rank.unsqueeze(-1),
                mean_static.unsqueeze(-1),
            ],
            dim=-1,
        )
        reliability_proxy = (
            stats[:, 1].clamp(0.0, 1.0)
            + (1.0 - stats[:, 2]).clamp(0.0, 1.0)
            + stats[:, 3].clamp(0.0, 1.0)
            + stats[:, 4].clamp(0.0, 1.0)
            + stats[:, 5].clamp(0.0, 1.0)
            + feature_gate.detach().clamp(0.0, 1.0)
        ) / 6.0
        prompt_prunable_prior = (
            hub_rank * (1.0 - 0.5 * reliability_proxy)
            + (1.0 - hub_rank) * 0.25 * (1.0 - reliability_proxy)
        ).clamp(0.0, 1.0)
        node_static = self._node_static(
            node_repr=node,
            active_node=active_node,
            original_edge_index=original_edge_index,
            no_prompt_logits=no_prompt_logits,
        )
        node_latent = self.node_projection(node)
        prompt_latent = self.prompt_projection(prompt)
        prompt_logit = self.prompt_scorer(
            torch.cat([prompt_latent, prompt_static], dim=-1)
        ).squeeze(-1)
        node_edge = node_latent[active_node]
        prompt_edge = prompt_latent[active_feature]
        edge_input = torch.cat(
            [
                node_edge * prompt_edge,
                (node_edge - prompt_edge).abs(),
                prompt_static[active_feature],
                node_static[active_node],
                base_edge_weight[active].detach().unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_logit = (
            self.edge_scorer(edge_input).squeeze(-1)
            if self.edge_selector_enabled
            else torch.zeros_like(active_node, dtype=base_edge_weight.dtype)
        )
        combined_logit = prompt_logit[active_feature] + edge_logit
        keep_probability = torch.sigmoid(combined_logit)
        prompt_logit_scale = 2.0 if self.edge_selector_enabled else 1.0
        prompt_keep_logit = prompt_logit * prompt_logit_scale
        prompt_keep_probability = torch.sigmoid(prompt_keep_logit)
        temperature, pruning_strength = self._schedule(epoch, base_edge_weight)

        if self.training:
            uniform = torch.rand_like(combined_logit).clamp(1e-6, 1.0 - 1e-6)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            soft_mask = torch.sigmoid((combined_logit + logistic_noise) / temperature)
            if self.straight_through:
                hard = (soft_mask >= self.keep_threshold).to(dtype=soft_mask.dtype)
                selection = hard.detach() - soft_mask.detach() + soft_mask
            else:
                selection = soft_mask
        else:
            selection = keep_probability
            if self.hard_eval:
                hard = (keep_probability >= self.keep_threshold).to(dtype=keep_probability.dtype)
                selection = self._ensure_minimum_hard_edges(
                    hard_mask=hard,
                    probability=keep_probability,
                    active_node=active_node,
                    num_nodes=int(node.size(0)),
                )

        edge_prunable_prior = prompt_prunable_prior[active_feature]
        active_gate = 1.0 - pruning_strength * edge_prunable_prior * (1.0 - selection)
        candidate_weight = base_edge_weight.clone()
        candidate_weight[active] = base_edge_weight[active] * active_gate
        edge_gate = torch.ones_like(base_edge_weight)
        edge_gate[active] = active_gate
        hard_mask = self._ensure_minimum_hard_edges(
            hard_mask=(keep_probability >= self.keep_threshold).to(dtype=keep_probability.dtype),
            probability=keep_probability,
            active_node=active_node,
            num_nodes=int(node.size(0)),
        )
        entropy = -(
            keep_probability * keep_probability.clamp_min(1e-12).log()
            + (1.0 - keep_probability) * (1.0 - keep_probability).clamp_min(1e-12).log()
        ).mean()
        # P23 is already useful: the budget is a minimum keep ratio, not a
        # pruning target that forces topology changes without utility evidence.
        keep_ratio_loss = F.relu(
            keep_probability.new_tensor(self.target_keep_ratio) - keep_probability.mean()
        ).pow(2)
        keep_static = (active_gate - 1.0).pow(2).mean()
        hub_edge = hub_rank[active_feature] > 0.0
        nonhub_edge = ~hub_edge
        effective_pruned = active_gate < 1.0 - 0.05 * float(self.pruning_strength_max)
        zero = base_edge_weight.new_tensor(0.0)
        return {
            "edge_weight": candidate_weight,
            "edge_gate": edge_gate,
            "candidate_edge_weight": candidate_weight,
            "prompt_keep_probability": prompt_keep_probability,
            "prompt_keep_logit": prompt_keep_logit,
            "edge_keep_probability": keep_probability,
            "edge_keep_logit": combined_logit,
            "prompt_prunable_prior": prompt_prunable_prior,
            "losses": {
                "keep_static": keep_static,
                "keep_ratio": keep_ratio_loss,
                "entropy": entropy,
            },
            "stats": {
                "p24_gumbel_enabled": base_edge_weight.new_tensor(1.0),
                "p24_gumbel_edge_selector_enabled": base_edge_weight.new_tensor(float(self.edge_selector_enabled)),
                "p24_gumbel_temperature": temperature,
                "p24_gumbel_pruning_strength": pruning_strength,
                "p24_gumbel_keep_probability_mean": keep_probability.mean(),
                "p24_gumbel_keep_probability_min": keep_probability.min(),
                "p24_gumbel_keep_probability_max": keep_probability.max(),
                "p24_gumbel_probability_entropy": entropy,
                "p24_gumbel_soft_keep_ratio": keep_probability.mean(),
                "p24_gumbel_hard_keep_ratio": hard_mask.mean(),
                "p24_gumbel_prompt_keep_mean": prompt_keep_probability.mean(),
                "p24_gumbel_prunable_prior_mean": edge_prunable_prior.mean(),
                "p24_gumbel_hub_prunable_prior_mean": edge_prunable_prior[hub_edge].mean() if bool(hub_edge.any()) else zero,
                "p24_gumbel_nonhub_prunable_prior_mean": edge_prunable_prior[nonhub_edge].mean() if bool(nonhub_edge.any()) else zero,
                "p24_gumbel_hub_keep_mean": keep_probability[hub_edge].mean() if bool(hub_edge.any()) else zero,
                "p24_gumbel_nonhub_keep_mean": keep_probability[nonhub_edge].mean() if bool(nonhub_edge.any()) else zero,
                "p24_gumbel_effectively_pruned_ratio": effective_pruned.to(dtype=base_edge_weight.dtype).mean(),
                "p24_gumbel_sample_variance": selection.var(unbiased=False),
                "p24_candidate_weight_delta_abs_mean": (
                    candidate_weight[active] - base_edge_weight[active]
                ).abs().mean(),
                "p24_candidate_gate_mean": active_gate.mean(),
                "p24_candidate_gate_min": active_gate.min(),
                "p24_candidate_gate_max": active_gate.max(),
            },
        }
