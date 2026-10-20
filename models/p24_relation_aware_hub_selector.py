"""Unsupervised relation-aware hub edge selection for P24."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class P24RelationAwareHubEdgeSelector(nn.Module):
    """Denoise P23 prompt-to-node edges without assuming feature homophily."""

    def __init__(self, repr_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.apply_to = str(self.config.get("apply_to", "prompt_to_node"))
        if self.apply_to != "prompt_to_node":
            raise ValueError("relation-aware P24 supports prompt_to_node only")
        if bool(self.config.get("allow_new_edges", False)):
            raise ValueError("relation-aware P24 only selects existing P23 prompt edges")

        self.repr_dim = int(repr_dim)
        clusters = self.config.get("relation_clusters", [8, 16])
        if isinstance(clusters, int):
            clusters = [clusters]
        self.relation_clusters = tuple(sorted({max(2, int(value)) for value in clusters}))
        self.kmeans_iterations = max(1, int(self.config.get("kmeans_iterations", 8)))
        self.hidden_dim = max(4, int(self.config.get("hidden_dim", 32)))
        self.dropout = float(self.config.get("dropout", 0.10))
        self.detach_context = bool(self.config.get("detach_context", True))
        self.keep_prob_init = float(self.config.get("keep_prob_init", 0.95))
        self.temperature_init = float(self.config.get("temperature_init", 1.0))
        self.temperature_min = float(self.config.get("temperature_min", 0.30))
        self.temperature_anneal_epochs = max(
            1, int(self.config.get("temperature_anneal_epochs", 100))
        )
        self.pruning_strength_init = float(self.config.get("pruning_strength_init", 0.0))
        self.pruning_strength_max = float(self.config.get("pruning_strength_max", 0.80))
        self.pruning_warmup_epochs = max(1, int(self.config.get("pruning_warmup_epochs", 50)))
        self.gate_floor = float(self.config.get("gate_floor", 0.20))
        self.keep_threshold = float(self.config.get("keep_threshold", 0.50))
        self.hub_quantile = float(self.config.get("hub_quantile", 0.80))
        self.hub_power = float(self.config.get("hub_power", 1.0))
        self.local_profile_weight = float(self.config.get("local_profile_weight", 0.50))
        self.compatibility_threshold = float(
            self.config.get("relation_compatibility_threshold", 0.85)
        )
        self.prior_temperature = float(self.config.get("relation_prior_temperature", 0.05))
        self.max_prune_target = float(self.config.get("max_prune_target", 0.80))
        self.noise_activation_threshold = float(
            self.config.get("noise_activation_threshold", 0.05)
        )
        self.uncertainty_threshold = float(self.config.get("uncertainty_threshold", 0.20))
        self.relation_smoothing = float(self.config.get("relation_smoothing", 1e-3))
        self.homophily_collapse_margin = float(
            self.config.get("homophily_collapse_margin", 0.02)
        )
        self.clean_prior_weight = float(self.config.get("clean_prior_weight", 0.25))
        self.minimum_edges_per_node = max(1, int(self.config.get("minimum_edges_per_node", 1)))
        self.straight_through = bool(self.config.get("straight_through", False))
        self.hard_eval = bool(self.config.get("hard_eval", False))

        if self.repr_dim <= 0:
            raise ValueError("repr_dim must be positive")
        if not 0.0 < self.keep_prob_init < 1.0:
            raise ValueError("keep_prob_init must be in (0, 1)")
        if not 0.0 < self.temperature_min <= self.temperature_init:
            raise ValueError("temperature must satisfy 0 < min <= init")
        if not 0.0 <= self.pruning_strength_init <= self.pruning_strength_max <= 1.0:
            raise ValueError("pruning strengths must satisfy 0 <= init <= max <= 1")
        if self.pruning_strength_max > 1.0 - self.gate_floor + 1e-8:
            raise ValueError("pruning_strength_max must preserve gate_floor")
        for name, value in {
            "hub_quantile": self.hub_quantile,
            "local_profile_weight": self.local_profile_weight,
            "max_prune_target": self.max_prune_target,
            "noise_activation_threshold": self.noise_activation_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "clean_prior_weight": self.clean_prior_weight,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.prior_temperature <= 0.0:
            raise ValueError("relation_prior_temperature must be positive")

        # The scorer sees relation statistics only. It never receives cosine
        # similarity as a positive edge signal.
        self.edge_feature_dim = 8
        self.edge_scorer = nn.Sequential(
            nn.LayerNorm(self.edge_feature_dim),
            nn.Linear(self.edge_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.edge_scorer[-1].weight)
        nn.init.constant_(
            self.edge_scorer[-1].bias,
            torch.logit(torch.tensor(self.keep_prob_init)).item(),
        )

        self._relation_cache_key: tuple[Any, ...] | None = None
        self._relation_cache: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _normalized_entropy(distribution: torch.Tensor) -> torch.Tensor:
        count = int(distribution.size(-1))
        entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=-1)
        if count > 1:
            entropy = entropy / distribution.new_tensor(float(count)).log()
        return entropy.clamp(0.0, 1.0)

    @staticmethod
    def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        middle = 0.5 * (left + right)
        left_kl = (left * (left.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum()
        right_kl = (right * (right.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum()
        return 0.5 * (left_kl + right_kl)

    @staticmethod
    def _schedule_value(
        *, start: float, end: float, epoch: int | None, duration: int, ref: torch.Tensor
    ) -> torch.Tensor:
        current = 0 if epoch is None else max(0, int(epoch))
        progress = min(1.0, current / max(1, duration))
        return ref.new_tensor(start + progress * (end - start))

    def _schedule(self, epoch: int | None, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = self._schedule_value(
            start=self.temperature_init,
            end=self.temperature_min,
            epoch=epoch,
            duration=self.temperature_anneal_epochs,
            ref=ref,
        )
        strength = self._schedule_value(
            start=self.pruning_strength_init,
            end=self.pruning_strength_max,
            epoch=epoch,
            duration=self.pruning_warmup_epochs,
            ref=ref,
        )
        return temperature, strength

    @torch.no_grad()
    def _deterministic_kmeans(self, node_repr: torch.Tensor, cluster_count: int) -> torch.Tensor:
        values = F.normalize(node_repr.detach().to(dtype=torch.float32), dim=-1, eps=1e-12)
        num_nodes = int(values.size(0))
        k = min(max(2, int(cluster_count)), num_nodes)
        centroids = [values[0]]
        min_distance = 1.0 - values @ centroids[0]
        for _ in range(1, k):
            next_index = int(torch.argmax(min_distance).item())
            next_centroid = values[next_index]
            centroids.append(next_centroid)
            min_distance = torch.minimum(min_distance, 1.0 - values @ next_centroid)
        centroid = torch.stack(centroids, dim=0)
        assignment = torch.zeros(num_nodes, dtype=torch.long, device=values.device)
        for _ in range(self.kmeans_iterations):
            assignment = torch.argmax(values @ centroid.t(), dim=-1)
            updated = torch.zeros_like(centroid)
            updated.index_add_(0, assignment, values)
            counts = torch.bincount(assignment, minlength=k).to(dtype=values.dtype).unsqueeze(-1)
            nonempty = counts.squeeze(-1) > 0
            updated[nonempty] = updated[nonempty] / counts[nonempty]
            updated[~nonempty] = centroid[~nonempty]
            centroid = F.normalize(updated, dim=-1, eps=1e-12)
        return assignment

    @torch.no_grad()
    def _build_relation_cache(
        self, node_repr: torch.Tensor, original_edge_index: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        node_count = int(node_repr.size(0))
        edge_index = original_edge_index.to(device=node_repr.device, dtype=torch.long)
        source = edge_index[0]
        target = edge_index[1]
        cache: list[dict[str, torch.Tensor]] = []
        for requested_k in self.relation_clusters:
            assignment = self._deterministic_kmeans(node_repr, requested_k)
            k = int(assignment.max().item()) + 1
            global_count = torch.bincount(assignment, minlength=k).to(dtype=node_repr.dtype)
            global_probability = global_count / global_count.sum().clamp_min(1.0)

            node_flat = target * k + assignment[source]
            node_count_matrix = torch.bincount(
                node_flat, minlength=node_count * k
            ).reshape(node_count, k).to(dtype=node_repr.dtype)
            node_count_matrix = node_count_matrix + self.relation_smoothing * global_probability
            node_profile = node_count_matrix / node_count_matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            transition_flat = assignment[target] * k + assignment[source]
            transition_count = torch.bincount(
                transition_flat, minlength=k * k
            ).reshape(k, k).to(dtype=node_repr.dtype)
            transition_count = transition_count + self.relation_smoothing * global_probability.unsqueeze(0)
            transition = transition_count / transition_count.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            transition_joint = transition_count / transition_count.sum().clamp_min(1e-12)
            cache.append(
                {
                    "assignment": assignment,
                    "node_profile": node_profile,
                    "transition": transition,
                    "transition_joint": transition_joint,
                    "global_probability": global_probability,
                }
            )
        return cache

    def _get_relation_cache(
        self, node_repr: torch.Tensor, original_edge_index: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        key = (
            str(node_repr.device),
            str(node_repr.dtype),
            int(node_repr.size(0)),
            int(node_repr.size(1)),
            int(original_edge_index.size(1)),
            round(float(node_repr.detach().sum().item()), 6),
            round(float(node_repr.detach().pow(2).sum().item()), 6),
            self.relation_clusters,
        )
        if self._relation_cache_key != key:
            self._relation_cache = self._build_relation_cache(node_repr, original_edge_index)
            self._relation_cache_key = key
        return self._relation_cache

    @staticmethod
    def _hub_rank(usage: torch.Tensor, quantile: float) -> torch.Tensor:
        count = int(usage.numel())
        if count <= 1:
            return torch.ones_like(usage)
        order = torch.argsort(usage.detach())
        rank = torch.empty_like(usage)
        rank[order] = torch.linspace(0.0, 1.0, count, device=usage.device, dtype=usage.dtype)
        return ((rank - quantile) / max(1e-6, 1.0 - quantile)).clamp(0.0, 1.0)

    def _ensure_minimum_hard_edges(
        self, hard_mask: torch.Tensor, probability: torch.Tensor, active_node: torch.Tensor
    ) -> torch.Tensor:
        output = hard_mask.clone()
        for node_index in torch.unique(active_node).tolist():
            positions = torch.where(active_node == int(node_index))[0]
            needed = min(self.minimum_edges_per_node, int(positions.numel())) - int(
                output[positions].sum().item()
            )
            if needed > 0:
                chosen = positions[torch.topk(probability[positions], k=needed).indices]
                output[chosen] = 1.0
        return output

    def _scale_features(
        self,
        *,
        context: dict[str, torch.Tensor],
        active_node: torch.Tensor,
        active_feature: torch.Tensor,
        active_weight: torch.Tensor,
        feature_count: int,
    ) -> dict[str, torch.Tensor]:
        assignment = context["assignment"]
        node_profile = context["node_profile"]
        transition = context["transition"]
        k = int(transition.size(0))
        edge_cluster = assignment[active_node]

        prompt_flat = active_feature * k + edge_cluster
        prompt_count = torch.zeros(feature_count * k, device=active_weight.device, dtype=active_weight.dtype)
        prompt_count.index_add_(0, prompt_flat, active_weight)
        prompt_count = prompt_count.reshape(feature_count, k)
        edge_prompt_count = prompt_count[active_feature].clone()
        edge_prompt_count[
            torch.arange(active_node.numel(), device=active_node.device), edge_cluster
        ] -= active_weight
        edge_prompt_count = edge_prompt_count.clamp_min(0.0)
        edge_prompt_mass = edge_prompt_count.sum(dim=-1, keepdim=True)
        fallback = context["global_probability"].unsqueeze(0).expand_as(edge_prompt_count)
        prompt_profile = torch.where(
            edge_prompt_mass > 1e-12,
            edge_prompt_count / edge_prompt_mass.clamp_min(1e-12),
            fallback,
        )

        global_relation_profile = transition[edge_cluster]
        local_relation_profile = node_profile[active_node]
        target_profile = (
            self.local_profile_weight * local_relation_profile
            + (1.0 - self.local_profile_weight) * global_relation_profile
        )
        target_profile = target_profile / target_profile.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        compatibility = torch.sqrt(
            prompt_profile.clamp_min(0.0) * target_profile.clamp_min(0.0)
        ).sum(dim=-1).clamp(0.0, 1.0)
        prompt_entropy = self._normalized_entropy(prompt_profile)
        node_entropy = self._normalized_entropy(target_profile)
        row = torch.arange(active_node.numel(), device=active_node.device)
        same_mode_excess = (
            prompt_profile[row, edge_cluster] - target_profile[row, edge_cluster]
        ).clamp_min(0.0)
        return {
            "compatibility": compatibility,
            "prompt_entropy": prompt_entropy,
            "node_entropy": node_entropy,
            "same_mode_excess": same_mode_excess,
            "prompt_profile": prompt_profile,
            "target_profile": target_profile,
            "edge_cluster": edge_cluster,
            "transition_joint": context["transition_joint"],
        }

    def _relation_preservation_losses(
        self,
        *,
        scale_outputs: list[dict[str, torch.Tensor]],
        active_weight: torch.Tensor,
        soft_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_losses: list[torch.Tensor] = []
        collapse_losses: list[torch.Tensor] = []
        original_diagonal: list[torch.Tensor] = []
        candidate_diagonal: list[torch.Tensor] = []
        weighted_edge = active_weight * soft_gate
        for scale in scale_outputs:
            prompt_profile = scale["prompt_profile"]
            receiver_cluster = scale["edge_cluster"]
            k = int(prompt_profile.size(-1))
            contribution = weighted_edge.unsqueeze(-1) * prompt_profile
            candidate = contribution.new_zeros((k, k))
            candidate.index_add_(0, receiver_cluster, contribution)
            candidate_joint = candidate / candidate.sum().clamp_min(1e-12)
            original_joint = scale["transition_joint"]
            relation_losses.append(self._js_divergence(candidate_joint, original_joint))
            original_diag = torch.diagonal(original_joint).sum()
            candidate_diag = torch.diagonal(candidate_joint).sum()
            collapse_losses.append(
                F.relu(candidate_diag - original_diag - self.homophily_collapse_margin).pow(2)
            )
            original_diagonal.append(original_diag)
            candidate_diagonal.append(candidate_diag)
        return (
            torch.stack(relation_losses).mean(),
            torch.stack(collapse_losses).mean(),
            torch.stack(original_diagonal).mean(),
            torch.stack(candidate_diagonal).mean(),
        )

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
        del feature_stats, feature_gate, prompt_repr, no_prompt_logits
        edge_node = edge_node_index.to(device=base_edge_weight.device, dtype=torch.long)
        edge_feature = edge_feature_local.to(device=base_edge_weight.device, dtype=torch.long)
        direction = edge_direction.to(device=base_edge_weight.device, dtype=torch.long)
        active = direction == 1
        zero = base_edge_weight.new_tensor(0.0)
        if (
            not self.enabled
            or base_edge_weight.numel() == 0
            or not bool(active.any())
            or not isinstance(original_edge_index, torch.Tensor)
            or original_edge_index.numel() == 0
        ):
            one = torch.ones_like(base_edge_weight)
            return {
                "edge_weight": base_edge_weight,
                "edge_gate": one,
                "candidate_edge_weight": base_edge_weight,
                "prompt_keep_probability": base_edge_weight.new_ones(0),
                "prompt_keep_logit": base_edge_weight.new_zeros(0),
                "edge_keep_probability": one,
                "edge_keep_logit": base_edge_weight.new_zeros(base_edge_weight.numel()),
                "prompt_prunable_prior": base_edge_weight.new_zeros(0),
                "losses": {
                    "keep_static": zero,
                    "keep_ratio": zero,
                    "entropy": zero,
                    "relation_prior": zero,
                    "relation_preservation": zero,
                    "homophily_collapse": zero,
                    "uncertain_keep": zero,
                },
                "stats": {"p24_relation_enabled": zero},
            }

        node = node_repr.to(device=base_edge_weight.device, dtype=base_edge_weight.dtype)
        if self.detach_context:
            node = node.detach()
        relation_cache = self._get_relation_cache(node, original_edge_index)
        active_node = edge_node[active]
        active_feature = edge_feature[active]
        active_weight = base_edge_weight[active].detach().clamp_min(0.0)
        feature_count = int(edge_feature.max().item()) + 1

        scale_outputs = [
            self._scale_features(
                context=context,
                active_node=active_node,
                active_feature=active_feature,
                active_weight=active_weight,
                feature_count=feature_count,
            )
            for context in relation_cache
        ]
        compatibility_stack = torch.stack(
            [output["compatibility"] for output in scale_outputs], dim=-1
        )
        compatibility_mean = compatibility_stack.mean(dim=-1)
        compatibility_min = compatibility_stack.min(dim=-1).values
        compatibility_std = compatibility_stack.std(dim=-1, unbiased=False)
        stability = (1.0 - compatibility_std / max(self.uncertainty_threshold, 1e-6)).clamp(0.0, 1.0)
        prompt_entropy = torch.stack(
            [output["prompt_entropy"] for output in scale_outputs], dim=-1
        ).mean(dim=-1)
        node_entropy = torch.stack(
            [output["node_entropy"] for output in scale_outputs], dim=-1
        ).mean(dim=-1)
        same_mode_excess = torch.stack(
            [output["same_mode_excess"] for output in scale_outputs], dim=-1
        ).mean(dim=-1)

        usage = base_edge_weight.new_zeros(feature_count)
        if isinstance(feature_usage, torch.Tensor) and feature_usage.numel() == feature_count:
            usage = feature_usage.to(device=usage.device, dtype=usage.dtype).clamp_min(0.0)
        else:
            usage.index_add_(0, active_feature, torch.ones_like(active_weight))
        hub_rank = self._hub_rank(usage, self.hub_quantile)
        edge_hub_rank = hub_rank[active_feature]
        low_compatibility = torch.sigmoid(
            (self.compatibility_threshold - compatibility_mean) / self.prior_temperature
        ) * torch.sigmoid(
            (self.compatibility_threshold - compatibility_min) / self.prior_temperature
        )
        noise_prior = (
            edge_hub_rank.pow(self.hub_power)
            * prompt_entropy
            * low_compatibility
            * stability
        ).clamp(0.0, 1.0)
        uncertain = compatibility_std > self.uncertainty_threshold
        noise_prior = torch.where(uncertain, torch.zeros_like(noise_prior), noise_prior)
        prior_keep_target = (1.0 - self.max_prune_target * noise_prior).clamp(1e-4, 1.0 - 1e-4)
        prunable = noise_prior > self.noise_activation_threshold

        normalized_weight = active_weight / active_weight.mean().clamp_min(1e-6)
        edge_features = torch.stack(
            [
                compatibility_mean,
                compatibility_min,
                stability,
                prompt_entropy,
                node_entropy,
                edge_hub_rank,
                same_mode_excess,
                normalized_weight.clamp(0.0, 4.0) / 4.0,
            ],
            dim=-1,
        ).detach()
        keep_logit = self.edge_scorer(edge_features).squeeze(-1)
        keep_probability = torch.sigmoid(keep_logit)
        temperature, pruning_strength = self._schedule(epoch, base_edge_weight)
        if self.training:
            uniform = torch.rand_like(keep_logit).clamp(1e-6, 1.0 - 1e-6)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            soft_selection = torch.sigmoid((keep_logit + logistic_noise) / temperature)
            if self.straight_through:
                hard = (soft_selection >= self.keep_threshold).to(dtype=soft_selection.dtype)
                selection = hard.detach() - soft_selection.detach() + soft_selection
            else:
                selection = soft_selection
        else:
            selection = keep_probability
            if self.hard_eval:
                hard = (keep_probability >= self.keep_threshold).to(dtype=keep_probability.dtype)
                selection = self._ensure_minimum_hard_edges(hard, keep_probability, active_node)

        selected_gate = 1.0 - pruning_strength * (1.0 - selection)
        selected_gate = selected_gate.clamp_min(self.gate_floor)
        # The unsupervised relation prior defines the support of the topology
        # search. Edges outside that support remain exactly equal to P23.
        active_gate = torch.where(prunable, selected_gate, torch.ones_like(selected_gate))
        candidate_weight = base_edge_weight.clone()
        candidate_weight[active] = base_edge_weight[active] * active_gate
        edge_gate = torch.ones_like(base_edge_weight)
        edge_gate[active] = active_gate

        hard_mask = self._ensure_minimum_hard_edges(
            (keep_probability >= self.keep_threshold).to(dtype=keep_probability.dtype),
            keep_probability,
            active_node,
        )
        edge_bce = F.binary_cross_entropy(
            keep_probability.clamp(1e-6, 1.0 - 1e-6), prior_keep_target, reduction="none"
        )
        noisy = prunable
        clean = ~noisy
        prior_parts: list[torch.Tensor] = []
        if bool(noisy.any()):
            prior_parts.append(edge_bce[noisy].mean())
        if bool(clean.any()):
            prior_parts.append(self.clean_prior_weight * edge_bce[clean].mean())
        relation_prior_loss = torch.stack(prior_parts).sum() if prior_parts else zero
        uncertain_keep_loss = (
            (1.0 - keep_probability[uncertain]).pow(2).mean() if bool(uncertain.any()) else zero
        )
        entropy = -(
            keep_probability * keep_probability.clamp_min(1e-12).log()
            + (1.0 - keep_probability) * (1.0 - keep_probability).clamp_min(1e-12).log()
        ).mean()
        selected_soft_gate = 1.0 - pruning_strength * (1.0 - keep_probability)
        soft_gate = torch.where(
            prunable, selected_soft_gate, torch.ones_like(selected_soft_gate)
        )
        relation_preservation, homophily_collapse, original_diag, candidate_diag = (
            self._relation_preservation_losses(
                scale_outputs=scale_outputs,
                active_weight=active_weight,
                soft_gate=soft_gate,
            )
        )

        prompt_probability_sum = base_edge_weight.new_zeros(feature_count)
        prompt_probability_count = base_edge_weight.new_zeros(feature_count)
        prompt_probability_sum.index_add_(0, active_feature, keep_probability)
        prompt_probability_count.index_add_(0, active_feature, torch.ones_like(keep_probability))
        prompt_keep_probability = prompt_probability_sum / prompt_probability_count.clamp_min(1.0)
        prompt_keep_probability = torch.where(
            prompt_probability_count > 0,
            prompt_keep_probability,
            torch.ones_like(prompt_keep_probability),
        )
        prompt_keep_logit = torch.logit(prompt_keep_probability.clamp(1e-6, 1.0 - 1e-6))
        prompt_noise_sum = base_edge_weight.new_zeros(feature_count)
        prompt_noise_sum.index_add_(0, active_feature, noise_prior)
        prompt_prunable_prior = prompt_noise_sum / prompt_probability_count.clamp_min(1.0)
        hub_edge = edge_hub_rank > 0.0
        nonhub_edge = ~hub_edge

        return {
            "edge_weight": candidate_weight,
            "edge_gate": edge_gate,
            "candidate_edge_weight": candidate_weight,
            "prompt_keep_probability": prompt_keep_probability,
            "prompt_keep_logit": prompt_keep_logit,
            "edge_keep_probability": keep_probability,
            "edge_keep_logit": keep_logit,
            "prompt_prunable_prior": prompt_prunable_prior,
            "edge_relation_compatibility": compatibility_mean,
            "edge_prior_keep_target": prior_keep_target,
            "losses": {
                "keep_static": (soft_gate - 1.0).pow(2).mean(),
                "keep_ratio": zero,
                "entropy": entropy,
                "relation_prior": relation_prior_loss,
                "relation_preservation": relation_preservation,
                "homophily_collapse": homophily_collapse,
                "uncertain_keep": uncertain_keep_loss,
            },
            "stats": {
                "p24_relation_enabled": base_edge_weight.new_tensor(1.0),
                "p24_relation_cluster_scale_count": base_edge_weight.new_tensor(
                    float(len(scale_outputs))
                ),
                "p24_relation_compatibility_mean": compatibility_mean.mean(),
                "p24_relation_compatibility_min": compatibility_mean.min(),
                "p24_relation_compatibility_p10": torch.quantile(compatibility_mean, 0.10),
                "p24_relation_compatibility_p50": torch.quantile(compatibility_mean, 0.50),
                "p24_relation_multiscale_stability_mean": stability.mean(),
                "p24_relation_prompt_entropy_mean": prompt_entropy.mean(),
                "p24_relation_node_entropy_mean": node_entropy.mean(),
                "p24_relation_same_mode_excess_mean": same_mode_excess.mean(),
                "p24_relation_noise_prior_mean": noise_prior.mean(),
                "p24_relation_noise_prior_max": noise_prior.max(),
                "p24_relation_noise_prior_p90": torch.quantile(noise_prior, 0.90),
                "p24_relation_noise_prior_p99": torch.quantile(noise_prior, 0.99),
                "p24_relation_prunable_edge_ratio": prunable.to(
                    dtype=base_edge_weight.dtype
                ).mean(),
                "p24_relation_prunable_gate_mean": active_gate[prunable].mean()
                if bool(prunable.any())
                else base_edge_weight.new_tensor(1.0),
                "p24_relation_protected_gate_mean": active_gate[~prunable].mean()
                if bool((~prunable).any())
                else base_edge_weight.new_tensor(1.0),
                "p24_relation_prior_keep_target_mean": prior_keep_target.mean(),
                "p24_relation_uncertain_edge_ratio": uncertain.to(dtype=base_edge_weight.dtype).mean(),
                "p24_relation_original_diagonal_mass": original_diag,
                "p24_relation_candidate_diagonal_mass": candidate_diag,
                "p24_relation_preservation_loss": relation_preservation,
                "p24_relation_homophily_collapse_loss": homophily_collapse,
                "p24_relation_prior_loss": relation_prior_loss,
                "p24_relation_uncertain_keep_loss": uncertain_keep_loss,
                "p24_gumbel_enabled": base_edge_weight.new_tensor(1.0),
                "p24_gumbel_edge_selector_enabled": base_edge_weight.new_tensor(1.0),
                "p24_gumbel_temperature": temperature,
                "p24_gumbel_pruning_strength": pruning_strength,
                "p24_gumbel_keep_probability_mean": keep_probability.mean(),
                "p24_gumbel_keep_probability_min": keep_probability.min(),
                "p24_gumbel_keep_probability_max": keep_probability.max(),
                "p24_gumbel_probability_entropy": entropy,
                "p24_gumbel_soft_keep_ratio": keep_probability.mean(),
                "p24_gumbel_hard_keep_ratio": hard_mask.mean(),
                "p24_gumbel_prompt_keep_mean": prompt_keep_probability.mean(),
                "p24_gumbel_prunable_prior_mean": noise_prior.mean(),
                "p24_gumbel_hub_prunable_prior_mean": noise_prior[hub_edge].mean()
                if bool(hub_edge.any())
                else zero,
                "p24_gumbel_nonhub_prunable_prior_mean": noise_prior[nonhub_edge].mean()
                if bool(nonhub_edge.any())
                else zero,
                "p24_gumbel_hub_keep_mean": keep_probability[hub_edge].mean()
                if bool(hub_edge.any())
                else zero,
                "p24_gumbel_nonhub_keep_mean": keep_probability[nonhub_edge].mean()
                if bool(nonhub_edge.any())
                else zero,
                "p24_gumbel_effectively_pruned_ratio": (
                    active_gate < 0.95
                ).to(dtype=base_edge_weight.dtype).mean(),
                "p24_gumbel_sample_variance": selection.var(unbiased=False),
                "p24_candidate_weight_delta_abs_mean": (
                    candidate_weight[active] - base_edge_weight[active]
                ).abs().mean(),
                "p24_candidate_gate_mean": active_gate.mean(),
                "p24_candidate_gate_min": active_gate.min(),
                "p24_candidate_gate_max": active_gate.max(),
            },
        }
