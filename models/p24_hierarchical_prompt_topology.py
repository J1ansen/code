"""Hierarchical, P23-anchored prompt topology adaptation for P24."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class P24HierarchicalPromptTopology(nn.Module):
    """Adapt P23 prompt edges with prompt, edge, and node-level controls.

    The selector starts as an exact identity. Prompt-to-node mass is
    redistributed within each receiving node and may only change globally
    inside a conservative, configured interval.
    """

    def __init__(self, repr_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.apply_to = str(self.config.get("apply_to", "prompt_to_node"))
        if self.apply_to not in {"prompt_to_node", "both"}:
            raise ValueError("hierarchical P24 apply_to must be 'prompt_to_node' or 'both'")
        if bool(self.config.get("allow_new_edges", False)):
            raise ValueError("hierarchical P24 only adapts P23 candidate edges")

        self.repr_dim = int(repr_dim)
        self.projection_dim = int(self.config.get("projection_dim", 16))
        self.hidden_dim = int(self.config.get("hidden_dim", 32))
        self.dropout = float(self.config.get("dropout", 0.10))
        self.delta_max = float(self.config.get("delta_max", 0.20))
        self.mass_floor = float(self.config.get("mass_floor", 0.80))
        self.mass_cap = float(self.config.get("mass_cap", 1.10))
        self.confidence_init_bias = float(self.config.get("confidence_init_bias", -2.0))
        self.adaptation_budget = float(self.config.get("adaptation_budget", 0.15))
        candidate_gate_floor = self.config.get("candidate_gate_floor")
        candidate_gate_cap = self.config.get("candidate_gate_cap")
        self.candidate_gate_floor = None if candidate_gate_floor is None else float(candidate_gate_floor)
        self.candidate_gate_cap = None if candidate_gate_cap is None else float(candidate_gate_cap)
        candidate_node_l1_budget = self.config.get("candidate_node_l1_budget")
        self.candidate_node_l1_budget = (
            None if candidate_node_l1_budget is None else float(candidate_node_l1_budget)
        )
        self.candidate_projection_steps = int(self.config.get("candidate_projection_steps", 6))
        self.keep_threshold = float(self.config.get("keep_threshold", 0.80))
        self.hub_quantile = float(self.config.get("hub_quantile", 0.80))
        self.prompt_noise_threshold = float(self.config.get("prompt_noise_threshold", 0.10))
        self.detach_context = bool(self.config.get("detach_context", True))
        self.center_prompt_scores = bool(self.config.get("center_prompt_scores", False))
        self.center_edge_scores_per_node = bool(self.config.get("center_edge_scores_per_node", False))
        self.mass_mode = str(self.config.get("mass_mode", "bounded")).lower()
        routing_cfg = dict(self.config.get("safe_routing", {}))
        self.safe_routing_enabled = bool(routing_cfg.get("enabled", False))
        self.route_init_bias = float(routing_cfg.get("init_bias", -4.0))
        self.route_max_adapt = float(routing_cfg.get("max_adapt", 1.0))
        self.route_max_no_prompt = float(routing_cfg.get("max_no_prompt", 0.50))
        multimodal_cfg = dict(self.config.get("multimodal_context", {}))
        self.multimodal_enabled = bool(multimodal_cfg.get("enabled", False))
        self.multimodal_ambiguity_threshold = float(multimodal_cfg.get("ambiguity_threshold", 0.65))
        self.route_warmup_epochs = int(routing_cfg.get("warmup_epochs", 0))
        self.detach_candidate_when_routing = bool(
            routing_cfg.get("detach_candidate_when_routing", False)
        )
        self.detach_route_context = bool(routing_cfg.get("detach_route_context", False))
        self.candidate_confidence_floor = float(self.config.get("candidate_confidence_floor", 0.0))

        if self.repr_dim <= 0 or self.projection_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("hierarchical P24 dimensions must be positive")
        if not 0.0 < self.mass_floor <= 1.0 <= self.mass_cap:
            raise ValueError("hierarchical P24 mass bounds must satisfy 0 < floor <= 1 <= cap")
        if self.delta_max < 0.0 or self.adaptation_budget < 0.0:
            raise ValueError("hierarchical P24 delta and budget must be non-negative")
        if (self.candidate_gate_floor is None) != (self.candidate_gate_cap is None):
            raise ValueError("candidate gate floor and cap must be configured together")
        if self.candidate_gate_floor is not None and not (
            0.0 < self.candidate_gate_floor <= 1.0 <= self.candidate_gate_cap
        ):
            raise ValueError("candidate gate bounds must satisfy 0 < floor <= 1 <= cap")
        if self.candidate_node_l1_budget is not None and self.candidate_node_l1_budget < 0.0:
            raise ValueError("candidate node L1 budget must be non-negative")
        if self.candidate_projection_steps <= 0:
            raise ValueError("candidate projection steps must be positive")
        if self.mass_mode not in {"bounded", "preserve"}:
            raise ValueError("hierarchical P24 mass_mode must be 'bounded' or 'preserve'")
        if not 0.0 <= self.route_max_adapt <= 1.0 or not 0.0 <= self.route_max_no_prompt <= 1.0:
            raise ValueError("hierarchical P24 route limits must be in [0, 1]")
        if not 0.0 <= self.candidate_confidence_floor <= 1.0:
            raise ValueError("hierarchical P24 candidate_confidence_floor must be in [0, 1]")

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

        # Six P23 feature statistics, graph priors, and two unsupervised
        # within-prompt mode descriptors.
        self.prompt_static_dim = 12
        prompt_input_dim = self.projection_dim + self.prompt_static_dim
        self.prompt_selector = nn.Sequential(
            nn.LayerNorm(prompt_input_dim),
            nn.Linear(prompt_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

        # Representation norm, graph degrees, predictive uncertainty, and the
        # frozen/adapted GP2F branch disagreement form the node-level context.
        self.node_static_dim = 8
        self.mode_edge_dim = 4
        edge_input_dim = (
            2 * self.projection_dim
            + self.prompt_static_dim
            + self.node_static_dim
            + self.mode_edge_dim
            + 1
        )
        self.edge_selector = nn.Sequential(
            nn.LayerNorm(edge_input_dim),
            nn.Linear(edge_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        confidence_input_dim = self.projection_dim + self.node_static_dim
        self.node_confidence = nn.Sequential(
            nn.LayerNorm(confidence_input_dim),
            nn.Linear(confidence_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.route_selector = nn.Sequential(
            nn.LayerNorm(confidence_input_dim),
            nn.Linear(confidence_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 2),
        )

        for head in (self.prompt_selector[-1], self.edge_selector[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.node_confidence[-1].weight)
        nn.init.constant_(self.node_confidence[-1].bias, self.confidence_init_bias)
        nn.init.zeros_(self.route_selector[-1].weight)
        nn.init.constant_(self.route_selector[-1].bias, self.route_init_bias)

    @staticmethod
    def _zero(ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    @staticmethod
    def _fit_feature_stats(stats: torch.Tensor, feature_count: int, ref: torch.Tensor) -> torch.Tensor:
        if stats.numel() == 0:
            return ref.new_zeros((feature_count, 6))
        stats = stats.to(device=ref.device, dtype=ref.dtype)
        if stats.size(0) != feature_count:
            raise ValueError("feature_stats must match the P23 prompt count")
        if stats.size(1) < 6:
            stats = torch.cat([stats, ref.new_zeros((feature_count, 6 - stats.size(1)))], dim=-1)
        return stats[:, :6]

    @staticmethod
    def _normalized_log(values: torch.Tensor) -> torch.Tensor:
        logged = torch.log1p(values.clamp_min(0.0))
        return logged / logged.mean().clamp_min(1e-6)

    def _usage_context(self, usage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        usage_norm = self._normalized_log(usage)
        if usage.numel() <= 1:
            return usage_norm, torch.zeros_like(usage_norm)
        order = torch.argsort(usage.detach())
        rank = torch.empty_like(usage_norm)
        rank[order] = torch.linspace(0.0, 1.0, usage.numel(), device=usage.device, dtype=usage.dtype)
        hub_rank = ((rank - self.hub_quantile) / max(1e-6, 1.0 - self.hub_quantile)).clamp(0.0, 1.0)
        return usage_norm, hub_rank

    def _node_context(
        self,
        *,
        node_repr: torch.Tensor,
        edge_node: torch.Tensor,
        active_mask: torch.Tensor,
        original_edge_index: torch.Tensor | None,
        no_prompt_logits: torch.Tensor | None,
        h_adp_no_prompt: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = node_repr.dtype
        device = node_repr.device
        num_nodes = int(node_repr.size(0))
        repr_norm = node_repr.norm(dim=-1)
        repr_norm = repr_norm / repr_norm.mean().clamp_min(1e-6)

        prompt_degree = node_repr.new_zeros(num_nodes)
        if bool(active_mask.any()):
            prompt_degree.index_add_(0, edge_node[active_mask], torch.ones_like(edge_node[active_mask], dtype=dtype))
        prompt_degree = self._normalized_log(prompt_degree)

        original_degree = node_repr.new_zeros(num_nodes)
        if isinstance(original_edge_index, torch.Tensor) and original_edge_index.numel() > 0:
            target = original_edge_index[1].to(device=device, dtype=torch.long)
            original_degree.index_add_(0, target, torch.ones_like(target, dtype=dtype))
        original_degree = self._normalized_log(original_degree)

        entropy = node_repr.new_zeros(num_nodes)
        margin = node_repr.new_zeros(num_nodes)
        confidence = node_repr.new_zeros(num_nodes)
        if isinstance(no_prompt_logits, torch.Tensor) and no_prompt_logits.size(0) == num_nodes:
            logits = no_prompt_logits.to(device=device, dtype=dtype).detach()
            probability = F.softmax(logits, dim=-1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            if probability.size(1) > 1:
                entropy = entropy / torch.log(probability.new_tensor(float(probability.size(1)))).clamp_min(1e-6)
                top_two = torch.topk(probability, k=2, dim=-1).values
                margin = top_two[:, 0] - top_two[:, 1]
            confidence = probability.max(dim=-1).values

        branch_cosine = node_repr.new_ones(num_nodes)
        branch_delta = node_repr.new_zeros(num_nodes)
        if isinstance(h_adp_no_prompt, torch.Tensor) and h_adp_no_prompt.shape == node_repr.shape:
            adapted = h_adp_no_prompt.to(device=device, dtype=dtype).detach()
            branch_cosine = F.cosine_similarity(node_repr, adapted, dim=-1)
            branch_delta = (node_repr - adapted).norm(dim=-1)
            branch_delta = branch_delta / node_repr.norm(dim=-1).mean().clamp_min(1e-6)

        return torch.stack(
            [
                repr_norm,
                prompt_degree,
                original_degree,
                entropy,
                margin,
                confidence,
                branch_cosine,
                branch_delta,
            ],
            dim=-1,
        )

    def _identity_output(self, base_edge_weight: torch.Tensor) -> dict[str, Any]:
        edge_gate = torch.ones_like(base_edge_weight)
        zero = self._zero(base_edge_weight)
        return {
            "edge_weight": base_edge_weight,
            "edge_gate": edge_gate,
            "prompt_gate": base_edge_weight.new_ones(0),
            "node_confidence": base_edge_weight.new_zeros(0),
            "losses": {
                "keep_static": zero,
                "hub_suppression": zero,
                "prompt_homophily": zero,
                "edge_budget": zero,
            },
            "stats": {
                "p24_enabled": zero,
                "p24_hierarchical_enabled": zero,
                "p24_edge_gate_mean": edge_gate.mean() if edge_gate.numel() > 0 else zero,
                "p24_edge_gate_min": edge_gate.min() if edge_gate.numel() > 0 else zero,
                "p24_edge_gate_max": edge_gate.max() if edge_gate.numel() > 0 else zero,
                "p24_prompt_to_node_gate_mean": edge_gate.mean() if edge_gate.numel() > 0 else zero,
                "p24_node_to_prompt_gate_mean": edge_gate.mean() if edge_gate.numel() > 0 else zero,
                "p24_hub_gate_mean": zero,
                "p24_nonhub_gate_mean": zero,
                "p24_edge_keep_ratio": edge_gate.new_tensor(1.0) if edge_gate.numel() > 0 else zero,
                "p24_learned_gate_mean": edge_gate.mean() if edge_gate.numel() > 0 else zero,
                "p24_effective_gate_mean": edge_gate.mean() if edge_gate.numel() > 0 else zero,
                "p24_mix_ratio": zero,
                "p24_gate_above_one_ratio": zero,
                "p24_weight_delta_mean": zero,
                "p24_weight_delta_abs_mean": zero,
                "p24_prompt_gate_mean": zero,
                "p24_node_confidence_mean": zero,
                "p24_p23_fallback_ratio": zero,
                "p24_node_mass_ratio_mean": base_edge_weight.new_tensor(1.0),
                "p24_node_mass_error_mean": zero,
                "p24_mass_at_lower_bound_ratio": zero,
                "p24_positive_delta_ratio": zero,
                "p24_negative_delta_ratio": zero,
                "p24_noisy_prompt_suppression_ratio": zero,
                "p24_helpful_prompt_enhancement_ratio": zero,
                "p24_train_edge_homophily_raw": zero,
                "p24_train_edge_homophily_weighted": zero,
                "p24_train_same_class_gate_mean": zero,
                "p24_train_offclass_gate_mean": zero,
                "p24_route_p23_mean": base_edge_weight.new_tensor(1.0),
                "p24_route_adapt_mean": zero,
                "p24_route_no_prompt_mean": zero,
                "p24_safe_routing_enabled": zero,
                "p24_multimodal_enabled": zero,
                "p24_prompt_mode_entropy_mean": zero,
                "p24_prompt_mode_separation_mean": zero,
                "p24_prompt_mode_ambiguous_ratio": zero,
            },
            "route_logits": base_edge_weight.new_zeros((0, 2)),
            "route_probs": base_edge_weight.new_zeros((0, 3)),
            "candidate_edge_weight": base_edge_weight,
        }

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
        feature_usage: torch.Tensor | None = None,
        original_edge_index: torch.Tensor | None = None,
        no_prompt_logits: torch.Tensor | None = None,
        h_adp_no_prompt: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        edge_count = int(base_edge_weight.numel())
        if not self.enabled or edge_count == 0:
            return self._identity_output(base_edge_weight)

        device = base_edge_weight.device
        dtype = base_edge_weight.dtype
        edge_node = edge_node_index.to(device=device, dtype=torch.long)
        edge_feature = edge_feature_local.to(device=device, dtype=torch.long)
        direction = edge_direction.to(device=device, dtype=torch.long)
        if edge_node.numel() != edge_count or edge_feature.numel() != edge_count or direction.numel() != edge_count:
            raise ValueError("P24 edge metadata must match base_edge_weight")
        p2n_mask = direction == 1
        n2p_mask = direction == 0
        active_mask = torch.ones_like(direction, dtype=torch.bool) if self.apply_to == "both" else p2n_mask
        if not bool(active_mask.any()):
            return self._identity_output(base_edge_weight)

        feature_count = int(edge_feature.max().item()) + 1
        node = node_repr.to(device=device, dtype=dtype)
        if self.detach_context:
            node = node.detach()
        feature_stats = self._fit_feature_stats(feature_stats, feature_count, base_edge_weight)
        feature_gate = feature_gate.to(device=device, dtype=dtype)
        if feature_gate.numel() != feature_count:
            feature_gate = base_edge_weight.new_ones(feature_count)
        usage = (
            feature_usage.to(device=device, dtype=dtype)
            if isinstance(feature_usage, torch.Tensor) and feature_usage.numel() == feature_count
            else base_edge_weight.new_ones(feature_count)
        )
        usage_norm, hub_rank = self._usage_context(usage)

        prototype = node.new_zeros((feature_count, node.size(1)))
        prototype_denom = node.new_zeros(feature_count)
        prototype_mask = p2n_mask if bool(p2n_mask.any()) else active_mask
        prototype_weight = base_edge_weight[prototype_mask].detach().clamp_min(0.0)
        prototype.index_add_(
            0,
            edge_feature[prototype_mask],
            node[edge_node[prototype_mask]] * prototype_weight.unsqueeze(-1),
        )
        prototype_denom.index_add_(0, edge_feature[prototype_mask], prototype_weight)
        prototype = prototype / prototype_denom.clamp_min(1e-6).unsqueeze(-1)

        mode_prototype = prototype.unsqueeze(1).expand(-1, 2, -1).clone()
        mode_entropy = base_edge_weight.new_zeros(feature_count)
        mode_separation = base_edge_weight.new_zeros(feature_count)
        if self.multimodal_enabled:
            second_moment = node.new_zeros((feature_count, node.size(1)))
            second_moment.index_add_(
                0,
                edge_feature[prototype_mask],
                node[edge_node[prototype_mask]].pow(2) * prototype_weight.unsqueeze(-1),
            )
            second_moment = second_moment / prototype_denom.clamp_min(1e-6).unsqueeze(-1)
            feature_variance = (second_moment - prototype.pow(2)).clamp_min(0.0)
            split_dimension = feature_variance.argmax(dim=-1)
            prototype_feature = edge_feature[prototype_mask]
            prototype_node = edge_node[prototype_mask]
            split_value = node[prototype_node, split_dimension[prototype_feature]]
            split_center = prototype[prototype_feature, split_dimension[prototype_feature]]
            mode_assignment = (split_value >= split_center).to(dtype=torch.long)
            mode_weight = node.new_zeros((feature_count, 2))
            mode_sum = node.new_zeros((feature_count, 2, node.size(1)))
            flat_mode = prototype_feature * 2 + mode_assignment
            mode_weight.view(-1).index_add_(0, flat_mode, prototype_weight)
            mode_sum.view(feature_count * 2, node.size(1)).index_add_(
                0,
                flat_mode,
                node[prototype_node] * prototype_weight.unsqueeze(-1),
            )
            learned_mode = mode_sum / mode_weight.clamp_min(1e-6).unsqueeze(-1)
            nonempty_mode = mode_weight > 0.0
            mode_prototype = torch.where(
                nonempty_mode.unsqueeze(-1),
                learned_mode,
                mode_prototype,
            )
            mode_probability = mode_weight / mode_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            mode_entropy = -(
                mode_probability * mode_probability.clamp_min(1e-12).log()
            ).sum(dim=-1) / torch.log(mode_probability.new_tensor(2.0))
            mode_separation = 1.0 - F.cosine_similarity(
                mode_prototype[:, 0],
                mode_prototype[:, 1],
                dim=-1,
            )

        base_feature_weight = base_edge_weight.new_zeros(feature_count)
        base_feature_count = base_edge_weight.new_zeros(feature_count)
        base_feature_weight.index_add_(0, edge_feature[prototype_mask], base_edge_weight[prototype_mask].detach())
        base_feature_count.index_add_(
            0,
            edge_feature[prototype_mask],
            torch.ones_like(base_edge_weight[prototype_mask]),
        )
        base_feature_weight = base_feature_weight / base_feature_count.clamp_min(1.0)
        base_feature_weight = base_feature_weight / base_feature_weight.mean().clamp_min(1e-6)

        prompt_static = torch.cat(
            [
                feature_stats,
                feature_gate.detach().unsqueeze(-1),
                usage_norm.unsqueeze(-1),
                hub_rank.unsqueeze(-1),
                base_feature_weight.unsqueeze(-1),
                mode_entropy.unsqueeze(-1),
                mode_separation.unsqueeze(-1),
            ],
            dim=-1,
        )
        node_static = self._node_context(
            node_repr=node,
            edge_node=edge_node,
            active_mask=active_mask,
            original_edge_index=original_edge_index,
            no_prompt_logits=no_prompt_logits,
            h_adp_no_prompt=h_adp_no_prompt,
        )
        node_latent = self.node_projection(node)
        prompt_latent = self.prompt_projection(prototype)
        mode_latent = self.prompt_projection(mode_prototype.reshape(feature_count * 2, -1)).reshape(
            feature_count,
            2,
            self.projection_dim,
        )
        raw_prompt_score = self.prompt_selector(torch.cat([prompt_latent, prompt_static], dim=-1)).squeeze(-1)
        prompt_score = raw_prompt_score
        if self.center_prompt_scores and prompt_score.numel() > 0:
            prompt_score = prompt_score - prompt_score.mean()
        confidence = torch.sigmoid(
            self.node_confidence(torch.cat([node_latent, node_static], dim=-1)).squeeze(-1)
        )

        active_node = edge_node[active_mask]
        active_feature = edge_feature[active_mask]
        num_nodes = int(node.size(0))
        node_edge = node_latent[active_node]
        prompt_edge = prompt_latent[active_feature]
        edge_mode = mode_latent[active_feature]
        mode_similarity = F.cosine_similarity(node_edge.unsqueeze(1), edge_mode, dim=-1)
        mode_features = torch.stack(
            [
                mode_similarity[:, 0],
                mode_similarity[:, 1],
                mode_similarity.max(dim=-1).values,
                (mode_similarity[:, 0] - mode_similarity[:, 1]).abs(),
            ],
            dim=-1,
        )
        edge_input = torch.cat(
            [
                node_edge * prompt_edge,
                (node_edge - prompt_edge).abs(),
                prompt_static[active_feature],
                node_static[active_node],
                mode_features,
                base_edge_weight[active_mask].detach().unsqueeze(-1),
            ],
            dim=-1,
        )
        raw_edge_score = self.edge_selector(edge_input).squeeze(-1)
        edge_score = raw_edge_score
        if self.center_edge_scores_per_node:
            edge_score_sum = base_edge_weight.new_zeros(num_nodes)
            edge_score_weight = base_edge_weight.new_zeros(num_nodes)
            detached_active_weight = base_edge_weight[active_mask].detach().clamp_min(0.0)
            edge_score_sum.index_add_(0, active_node, detached_active_weight * edge_score)
            edge_score_weight.index_add_(0, active_node, detached_active_weight)
            edge_score = edge_score - edge_score_sum[active_node] / edge_score_weight[active_node].clamp_min(1e-6)
        combined_score = prompt_score[active_feature] + edge_score
        candidate_confidence = self.candidate_confidence_floor + (
            1.0 - self.candidate_confidence_floor
        ) * confidence
        log_adjustment = candidate_confidence[active_node] * self.delta_max * torch.tanh(combined_score)
        raw_weight = base_edge_weight[active_mask] * torch.exp(log_adjustment)

        base_mass = base_edge_weight.new_zeros(num_nodes)
        raw_mass = base_edge_weight.new_zeros(num_nodes)
        base_mass.index_add_(0, active_node, base_edge_weight[active_mask])
        raw_mass.index_add_(0, active_node, raw_weight)

        prompt_delta = torch.tanh(prompt_score)
        mass_signal_sum = base_edge_weight.new_zeros(num_nodes)
        mass_signal_sum.index_add_(
            0,
            active_node,
            base_edge_weight[active_mask] * prompt_delta[active_feature],
        )
        mass_signal = mass_signal_sum / base_mass.clamp_min(1e-6)
        if self.mass_mode == "preserve":
            mass_scale = torch.ones_like(confidence)
        else:
            negative_scale = (1.0 - self.mass_floor) * mass_signal
            positive_scale = (self.mass_cap - 1.0) * mass_signal
            mass_scale = 1.0 + confidence * torch.where(mass_signal < 0.0, negative_scale, positive_scale)
            mass_scale = mass_scale.clamp(self.mass_floor, self.mass_cap)
        target_mass = base_mass * mass_scale
        candidate_active_weight = raw_weight * target_mass[active_node] / raw_mass[active_node].clamp_min(1e-6)

        candidate_gate_before_projection = candidate_active_weight / base_edge_weight[active_mask].clamp_min(1e-12)
        candidate_gate_clamp_ratio = candidate_active_weight.new_tensor(0.0)
        if self.candidate_gate_floor is not None and self.candidate_gate_cap is not None:
            lower = float(self.candidate_gate_floor)
            upper = float(self.candidate_gate_cap)
            candidate_gate_clamp_ratio = (
                (candidate_gate_before_projection < lower) | (candidate_gate_before_projection > upper)
            ).to(dtype=dtype).mean()
            projected_gate = candidate_gate_before_projection.clamp(lower, upper)

            # Alternating bounded scaling keeps each receiving node close to its
            # requested prompt mass without allowing normalization to recreate
            # extreme per-edge gates.
            feasible_target_mass = target_mass.clamp(base_mass * lower, base_mass * upper)
            for _ in range(self.candidate_projection_steps):
                projected_mass = base_edge_weight.new_zeros(num_nodes)
                projected_mass.index_add_(
                    0,
                    active_node,
                    base_edge_weight[active_mask] * projected_gate,
                )
                correction = feasible_target_mass / projected_mass.clamp_min(1e-12)
                projected_gate = (projected_gate * correction[active_node]).clamp(lower, upper)
            candidate_active_weight = base_edge_weight[active_mask] * projected_gate

        candidate_node_l1 = base_edge_weight.new_zeros(num_nodes)
        candidate_node_l1.index_add_(
            0,
            active_node,
            (candidate_active_weight - base_edge_weight[active_mask]).abs(),
        )
        candidate_node_l1 = candidate_node_l1 / base_mass.clamp_min(1e-12)
        if self.candidate_node_l1_budget is not None:
            budget = float(self.candidate_node_l1_budget)
            shrink = (budget / candidate_node_l1.clamp_min(1e-12)).clamp(max=1.0)
            candidate_active_weight = base_edge_weight[active_mask] + shrink[active_node] * (
                candidate_active_weight - base_edge_weight[active_mask]
            )
            candidate_node_l1 = base_edge_weight.new_zeros(num_nodes)
            candidate_node_l1.index_add_(
                0,
                active_node,
                (candidate_active_weight - base_edge_weight[active_mask]).abs(),
            )
            candidate_node_l1 = candidate_node_l1 / base_mass.clamp_min(1e-12)

        # The projected candidate mass is the actual target used by routing and
        # diagnostics. This is exact for preserve mode and remains feasible for
        # bounded-mass configurations.
        target_mass = base_edge_weight.new_zeros(num_nodes)
        target_mass.index_add_(0, active_node, candidate_active_weight)
        mass_scale = torch.where(base_mass > 0.0, target_mass / base_mass.clamp_min(1e-12), torch.ones_like(base_mass))

        route_input = torch.cat([node_latent, node_static], dim=-1)
        if self.detach_route_context:
            route_input = route_input.detach()
        route_logits = self.route_selector(route_input)
        current_epoch = 0 if epoch is None else int(epoch)
        route_stage_active = self.safe_routing_enabled and current_epoch > self.route_warmup_epochs
        if route_stage_active:
            init_probability = torch.sigmoid(route_logits.new_tensor(self.route_init_bias))
            route_signal = (
                (torch.sigmoid(route_logits) - init_probability) / (1.0 - init_probability).clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            adapt_route = self.route_max_adapt * route_signal[:, 0]
            raw_no_prompt_route = self.route_max_no_prompt * route_signal[:, 1]
            p24_route = adapt_route
            no_prompt_route = (1.0 - adapt_route) * raw_no_prompt_route
            p23_route = (1.0 - adapt_route) * (1.0 - raw_no_prompt_route)
            routed_candidate_weight = (
                candidate_active_weight.detach()
                if self.detach_candidate_when_routing
                else candidate_active_weight
            )
            final_active_weight = (
                p23_route[active_node] * base_edge_weight[active_mask]
                + p24_route[active_node] * routed_candidate_weight
            )
        elif self.safe_routing_enabled:
            p23_route = confidence.new_ones(num_nodes)
            p24_route = confidence.new_zeros(num_nodes)
            no_prompt_route = confidence.new_zeros(num_nodes)
            final_active_weight = base_edge_weight[active_mask]
        else:
            p23_route = confidence.new_zeros(num_nodes)
            p24_route = confidence.new_ones(num_nodes)
            no_prompt_route = confidence.new_zeros(num_nodes)
            final_active_weight = candidate_active_weight
        route_probs = torch.stack([p23_route, p24_route, no_prompt_route], dim=-1)

        final_weight = base_edge_weight.clone()
        final_weight[active_mask] = final_active_weight
        edge_gate = torch.ones_like(base_edge_weight)
        valid_base = active_mask & (base_edge_weight.abs() > 1e-12)
        edge_gate[valid_base] = final_weight[valid_base] / base_edge_weight[valid_base]

        final_mass = base_edge_weight.new_zeros(num_nodes)
        final_mass.index_add_(0, active_node, final_active_weight)
        active_nodes = base_mass > 0.0
        mass_ratio = final_mass[active_nodes] / base_mass[active_nodes].clamp_min(1e-6)
        if self.safe_routing_enabled:
            mass_target = (
                p23_route[active_nodes]
                + p24_route[active_nodes] * mass_scale[active_nodes]
            )
        else:
            mass_target = mass_scale[active_nodes]
        mass_error = (mass_ratio - mass_target).abs()

        active_gate = edge_gate[active_mask]
        candidate_gate = torch.ones_like(base_edge_weight)
        active_base_valid = base_edge_weight[active_mask].abs() > 1e-12
        candidate_valid = active_mask & (base_edge_weight.abs() > 1e-12)
        candidate_gate[candidate_valid] = (
            candidate_active_weight[active_base_valid]
            / base_edge_weight[active_mask][active_base_valid]
        )
        gate_delta = active_gate - 1.0
        keep_static = gate_delta.pow(2).mean()
        mean_abs_delta = gate_delta.abs().mean()
        edge_budget = F.relu(mean_abs_delta - self.adaptation_budget).pow(2)
        hub_edge_mask = p2n_mask & (hub_rank[edge_feature] > 0.0)
        nonhub_edge_mask = p2n_mask & ~hub_edge_mask
        zero = self._zero(base_edge_weight)
        weight_delta = final_weight - base_edge_weight
        prompt_gate = torch.exp(self.delta_max * prompt_delta)
        stats = {
            "p24_enabled": base_edge_weight.new_tensor(1.0),
            "p24_hierarchical_enabled": base_edge_weight.new_tensor(1.0),
            "p24_edge_gate_mean": edge_gate.mean(),
            "p24_edge_gate_min": edge_gate.min(),
            "p24_edge_gate_max": edge_gate.max(),
            "p24_prompt_to_node_gate_mean": edge_gate[p2n_mask].mean() if bool(p2n_mask.any()) else zero,
            "p24_node_to_prompt_gate_mean": edge_gate[n2p_mask].mean() if bool(n2p_mask.any()) else zero,
            "p24_hub_gate_mean": edge_gate[hub_edge_mask].mean() if bool(hub_edge_mask.any()) else zero,
            "p24_nonhub_gate_mean": edge_gate[nonhub_edge_mask].mean() if bool(nonhub_edge_mask.any()) else zero,
            "p24_edge_keep_ratio": (active_gate >= self.keep_threshold).to(dtype=dtype).mean(),
            "p24_learned_gate_mean": active_gate.mean(),
            "p24_effective_gate_mean": active_gate.mean(),
            "p24_mix_ratio": base_edge_weight.new_tensor(1.0),
            "p24_gate_above_one_ratio": (active_gate > 1.0).to(dtype=dtype).mean(),
            "p24_weight_delta_mean": weight_delta[active_mask].mean(),
            "p24_weight_delta_abs_mean": weight_delta[active_mask].abs().mean(),
            "p24_prompt_gate_mean": prompt_gate.mean(),
            "p24_prompt_gate_min": prompt_gate.min(),
            "p24_prompt_gate_max": prompt_gate.max(),
            "p24_node_confidence_mean": confidence.mean(),
            "p24_p23_fallback_ratio": (confidence < 0.20).to(dtype=dtype).mean(),
            "p24_node_mass_ratio_mean": mass_ratio.mean() if mass_ratio.numel() > 0 else base_edge_weight.new_tensor(1.0),
            "p24_node_mass_error_mean": mass_error.mean() if mass_error.numel() > 0 else zero,
            "p24_mass_at_lower_bound_ratio": (
                (mass_target <= self.mass_floor + 1e-5).to(dtype=dtype).mean() if mass_target.numel() > 0 else zero
            ),
            "p24_positive_delta_ratio": (gate_delta > 1e-6).to(dtype=dtype).mean(),
            "p24_negative_delta_ratio": (gate_delta < -1e-6).to(dtype=dtype).mean(),
            "p24_noisy_prompt_suppression_ratio": (
                (prompt_delta < -self.prompt_noise_threshold).to(dtype=dtype).mean()
            ),
            "p24_helpful_prompt_enhancement_ratio": (
                (prompt_delta > self.prompt_noise_threshold).to(dtype=dtype).mean()
            ),
            "p24_prompt_score_abs_mean": prompt_score.abs().mean(),
            "p24_edge_score_abs_mean": edge_score.abs().mean(),
            "p24_raw_prompt_score_abs_mean": raw_prompt_score.abs().mean(),
            "p24_raw_edge_score_abs_mean": raw_edge_score.abs().mean(),
            "p24_prompt_score_centered": base_edge_weight.new_tensor(float(self.center_prompt_scores)),
            "p24_edge_score_centered": base_edge_weight.new_tensor(float(self.center_edge_scores_per_node)),
            "p24_mass_preserved": base_edge_weight.new_tensor(float(self.mass_mode == "preserve")),
            "p24_train_edge_homophily_raw": zero,
            "p24_train_edge_homophily_weighted": zero,
            "p24_train_same_class_gate_mean": zero,
            "p24_train_offclass_gate_mean": zero,
            "p24_route_p23_mean": p23_route.mean(),
            "p24_route_adapt_mean": p24_route.mean(),
            "p24_route_no_prompt_mean": no_prompt_route.mean(),
            "p24_route_p23_active_ratio": (p23_route >= 0.50).to(dtype=dtype).mean(),
            "p24_route_adapt_active_ratio": (p24_route >= 0.10).to(dtype=dtype).mean(),
            "p24_route_no_prompt_active_ratio": (no_prompt_route >= 0.10).to(dtype=dtype).mean(),
            "p24_safe_routing_enabled": base_edge_weight.new_tensor(float(self.safe_routing_enabled)),
            "p24_multimodal_enabled": base_edge_weight.new_tensor(float(self.multimodal_enabled)),
            "p24_prompt_mode_entropy_mean": mode_entropy.mean(),
            "p24_prompt_mode_separation_mean": mode_separation.mean(),
            "p24_prompt_mode_ambiguous_ratio": (
                mode_entropy >= self.multimodal_ambiguity_threshold
            ).to(dtype=dtype).mean(),
            "p24_edge_nearest_mode_similarity_mean": mode_similarity.max(dim=-1).values.mean(),
            "p24_route_stage_active": base_edge_weight.new_tensor(float(route_stage_active)),
            "p24_route_warmup_epochs": base_edge_weight.new_tensor(float(self.route_warmup_epochs)),
            "p24_route_candidate_detached": base_edge_weight.new_tensor(
                float(self.detach_candidate_when_routing and route_stage_active)
            ),
            "p24_route_context_detached": base_edge_weight.new_tensor(float(self.detach_route_context)),
            "p24_candidate_confidence_mean": candidate_confidence.mean(),
            "p24_candidate_gate_mean": candidate_gate[active_mask].mean(),
            "p24_candidate_gate_min": candidate_gate[active_mask].min(),
            "p24_candidate_gate_max": candidate_gate[active_mask].max(),
            "p24_candidate_weight_delta_abs_mean": (
                candidate_active_weight - base_edge_weight[active_mask]
            ).abs().mean(),
            "p24_candidate_gate_clamp_ratio": candidate_gate_clamp_ratio,
            "p24_candidate_node_l1_mean": (
                candidate_node_l1[active_nodes].mean() if bool(active_nodes.any()) else zero
            ),
            "p24_candidate_node_l1_max": (
                candidate_node_l1[active_nodes].max() if bool(active_nodes.any()) else zero
            ),
            "p24_candidate_projection_enabled": base_edge_weight.new_tensor(
                float(self.candidate_gate_floor is not None or self.candidate_node_l1_budget is not None)
            ),
        }
        candidate_weight = base_edge_weight.clone()
        candidate_weight[active_mask] = candidate_active_weight
        return {
            "edge_weight": final_weight,
            "edge_gate": edge_gate,
            "prompt_gate": prompt_gate,
            "node_confidence": confidence,
            "route_logits": route_logits,
            "route_probs": route_probs,
            "candidate_edge_weight": candidate_weight,
            "losses": {
                "keep_static": keep_static,
                "hub_suppression": zero,
                "prompt_homophily": zero,
                "edge_budget": edge_budget,
            },
            "stats": stats,
        }
