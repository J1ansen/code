"""Prompting Adaptation on the candidate prompt graph."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from models.p24_counterfactual_utility_router import P24CounterfactualUtilityRouter
from models.p24_cross_fitted_utility import P24CrossFittedUtilityLearner
from models.p24_degree_relative_receiver import P24DegreeRelativeReceiver
from models.p24_gumbel_prompt_edge_selector import P24GumbelPromptEdgeSelector
from models.p24_hierarchical_prompt_topology import P24HierarchicalPromptTopology
from models.p24_relation_aware_hub_selector import P24RelationAwareHubEdgeSelector
from models.prompt_graph_construction import PromptGraphConstruction


class PromptingAdaptation(PromptGraphConstruction):
    """Combine prompt construction with target-specific prompting weight learning."""

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        resolved = dict(config or {})
        denoising_cfg = dict(resolved.get("prompt_edge_denoising", {}))
        denoising_cfg.setdefault("enabled", True)
        denoising_cfg.setdefault("allow_new_edges", False)
        denoising_cfg.setdefault("apply_to", "prompt_to_node")
        architecture = str(denoising_cfg.get("architecture", "legacy_gate")).lower()
        if architecture == "independent_edge":
            denoising_cfg["descriptor_type"] = "independent_edge"
            if not resolved.get("static_graph", True):
                raise ValueError("Independent edge parameters require a static candidate graph")
        elif denoising_cfg.get("descriptor_type") == "independent_edge":
            raise ValueError("Independent descriptor requires architecture='independent_edge'")
        if (
            denoising_cfg.get("descriptor_type") in {"representation_concat", "raw_representation_concat"}
            and architecture != "legacy_gate"
        ):
            raise ValueError("Representation descriptors require architecture='legacy_gate'")
        self._hierarchical_mode = architecture in {
            "hierarchical",
            "hierarchical_conditional",
            "p23_anchored_hierarchical",
        }
        self._gumbel_mode = architecture in {
            "hierarchical_gumbel",
            "gumbel_prompt_edge_selector",
            "p23_anchored_gumbel",
        }
        self._relation_mode = architecture in {
            "relation_aware_hub",
            "relation_aware_gumbel",
            "unsupervised_relation_hub",
        }
        self._candidate_mode = self._hierarchical_mode or self._gumbel_mode or self._relation_mode
        self._hierarchical_config = dict(denoising_cfg)
        self._freeze_p23_core = bool(denoising_cfg.get("freeze_p23_core", False))
        self._selector_grad_scale = float(denoising_cfg.get("selector_grad_scale", 1.0))
        safe_routing_cfg = dict(denoising_cfg.get("safe_routing", {}))
        counterfactual_router_cfg = dict(denoising_cfg.get("counterfactual_router", {}))
        receiver_cfg = dict(denoising_cfg.get("degree_relative_receiver", {}))
        crossfit_cfg = dict(denoising_cfg.get("crossfit_utility", {}))
        self._counterfactual_router_config = counterfactual_router_cfg
        self.counterfactual_utility_enabled = bool(
            counterfactual_router_cfg.get("enabled", False)
            or (
                safe_routing_cfg.get("enabled", False)
                and safe_routing_cfg.get("counterfactual_teacher", False)
            )
        )
        if self._selector_grad_scale <= 0.0:
            raise ValueError("prompt_edge_denoising.selector_grad_scale must be positive")
        if self._candidate_mode:
            # Let the P23 parent produce its untouched static prompt graph. The
            # P24 selector is created only after P23 initialization completes.
            resolved["prompt_edge_denoising"] = {"enabled": False}
        else:
            resolved["prompt_edge_denoising"] = denoising_cfg
        super().__init__(source_dim, hidden_dim, resolved)

        self.hierarchical_selector: P24HierarchicalPromptTopology | None = None
        self.gumbel_selector: P24GumbelPromptEdgeSelector | None = None
        self.relation_selector: P24RelationAwareHubEdgeSelector | None = None
        self.counterfactual_router: P24CounterfactualUtilityRouter | None = None
        self.degree_relative_receiver: P24DegreeRelativeReceiver | None = None
        self.crossfit_utility_learner: P24CrossFittedUtilityLearner | None = None
        if self._candidate_mode:
            if self._freeze_p23_core:
                for parameter in super().parameters():
                    parameter.requires_grad = False
            if self._hierarchical_mode:
                self.hierarchical_selector = P24HierarchicalPromptTopology(hidden_dim, self._hierarchical_config)
                candidate_selector: nn.Module = self.hierarchical_selector
            elif self._relation_mode:
                self.relation_selector = P24RelationAwareHubEdgeSelector(
                    hidden_dim, self._hierarchical_config
                )
                candidate_selector = self.relation_selector
            else:
                self.gumbel_selector = P24GumbelPromptEdgeSelector(hidden_dim, self._hierarchical_config)
                candidate_selector = self.gumbel_selector
            if self._selector_grad_scale != 1.0:
                for parameter in candidate_selector.parameters():
                    parameter.register_hook(lambda gradient, scale=self._selector_grad_scale: gradient * scale)
            # This asks the existing runner to provide the frozen no-prompt
            # branch context without changing the P23 runner path.
            self.pool_strategy = "utility_structural"
            if bool(counterfactual_router_cfg.get("enabled", False)):
                self.counterfactual_router = P24CounterfactualUtilityRouter(counterfactual_router_cfg)
            if bool(receiver_cfg.get("enabled", False)):
                self.degree_relative_receiver = P24DegreeRelativeReceiver(receiver_cfg)
            if bool(crossfit_cfg.get("enabled", False)):
                self.crossfit_utility_learner = P24CrossFittedUtilityLearner(crossfit_cfg)

    def _build_graph(self, *, z, x_raw, edge_index, train_mask, labels):
        filtering = dict(self.config.get("filtering_support", {}))
        if not filtering.get("enabled", False):
            return super()._build_graph(z=z, x_raw=x_raw, edge_index=edge_index,
                                       train_mask=train_mask, labels=labels)
        if self.static_pool.enabled or self._candidate_mode:
            raise ValueError("Balanced filtering requires legacy_gate and disabled static_pool")
        if labels is None:
            raise ValueError("Balanced filtering requires training labels")
        shots = int(filtering.get("shots_per_class", 5))
        if shots < 1:
            raise ValueError("filtering_support.shots_per_class must be positive")
        # A local generator leaves model/dropout random streams unchanged.
        generator = torch.Generator(device="cpu").manual_seed(int(filtering.get("seed", 12345)))
        train = train_mask.bool()
        selected = torch.zeros_like(train)
        for class_id in range(self.num_classes):
            indices = torch.where(train & (labels == class_id))[0].cpu()
            if indices.numel() < shots:
                raise ValueError(f"Class {class_id} has fewer than {shots} training nodes")
            indices = indices[torch.randperm(indices.numel(), generator=generator)[:shots]]
            selected[indices.to(selected.device)] = True
        graph = super()._build_graph(z=z, x_raw=x_raw, edge_index=edge_index,
                                    train_mask=selected, labels=labels)
        ids = graph["feature_ids"]
        # Only reliability statistics use the subset. Initialization retains all
        # training evidence, with the subset-estimated dominant class.
        if ids.numel():
            membership = self._token_membership(x_raw.to(z.device))[:, ids]
            weights = z.new_full(membership.shape, self.unlabeled_node_weight)
            match = labels[train, None] == graph["feature_dominant_class"][None, :]
            weights[train] = torch.where(match, self.label_node_weight, self.offclass_label_weight)
            weighted = membership.to(z.dtype) * weights
            prompt_x = weighted.t() @ z / weighted.sum(0).clamp_min(1.0)[:, None]
            graph["prompt_x"] = prompt_x.detach() if self.detach_feature_prompt else prompt_x
        graph["filtering_support_mask"] = selected.detach()
        graph["aux"]["p24_filtering_support_count"] = z.new_tensor(float(selected.sum()))
        graph["aux"]["p24_initialization_support_count"] = z.new_tensor(float(train.sum()))
        return graph

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        denoiser = self.prompt_edge_denoiser
        if denoiser.descriptor_type in {"representation_concat", "raw_representation_concat", "independent_edge"}:
            # The shared runner toggles trainability through this interface.
            # Keep the checkpoint-compatible, unused legacy head frozen.
            denoiser.edge_gate.requires_grad_(False)
            unused = {id(parameter) for parameter in denoiser.edge_gate.parameters()}
            yield from (parameter for parameter in super().parameters(recurse=recurse) if id(parameter) not in unused)
            return
        selector = (
            getattr(self, "hierarchical_selector", None)
            or getattr(self, "gumbel_selector", None)
            or getattr(self, "relation_selector", None)
        )
        if self._candidate_mode and self._freeze_p23_core and selector is not None:
            yield from selector.parameters(recurse=recurse)
            receiver = getattr(self, "degree_relative_receiver", None)
            if receiver is not None:
                yield from receiver.parameters(recurse=recurse)
            router = getattr(self, "counterfactual_router", None)
            if router is not None:
                yield from router.parameters(recurse=recurse)
            return
        yield from super().parameters(recurse=recurse)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        if self._hierarchical_mode:
            selector_prefix = "hierarchical_selector."
        elif self._relation_mode:
            selector_prefix = "relation_selector."
        else:
            selector_prefix = "gumbel_selector."
        if self._candidate_mode and not any(key.startswith(selector_prefix) for key in state_dict):
            strict = False
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        *,
        z: torch.Tensor,
        h_pre: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        edge_scale_multiplier: float | torch.Tensor = 1.0,
        no_prompt_logits: torch.Tensor | None = None,
        h_adp_no_prompt: torch.Tensor | None = None,
        x_raw: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        independent = self.prompt_edge_denoiser.independent_edge_weights
        if independent is not None and self._cached_graph is None:
            # Enumerate with the original P23 logic; do not change/cache prompt initialization.
            graph = self._build_graph(z=z, x_raw=z if x_raw is None else x_raw,
                                      edge_index=edge_index, train_mask=train_mask, labels=labels)
            independent.validate_features(graph["feature_ids"])
        base_out = super().forward(
            z=z,
            h_pre=h_pre,
            edge_index=edge_index,
            train_mask=train_mask,
            edge_scale_multiplier=edge_scale_multiplier,
            no_prompt_logits=no_prompt_logits,
            h_adp_no_prompt=h_adp_no_prompt,
            x_raw=x_raw,
            labels=labels,
            epoch=epoch,
        )
        if independent is not None:
            independent.validate_features(self._cached_graph["feature_ids"])
        selector = self.hierarchical_selector
        gumbel_selector = self.gumbel_selector
        relation_selector = self.relation_selector
        if not self._candidate_mode or (
            selector is None and gumbel_selector is None and relation_selector is None
        ):
            return base_out

        graph = self._cached_graph
        if graph is None:
            raise RuntimeError("P23 graph cache is unavailable for hierarchical P24")
        aux = base_out["aux"]
        static_prompt_edge_weight = aux["prompt_edge_weight"]
        selector_kwargs = {
            "edge_node_index": graph["edge_node_index"].to(device=z.device),
            "edge_feature_local": graph["edge_feature_local"].to(device=z.device),
            "edge_direction": graph["edge_direction"].to(device=z.device),
            "base_edge_weight": static_prompt_edge_weight,
            "feature_stats": graph["feature_stats"].to(device=z.device),
            "feature_gate": aux["p23_selective_feature_gate"],
            "node_repr": h_pre,
            "feature_usage": graph.get("usage"),
            "original_edge_index": edge_index,
            "no_prompt_logits": no_prompt_logits,
            "epoch": epoch,
        }
        if relation_selector is not None:
            selector_out = relation_selector(
                **selector_kwargs,
                prompt_repr=base_out["prompt_node_x"],
            )
        elif gumbel_selector is not None:
            selector_out = gumbel_selector(
                **selector_kwargs,
                prompt_repr=base_out["prompt_node_x"],
            )
        else:
            selector_out = selector(
                **selector_kwargs,
                h_adp_no_prompt=h_adp_no_prompt,
            )
        candidate_prompt_edge_weight = selector_out["candidate_edge_weight"]
        receiver = self.degree_relative_receiver
        if receiver is not None:
            receiver_out = receiver(
                static_weight=static_prompt_edge_weight,
                candidate_weight=candidate_prompt_edge_weight,
                edge_node=graph["edge_node_index"].to(device=z.device),
                edge_direction=graph["edge_direction"].to(device=z.device),
                original_edge_index=edge_index,
                epoch=epoch,
            )
            candidate_prompt_edge_weight = receiver_out["edge_weight"]
        else:
            receiver_out = {
                "node_scale": static_prompt_edge_weight.new_ones(h_pre.size(0)),
                "mix_ratio": static_prompt_edge_weight.new_tensor(0.0),
                "budget_loss": static_prompt_edge_weight.new_tensor(0.0),
                "enabled": static_prompt_edge_weight.new_tensor(0.0),
            }
        prompt_edge_weight = (
            static_prompt_edge_weight
            if self.counterfactual_router is not None
            else (
                candidate_prompt_edge_weight
                if receiver is not None
                else selector_out["edge_weight"]
            )
        )
        original_edge_count = int(edge_index.size(1))
        original_weight = base_out["adapted_edge_weight"][:original_edge_count]
        base_out["adapted_edge_weight"] = torch.cat([original_weight, prompt_edge_weight], dim=0)

        aux["prompt_edge_weight"] = prompt_edge_weight
        aux["p24_static_prompt_edge_weight"] = static_prompt_edge_weight
        effective_gate = torch.ones_like(static_prompt_edge_weight)
        valid = static_prompt_edge_weight.abs() > 1e-12
        effective_gate[valid] = prompt_edge_weight[valid] / static_prompt_edge_weight[valid]
        aux["p24_edge_gate"] = effective_gate
        aux["p24_effective_edge_gate"] = effective_gate
        aux["p24_learned_prompt_edge_weight"] = prompt_edge_weight
        aux["p24_prompt_gate"] = selector_out.get(
            "prompt_keep_probability", selector_out.get("prompt_gate", static_prompt_edge_weight.new_ones(0))
        )
        aux["p24_node_confidence"] = selector_out.get("node_confidence", h_pre.new_zeros(h_pre.size(0)))
        aux["p24_route_logits"] = selector_out.get("route_logits", h_pre.new_zeros((h_pre.size(0), 2)))
        aux["p24_route_probs"] = selector_out.get(
            "route_probs",
            torch.stack([torch.ones(h_pre.size(0), device=h_pre.device), torch.zeros(h_pre.size(0), device=h_pre.device), torch.zeros(h_pre.size(0), device=h_pre.device)], dim=-1),
        )
        aux["p24_candidate_prompt_edge_weight"] = candidate_prompt_edge_weight
        receiver_node_scale = receiver_out["node_scale"]
        receiver_active = receiver_out.get(
            "active_node_mask", torch.ones_like(receiver_node_scale, dtype=torch.bool)
        )
        receiver_selected_scale = receiver_node_scale[receiver_active]
        if receiver_selected_scale.numel() == 0:
            receiver_selected_scale = receiver_node_scale
        aux["p24_receiver_node_scale"] = receiver_node_scale
        aux["p24_receiver_budget_loss"] = receiver_out["budget_loss"]
        aux["p24_receiver_enabled"] = receiver_out["enabled"]
        aux["p24_receiver_mix_ratio"] = receiver_out["mix_ratio"]
        aux["p24_receiver_scale_mean"] = receiver_selected_scale.mean()
        aux["p24_receiver_scale_min"] = receiver_selected_scale.min()
        aux["p24_receiver_scale_max"] = receiver_selected_scale.max()
        aux["p24_receiver_scale_above_one_ratio"] = (
            receiver_selected_scale > 1.0
        ).to(dtype=receiver_selected_scale.dtype).mean()
        crossfit_learner = self.crossfit_utility_learner
        if crossfit_learner is not None and isinstance(labels, torch.Tensor):
            eligible, minimum, class_count = crossfit_learner.eligibility(labels, train_mask)
            gain_ema = crossfit_learner.gain_ema
            coverage = (
                torch.isfinite(gain_ema[train_mask.bool()]).to(dtype=z.dtype).mean()
                if gain_ema.numel() == labels.numel() and bool(train_mask.any())
                else z.new_tensor(0.0)
            )
            aux["p24_crossfit_enabled"] = z.new_tensor(1.0)
            aux["p24_crossfit_eligible"] = z.new_tensor(float(eligible))
            aux["p24_crossfit_min_train_per_class"] = z.new_tensor(float(minimum))
            aux["p24_crossfit_class_count"] = z.new_tensor(float(class_count))
            aux["p24_crossfit_target_coverage"] = coverage
        else:
            aux["p24_crossfit_enabled"] = z.new_tensor(0.0)
            aux["p24_crossfit_eligible"] = z.new_tensor(0.0)
            aux["p24_crossfit_min_train_per_class"] = z.new_tensor(0.0)
            aux["p24_crossfit_class_count"] = z.new_tensor(0.0)
            aux["p24_crossfit_target_coverage"] = z.new_tensor(0.0)
        aux["p24_selector_grad_scale"] = z.new_tensor(self._selector_grad_scale)
        aux["p24_keep_static_loss"] = selector_out["losses"]["keep_static"]
        aux["p24_hub_suppression_loss"] = selector_out["losses"].get("hub_suppression", z.new_tensor(0.0))
        aux["p24_prompt_homophily_loss"] = selector_out["losses"].get("prompt_homophily", z.new_tensor(0.0))
        aux["p24_edge_budget_loss"] = selector_out["losses"].get("edge_budget", z.new_tensor(0.0))
        aux["p24_gumbel_keep_ratio_loss"] = selector_out["losses"].get("keep_ratio", z.new_tensor(0.0))
        aux["p24_gumbel_entropy_loss"] = selector_out["losses"].get("entropy", z.new_tensor(0.0))
        aux["p24_relation_prior_loss"] = selector_out["losses"].get(
            "relation_prior", z.new_tensor(0.0)
        )
        aux["p24_relation_preservation_loss"] = selector_out["losses"].get(
            "relation_preservation", z.new_tensor(0.0)
        )
        aux["p24_relation_homophily_collapse_loss"] = selector_out["losses"].get(
            "homophily_collapse", z.new_tensor(0.0)
        )
        aux["p24_relation_uncertain_keep_loss"] = selector_out["losses"].get(
            "uncertain_keep", z.new_tensor(0.0)
        )
        if "edge_relation_compatibility" in selector_out:
            aux["p24_relation_edge_compatibility"] = selector_out[
                "edge_relation_compatibility"
            ]
            aux["p24_relation_edge_prior_keep_target"] = selector_out[
                "edge_prior_keep_target"
            ]
        if "prompt_keep_logit" in selector_out:
            aux["p24_gumbel_prompt_keep_logit"] = selector_out["prompt_keep_logit"]
            aux["p24_gumbel_prompt_keep_probability"] = selector_out["prompt_keep_probability"]
            aux["p24_gumbel_edge_keep_logit"] = selector_out["edge_keep_logit"]
            aux["p24_gumbel_edge_keep_probability"] = selector_out["edge_keep_probability"]
            aux["p24_gumbel_prompt_prunable_prior"] = selector_out["prompt_prunable_prior"]
        aux.update(selector_out["stats"])
        return base_out

    def apply_counterfactual_routing(
        self,
        *,
        prompt_out: dict[str, Any],
        static_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        original_edge_count: int,
        epoch: int | None,
    ) -> dict[str, Any]:
        router = self.counterfactual_router
        if router is None or not router.enabled:
            return prompt_out
        aux = prompt_out["aux"]
        static_weight = aux["p24_static_prompt_edge_weight"]
        candidate_weight = aux["p24_candidate_prompt_edge_weight"]
        edge_node = aux["p23_selective_edge_node_index"]
        edge_feature = aux["p23_selective_edge_feature_local"]
        edge_direction = aux["p23_selective_edge_direction"]
        router_out = router(
            static_logits=static_logits,
            candidate_logits=candidate_logits,
            static_weight=static_weight,
            candidate_weight=candidate_weight,
            edge_node=edge_node,
            edge_feature=edge_feature,
            edge_direction=edge_direction,
            epoch=epoch,
        )
        crossfit_learner = self.crossfit_utility_learner
        if crossfit_learner is not None:
            eligible_value = aux.get("p24_crossfit_eligible")
            eligible = bool(
                isinstance(eligible_value, torch.Tensor)
                and eligible_value.detach().item() >= 0.5
            )
            if not eligible:
                router_out["adapt_ratio"] = torch.zeros_like(router_out["adapt_ratio"])
                router_out["p23_ratio"] = torch.ones_like(router_out["p23_ratio"])
        routed_weight = router.blend_weights(
            static_weight=static_weight,
            candidate_weight=candidate_weight,
            edge_node=edge_node,
            edge_direction=edge_direction,
            adapt_ratio=router_out["adapt_ratio"],
        )
        original_weight = prompt_out["adapted_edge_weight"][:original_edge_count]
        prompt_out["adapted_edge_weight"] = torch.cat([original_weight, routed_weight], dim=0)
        aux["prompt_edge_weight"] = routed_weight
        aux["p24_learned_prompt_edge_weight"] = routed_weight
        aux["p24_counterfactual_predicted_gain"] = router_out["predicted_gain"]
        aux["p24_counterfactual_features"] = router_out["features"]
        aux["p24_counterfactual_topology_abs_delta"] = router_out["topology_abs_delta"]
        aux["p24_counterfactual_hub_exposure"] = router_out["hub_exposure"]
        aux["p24_route_adapt"] = router_out["adapt_ratio"]
        aux["p24_route_p23"] = router_out["p23_ratio"]
        aux["p24_route_probs"] = torch.stack(
            [router_out["p23_ratio"], router_out["adapt_ratio"], torch.zeros_like(router_out["adapt_ratio"])],
            dim=-1,
        )
        valid = static_weight.abs() > 1e-12
        effective_gate = torch.ones_like(static_weight)
        effective_gate[valid] = routed_weight[valid] / static_weight[valid]
        weight_delta = routed_weight - static_weight
        aux.update(
            {
                "p24_counterfactual_router_enabled": static_weight.new_tensor(1.0),
                "p24_counterfactual_predicted_gain_mean": router_out["predicted_gain"].mean(),
                "p24_counterfactual_predicted_gain_std": router_out["predicted_gain"].std(unbiased=False),
                "p24_counterfactual_deployment_mix": router_out["deployment_mix"],
                "p24_route_stage_active": router_out["route_stage_active"],
                "p24_route_adapt_mean": router_out["adapt_ratio"].mean(),
                "p24_route_p23_mean": router_out["p23_ratio"].mean(),
                "p24_route_no_prompt_mean": static_weight.new_tensor(0.0),
                "p24_route_candidate_detached": static_weight.new_tensor(float(router.detach_candidate_weight)),
                "p24_weight_delta_mean": weight_delta.mean(),
                "p24_weight_delta_abs_mean": weight_delta.abs().mean(),
                "p24_effective_gate_mean": effective_gate[valid].mean() if bool(valid.any()) else static_weight.new_tensor(1.0),
                "p24_edge_gate_mean": effective_gate.mean(),
                "p24_edge_gate_min": effective_gate.min(),
                "p24_edge_gate_max": effective_gate.max(),
            }
        )
        return prompt_out
