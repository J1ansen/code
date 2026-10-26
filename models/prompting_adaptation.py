"""Prompting Adaptation over a fixed candidate prompt graph."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import torch
from torch import nn
from models.prompt_graph_construction import PromptGraphConstruction


class PromptingAdaptation(PromptGraphConstruction):
    """Combine Prompt Graph Construction and Prompting Weight Learning."""

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        resolved = dict(config or {})
        denoising_cfg = dict(resolved.get("prompt_edge_denoising", {}))
        denoising_cfg.setdefault("enabled", True)
        denoising_cfg.setdefault("allow_new_edges", False)
        denoising_cfg.setdefault("apply_to", "prompt_to_node")
        if str(denoising_cfg.get("architecture", "legacy_gate")).lower() != "legacy_gate":
            raise ValueError("The release supports only the shared edge-calibrator architecture")
        if denoising_cfg.get("descriptor_type") == "independent_edge":
            raise ValueError("Independent edge parameters are not part of the release model")
        for key in ("counterfactual_router", "degree_relative_receiver", "crossfit_utility", "safe_routing"):
            if denoising_cfg.get(key, {}).get("enabled", False):
                raise ValueError(f"Unsupported historical selector: {key}")
        if float(denoising_cfg.get("selector_grad_scale", 1.0)) <= 0:
            raise ValueError("selector_grad_scale must be positive")
        self._candidate_mode = False
        self.counterfactual_utility_enabled = False
        resolved["prompt_edge_denoising"] = denoising_cfg
        super().__init__(source_dim, hidden_dim, resolved)
        self.hierarchical_selector = None
        self.gumbel_selector = None
        self.relation_selector = None
        self.counterfactual_router = None
        self.degree_relative_receiver = None
        self.crossfit_utility_learner = None

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
        graph["aux"]["prompting_filtering_support_count"] = z.new_tensor(float(selected.sum()))
        graph["aux"]["prompting_initialization_support_count"] = z.new_tensor(float(train.sum()))
        return graph

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        denoiser = self.prompt_edge_denoiser
        if denoiser.descriptor_type in {"representation_concat", "raw_representation_concat"}:
            # Preserve initialization and state keys while excluding the unused head.
            denoiser.edge_gate.requires_grad_(False)
            unused = {id(parameter) for parameter in denoiser.edge_gate.parameters()}
            yield from (parameter for parameter in super().parameters(recurse=recurse)
                        if id(parameter) not in unused)
            return
        yield from super().parameters(recurse=recurse)
