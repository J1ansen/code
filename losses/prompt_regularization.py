"""Prompt regularization formulas shared with the training objective."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F


def _as_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _init_logit(value: float, max_value: float) -> torch.Tensor:
    max_value = float(max_value)
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    ratio = min(max(float(value) / max_value, 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(ratio, dtype=torch.float32))


def _safe_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x, y, dim=-1, eps=1e-12)


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


def _deterministic_random_scores(num_nodes: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(num_nodes, dtype=dtype, device=device)
    scores = torch.sin(idx * 12.9898 + float(seed) * 78.233) * 43758.5453
    return scores - torch.floor(scores)


def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    entropy = -(values * values.clamp_min(1e-12).log()).sum()
    if values.numel() > 1:
        entropy = entropy / math.log(float(values.numel()))
    return entropy


def _mean_route_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.numel() == 0 or logits.size(1) < 2:
        return logits.new_tensor(0.0)
    top2 = torch.topk(logits, k=2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).mean()


def prompt_edge_l1_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    weights = prompt_out.get("aux", {}).get("prompt_edge_weight")
    if isinstance(weights, torch.Tensor) and weights.numel() > 0:
        return weights.mean()
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_balance_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    aux = prompt_out.get("aux", {})
    usage = aux.get("prompt_usage_full", aux.get("prompt_usage"))
    if isinstance(usage, torch.Tensor) and usage.numel() > 0:
        target = usage.new_full(usage.shape, 1.0 / float(usage.numel()))
        return (usage - target).pow(2).sum()
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_role_diversity_loss(module: nn.Module | None) -> torch.Tensor:
    """Penalize highly similar prompt keys to encourage role separation."""

    if module is None:
        return torch.tensor(0.0)
    if bool(getattr(module, "use_multiview_routing", False)):
        keys = torch.cat(
            [
                module.semantic_prompt_keys,
                module.structural_prompt_keys,
                module.role_prompt_keys,
            ],
            dim=0,
        )
    else:
        keys = module.prompt_keys
    if keys.size(0) < 2:
        return keys.new_tensor(0.0)
    norm_keys = F.normalize(keys, dim=-1, eps=1e-12)
    sim = norm_keys @ norm_keys.t()
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    off_diag = sim[~eye]
    return off_diag.pow(2).mean()


def prompt_acceptance_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    """Small regularizer that discourages accepting every pool node by default."""

    aux = prompt_out.get("aux", {})
    gates = aux.get("pool_acceptance_gate")
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(gates, torch.Tensor) and gates.numel() > 0:
        return gates.mean()
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_acceptance_budget_loss(
    prompt_out: dict[str, Any],
    *,
    min_acceptance: float | None = None,
    max_acceptance: float | None = None,
) -> torch.Tensor:
    """Soft safety budget over the mean acceptance gate.

    This is a null-route style constraint for prompt graphs. It does not decide
    which nodes are correct; it only prevents the acceptance gate from drifting
    into always-off or always-on behavior when the available supervision is weak.
    """

    aux = prompt_out.get("aux", {})
    gates = aux.get("pool_acceptance_gate")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(gates, torch.Tensor) and gates.numel() > 0):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    mean_gate = gates.mean()
    losses: list[torch.Tensor] = []
    if min_acceptance is not None:
        losses.append(F.relu(float(min_acceptance) - mean_gate).pow(2))
    if max_acceptance is not None:
        losses.append(F.relu(mean_gate - float(max_acceptance)).pow(2))
    if not losses:
        return mean_gate.new_tensor(0.0)
    return torch.stack(losses).sum()


def prompt_usage_consistency_loss(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    margin: float = 0.25,
    negative_weight: float = 0.05,
) -> torch.Tensor:
    """Train-only class consistency over prompt usage distributions.

    This does not assign a class to any prompt node. It only nudges labeled
    train-pool nodes from the same class to use similar prompt distributions.
    Validation/test labels are ignored because the mask is always restricted
    to ``train_mask``.
    """

    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != full_prob.size(0):
        raise ValueError("routing_full_prob rows must match pool_mask true count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) < 2:
        return full_prob.new_tensor(0.0)
    probs = full_prob[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx][train_pool]
    dist = torch.cdist(probs, probs, p=2).pow(2)
    same = y[:, None] == y[None, :]
    eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    same = same & ~eye
    diff = ~same & ~eye
    losses: list[torch.Tensor] = []
    if bool(same.any()):
        losses.append(dist[same].mean())
    if negative_weight > 0 and bool(diff.any()):
        diff_dist = torch.sqrt(dist[diff].clamp_min(1e-12))
        losses.append(float(negative_weight) * F.relu(float(margin) - diff_dist).pow(2).mean())
    if not losses:
        return full_prob.new_tensor(0.0)
    return torch.stack(losses).sum()


def _train_pool_rows(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        return None
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != full_prob.size(0):
        raise ValueError("routing rows must match pool index count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return None
    y = labels.to(device=pool_mask.device)[pool_idx][train_pool]
    return train_pool, y, edge_scale if isinstance(edge_scale, torch.Tensor) else full_prob.new_tensor(0.0)


def prompt_class_route_loss(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    """Train-only class-slot routing loss for labeled pool nodes.

    Only labeled training nodes inside the prompt pool are supervised. The first
    ``num_class_prompt_slots`` prompt slots are treated as routing anchors; all
    remaining prompt slots stay residual and are never class targets.
    """

    aux = prompt_out.get("aux", {})
    logits = aux.get("routing_logits")
    class_slots = int(aux.get("num_class_prompt_slots", 0))
    if not (isinstance(logits, torch.Tensor) and class_slots > 0):
        edge_scale = prompt_out.get("edge_scale")
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    rows = _train_pool_rows(prompt_out, labels, train_mask)
    if rows is None:
        return logits.new_tensor(0.0)
    train_pool, y, _ = rows
    valid = (y >= 0) & (y < class_slots)
    if int(valid.sum().item()) == 0:
        return logits.new_tensor(0.0)
    class_logits = logits[train_pool][:, :class_slots][valid]
    targets = y[valid].long()
    return F.cross_entropy(class_logits, targets)


def prompt_key_proto_loss(
    module: nn.Module | None,
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    """Align class prompt keys to train-only structural query prototypes."""

    if module is None:
        return torch.tensor(0.0)
    class_slots = int(getattr(module, "num_class_prompt_slots", 0))
    if class_slots <= 0:
        return module.prompt_keys.new_tensor(0.0)
    aux = prompt_out.get("aux", {})
    query = aux.get("structural_query", aux.get("query"))
    if not isinstance(query, torch.Tensor):
        return module.prompt_keys.new_tensor(0.0)
    rows = _train_pool_rows(prompt_out, labels, train_mask)
    if rows is None:
        return query.new_tensor(0.0)
    train_pool, y, _ = rows
    train_query = query[train_pool]
    if train_query.numel() == 0:
        return query.new_tensor(0.0)
    if bool(getattr(module, "use_multiview_routing", False)):
        class_keys = module.structural_prompt_keys[:class_slots]
    else:
        class_keys = module.prompt_keys[:class_slots]
    losses: list[torch.Tensor] = []
    for class_id in range(class_slots):
        mask = y == class_id
        if not bool(mask.any()):
            continue
        proto = train_query[mask].detach().mean(dim=0)
        key = class_keys[class_id].to(dtype=query.dtype, device=query.device)
        losses.append(1.0 - F.cosine_similarity(key.unsqueeze(0), proto.unsqueeze(0), dim=-1, eps=1e-12).squeeze(0))
    if not losses:
        return query.new_tensor(0.0)
    return torch.stack(losses).mean()


def prompt_view_entropy_loss(prompt_out: dict[str, Any]) -> torch.Tensor:
    """Optional small loss encouraging node-wise view gates to make choices."""

    aux = prompt_out.get("aux", {})
    entropy = aux.get("view_gate_entropy")
    edge_scale = prompt_out.get("edge_scale")
    if isinstance(entropy, torch.Tensor):
        return entropy
    if isinstance(edge_scale, torch.Tensor):
        return edge_scale.new_tensor(0.0)
    return torch.tensor(0.0)


def prompt_view_prior_loss(prompt_out: dict[str, Any], prior: list[float] | tuple[float, ...]) -> torch.Tensor:
    """Match the average view gate to a configured soft prior."""

    aux = prompt_out.get("aux", {})
    view_gate = aux.get("view_gate")
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(view_gate, torch.Tensor) and view_gate.numel() > 0):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    target = view_gate.new_tensor(list(prior), dtype=view_gate.dtype)
    if target.numel() != view_gate.size(1):
        raise ValueError("view_prior length must match the number of routing views")
    target = target.clamp_min(0.0)
    target = target / target.sum().clamp_min(1e-12)
    return F.mse_loss(view_gate.mean(dim=0), target)


def utility_receive_gate_budget_loss(
    prompt_out: dict[str, Any],
    *,
    min_receive: float | None = None,
    max_receive: float | None = None,
) -> torch.Tensor:
    """Soft budget over the node-level effective receive gate."""

    aux = prompt_out.get("aux", {})
    gates = aux.get("effective_receive_gate", aux.get("receive_gate_score"))
    edge_scale = prompt_out.get("edge_scale")
    if not (isinstance(gates, torch.Tensor) and gates.numel() > 0):
        if isinstance(edge_scale, torch.Tensor):
            return edge_scale.new_tensor(0.0)
        return torch.tensor(0.0)
    mean_gate = gates.mean()
    losses: list[torch.Tensor] = []
    if min_receive is not None:
        losses.append(F.relu(float(min_receive) - mean_gate).pow(2))
    if max_receive is not None:
        losses.append(F.relu(mean_gate - float(max_receive)).pow(2))
    if not losses:
        return mean_gate.new_tensor(0.0)
    return torch.stack(losses).sum()


def prompt_adapter_update_norm_loss(adapter_out: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
    update = adapter_out.get("update")
    if not isinstance(update, torch.Tensor):
        return torch.tensor(0.0)
    if mask is not None:
        mask = mask.to(device=update.device, dtype=torch.bool)
        if int(mask.sum().item()) == 0:
            return update.new_tensor(0.0)
        update = update[mask]
    return update.pow(2).sum(dim=-1).mean()


def prompt_adapter_gate_budget_loss(
    adapter_out: dict[str, torch.Tensor],
    *,
    max_gate: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    gate = adapter_out.get("gate")
    if not isinstance(gate, torch.Tensor) or gate.numel() == 0:
        return torch.tensor(0.0)
    if mask is not None:
        mask = mask.to(device=gate.device, dtype=torch.bool)
        if int(mask.sum().item()) == 0:
            return gate.new_tensor(0.0)
        gate = gate[mask]
    return F.relu(gate.mean() - float(max_gate)).pow(2)


def prompt_adapter_message_help_loss(
    *,
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
    anti_harm_weight: float = 0.0,
    anti_harm_margin: float = 0.0,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_message_help_loss": 0.0,
            "prompt_adapter_message_help_mean_delta_ce": 0.0,
            "prompt_adapter_message_help_positive_ratio": 0.0,
            "prompt_adapter_message_help_count": 0.0,
            "prompt_adapter_message_help_anti_harm_loss": 0.0,
        }
    y = labels.to(device=logits_prompt.device)[idx]
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt[idx], y, reduction="none")
    delta = ce_no - ce_prompt
    losses = F.relu(float(margin) - delta)
    anti_harm_losses = F.relu(float(anti_harm_margin) - delta).pow(2)
    if class_balanced:
        per_class = []
        per_class_anti_harm = []
        for class_id in torch.unique(y.detach()).tolist():
            class_mask = y == int(class_id)
            if bool(class_mask.any()):
                per_class.append(losses[class_mask].mean())
                per_class_anti_harm.append(anti_harm_losses[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else losses.mean()
        anti_harm_loss = torch.stack(per_class_anti_harm).mean() if per_class_anti_harm else anti_harm_losses.mean()
    else:
        loss = losses.mean()
        anti_harm_loss = anti_harm_losses.mean()
    if float(anti_harm_weight) > 0.0:
        loss = loss + float(anti_harm_weight) * anti_harm_loss
    stats = {
        "prompt_adapter_message_help_loss": float(loss.detach().item()),
        "prompt_adapter_message_help_mean_delta_ce": float(delta.detach().mean().item()),
        "prompt_adapter_message_help_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
        "prompt_adapter_message_help_count": float(idx.numel()),
        "prompt_adapter_message_help_anti_harm_loss": float(anti_harm_loss.detach().item()),
    }
    return loss, stats


def prompt_adapter_utility_gate_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.02,
    margin: float = 0.0,
    class_balanced: bool = True,
    gate_source: str = "raw_gate",
) -> tuple[torch.Tensor, dict[str, float]]:
    if gate_source == "effective_gate":
        gate = adapter_out.get("gate", adapter_out.get("raw_gate"))
    elif gate_source == "raw_gate":
        gate = adapter_out.get("raw_gate", adapter_out.get("gate"))
    else:
        raise ValueError("gate_source must be 'raw_gate' or 'effective_gate'")
    if not isinstance(gate, torch.Tensor) or gate.numel() == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": 0.0,
            "prompt_adapter_utility_gate_positive_ratio": 0.0,
        }
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": 0.0,
            "prompt_adapter_utility_gate_positive_ratio": 0.0,
        }
    y = labels.to(device=logits_prompt.device)[idx]
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt.detach()[idx], y, reduction="none")
    delta = ce_no - ce_prompt
    temp = max(float(temperature), 1e-6)
    if margin > 0.0:
        supervised = delta.abs() >= float(margin)
    else:
        supervised = torch.ones_like(delta, dtype=torch.bool)
    if int(supervised.sum().item()) == 0:
        fallback = logits_prompt.new_tensor(0.0)
        return fallback, {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": float(delta.detach().mean().item()),
            "prompt_adapter_utility_gate_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
        }
    target = torch.sigmoid(delta.detach() / temp)
    gate_values = gate.to(device=logits_prompt.device)[idx].clamp(1e-6, 1.0 - 1e-6)
    losses = F.binary_cross_entropy(gate_values[supervised], target[supervised], reduction="none")
    y_supervised = y[supervised]
    if class_balanced:
        per_class = []
        for class_id in torch.unique(y_supervised.detach()).tolist():
            class_mask = y_supervised == int(class_id)
            if bool(class_mask.any()):
                per_class.append(losses[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else losses.mean()
    else:
        loss = losses.mean()
    stats = {
        "prompt_adapter_utility_gate_loss": float(loss.detach().item()),
        "prompt_adapter_utility_gate_target_mean": float(target[supervised].detach().mean().item()),
        "prompt_adapter_utility_gate_count": float(supervised.sum().item()),
        "prompt_adapter_utility_gate_delta_mean": float(delta.detach().mean().item()),
        "prompt_adapter_utility_gate_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
    }
    return loss, stats


PATTERN_NAMES = [
    "reject",  # 0: no correction / reject
    "ego",  # 1: ego / self expert
    "high_pass",  # 2: high-pass (h_i - neighbor_mean_i)
    "two_hop",  # 3: two-hop aggregation
    "class_compat",  # 4: neighbor soft class distribution
    "role",  # 5: role / structure features
]


def prompt_router_pattern_balance_loss(
    adapter_out: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    *,
    entropy_floor: float = 0.5,
) -> torch.Tensor:
    """Anti-collapse regulariser as a one-sided entropy floor.

    Unlike a KL-to-uniform penalty (which *rewards* a uniform distribution and
    therefore suppresses any routing specialisation), this only penalises the
    *extreme* case where the batch-averaged pattern usage collapses below a small
    normalised-entropy floor. When usage is sufficiently diverse the loss is 0,
    so the PatternRouter is free to specialise.

    ``entropy_floor`` is the normalised-entropy threshold in [0, 1]; set it to 0
    to disable the floor entirely.
    """
    q = adapter_out.get("pattern_weights")
    if not isinstance(q, torch.Tensor) or q.numel() == 0:
        return torch.tensor(0.0)
    if mask is not None:
        mask = mask.to(device=q.device, dtype=torch.bool)
        if int(mask.sum().item()) == 0:
            return q.new_tensor(0.0)
        q = q[mask]
    qbar = q.mean(dim=0)
    num_patterns = qbar.numel()
    entropy = -(qbar * qbar.clamp_min(1e-12).log()).sum()
    if num_patterns > 1:
        entropy = entropy / math.log(float(num_patterns))
    floor = min(max(float(entropy_floor), 0.0), 1.0)
    # Hinge: only pay when batch usage entropy drops below the floor (collapse).
    return F.relu(floor - entropy)
