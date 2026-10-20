"""Class-conditioned heterophily pattern prompt router (p20).

Unlike the free residual ``HeterophilyAwarePromptAdapter`` (P14-P18) which
produces an unconstrained ``delta`` per node, this module produces a
*constrained* correction message that is:

1. routed to a heterophily pattern via a trainable ``PatternRouter``;
2. produced by a bounded heterophily *expert* (no free delta);
3. conditioned on a class-conditioned prompt bank with parameter sharing
   ``P_{c,k} = class_prompt[c] + pattern_prompt[k] + low_rank_interaction[c,k]``;
4. accepted or rejected via a trainable ``ReceiveGate``.

The module is intentionally a duck-typed drop-in for
``HeterophilyAwarePromptAdapter``: its ``forward`` accepts the same keyword
arguments (plus optional ``base_logits`` / ``h_pre`` injected by the runner)
and returns a dict that exposes the same ``h_adp / delta / update / gate /
raw_gate`` contract and adapter diagnostic scalars, so the existing prompt
adapter training loop, evaluation and losses keep working unchanged.

It never builds prompt nodes/edges and never modifies ``edge_index``; the
correction is applied functionally on the adapted GP2F branch. pool/query/test
nodes only soft-read the prompt bank through stop-gradient soft class
probabilities; no hard pseudo-labels are written back.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary, mean_neighbor_variance


def _as_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


def _init_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


# Pattern index -> heterophily expert semantics (v1).
PATTERN_NAMES = [
    "reject",  # 0: no correction / reject
    "ego",  # 1: ego / self expert
    "high_pass",  # 2: high-pass (h_i - neighbor_mean_i)
    "two_hop",  # 3: two-hop aggregation
    "class_compat",  # 4: neighbor soft class distribution
    "role",  # 5: role / structure features
]
MAX_PATTERNS = len(PATTERN_NAMES)


class _BoundedExpert(nn.Module):
    """Bounded heterophily expert.

    Maps a pattern-specific descriptor ``feat`` plus a class-conditioned prompt
    ``cond`` into a hidden-space correction. The message magnitude is governed by
    a learnable ``out_scale`` rather than by zero-initialising the whole MLP, so
    the expert produces a *small but non-zero* correction from the start. This
    breaks the routing symmetry and lets ``q_i`` receive gradient early (a fully
    zero-initialised message would emit no routing gradient at all).

    Set ``init_scale=0.0`` to recover the exact no-op-at-init behaviour.
    """

    def __init__(self, in_dim: int, hidden_dim: int, *, dropout: float, init_scale: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        # Small final-layer init keeps the raw direction non-degenerate but tame;
        # overall magnitude is then controlled by the learnable out_scale.
        nn.init.normal_(self.mlp[-1].weight, std=0.05)
        nn.init.zeros_(self.mlp[-1].bias)
        self.out_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.out_scale * self.mlp(F.relu(self.in_proj(feat) + cond))


class ClassConditionedPatternPromptRouter(nn.Module):
    """Class-conditioned, pattern-routed, constrained prompt message generator."""

    # The runner injects stop-gradient base logits (and h_pre) when this is set.
    consumes_base_logits = True

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = _as_config(config)

        self.context_base = str(self.config.get("context_base", "z_detached"))
        if self.context_base not in {"z", "z_detached"}:
            raise ValueError("prompt_adapter.context_base must be 'z' or 'z_detached'")
        self.num_classes = int(self.config.get("num_classes", 0))
        if self.num_classes <= 0:
            raise ValueError("prompt_router.num_classes must be positive")
        self.num_patterns = int(self.config.get("num_patterns", MAX_PATTERNS))
        if not 1 <= self.num_patterns <= MAX_PATTERNS:
            raise ValueError(f"prompt_router.num_patterns must be in [1, {MAX_PATTERNS}]")
        self.hidden = int(self.config.get("hidden_dim", self.hidden_dim))
        self.low_rank_dim = int(self.config.get("low_rank_dim", 8))
        self.dropout = float(self.config.get("dropout", 0.1))
        self.gate_init = float(self.config.get("gate_init", 0.05))
        self.max_update_norm = float(self.config.get("max_update_norm", 0.08))
        self.message_scale = float(self.config.get("message_scale", 1.0))
        # zero_init_message=True recovers an exact no-op at init (no routing
        # gradient). By default we use a small non-zero message scale so the
        # PatternRouter can specialise early.
        self.zero_init_message = bool(self.config.get("zero_init_message", False))
        self.message_init_scale = float(self.config.get("message_init_scale", 0.1))
        self.easy_init = float(self.config.get("easy_init", 0.5))
        self.pattern_reject_init_prob = min(
            max(float(self.config.get("pattern_reject_init_prob", 1.0 / max(1, self.num_patterns))), 1e-6),
            1.0 - 1e-6,
        )
        self.use_disagreement = bool(self.config.get("use_frozen_disagreement", True))

        c = self.num_classes
        h = self.hidden_dim
        # Descriptor r_i: ego/low/two/high (hidden each) + role(5) + entropy/disagreement(2)
        # + neighbor soft class dist (C) + own soft class prob (C).
        self.descriptor_dim = 4 * h + 5 + 2 + 2 * c

        def _router(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.descriptor_dim, self.hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, int(out_dim)),
            )

        self.easy_router = _router(1)
        self.pattern_router = _router(self.num_patterns)
        self.receive_gate = _router(1)
        nn.init.zeros_(self.receive_gate[-1].weight)
        nn.init.constant_(self.receive_gate[-1].bias, _init_logit(self.gate_init))
        nn.init.zeros_(self.easy_router[-1].weight)
        nn.init.constant_(self.easy_router[-1].bias, _init_logit(self.easy_init))
        nn.init.zeros_(self.pattern_router[-1].weight)
        with torch.no_grad():
            pattern_prior = torch.full(
                (self.num_patterns,),
                (1.0 - self.pattern_reject_init_prob) / max(1, self.num_patterns - 1),
            )
            pattern_prior[0] = self.pattern_reject_init_prob
            self.pattern_router[-1].bias.copy_(pattern_prior.clamp_min(1e-12).log())

        # Class-conditioned prompt bank with parameter sharing.
        self.class_prompt = nn.Parameter(torch.zeros(c, h))
        self.pattern_prompt = nn.Parameter(torch.zeros(self.num_patterns, h))
        self.low_rank_u = nn.Parameter(torch.randn(c, self.low_rank_dim) * 0.02)
        self.low_rank_v = nn.Parameter(torch.randn(self.num_patterns, self.low_rank_dim) * 0.02)
        self.low_rank_proj = nn.Linear(self.low_rank_dim, h)
        nn.init.normal_(self.class_prompt, std=0.02)
        nn.init.normal_(self.pattern_prompt, std=0.02)

        # Experts. Index 0 (reject) has no parameters and emits a zero message.
        # Class prompt uses a dedicated ego-style expert for prototype nodes.
        init_scale = 0.0 if self.zero_init_message else self.message_init_scale
        self.class_expert = _BoundedExpert(h, h, dropout=self.dropout, init_scale=init_scale)
        feat_dims = {
            "ego": h,
            "high_pass": h,
            "two_hop": h,
            "class_compat": c,
            "role": 5,
        }
        self.pattern_experts = nn.ModuleDict()
        for k in range(1, self.num_patterns):
            name = PATTERN_NAMES[k]
            self.pattern_experts[name] = _BoundedExpert(
                feat_dims[name], h, dropout=self.dropout, init_scale=init_scale
            )

    def _base(self, rep: torch.Tensor) -> torch.Tensor:
        return rep.detach() if self.context_base == "z_detached" else rep

    def _soft_class_prob(
        self,
        *,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        base_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (soft_prob, base_entropy_normalised) using stop-gradient base logits."""
        if isinstance(base_logits, torch.Tensor):
            logits = base_logits.detach().to(device=device, dtype=dtype)
            prob = F.softmax(logits, dim=-1)
        else:
            prob = torch.full((num_nodes, self.num_classes), 1.0 / self.num_classes, device=device, dtype=dtype)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        if self.num_classes > 1:
            entropy = entropy / math.log(float(self.num_classes))
        return prob, entropy

    def _descriptor(
        self,
        *,
        rep: torch.Tensor,
        edge_index: torch.Tensor,
        soft_prob: torch.Tensor,
        base_entropy: torch.Tensor,
        h_pre: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        num_nodes = rep.size(0)
        low = mean_neighbor_summary(rep, edge_index, num_nodes=num_nodes)
        two = mean_neighbor_summary(low, edge_index, num_nodes=num_nodes)
        high = rep - low
        degree = rep.new_zeros(num_nodes)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1], rep.new_ones(edge_index.size(1)))
        neighbor_var = mean_neighbor_variance(rep, edge_index, num_nodes=num_nodes).mean(dim=-1)
        cos_low = F.cosine_similarity(rep, low, dim=-1, eps=1e-12)
        cos_two = F.cosine_similarity(low, two, dim=-1, eps=1e-12)
        role = torch.stack(
            [_minmax(degree), _minmax(torch.log1p(degree)), _minmax(neighbor_var), cos_low, cos_two], dim=-1
        )
        if self.use_disagreement and isinstance(h_pre, torch.Tensor):
            disagreement = 1.0 - F.cosine_similarity(
                rep, h_pre.detach().to(device=rep.device, dtype=rep.dtype), dim=-1, eps=1e-12
            )
        else:
            disagreement = rep.new_zeros(num_nodes)
        neighbor_class = mean_neighbor_summary(soft_prob, edge_index, num_nodes=num_nodes)
        descriptor = torch.cat(
            [
                rep,
                low,
                two,
                high,
                role,
                base_entropy.unsqueeze(-1),
                disagreement.unsqueeze(-1),
                neighbor_class,
                soft_prob,
            ],
            dim=-1,
        )
        return {
            "descriptor": descriptor,
            "ego": rep,
            "low": low,
            "two": two,
            "high": high,
            "role": role,
            "neighbor_class": neighbor_class,
            "degree": degree,
            "neighbor_variance": neighbor_var,
        }

    def _prompt_bank(self) -> torch.Tensor:
        """Return P_{c,k} of shape [C, K, hidden] with parameter sharing."""
        inter = self.low_rank_proj(
            self.low_rank_u.unsqueeze(1) * self.low_rank_v.unsqueeze(0)
        )  # [C, K, hidden]
        return self.class_prompt.unsqueeze(1) + self.pattern_prompt.unsqueeze(0) + inter

    def forward(
        self,
        *,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        h_adp: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        message_scale: float | None = None,
        support_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        h_pre: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        num_nodes = h_adp.size(0)
        rep = self._base(h_adp)
        soft_prob, base_entropy = self._soft_class_prob(
            num_nodes=num_nodes, device=rep.device, dtype=rep.dtype, base_logits=base_logits
        )
        desc = self._descriptor(
            rep=rep, edge_index=edge_index, soft_prob=soft_prob, base_entropy=base_entropy, h_pre=h_pre
        )
        descriptor = desc["descriptor"]

        # Class routing: support nodes are forced onto their true class.
        p_i = soft_prob
        if support_mask is not None and labels is not None:
            support_mask_b = support_mask.to(device=rep.device, dtype=torch.bool)
            labels_l = labels.to(device=rep.device, dtype=torch.long)
            valid = support_mask_b & (labels_l >= 0) & (labels_l < self.num_classes)
            if bool(valid.any()):
                one_hot = F.one_hot(labels_l[valid], num_classes=self.num_classes).to(dtype=rep.dtype)
                p_i = soft_prob.clone()
                p_i[valid] = one_hot

        a_i = torch.sigmoid(self.easy_router(descriptor)).squeeze(-1)  # [N]
        q_i = torch.softmax(self.pattern_router(descriptor), dim=-1)  # [N, K]
        raw_gate = torch.sigmoid(self.receive_gate(descriptor)).squeeze(-1)  # [N]
        gate = raw_gate

        bank = self._prompt_bank()  # [C, K, hidden]
        p_class = p_i @ self.class_prompt  # [N, hidden]
        inter_pk = torch.einsum("nc,ckh->nkh", p_i, bank)  # [N, K, hidden]
        # inter_pk already = sum_c p_i[c] * P[c,k]; this folds class conditioning into the
        # prompt vector (linear in p_i), so the message reduces to N*K expert evaluations.

        class_message = self.class_expert(desc["ego"], p_class)  # [N, hidden]

        feat_by_pattern = {
            "ego": desc["ego"],
            "high_pass": desc["high"],
            "two_hop": desc["two"],
            "class_compat": desc["neighbor_class"],
            "role": desc["role"],
        }
        messages = rep.new_zeros((num_nodes, self.num_patterns, self.hidden_dim))
        for k in range(1, self.num_patterns):
            name = PATTERN_NAMES[k]
            messages[:, k, :] = self.pattern_experts[name](feat_by_pattern[name], inter_pk[:, k, :])
        pattern_message = torch.einsum("nk,nkh->nh", q_i, messages)  # [N, hidden]

        raw_message = a_i.unsqueeze(-1) * class_message + (1.0 - a_i).unsqueeze(-1) * pattern_message

        max_norm = max(float(self.max_update_norm), 0.0)
        raw_norm = raw_message.norm(dim=-1, keepdim=True)
        if max_norm > 0.0:
            scale = (max_norm / raw_norm.clamp_min(1e-12)).clamp_max(1.0)
            message = raw_message * scale
            clip_ratio = (raw_norm.squeeze(-1) > max_norm).to(dtype=rep.dtype).mean()
        else:
            message = raw_message
            clip_ratio = raw_message.new_tensor(0.0)

        effective_mask = (
            torch.ones_like(gate, dtype=torch.bool) if update_mask is None else update_mask.bool().to(gate.device)
        )
        scale_value = self.message_scale if message_scale is None else float(message_scale)
        update = float(scale_value) * gate.unsqueeze(-1) * message
        update = update * effective_mask.to(dtype=update.dtype).unsqueeze(-1)
        h_prompted = h_adp + update
        update_norm = update.norm(dim=-1)

        pattern_usage_mean = q_i.mean(dim=0)  # [K]
        pattern_usage_entropy = -(pattern_usage_mean * pattern_usage_mean.clamp_min(1e-12).log()).sum()
        if self.num_patterns > 1:
            pattern_usage_entropy = pattern_usage_entropy / math.log(float(self.num_patterns))

        return {
            # ---- duck-typed adapter contract ----
            "h_adp": h_prompted,
            "delta": message,
            "raw_delta": raw_message,
            "update": update,
            "gate": gate,
            "raw_gate": raw_gate,
            "update_mask": effective_mask,
            "ego": desc["ego"],
            "low": desc["low"],
            "two_step": desc["two"],
            "high": desc["high"],
            "role": desc["role"],
            "prompt_update_norm": update_norm.mean(),
            "prompt_update_max_norm": update_norm.max() if update_norm.numel() > 0 else update_norm.new_tensor(0.0),
            "prompt_raw_delta_norm": raw_message.norm(dim=-1).mean(),
            "prompt_delta_norm": message.norm(dim=-1).mean(),
            "prompt_gate_mean": gate.mean(),
            "prompt_gate_min": gate.min() if gate.numel() > 0 else gate.new_tensor(0.0),
            "prompt_gate_max": gate.max() if gate.numel() > 0 else gate.new_tensor(0.0),
            "prompt_raw_gate_mean": raw_gate.mean(),
            "prompt_raw_gate_min": raw_gate.min() if raw_gate.numel() > 0 else raw_gate.new_tensor(0.0),
            "prompt_raw_gate_max": raw_gate.max() if raw_gate.numel() > 0 else raw_gate.new_tensor(0.0),
            "prompt_update_mask_ratio": effective_mask.to(dtype=rep.dtype).mean(),
            "prompt_update_clip_ratio": clip_ratio,
            "high_frequency_norm": desc["high"].norm(dim=-1).mean(),
            "low_frequency_norm": desc["low"].norm(dim=-1).mean(),
            # support context is not used by this module; provide neutral placeholders
            # so the shared adapter diagnostics keep working.
            "support_context_enabled": rep.new_tensor(0.0),
            "support_context_available": rep.new_tensor(0.0),
            "support_context_coverage": rep.new_tensor(0.0),
            "support_context_count": rep.new_tensor(0.0),
            "support_similarity_margin": rep.new_tensor(0.0),
            "support_similarity_entropy": rep.new_tensor(0.0),
            "support_reliability_mean": rep.new_tensor(1.0),
            "support_reliability_min": rep.new_tensor(1.0),
            "support_reliability_max": rep.new_tensor(1.0),
            "support_topk_mean_score": rep.new_tensor(0.0),
            "message_scale": rep.new_tensor(float(scale_value)),
            "max_update_norm": rep.new_tensor(float(max_norm)),
            # ---- router-specific outputs / diagnostics ----
            "easy_prob": a_i,
            "pattern_weights": q_i,
            "soft_class_prob": p_i,
            # Per-pattern (pre-q-weighting) and class messages, used by the
            # pattern-routing supervision teacher. Row k=0 (reject) is zeros.
            "pattern_messages": messages,
            "class_message": class_message,
            "class_prompt_usage": a_i.mean(),
            "hetero_prompt_usage": (1.0 - a_i).mean(),
            "pattern_usage_mean": pattern_usage_mean,
            "pattern_usage_entropy": pattern_usage_entropy,
        }


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
