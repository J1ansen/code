"""P22 class-conditional heterophily pattern enrichment bank."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.prompt_module import mean_neighbor_summary


def _logit_from_fraction(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


@torch.no_grad()
def estimate_class_transition_from_support(
    *,
    labels: torch.Tensor,
    support_mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.5,
) -> torch.Tensor:
    """Estimate T[c, d] = P(neighbor class=d | center class=c) from support edges."""

    device = labels.device
    transition = torch.full(
        (int(num_classes), int(num_classes)),
        float(smoothing),
        device=device,
        dtype=torch.float32,
    )
    if edge_index.numel() > 0:
        src, dst = edge_index.to(device)
        support = support_mask.to(device=device, dtype=torch.bool)
        edge_mask = support[src] & support[dst]
        src_y = labels[src[edge_mask]].long()
        dst_y = labels[dst[edge_mask]].long()
        valid = (src_y >= 0) & (src_y < int(num_classes)) & (dst_y >= 0) & (dst_y < int(num_classes))
        src_y = src_y[valid]
        dst_y = dst_y[valid]
        if src_y.numel() > 0:
            flat_idx = src_y * int(num_classes) + dst_y
            counts = torch.bincount(flat_idx, minlength=int(num_classes) * int(num_classes)).float()
            transition = transition + counts.view(int(num_classes), int(num_classes))
    return transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def compute_support_uniform_compatibility_matrix(
    *,
    labels: torch.Tensor,
    support_mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_classes: int,
    alpha: float = 5.0,
    prior: str = "uniform",
    support_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate C[neighbor_class, ego_class] with a configurable row prior."""

    k = int(num_classes)
    if k <= 0:
        raise ValueError("num_classes must be positive")
    device = labels.device
    dtype = torch.float32
    counts = torch.zeros(k, k, device=device, dtype=dtype)
    support = support_mask.to(device=device, dtype=torch.bool)
    y = labels.to(device=device, dtype=torch.long)
    if edge_index.numel() > 0:
        src, dst = edge_index.to(device)
        edge_mask = support[src] & support[dst]
        neigh_y = y[src[edge_mask]]
        ego_y = y[dst[edge_mask]]
        valid = (neigh_y >= 0) & (neigh_y < k) & (ego_y >= 0) & (ego_y < k)
        neigh_y = neigh_y[valid]
        ego_y = ego_y[valid]
        if neigh_y.numel() > 0:
            flat_idx = neigh_y * k + ego_y
            counts = counts + torch.bincount(flat_idx, minlength=k * k).to(dtype=dtype).view(k, k)
    prior_name = str(prior)
    if prior_name == "uniform":
        prior_counts = torch.full((k, k), 1.0 / float(k), device=device, dtype=dtype)
    elif prior_name == "identity":
        prior_counts = torch.eye(k, device=device, dtype=dtype)
    elif prior_name == "anti_identity":
        if k == 1:
            prior_counts = torch.ones(1, 1, device=device, dtype=dtype)
        else:
            prior_counts = torch.full((k, k), 1.0 / float(k - 1), device=device, dtype=dtype)
            prior_counts.fill_diagonal_(0.0)
    else:
        raise ValueError(f"Unsupported transition_prior={prior_name!r}")
    total = max(0.0, float(support_weight)) * counts + max(0.0, float(alpha)) * prior_counts
    transition = total / total.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    row_entropy = -(transition * transition.clamp_min(1e-12).log()).sum(dim=-1)
    if k > 1:
        row_entropy = row_entropy / math.log(float(k))
    diag = torch.diagonal(transition)
    offdiag_mask = ~torch.eye(k, dtype=torch.bool, device=device)
    offdiag = transition[offdiag_mask]
    support_edge_count = counts.sum()
    stats = {
        "transition_support_edge_count": support_edge_count,
        "transition_support_nonzero_row_ratio": (counts.sum(dim=-1) > 0).to(dtype=dtype).mean(),
        "transition_support_class_pair_coverage": (counts > 0).to(dtype=dtype).mean(),
        "transition_C_row_entropy": row_entropy.mean(),
        "transition_C_diag_mean": diag.mean(),
        "transition_C_offdiag_mean": offdiag.mean() if offdiag.numel() > 0 else transition.new_tensor(0.0),
        "transition_C_max_mean": transition.max(dim=-1).values.mean(),
    }
    return transition, stats


class P22ClassPatternEnrichmentBank(nn.Module):
    """Basis-supervised logit-evidence adapter for P22.

    The module leaves ``edge_index`` and hidden states unchanged. It builds a
    detached structural-predictive signature from NoPrompt logits and hidden
    relations, learns shared pattern tokens, then lets patterns choose explicit
    heterophily evidence bases before emitting a bounded class-level logit bias.
    """

    consumes_base_logits = True
    emits_logits = True

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        self.num_classes = int(self.config.get("num_classes", 0))
        if self.num_classes <= 0:
            raise ValueError("P22 requires prompt_adapter.num_classes to be positive")
        self.num_patterns = int(self.config.get("num_patterns", 6))
        if self.num_patterns <= 0:
            raise ValueError("P22 num_patterns must be positive")
        self.pattern_dim = int(self.config.get("pattern_dim", 32))
        self.pattern_init = str(self.config.get("pattern_init", "kmeans_all_signature"))
        self.pattern_init_use_labels = bool(self.config.get("pattern_init_use_labels", False))
        if self.pattern_init_use_labels:
            raise ValueError("P22 pattern_init_use_labels must remain false")
        self.class_pattern_smoothing = float(self.config.get("class_pattern_smoothing", 0.5))
        self.detach_class_pattern = bool(self.config.get("detach_class_pattern", True))
        self.eps = float(self.config.get("eps", 1e-8))
        self.use_basis_evidence = bool(self.config.get("use_basis_evidence", True))
        self.basis_types = list(
            self.config.get(
                "basis_types",
                [
                    "ego_logprob",
                    "onehop_logprob",
                    "twohop_logprob",
                    "highpass_ego_onehop",
                    "highpass_onehop_twohop",
                    "class_transition",
                ],
            )
        )
        self.num_bases = int(self.config.get("num_bases", len(self.basis_types)))
        if self.num_bases != len(self.basis_types):
            raise ValueError("P22 num_bases must match len(basis_types)")
        self.enrichment_weight = float(self.config.get("enrichment_weight", 0.5))
        self.basis_weight_scale = float(self.config.get("basis_weight_scale", 1.0))
        self.use_class_transition = bool(self.config.get("use_class_transition", True))
        self.transition_type = str(self.config.get("transition_type", "support_edges"))
        self.class_transition_smoothing = float(self.config.get("class_transition_smoothing", 0.5))
        self.use_transition_basis = bool(self.config.get("use_transition_basis", False))
        self.transition_matrix_mode = str(self.config.get("transition_matrix_mode", "support_edges"))
        self.transition_prior = str(self.config.get("transition_prior", "uniform"))
        self.transition_alpha = float(self.config.get("transition_alpha", 5.0))
        self.transition_lambda_support = float(self.config.get("transition_lambda_support", 1.0))
        self.transition_lambda_pseudo = float(self.config.get("transition_lambda_pseudo", 0.0))
        if self.transition_lambda_pseudo != 0.0 and self.transition_matrix_mode == "support_uniform":
            raise ValueError("P22 support_uniform transition basis keeps transition_lambda_pseudo at 0.0")
        self.basis_usage_entropy_floor = float(self.config.get("basis_usage_entropy_floor", 0.60))
        self.normalize_basis_evidence = bool(self.config.get("normalize_basis_evidence", False))
        self.basis_evidence_std_floor = float(self.config.get("basis_evidence_std_floor", 0.5))
        self.pattern_scale_max = float(self.config.get("pattern_scale_max", 1.0))
        pattern_scale_init = float(self.config.get("pattern_scale_init", 0.10))
        self.raw_pattern_scale = nn.Parameter(
            torch.tensor(_logit_from_fraction(pattern_scale_init / max(self.pattern_scale_max, 1e-12)))
        )
        self.pattern_scale_warmup_epochs = int(self.config.get("pattern_scale_warmup_epochs", 20))
        self.register_buffer("current_epoch", torch.tensor(0.0), persistent=False)
        self.temperature_init = float(self.config.get("pattern_temperature_init", 0.70))
        self.temperature_final = float(self.config.get("pattern_temperature_final", 0.25))
        self.temperature_warmdown_epochs = int(self.config.get("pattern_temperature_warmdown_epochs", 50))
        self.normalize_evidence = bool(self.config.get("normalize_pattern_evidence", True))
        self.evidence_std_floor = float(self.config.get("pattern_evidence_std_floor", 0.5))
        self.use_reliability_gate = bool(self.config.get("use_reliability_gate", False))
        self.reliability_gate_min = float(self.config.get("reliability_gate_min", 0.0))
        self.reliability_gate_max = float(self.config.get("reliability_gate_max", 1.0))
        self.reliability_gate_detach_input = bool(self.config.get("reliability_gate_detach_input", True))
        self.use_pattern_gate = bool(self.config.get("use_pattern_gate", False))
        if self.use_pattern_gate:
            raise ValueError("P22 keeps use_pattern_gate disabled in the basis-supervised version")

        signature_dim = 2 * self.num_classes + 8
        dropout = float(self.config.get("dropout", 0.0))
        encoder_hidden = int(self.config.get("pattern_encoder_hidden_dim", max(32, self.pattern_dim)))
        self.signature_norm = nn.LayerNorm(signature_dim)
        self.pattern_encoder = nn.Sequential(
            nn.Linear(signature_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden, self.pattern_dim),
        )
        self.pattern_tokens = nn.Parameter(torch.randn(self.num_patterns, self.pattern_dim) * 0.02)
        self.pattern_basis_logits = nn.Parameter(torch.empty(self.num_patterns, self.num_bases))
        pattern_basis_init = str(self.config.get("pattern_basis_init", "zeros"))
        with torch.no_grad():
            if pattern_basis_init == "cyclic_anchor":
                self.pattern_basis_logits.fill_(float(self.config.get("pattern_basis_off_logit", -2.0)))
                anchor_logit = float(self.config.get("pattern_basis_anchor_logit", 2.0))
                for pattern_idx in range(self.num_patterns):
                    self.pattern_basis_logits[pattern_idx, pattern_idx % self.num_bases] = anchor_logit
            elif pattern_basis_init == "zeros":
                self.pattern_basis_logits.zero_()
            else:
                raise ValueError(f"Unsupported P22 pattern_basis_init={pattern_basis_init!r}")
        self.reliability_feature_dim = 10
        if self.use_reliability_gate:
            gate_hidden = int(self.config.get("reliability_gate_hidden_dim", 32))
            self.reliability_norm = nn.LayerNorm(self.reliability_feature_dim)
            self.reliability_gate = nn.Sequential(
                nn.Linear(self.reliability_feature_dim, gate_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(
                self.reliability_gate[-1].bias,
                float(self.config.get("reliability_gate_init_bias", -2.0)),
            )
        else:
            self.reliability_norm = nn.Identity()
            self.reliability_gate = None
        self.register_buffer("pattern_tokens_initialized", torch.tensor(False), persistent=True)

    def set_epoch(self, epoch: int | float) -> None:
        self.current_epoch.fill_(float(epoch))

    def _temperature(self) -> torch.Tensor:
        if self.temperature_warmdown_epochs <= 0:
            value = self.temperature_final
        else:
            t = min(1.0, max(0.0, float(self.current_epoch.item()) / float(self.temperature_warmdown_epochs)))
            value = self.temperature_init + t * (self.temperature_final - self.temperature_init)
        return self.pattern_tokens.new_tensor(max(value, 1e-4))

    def _scale(self) -> torch.Tensor:
        if self.pattern_scale_warmup_epochs <= 0:
            warmup = 1.0
        else:
            warmup = min(1.0, max(0.0, (float(self.current_epoch.item()) + 1.0) / float(self.pattern_scale_warmup_epochs)))
        return self.pattern_scale_max * torch.sigmoid(self.raw_pattern_scale) * self.raw_pattern_scale.new_tensor(warmup)

    def _degree_norm(self, edge_index: torch.Tensor, num_nodes: int, ref: torch.Tensor) -> torch.Tensor:
        degree = ref.new_zeros(num_nodes)
        if edge_index.numel() > 0:
            degree.index_add_(0, edge_index[1].to(ref.device), ref.new_ones(edge_index.size(1)))
        return _minmax(torch.log1p(degree))

    def _signature(
        self,
        *,
        h_adp: torch.Tensor,
        edge_index: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        h = h_adp.detach()
        logits = base_logits.detach().to(device=h.device, dtype=h.dtype)
        prob = F.softmax(logits, dim=-1)
        neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=int(prob.size(0)))
        two_prob = mean_neighbor_summary(neigh_prob, edge_index, num_nodes=int(prob.size(0)))
        low = mean_neighbor_summary(h, edge_index, num_nodes=int(h.size(0)))
        two = mean_neighbor_summary(low, edge_index, num_nodes=int(h.size(0)))

        prob_diff = (prob - neigh_prob).abs()
        neigh_two_diff = (neigh_prob - two_prob).abs()
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        neigh_entropy = -(neigh_prob * neigh_prob.clamp_min(1e-12).log()).sum(dim=-1)
        if prob.size(-1) > 1:
            norm = math.log(float(prob.size(-1)))
            entropy = entropy / norm
            neigh_entropy = neigh_entropy / norm
        top2 = torch.topk(prob, k=min(2, prob.size(-1)), dim=-1).values
        margin = top2[:, 0] if top2.size(-1) == 1 else top2[:, 0] - top2[:, 1]
        degree_norm = self._degree_norm(edge_index, int(h.size(0)), h)
        ego_low_norm = _minmax((h - low).norm(dim=-1))
        low_two_norm = _minmax((low - two).norm(dim=-1))
        cos_ego_low = F.cosine_similarity(h, low, dim=-1, eps=1e-12)
        cos_ego_two = F.cosine_similarity(h, two, dim=-1, eps=1e-12)
        signature = torch.cat(
            [
                prob_diff,
                neigh_two_diff,
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                neigh_entropy.unsqueeze(-1),
                degree_norm.unsqueeze(-1),
                ego_low_norm.unsqueeze(-1),
                low_two_norm.unsqueeze(-1),
                cos_ego_low.unsqueeze(-1),
                cos_ego_two.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.signature_norm(signature)

    @torch.no_grad()
    def _kmeans_centers(self, x: torch.Tensor) -> torch.Tensor:
        n = int(x.size(0))
        k = self.num_patterns
        if n == 0:
            return x.new_zeros(k, x.size(-1))
        if n < k:
            reps = (k + n - 1) // n
            return x.repeat(reps, 1)[:k]
        # Deterministic farthest-ish seed spread over signature norm order.
        order = torch.argsort(x.norm(dim=-1))
        seed_pos = torch.linspace(0, n - 1, steps=k, device=x.device).round().long()
        centers = x[order[seed_pos]].clone()
        iters = int(self.config.get("kmeans_iters", 15))
        for _ in range(max(1, iters)):
            dist = torch.cdist(x, centers)
            assign = dist.argmin(dim=-1)
            new_centers = centers.clone()
            for idx in range(k):
                mask = assign == idx
                if bool(mask.any()):
                    new_centers[idx] = x[mask].mean(dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        return centers

    @torch.no_grad()
    def _maybe_initialize_tokens(self, signature: torch.Tensor) -> None:
        if bool(self.pattern_tokens_initialized.item()):
            return
        if self.pattern_init == "random":
            self.pattern_tokens_initialized.fill_(True)
            return
        if self.pattern_init != "kmeans_all_signature":
            raise ValueError(f"Unsupported P22 pattern_init={self.pattern_init!r}")
        was_training = self.pattern_encoder.training
        self.pattern_encoder.eval()
        encoded_all = F.normalize(self.pattern_encoder(signature.detach()), dim=-1, eps=1e-12)
        centers = self._kmeans_centers(encoded_all)
        self.pattern_tokens.copy_(F.normalize(centers, dim=-1, eps=1e-12))
        if was_training:
            self.pattern_encoder.train()
        self.pattern_tokens_initialized.fill_(True)

    def _class_pattern_enrichment(
        self,
        *,
        pattern_weights: torch.Tensor,
        support_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = pattern_weights.shape
        if support_mask is None:
            support_mask = torch.ones(n, dtype=torch.bool, device=pattern_weights.device)
        support_mask = support_mask.to(device=pattern_weights.device, dtype=torch.bool)
        if labels is None:
            raise ValueError("P22 requires labels to build class-pattern enrichment")
        labels = labels.to(device=pattern_weights.device, dtype=torch.long)
        if not bool(support_mask.any()):
            support_mask = torch.ones(n, dtype=torch.bool, device=pattern_weights.device)
        global_pattern = pattern_weights[support_mask].mean(dim=0).detach().clamp_min(self.eps)
        global_pattern = global_pattern / global_pattern.sum().clamp_min(self.eps)
        class_pattern = pattern_weights.new_zeros(self.num_classes, k)
        class_count = pattern_weights.new_zeros(self.num_classes, 1)
        alpha = pattern_weights.new_tensor(max(0.0, self.class_pattern_smoothing))
        for class_id in range(self.num_classes):
            class_mask = support_mask & (labels == class_id)
            count = class_mask.to(dtype=pattern_weights.dtype).sum()
            class_count[class_id, 0] = count
            numerator = pattern_weights[class_mask].sum(dim=0) if bool(class_mask.any()) else pattern_weights.new_zeros(k)
            class_pattern[class_id] = (numerator + alpha * global_pattern) / (count + alpha).clamp_min(self.eps)
        enrichment = (class_pattern.clamp_min(self.eps).log() - global_pattern.clamp_min(self.eps).log().unsqueeze(0))
        return enrichment, class_pattern, global_pattern

    def _basis_evidence(
        self,
        *,
        base_logits: torch.Tensor,
        edge_index: torch.Tensor,
        support_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        logits = base_logits.detach()
        num_nodes = int(logits.size(0))
        prob = F.softmax(logits, dim=-1)
        neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=num_nodes)
        two_prob = mean_neighbor_summary(neigh_prob, edge_index, num_nodes=num_nodes)
        if support_mask is None:
            support = torch.ones(num_nodes, dtype=torch.bool, device=logits.device)
        else:
            support = support_mask.to(device=logits.device, dtype=torch.bool)
        if labels is None:
            labels_t = torch.zeros(num_nodes, dtype=torch.long, device=logits.device)
        else:
            labels_t = labels.to(device=logits.device, dtype=torch.long)
        transition_stats: dict[str, torch.Tensor] = {}
        if self.use_transition_basis and self.transition_matrix_mode == "support_uniform":
            transition, transition_stats = compute_support_uniform_compatibility_matrix(
                labels=labels_t,
                support_mask=support,
                edge_index=edge_index,
                num_classes=self.num_classes,
                alpha=self.transition_alpha,
                prior=self.transition_prior,
                support_weight=self.transition_lambda_support,
            )
            transition = transition.to(device=logits.device, dtype=logits.dtype).detach()
            transition_stats = {
                key: value.to(device=logits.device, dtype=logits.dtype)
                for key, value in transition_stats.items()
            }
        elif self.transition_type == "support_neighbor_prob":
            transition = logits.new_full(
                (self.num_classes, self.num_classes),
                float(self.class_transition_smoothing),
            )
            for class_id in range(self.num_classes):
                class_mask = support & (labels_t == class_id)
                if bool(class_mask.any()):
                    transition[class_id] = transition[class_id] + neigh_prob[class_mask].sum(dim=0)
            transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        elif self.transition_type == "support_edges":
            transition = estimate_class_transition_from_support(
                labels=labels_t,
                support_mask=support,
                edge_index=edge_index,
                num_classes=self.num_classes,
                smoothing=self.class_transition_smoothing,
            ).to(device=logits.device, dtype=logits.dtype)
        else:
            raise ValueError(f"Unsupported P22 transition_type={self.transition_type!r}")

        onehop_transition_prob = (neigh_prob @ transition).clamp_min(self.eps)
        onehop_transition_logprob = onehop_transition_prob.log()
        ego_logprob = prob.clamp_min(self.eps).log()
        values: list[torch.Tensor] = []
        for basis_type in self.basis_types:
            if basis_type == "ego_logprob":
                values.append(ego_logprob)
            elif basis_type == "onehop_logprob":
                values.append(neigh_prob.clamp_min(self.eps).log())
            elif basis_type == "twohop_logprob":
                values.append(two_prob.clamp_min(self.eps).log())
            elif basis_type == "highpass_ego_onehop":
                values.append(prob - neigh_prob)
            elif basis_type == "highpass_onehop_twohop":
                values.append(neigh_prob - two_prob)
            elif basis_type == "class_transition":
                if self.use_class_transition:
                    transition_evidence = neigh_prob @ transition.t()
                    values.append(transition_evidence.clamp_min(self.eps).log())
                else:
                    values.append(logits.new_zeros(num_nodes, self.num_classes))
            elif basis_type == "onehop_transition_logprob":
                values.append(onehop_transition_logprob)
            elif basis_type == "highpass_ego_transition_onehop":
                values.append(ego_logprob - onehop_transition_logprob)
            else:
                raise ValueError(f"Unsupported P22 basis type: {basis_type!r}")
        basis_evidence = torch.stack(values, dim=1)
        if self.normalize_basis_evidence:
            basis_evidence = basis_evidence - basis_evidence.mean(dim=-1, keepdim=True)
            std = basis_evidence.std(dim=-1, keepdim=True).detach().clamp_min(self.basis_evidence_std_floor)
            basis_evidence = basis_evidence / std
        return basis_evidence, transition, transition_stats

    def _reliability_features(
        self,
        *,
        base_logits: torch.Tensor,
        edge_index: torch.Tensor,
        pattern_weights: torch.Tensor,
        pattern_evidence: torch.Tensor,
        ungated_logit_bias: torch.Tensor,
    ) -> torch.Tensor:
        logits = base_logits.detach().to(device=pattern_evidence.device, dtype=pattern_evidence.dtype)
        prob = F.softmax(logits, dim=-1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        pattern_entropy = -(pattern_weights * pattern_weights.clamp_min(1e-12).log()).sum(dim=-1)
        if prob.size(-1) > 1:
            entropy = entropy / math.log(float(prob.size(-1)))
        if pattern_weights.size(-1) > 1:
            pattern_entropy = pattern_entropy / math.log(float(pattern_weights.size(-1)))
        top2 = torch.topk(prob, k=min(2, prob.size(-1)), dim=-1).values
        confidence = top2[:, 0]
        margin = top2[:, 0] if top2.size(-1) == 1 else top2[:, 0] - top2[:, 1]
        pattern_max = pattern_weights.max(dim=-1).values
        evidence_norm = _minmax(pattern_evidence.detach().norm(dim=-1))
        bias_norm = _minmax(ungated_logit_bias.detach().norm(dim=-1))
        cosine = F.cosine_similarity(logits, pattern_evidence.detach(), dim=-1, eps=1e-12)
        agree = (
            logits.argmax(dim=-1) == pattern_evidence.detach().argmax(dim=-1)
        ).to(dtype=pattern_evidence.dtype)
        neigh_prob = mean_neighbor_summary(prob, edge_index, num_nodes=int(prob.size(0)))
        neigh_entropy = -(neigh_prob * neigh_prob.clamp_min(1e-12).log()).sum(dim=-1)
        if prob.size(-1) > 1:
            neigh_entropy = neigh_entropy / math.log(float(prob.size(-1)))
        degree_norm = self._degree_norm(edge_index, int(prob.size(0)), pattern_evidence)
        features = torch.stack(
            [
                entropy,
                margin,
                confidence,
                pattern_entropy,
                pattern_max,
                evidence_norm,
                bias_norm,
                cosine,
                agree,
                0.5 * (neigh_entropy + degree_norm),
            ],
            dim=-1,
        )
        features = features.detach() if self.reliability_gate_detach_input else features
        return self.reliability_norm(features)

    def forward(
        self,
        *,
        z: torch.Tensor | None = None,
        edge_index: torch.Tensor,
        h_adp: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if base_logits is None:
            raise ValueError("P22 requires detached NoPrompt base_logits")
        signature = self._signature(h_adp=h_adp, edge_index=edge_index, base_logits=base_logits)
        self._maybe_initialize_tokens(signature)
        encoded = F.normalize(self.pattern_encoder(signature), dim=-1)
        tokens = F.normalize(self.pattern_tokens, dim=-1)
        pattern_logits = encoded @ tokens.t() / self._temperature()
        pattern_weights = F.softmax(pattern_logits, dim=-1)
        pattern_weights_for_stats = pattern_weights.detach() if self.detach_class_pattern else pattern_weights
        enrichment, class_pattern, global_pattern = self._class_pattern_enrichment(
            pattern_weights=pattern_weights_for_stats,
            support_mask=support_mask,
            labels=labels,
        )
        enrichment_evidence = pattern_weights @ enrichment.t()
        if self.use_basis_evidence:
            basis_evidence, transition_matrix, transition_stats = self._basis_evidence(
                base_logits=base_logits,
                edge_index=edge_index,
                support_mask=support_mask,
                labels=labels,
            )
            pattern_basis_weight = F.softmax(self.pattern_basis_logits, dim=-1)
            basis_pattern_evidence = torch.einsum("nk,kb,nbc->nc", pattern_weights, pattern_basis_weight, basis_evidence)
            student_basis = pattern_weights @ pattern_basis_weight
            basis_usage = pattern_weights.mean(dim=0) @ pattern_basis_weight
            basis_usage = basis_usage / basis_usage.sum().clamp_min(self.eps)
            basis_usage_entropy = -(basis_usage * basis_usage.clamp_min(1e-12).log()).sum()
            if self.num_bases > 1:
                basis_usage_entropy = basis_usage_entropy / math.log(float(self.num_bases))
            basis_usage_loss = F.relu(self.basis_usage_entropy_floor - basis_usage_entropy).pow(2)
        else:
            basis_evidence = h_adp.new_zeros(h_adp.size(0), self.num_bases, self.num_classes)
            transition_matrix = h_adp.new_zeros(self.num_classes, self.num_classes)
            transition_stats = {}
            pattern_basis_weight = F.softmax(self.pattern_basis_logits, dim=-1)
            basis_pattern_evidence = h_adp.new_zeros(h_adp.size(0), self.num_classes)
            student_basis = pattern_weights @ pattern_basis_weight
            basis_usage = student_basis.mean(dim=0)
            basis_usage = basis_usage / basis_usage.sum().clamp_min(self.eps)
            basis_usage_entropy = h_adp.new_tensor(0.0)
            basis_usage_loss = h_adp.new_tensor(0.0)

        pattern_evidence = self.enrichment_weight * enrichment_evidence + self.basis_weight_scale * basis_pattern_evidence
        pattern_evidence = pattern_evidence - pattern_evidence.mean(dim=-1, keepdim=True)
        if self.normalize_evidence:
            std = pattern_evidence.std(dim=-1, keepdim=True).detach().clamp_min(self.evidence_std_floor)
            pattern_evidence = pattern_evidence / std
        pattern_scale = self._scale()
        ungated_logit_bias = pattern_scale * pattern_evidence
        if self.use_reliability_gate and self.reliability_gate is not None:
            reliability_features = self._reliability_features(
                base_logits=base_logits,
                edge_index=edge_index,
                pattern_weights=pattern_weights,
                pattern_evidence=pattern_evidence,
                ungated_logit_bias=ungated_logit_bias,
            )
            reliability_gate_logit = self.reliability_gate(reliability_features).squeeze(-1)
            raw_reliability_gate = torch.sigmoid(reliability_gate_logit)
            reliability_gate = self.reliability_gate_min + (
                self.reliability_gate_max - self.reliability_gate_min
            ) * raw_reliability_gate
        else:
            reliability_features = h_adp.new_zeros(h_adp.size(0), self.reliability_feature_dim)
            reliability_gate_logit = h_adp.new_full((h_adp.size(0),), 20.0)
            raw_reliability_gate = h_adp.new_ones(h_adp.size(0))
            reliability_gate = h_adp.new_ones(h_adp.size(0))
        effective_gate = reliability_gate
        if update_mask is not None:
            # The mask only gates deployment of evidence; support statistics stay
            # controlled by support_mask.
            mask = update_mask.to(device=h_adp.device, dtype=h_adp.dtype).unsqueeze(-1)
            effective_gate = effective_gate * mask.squeeze(-1)
        logit_bias = ungated_logit_bias * effective_gate.unsqueeze(-1)
        logits = base_logits.detach().to(device=h_adp.device, dtype=h_adp.dtype) + logit_bias
        ungated_logits = base_logits.detach().to(device=h_adp.device, dtype=h_adp.dtype) + ungated_logit_bias

        zero = h_adp.new_zeros(h_adp.shape)
        pattern_usage_mean = pattern_weights.mean(dim=0)
        pattern_usage_entropy = -(pattern_usage_mean * pattern_usage_mean.clamp_min(1e-12).log()).sum()
        if self.num_patterns > 1:
            pattern_usage_entropy = pattern_usage_entropy / math.log(float(self.num_patterns))
        update_mask_eff = (
            torch.ones(h_adp.size(0), dtype=torch.bool, device=h_adp.device)
            if update_mask is None
            else update_mask.to(device=h_adp.device, dtype=torch.bool)
        )
        pattern_reg = F.relu(pattern_usage_mean.max() - float(self.config.get("pattern_usage_max", 0.80)))
        pattern_reg = pattern_reg + F.relu(float(self.config.get("pattern_usage_entropy_floor", 0.35)) - pattern_usage_entropy)
        class_pattern_kl = (
            class_pattern.clamp_min(self.eps)
            * (class_pattern.clamp_min(self.eps).log() - global_pattern.clamp_min(self.eps).log().unsqueeze(0))
        ).sum(dim=-1).mean()
        topk = min(3, self.num_patterns)
        class_pattern_topk_value, class_pattern_topk_index = torch.topk(class_pattern.detach(), k=topk, dim=-1)
        return {
            "h_adp": h_adp,
            "logits": logits,
            "pattern_evidence": pattern_evidence,
            "enrichment_evidence": enrichment_evidence,
            "basis_pattern_evidence": basis_pattern_evidence,
            "basis_evidence": basis_evidence,
            "logit_bias": logit_bias,
            "ungated_logit_bias": ungated_logit_bias,
            "ungated_logits": ungated_logits,
            "pattern_scale": pattern_scale,
            "reliability_features": reliability_features.detach(),
            "reliability_gate_logit": reliability_gate_logit,
            "reliability_gate": reliability_gate,
            "raw_reliability_gate": raw_reliability_gate,
            "effective_reliability_gate": effective_gate,
            "pattern_weights": pattern_weights,
            "pattern_logits": pattern_logits,
            "pattern_basis_weight": pattern_basis_weight,
            "student_basis": student_basis,
            "basis_usage": basis_usage,
            "basis_usage_entropy": basis_usage_entropy,
            "basis_usage_loss": basis_usage_loss,
            "transition_matrix": transition_matrix,
            "basis_names": list(self.basis_types),
            "class_pattern_enrichment": enrichment,
            "class_pattern": class_pattern,
            "class_pattern_kl_to_global": class_pattern_kl,
            "class_pattern_topk_value": class_pattern_topk_value,
            "class_pattern_topk_index": class_pattern_topk_index,
            "global_pattern": global_pattern,
            "pattern_usage_mean": pattern_usage_mean,
            "pattern_usage_entropy": pattern_usage_entropy,
            "pattern_max_prob_mean": pattern_weights.max(dim=-1).values.mean(),
            "pattern_reg": pattern_reg,
            "signature": signature.detach(),
            "delta": zero,
            "raw_delta": zero,
            "update": zero,
            "gate": effective_gate,
            "raw_gate": reliability_gate,
            "prompt_update_norm": h_adp.new_tensor(0.0),
            "prompt_update_max_norm": h_adp.new_tensor(0.0),
            "prompt_raw_delta_norm": h_adp.new_tensor(0.0),
            "prompt_delta_norm": h_adp.new_tensor(0.0),
            "prompt_gate_mean": effective_gate.mean(),
            "prompt_gate_min": effective_gate.min() if effective_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_gate_max": effective_gate.max() if effective_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_raw_gate_mean": reliability_gate.mean(),
            "prompt_raw_gate_min": reliability_gate.min() if reliability_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "prompt_raw_gate_max": reliability_gate.max() if reliability_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "p22_gate_std": effective_gate.std(unbiased=False) if effective_gate.numel() > 0 else h_adp.new_tensor(0.0),
            "p22_gate_open_ratio": (effective_gate > 0.5).to(dtype=h_adp.dtype).mean(),
            "prompt_update_mask_ratio": update_mask_eff.to(dtype=h_adp.dtype).mean(),
            "prompt_update_clip_ratio": h_adp.new_tensor(0.0),
            "high_frequency_norm": h_adp.new_tensor(0.0),
            "low_frequency_norm": h_adp.new_tensor(0.0),
            **transition_stats,
        }
