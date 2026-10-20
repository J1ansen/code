"""Hub-aware prompt receiver for P23 v0.1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.p23_static_prompt_graph import P23PromptGraphState


class P23HubAwarePromptReceiver(nn.Module):
    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        receiver_cfg = self.config.get("receiver", {})
        node_to_prompt_cfg = self.config.get("node_to_prompt", {})
        self.max_update_norm = float(receiver_cfg.get("max_update_norm", 0.05))
        self.prompt_dropout = float(receiver_cfg.get("prompt_dropout", 0.10))
        self.node_to_prompt_enabled = bool(node_to_prompt_cfg.get("enabled", False))
        self.node_to_prompt_scale = float(node_to_prompt_cfg.get("scale", 1.0))
        init_gate_bias = float(receiver_cfg.get("init_gate_bias", -2.0))

        self.prompt_proj = nn.Linear(self.source_dim, self.hidden_dim)
        self.node_to_prompt_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.prompt_context_norm = nn.LayerNorm(self.hidden_dim)
        self.hub_gate = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, max(8, self.hidden_dim // 4)),
            nn.ReLU(),
            nn.Dropout(self.prompt_dropout),
            nn.Linear(max(8, self.hidden_dim // 4), 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim + 3),
            nn.Linear(4 * self.hidden_dim + 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.prompt_dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.node_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + 3),
            nn.Linear(self.hidden_dim + 3, max(8, self.hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(self.prompt_dropout),
            nn.Linear(max(8, self.hidden_dim // 2), 1),
        )
        nn.init.constant_(self.hub_gate[-1].bias, init_gate_bias)
        nn.init.constant_(self.node_gate[-1].bias, init_gate_bias)

    def _hub_features(self, state: P23PromptGraphState, ref: torch.Tensor) -> torch.Tensor:
        stats = state.feature_static_stats
        keys = ["idf", "cohesion", "pool_rel", "hub_score", "static_reliability"]
        if state.prompt_x.size(0) == 0:
            return ref.new_zeros((0, len(keys)))
        return torch.stack(
            [stats.get(key, ref.new_zeros(state.prompt_x.size(0))).to(device=ref.device, dtype=ref.dtype) for key in keys],
            dim=-1,
        )

    def _edge_softmax(self, scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        alpha = scores.new_zeros(scores.shape)
        if scores.numel() == 0:
            return alpha
        unique_dst = torch.unique(dst)
        for node in unique_dst.tolist():
            mask = dst == int(node)
            alpha[mask] = F.softmax(scores[mask], dim=0)
        return alpha

    def _group_entropy(self, probs: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        if probs.numel() == 0:
            return probs.new_tensor(0.0)
        entropy_terms = []
        for item in torch.unique(group).tolist():
            mask = group == int(item)
            group_probs = probs[mask]
            entropy_terms.append(-(group_probs * group_probs.clamp_min(1e-12).log()).sum())
        return torch.stack(entropy_terms).mean() if entropy_terms else probs.new_tensor(0.0)

    def forward(
        self,
        *,
        h_base: torch.Tensor,
        state: P23PromptGraphState,
        edge_scale: torch.Tensor | float = 1.0,
        prompt_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = h_base.device
        dtype = h_base.dtype
        state = state.to(device, dtype)
        num_nodes = int(h_base.size(0))
        num_prompts = int(state.prompt_x.size(0))
        prompt_x = state.prompt_x
        if prompt_delta is not None and prompt_delta.numel() > 0:
            prompt_x = prompt_x + prompt_delta.to(device=device, dtype=dtype)
        prompt_h = self.prompt_proj(prompt_x)
        prompt_h = F.dropout(prompt_h, p=self.prompt_dropout, training=self.training)

        hub_features = self._hub_features(state, h_base)
        hub_gate = torch.sigmoid(self.hub_gate(hub_features).squeeze(-1)) if num_prompts > 0 else h_base.new_zeros(0)

        edge_index = state.prompt_edge_index
        if edge_index.numel() == 0:
            update = h_base.new_zeros(h_base.shape)
            node_gate = h_base.new_zeros(num_nodes)
            edge_attention = h_base.new_zeros(0)
            reverse_attention = h_base.new_zeros(0)
            edge_entropy = h_base.new_tensor(0.0)
            reverse_entropy = h_base.new_tensor(0.0)
            message = h_base.new_zeros(h_base.shape)
            prompt_context = h_base.new_zeros((num_prompts, self.hidden_dim))
        else:
            src_prompt = edge_index[0]
            dst_node = edge_index[1]
            h_i = h_base[dst_node]
            p_b = prompt_h[src_prompt]
            static_rel = state.feature_static_stats["static_reliability"][src_prompt].to(device=device, dtype=dtype)
            edge_features = torch.cat(
                [
                    h_i,
                    p_b,
                    (h_i - p_b).abs(),
                    h_i * p_b,
                    state.heterophily_risk[dst_node].unsqueeze(-1),
                    static_rel.unsqueeze(-1),
                    hub_gate[src_prompt].unsqueeze(-1),
                ],
                dim=-1,
            )
            score = self.edge_mlp(edge_features).squeeze(-1)
            score = score + (hub_gate[src_prompt] + 1e-6).log() + (static_rel + 1e-6).log()
            reverse_attention = h_base.new_zeros(score.shape)
            prompt_context = h_base.new_zeros((num_prompts, self.hidden_dim))
            if self.node_to_prompt_enabled:
                reverse_attention = self._edge_softmax(score, src_prompt, num_prompts)
                node_msg = reverse_attention.unsqueeze(-1) * self.node_to_prompt_proj(h_i)
                prompt_context.index_add_(0, src_prompt, node_msg)
                prompt_context = self.prompt_context_norm(prompt_context)
                prompt_h = prompt_h + float(self.node_to_prompt_scale) * prompt_context
                prompt_h = F.dropout(prompt_h, p=self.prompt_dropout, training=self.training)
                p_b = prompt_h[src_prompt]
            edge_attention = self._edge_softmax(score, dst_node, num_nodes)
            msg_edge = edge_attention.unsqueeze(-1) * hub_gate[src_prompt].unsqueeze(-1) * p_b
            message = h_base.new_zeros(h_base.shape)
            message.index_add_(0, dst_node, msg_edge)

            node_features = torch.cat(
                [
                    h_base,
                    state.heterophily_risk.unsqueeze(-1),
                    state.node_uncertainty.unsqueeze(-1),
                    state.branch_disagreement.unsqueeze(-1),
                ],
                dim=-1,
            )
            node_gate = torch.sigmoid(self.node_gate(node_features).squeeze(-1)) * state.pool_mask.to(dtype=dtype)
            update = node_gate.unsqueeze(-1) * message
            update_norm = update.norm(dim=-1, keepdim=True)
            scale = torch.clamp(self.max_update_norm / update_norm.clamp_min(1e-12), max=1.0)
            update = update * scale
            edge_entropy = self._group_entropy(edge_attention, dst_node)
            reverse_entropy = self._group_entropy(reverse_attention, src_prompt)

        edge_scale_tensor = (
            edge_scale.to(device=device, dtype=dtype)
            if isinstance(edge_scale, torch.Tensor)
            else h_base.new_tensor(float(edge_scale))
        )
        update = update * edge_scale_tensor.clamp_min(0.0)
        h_adp = h_base + update
        update_norm_flat = update.norm(dim=-1)
        pool = state.pool_mask.to(device=device, dtype=torch.bool)
        pool_update_norm = update_norm_flat[pool] if bool(pool.any()) else update_norm_flat
        hub_score = state.feature_static_stats.get("hub_score", hub_gate.new_zeros(hub_gate.shape)).to(device=device, dtype=dtype)
        hub_budget_loss = (hub_gate * hub_score).mean() if hub_gate.numel() > 0 else h_base.new_tensor(0.0)
        norm_loss = pool_update_norm.mean() if pool_update_norm.numel() > 0 else h_base.new_tensor(0.0)
        return {
            "h_adp": h_adp,
            "prompt_update": update,
            "node_receive_gate": node_gate,
            "hub_gate": hub_gate,
            "edge_attention": edge_attention,
            "node_to_prompt_attention": reverse_attention,
            "prompt_context": prompt_context,
            "prompt_message": message,
            "prompt_message_norm": message.norm(dim=-1),
            "prompt_update_norm": update_norm_flat,
            "prompt_update_norm_mean": update_norm_flat.mean() if update_norm_flat.numel() > 0 else h_base.new_tensor(0.0),
            "prompt_update_norm_max": update_norm_flat.max() if update_norm_flat.numel() > 0 else h_base.new_tensor(0.0),
            "edge_attention_entropy": edge_entropy,
            "node_to_prompt_attention_entropy": reverse_entropy,
            "prompt_context_norm_mean": prompt_context.norm(dim=-1).mean() if prompt_context.numel() > 0 else h_base.new_tensor(0.0),
            "node_to_prompt_enabled": h_base.new_tensor(float(self.node_to_prompt_enabled)),
            "learn_prompt_delta_enabled": h_base.new_tensor(float(prompt_delta is not None and prompt_delta.numel() > 0)),
            "prompt_delta_norm_mean": prompt_delta.norm(dim=-1).mean().to(device=device, dtype=dtype) if prompt_delta is not None and prompt_delta.numel() > 0 else h_base.new_tensor(0.0),
            "hub_budget_loss": hub_budget_loss,
            "norm_loss": norm_loss,
            "gate_closed_ratio": (node_gate < 0.05).to(dtype=dtype).mean() if node_gate.numel() > 0 else h_base.new_tensor(0.0),
            "large_update_ratio": (update_norm_flat > self.max_update_norm * 0.95).to(dtype=dtype).mean() if update_norm_flat.numel() > 0 else h_base.new_tensor(0.0),
            "hub_suppression_ratio": (hub_gate < 0.05).to(dtype=dtype).mean() if hub_gate.numel() > 0 else h_base.new_tensor(0.0),
        }


class P23V01PromptModule(nn.Module):
    """P23 v0.1: static discrete feature prompt graph plus dynamic receiver."""

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        from models.p23_static_prompt_graph import P23StaticPromptGraphBuilder

        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = dict(config or {})
        prompt_token_cfg = self.config.get("prompt_token", {})
        self.learn_prompt_delta = bool(
            prompt_token_cfg.get("learn_delta", self.config.get("learn_prompt_delta", False))
        )
        self.prompt_delta_init_std = float(prompt_token_cfg.get("delta_init_std", 0.0))
        self.builder = P23StaticPromptGraphBuilder(self.config)
        self.receiver = P23HubAwarePromptReceiver(source_dim, hidden_dim, self.config)
        self.state: P23PromptGraphState | None = None
        self.prompt_delta: nn.Parameter | None = None

    def _reset_prompt_delta(self, state: P23PromptGraphState) -> None:
        if not self.learn_prompt_delta:
            self.prompt_delta = None
            return
        delta = torch.zeros_like(state.prompt_x)
        if self.prompt_delta_init_std > 0.0 and delta.numel() > 0:
            delta.normal_(mean=0.0, std=self.prompt_delta_init_std)
        self.prompt_delta = nn.Parameter(delta)

    def build_state(
        self,
        *,
        x_raw: torch.Tensor,
        z_snapshot: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        no_prompt_logits: torch.Tensor | None = None,
        h_pre_snapshot: torch.Tensor | None = None,
        h_adp0_snapshot: torch.Tensor | None = None,
    ) -> P23PromptGraphState:
        self.state = self.builder.build(
            x_raw=x_raw,
            z_snapshot=z_snapshot,
            edge_index=edge_index,
            train_mask=train_mask,
            no_prompt_logits=no_prompt_logits,
            h_pre_snapshot=h_pre_snapshot,
            h_adp0_snapshot=h_adp0_snapshot,
        )
        self._reset_prompt_delta(self.state)
        return self.state

    def _state(self, ref: torch.Tensor) -> P23PromptGraphState:
        if self.state is None:
            raise RuntimeError("P23 v0.1 prompt graph state must be built once before training")
        return self.state.to(ref.device, ref.dtype)

    def _static_aux(self, state: P23PromptGraphState, z: torch.Tensor, edge_index: torch.Tensor, train_mask: torch.Tensor) -> dict[str, Any]:
        stats = state.feature_static_stats
        pool = state.pool_mask.to(device=z.device, dtype=torch.bool)
        train = train_mask.to(device=z.device, dtype=torch.bool)
        risk = state.heterophily_risk
        nonpool = ~pool

        def mean_or_zero(values: torch.Tensor) -> torch.Tensor:
            return values.mean() if values.numel() > 0 else z.new_tensor(0.0)

        return {
            "prompt_edge_weight": state.prompt_edge_prior,
            "prompt_usage": state.prompt_edge_prior.new_zeros(state.prompt_x.size(0)),
            "prompt_usage_entropy": z.new_tensor(0.0),
            "connected_edge_count": int(state.prompt_edge_index.size(1)),
            "edge_type_counts": [int(edge_index.size(1)), 0, int(state.prompt_edge_index.size(1))],
            "p23_feature_ids_tensor": state.feature_ids,
            "p23_prompt_edge_index_tensor": state.prompt_edge_index,
            "p23_df_pool_tensor": stats.get("df_pool", z.new_zeros(0)),
            "p23_df_global_tensor": stats.get("df_global", z.new_zeros(0)),
            "p23_pool_rel_tensor": stats.get("pool_rel", z.new_zeros(0)),
            "p23_static_reliability_tensor": stats.get("static_reliability", z.new_zeros(0)),
            "p23_log_topk_feature_prompt_delta": int(self.config.get("p23_log_topk_feature_prompt_delta", 20)),
            "use_receiver_only_prompt": 1.0,
            "p23_static_topology": z.new_tensor(1.0),
            "p23_graph_risk": state.graph_risk,
            "p23_pool_ratio": pool.to(dtype=z.dtype).mean() if pool.numel() > 0 else z.new_tensor(0.0),
            "p23_pool_size": z.new_tensor(float(pool.sum().item())),
            "p23_train_nodes_in_pool_ratio": (
                (pool & train).to(dtype=z.dtype).sum() / train.to(dtype=z.dtype).sum().clamp_min(1.0)
            ),
            "p23_pool_risk_mean": mean_or_zero(risk[pool]),
            "p23_nonpool_risk_mean": mean_or_zero(risk[nonpool]),
            "p23_feature_prompt_count": z.new_tensor(float(state.prompt_x.size(0))),
            "p23_prompt_edge_count": z.new_tensor(float(state.prompt_edge_index.size(1))),
            "p23_df_pool_mean": mean_or_zero(stats.get("df_pool", z.new_zeros(0))),
            "p23_df_global_mean": mean_or_zero(stats.get("df_global", z.new_zeros(0))),
            "p23_pool_freq_mean": mean_or_zero(stats.get("pool_freq", z.new_zeros(0))),
            "p23_pool_concentration_mean": mean_or_zero(stats.get("pool_rel", z.new_zeros(0))),
            "p23_static_reliability_mean": mean_or_zero(stats.get("static_reliability", z.new_zeros(0))),
            "p23_hub_score_mean": mean_or_zero(stats.get("hub_score", z.new_zeros(0))),
            "p23_learn_prompt_delta": z.new_tensor(float(self.learn_prompt_delta)),
            "p23_node_to_prompt_enabled": z.new_tensor(float(self.receiver.node_to_prompt_enabled)),
            "p23_ce_delta_pool_mean": z.new_tensor(0.0),
            "p23_ce_delta_train_mean": z.new_tensor(0.0),
        }

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
    ) -> dict[str, Any]:
        state = self._state(z)
        edge_weight = torch.ones(edge_index.size(1), dtype=z.dtype, device=z.device)
        edge_scale = (
            edge_scale_multiplier.to(device=z.device, dtype=z.dtype)
            if isinstance(edge_scale_multiplier, torch.Tensor)
            else z.new_tensor(float(edge_scale_multiplier))
        )
        return {
            "adapted_x": z,
            "adapted_edge_index": edge_index,
            "adapted_edge_weight": edge_weight,
            "adapted_edge_type": torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
            "prompt_node_x": state.prompt_x,
            "pool_mask": state.pool_mask,
            "prompt_edge_count": int(state.prompt_edge_index.size(1)),
            "edge_scale": edge_scale,
            "aux": self._static_aux(state, z, edge_index, train_mask),
        }

    def apply_receiver(self, h_base: torch.Tensor, edge_scale: torch.Tensor | float = 1.0) -> dict[str, torch.Tensor]:
        prompt_delta = self.prompt_delta if self.learn_prompt_delta else None
        return self.receiver(h_base=h_base, state=self._state(h_base), edge_scale=edge_scale, prompt_delta=prompt_delta)

    def receiver_aux(self, receiver_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hub_gate = receiver_out["hub_gate"]
        node_gate = receiver_out["node_receive_gate"]
        return {
            "p23_hub_gate_mean": hub_gate.mean() if hub_gate.numel() > 0 else node_gate.new_tensor(0.0),
            "p23_hub_gate_min": hub_gate.min() if hub_gate.numel() > 0 else node_gate.new_tensor(0.0),
            "p23_hub_gate_max": hub_gate.max() if hub_gate.numel() > 0 else node_gate.new_tensor(0.0),
            "p23_node_gate_mean": node_gate.mean() if node_gate.numel() > 0 else node_gate.new_tensor(0.0),
            "p23_node_gate_min": node_gate.min() if node_gate.numel() > 0 else node_gate.new_tensor(0.0),
            "p23_node_gate_max": node_gate.max() if node_gate.numel() > 0 else node_gate.new_tensor(0.0),
            "p23_edge_attention_entropy": receiver_out["edge_attention_entropy"],
            "p23_node_to_prompt_attention_entropy": receiver_out["node_to_prompt_attention_entropy"],
            "p23_prompt_context_norm_mean": receiver_out["prompt_context_norm_mean"],
            "p23_node_to_prompt_enabled": receiver_out["node_to_prompt_enabled"],
            "p23_learn_prompt_delta_enabled": receiver_out["learn_prompt_delta_enabled"],
            "p23_prompt_delta_norm_mean": receiver_out["prompt_delta_norm_mean"],
            "p23_prompt_message_norm_mean": receiver_out["prompt_message_norm"].mean(),
            "p23_prompt_update_norm_mean": receiver_out["prompt_update_norm_mean"],
            "p23_prompt_update_norm_max": receiver_out["prompt_update_norm_max"],
            "p23_norm_loss": receiver_out["norm_loss"],
            "p23_hub_budget_loss": receiver_out["hub_budget_loss"],
            "p23_gate_closed_ratio": receiver_out["gate_closed_ratio"],
            "p23_large_update_ratio": receiver_out["large_update_ratio"],
            "p23_hub_suppression_ratio": receiver_out["hub_suppression_ratio"],
        }
