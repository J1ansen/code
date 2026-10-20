"""Projector-synchronized representation adapter for the static P23 topology."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.prompt_graph_construction import PromptGraphConstruction
from models.p23_projector_prompt_receiver import P23ProjectorPromptReceiver


class P23ProjectorSynchronizedPromptGraphAdapter(PromptGraphConstruction):
    """Keep P23 topology static while rematerializing prompt vectors from current z."""

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(source_dim, hidden_dim, config)
        self.refresh_prompt_repr = bool(self.config.get("refresh_prompt_repr_from_current_z", True))
        self.sync_fusion = str(self.config.get("projector_sync_fusion", "direct")).lower()
        if self.sync_fusion not in {
            "direct",
            "anchored_bootstrap",
            "centered_residual",
            "bank_standardized",
        }:
            raise ValueError(f"Unsupported projector_sync_fusion: {self.sync_fusion}")
        self.sync_mix_init = float(self.config.get("projector_sync_mix_init", 0.0))
        self.sync_mix_max = float(self.config.get("projector_sync_mix_max", 0.25))
        self.sync_warmup_epochs = max(0, int(self.config.get("projector_sync_warmup_epochs", 40)))
        self.sync_norm_mode = str(self.config.get("projector_sync_norm_mode", "cached_per_prompt")).lower()
        if self.sync_norm_mode not in {"none", "cached_per_prompt", "cached_global"}:
            raise ValueError(f"Unsupported projector_sync_norm_mode: {self.sync_norm_mode}")
        self.sync_max_delta_ratio = max(0.0, float(self.config.get("projector_sync_max_delta_ratio", 0.50)))
        self.sync_center_detach = bool(self.config.get("projector_sync_center_detach", True))
        self.projector_receiver = P23ProjectorPromptReceiver(
            hidden_dim,
            dict(self.config.get("projector_receiver", {})),
        )
        if self.prompt_residual.enabled:
            raise ValueError("Projector-synchronized P23 currently requires prompt_residual.enabled=false")
        if self.prompt_edge_denoiser.enabled:
            raise ValueError("Projector-synchronized P23 currently requires prompt_edge_denoising.enabled=false")

    def _build_graph(
        self,
        *,
        z: torch.Tensor,
        x_raw: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> dict[str, Any]:
        graph = super()._build_graph(
            z=z,
            x_raw=x_raw,
            edge_index=edge_index,
            train_mask=train_mask,
            labels=labels,
        )
        feature_ids = graph["feature_ids"].to(device=z.device, dtype=torch.long)
        if feature_ids.numel() == 0:
            graph["prompt_aggregation_node_index"] = torch.empty(0, dtype=torch.long, device=z.device)
            graph["prompt_aggregation_feature_local"] = torch.empty(0, dtype=torch.long, device=z.device)
            graph["prompt_aggregation_weight"] = z.new_zeros(0)
            graph["prompt_aggregation_denom"] = z.new_zeros(0)
            return graph

        membership = self._token_membership(x_raw.to(device=z.device))[:, feature_ids]
        node_index, feature_local = torch.nonzero(membership, as_tuple=True)
        aggregation_weight = z.new_full((node_index.numel(),), float(self.unlabeled_node_weight))
        train = train_mask.to(device=z.device, dtype=torch.bool)
        if labels is not None and node_index.numel() > 0 and bool(train.any()):
            y = labels.to(device=z.device, dtype=torch.long)
            dominant = graph["feature_dominant_class"].to(device=z.device, dtype=torch.long)
            labelled_edge = train[node_index]
            label_match = y[node_index] == dominant[feature_local]
            aggregation_weight = torch.where(
                labelled_edge & label_match,
                aggregation_weight.new_tensor(float(self.label_node_weight)),
                torch.where(
                    labelled_edge,
                    aggregation_weight.new_tensor(float(self.offclass_label_weight)),
                    aggregation_weight,
                ),
            )

        denom = z.new_zeros(feature_ids.numel())
        denom.index_add_(0, feature_local, aggregation_weight)
        graph["prompt_aggregation_node_index"] = node_index
        graph["prompt_aggregation_feature_local"] = feature_local
        graph["prompt_aggregation_weight"] = aggregation_weight
        graph["prompt_aggregation_denom"] = denom.clamp_min(1.0)
        return graph

    def _current_prompt_repr(self, graph: dict[str, Any], z: torch.Tensor) -> torch.Tensor:
        node_index = graph["prompt_aggregation_node_index"].to(device=z.device, dtype=torch.long)
        feature_local = graph["prompt_aggregation_feature_local"].to(device=z.device, dtype=torch.long)
        aggregation_weight = graph["prompt_aggregation_weight"].to(device=z.device, dtype=z.dtype)
        denom = graph["prompt_aggregation_denom"].to(device=z.device, dtype=z.dtype)
        prompt_x = z.new_zeros((denom.numel(), z.size(-1)))
        if node_index.numel() > 0:
            prompt_x.index_add_(0, feature_local, z[node_index] * aggregation_weight.unsqueeze(-1))
            prompt_x = prompt_x / denom.unsqueeze(-1)
        if self.detach_feature_prompt:
            prompt_x = prompt_x.detach()
        return prompt_x

    def _sync_mix(self, epoch: int | None, ref: torch.Tensor) -> torch.Tensor:
        if self.sync_fusion == "direct":
            return ref.new_tensor(1.0)
        current_epoch = max(0, int(epoch or 0))
        if self.sync_warmup_epochs <= 0:
            progress = 1.0
        else:
            progress = min(float(current_epoch) / float(self.sync_warmup_epochs), 1.0)
        mix = self.sync_mix_init + (self.sync_mix_max - self.sync_mix_init) * progress
        return ref.new_tensor(min(max(mix, 0.0), 1.0))

    def _align_current_prompt(
        self,
        current_prompt_x: torch.Tensor,
        cached_prompt_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current_prompt_x.numel() == 0 or self.sync_norm_mode == "none":
            one = current_prompt_x.new_tensor(1.0)
            return current_prompt_x, one, current_prompt_x.new_tensor(0.0)

        eps = 1e-12
        current_norm = current_prompt_x.norm(dim=-1, keepdim=True)
        cached_norm = cached_prompt_x.norm(dim=-1, keepdim=True)
        if self.sync_norm_mode == "cached_per_prompt":
            scale = cached_norm / current_norm.clamp_min(eps)
        else:
            scale = cached_norm.mean() / current_norm.mean().clamp_min(eps)
        aligned = current_prompt_x * scale
        return aligned, scale.mean().detach(), scale.std(unbiased=False).detach()

    def _anchored_prompt_repr(
        self,
        current_prompt_x: torch.Tensor,
        cached_prompt_x: torch.Tensor,
        epoch: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aligned_prompt_x, norm_scale_mean, norm_scale_std = self._align_current_prompt(
            current_prompt_x,
            cached_prompt_x,
        )
        delta = aligned_prompt_x - cached_prompt_x
        cached_norm = cached_prompt_x.norm(dim=-1, keepdim=True)
        delta_norm = delta.norm(dim=-1, keepdim=True)
        max_delta_norm = self.sync_max_delta_ratio * cached_norm
        clip_scale = (max_delta_norm / delta_norm.clamp_min(1e-12)).clamp(max=1.0)
        if self.sync_max_delta_ratio <= 0.0:
            clip_scale = torch.zeros_like(clip_scale)
        clipped_delta = delta * clip_scale
        mix = self._sync_mix(epoch, cached_prompt_x)
        prompt_x = cached_prompt_x + mix * clipped_delta
        stats = {
            "mix": mix.detach(),
            "norm_scale_mean": norm_scale_mean,
            "norm_scale_std": norm_scale_std,
            "clip_ratio": (clip_scale < 1.0).to(dtype=cached_prompt_x.dtype).mean().detach(),
            "aligned_delta_ratio": (
                clipped_delta.norm(dim=-1).mean()
                / cached_prompt_x.norm(dim=-1).mean().clamp_min(1e-12)
            ).detach(),
        }
        return prompt_x, stats

    def _centered_residual_prompt_repr(
        self,
        current_prompt_x: torch.Tensor,
        cached_prompt_x: torch.Tensor,
        z: torch.Tensor,
        epoch: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        center = z.mean(dim=0, keepdim=True)
        if self.sync_center_detach:
            center = center.detach()
        residual = current_prompt_x - center
        cached_norm = cached_prompt_x.norm(dim=-1, keepdim=True)
        cached_unit = cached_prompt_x / cached_norm.clamp_min(1e-12)
        residual = residual - (residual * cached_unit).sum(dim=-1, keepdim=True) * cached_unit

        residual_norm = residual.norm(dim=-1, keepdim=True)
        max_delta_norm = self.sync_max_delta_ratio * cached_norm
        clip_scale = (max_delta_norm / residual_norm.clamp_min(1e-12)).clamp(max=1.0)
        if self.sync_max_delta_ratio <= 0.0:
            clip_scale = torch.zeros_like(clip_scale)
        clipped_residual = residual * clip_scale
        mix = self._sync_mix(epoch, cached_prompt_x)
        prompt_x = cached_prompt_x + mix * clipped_residual
        stats = {
            "mix": mix.detach(),
            "norm_scale_mean": z.new_tensor(1.0),
            "norm_scale_std": z.new_tensor(0.0),
            "clip_ratio": (clip_scale < 1.0).to(dtype=cached_prompt_x.dtype).mean().detach(),
            "aligned_delta_ratio": (
                clipped_residual.norm(dim=-1).mean()
                / cached_prompt_x.norm(dim=-1).mean().clamp_min(1e-12)
            ).detach(),
            "centered_residual_norm": residual_norm.mean().detach(),
            "centered_residual_cached_cosine": (
                F.cosine_similarity(clipped_residual, cached_prompt_x, dim=-1).mean().detach()
                if clipped_residual.numel() > 0
                else z.new_tensor(0.0)
            ),
        }
        return prompt_x, stats

    def _bank_standardized_prompt_repr(
        self,
        current_prompt_x: torch.Tensor,
        cached_prompt_x: torch.Tensor,
        epoch: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if current_prompt_x.numel() == 0:
            return self._anchored_prompt_repr(current_prompt_x, cached_prompt_x, epoch)
        current_mean = current_prompt_x.mean(dim=0, keepdim=True)
        cached_mean = cached_prompt_x.mean(dim=0, keepdim=True)
        current_std = current_prompt_x.std(dim=0, keepdim=True, unbiased=False)
        cached_std = cached_prompt_x.std(dim=0, keepdim=True, unbiased=False)
        stable_dim = (current_std > 1e-6) & (cached_std > 1e-6)
        standardized = (current_prompt_x - current_mean) / current_std.clamp_min(1e-6)
        aligned_prompt_x = standardized * cached_std + cached_mean
        aligned_prompt_x = torch.where(stable_dim, aligned_prompt_x, cached_prompt_x)

        delta = aligned_prompt_x - cached_prompt_x
        cached_norm = cached_prompt_x.norm(dim=-1, keepdim=True)
        delta_norm = delta.norm(dim=-1, keepdim=True)
        max_delta_norm = self.sync_max_delta_ratio * cached_norm
        clip_scale = (max_delta_norm / delta_norm.clamp_min(1e-12)).clamp(max=1.0)
        if self.sync_max_delta_ratio <= 0.0:
            clip_scale = torch.zeros_like(clip_scale)
        clipped_delta = delta * clip_scale
        mix = self._sync_mix(epoch, cached_prompt_x)
        prompt_x = cached_prompt_x + mix * clipped_delta
        stats = {
            "mix": mix.detach(),
            "norm_scale_mean": (cached_std / current_std.clamp_min(1e-6))[stable_dim].mean().detach()
            if bool(stable_dim.any())
            else current_prompt_x.new_tensor(1.0),
            "norm_scale_std": (cached_std / current_std.clamp_min(1e-6))[stable_dim].std(unbiased=False).detach()
            if bool(stable_dim.any())
            else current_prompt_x.new_tensor(0.0),
            "clip_ratio": (clip_scale < 1.0).to(dtype=cached_prompt_x.dtype).mean().detach(),
            "aligned_delta_ratio": (
                clipped_delta.norm(dim=-1).mean()
                / cached_prompt_x.norm(dim=-1).mean().clamp_min(1e-12)
            ).detach(),
            "bank_stable_dim_ratio": stable_dim.to(dtype=cached_prompt_x.dtype).mean().detach(),
            "bank_aligned_mean_error": (prompt_x.mean(dim=0) - cached_mean.squeeze(0)).norm().detach(),
        }
        return prompt_x, stats

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        out = super().forward(**kwargs)
        z = kwargs["z"]
        cached_prompt_x = out["prompt_node_x"]
        aux = out["aux"]
        aux["p23_projector_sync_enabled"] = z.new_tensor(float(self.refresh_prompt_repr))
        if not self.refresh_prompt_repr or self._cached_graph is None:
            aux["p23_projector_sync_delta_norm"] = z.new_tensor(0.0)
            aux["p23_projector_sync_delta_ratio"] = z.new_tensor(0.0)
            aux["p23_projector_sync_cached_cosine"] = z.new_tensor(1.0)
            aux["p23_projector_sync_mix"] = z.new_tensor(0.0)
            return out

        current_prompt_x = self._current_prompt_repr(self._cached_graph, z)
        if current_prompt_x.shape != cached_prompt_x.shape:
            raise RuntimeError(
                "Projector-synchronized prompt shape mismatch: "
                f"current={tuple(current_prompt_x.shape)} cached={tuple(cached_prompt_x.shape)}"
            )
        raw_delta = current_prompt_x - cached_prompt_x
        delta_norm = raw_delta.norm(dim=-1).mean() if raw_delta.numel() > 0 else z.new_tensor(0.0)
        cached_norm = cached_prompt_x.norm(dim=-1).mean() if cached_prompt_x.numel() > 0 else z.new_tensor(0.0)
        cosine = (
            F.cosine_similarity(current_prompt_x, cached_prompt_x, dim=-1).mean()
            if current_prompt_x.numel() > 0
            else z.new_tensor(1.0)
        )

        if self.sync_fusion == "anchored_bootstrap":
            prompt_x, sync_stats = self._anchored_prompt_repr(
                current_prompt_x,
                cached_prompt_x,
                kwargs.get("epoch"),
            )
        elif self.sync_fusion == "centered_residual":
            prompt_x, sync_stats = self._centered_residual_prompt_repr(
                current_prompt_x,
                cached_prompt_x,
                z,
                kwargs.get("epoch"),
            )
        elif self.sync_fusion == "bank_standardized":
            prompt_x, sync_stats = self._bank_standardized_prompt_repr(
                current_prompt_x,
                cached_prompt_x,
                kwargs.get("epoch"),
            )
        else:
            prompt_x = current_prompt_x
            sync_stats = {
                "mix": z.new_tensor(1.0),
                "norm_scale_mean": z.new_tensor(1.0),
                "norm_scale_std": z.new_tensor(0.0),
                "clip_ratio": z.new_tensor(0.0),
                "aligned_delta_ratio": (delta_norm / cached_norm.clamp_min(1e-12)).detach(),
                "centered_residual_norm": z.new_tensor(0.0),
                "centered_residual_cached_cosine": z.new_tensor(0.0),
                "bank_stable_dim_ratio": z.new_tensor(0.0),
                "bank_aligned_mean_error": z.new_tensor(0.0),
            }

        effective_delta = prompt_x - cached_prompt_x
        effective_delta_norm = (
            effective_delta.norm(dim=-1).mean() if effective_delta.numel() > 0 else z.new_tensor(0.0)
        )

        out["adapted_x"] = torch.cat([z, prompt_x], dim=0)
        out["prompt_node_x"] = prompt_x
        aux["p23_projector_sync_delta_norm"] = delta_norm.detach()
        aux["p23_projector_sync_delta_ratio"] = (delta_norm / cached_norm.clamp_min(1e-12)).detach()
        aux["p23_projector_sync_cached_cosine"] = cosine.detach()
        aux["p23_projector_sync_current_norm"] = (
            current_prompt_x.norm(dim=-1).mean().detach()
            if current_prompt_x.numel() > 0
            else z.new_tensor(0.0)
        )
        aux["p23_projector_sync_mix"] = sync_stats["mix"]
        aux["p23_projector_sync_norm_scale_mean"] = sync_stats["norm_scale_mean"]
        aux["p23_projector_sync_norm_scale_std"] = sync_stats["norm_scale_std"]
        aux["p23_projector_sync_clip_ratio"] = sync_stats["clip_ratio"]
        aux["p23_projector_sync_aligned_delta_ratio"] = sync_stats["aligned_delta_ratio"]
        aux["p23_projector_sync_effective_delta_norm"] = effective_delta_norm.detach()
        aux["p23_projector_sync_effective_delta_ratio"] = (
            effective_delta_norm / cached_norm.clamp_min(1e-12)
        ).detach()
        aux["p23_projector_sync_effective_cosine"] = (
            F.cosine_similarity(prompt_x, cached_prompt_x, dim=-1).mean().detach()
            if prompt_x.numel() > 0
            else z.new_tensor(1.0)
        )
        aux["p23_projector_sync_centered_residual_norm"] = sync_stats.get(
            "centered_residual_norm", z.new_tensor(0.0)
        )
        aux["p23_projector_sync_centered_residual_cached_cosine"] = sync_stats.get(
            "centered_residual_cached_cosine", z.new_tensor(0.0)
        )
        aux["p23_projector_sync_bank_stable_dim_ratio"] = sync_stats.get(
            "bank_stable_dim_ratio", z.new_tensor(0.0)
        )
        aux["p23_projector_sync_bank_aligned_mean_error"] = sync_stats.get(
            "bank_aligned_mean_error", z.new_tensor(0.0)
        )
        return out

    @property
    def projector_receiver_enabled(self) -> bool:
        return self.projector_receiver.enabled

    def apply_projector_receiver(
        self,
        *,
        h_base: torch.Tensor,
        prompt_out: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if self._cached_graph is None:
            raise RuntimeError("Projector receiver requires the P23 graph to be materialized first")
        receiver_out = self.projector_receiver(
            h_base=h_base,
            graph=self._cached_graph,
            prompt_edge_weight=prompt_out.get("aux", {}).get("prompt_edge_weight"),
        )
        aux = prompt_out.setdefault("aux", {})
        aux["p23_projector_receiver_enabled"] = h_base.new_tensor(
            float(self.projector_receiver_enabled)
        )
        aux["p23_projector_receiver_signed_scale"] = receiver_out["signed_scale"].detach()
        aux["p23_projector_receiver_connected_ratio"] = receiver_out["connected_ratio"].detach()
        aux["p23_projector_receiver_context_delta_norm_mean"] = receiver_out[
            "context_delta_norm_mean"
        ].detach()
        aux["p23_projector_receiver_edge_context_norm_mean"] = receiver_out[
            "edge_context_norm_mean"
        ].detach()
        aux["p23_projector_receiver_update_norm_mean"] = receiver_out["update_norm_mean"].detach()
        aux["p23_projector_receiver_update_norm_max"] = receiver_out["update_norm_max"].detach()
        aux["p23_projector_receiver_update_to_base_ratio"] = receiver_out[
            "update_to_base_ratio"
        ].detach()
        return receiver_out
