"""Run main experiments for the prompt graph adaptation framework."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import build_few_shot_split, load_gp2f_precomputed_split, load_node_dataset
from experiments.dual_branch_training import (
    InputAligner,
    _adapter_stats,
    _deep_update,
    _environment_info,
    _model_variant,
    _resolve_path,
    _resolve_run_seeds,
    _split_counts,
    set_seed,
)
from losses import GP2FLossConfig, compute_gp2f_loss
from models import (
    ClassConditionedPatternPromptRouter,
    PromptGraphConstruction,
    DualBranchFramework,
    HeterophilyAwarePromptAdapter,
    P21LiteAdaptiveFilter,
    P21V2HeteroFilter,
    P22ClassPatternEnrichmentBank,
    P23V01PromptModule,
    P23ProjectorSynchronizedPromptGraphAdapter,
    PromptingAdaptation,
    PromptAwareGP2F,
    PromptGraphModuleP1,
    GraphiteStylePromptGraphAdapter,
    SelectiveDiscreteFeaturePromptGraph,
    UtilitySupervisedPatternPromptRouter,
    infer_backbone_dimensions,
    load_pretrained_backbone,
)
from models.p21_adaptive_filter import CHANNEL_NAMES
from models.p21_v2_hetero_filter import P21_V2_CHANNEL_NAMES
from models.class_conditioned_pattern_prompt_router import PATTERN_NAMES, prompt_router_pattern_balance_loss
from models.hetero_prompt_adapter import (
    prompt_adapter_gate_budget_loss,
    prompt_adapter_message_help_loss,
    prompt_adapter_update_norm_loss,
    prompt_adapter_utility_gate_loss,
)
from models.prompt_graph_module import (
    prompt_acceptance_budget_loss,
    prompt_acceptance_loss,
    prompt_balance_loss,
    prompt_class_route_loss,
    prompt_edge_l1_loss,
    prompt_key_proto_loss,
    prompt_role_diversity_loss,
    prompt_usage_consistency_loss,
    prompt_view_entropy_loss,
    prompt_view_prior_loss,
    utility_receive_gate_budget_loss,
)
from models.prompt_module import count_trainable_parameters, mean_neighbor_summary, mean_neighbor_variance
from utils.io import read_yaml, write_json, write_yaml
from utils.metrics import split_metrics


def _format_mean_std(values: list[float], *, scale: float = 100.0, precision: int = 2) -> str:
    if not values:
        return f"{0.0:.{precision}f} ± {0.0:.{precision}f}"
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else 0.0
    return f"{mean * scale:.{precision}f} ± {std * scale:.{precision}f}"


def _format_gp2f_metric(values: list[float]) -> str:
    if not values:
        return "0.0000 ± 0.0000(std)"
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}(std)"


PROMPT_GRAPH_VARIANTS = {
    "noprompt",
    "p1_graph",
    "p1_random_pool",
    "p2_prompt_aware",
    "p2_strength",
    "p2_multiview",
    "p2_pattern_benefit",
    "p5_multi_expert",
    "p5_lite_receiver",
    "p13_utility_correction",
    "p14_freeze_prompt_adapter",
    "p15_hetero_prompt_adapter",
    "p16_support_prompt_adapter",
    "p17_selective_support_adapter",
    "p18_episode_consistency_adapter",
    "p20_class_conditioned_pattern_prompt_router",
    "p20_utility_supervised_pattern_prompt_router",
    "p21_lite_adaptive_filter",
    "p21_v2_hetero_filter",
    "p22_class_pattern_enrichment_bank",
    "p22_reliability_calibrated_basis_bank",
    "p22_v031_conservative_reliability_basis_bank",
    "p22_v04_minimal_transition_basis",
    "p23_v0_2_message_passing",
    "p23_v0_1",
    "p23_selective_discrete_feature_prompting",
    "p23_graphite_adapter",
    "p23_selective_graphite_adapter",
    "p23_projector_synchronized_graphite_adapter",
    "p24_prompt_edge_denoising",
    "p2_strength_random_pool",
    "p2_no_node_to_prompt",
    "p2_no_prompt_to_node",
}

COMPONENT_ABLATIONS = {
    "none",
    "no_prompt_graph",
    "no_adapter",
    "no_target_adaptation_branch",
    "no_frozen_branch",
}

TRIAL_SEED_MODES = {
    "stream",
    "split",
    "increment",
}


def _prompt_variant(config: dict[str, Any]) -> str:
    experiment_cfg = config.get("experiment", {})
    prompt_graph_cfg = config.get("prompt_graph", {})
    return str(experiment_cfg.get("prompt_variant", prompt_graph_cfg.get("variant", "p1_graph")))


def _apply_component_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Apply one isolated component ablation after variant defaults are resolved."""

    out = copy.deepcopy(config)
    experiment = out.setdefault("experiment", {})
    ablation = str(experiment.get("ablation", "none"))
    if ablation not in COMPONENT_ABLATIONS:
        raise ValueError(f"Unsupported component ablation={ablation!r}")
    experiment["ablation"] = ablation

    if ablation == "no_prompt_graph":
        out.setdefault("prompt_graph", {})["enabled"] = False
    elif ablation == "no_adapter":
        model = out.setdefault("model", {})
        model["use_adapters"] = False
        model["fusion_mode"] = "dual_branch"

        # Formal no-adapter protocol: retain the P23 prompt topology and its
        # edge weighting, but remove every learned representation update from
        # the prompt branch before the shared frozen GNN propagation.
        prompt_graph = out.setdefault("prompt_graph", {})
        prompt_graph["enabled"] = True
        prompt_graph.setdefault("prompt_residual", {})["enabled"] = False
        prompt_graph.setdefault("prompt_edge_denoising", {})["enabled"] = False
        prompt_graph.setdefault("projector_receiver", {})["enabled"] = False
        out.setdefault("prompt_aware", {})["enabled"] = False
        out.setdefault("prompt_adapter", {})["enabled"] = False

        training = out.setdefault("training", {})
        training["train_prompt_graph_module"] = True
        training["train_prompt_adapter"] = False
        training["train_prompt_aware_module"] = False
        training["prompt_graph_trainable_scope"] = "edge_weight_only"
    elif ablation == "no_target_adaptation_branch":
        model = out.setdefault("model", {})
        model["use_adapters"] = False
        model["fusion_mode"] = "frozen_only"

        prompt_graph = out.setdefault("prompt_graph", {})
        prompt_graph["enabled"] = False
        prompt_graph.setdefault("prompt_residual", {})["enabled"] = False
        prompt_graph.setdefault("prompt_edge_denoising", {})["enabled"] = False
        prompt_graph.setdefault("projector_receiver", {})["enabled"] = False
        out.setdefault("prompt_aware", {})["enabled"] = False
        out.setdefault("prompt_adapter", {})["enabled"] = False

        training = out.setdefault("training", {})
        training["train_prompt_graph_module"] = False
        training["train_prompt_adapter"] = False
        training["train_prompt_aware_module"] = False

        loss = out.setdefault("loss", {})
        loss["use_original_contrastive"] = False
        loss["use_original_topology_fusion"] = False
        loss["lambda_ctr"] = 0.0
        loss["lambda_fus"] = 0.0
    elif ablation == "no_frozen_branch":
        model = out.setdefault("model", {})
        model["use_adapters"] = True
        model["fusion_mode"] = "adapted_only"
        out.setdefault("prompt_graph", {})["enabled"] = True

        loss = out.setdefault("loss", {})
        loss["use_original_contrastive"] = False
        loss["use_original_topology_fusion"] = False
        loss["lambda_ctr"] = 0.0
        loss["lambda_fus"] = 0.0
    return out


def _resolve_trial_seed_protocol(
    experiment_cfg: dict[str, Any],
    *,
    base_seed: int,
    trial_index: int,
    trials_per_seed: int,
) -> tuple[int, int | None, str]:
    """Resolve data-split and model RNG seeds without changing legacy defaults."""

    fixed_split_seed = experiment_cfg.get("fixed_split_seed")
    if fixed_split_seed is not None:
        split_seed = int(fixed_split_seed)
    elif trials_per_seed == 1:
        split_seed = int(base_seed)
    else:
        split_seed = int(base_seed) * 1000 + int(trial_index) + 1

    configured_mode = experiment_cfg.get("trial_seed_mode")
    if configured_mode is None:
        mode = "split" if bool(experiment_cfg.get("reset_seed_per_trial", False)) else "stream"
    else:
        mode = str(configured_mode)
    if mode not in TRIAL_SEED_MODES:
        raise ValueError(
            f"Unsupported trial_seed_mode={mode!r}; expected one of {sorted(TRIAL_SEED_MODES)}"
        )

    if mode == "stream":
        model_seed = None
    elif mode == "split":
        model_seed = split_seed
    else:
        model_seed = int(base_seed) + int(trial_index)
    return split_seed, model_seed, mode


def _resolve_shot_setting(data_cfg: dict[str, Any], experiment_cfg: dict[str, Any], target_dataset: str) -> tuple[int, float | None, str]:
    shots = int(data_cfg.get("shots", experiment_cfg.get("shots", 5)))
    shot_ratio_raw = data_cfg.get("shot_ratio", experiment_cfg.get("shot_ratio"))
    shot_ratio = None if shot_ratio_raw is None else float(shot_ratio_raw)
    if not bool(data_cfg.get("auto_shot_by_dataset", False)):
        return shots, shot_ratio, "explicit_ratio" if shot_ratio is not None else "explicit_shots"

    hetero_names = {
        str(name).lower()
        for name in data_cfg.get(
            "heterophily_datasets",
            [
                "Actor",
                "Squirrel",
                "Chameleon",
                "Texas",
                "Cornell",
                "Wisconsin",
                "Minesweeper",
                "Roman-empire",
                "RomanEmpire",
                "roman_empire",
                "amazon_ratings",
                "AmazonRatings",
                "Amazon-Ratings",
                "Tolokers",
                "Questions",
            ],
        )
    }
    if str(target_dataset).lower() in hetero_names:
        return shots, float(data_cfg.get("heterophily_shot_ratio", 0.10)), "heterophily_10pct"
    return int(data_cfg.get("homophily_shots", shots)), None, "homophily_5shot"


def _config_for_variant(config: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in PROMPT_GRAPH_VARIANTS:
        raise ValueError(f"Unsupported prompt_variant={variant!r}")
    out = copy.deepcopy(config)
    out.setdefault("experiment", {})["prompt_variant"] = variant
    training = out.setdefault("training", {})
    explicit_training_keys = set(training.keys())
    prompt_graph = out.setdefault("prompt_graph", {})
    adapter_variants = {
        "p14_freeze_prompt_adapter",
        "p15_hetero_prompt_adapter",
        "p16_support_prompt_adapter",
        "p17_selective_support_adapter",
        "p18_episode_consistency_adapter",
        "p20_class_conditioned_pattern_prompt_router",
        "p20_utility_supervised_pattern_prompt_router",
        "p21_lite_adaptive_filter",
        "p21_v2_hetero_filter",
        "p22_class_pattern_enrichment_bank",
        "p22_reliability_calibrated_basis_bank",
        "p22_v031_conservative_reliability_basis_bank",
        "p22_v04_minimal_transition_basis",
    }
    if variant == "noprompt" or variant in adapter_variants:
        prompt_graph["enabled"] = False
    else:
        prompt_graph["enabled"] = True
    prompt_aware = out.setdefault("prompt_aware", {})
    prompt_aware["enabled"] = variant.startswith("p2_") or variant.startswith("p5_") or variant.startswith("p13_")
    prompt_adapter = out.setdefault("prompt_adapter", {})
    explicit_prompt_adapter_keys = set(prompt_adapter.keys())
    if variant == "p23_selective_discrete_feature_prompting":
        prompt_graph["enabled"] = True
        prompt_aware["enabled"] = False
        prompt_adapter["enabled"] = False
        prompt_graph.setdefault("module_type", "selective_discrete_feature_prompt")
        prompt_graph.setdefault("static_graph", True)
        prompt_graph.setdefault("tokenizer", "binary_nonzero")
        prompt_graph.setdefault("topk_feature_dims", 32)
        prompt_graph.setdefault("min_df", 2)
        prompt_graph.setdefault("max_df_ratio", 0.20)
        prompt_graph.setdefault("rho", 0.15)
        prompt_graph.setdefault("include_train_in_pool", True)
        prompt_graph.setdefault("topk_feature_prompt_per_node", 3)
        prompt_graph.setdefault("direction", "prompt_to_node")
        prompt_graph.setdefault("feature_node_init", "avg_z_detached")
        prompt_graph.setdefault("feature_edge_weight", 0.10)
        prompt_graph.setdefault("difficulty_edge_weight", 0.50)
        prompt_graph.setdefault("utility_pool_structural_weight", 0.40)
        prompt_graph.setdefault("utility_pool_uncertainty_weight", 0.30)
        prompt_graph.setdefault("utility_pool_disagreement_weight", 0.30)
        prompt_graph.setdefault("use_idf", True)
        prompt_graph.setdefault("use_feature_reliability", True)
        prompt_graph.setdefault("lambda_edge_l1", 0.0)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_role_diversity", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_view_prior", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_correction", 0.0)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help_query", 0.0)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_utility_receive_gate", 0.0)
        prompt_graph.setdefault("lambda_utility_receive_gate_query", 0.0)
        prompt_graph.setdefault("lambda_receive_gate_budget", 0.0)
        prompt_graph.setdefault("lambda_query_proto_alignment", 0.0)
        prompt_graph.setdefault("lambda_edge_utility_supervision", 0.0)
        prompt_graph.setdefault("lambda_correction_alignment", 0.0)
        prompt_graph.setdefault("lambda_correction_anti_harm", 0.0)
        training.setdefault("freeze_base_model", False)
        training.setdefault("train_prompt_graph_module", False)
        training.setdefault("train_prompt_adapter", False)
        training.setdefault("train_prompt_aware_module", False)
        training.setdefault("epochs", 250)
        training.setdefault("early_stop_min_epochs", 80)
        training.setdefault("early_stop_patience", 60)
        training.setdefault("output_dir", "outputs/gp2f_prompt_p23_selective_discrete_feature_prompting")
    if variant == "p23_graphite_adapter":
        prompt_graph["enabled"] = True
        prompt_aware["enabled"] = False
        prompt_adapter["enabled"] = False
        prompt_graph.setdefault("module_type", "graphite_style_prompt_graph")
        prompt_graph.setdefault("static_graph", True)
        prompt_graph.setdefault(
            "feature_tokenizer",
            {"mode": "binary_nonzero", "binary_threshold": 0.0, "binary_topk": 0, "topk": 0},
        )
        prompt_graph.setdefault("feature_filter", {"min_df_global": 2, "max_df_global_ratio": 0.80})
        prompt_graph.setdefault("use_feature_filter", True)
        prompt_graph.setdefault("feature_edge_weight", 1.0)
        prompt_graph.setdefault("original_edge_weight", 1.0)
        prompt_graph.setdefault("learn_feature_edge_weight", True)
        prompt_graph.setdefault("detach_feature_prompt", True)
        prompt_graph.setdefault("edge_scale_warmup_epochs", 0)
        prompt_graph.setdefault("edge_scale_warmup_start", 1.0)
        prompt_graph.setdefault("lambda_edge_l1", 0.0)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_p23_feature_harm", 0.05)
        prompt_graph.setdefault("lambda_p23_feature_gate_supervision", 0.0)
        prompt_graph.setdefault("p23_feature_harm_margin", 0.02)
        prompt_graph.setdefault("p23_feature_harm_temperature", 0.15)
        prompt_graph.setdefault("p23_feature_harm_quantile", 0.70)
        prompt_graph.setdefault("p23_feature_help_quantile", 0.70)
        prompt_graph.setdefault("p23_feature_min_signal", 0.005)
        prompt_graph.setdefault("p23_feature_help_weight", 0.25)
        prompt_graph.setdefault("p23_feature_harm_warmup_epochs", 5)
        prompt_graph.setdefault("p23_feature_harm_detach_delta", True)
        prompt_graph.setdefault("p23_feature_harm_class_balanced", True)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        training.setdefault("freeze_base_model", False)
        training.setdefault("freeze_input_aligner_for_p23", False)
        training.setdefault("train_prompt_graph_module", True)
        training.setdefault("train_prompt_adapter", False)
        training.setdefault("train_prompt_aware_module", False)
        training.setdefault("epochs", 160)
        training.setdefault("early_stop_min_epochs", 35)
        training.setdefault("early_stop_patience", 35)
        training.setdefault("output_dir", "outputs/gp2f_prompt_p23_graphite_adapter")
    if variant in {
        "p23_selective_graphite_adapter",
        "p23_projector_synchronized_graphite_adapter",
        "p24_prompt_edge_denoising",
    }:
        prompt_graph["enabled"] = True
        prompt_aware["enabled"] = False
        prompt_adapter["enabled"] = False
        module_type = "class_aware_selective_graphite_prompt_graph"
        if variant == "p24_prompt_edge_denoising":
            module_type = "p24_prompt_edge_denoising_adapter"
        elif variant == "p23_projector_synchronized_graphite_adapter":
            module_type = "p23_projector_synchronized_prompt_graph"
        prompt_graph.setdefault("module_type", module_type)
        prompt_graph.setdefault("static_graph", True)
        prompt_graph.setdefault(
            "feature_tokenizer",
            {"mode": "topk_activation", "topk": 16},
        )
        prompt_graph.setdefault(
            "feature_filter",
            {
                "min_df_global": 2,
                "max_df_global_ratio": 0.95,
                "min_label_count": 2,
                "min_purity": 0.35,
                "max_entropy": 0.95,
                "max_feature_nodes": 512,
            },
        )
        prompt_graph.setdefault("feature_edge_weight", 1.0)
        prompt_graph.setdefault("feature_edge_min_scale", 0.05)
        prompt_graph.setdefault("feature_gate_floor", 0.20)
        prompt_graph.setdefault("original_edge_weight", 1.0)
        prompt_graph.setdefault("learn_feature_edge_weight", True)
        prompt_graph.setdefault("use_feature_gate", True)
        prompt_graph.setdefault("use_node_feature_edge_gate", True)
        prompt_graph.setdefault("node_feature_edge_gate_strength", 0.25)
        prompt_graph.setdefault("anti_hub_power", 0.5)
        prompt_graph.setdefault("label_node_weight", 4.0)
        prompt_graph.setdefault("unlabeled_node_weight", 0.25)
        prompt_graph.setdefault("offclass_label_weight", 0.05)
        prompt_graph.setdefault(
            "static_pool",
            {
                "enabled": False,
                "mode": "hard",
                "strategy": "feature_label_ambiguity",
                "ratio": 0.15,
                "core_ratio": 0.15,
                "expand_ratio": 0.35,
                "core_scale": 1.2,
                "expand_scale": 1.1,
                "non_pool_scale": 1.0,
                "apply_to": "prompt_to_node",
                "feature_topk": 16,
            },
        )
        prompt_graph.setdefault("detach_feature_prompt", True)
        prompt_graph.setdefault("edge_scale_warmup_epochs", 0)
        prompt_graph.setdefault("edge_scale_warmup_start", 1.0)
        prompt_graph.setdefault("lambda_edge_l1", 0.0)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault(
            "prompt_residual",
            {
                "enabled": False,
                "rank": 4,
                "hidden_dim": 32,
                "dropout": 0.10,
                "gate_init_bias": -3.5,
                "max_delta_ratio": 0.05,
                "delta_scale": 1.0,
                "basis_init_std": 0.01,
                "use_reliability_prior": True,
                "reliability_prior_strength": 1.0,
            },
        )
        prompt_graph.setdefault("lambda_p23_prompt_residual_delta_norm", 0.0)
        prompt_graph.setdefault("lambda_p23_prompt_residual_delta_align", 0.0)
        prompt_graph.setdefault("lambda_p23_prompt_residual_gate_budget", 0.0)
        training.setdefault("freeze_base_model", False)
        training.setdefault("freeze_input_aligner_for_p23", False)
        training.setdefault("train_prompt_graph_module", True)
        training.setdefault("train_prompt_adapter", False)
        training.setdefault("train_prompt_aware_module", False)
        training.setdefault("class_balanced_ce", True)
        training.setdefault("class_weight_mode", "inverse_sqrt")
        training.setdefault("epochs", 160)
        training.setdefault("early_stop_metric", "val_macro_f1")
        training.setdefault("early_stop_min_epochs", 35)
        training.setdefault("early_stop_patience", 35)
        if variant == "p24_prompt_edge_denoising":
            prompt_graph.setdefault(
                "prompt_edge_denoising",
                {
                    "enabled": True,
                    "gate_init": 0.95,
                    "gate_floor": 0.20,
                    "hidden_dim": 16,
                    "dropout": 0.10,
                    "detach_node_repr": True,
                    "detach_prompt_repr": True,
                    "apply_to": "prompt_to_node",
                    "allow_new_edges": False,
                    "hub_quantile": 0.80,
                    "homophily_keep_weight": 0.25,
                },
            )
            prompt_graph.setdefault("lambda_p24_keep_static", 0.01)
            prompt_graph.setdefault("lambda_p24_hub_suppression", 0.01)
            prompt_graph.setdefault("lambda_p24_prompt_homophily", 0.05)
            prompt_graph.setdefault("lambda_p24_edge_budget", 0.0)
            prompt_graph.setdefault("lambda_p24_candidate_cls", 0.0)
            prompt_graph.setdefault("lambda_p24_candidate_anti_harm", 0.0)
            prompt_graph.setdefault("lambda_p24_relation_prior", 0.0)
            prompt_graph.setdefault("lambda_p24_relation_preservation", 0.0)
            prompt_graph.setdefault("lambda_p24_relation_homophily_collapse", 0.0)
            prompt_graph.setdefault("lambda_p24_relation_uncertain_keep", 0.0)
            prompt_graph.setdefault("p24_candidate_anti_harm_margin", 0.001)
            prompt_graph.setdefault("lambda_p24_route_utility", 0.0)
            prompt_graph.setdefault("p24_route_utility_warmup_epochs", 0)
            prompt_graph.setdefault("p24_route_positive_margin", 0.005)
            prompt_graph.setdefault("p24_route_negative_margin", 0.005)
            prompt_graph.setdefault("p24_route_class_balanced", True)
            prompt_graph.setdefault("p24_route_supervise_no_prompt", True)
            prompt_graph.setdefault("p24_route_target_mode", "hard")
            prompt_graph.setdefault("p24_route_gain_scale", 0.01)
            prompt_graph.setdefault(
                "p24_class_coverage_guard",
                {
                    "enabled": False,
                    "temperature": 1.0,
                    "min_entropy_ratio": 0.75,
                    "max_dominant_ratio": 0.70,
                    "class_separation_weight": 1.0,
                    "marginal_balance_weight": 0.25,
                    "warmup_epochs": 0,
                    "loss_mask": "all_train",
                },
            )
            prompt_graph.setdefault("lambda_p24_class_coverage_guard", 0.0)
            training.setdefault(
                "p24_collapse_recovery",
                {
                    "train_input_aligner": False,
                    "train_adapters": False,
                    "train_fusion_alpha": False,
                },
            )
        training.setdefault(
            "output_dir",
            "outputs/gp2f_prompt_p24_prompt_edge_denoising"
            if variant == "p24_prompt_edge_denoising"
            else "outputs/gp2f_prompt_p23_selective_graphite_adapter",
        )
    if variant in {"p23_v0_1", "p23_v0_2_message_passing"}:
        prompt_graph["enabled"] = True
        prompt_aware["enabled"] = False
        prompt_adapter["enabled"] = False
        prompt_graph.setdefault("module_type", "p23_v0_1")
        prompt_graph.setdefault(
            "feature_tokenizer",
            {"mode": "binary_nonzero", "topk": 8, "binary_threshold": 0.0, "binary_topk": 16},
        )
        prompt_graph.setdefault(
            "risk",
            {
                "lambda_homo": [0.35, 0.25, 0.20, 0.10, 0.10],
                "lambda_hetero": [0.25, 0.20, 0.20, 0.15, 0.20],
            },
        )
        prompt_graph.setdefault("graph_risk_weights", {"ego_neighbor": 0.40, "onehop_twohop": 0.30, "neighbor_variance": 0.30})
        prompt_graph.setdefault("pool", {"adaptive_ratio": True, "rho_min": 0.05, "rho_max": 0.20, "rho_power": 1.0, "force_train_nodes": True})
        prompt_graph.setdefault("feature_filter", {"min_df_pool": 2, "max_df_pool_ratio": 0.50, "max_df_global_ratio": 0.80})
        prompt_graph.setdefault(
            "prompt_graph",
            {"init_scope": "global_same_feature", "edge_type_prompt_to_node": 2, "topk_feature_prompt_per_node": 4},
        )
        prompt_graph.setdefault(
            "receiver",
            {"max_update_norm": 0.03, "prompt_dropout": 0.05, "init_gate_bias": -4.5}
            if variant == "p23_v0_2_message_passing"
            else {"max_update_norm": 0.02, "prompt_dropout": 0.10, "init_gate_bias": -5.0},
        )
        prompt_graph.setdefault("prompt_token", {"learn_delta": False, "delta_init_std": 0.0})
        prompt_graph.setdefault(
            "node_to_prompt",
            {"enabled": True, "scale": 0.5} if variant == "p23_v0_2_message_passing" else {"enabled": False, "scale": 1.0},
        )
        prompt_graph.setdefault("edge_scale_warmup_epochs", 20)
        prompt_graph.setdefault("edge_scale_warmup_start", 0.0)
        prompt_graph.setdefault("lambda_edge_l1", 0.0)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_role_diversity", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_view_prior", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_correction", 0.0)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help_query", 0.0)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_utility_receive_gate", 0.0)
        prompt_graph.setdefault("lambda_utility_receive_gate_query", 0.0)
        prompt_graph.setdefault("lambda_receive_gate_budget", 0.0)
        prompt_graph.setdefault("lambda_query_proto_alignment", 0.0)
        prompt_graph.setdefault("lambda_edge_utility_supervision", 0.0)
        prompt_graph.setdefault("lambda_correction_alignment", 0.0)
        prompt_graph.setdefault("lambda_correction_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_p23_norm", 0.01)
        prompt_graph.setdefault("lambda_p23_hub_budget", 0.01)
        prompt_graph.setdefault("lambda_p23_node_gate_utility", 0.0)
        prompt_graph.setdefault("p23_node_gate_positive_margin", 0.001)
        prompt_graph.setdefault("p23_node_gate_negative_margin", -0.001)
        prompt_graph.setdefault("p23_node_gate_utility_warmup_epochs", 20)
        prompt_graph.setdefault("p23_node_gate_utility_class_balanced", True)
        prompt_graph.setdefault("p23_log_topk_feature_prompt_delta", 20)
        training.setdefault("freeze_base_model", False)
        training.setdefault("freeze_input_aligner_for_p23", False)
        training.setdefault("train_prompt_graph_module", True)
        training.setdefault("train_prompt_adapter", False)
        training.setdefault("train_prompt_aware_module", False)
        training.setdefault("epochs", 250)
        training.setdefault("early_stop_min_epochs", 80)
        training.setdefault("early_stop_patience", 60)
        training.setdefault("output_dir", "outputs/gp2f_prompt_p23_v0_1")
        if variant == "p23_v0_2_message_passing":
            prompt_graph["prompt_token"] = {**dict(prompt_graph.get("prompt_token", {})), "learn_delta": True}
            prompt_graph["node_to_prompt"] = {
                **dict(prompt_graph.get("node_to_prompt", {})),
                "enabled": True,
            }
            prompt_graph["receiver"] = {
                **dict(prompt_graph.get("receiver", {})),
            }
            prompt_graph["lambda_p23_hub_budget"] = float(prompt_graph.get("lambda_p23_hub_budget", 0.005))
            training["output_dir"] = str(training.get("output_dir", "outputs/gp2f_prompt_p23_v0_2_message_passing"))
    if variant in adapter_variants:
        prompt_adapter["enabled"] = True
        prompt_aware["enabled"] = False
        prompt_graph["enabled"] = False
        prompt_graph.setdefault("support_query_split", {"enabled": True, "support_ratio": 0.55, "min_query_per_class": 2, "resample_each_epoch": True})
        prompt_graph.setdefault("support_only_prompt_graph", False)
        prompt_graph.setdefault("lambda_edge_l1", 0.0)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_role_diversity", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_view_prior", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_correction", 0.0)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help_query", 0.0)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_utility_receive_gate", 0.0)
        prompt_graph.setdefault("lambda_utility_receive_gate_query", 0.0)
        prompt_graph.setdefault("lambda_receive_gate_budget", 0.0)
        prompt_graph.setdefault("lambda_query_proto_alignment", 0.0)
        prompt_graph.setdefault("lambda_edge_utility_supervision", 0.0)
        prompt_graph.setdefault("lambda_correction_alignment", 0.0)
        prompt_graph.setdefault("lambda_correction_anti_harm", 0.0)
        prompt_adapter.setdefault("context_base", "z_detached")
        prompt_adapter.setdefault("use_ego", True)
        prompt_adapter.setdefault("use_low_frequency", True)
        prompt_adapter.setdefault("use_two_step", True)
        prompt_adapter.setdefault("use_high_frequency", True)
        prompt_adapter.setdefault("use_role_features", True)
        prompt_adapter.setdefault("hidden_dim", 128)
        prompt_adapter.setdefault("dropout", 0.2)
        prompt_adapter.setdefault("zero_init_delta", True)
        prompt_adapter.setdefault("gate_init", 0.05)
        prompt_adapter.setdefault("max_update_norm", 0.05)
        prompt_adapter.setdefault("gate_budget", 0.35)
        prompt_adapter.setdefault("message_scale", 1.0)
        training = out.setdefault("training", {})
        training.setdefault("lambda_prompt_adapter_update_norm", 0.01)
        training.setdefault("lambda_prompt_adapter_gate_budget", 0.01)
        training.setdefault("lambda_prompt_adapter_message_help", 0.0)
        training.setdefault("prompt_adapter_message_help_margin", 0.0)
        training.setdefault("prompt_adapter_message_help_class_balanced", True)
        training.setdefault("prompt_adapter_update_mask", "all")
        training.setdefault("prompt_adapter_loss_mask", "query")
        if variant == "p14_freeze_prompt_adapter":
            training["freeze_base_model"] = True
            training["train_prompt_adapter"] = True
        elif variant in {
            "p16_support_prompt_adapter",
            "p17_selective_support_adapter",
            "p18_episode_consistency_adapter",
        }:
            training.setdefault("freeze_base_model", False)
            training["train_prompt_adapter"] = True
            if float(training.get("lambda_prompt_adapter_message_help", 0.0)) <= 0.0:
                training["lambda_prompt_adapter_message_help"] = 0.10
            if float(training.get("prompt_adapter_message_help_margin", 0.0)) <= 0.0:
                training["prompt_adapter_message_help_margin"] = 0.005
            prompt_adapter.setdefault("use_support_context", True)
            prompt_adapter.setdefault("support_tau", 0.5)
            prompt_adapter.setdefault("use_support_class_similarity", True)
            prompt_adapter.setdefault("use_support_proto_residual", True)
            prompt_adapter.setdefault("use_support_high_residual", True)
            prompt_adapter.setdefault("gate_init", 0.10)
            prompt_adapter.setdefault("max_update_norm", 0.08)
            if variant in {"p17_selective_support_adapter", "p18_episode_consistency_adapter"}:
                prompt_adapter.setdefault("support_context_mode", "topk_attention")
                prompt_adapter.setdefault("support_topk", 8)
                prompt_adapter.setdefault("support_tau", 0.25)
                prompt_adapter.setdefault("use_support_uncertainty_features", True)
                prompt_adapter.setdefault("use_support_reliability_gate", True)
                prompt_adapter.setdefault("support_reliability_strength", 0.5)
                prompt_adapter.setdefault("support_reliability_floor", 0.15)
                prompt_adapter.setdefault("support_reliability_margin_weight", 0.7)
                prompt_adapter.setdefault("gate_budget", 0.55)
                training.setdefault("lambda_prompt_adapter_utility_gate", 0.05)
                training.setdefault("prompt_adapter_utility_gate_temperature", 0.02)
                training.setdefault("prompt_adapter_utility_gate_margin", 0.001)
                training.setdefault("prompt_adapter_utility_gate_class_balanced", True)
                training.setdefault("prompt_adapter_utility_gate_source", "effective_gate")
                training["lambda_prompt_adapter_gate_budget"] = max(
                    float(training.get("lambda_prompt_adapter_gate_budget", 0.0)),
                    0.05,
                )
                if variant == "p18_episode_consistency_adapter":
                    training.setdefault("prompt_adapter_episode_count_per_epoch", 3)
                    training.setdefault("lambda_prompt_adapter_gate_consistency", 0.02)
                    training.setdefault("lambda_prompt_adapter_delta_consistency", 0.01)
        elif variant in {
            "p20_class_conditioned_pattern_prompt_router",
            "p20_utility_supervised_pattern_prompt_router",
            "p21_lite_adaptive_filter",
            "p21_v2_hetero_filter",
            "p22_class_pattern_enrichment_bank",
            "p22_reliability_calibrated_basis_bank",
            "p22_v031_conservative_reliability_basis_bank",
            "p22_v04_minimal_transition_basis",
        }:
            training.setdefault("freeze_base_model", False)
            training.setdefault("log_every", 5)
            training["train_prompt_adapter"] = True
            p22_variants = {
                "p22_class_pattern_enrichment_bank",
                "p22_reliability_calibrated_basis_bank",
                "p22_v031_conservative_reliability_basis_bank",
                "p22_v04_minimal_transition_basis",
            }
            if variant in p22_variants:
                prompt_adapter.setdefault("module_type", "p22_class_pattern_enrichment_bank")
                prompt_adapter.setdefault("num_patterns", 8)
                prompt_adapter.setdefault("pattern_dim", 64)
                prompt_adapter.setdefault("pattern_encoder_hidden_dim", 128)
                prompt_adapter.setdefault("dropout", 0.0)
                prompt_adapter.setdefault("pattern_init", "kmeans_all_signature")
                prompt_adapter.setdefault("pattern_init_use_labels", False)
                prompt_adapter.setdefault("detach_class_pattern", True)
                prompt_adapter.setdefault("use_basis_evidence", True)
                default_basis_types = [
                    "ego_logprob",
                    "onehop_logprob",
                    "twohop_logprob",
                    "highpass_ego_onehop",
                    "highpass_onehop_twohop",
                ]
                if variant == "p22_class_pattern_enrichment_bank":
                    default_basis_types = [*default_basis_types, "class_transition"]
                elif variant == "p22_v04_minimal_transition_basis":
                    default_basis_types = [
                        *default_basis_types,
                        "onehop_transition_logprob",
                        "highpass_ego_transition_onehop",
                    ]
                prompt_adapter.setdefault("basis_types", default_basis_types)
                prompt_adapter.setdefault("num_bases", len(prompt_adapter["basis_types"]))
                prompt_adapter.setdefault("enrichment_weight", 0.5 if variant == "p22_class_pattern_enrichment_bank" else 0.0)
                prompt_adapter.setdefault("basis_weight_scale", 1.0)
                prompt_adapter.setdefault("use_class_transition", variant == "p22_class_pattern_enrichment_bank")
                prompt_adapter.setdefault("use_transition_basis", variant == "p22_v04_minimal_transition_basis")
                prompt_adapter.setdefault("transition_matrix_mode", "support_uniform")
                prompt_adapter.setdefault("transition_prior", "uniform")
                prompt_adapter.setdefault("transition_alpha", 5.0)
                prompt_adapter.setdefault("transition_lambda_support", 1.0)
                prompt_adapter.setdefault("transition_lambda_pseudo", 0.0)
                prompt_adapter.setdefault("use_c2_matrix", False)
                prompt_adapter.setdefault("use_transition_ema", False)
                prompt_adapter.setdefault("log_transition_matrix_stats", variant == "p22_v04_minimal_transition_basis")
                prompt_adapter.setdefault("log_single_basis_delta_ce", variant == "p22_v04_minimal_transition_basis")
                prompt_adapter.setdefault("basis_delta_scale_grid", [0.01, 0.03, 0.05, 0.10, 0.20])
                prompt_adapter.setdefault("class_transition_smoothing", 0.5)
                prompt_adapter.setdefault("class_pattern_smoothing", 0.5)
                prompt_adapter.setdefault("use_basis_teacher", True)
                prompt_adapter.setdefault("basis_teacher_temperature", 0.5)
                prompt_adapter.setdefault("basis_teacher_scale", 1.0)
                prompt_adapter.setdefault("basis_usage_entropy_floor", 0.60)
                prompt_adapter.setdefault("normalize_basis_evidence", variant != "p22_class_pattern_enrichment_bank")
                prompt_adapter.setdefault("basis_evidence_std_floor", 0.5)
                prompt_adapter.setdefault(
                    "pattern_basis_init",
                    "cyclic_anchor" if variant in {
                        "p22_v031_conservative_reliability_basis_bank",
                        "p22_v04_minimal_transition_basis",
                    } else "zeros",
                )
                prompt_adapter.setdefault("pattern_basis_anchor_logit", 2.0)
                prompt_adapter.setdefault("pattern_basis_off_logit", -2.0)
                prompt_adapter.setdefault("pattern_temperature_init", 0.70)
                prompt_adapter.setdefault(
                    "pattern_temperature_final",
                    0.40 if variant in {
                        "p22_v031_conservative_reliability_basis_bank",
                        "p22_v04_minimal_transition_basis",
                    } else (0.35 if variant == "p22_reliability_calibrated_basis_bank" else 0.30),
                )
                prompt_adapter.setdefault(
                    "pattern_temperature_warmdown_epochs",
                    80 if variant in {
                        "p22_v031_conservative_reliability_basis_bank",
                        "p22_v04_minimal_transition_basis",
                    } else (60 if variant == "p22_reliability_calibrated_basis_bank" else 80),
                )
                prompt_adapter.setdefault(
                    "pattern_scale_init",
                    0.01 if variant in {
                        "p22_v031_conservative_reliability_basis_bank",
                        "p22_v04_minimal_transition_basis",
                    } else (0.03 if variant == "p22_reliability_calibrated_basis_bank" else 0.05),
                )
                prompt_adapter.setdefault(
                    "pattern_scale_max",
                    0.30 if variant in {
                        "p22_v031_conservative_reliability_basis_bank",
                        "p22_v04_minimal_transition_basis",
                    } else (0.60 if variant == "p22_reliability_calibrated_basis_bank" else 1.0),
                )
                prompt_adapter.setdefault("pattern_scale_warmup_epochs", 120 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else 80)
                prompt_adapter.setdefault("use_reliability_gate", variant != "p22_class_pattern_enrichment_bank")
                prompt_adapter.setdefault("reliability_gate_hidden_dim", 32)
                prompt_adapter.setdefault("reliability_gate_init_bias", -2.5 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else -2.0)
                prompt_adapter.setdefault("reliability_gate_min", 0.0)
                prompt_adapter.setdefault("reliability_gate_max", 1.0)
                prompt_adapter.setdefault("reliability_gate_detach_input", True)
                prompt_adapter.setdefault("use_pattern_gate", False)
                prompt_adapter["use_candidate_pool"] = False
                prompt_adapter["candidate_pool_ratio"] = 1.0
                training["freeze_base_model"] = True
                training["train_prompt_adapter"] = True
                training["prompt_adapter_update_mask"] = "all"
                training["prompt_adapter_loss_mask"] = "query"
                training.setdefault("epochs", 250 if variant != "p22_class_pattern_enrichment_bank" else 300)
                training.setdefault("early_stop_min_epochs", 80 if variant != "p22_class_pattern_enrichment_bank" else 120)
                training.setdefault("early_stop_patience", 60 if variant != "p22_class_pattern_enrichment_bank" else 80)
                training.setdefault("prompt_weight_decay", 0.0)
                training.setdefault("p22_stage1_epochs", 30 if variant != "p22_class_pattern_enrichment_bank" else 80)
                training.setdefault("p22_stage1_pattern_only", True)
                training.setdefault("p22_support_source", "episode" if variant != "p22_class_pattern_enrichment_bank" else "full_train")
                training.setdefault("p22_loss_source", "episode" if variant != "p22_class_pattern_enrichment_bank" else "train")
                training.setdefault("p22_crossfit_enabled", variant != "p22_class_pattern_enrichment_bank")
                training.setdefault("p22_crossfit_num_folds", 5)
                training.setdefault("p22_crossfit_resample_each_epoch", True)
                training.setdefault("p22_crossfit_class_balanced", True)
                training.setdefault("p22_freeze_pattern_after_epoch", 180 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else (200 if variant == "p22_reliability_calibrated_basis_bank" else 0))
                training["lambda_prompt_adapter_update_norm"] = 0.0
                training["lambda_prompt_adapter_gate_budget"] = 0.0
                training["lambda_prompt_adapter_message_help"] = 0.0
                training["lambda_prompt_adapter_utility_gate"] = 0.0
                training["lambda_prompt_router_pattern_balance"] = 0.0
                training["lambda_prompt_router_pattern_supervision"] = 0.0
                training["lambda_prompt_router_pattern_utility"] = 0.0
                training["lambda_prompt_router_class_pattern_reliability"] = 0.0
                training.setdefault("lambda_p22_pattern_only", 0.2 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else (0.3 if variant == "p22_reliability_calibrated_basis_bank" else 1.0))
                training.setdefault("lambda_p22_pattern_reg", 0.005)
                training.setdefault("lambda_p22_basis_teacher", 0.2 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else (0.3 if variant == "p22_reliability_calibrated_basis_bank" else 0.5))
                training.setdefault("lambda_p22_basis_usage", 0.01)
                training.setdefault("lambda_p22_deployment", 1.0 if variant != "p22_class_pattern_enrichment_bank" else 0.0)
                training.setdefault("lambda_p22_gate", 0.5 if variant != "p22_class_pattern_enrichment_bank" else 0.0)
                training.setdefault("lambda_p22_anti_harm", 1.0 if variant != "p22_class_pattern_enrichment_bank" else 0.0)
                training.setdefault("lambda_p22_gain_reward", 0.05 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else (0.2 if variant == "p22_reliability_calibrated_basis_bank" else 0.0))
                training.setdefault("p22_anti_harm_margin", 0.0)
                training.setdefault("p22_gate_margin", 0.0005)
                training.setdefault("p22_gate_target_mode", "tri_state" if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else "binary")
                training.setdefault("p22_gate_positive_margin", 0.010)
                training.setdefault("p22_gate_negative_margin", -0.005)
                training.setdefault("p22_gate_ignore_neutral", True)
                training.setdefault("p22_gate_target_source", "ungated_delta")
                training.setdefault("p22_gate_use_crossfit_stability", variant == "p22_v031_conservative_reliability_basis_bank")
                training.setdefault("p22_gate_helpful_stability_threshold", 0.70)
                training.setdefault("p22_gate_harmful_stability_threshold", 0.50)
                training.setdefault("p22_gate_stability_min_seen", 2)
                training.setdefault("lambda_p22_gate_budget", 0.2 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else 0.0)
                training.setdefault("p22_gate_budget_max", 0.35)
                training.setdefault("p22_gate_budget_warmup_epochs", 30)
                training.setdefault("lambda_p22_gate_harm", 0.5 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else 0.0)
                training.setdefault("p22_gate_harm_negative_margin", -0.005)
                training.setdefault("lambda_p22_scale_reg", 0.01 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else 0.0)
                training.setdefault("p22_gain_reward_cap", 0.01 if variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                } else 0.02)
                training.setdefault("p22_safe_checkpoint_enabled", variant in {
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                })
                training.setdefault("p22_safe_checkpoint_metric", "val_acc_plus_val_delta_ce")
                training.setdefault("p22_safe_checkpoint_min_val_delta_ce", -0.0005)
                training.setdefault("p22_safe_checkpoint_delta_weight", 0.5)
                training.setdefault("p22_safe_checkpoint_delta_cap", 0.01)
                training.setdefault("prompt_adapter_episode_count_per_epoch", 5)
                training.setdefault("lambda_prompt_adapter_gate_consistency", 0.0)
                training.setdefault("lambda_prompt_adapter_delta_consistency", 0.0)
            elif variant in {"p21_lite_adaptive_filter", "p21_v2_hetero_filter"}:
                prompt_adapter.setdefault(
                    "module_type",
                    "p21_v2_hetero_filter" if variant == "p21_v2_hetero_filter" else "p21_lite_adaptive_filter",
                )
                training["freeze_base_model"] = True
                training["early_stop_metric"] = str(training.get("early_stop_metric", "val_acc"))
            elif variant == "p20_utility_supervised_pattern_prompt_router":
                prompt_adapter.setdefault("module_type", "utility_supervised_pattern_router")
                training["freeze_base_model"] = True
                training["early_stop_metric"] = "train_loss"
            else:
                prompt_adapter.setdefault("module_type", "class_conditioned_pattern_router")
            prompt_adapter.setdefault("num_patterns", 6)
            prompt_adapter.setdefault("low_rank_dim", 8)
            prompt_adapter.setdefault("dropout", 0.1)
            # Non-zero message at init breaks routing symmetry so q_i gets gradient.
            prompt_adapter.setdefault("zero_init_message", False)
            prompt_adapter.setdefault("message_init_scale", 0.1)
            prompt_adapter.setdefault("easy_init", 0.30)
            prompt_adapter.setdefault("pattern_reject_init_prob", 0.05)
            prompt_adapter.setdefault("use_frozen_disagreement", True)
            # Conservative gate so the prompt does not open up too early.
            prompt_adapter.setdefault("gate_init", 0.05)
            prompt_adapter.setdefault("max_update_norm", 0.04)
            prompt_adapter.setdefault("gate_budget", 0.25)
            prompt_adapter.setdefault("message_scale", 1.0)
            prompt_adapter.setdefault("use_candidate_pool", True)
            prompt_adapter.setdefault("candidate_pool_strategy", "utility_structural")
            prompt_adapter.setdefault("candidate_pool_ratio", 0.30)
            prompt_adapter.setdefault("candidate_pool_include_train", True)
            prompt_adapter.setdefault("candidate_pool_structural_weight", 0.35)
            prompt_adapter.setdefault("candidate_pool_uncertainty_weight", 0.35)
            prompt_adapter.setdefault("candidate_pool_disagreement_weight", 0.30)
            # The router does not use support-context features.
            prompt_adapter.setdefault("use_support_context", False)
            # L_anti_harm reuses the message-help (relu(margin - delta_CE)) loss.
            if float(training.get("lambda_prompt_adapter_message_help", 0.0)) <= 0.0:
                training["lambda_prompt_adapter_message_help"] = 0.10
            if float(training.get("prompt_adapter_message_help_margin", 0.0)) <= 0.0:
                training["prompt_adapter_message_help_margin"] = 0.005
            training.setdefault("prompt_adapter_message_help_class_balanced", True)
            training.setdefault("prompt_adapter_message_help_anti_harm_weight", 0.5)
            training.setdefault("prompt_adapter_message_help_anti_harm_margin", 0.0)
            training.setdefault("lambda_prompt_adapter_utility_gate", 0.10)
            training.setdefault("prompt_adapter_utility_gate_temperature", 0.02)
            training.setdefault("prompt_adapter_utility_gate_margin", 1e-5)
            training.setdefault("prompt_adapter_utility_gate_class_balanced", True)
            training.setdefault("prompt_adapter_utility_gate_source", "effective_gate")
            training["lambda_prompt_adapter_update_norm"] = max(
                float(training.get("lambda_prompt_adapter_update_norm", 0.0)),
                0.02,
            )
            training["lambda_prompt_adapter_gate_budget"] = max(
                float(training.get("lambda_prompt_adapter_gate_budget", 0.0)),
                0.10,
            )
            if str(training.get("prompt_adapter_update_mask", "all")) == "all":
                training["prompt_adapter_update_mask"] = "candidate_pool"
            # L_pattern_balance: anti-collapse entropy-floor hinge (does NOT reward
            # uniform). Default off (0.0) so bare routing behaviour is observable;
            # the floor only activates when lambda > 0.
            training.setdefault("lambda_prompt_router_pattern_balance", 0.02)
            training.setdefault("prompt_router_pattern_balance_entropy_floor", 0.5)
            # L_pattern_supervision: directly teach q_i via per-expert utility
            # (train/query labels only). This is what makes the routing actually
            # specialise instead of collapsing to a uniform mix.
            training.setdefault("lambda_prompt_router_pattern_supervision", 0.1)
            # Differentiable per-pattern utility trains the experts themselves;
            # pattern supervision above only trains q_i from a stop-gradient teacher.
            training.setdefault("lambda_prompt_router_pattern_utility", 0.05)
            training.setdefault("prompt_router_pattern_utility_margin", 0.001)
            training.setdefault("prompt_router_pattern_utility_anti_harm_weight", 0.5)
            training.setdefault("prompt_router_pattern_utility_min_teacher_delta", 1e-5)
            training.setdefault("prompt_router_pattern_utility_helpful_fraction", 0.30)
            training.setdefault("prompt_router_pattern_utility_unhelpful_node_weight", 0.1)
            # Class-level reliability smooths noisy node-wise pattern utility:
            # it teaches the router which class x pattern pairs are broadly safe.
            training.setdefault("lambda_prompt_router_class_pattern_reliability", 0.03)
            training.setdefault("prompt_router_class_pattern_reliability_temperature", 0.5)
            training.setdefault("prompt_router_class_pattern_reliability_positive_margin", 1e-5)
            training.setdefault("prompt_router_class_pattern_reliability_harmful_margin", 0.0)
            training.setdefault("prompt_router_class_pattern_reliability_min_class_count", 2)
            # Temperature applies to per-node standardised utilities (z-scores),
            # so ~0.5 gives meaningful contrast without collapsing to one-hot.
            training.setdefault("prompt_router_pattern_supervision_temperature", 0.5)
            training.setdefault("prompt_router_pattern_utility_temperature", 0.5)
            training.setdefault("prompt_router_pattern_supervision_class_balanced", True)
            training.setdefault("prompt_router_pattern_utility_class_balanced", True)
            if variant in {"p21_lite_adaptive_filter", "p21_v2_hetero_filter"}:
                p21_defaults = {
                    "context_detach": True,
                    "residual_scale": 0.30,
                    "use_node_wise_gate": True,
                    "beta_init": 0.10,
                    "beta_max": 0.30,
                    "gate_max": 0.30,
                    "max_update_norm": 0.08,
                }
                for key, value in p21_defaults.items():
                    if key not in explicit_prompt_adapter_keys:
                        prompt_adapter[key] = value
                if variant == "p21_v2_hetero_filter":
                    prompt_adapter.setdefault("channel_set", "reject_low_two_high_compat_role")
                    prompt_adapter.setdefault("channel_prior", [0.50, 0.10, 0.12, 0.10, 0.10, 0.08])
                    prompt_adapter.setdefault("use_compat_channel", True)
                    prompt_adapter.setdefault("compat_support_source", "full_train")
                    prompt_adapter.setdefault("compat_source", "no_prompt_logits")
                    prompt_adapter.setdefault("compat_matrix", "support_estimated")
                    prompt_adapter.setdefault("compat_smoothing", 0.50)
                    prompt_adapter.setdefault("compat_detach_logits", True)
                    prompt_adapter.setdefault("compat_detach_prototypes", True)
                    prompt_adapter.setdefault("use_role_channel", True)
                    prompt_adapter.setdefault("role_hidden_dim", 64)
                else:
                    prompt_adapter.setdefault("channel_set", "reject_low_two_high")
                    prompt_adapter.setdefault("channel_prior", [0.50, 0.15, 0.20, 0.15])
                prompt_adapter.setdefault("use_reject_channel", True)
                prompt_adapter.setdefault("reject_channel_index", 0)
                prompt_adapter["use_candidate_pool"] = False
                prompt_adapter["candidate_pool_ratio"] = 1.0
                training["lambda_prompt_router_pattern_balance"] = 0.0
                training["lambda_prompt_router_pattern_supervision"] = 0.0
                training["lambda_prompt_router_pattern_utility"] = 0.0
                training["lambda_prompt_router_class_pattern_reliability"] = 0.0
                training.setdefault("lambda_p22_pattern_only", 1.0)
                training.setdefault("lambda_p22_pattern_reg", 0.005)
                training.setdefault("lambda_p22_basis_teacher", 0.5)
                training.setdefault("lambda_p22_basis_usage", 0.01)
                if variant == "p21_v2_hetero_filter":
                    training.setdefault("expert_warmup_epochs", 30)
                    training.setdefault("lambda_p21_channel_expert_utility", 0.20)
                    training.setdefault("lambda_p21_channel_expert_utility_after_warmup", 0.05)
                    training.setdefault("p21_channel_expert_probe_scale", 1.0)
                    training.setdefault("p21_channel_expert_temperature", 0.10)
                    training.setdefault("p21_channel_expert_margin", 0.001)
                    training.setdefault("p21_channel_expert_anti_harm_weight", 0.5)
                    training.setdefault("p21_oracle_gate_source", "max")
                    training.setdefault("p21_oracle_scale_grid", [0.25, 0.5, 1.0, 2.0, 4.0])
                training.setdefault("lambda_p21_channel_utility", 0.30 if variant == "p21_v2_hetero_filter" else 0.20)
                training.setdefault("p21_channel_utility_temperature", 0.10)
                training.setdefault("p21_channel_utility_margin", 0.001)
                training.setdefault("p21_channel_utility_min_teacher_delta", 0.001)
                training.setdefault("p21_channel_utility_target_mode", "hard_reject_or_best")
                training.setdefault("p21_channel_utility_gate_source", "max" if variant == "p21_v2_hetero_filter" else "actual")
                training.setdefault("p21_channel_utility_gate_source_warmup", "max")
                training.setdefault("p21_channel_utility_actual_gate_start_epoch", 80 if variant == "p21_v2_hetero_filter" else 0)
                training.setdefault("p21_channel_utility_class_balanced", True)
                training.setdefault("lambda_p21_gate_utility", 0.10)
                training.setdefault("p21_gate_utility_temperature", 0.02)
                training.setdefault("p21_gate_utility_margin", 0.001)
                training.setdefault("p21_gate_target_mode", "soft")
                for key in (
                    "p21_channel_utility_temperature",
                    "p21_channel_utility_margin",
                    "p21_channel_utility_min_teacher_delta",
                    "p21_channel_utility_target_mode",
                    "p21_channel_utility_gate_source",
                    "p21_channel_utility_gate_source_warmup",
                    "p21_channel_utility_actual_gate_start_epoch",
                    "p21_gate_utility_temperature",
                    "p21_gate_utility_margin",
                    "p21_gate_target_mode",
                    "p21_oracle_gate_source",
                    "p21_oracle_scale_grid",
                ):
                    if key in training:
                        prompt_adapter.setdefault(key, training[key])
                training.setdefault("lambda_prompt_router_deployment_utility", 0.05 if variant == "p21_v2_hetero_filter" else 0.10)
                training.setdefault("prompt_router_deployment_utility_margin", 0.0005)
                training.setdefault("prompt_router_deployment_utility_anti_harm_weight", 0.5)
                training.setdefault("prompt_router_deployment_utility_anti_harm_margin", 0.0)
                training.setdefault("prompt_router_deployment_utility_gain_reward_weight", 0.10)
                training.setdefault("prompt_router_deployment_utility_gain_reward_cap", 0.02)
                training.setdefault("prompt_router_deployment_utility_class_balanced", True)
                training.setdefault("prompt_adapter_episode_count_per_epoch", 1)
                if "lambda_prompt_adapter_update_norm" not in explicit_training_keys:
                    training["lambda_prompt_adapter_update_norm"] = 0.005 if variant == "p21_v2_hetero_filter" else 0.01
                training["lambda_prompt_adapter_gate_budget"] = 0.0
                training["lambda_prompt_adapter_utility_gate"] = 0.0
                training["prompt_adapter_update_mask"] = "all"
                training["lambda_prompt_adapter_message_help"] = min(
                    float(training.get("lambda_prompt_adapter_message_help", 0.02 if variant == "p21_v2_hetero_filter" else 0.05)),
                    0.05,
                )
            elif variant == "p20_utility_supervised_pattern_prompt_router":
                training.setdefault("lambda_prompt_router_deployment_utility", 0.10)
                training.setdefault("prompt_router_deployment_utility_margin", 0.0005)
                training.setdefault("prompt_router_deployment_utility_anti_harm_weight", 1.0)
                training.setdefault("prompt_router_deployment_utility_anti_harm_margin", 0.0)
                training.setdefault("prompt_router_deployment_utility_gain_reward_weight", 0.25)
                training.setdefault("prompt_router_deployment_utility_gain_reward_cap", 0.02)
                training.setdefault("prompt_router_deployment_utility_class_balanced", True)
                training.setdefault("lambda_prompt_router_expert_utility_supervision", 0.20)
                training.setdefault("prompt_router_expert_utility_temperature", 0.5)
                training.setdefault("prompt_router_expert_utility_gain_temperature", 0.02)
                training.setdefault("prompt_router_expert_utility_margin", 0.0005)
                training.setdefault("prompt_router_expert_utility_target", "margin_softmax")
                training.setdefault("prompt_router_expert_utility_class_balanced", True)
                training.setdefault("prompt_router_expert_utility_use_actual_message", True)
                training.setdefault("prompt_router_expert_utility_gate_weight", 0.25)
                training.setdefault("prompt_router_expert_utility_gate_target", "margin_sigmoid")
                training.setdefault("prompt_router_expert_utility_gate_temperature", 0.02)
                training["lambda_prompt_router_pattern_balance"] = 0.0
                training["lambda_prompt_router_pattern_supervision"] = 0.0
                training["lambda_prompt_router_pattern_utility"] = 0.0
                training["lambda_prompt_router_class_pattern_reliability"] = 0.0
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_temperature",
                    training["prompt_router_expert_utility_temperature"],
                )
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_gain_temperature",
                    training["prompt_router_expert_utility_gain_temperature"],
                )
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_margin",
                    training["prompt_router_expert_utility_margin"],
                )
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_target",
                    training["prompt_router_expert_utility_target"],
                )
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_gate_weight",
                    training["prompt_router_expert_utility_gate_weight"],
                )
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_gate_target",
                    training["prompt_router_expert_utility_gate_target"],
                )
                prompt_adapter.setdefault(
                    "prompt_router_expert_utility_gate_temperature",
                    training["prompt_router_expert_utility_gate_temperature"],
                )
                prompt_adapter.setdefault("prompt_router_expert_oracle_scale_grid", [0.5, 1.0, 2.0, 4.0])
                if "prompt_router_expert_utility_probe_norm" in training:
                    prompt_adapter.setdefault(
                        "prompt_router_expert_utility_probe_norm",
                        training["prompt_router_expert_utility_probe_norm"],
                    )
            if variant in p22_variants:
                prompt_adapter["use_candidate_pool"] = False
                prompt_adapter["candidate_pool_ratio"] = 1.0
                training["freeze_base_model"] = True
                training["prompt_adapter_update_mask"] = "all"
                training["prompt_adapter_loss_mask"] = "query"
                training["lambda_prompt_adapter_update_norm"] = 0.0
                training["lambda_prompt_adapter_gate_budget"] = 0.0
                training["lambda_prompt_adapter_message_help"] = 0.0
                training["lambda_prompt_adapter_utility_gate"] = 0.0
                training["lambda_prompt_router_pattern_balance"] = 0.0
                training["lambda_prompt_router_pattern_supervision"] = 0.0
                training["lambda_prompt_router_pattern_utility"] = 0.0
                training["lambda_prompt_router_class_pattern_reliability"] = 0.0
                training["lambda_prompt_router_deployment_utility"] = 0.0
                training["lambda_prompt_adapter_gate_consistency"] = 0.0
                training["lambda_prompt_adapter_delta_consistency"] = 0.0
            else:
                # Reuse p18-style episodic consistency.
                training.setdefault("prompt_adapter_episode_count_per_epoch", 3)
                training.setdefault("lambda_prompt_adapter_gate_consistency", 0.02)
                training.setdefault("lambda_prompt_adapter_delta_consistency", 0.01)
        else:
            training.setdefault("freeze_base_model", False)
            training["train_prompt_adapter"] = True
    else:
        prompt_adapter["enabled"] = False
    if variant == "p2_no_node_to_prompt":
        prompt_aware["use_node_to_prompt"] = False
        prompt_aware.setdefault("use_prompt_to_node", True)
    elif variant == "p2_no_prompt_to_node":
        prompt_aware["use_prompt_to_node"] = False
        prompt_aware.setdefault("use_node_to_prompt", True)
    elif variant in {
        "p2_prompt_aware",
        "p2_strength",
        "p2_strength_random_pool",
        "p2_multiview",
        "p2_pattern_benefit",
        "p5_multi_expert",
        "p5_lite_receiver",
        "p13_utility_correction",
    }:
        prompt_aware.setdefault("use_node_to_prompt", True)
        prompt_aware.setdefault("use_prompt_to_node", True)
    if variant == "p2_pattern_benefit":
        prompt_aware["use_node_to_prompt"] = False
        prompt_aware["use_prompt_to_node"] = True
    if variant in {
        "p2_strength",
        "p2_strength_random_pool",
        "p2_multiview",
        "p2_pattern_benefit",
        "p5_multi_expert",
        "p5_lite_receiver",
    }:
        prompt_aware.setdefault("gate_init", 0.05)
        prompt_aware.setdefault("message_scale", 5.0)
        prompt_aware.setdefault("prompt_message_norm", "weighted_mean")
        prompt_graph.setdefault("edge_scale_init", 0.05)
        prompt_graph.setdefault("edge_scale_max", 0.75)
        prompt_graph.setdefault("lambda_prompt_balance", 0.02)
    if variant in {"p2_multiview", "p5_multi_expert", "p5_lite_receiver"}:
        prompt_graph["use_multiview_routing"] = True
        prompt_graph.setdefault("use_class_aware_routing", True)
        prompt_graph.setdefault("residual_prompt_count", 2)
        prompt_graph.setdefault("lambda_class_route", 0.05)
        prompt_graph.setdefault("lambda_key_proto", 0.001)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.02)
    if variant == "p5_multi_expert":
        prompt_aware["receiver_version"] = "v4_multi_expert_residual"
        prompt_aware.setdefault("prompt_fusion", "multi_expert_residual_correction")
        prompt_aware.setdefault("zero_init_prompt_messages", True)
        prompt_aware.setdefault("prompt_message_norm", "layernorm")
        prompt_aware.setdefault("use_bounded_prompt_update", True)
        prompt_aware.setdefault("max_prompt_update_norm", 0.05)
        prompt_graph.setdefault("residual_prompt_count", 8)
        prompt_graph.setdefault("num_prompt_nodes", 16)
        prompt_graph.setdefault("init_class_keys_from_train_proto", True)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("use_benefit_gate", True)
        prompt_graph.setdefault("use_hard_receive_gate", True)
        prompt_graph.setdefault("hard_receive_ratio", 0.20)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.10)
        prompt_graph.setdefault("prompt_message_help_margin", 0.01)
        prompt_graph.setdefault("prompt_message_help_warmup_epochs", 10)
        prompt_graph.setdefault("prompt_message_help_class_balanced", True)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.10)
        prompt_graph.setdefault("prompt_class_anti_harm_floor", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.05)
        prompt_graph.setdefault("benefit_supervision_warmup_epochs", 10)
        prompt_graph.setdefault("benefit_supervision_label_strategy", "hybrid_quantile_margin")
        prompt_graph.setdefault("benefit_supervision_quantile", 0.20)
        prompt_graph.setdefault("benefit_delta_eps", 1e-4)
    if variant == "p5_lite_receiver":
        prompt_aware["receiver_version"] = "v4_multi_expert_residual"
        prompt_aware.setdefault("prompt_fusion", "multi_expert_residual_correction")
        prompt_aware.setdefault("zero_init_prompt_messages", True)
        prompt_aware.setdefault("prompt_message_norm", "weighted_mean")
        prompt_aware.setdefault("use_bounded_prompt_update", True)
        prompt_aware.setdefault("max_prompt_update_norm", 0.20)
        prompt_aware.setdefault("prompt_to_node_gate_init", 0.10)
        prompt_graph.setdefault("residual_prompt_count", 8)
        prompt_graph.setdefault("num_prompt_nodes", 16)
        prompt_graph.setdefault("init_class_keys_from_train_proto", False)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("use_benefit_gate", False)
        prompt_graph.setdefault("use_hard_receive_gate", False)
        prompt_graph.setdefault("use_hard_acceptance", False)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_role_diversity", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_correction", 0.0)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_message_help", 0.10)
        prompt_graph.setdefault("prompt_message_help_margin", 0.005)
        prompt_graph.setdefault("prompt_message_help_warmup_epochs", 5)
        prompt_graph.setdefault("prompt_message_help_class_balanced", True)
    if variant == "p13_utility_correction":
        prompt_aware.setdefault("receiver_version", "v6_classifier_directional")
        prompt_aware.setdefault("prompt_fusion", "classifier_directional_residual")
        prompt_aware.setdefault("zero_init_prompt_messages", False)
        prompt_aware.setdefault("prompt_message_norm", "weighted_mean")
        prompt_aware.setdefault("use_bounded_prompt_update", True)
        prompt_aware.setdefault("max_prompt_update_norm", 0.04)
        prompt_aware.setdefault("use_node_to_prompt", False)
        prompt_aware.setdefault("use_prompt_to_node", True)
        prompt_graph.setdefault("use_multiview_routing", True)
        prompt_graph.setdefault("use_attribute_view", True)
        prompt_graph.setdefault("use_enhanced_role_view", True)
        prompt_graph.setdefault("use_class_aware_routing", False)
        prompt_graph.setdefault("use_pattern_prompt_bank", True)
        prompt_graph.setdefault("init_class_keys_from_train_proto", False)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("use_benefit_gate", False)
        prompt_graph.setdefault("use_hard_receive_gate", False)
        prompt_graph.setdefault("use_hard_acceptance", False)
        prompt_graph.setdefault("lambda_prompt_balance", 0.0)
        prompt_graph.setdefault("lambda_prompt_role_diversity", 0.001)
        prompt_graph.setdefault("lambda_prompt_acceptance", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_acceptance_budget", 0.0)
        prompt_graph.setdefault("lambda_prompt_usage_consistency", 0.0)
        prompt_graph.setdefault("lambda_prompt_view_entropy", 0.0)
        prompt_graph.setdefault("lambda_class_route", 0.0)
        prompt_graph.setdefault("lambda_key_proto", 0.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.0)
        prompt_graph.setdefault("lambda_prompt_correction", 0.0)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.0)
        prompt_graph.setdefault("lambda_prompt_class_anti_harm", 0.0)
        prompt_graph.setdefault("pool_strategy", "utility_structural")
        prompt_graph.setdefault("use_edge_utility", True)
        prompt_graph.setdefault("edge_utility_init", 0.50)
        prompt_graph.setdefault("utility_pool_structural_weight", 0.35)
        prompt_graph.setdefault("utility_pool_uncertainty_weight", 0.35)
        prompt_graph.setdefault("utility_pool_disagreement_weight", 0.30)
        prompt_graph.setdefault("lambda_edge_utility_supervision", 0.05)
        prompt_graph.setdefault("edge_utility_margin", 0.0)
        prompt_graph.setdefault("edge_utility_warmup_epochs", 5)
        prompt_graph.setdefault("lambda_correction_alignment", 0.05)
        prompt_graph.setdefault("lambda_correction_anti_harm", 0.05)
        prompt_graph.setdefault("correction_alignment_margin", 0.0)
        prompt_graph.setdefault("correction_alignment_warmup_epochs", 5)
        prompt_aware.setdefault("message_scale", 0.10)
        prompt_aware.setdefault("message_scale_grid", [0.0, 0.05, 0.10, 0.25])
    if variant == "p2_pattern_benefit":
        prompt_graph["use_multiview_routing"] = True
        prompt_graph["use_pattern_prompt_bank"] = True
        prompt_graph["use_receiver_only_prompt"] = True
        prompt_graph["use_benefit_gate"] = True
        prompt_aware.setdefault("zero_init_prompt_messages", False)
        prompt_aware.setdefault("prompt_to_node_gate_init", 0.20)
        prompt_graph.setdefault("init_pattern_keys_from_pool_medoids", True)
        prompt_graph.setdefault("benefit_gate_bias_init", -1.0)
        prompt_graph.setdefault("lambda_prompt_benefit_supervision", 0.10)
        prompt_graph.setdefault("benefit_supervision_warmup_epochs", 20)
        prompt_graph.setdefault("benefit_delta_margin", 0.0)
        prompt_graph.setdefault("benefit_supervision_label_strategy", "hybrid_quantile_margin")
        prompt_graph.setdefault("benefit_supervision_quantile", 0.20)
        prompt_graph.setdefault("benefit_delta_eps", 1e-4)
        prompt_graph.setdefault("benefit_supervision_quantile_warmup_epochs", 20)
        prompt_graph.setdefault("benefit_supervision_probe_scale", 2.0)
        prompt_graph.setdefault("benefit_supervision_balance_targets", True)
        prompt_graph.setdefault("lambda_prompt_correction", 0.10)
        prompt_graph.setdefault("lambda_prompt_anti_harm", 0.02)
        prompt_graph.setdefault("prompt_correction_warmup_epochs", 20)
        prompt_graph.setdefault("prompt_correction_eps", 0.0001)
        prompt_graph.setdefault("prompt_correction_target", "delta_prob_to_label")
        prompt_graph.setdefault("use_hard_receive_gate", True)
        prompt_graph.setdefault("hard_receive_ratio", 0.15)
        prompt_graph.setdefault("hard_receive_straight_through", True)
    if variant in {"p1_random_pool", "p2_strength_random_pool"}:
        prompt_graph["pool_strategy"] = "random"
    elif variant in {
        "p1_graph",
        "p2_prompt_aware",
        "p2_strength",
        "p2_multiview",
        "p2_pattern_benefit",
        "p5_multi_expert",
        "p5_lite_receiver",
        "p13_utility_correction",
        "p2_no_node_to_prompt",
        "p2_no_prompt_to_node",
    }:
        prompt_graph.setdefault("pool_strategy", "structural")
    return out


def _resolve_p21_utility_cfg(
    prompt_adapter_cfg: dict[str, Any],
    training_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve P21 utility/oracle knobs with training config priority."""
    training_cfg = training_cfg or {}

    def get(key: str, default: Any) -> Any:
        if key in training_cfg:
            return training_cfg[key]
        if key in prompt_adapter_cfg:
            return prompt_adapter_cfg[key]
        return default

    return {
        "p21_channel_utility_temperature": float(get("p21_channel_utility_temperature", 0.10)),
        "p21_channel_utility_margin": float(get("p21_channel_utility_margin", 0.001)),
        "p21_channel_utility_min_teacher_delta": float(get("p21_channel_utility_min_teacher_delta", 0.001)),
        "p21_channel_utility_target_mode": str(get("p21_channel_utility_target_mode", "hard_reject_or_best")),
        "p21_channel_utility_gate_source": str(get("p21_channel_utility_gate_source", "actual")),
        "p21_channel_utility_gate_source_warmup": str(get("p21_channel_utility_gate_source_warmup", "max")),
        "p21_channel_utility_actual_gate_start_epoch": int(get("p21_channel_utility_actual_gate_start_epoch", 0)),
        "p21_gate_utility_temperature": float(get("p21_gate_utility_temperature", 0.02)),
        "p21_gate_utility_margin": float(get("p21_gate_utility_margin", 0.001)),
        "p21_gate_target_mode": str(get("p21_gate_target_mode", "soft")),
        "p21_oracle_gate_source": str(get("p21_oracle_gate_source", get("p21_channel_utility_gate_source", "actual"))),
        "p21_oracle_scale_grid": get("p21_oracle_scale_grid", [0.25, 0.5, 1.0, 2.0, 4.0]),
    }


def _class_balanced_support_query_split(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    support_ratio: float = 0.70,
    min_query_per_class: int = 1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Split training nodes into support/query masks without using val/test labels."""

    device = labels.device
    support = torch.zeros_like(train_mask, dtype=torch.bool, device=device)
    query = torch.zeros_like(train_mask, dtype=torch.bool, device=device)
    support_ratio = min(max(float(support_ratio), 0.0), 1.0)
    min_query_per_class = max(0, int(min_query_per_class))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    by_class: dict[str, dict[str, int]] = {}
    train_idx_all = torch.where(train_mask.bool())[0]
    if train_idx_all.numel() == 0:
        return support, query, {"enabled": True, "support_count": 0, "query_count": 0, "by_class": by_class}

    for class_id in torch.unique(labels[train_idx_all].detach().cpu()).tolist():
        class_id = int(class_id)
        idx = torch.where(train_mask.bool() & (labels == class_id))[0]
        count = int(idx.numel())
        if count == 0:
            continue
        if count <= min_query_per_class:
            support_idx = idx
            query_idx = idx.new_empty(0)
        else:
            perm = torch.randperm(count, generator=generator, device="cpu").to(device=idx.device)
            query_count = min(max(min_query_per_class, int(round(count * (1.0 - support_ratio)))), count - 1)
            query_idx = idx[perm[:query_count]]
            support_idx = idx[perm[query_count:]]
        support[support_idx] = True
        query[query_idx] = True
        by_class[str(class_id)] = {
            "total": count,
            "support": int(support_idx.numel()),
            "query": int(query_idx.numel()),
        }

    # Tiny-shot fallback: if every class had too few nodes, keep the old train-only behavior.
    if int(query.sum().item()) == 0:
        support = train_mask.bool().clone()

    return support, query, {
        "enabled": True,
        "support_count": int(support.sum().item()),
        "query_count": int(query.sum().item()),
        "support_ratio": float(support.float().mean().item()),
        "train_support_ratio": float(
            support.float().sum().item() / max(1, int(train_mask.bool().sum().item()))
        ),
        "by_class": by_class,
    }


def _class_balanced_rotating_kfold_split(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return a deterministic class-stratified held-out fold for P24 cross-fitting."""

    device = labels.device
    support = train_mask.bool().clone()
    query = torch.zeros_like(train_mask, dtype=torch.bool, device=device)
    num_folds = max(2, int(num_folds))
    fold_index = int(fold_index) % num_folds
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    by_class: dict[str, dict[str, int]] = {}
    train_idx_all = torch.where(train_mask.bool())[0]

    for class_id in torch.unique(labels[train_idx_all].detach().cpu()).tolist():
        class_id = int(class_id)
        idx = torch.where(train_mask.bool() & (labels == class_id))[0]
        count = int(idx.numel())
        if count < 2:
            by_class[str(class_id)] = {"total": count, "support": count, "query": 0}
            continue
        perm = torch.randperm(count, generator=generator, device="cpu").to(device=idx.device)
        class_query = idx[perm[fold_index::num_folds]]
        if class_query.numel() == 0:
            class_query = idx[perm[fold_index % count : fold_index % count + 1]]
        query[class_query] = True
        support[class_query] = False
        by_class[str(class_id)] = {
            "total": count,
            "support": count - int(class_query.numel()),
            "query": int(class_query.numel()),
        }

    if not bool(query.any()):
        return _class_balanced_support_query_split(
            labels,
            train_mask,
            support_ratio=(num_folds - 1.0) / num_folds,
            min_query_per_class=1,
            seed=seed,
        )
    return support, query, {
        "enabled": True,
        "mode": "rotating_kfold",
        "num_folds": num_folds,
        "fold_index": fold_index,
        "support_count": int(support.sum().item()),
        "query_count": int(query.sum().item()),
        "train_support_ratio": float(
            support.sum().item() / max(1, int(train_mask.bool().sum().item()))
        ),
        "by_class": by_class,
    }


def _support_query_masks_for_epoch(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_graph_cfg: dict[str, Any],
    *,
    seed: int,
    epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    split_cfg = prompt_graph_cfg.get("support_query_split", {})
    enabled = bool(split_cfg.get("enabled", prompt_graph_cfg.get("support_query_split_enabled", False)))
    if not enabled:
        train = train_mask.bool().clone()
        empty = torch.zeros_like(train, dtype=torch.bool)
        return train, empty, {
            "enabled": False,
            "support_count": int(train.sum().item()),
            "query_count": 0,
            "train_support_ratio": 1.0,
            "by_class": {},
        }
    support_ratio = float(split_cfg.get("support_ratio", prompt_graph_cfg.get("support_query_support_ratio", 0.70)))
    min_query = int(split_cfg.get("min_query_per_class", prompt_graph_cfg.get("support_query_min_query_per_class", 1)))
    mode = str(split_cfg.get("mode", "random")).lower()
    if mode == "rotating_kfold":
        num_folds = int(split_cfg.get("num_folds", 3))
        fold_hold_epochs = max(1, int(split_cfg.get("fold_hold_epochs", 1)))
        fold_index = max(0, int(epoch) - 1) // fold_hold_epochs
        return _class_balanced_rotating_kfold_split(
            labels,
            train_mask,
            num_folds=num_folds,
            fold_index=fold_index,
            seed=int(split_cfg.get("seed", seed)),
        )
    if mode not in {"random", "resampled_random"}:
        raise ValueError(f"Unsupported support_query_split mode: {mode}")
    resample = bool(split_cfg.get("resample_each_epoch", prompt_graph_cfg.get("support_query_resample_each_epoch", True)))
    split_seed = int(seed if resample else int(split_cfg.get("seed", seed)))
    if resample:
        split_seed = split_seed * 1000003 + int(epoch)
    return _class_balanced_support_query_split(
        labels,
        train_mask,
        support_ratio=support_ratio,
        min_query_per_class=min_query,
        seed=split_seed,
    )


def _build_prompt_graph_module(
    *,
    variant: str,
    source_dim: int,
    hidden_dim: int,
    num_classes: int,
    prompt_graph_cfg: dict[str, Any],
    device: torch.device,
) -> (
    PromptGraphModuleP1
    | SelectiveDiscreteFeaturePromptGraph
    | GraphiteStylePromptGraphAdapter
    | PromptGraphConstruction
    | P23ProjectorSynchronizedPromptGraphAdapter
    | PromptingAdaptation
    | P23V01PromptModule
    | None
):
    if variant == "noprompt" or not bool(prompt_graph_cfg.get("enabled", True)):
        return None
    resolved_cfg = dict(prompt_graph_cfg)
    resolved_cfg.setdefault("num_classes", int(num_classes))
    if variant in {"p23_v0_1", "p23_v0_2_message_passing"} or str(resolved_cfg.get("module_type", "")) == "p23_v0_1":
        return P23V01PromptModule(source_dim, hidden_dim, resolved_cfg).to(device)
    if variant == "p23_graphite_adapter" or str(resolved_cfg.get("module_type", "")) == "graphite_style_prompt_graph":
        return GraphiteStylePromptGraphAdapter(source_dim, hidden_dim, resolved_cfg).to(device)
    if (
        variant == "p23_projector_synchronized_graphite_adapter"
        or str(resolved_cfg.get("module_type", "")) == "p23_projector_synchronized_prompt_graph"
    ):
        return P23ProjectorSynchronizedPromptGraphAdapter(source_dim, hidden_dim, resolved_cfg).to(device)
    if (
        variant == "p23_selective_graphite_adapter"
        or str(resolved_cfg.get("module_type", "")) == "class_aware_selective_graphite_prompt_graph"
    ):
        return PromptGraphConstruction(source_dim, hidden_dim, resolved_cfg).to(device)
    if variant == "p24_prompt_edge_denoising" or str(resolved_cfg.get("module_type", "")) == "p24_prompt_edge_denoising_adapter":
        return PromptingAdaptation(source_dim, hidden_dim, resolved_cfg).to(device)
    if str(resolved_cfg.get("module_type", "")) == "selective_discrete_feature_prompt":
        return SelectiveDiscreteFeaturePromptGraph(source_dim, hidden_dim, resolved_cfg).to(device)
    return PromptGraphModuleP1(source_dim, hidden_dim, resolved_cfg).to(device)


def _build_prompt_adapter_module(
    *,
    source_dim: int,
    hidden_dim: int,
    num_classes: int,
    prompt_adapter_cfg: dict[str, Any],
    device: torch.device,
) -> (
    HeterophilyAwarePromptAdapter
    | ClassConditionedPatternPromptRouter
    | UtilitySupervisedPatternPromptRouter
    | P21LiteAdaptiveFilter
    | P21V2HeteroFilter
    | P22ClassPatternEnrichmentBank
    | None
):
    if not bool(prompt_adapter_cfg.get("enabled", False)):
        return None
    resolved_cfg = dict(prompt_adapter_cfg)
    resolved_cfg.setdefault("num_classes", int(num_classes))
    module_type = str(resolved_cfg.get("module_type", "hetero_adapter"))
    if module_type == "utility_supervised_pattern_router":
        return UtilitySupervisedPatternPromptRouter(source_dim, hidden_dim, resolved_cfg).to(device)
    if module_type == "class_conditioned_pattern_router":
        return ClassConditionedPatternPromptRouter(source_dim, hidden_dim, resolved_cfg).to(device)
    if module_type == "p21_lite_adaptive_filter":
        return P21LiteAdaptiveFilter(source_dim, hidden_dim, resolved_cfg).to(device)
    if module_type == "p21_v2_hetero_filter":
        return P21V2HeteroFilter(source_dim, hidden_dim, resolved_cfg).to(device)
    if module_type in {
        "p22_class_pattern_enrichment_bank",
        "p22_reliability_calibrated_basis_bank",
        "p22_v031_conservative_reliability_basis_bank",
        "p22_v04_minimal_transition_basis",
    }:
        return P22ClassPatternEnrichmentBank(source_dim, hidden_dim, resolved_cfg).to(device)
    if module_type == "hetero_adapter":
        return HeterophilyAwarePromptAdapter(source_dim, hidden_dim, resolved_cfg).to(device)
    raise ValueError(f"Unsupported prompt_adapter.module_type={module_type!r}")


@torch.no_grad()
def _maybe_initialize_class_keys(
    *,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    model: DualBranchFramework,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_graph_cfg: dict[str, Any],
) -> dict[str, Any]:
    if prompt_graph_module is None or not bool(prompt_graph_cfg.get("init_class_keys_from_train_proto", False)):
        return {"class_key_proto_init_coverage": 0.0, "class_key_proto_init_missing_classes": []}
    was_training = prompt_graph_module.training
    input_was_training = input_aligner.training
    model_was_training = model.training
    input_aligner.eval()
    model.eval()
    prompt_graph_module.eval()
    z = input_aligner(x)
    h_pre = model.encode_frozen(z, edge_index)
    stats = prompt_graph_module.initialize_class_keys_from_train_prototypes(
        z=z,
        h_pre=h_pre.detach(),
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        normalize=bool(prompt_graph_cfg.get("class_key_init_normalize", True)),
        source=str(prompt_graph_cfg.get("class_key_init_source", "structural_query")),
    )
    input_aligner.train(input_was_training)
    model.train(model_was_training)
    prompt_graph_module.train(was_training)
    return stats


@torch.no_grad()
def _maybe_initialize_pattern_keys(
    *,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    model: DualBranchFramework,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_graph_cfg: dict[str, Any],
) -> dict[str, Any]:
    if prompt_graph_module is None or not bool(prompt_graph_cfg.get("init_pattern_keys_from_pool_medoids", False)):
        return {"pattern_key_init_coverage": 0.0, "pattern_key_init_selected_nodes": []}
    was_training = prompt_graph_module.training
    input_was_training = input_aligner.training
    model_was_training = model.training
    input_aligner.eval()
    model.eval()
    prompt_graph_module.eval()
    z = input_aligner(x)
    h_pre = model.encode_frozen(z, edge_index)
    stats = prompt_graph_module.initialize_pattern_keys_from_pool_medoids(
        z=z,
        h_pre=h_pre.detach(),
        edge_index=edge_index,
        train_mask=train_mask,
        normalize=bool(prompt_graph_cfg.get("pattern_key_init_normalize", True)),
    )
    input_aligner.train(input_was_training)
    model.train(model_was_training)
    prompt_graph_module.train(was_training)
    return stats


def _default_prompt_graph_out(z: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
    edge_weight = torch.ones(edge_index.size(1), dtype=z.dtype, device=z.device)
    return {
        "adapted_x": z,
        "adapted_edge_index": edge_index,
        "adapted_edge_weight": edge_weight,
        "adapted_edge_type": torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device),
        "prompt_node_x": z.new_zeros((0, z.size(-1))),
        "pool_mask": torch.zeros(z.size(0), dtype=torch.bool, device=z.device),
        "prompt_edge_count": 0,
        "edge_scale": z.new_tensor(0.0),
        "aux": {
            "prompt_edge_weight": z.new_zeros(0),
            "prompt_usage": z.new_zeros(0),
            "prompt_usage_entropy": z.new_tensor(0.0),
            "connected_edge_count": 0,
            "edge_type_counts": [int(edge_index.size(1)), 0, 0],
        },
    }


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-12)


def _adapter_candidate_pool(
    *,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    prompt_adapter_cfg: dict[str, Any],
    no_prompt_logits: torch.Tensor | None = None,
    h_pre: torch.Tensor | None = None,
    h_adp_no_prompt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Old-structure-style candidate pool for node-level prompt adapters.

    This does not create prompt nodes/edges. It only builds an update mask:
    labeled train nodes plus the top-ratio unlabeled/high-score candidates.
    """
    device = z.device
    num_nodes = int(z.size(0))
    train = train_mask.to(device=device, dtype=torch.bool)
    if not bool(prompt_adapter_cfg.get("use_candidate_pool", False)):
        mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        return mask, {
            "candidate_pool_enabled": 0.0,
            "candidate_pool_ratio": 1.0,
            "candidate_pool_count": float(mask.sum().item()),
            "candidate_pool_topk_count": float(num_nodes),
            "candidate_pool_strategy_id": -1.0,
        }

    strategy = str(prompt_adapter_cfg.get("candidate_pool_strategy", "structural"))
    if strategy not in {"structural", "utility_structural"}:
        raise ValueError(f"Unsupported prompt_adapter.candidate_pool_strategy={strategy!r}")
    ratio = min(max(float(prompt_adapter_cfg.get("candidate_pool_ratio", 0.30)), 0.0), 1.0)
    include_train = bool(prompt_adapter_cfg.get("candidate_pool_include_train", True))

    base = z.detach()
    m1 = mean_neighbor_summary(base, edge_index, num_nodes=num_nodes)
    m2 = mean_neighbor_summary(m1, edge_index, num_nodes=num_nodes)
    var = mean_neighbor_variance(base, edge_index, num_nodes=num_nodes).mean(dim=-1)
    sim_1 = 1.0 - F.cosine_similarity(base, m1, dim=-1, eps=1e-12)
    sim_2 = 1.0 - F.cosine_similarity(m1, m2, dim=-1, eps=1e-12)
    structural = torch.stack(
        [sim_1.clamp_min(0.0), sim_2.clamp_min(0.0), _minmax_normalize(var)],
        dim=-1,
    ).mean(dim=-1)
    structural_component = _minmax_normalize(structural.detach()).to(dtype=z.dtype)
    uncertainty = torch.zeros_like(structural_component)
    disagreement = torch.zeros_like(structural_component)

    if strategy == "utility_structural":
        if isinstance(no_prompt_logits, torch.Tensor) and no_prompt_logits.numel() > 0:
            logits = no_prompt_logits.detach().to(device=device, dtype=z.dtype)
            prob = torch.softmax(logits, dim=-1)
            entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
            if logits.size(-1) > 1:
                entropy = entropy / math.log(float(logits.size(-1)))
            uncertainty = entropy.clamp_min(0.0)
        if isinstance(h_pre, torch.Tensor) and isinstance(h_adp_no_prompt, torch.Tensor) and h_pre.shape == h_adp_no_prompt.shape:
            disagreement = 1.0 - F.cosine_similarity(
                h_pre.detach().to(device=device, dtype=z.dtype),
                h_adp_no_prompt.detach().to(device=device, dtype=z.dtype),
                dim=-1,
                eps=1e-12,
            )
            disagreement = _minmax_normalize(disagreement.clamp_min(0.0))
        score = (
            float(prompt_adapter_cfg.get("candidate_pool_structural_weight", 0.35)) * structural_component
            + float(prompt_adapter_cfg.get("candidate_pool_uncertainty_weight", 0.35)) * uncertainty
            + float(prompt_adapter_cfg.get("candidate_pool_disagreement_weight", 0.30)) * disagreement
        )
    else:
        score = structural_component

    topk = int(math.ceil(ratio * num_nodes))
    topk = min(max(topk, 0), num_nodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if topk > 0:
        top_idx = torch.topk(score, k=topk, largest=True).indices
        mask[top_idx] = True
    if include_train:
        mask = mask | train

    selected = score[mask] if bool(mask.any()) else score.new_zeros(0)
    return mask, {
        "candidate_pool_enabled": 1.0,
        "candidate_pool_ratio": float(mask.float().mean().item()),
        "candidate_pool_count": float(mask.sum().item()),
        "candidate_pool_topk_count": float(topk),
        "candidate_pool_include_train": float(include_train),
        "candidate_pool_strategy_id": {"structural": 0.0, "utility_structural": 1.0}[strategy],
        "candidate_pool_score_mean": float(score.mean().detach().item()) if score.numel() > 0 else 0.0,
        "candidate_pool_score_selected_mean": float(selected.mean().detach().item()) if selected.numel() > 0 else 0.0,
        "candidate_pool_structural_mean": float(structural_component.mean().detach().item()) if structural_component.numel() > 0 else 0.0,
        "candidate_pool_uncertainty_mean": float(uncertainty.mean().detach().item()) if uncertainty.numel() > 0 else 0.0,
        "candidate_pool_disagreement_mean": float(disagreement.mean().detach().item()) if disagreement.numel() > 0 else 0.0,
    }


def _prompt_out_with_pool(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    pool_mask: torch.Tensor | None,
    pool_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_out = _default_prompt_graph_out(z, edge_index)
    if isinstance(pool_mask, torch.Tensor):
        mask = pool_mask.to(device=z.device, dtype=torch.bool)
        prompt_out["pool_mask"] = mask
        prompt_out["aux"]["pool_idx"] = torch.where(mask)[0]
        prompt_out["aux"]["pool_selected_ratio"] = z.new_tensor(float(mask.float().mean().item()))
    if pool_stats:
        prompt_out["aux"].update(pool_stats)
    return prompt_out


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def _set_prompt_graph_trainable_scope(
    module: torch.nn.Module | None,
    *,
    trainable: bool,
    scope: str = "all",
) -> None:
    """Restrict prompt-module optimization without changing its forward graph."""

    normalized_scope = str(scope).lower()
    if normalized_scope not in {"all", "edge_weight_only"}:
        raise ValueError(
            "training.prompt_graph_trainable_scope must be 'all' or 'edge_weight_only'"
        )
    if module is None or not trainable:
        _set_module_trainable(module, False)
        return
    if normalized_scope == "all":
        _set_module_trainable(module, True)
        return

    _set_module_trainable(module, False)
    use_feature_gate = bool(getattr(module, "use_feature_gate", False))
    use_node_feature_edge_gate = bool(
        getattr(module, "use_node_feature_edge_gate", False)
    )
    for name, parameter in module.named_parameters():
        path = name.split(".")
        is_dynamic_edge_scale = "feature_edge_log_scale" in path
        is_feature_gate = use_feature_gate and "feature_gate" in path
        is_node_edge_gate = (
            use_node_feature_edge_gate and "node_feature_edge_gate" in path
        )
        parameter.requires_grad = bool(
            is_dynamic_edge_scale or is_feature_gate or is_node_edge_gate
        )


def _trainable_parameters(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    if module is None:
        return []
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _prompt_aware_parameter_names(model: DualBranchFramework) -> set[str]:
    if not isinstance(model, PromptAwareGP2F):
        return set()
    prefixes = (
        "prompt_gate_logit",
        "node_to_prompt_msgs.",
        "prompt_to_node_msgs.",
        "node_to_prompt_conditioned_msgs.",
        "prompt_to_node_conditioned_msgs.",
        "prompt_node_residual_corrections.",
        "prompt_slot_residual_corrections.",
        "prototype_direction_projections.",
        "prototype_direction_strength",
        "prompt_update_norms.",
    )
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)
    }


def _set_prompt_aware_trainable(model: DualBranchFramework, trainable: bool) -> None:
    if not isinstance(model, PromptAwareGP2F):
        return
    names = _prompt_aware_parameter_names(model)
    if names:
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.requires_grad = bool(trainable)
        return
    prefixes = (
        "prompt_gate_logit",
        "node_to_prompt_msgs.",
        "prompt_to_node_msgs.",
        "node_to_prompt_conditioned_msgs.",
        "prompt_to_node_conditioned_msgs.",
        "prompt_node_residual_corrections.",
        "prompt_slot_residual_corrections.",
        "prototype_direction_projections.",
        "prototype_direction_strength",
        "prompt_update_norms.",
    )
    for name, parameter in model.named_parameters():
        if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad = bool(trainable)


def _trainable_parameter_summary(
    *,
    input_aligner: InputAligner,
    model: DualBranchFramework,
    prompt_graph_module: torch.nn.Module | None,
    prompt_adapter_module: torch.nn.Module | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    modules: list[tuple[str, torch.nn.Module | None]] = [
        ("input_aligner", input_aligner),
        ("model", model),
        ("prompt_graph_module", prompt_graph_module),
        ("prompt_adapter_module", prompt_adapter_module),
    ]
    for group_name, module in modules:
        names: list[str] = []
        total = 0
        if module is not None:
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    names.append(f"{group_name}.{name}")
                    total += int(parameter.numel())
        groups[group_name] = names
        counts[f"{group_name}_trainable_parameter_count"] = total
    all_names = [name for names in groups.values() for name in names]
    return {
        "trainable_parameter_names": all_names,
        "trainable_parameter_groups": groups,
        "actual_trainable_parameter_count": sum(counts.values()),
        **counts,
    }


def _optimizer_groups(
    *,
    input_aligner: InputAligner,
    model: DualBranchFramework,
    prompt_graph_module: torch.nn.Module | None,
    prompt_adapter_module: torch.nn.Module | None = None,
    training_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter], dict[str, float]]:
    base_lr = float(training_cfg.get("lr", 0.001))
    base_weight_decay = float(training_cfg.get("weight_decay", 5e-4))
    fusion_alpha_lr_cfg = training_cfg.get("fusion_alpha_lr")
    fusion_alpha_lr = (
        None if fusion_alpha_lr_cfg is None else float(fusion_alpha_lr_cfg)
    )
    if fusion_alpha_lr is not None and fusion_alpha_lr < 0.0:
        raise ValueError("training.fusion_alpha_lr must be non-negative")
    prompt_lr_cfg = training_cfg.get("prompt_lr")
    if prompt_lr_cfg is None:
        prompt_lr = base_lr * float(training_cfg.get("prompt_lr_multiplier", 1.0))
    else:
        prompt_lr = float(prompt_lr_cfg)
    prompt_weight_decay = float(training_cfg.get("prompt_weight_decay", base_weight_decay))

    prompt_aware_names = _prompt_aware_parameter_names(model)
    prompt_aware_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in prompt_aware_names
    ]
    prompt_aware_param_ids = {id(parameter) for parameter in prompt_aware_params}
    fusion_alpha_params = [
        parameter
        for parameter in (model.alpha_logit, model.raw_alpha)
        if parameter is not None and parameter.requires_grad
    ]
    separate_fusion_alpha = fusion_alpha_lr is not None and bool(fusion_alpha_params)
    fusion_alpha_param_ids = (
        {id(parameter) for parameter in fusion_alpha_params}
        if separate_fusion_alpha
        else set()
    )
    model_base_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in prompt_aware_param_ids
        and id(parameter) not in fusion_alpha_param_ids
    ]
    base_params = _trainable_parameters(input_aligner) + model_base_params
    prompt_params = prompt_aware_params + _trainable_parameters(prompt_graph_module) + _trainable_parameters(prompt_adapter_module)
    groups: list[dict[str, Any]] = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr, "weight_decay": base_weight_decay, "name": "base"})
    if separate_fusion_alpha:
        groups.append(
            {
                "params": fusion_alpha_params,
                "lr": float(fusion_alpha_lr),
                "weight_decay": base_weight_decay,
                "name": "fusion_alpha",
            }
        )
    if prompt_params:
        groups.append({"params": prompt_params, "lr": prompt_lr, "weight_decay": prompt_weight_decay, "name": "prompt_graph"})
    trainable_params = [parameter for group in groups for parameter in group["params"]]
    return groups, trainable_params, {
        "base_lr": base_lr if base_params else 0.0,
        "fusion_alpha_lr": (
            float(fusion_alpha_lr)
            if separate_fusion_alpha
            else (base_lr if fusion_alpha_params else 0.0)
        ),
        "fusion_alpha_separate_group": float(separate_fusion_alpha),
        "prompt_lr": prompt_lr if prompt_params else 0.0,
        "base_weight_decay": base_weight_decay if base_params else 0.0,
        "prompt_weight_decay": prompt_weight_decay if prompt_params else 0.0,
        "prompt_aware_parameter_count": sum(int(parameter.numel()) for parameter in prompt_aware_params),
        "prompt_adapter_parameter_count": sum(int(parameter.numel()) for parameter in _trainable_parameters(prompt_adapter_module)),
    }


def _resolve_base_checkpoint(
    training_cfg: dict[str, Any],
    *,
    repo_root: Path,
    seed: int,
    split_seed: int | None = None,
    trial_index: int | None = None,
) -> Path | None:
    resolved_split_seed = int(seed if split_seed is None else split_seed)
    resolved_trial = 0 if trial_index is None else int(trial_index)
    checkpoint_path = training_cfg.get("base_checkpoint_path")
    if checkpoint_path:
        formatted = str(checkpoint_path).format(
            seed=int(seed),
            split_seed=resolved_split_seed,
            trial_index=resolved_trial,
        )
        return _resolve_path(formatted, base_dir=repo_root)
    checkpoint_root = training_cfg.get("base_checkpoint_root")
    if not checkpoint_root:
        return None
    root = _resolve_path(checkpoint_root, base_dir=repo_root)
    direct_candidates = [
        root / f"seed_{resolved_split_seed}" / f"trial_{resolved_trial}" / "best_model.pt",
        root / f"seed_{resolved_split_seed}" / "best_model.pt",
        root / f"seed_{int(seed)}" / f"trial_{resolved_trial}" / "best_model.pt",
        root / f"seed_{int(seed)}" / "best_model.pt",
    ]
    candidates = [path for path in direct_candidates if path.is_file()]
    if not candidates:
        candidates = sorted(
            root.glob(f"**/seed_{resolved_split_seed}/trial_{resolved_trial}/best_model.pt")
        )
    if not candidates:
        candidates = sorted(root.glob(f"**/seed_{resolved_split_seed}/best_model.pt"))
    if not candidates:
        candidates = sorted(root.glob(f"**/seed_{int(seed)}/trial_{resolved_trial}/best_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No base checkpoint found for "
            f"seed={int(seed)}, split_seed={resolved_split_seed}, trial={resolved_trial} under {root}"
        )
    return candidates[0]


def _load_base_checkpoint(
    checkpoint_path: Path,
    *,
    model: DualBranchFramework,
    input_aligner: InputAligner,
    prompt_graph_module: torch.nn.Module | None,
    prompt_adapter_module: torch.nn.Module | None = None,
    device: torch.device,
    load_prompt: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    input_aligner.load_state_dict(checkpoint["input_aligner"])
    if load_prompt and prompt_graph_module is not None and checkpoint.get("prompt_graph_module") is not None:
        prompt_graph_module.load_state_dict(checkpoint["prompt_graph_module"])
    if load_prompt and prompt_adapter_module is not None and checkpoint.get("prompt_adapter_module") is not None:
        prompt_adapter_module.load_state_dict(checkpoint["prompt_adapter_module"])


def _save_checkpoint(
    path: Path,
    *,
    model: DualBranchFramework,
    input_aligner: InputAligner,
    prompt_graph_module: torch.nn.Module | None,
    prompt_adapter_module: torch.nn.Module | None = None,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_aligner": input_aligner.state_dict(),
            "prompt_graph_module": prompt_graph_module.state_dict() if prompt_graph_module is not None else None,
            "prompt_adapter_module": prompt_adapter_module.state_dict() if prompt_adapter_module is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def _monitor_improved(monitor: str, value: float, best_value: float, has_best: bool) -> bool:
    if not has_best:
        return True
    return value < best_value if monitor in {"train_loss", "total", "cls", "edge_l1", "prompt_balance"} else value > best_value


def _update_test_acc_diagnostic(
    current: dict[str, Any],
    *,
    epoch: int,
    train_loss: float,
    metrics: dict[str, Any],
    selection_metric: str,
    maximize: bool,
) -> dict[str, Any]:
    selection_value = (
        float(train_loss)
        if selection_metric == "train_loss"
        else float(metrics[selection_metric])
    )
    if current:
        incumbent = float(current["selection_value"])
        improved = selection_value > incumbent if maximize else selection_value < incumbent
        if not improved:
            return current
    return {
        "epoch": float(epoch),
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "train_loss": float(train_loss),
        "val_acc": float(metrics.get("val_acc", 0.0)),
        "val_macro_f1": float(metrics.get("val_macro_f1", 0.0)),
        "test_acc": float(metrics.get("test_acc", 0.0)),
        "test_macro_f1": float(metrics.get("test_macro_f1", 0.0)),
        "test_auroc": float(metrics.get("test_auroc", 0.0)),
        "test_auprc": float(metrics.get("test_auprc", 0.0)),
    }


def _pool_label_distribution(pool_mask: torch.Tensor, labels: torch.Tensor, num_classes: int) -> list[float]:
    pool_labels = labels[pool_mask]
    if pool_labels.numel() == 0:
        return [0.0 for _ in range(int(num_classes))]
    counts = torch.stack([(pool_labels == class_id).sum() for class_id in range(int(num_classes))]).float()
    return [float(value) for value in (counts / counts.sum().clamp_min(1.0)).tolist()]


def _prompt_usage_by_true_class(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    num_classes: int,
) -> list[list[float]]:
    aux = prompt_out.get("aux", {})
    top_prompt_ids = aux.get("top_prompt_ids")
    assignment_prob = aux.get("assignment_prob")
    pool_mask = prompt_out.get("pool_mask")
    prompt_node_x = prompt_out.get("prompt_node_x")
    if not (
        isinstance(top_prompt_ids, torch.Tensor)
        and isinstance(assignment_prob, torch.Tensor)
        and isinstance(pool_mask, torch.Tensor)
        and isinstance(prompt_node_x, torch.Tensor)
    ):
        return []
    num_prompt_nodes = int(prompt_node_x.size(0))
    usage = torch.zeros((int(num_classes), num_prompt_nodes), dtype=assignment_prob.dtype, device=assignment_prob.device)
    pool_idx = torch.where(pool_mask)[0]
    if pool_idx.numel() == 0:
        return [[0.0 for _ in range(num_prompt_nodes)] for _ in range(int(num_classes))]
    pool_labels = labels[pool_idx]
    for row in range(top_prompt_ids.size(0)):
        class_id = int(pool_labels[row].item())
        usage[class_id].index_add_(0, top_prompt_ids[row], assignment_prob[row])
    usage = usage / usage.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return [[float(value) for value in row] for row in usage.detach().cpu().tolist()]


def _prompt_mediated_same_label_reachability(prompt_out: dict[str, Any], labels: torch.Tensor) -> float:
    aux = prompt_out.get("aux", {})
    top_prompt_ids = aux.get("top_prompt_ids")
    assignment_prob = aux.get("assignment_prob")
    pool_mask = prompt_out.get("pool_mask")
    prompt_node_x = prompt_out.get("prompt_node_x")
    if not (
        isinstance(top_prompt_ids, torch.Tensor)
        and isinstance(assignment_prob, torch.Tensor)
        and isinstance(pool_mask, torch.Tensor)
        and isinstance(prompt_node_x, torch.Tensor)
    ):
        return 0.0
    pool_idx = torch.where(pool_mask)[0]
    if pool_idx.numel() < 2:
        return 0.0
    top1 = top_prompt_ids[torch.arange(top_prompt_ids.size(0), device=top_prompt_ids.device), assignment_prob.argmax(dim=1)]
    pool_labels = labels[pool_idx]
    total_pairs = 0
    same_pairs = 0
    for prompt_id in range(int(prompt_node_x.size(0))):
        group_labels = pool_labels[top1 == prompt_id]
        group_size = int(group_labels.numel())
        if group_size < 2:
            continue
        for left in range(group_size):
            for right in range(left + 1, group_size):
                total_pairs += 1
                same_pairs += int(group_labels[left].item() == group_labels[right].item())
    if total_pairs == 0:
        return 0.0
    return float(same_pairs / total_pairs)


def _class_router_hit_rate(
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    class_slots = int(aux.get("num_class_prompt_slots", 0))
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor) and class_slots > 0):
        return 0.0
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
    selected = mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(selected.sum().item()) == 0:
        return 0.0
    y = labels.to(device=pool_mask.device)[pool_idx][selected]
    valid = (y >= 0) & (y < class_slots)
    if int(valid.sum().item()) == 0:
        return 0.0
    top1 = full_prob[selected].argmax(dim=-1)[valid]
    return float((top1 == y[valid]).float().mean().detach().item())


def _prompt_label_purity(prompt_out: dict[str, Any], labels: torch.Tensor, mask: torch.Tensor) -> float:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        return 0.0
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
    selected = mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(selected.sum().item()) < 2:
        return 0.0
    top1 = full_prob[selected].argmax(dim=-1)
    y = labels.to(device=pool_mask.device)[pool_idx][selected]
    total_pairs = 0
    same_pairs = 0
    for prompt_id in top1.unique().tolist():
        group_labels = y[top1 == int(prompt_id)]
        group_size = int(group_labels.numel())
        if group_size < 2:
            continue
        same = group_labels[:, None] == group_labels[None, :]
        pair_count = group_size * (group_size - 1) // 2
        total_pairs += pair_count
        same_pairs += int(torch.triu(same, diagonal=1).sum().item())
    if total_pairs == 0:
        return 0.0
    return float(same_pairs / total_pairs)


def _route_compactness_stats(prompt_out: dict[str, Any], labels: torch.Tensor, train_mask: torch.Tensor) -> dict[str, float]:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    pool_mask = prompt_out.get("pool_mask")
    if not (isinstance(full_prob, torch.Tensor) and isinstance(pool_mask, torch.Tensor)):
        return {"same_class_route_compactness": 0.0, "different_class_route_separation": 0.0}
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask)[0]
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) < 2:
        return {"same_class_route_compactness": 0.0, "different_class_route_separation": 0.0}
    probs = full_prob[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx][train_pool]
    dist = torch.cdist(probs, probs, p=2)
    same = y[:, None] == y[None, :]
    eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    same = same & ~eye
    diff = (~same) & ~eye
    return {
        "same_class_route_compactness": float(dist[same].mean().detach().item()) if bool(same.any()) else 0.0,
        "different_class_route_separation": float(dist[diff].mean().detach().item()) if bool(diff.any()) else 0.0,
    }


def _prompt_slot_usage_stats(
    usage: torch.Tensor | None,
    *,
    active_threshold: float = 0.05,
) -> dict[str, float]:
    if not isinstance(usage, torch.Tensor) or usage.numel() == 0:
        return {
            "dominant_prompt_slot_ratio": 0.0,
            "active_prompt_slot_count@0.05": 0.0,
        }
    detached = usage.detach()
    return {
        "dominant_prompt_slot_ratio": float(detached.max().item()),
        "active_prompt_slot_count@0.05": float((detached >= float(active_threshold)).sum().item()),
    }


def _pattern_usage_by_structural_bin(prompt_out: dict[str, Any], bins: int = 3) -> list[list[float]]:
    aux = prompt_out.get("aux", {})
    full_prob = aux.get("routing_full_prob")
    score = aux.get("structural_score")
    pool_idx = aux.get("pool_idx")
    if not (
        isinstance(full_prob, torch.Tensor)
        and isinstance(score, torch.Tensor)
        and isinstance(pool_idx, torch.Tensor)
        and full_prob.numel() > 0
        and pool_idx.numel() == full_prob.size(0)
    ):
        return []
    pool_score = score[pool_idx].detach()
    if pool_score.numel() == 0:
        return []
    ranked = torch.argsort(pool_score)
    chunks = torch.chunk(ranked, max(1, int(bins)))
    out: list[list[float]] = []
    for chunk in chunks:
        if chunk.numel() == 0:
            out.append([])
        else:
            out.append([float(value) for value in full_prob[chunk].mean(dim=0).detach().cpu().tolist()])
    return out


def _prompt_graph_diagnostics(
    prompt_out: dict[str, Any],
    *,
    z: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    test_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    pool_mask = prompt_out["pool_mask"].detach()
    aux = prompt_out.get("aux", {})
    prompt_edge_weight = aux.get("prompt_edge_weight")
    prompt_usage = aux.get("prompt_usage")
    prompt_usage_full = aux.get("prompt_usage_full")
    prompt_node_x = prompt_out.get("prompt_node_x")
    prompt_edge_count = int(prompt_out.get("prompt_edge_count", 0))
    prompt_node_count = int(prompt_node_x.size(0)) if isinstance(prompt_node_x, torch.Tensor) else 0
    edge_weight_mean = (
        float(prompt_edge_weight.detach().mean().item())
        if isinstance(prompt_edge_weight, torch.Tensor) and prompt_edge_weight.numel() > 0
        else 0.0
    )
    diagnostics: dict[str, Any] = {
        "prompt_node_count": prompt_node_count,
        "prompt_edge_count": prompt_edge_count,
        "pool_ratio": float(pool_mask.float().mean().item()),
        "train_pool_ratio": float((pool_mask & train_mask.bool()).float().mean().item()),
        "edge_scale": float(prompt_out["edge_scale"].detach().item()),
        "raw_edge_scale": (
            float(aux["raw_edge_scale"].detach().item())
            if isinstance(aux.get("raw_edge_scale"), torch.Tensor)
            else float(prompt_out["edge_scale"].detach().item())
        ),
        "edge_scale_multiplier": (
            float(aux["edge_scale_multiplier"].detach().item())
            if isinstance(aux.get("edge_scale_multiplier"), torch.Tensor)
            else 1.0
        ),
        "mean_prompt_edge_weight": edge_weight_mean,
        "pool_acceptance_mean": (
            float(aux["pool_acceptance_mean"].detach().item())
            if isinstance(aux.get("pool_acceptance_mean"), torch.Tensor)
            else 1.0
        ),
        "pool_acceptance_min": (
            float(aux["pool_acceptance_min"].detach().item())
            if isinstance(aux.get("pool_acceptance_min"), torch.Tensor)
            else 1.0
        ),
        "pool_acceptance_max": (
            float(aux["pool_acceptance_max"].detach().item())
            if isinstance(aux.get("pool_acceptance_max"), torch.Tensor)
            else 1.0
        ),
        "use_hard_acceptance": float(aux.get("use_hard_acceptance", 0)),
        "hard_acceptance_ratio": (
            float(aux["hard_acceptance_ratio"].detach().item())
            if isinstance(aux.get("hard_acceptance_ratio"), torch.Tensor)
            else 0.0
        ),
        "hard_acceptance_selected_ratio": (
            float(aux["hard_acceptance_selected_ratio"].detach().item())
            if isinstance(aux.get("hard_acceptance_selected_ratio"), torch.Tensor)
            else 0.0
        ),
        "prompt_usage_entropy": (
            float(aux["prompt_usage_entropy"].detach().item())
            if isinstance(aux.get("prompt_usage_entropy"), torch.Tensor)
            else 0.0
        ),
        "prompt_usage_full_entropy": (
            float(aux["prompt_usage_full_entropy"].detach().item())
            if isinstance(aux.get("prompt_usage_full_entropy"), torch.Tensor)
            else 0.0
        ),
        "capacity_routing_enabled": float(aux.get("capacity_routing_enabled", 0)),
        "prompt_capacity": float(aux.get("prompt_capacity", 0)),
        "capacity_overflow_count": float(aux.get("capacity_overflow_count", 0)),
        "adapted_node_count": int(prompt_out["adapted_x"].size(0)),
        "connected_edge_count": int(aux.get("connected_edge_count", prompt_edge_count)),
        "use_multiview_routing": float(aux.get("use_multiview_routing", 0)),
        "use_class_aware_routing": float(aux.get("use_class_aware_routing", 0)),
        "use_attribute_view": float(aux.get("use_attribute_view", 0)),
        "use_enhanced_role_view": float(aux.get("use_enhanced_role_view", 0)),
        "use_pattern_prompt_bank": float(aux.get("use_pattern_prompt_bank", 0)),
        "use_receiver_only_prompt": float(aux.get("use_receiver_only_prompt", 0)),
        "use_benefit_gate": float(aux.get("use_benefit_gate", 0)),
        "num_class_prompt_slots": float(aux.get("num_class_prompt_slots", 0)),
        "residual_prompt_count": float(aux.get("residual_prompt_count", 0)),
        "class_key_proto_init_coverage": float(aux.get("class_key_proto_init_coverage", 0.0)),
        "pattern_key_init_coverage": float(aux.get("pattern_key_init_coverage", 0.0)),
        "benefit_gate_mean": (
            float(aux["benefit_gate_mean"].detach().item())
            if isinstance(aux.get("benefit_gate_mean"), torch.Tensor)
            else 1.0
        ),
        "benefit_gate_min": (
            float(aux["benefit_gate_min"].detach().item())
            if isinstance(aux.get("benefit_gate_min"), torch.Tensor)
            else 1.0
        ),
        "benefit_gate_max": (
            float(aux["benefit_gate_max"].detach().item())
            if isinstance(aux.get("benefit_gate_max"), torch.Tensor)
            else 1.0
        ),
        "use_edge_utility": float(aux.get("use_edge_utility", 0)),
        "edge_utility_mean": (
            float(aux["edge_utility_mean"].detach().item())
            if isinstance(aux.get("edge_utility_mean"), torch.Tensor)
            else 1.0
        ),
        "edge_utility_min": (
            float(aux["edge_utility_min"].detach().item())
            if isinstance(aux.get("edge_utility_min"), torch.Tensor)
            else 1.0
        ),
        "edge_utility_max": (
            float(aux["edge_utility_max"].detach().item())
            if isinstance(aux.get("edge_utility_max"), torch.Tensor)
            else 1.0
        ),
        "pool_strategy_id": float(aux.get("pool_strategy_id", 0)),
        "pool_selected_ratio": (
            float(aux["pool_selected_ratio"].detach().item())
            if isinstance(aux.get("pool_selected_ratio"), torch.Tensor)
            else 0.0
        ),
        "pool_score_mean": (
            float(aux["pool_score_mean"].detach().item())
            if isinstance(aux.get("pool_score_mean"), torch.Tensor)
            else 0.0
        ),
        "pool_score_selected_mean": (
            float(aux["pool_score_selected_mean"].detach().item())
            if isinstance(aux.get("pool_score_selected_mean"), torch.Tensor)
            else 0.0
        ),
        "pool_uncertainty_mean": (
            float(aux["pool_uncertainty_component"].detach().mean().item())
            if isinstance(aux.get("pool_uncertainty_component"), torch.Tensor)
            else 0.0
        ),
        "pool_disagreement_mean": (
            float(aux["pool_disagreement_component"].detach().mean().item())
            if isinstance(aux.get("pool_disagreement_component"), torch.Tensor)
            else 0.0
        ),
        "use_utility_receive_gate": float(aux.get("use_utility_receive_gate", 0)),
        "utility_receive_gate_mean": (
            float(aux["utility_receive_gate_mean"].detach().item())
            if isinstance(aux.get("utility_receive_gate_mean"), torch.Tensor)
            else 1.0
        ),
        "utility_receive_gate_min": (
            float(aux["utility_receive_gate_min"].detach().item())
            if isinstance(aux.get("utility_receive_gate_min"), torch.Tensor)
            else 1.0
        ),
        "utility_receive_gate_max": (
            float(aux["utility_receive_gate_max"].detach().item())
            if isinstance(aux.get("utility_receive_gate_max"), torch.Tensor)
            else 1.0
        ),
        "utility_receive_gate_floor": (
            float(aux["utility_receive_gate_floor"].detach().item())
            if isinstance(aux.get("utility_receive_gate_floor"), torch.Tensor)
            else 0.0
        ),
        "use_hard_receive_gate": float(aux.get("use_hard_receive_gate", 0)),
        "hard_receive_ratio": (
            float(aux["hard_receive_ratio"].detach().item())
            if isinstance(aux.get("hard_receive_ratio"), torch.Tensor)
            else 0.0
        ),
        "hard_receive_selected_ratio": (
            float(aux["hard_receive_selected_ratio"].detach().item())
            if isinstance(aux.get("hard_receive_selected_ratio"), torch.Tensor)
            else 0.0
        ),
        "receive_gate_mean": (
            float(aux["receive_gate_mean"].detach().item())
            if isinstance(aux.get("receive_gate_mean"), torch.Tensor)
            else 1.0
        ),
        "receive_gate_min": (
            float(aux["receive_gate_min"].detach().item())
            if isinstance(aux.get("receive_gate_min"), torch.Tensor)
            else 1.0
        ),
        "receive_gate_max": (
            float(aux["receive_gate_max"].detach().item())
            if isinstance(aux.get("receive_gate_max"), torch.Tensor)
            else 1.0
        ),
        "view_gate_entropy": (
            float(aux["view_gate_entropy"].detach().item())
            if isinstance(aux.get("view_gate_entropy"), torch.Tensor)
            else 0.0
        ),
    }
    view_gate_mean = aux.get("view_gate_mean")
    if isinstance(view_gate_mean, torch.Tensor) and view_gate_mean.numel() in {3, 4}:
        view_values = [float(value) for value in view_gate_mean.detach().cpu().tolist()]
    else:
        view_values = [0.0, 1.0, 0.0]
    diagnostics["semantic_view_weight"] = view_values[0]
    diagnostics["structural_view_weight"] = view_values[1]
    diagnostics["role_view_weight"] = view_values[2]
    diagnostics["attribute_view_weight"] = view_values[3] if len(view_values) > 3 else 0.0
    for key, value in aux.items():
        key_str = str(key)
        if (
            not (key_str.startswith("p23_") or key_str.startswith("p24_"))
            or key_str.endswith("_tensor")
            or not isinstance(value, torch.Tensor)
        ):
            continue
        tensor = value.detach()
        if tensor.numel() == 1:
            diagnostics[key_str] = float(tensor.item())
        elif tensor.numel() > 0:
            diagnostics[f"{key}_mean"] = float(tensor.to(dtype=torch.float32).mean().item())
    for key in ["p23_top_helpful_feature_prompts", "p23_top_harmful_feature_prompts"]:
        value = aux.get(key)
        if isinstance(value, list):
            diagnostics[key] = value
    for name in ["semantic_route_margin", "structural_route_margin", "role_route_margin", "attribute_route_margin"]:
        value = aux.get(name)
        diagnostics[name] = float(value.detach().item()) if isinstance(value, torch.Tensor) else 0.0
    if isinstance(prompt_usage, torch.Tensor):
        diagnostics["prompt_usage_distribution"] = [float(value) for value in prompt_usage.detach().cpu().tolist()]
    else:
        diagnostics["prompt_usage_distribution"] = []
    if isinstance(prompt_usage_full, torch.Tensor):
        diagnostics["prompt_usage_full_distribution"] = [float(value) for value in prompt_usage_full.detach().cpu().tolist()]
    else:
        diagnostics["prompt_usage_full_distribution"] = []
    slot_usage = prompt_usage_full if isinstance(prompt_usage_full, torch.Tensor) else prompt_usage
    diagnostics.update(_prompt_slot_usage_stats(slot_usage))
    diagnostics["pattern_prompt_usage_entropy"] = diagnostics["prompt_usage_full_entropy"]
    diagnostics["pattern_prompt_usage_by_structural_bin"] = _pattern_usage_by_structural_bin(prompt_out)
    if labels is not None and num_classes is not None:
        diagnostics["pool_label_distribution"] = _pool_label_distribution(pool_mask, labels, int(num_classes))
        diagnostics["prompt_usage_by_true_class"] = _prompt_usage_by_true_class(prompt_out, labels, int(num_classes))
        diagnostics["prompt_mediated_same_label_reachability"] = _prompt_mediated_same_label_reachability(prompt_out, labels)
        diagnostics["class_router_hit_rate_train"] = _class_router_hit_rate(prompt_out, labels, train_mask)
        diagnostics["prompt_label_purity_train"] = _prompt_label_purity(prompt_out, labels, train_mask)
        diagnostics["residual_prompt_usage_ratio"] = sum(
            diagnostics.get("prompt_usage_full_distribution", [])[int(aux.get("num_class_prompt_slots", 0)) :]
        )
        diagnostics.update(_route_compactness_stats(prompt_out, labels, train_mask))
        if val_mask is not None:
            diagnostics["class_router_hit_rate_val"] = _class_router_hit_rate(prompt_out, labels, val_mask)
        if test_mask is not None:
            diagnostics["class_router_hit_rate_test"] = _class_router_hit_rate(prompt_out, labels, test_mask)
    edge_type_counts = aux.get("edge_type_counts")
    if isinstance(edge_type_counts, list):
        diagnostics["edge_type_counts"] = [int(value) for value in edge_type_counts]
    else:
        edge_type = prompt_out.get("adapted_edge_type")
        if isinstance(edge_type, torch.Tensor):
            diagnostics["edge_type_counts"] = [
                int((edge_type == type_id).sum().item()) for type_id in range(3)
            ]
        else:
            diagnostics["edge_type_counts"] = [int(prompt_out["adapted_edge_index"].size(1)), 0, 0]
    return diagnostics


def _prompt_aware_diagnostics(model_out: dict[str, Any]) -> dict[str, Any]:
    aux = model_out.get("prompt_aware", {})
    if not isinstance(aux, dict):
        aux = {}

    def scalar(name: str) -> float:
        value = aux.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().item())
        return 0.0

    gates = aux.get("prompt_gate_by_layer")
    if isinstance(gates, torch.Tensor):
        gate_by_layer = [
            [float(item) for item in row]
            for row in gates.detach().cpu().tolist()
        ]
    else:
        gate_by_layer = []
    return {
        "prompt_msg_norm": scalar("prompt_msg_norm"),
        "prompt_to_original_msg_norm": scalar("prompt_to_original_msg_norm"),
        "node_to_prompt_msg_norm": scalar("node_to_prompt_msg_norm"),
        "prompt_to_original_update_norm": scalar("prompt_to_original_update_norm"),
        "correction_norm": scalar("prompt_to_original_update_norm"),
        "node_to_prompt_update_norm": scalar("node_to_prompt_update_norm"),
        "raw_prompt_update_norm": scalar("raw_prompt_update_norm"),
        "unbounded_prompt_update_norm": scalar("unbounded_prompt_update_norm"),
        "prompt_update_clip_ratio": scalar("prompt_update_clip_ratio"),
        "bounded_correction_clip_ratio": scalar("prompt_update_clip_ratio"),
        "prompt_gate_mean": scalar("prompt_gate_mean"),
        "prompt_gate_node_to_prompt_mean": scalar("prompt_gate_node_to_prompt_mean"),
        "prompt_gate_prompt_to_node_mean": scalar("prompt_gate_prompt_to_node_mean"),
        "prompt_gate_by_layer": gate_by_layer,
        "adapted_branch_delta_norm": scalar("adapted_branch_delta_norm"),
        "prompt_message_scale": scalar("prompt_message_scale"),
        "prompt_message_norm": str(aux.get("prompt_message_norm", "")),
        "bounded_prompt_update": scalar("bounded_prompt_update"),
        "max_prompt_update_norm": scalar("max_prompt_update_norm"),
        "prompt_update_bound_mode": str(aux.get("prompt_update_bound_mode", "")),
        "receiver_version": str(aux.get("receiver_version", "")),
        "prompt_fusion": str(aux.get("prompt_fusion", "")),
        "prompt_slot_head_count": scalar("prompt_slot_head_count"),
        "prompt_receiver_gate_mean": scalar("prompt_receiver_gate_mean"),
        "prototype_direction_strength_mean": scalar("prototype_direction_strength_mean"),
        "prototype_direction_normalize": scalar("prototype_direction_normalize"),
        "pool_only_prompt_update": scalar("pool_only_prompt_update"),
        "zero_init_prompt_messages": scalar("zero_init_prompt_messages"),
    }


def _attach_p23_ce_delta_diagnostics(
    *,
    model_out: dict[str, Any],
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    test_mask: torch.Tensor | None = None,
) -> None:
    base_logits = model_out.get("p23_no_receiver_logits")
    prompt_logits = model_out.get("logits")
    if not isinstance(base_logits, torch.Tensor) or not isinstance(prompt_logits, torch.Tensor):
        return
    aux = prompt_out.setdefault("aux", {})
    with torch.no_grad():
        y = labels.to(device=prompt_logits.device, dtype=torch.long)
        ce_base = F.cross_entropy(base_logits.to(device=prompt_logits.device), y, reduction="none")
        ce_prompt = F.cross_entropy(prompt_logits.detach(), y, reduction="none")
        delta = ce_base - ce_prompt
    pool = prompt_out.get("pool_mask")
    pool_mask = (
        pool.to(device=prompt_logits.device, dtype=torch.bool)
        if isinstance(pool, torch.Tensor)
        else torch.zeros(delta.size(0), dtype=torch.bool, device=prompt_logits.device)
    )

    def put(name: str, mask: torch.Tensor | None) -> None:
        if mask is None:
            return
        m = mask.to(device=prompt_logits.device, dtype=torch.bool)
        if not bool(m.any()):
            aux[f"p23_{name}_delta_ce_mean"] = delta.new_tensor(0.0)
            aux[f"p23_{name}_positive_delta_ratio"] = delta.new_tensor(0.0)
            return
        values = delta[m]
        aux[f"p23_{name}_delta_ce_mean"] = values.mean()
        aux[f"p23_{name}_positive_delta_ratio"] = (values > 0).to(dtype=delta.dtype).mean()

    put("train", train_mask)
    put("val", val_mask)
    put("test", test_mask)
    put("train_pool", train_mask.to(device=prompt_logits.device, dtype=torch.bool) & pool_mask)
    if val_mask is not None:
        put("val_pool", val_mask.to(device=prompt_logits.device, dtype=torch.bool) & pool_mask)
    if test_mask is not None:
        put("test_pool", test_mask.to(device=prompt_logits.device, dtype=torch.bool) & pool_mask)
    put("pool", pool_mask)
    put("nonpool", ~pool_mask)
    aux["p23_ce_delta_train_mean"] = aux.get("p23_train_delta_ce_mean", delta.new_tensor(0.0))
    aux["p23_ce_delta_pool_mean"] = aux.get("p23_pool_delta_ce_mean", delta.new_tensor(0.0))

    receiver = model_out.get("p23_receiver", {})
    node_gate = receiver.get("node_receive_gate") if isinstance(receiver, dict) else None
    if isinstance(node_gate, torch.Tensor):
        gate = node_gate.detach().to(device=delta.device, dtype=delta.dtype)

        def corr(name: str, mask: torch.Tensor | None) -> None:
            if mask is None:
                return
            m = mask.to(device=delta.device, dtype=torch.bool)
            aux[f"p23_node_gate_delta_corr_{name}"] = delta.new_tensor(_pearson_corr(gate[m], delta[m]) if bool(m.any()) else 0.0)

        corr("train", train_mask)
        corr("val", val_mask)
        corr("test", test_mask)

    prompt_edge_index = aux.get("p23_prompt_edge_index_tensor")
    feature_ids = aux.get("p23_feature_ids_tensor")
    if isinstance(prompt_edge_index, torch.Tensor) and isinstance(feature_ids, torch.Tensor) and prompt_edge_index.numel() > 0:
        topk = int(aux.get("p23_log_topk_feature_prompt_delta", 20))
        if topk > 0:
            edge_prompt = prompt_edge_index[0].to(device=delta.device, dtype=torch.long)
            edge_node = prompt_edge_index[1].to(device=delta.device, dtype=torch.long)
            edge_delta = delta[edge_node]
            prompt_count = int(feature_ids.numel())
            counts = torch.bincount(edge_prompt, minlength=prompt_count).to(device=delta.device, dtype=delta.dtype)
            sums = delta.new_zeros(prompt_count)
            sums.index_add_(0, edge_prompt, edge_delta)
            mean_delta = sums / counts.clamp_min(1.0)
            valid = counts > 0
            if bool(valid.any()):
                hub_gate = receiver.get("hub_gate") if isinstance(receiver, dict) else None
                hub = hub_gate.detach().to(device=delta.device, dtype=delta.dtype) if isinstance(hub_gate, torch.Tensor) else delta.new_zeros(prompt_count)
                df_global = aux.get("p23_df_global_tensor", delta.new_zeros(prompt_count)).to(device=delta.device, dtype=delta.dtype)
                df_pool = aux.get("p23_df_pool_tensor", delta.new_zeros(prompt_count)).to(device=delta.device, dtype=delta.dtype)
                pool_rel = aux.get("p23_pool_rel_tensor", delta.new_zeros(prompt_count)).to(device=delta.device, dtype=delta.dtype)
                static_rel = aux.get("p23_static_reliability_tensor", delta.new_zeros(prompt_count)).to(device=delta.device, dtype=delta.dtype)
                feature_ids_dev = feature_ids.to(device=delta.device, dtype=torch.long)

                def records(indices: torch.Tensor) -> list[dict[str, float | int]]:
                    out: list[dict[str, float | int]] = []
                    for idx in indices.detach().cpu().tolist():
                        local = int(idx)
                        out.append(
                            {
                                "feature_id": int(feature_ids_dev[local].detach().cpu().item()),
                                "mean_delta_ce": float(mean_delta[local].detach().cpu().item()),
                                "count": int(counts[local].detach().cpu().item()),
                                "hub_gate": float(hub[local].detach().cpu().item()) if hub.numel() > local else 0.0,
                                "df_global": float(df_global[local].detach().cpu().item()) if df_global.numel() > local else 0.0,
                                "df_pool": float(df_pool[local].detach().cpu().item()) if df_pool.numel() > local else 0.0,
                                "pool_rel": float(pool_rel[local].detach().cpu().item()) if pool_rel.numel() > local else 0.0,
                                "static_reliability": float(static_rel[local].detach().cpu().item()) if static_rel.numel() > local else 0.0,
                            }
                        )
                    return out

                valid_idx = torch.where(valid)[0]
                k = min(topk, int(valid_idx.numel()))
                helpful = valid_idx[torch.topk(mean_delta[valid_idx], k=k, largest=True).indices]
                harmful = valid_idx[torch.topk(mean_delta[valid_idx], k=k, largest=False).indices]
                aux["p23_top_helpful_feature_prompts"] = records(helpful)
                aux["p23_top_harmful_feature_prompts"] = records(harmful)


def _p23_node_gate_utility_loss(
    *,
    model_out: dict[str, Any],
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    positive_margin: float,
    negative_margin: float,
    class_balanced: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt_logits = model_out.get("logits")
    base_logits = model_out.get("p23_no_receiver_logits")
    receiver = model_out.get("p23_receiver", {})
    node_gate = receiver.get("node_receive_gate") if isinstance(receiver, dict) else None
    if not isinstance(prompt_logits, torch.Tensor) or not isinstance(base_logits, torch.Tensor) or not isinstance(node_gate, torch.Tensor):
        fallback = labels.new_tensor(0.0, dtype=torch.float32)
        return fallback, {
            "p23_node_gate_utility_loss": 0.0,
            "p23_node_gate_utility_count": 0.0,
            "p23_node_gate_positive_count": 0.0,
            "p23_node_gate_negative_count": 0.0,
            "p23_node_gate_target_mean": 0.0,
            "p23_node_gate_delta_corr_train": 0.0,
        }

    device = prompt_logits.device
    y = labels.to(device=device, dtype=torch.long)
    mask = train_mask.to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        return prompt_logits.new_tensor(0.0), {
            "p23_node_gate_utility_loss": 0.0,
            "p23_node_gate_utility_count": 0.0,
            "p23_node_gate_positive_count": 0.0,
            "p23_node_gate_negative_count": 0.0,
            "p23_node_gate_target_mean": 0.0,
            "p23_node_gate_delta_corr_train": 0.0,
        }

    with torch.no_grad():
        ce_base = F.cross_entropy(base_logits.to(device=device), y, reduction="none")
        ce_prompt = F.cross_entropy(prompt_logits.detach(), y, reduction="none")
        delta = ce_base - ce_prompt
        positive = delta > float(positive_margin)
        negative = delta < float(negative_margin)
        supervised = mask & (positive | negative)
        target = positive.to(dtype=prompt_logits.dtype)

    if not bool(supervised.any()):
        return prompt_logits.new_tensor(0.0), {
            "p23_node_gate_utility_loss": 0.0,
            "p23_node_gate_utility_count": 0.0,
            "p23_node_gate_positive_count": 0.0,
            "p23_node_gate_negative_count": 0.0,
            "p23_node_gate_target_mean": 0.0,
            "p23_node_gate_delta_corr_train": _pearson_corr(node_gate.detach()[mask], delta[mask]),
        }

    gate = node_gate.to(device=device, dtype=prompt_logits.dtype).clamp(1e-6, 1.0 - 1e-6)
    losses = F.binary_cross_entropy(gate[supervised], target[supervised], reduction="none")
    if class_balanced:
        pos_mask = supervised & positive
        neg_mask = supervised & negative
        parts = []
        if bool(pos_mask.any()):
            parts.append(F.binary_cross_entropy(gate[pos_mask], target[pos_mask], reduction="mean"))
        if bool(neg_mask.any()):
            parts.append(F.binary_cross_entropy(gate[neg_mask], target[neg_mask], reduction="mean"))
        loss = torch.stack(parts).mean() if parts else losses.mean()
    else:
        loss = losses.mean()

    aux = prompt_out.setdefault("aux", {})
    aux["p23_node_gate_utility_loss"] = loss.detach()
    aux["p23_node_gate_utility_count"] = prompt_logits.new_tensor(float(supervised.sum().item()))
    aux["p23_node_gate_positive_count"] = prompt_logits.new_tensor(float((supervised & positive).sum().item()))
    aux["p23_node_gate_negative_count"] = prompt_logits.new_tensor(float((supervised & negative).sum().item()))
    aux["p23_node_gate_target_mean"] = target[supervised].mean().detach()
    aux["p23_node_gate_delta_corr_train"] = prompt_logits.new_tensor(_pearson_corr(node_gate.detach()[mask], delta[mask]))
    return loss, {
        "p23_node_gate_utility_loss": float(loss.detach().item()),
        "p23_node_gate_utility_count": float(supervised.sum().item()),
        "p23_node_gate_positive_count": float((supervised & positive).sum().item()),
        "p23_node_gate_negative_count": float((supervised & negative).sum().item()),
        "p23_node_gate_target_mean": float(target[supervised].mean().detach().item()),
        "p23_node_gate_delta_corr_train": _pearson_corr(node_gate.detach()[mask], delta[mask]),
    }


def _p24_route_utility_loss(
    *,
    model_out: dict[str, Any],
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    adapt_positive_margin: float,
    adapt_negative_margin: float,
    no_prompt_positive_margin: float,
    no_prompt_negative_margin: float,
    class_balanced: bool,
    supervise_no_prompt: bool,
    target_mode: str = "hard",
    gain_scale: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    static_logits = model_out.get("p24_static_logits")
    candidate_logits = model_out.get("p24_candidate_logits")
    no_prompt_logits = model_out.get("p24_no_prompt_logits")
    route_logits = prompt_out.get("aux", {}).get("p24_route_logits")
    tensors = (static_logits, candidate_logits, no_prompt_logits, route_logits)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        fallback = labels.new_tensor(0.0, dtype=torch.float32)
        return fallback, {
            "p24_route_utility_loss": 0.0,
            "p24_route_utility_count": 0.0,
            "p24_route_adapt_positive_ratio": 0.0,
            "p24_route_no_prompt_positive_ratio": 0.0,
            "p24_candidate_gain_mean": 0.0,
            "p24_no_prompt_gain_mean": 0.0,
            "p24_route_target_mean": 0.0,
        }

    device = route_logits.device
    y = labels.to(device=device, dtype=torch.long)
    mask = train_mask.to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        return route_logits.sum() * 0.0, {
            "p24_route_utility_loss": 0.0,
            "p24_route_utility_count": 0.0,
            "p24_route_adapt_positive_ratio": 0.0,
            "p24_route_no_prompt_positive_ratio": 0.0,
            "p24_candidate_gain_mean": 0.0,
            "p24_no_prompt_gain_mean": 0.0,
            "p24_route_target_mean": 0.0,
        }

    with torch.no_grad():
        ce_static = F.cross_entropy(static_logits.detach(), y, reduction="none")
        ce_candidate = F.cross_entropy(candidate_logits.detach(), y, reduction="none")
        ce_no_prompt = F.cross_entropy(no_prompt_logits.detach(), y, reduction="none")
        candidate_gain = ce_static - ce_candidate
        no_prompt_gain = torch.minimum(ce_static, ce_candidate) - ce_no_prompt
        gains = torch.stack([candidate_gain, no_prompt_gain], dim=-1)
        positive_margin = gains.new_tensor(
            [float(adapt_positive_margin), float(no_prompt_positive_margin)]
        )
        negative_margin = gains.new_tensor(
            [abs(float(adapt_negative_margin)), abs(float(no_prompt_negative_margin))]
        )
        positive = gains > positive_margin.unsqueeze(0)
        negative = gains < -negative_margin.unsqueeze(0)
        supervised = mask.unsqueeze(-1) & (positive | negative)
        if not supervise_no_prompt:
            supervised[:, 1] = False
        if str(target_mode).lower() == "gain_scaled":
            targets = ((gains - positive_margin.unsqueeze(0)) / max(float(gain_scale), 1e-6)).clamp(0.0, 1.0)
            targets = torch.where(positive, targets, torch.zeros_like(targets))
        else:
            targets = positive.to(dtype=route_logits.dtype)

    if not bool(supervised.any()):
        return route_logits.sum() * 0.0, {
            "p24_route_utility_loss": 0.0,
            "p24_route_utility_count": 0.0,
            "p24_route_adapt_positive_ratio": 0.0,
            "p24_route_no_prompt_positive_ratio": 0.0,
            "p24_candidate_gain_mean": float(candidate_gain[mask].mean().item()),
            "p24_no_prompt_gain_mean": float(no_prompt_gain[mask].mean().item()),
            "p24_route_target_mean": 0.0,
        }

    per_entry = F.binary_cross_entropy_with_logits(route_logits, targets, reduction="none")
    if class_balanced:
        parts: list[torch.Tensor] = []
        for class_id in torch.unique(y[mask]):
            class_mask = mask & (y == class_id)
            for route_index in range(2):
                selected = class_mask & supervised[:, route_index]
                if bool(selected.any()):
                    parts.append(per_entry[selected, route_index].mean())
        loss = torch.stack(parts).mean() if parts else per_entry[supervised].mean()
    else:
        loss = per_entry[supervised].mean()

    adapt_selected = supervised[:, 0]
    none_selected = supervised[:, 1]
    return loss, {
        "p24_route_utility_loss": float(loss.detach().item()),
        "p24_route_utility_count": float(supervised.sum().item()),
        "p24_route_adapt_positive_ratio": float(
            positive[adapt_selected, 0].float().mean().item() if bool(adapt_selected.any()) else 0.0
        ),
        "p24_route_no_prompt_positive_ratio": float(
            positive[none_selected, 1].float().mean().item() if bool(none_selected.any()) else 0.0
        ),
        "p24_candidate_gain_mean": float(candidate_gain[mask].mean().item()),
        "p24_no_prompt_gain_mean": float(no_prompt_gain[mask].mean().item()),
        "p24_route_target_mean": float(targets[supervised].mean().item()),
    }


def _p24_counterfactual_route_loss(
    *,
    model_out: dict[str, Any],
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    query_mask: torch.Tensor,
    gain_ema: torch.Tensor,
    ema_decay: float,
    gain_cap: float,
    gain_margin: float,
    harm_weight: float,
    rank_weight: float,
    budget_weight: float,
    positive_adapt_floor: float,
    class_balanced: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    static_logits = model_out.get("p24_static_logits")
    candidate_logits = model_out.get("p24_candidate_logits")
    aux = prompt_out.get("aux", {})
    predicted_gain = aux.get("p24_counterfactual_predicted_gain")
    adapt_ratio = aux.get("p24_route_adapt")
    tensors = (static_logits, candidate_logits, predicted_gain, adapt_ratio)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        fallback = labels.new_tensor(0.0, dtype=torch.float32)
        return fallback, {
            "p24_counterfactual_route_loss": 0.0,
            "p24_counterfactual_route_count": 0.0,
            "p24_counterfactual_true_gain_mean": 0.0,
            "p24_counterfactual_gain_corr": 0.0,
            "p24_counterfactual_harm_loss": 0.0,
            "p24_counterfactual_rank_loss": 0.0,
            "p24_counterfactual_budget_loss": 0.0,
        }

    device = predicted_gain.device
    y = labels.to(device=device, dtype=torch.long)
    mask = query_mask.to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        return predicted_gain.sum() * 0.0, {
            "p24_counterfactual_route_loss": 0.0,
            "p24_counterfactual_route_count": 0.0,
            "p24_counterfactual_true_gain_mean": 0.0,
            "p24_counterfactual_gain_corr": 0.0,
            "p24_counterfactual_harm_loss": 0.0,
            "p24_counterfactual_rank_loss": 0.0,
            "p24_counterfactual_budget_loss": 0.0,
        }

    with torch.no_grad():
        static_ce = F.cross_entropy(static_logits.detach(), y, reduction="none")
        candidate_ce = F.cross_entropy(candidate_logits.detach(), y, reduction="none")
        current_gain = (static_ce - candidate_ce).clamp(-abs(float(gain_cap)), abs(float(gain_cap)))
        old = gain_ema[mask]
        initialized = torch.isfinite(old)
        updated = torch.where(
            initialized,
            float(ema_decay) * old + (1.0 - float(ema_decay)) * current_gain[mask],
            current_gain[mask],
        )
        gain_ema[mask] = updated
        target = gain_ema.detach()

    target_for_loss = torch.where(torch.isfinite(target), target, predicted_gain.detach())
    per_node = F.smooth_l1_loss(predicted_gain, target_for_loss, reduction="none")
    if class_balanced:
        parts = [per_node[mask & (y == class_id)].mean() for class_id in torch.unique(y[mask])]
        regression = torch.stack(parts).mean() if parts else per_node[mask].mean()
    else:
        regression = per_node[mask].mean()

    positive = mask & (target > abs(float(gain_margin)))
    harmful = mask & (target < -abs(float(gain_margin)))
    harm_loss = F.relu(predicted_gain[harmful]).mean() if bool(harmful.any()) else predicted_gain.sum() * 0.0
    if bool(positive.any()) and bool(harmful.any()):
        rank_loss = F.relu(
            predicted_gain.new_tensor(abs(float(gain_margin)))
            - (predicted_gain[positive].mean() - predicted_gain[harmful].mean())
        )
    else:
        rank_loss = predicted_gain.sum() * 0.0
    budget_loss = (
        F.relu(adapt_ratio.new_tensor(float(positive_adapt_floor)) - adapt_ratio[positive].mean())
        if bool(positive.any())
        else predicted_gain.sum() * 0.0
    )
    total = (
        regression
        + float(harm_weight) * harm_loss
        + float(rank_weight) * rank_loss
        + float(budget_weight) * budget_loss
    )
    with torch.no_grad():
        pred = predicted_gain[mask]
        truth = target[mask]
        pred_centered = pred - pred.mean()
        truth_centered = truth - truth.mean()
        corr = (pred_centered * truth_centered).sum() / (
            pred_centered.norm() * truth_centered.norm()
        ).clamp_min(1e-12)
    return total, {
        "p24_counterfactual_route_loss": float(total.detach().item()),
        "p24_counterfactual_route_count": float(mask.sum().item()),
        "p24_counterfactual_true_gain_mean": float(target[mask].mean().item()),
        "p24_counterfactual_true_gain_std": float(target[mask].std(unbiased=False).item()),
        "p24_counterfactual_positive_ratio": float(positive[mask].float().mean().item()),
        "p24_counterfactual_harmful_ratio": float(harmful[mask].float().mean().item()),
        "p24_counterfactual_gain_corr": float(corr.item()),
        "p24_counterfactual_harm_loss": float(harm_loss.detach().item()),
        "p24_counterfactual_rank_loss": float(rank_loss.detach().item()),
        "p24_counterfactual_budget_loss": float(budget_loss.detach().item()),
        "p24_counterfactual_positive_adapt_mean": float(
            adapt_ratio[positive].detach().mean().item() if bool(positive.any()) else 0.0
        ),
        "p24_counterfactual_harmful_adapt_mean": float(
            adapt_ratio[harmful].detach().mean().item() if bool(harmful.any()) else 0.0
        ),
    }


def _p24_candidate_anti_harm_loss(
    *,
    model_out: dict[str, Any],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float,
    class_balanced: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    static_logits = model_out.get("p24_static_logits")
    candidate_logits = model_out.get("p24_candidate_logits")
    if not isinstance(static_logits, torch.Tensor) or not isinstance(candidate_logits, torch.Tensor):
        fallback = labels.new_tensor(0.0, dtype=torch.float32)
        return fallback, {
            "p24_candidate_anti_harm_loss": 0.0,
            "p24_candidate_gain_mean": 0.0,
            "p24_candidate_positive_ratio": 0.0,
        }
    device = candidate_logits.device
    y = labels.to(device=device, dtype=torch.long)
    mask = train_mask.to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        return candidate_logits.sum() * 0.0, {
            "p24_candidate_anti_harm_loss": 0.0,
            "p24_candidate_gain_mean": 0.0,
            "p24_candidate_positive_ratio": 0.0,
        }
    ce_static = F.cross_entropy(static_logits.detach(), y, reduction="none")
    ce_candidate = F.cross_entropy(candidate_logits, y, reduction="none")
    per_node = F.relu(ce_candidate - ce_static + float(margin))
    if class_balanced:
        parts = [per_node[mask & (y == class_id)].mean() for class_id in torch.unique(y[mask])]
        loss = torch.stack(parts).mean() if parts else per_node[mask].mean()
    else:
        loss = per_node[mask].mean()
    with torch.no_grad():
        gain = ce_static - ce_candidate.detach()
    return loss, {
        "p24_candidate_anti_harm_loss": float(loss.detach().item()),
        "p24_candidate_gain_mean": float(gain[mask].mean().item()),
        "p24_candidate_positive_ratio": float((gain[mask] > 0.0).float().mean().item()),
    }


def _p24_prompt_utility_loss(
    *,
    model_out: dict[str, Any],
    prompt_out: dict[str, Any],
    labels: torch.Tensor,
    query_mask: torch.Tensor,
    shrinkage_tau: float,
    gain_scale: float,
    target_log_scale: float,
    hub_quantile: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit prompt gates to held-out candidate utility with degree shrinkage."""

    aux = prompt_out.get("aux", {})
    static_logits = model_out.get("p24_static_logits")
    candidate_logits = model_out.get("p24_candidate_logits")
    prompt_gate = aux.get("p24_prompt_gate")
    prompt_keep_logit = aux.get("p24_gumbel_prompt_keep_logit")
    edge_node = aux.get("p23_selective_edge_node_index")
    edge_feature = aux.get("p23_selective_edge_feature_local")
    edge_direction = aux.get("p23_selective_edge_direction")
    edge_weight = aux.get("p24_static_prompt_edge_weight")
    tensors = (
        static_logits,
        candidate_logits,
        prompt_gate,
        edge_node,
        edge_feature,
        edge_direction,
        edge_weight,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        fallback = labels.new_tensor(0.0, dtype=torch.float32)
        return fallback, {
            "p24_prompt_utility_loss": 0.0,
            "p24_prompt_utility_count": 0.0,
            "p24_prompt_utility_mean": 0.0,
            "p24_prompt_utility_reliability_mean": 0.0,
            "p24_noisy_hub_prompt_ratio": 0.0,
            "p24_noisy_hub_prompt_gate_mean": 1.0,
            "p24_reliable_prompt_gate_mean": 1.0,
        }

    device = candidate_logits.device
    y = labels.to(device=device, dtype=torch.long)
    query = query_mask.to(device=device, dtype=torch.bool)
    node = edge_node.to(device=device, dtype=torch.long)
    feature = edge_feature.to(device=device, dtype=torch.long)
    direction = edge_direction.to(device=device, dtype=torch.long)
    static_weight = edge_weight.to(device=device, dtype=candidate_logits.dtype).detach().clamp_min(0.0)
    num_prompts = int(prompt_gate.numel())
    selected = (direction == 1) & query[node]
    if not bool(selected.any()) or num_prompts == 0:
        return prompt_gate.sum() * 0.0, {
            "p24_prompt_utility_loss": 0.0,
            "p24_prompt_utility_count": 0.0,
            "p24_prompt_utility_mean": 0.0,
            "p24_prompt_utility_reliability_mean": 0.0,
            "p24_noisy_hub_prompt_ratio": 0.0,
            "p24_noisy_hub_prompt_gate_mean": 1.0,
            "p24_reliable_prompt_gate_mean": 1.0,
        }

    with torch.no_grad():
        ce_static = F.cross_entropy(static_logits.detach(), y, reduction="none")
        ce_candidate = F.cross_entropy(candidate_logits.detach(), y, reduction="none")
        node_gain = ce_static - ce_candidate
        selected_feature = feature[selected]
        selected_weight = static_weight[selected]
        weighted_count = candidate_logits.new_zeros(num_prompts)
        weighted_gain = candidate_logits.new_zeros(num_prompts)
        edge_count = candidate_logits.new_zeros(num_prompts)
        weighted_count.index_add_(0, selected_feature, selected_weight)
        weighted_gain.index_add_(0, selected_feature, selected_weight * node_gain[node[selected]])
        edge_count.index_add_(0, selected_feature, torch.ones_like(selected_weight))
        utility = weighted_gain / weighted_count.clamp_min(1e-12)
        reliability = edge_count / (edge_count + max(float(shrinkage_tau), 0.0))
        if isinstance(prompt_keep_logit, torch.Tensor):
            harmful_strength = torch.sigmoid(-utility / max(float(gain_scale), 1e-6)) * reliability
            target = 1.0 - min(max(float(target_log_scale), 0.0), 1.0) * harmful_strength
        else:
            target = (
                (utility / max(float(gain_scale), 1e-6)).clamp(-1.0, 1.0)
                * reliability
                * float(target_log_scale)
            )
        supervised = edge_count > 0.0

    if isinstance(prompt_keep_logit, torch.Tensor):
        predicted = prompt_keep_logit
        per_prompt = F.binary_cross_entropy_with_logits(predicted, target, reduction="none")
    else:
        predicted = prompt_gate.clamp_min(1e-8).log()
        per_prompt = F.smooth_l1_loss(predicted, target, reduction="none")
    loss = (per_prompt[supervised] * reliability[supervised]).sum() / reliability[supervised].sum().clamp_min(1e-12)

    usage = candidate_logits.new_zeros(num_prompts)
    all_p2n = direction == 1
    usage.index_add_(0, feature[all_p2n], torch.ones_like(feature[all_p2n], dtype=candidate_logits.dtype))
    used = usage > 0.0
    hub_threshold = torch.quantile(usage[used], min(max(float(hub_quantile), 0.0), 1.0)) if bool(used.any()) else usage.new_tensor(0.0)
    hub = used & (usage >= hub_threshold)
    noisy_hub = hub & supervised & (utility < 0.0)
    reliable_prompt = supervised & (utility > 0.0) & (reliability >= 0.50)
    return loss, {
        "p24_prompt_utility_loss": float(loss.detach().item()),
        "p24_prompt_utility_count": float(supervised.sum().item()),
        "p24_prompt_utility_mean": float(utility[supervised].mean().item()),
        "p24_prompt_utility_reliability_mean": float(reliability[supervised].mean().item()),
        "p24_noisy_hub_prompt_ratio": float(noisy_hub.float().mean().item()),
        "p24_noisy_hub_prompt_gate_mean": float(
            prompt_gate.detach()[noisy_hub].mean().item() if bool(noisy_hub.any()) else 1.0
        ),
        "p24_reliable_prompt_gate_mean": float(
            prompt_gate.detach()[reliable_prompt].mean().item() if bool(reliable_prompt.any()) else 1.0
        ),
    }


def _p24_causal_prompt_teacher_loss(
    *,
    model: DualBranchFramework,
    model_out: dict[str, Any],
    prompt_out: dict[str, Any],
    z: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    query_mask: torch.Tensor,
    state: dict[str, torch.Tensor] | None,
    epoch: int,
    update_interval: int,
    warmup_epochs: int,
    max_prompts_per_update: int,
    candidate_pool_size: int,
    probe_strength: float,
    ema_decay: float,
    utility_margin: float,
    target_temperature: float,
    max_prune_target: float,
    min_observations: int,
    harmful_stability_threshold: float = 0.70,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Supervise prompt keep logits with isolated held-out ablations."""

    aux = prompt_out.get("aux", {})
    prompt_logit = aux.get("p24_gumbel_prompt_keep_logit")
    static_logits = model_out.get("p24_static_logits")
    static_weight = aux.get("p24_static_prompt_edge_weight")
    edge_node = aux.get("p23_selective_edge_node_index")
    edge_feature = aux.get("p23_selective_edge_feature_local")
    edge_direction = aux.get("p23_selective_edge_direction")
    prunable_prior = aux.get("p24_gumbel_prompt_prunable_prior")
    tensors = (prompt_logit, static_logits, static_weight, edge_node, edge_feature, edge_direction)
    fallback = z.new_tensor(0.0)
    empty = {
        "p24_causal_prompt_teacher_loss": 0.0,
        "p24_causal_prompt_teacher_count": 0.0,
        "p24_causal_prompt_teacher_updated": 0.0,
        "p24_causal_prompt_removal_gain_mean": 0.0,
        "p24_causal_prompt_harmful_ratio": 0.0,
        "p24_causal_prompt_harmful_stability_mean": 0.0,
        "p24_causal_prompt_target_keep_mean": 1.0,
        "p24_causal_prompt_harmful_keep_mean": 1.0,
        "p24_causal_prompt_helpful_keep_mean": 1.0,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return fallback, state or {}, empty
    prompt_count = int(prompt_logit.numel())
    if prompt_count == 0:
        return prompt_logit.sum() * 0.0, state or {}, empty
    if state is None or state.get("ema", fallback).numel() != prompt_count:
        state = {
            "ema": prompt_logit.new_full((prompt_count,), float("nan")),
            "seen": prompt_logit.new_zeros(prompt_count),
            "harmful_seen": prompt_logit.new_zeros(prompt_count),
        }
    elif "harmful_seen" not in state:
        state["harmful_seen"] = prompt_logit.new_zeros(prompt_count)

    device = prompt_logit.device
    node = edge_node.to(device=device, dtype=torch.long)
    feature = edge_feature.to(device=device, dtype=torch.long)
    direction = edge_direction.to(device=device, dtype=torch.long)
    query = query_mask.to(device=device, dtype=torch.bool)
    y = labels.to(device=device, dtype=torch.long)
    p2n = direction == 1
    usage = prompt_logit.new_zeros(prompt_count)
    usage.index_add_(0, feature[p2n], torch.ones_like(feature[p2n], dtype=prompt_logit.dtype))
    if isinstance(prunable_prior, torch.Tensor) and prunable_prior.numel() == prompt_count:
        selection_score = usage * (0.25 + prunable_prior.detach().clamp(0.0, 1.0))
    else:
        selection_score = usage
    pool_size = min(prompt_count, max(1, int(candidate_pool_size)))
    pool = torch.topk(selection_score, k=pool_size).indices
    should_update = int(epoch) > int(warmup_epochs) and (
        (int(epoch) - int(warmup_epochs) - 1) % max(1, int(update_interval)) == 0
    )
    updated_count = 0
    if should_update:
        update_count = min(pool_size, max(1, int(max_prompts_per_update)))
        update_index = (int(epoch) - int(warmup_epochs) - 1) // max(1, int(update_interval))
        start = (update_index * update_count) % pool_size
        positions = (torch.arange(update_count, device=device) + start) % pool_size
        selected_prompts = pool[positions]
        original_count = int(edge_index.size(1))
        original_weight = prompt_out["adapted_edge_weight"][:original_count]
        static_weight = static_weight.to(device=device, dtype=z.dtype)
        static_ce = F.cross_entropy(static_logits.detach(), y, reduction="none")
        h_pre = model_out["h_pre_shared"]
        was_training = model.training
        model.eval()
        try:
            for prompt_id in selected_prompts.tolist():
                prompt_edges = p2n & (feature == int(prompt_id))
                connected_query = query.clone()
                connected = torch.zeros_like(query)
                connected[node[prompt_edges]] = True
                connected_query &= connected
                if not bool(connected_query.any()):
                    continue
                probe_weight = static_weight.clone()
                probe_weight[prompt_edges] = probe_weight[prompt_edges] * (1.0 - float(probe_strength))
                forward_kwargs: dict[str, Any] = {
                    "h_pre": h_pre,
                    "adapted_x": prompt_out["adapted_x"],
                    "adapted_edge_index": prompt_out["adapted_edge_index"],
                    "adapted_edge_weight": torch.cat([original_weight, probe_weight], dim=0),
                    "return_aux": True,
                }
                if isinstance(model, PromptAwareGP2F):
                    forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
                    forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
                with torch.no_grad():
                    probe_out = model.forward_with_h_pre(z, edge_index, **forward_kwargs)
                    probe_ce = F.cross_entropy(probe_out["logits"], y, reduction="none")
                    class_gains = []
                    for class_id in torch.unique(y[connected_query]):
                        class_mask = connected_query & (y == class_id)
                        class_gains.append((static_ce[class_mask] - probe_ce[class_mask]).mean())
                    removal_gain = torch.stack(class_gains).mean()
                    old = state["ema"][prompt_id]
                    state["ema"][prompt_id] = (
                        float(ema_decay) * old + (1.0 - float(ema_decay)) * removal_gain
                        if bool(torch.isfinite(old))
                        else removal_gain
                    )
                    state["seen"][prompt_id] += 1.0
                    state["harmful_seen"][prompt_id] += float(
                        removal_gain > float(utility_margin)
                    )
                    updated_count += 1
        finally:
            model.train(was_training)

    supervised = torch.isfinite(state["ema"]) & (state["seen"] >= float(min_observations))
    if not bool(supervised.any()):
        empty["p24_causal_prompt_teacher_updated"] = float(updated_count)
        return prompt_logit.sum() * 0.0, state, empty
    gain = state["ema"].detach()
    harmful_stability = state["harmful_seen"] / state["seen"].clamp_min(1.0)
    stable_harmful = (
        supervised
        & (gain > float(utility_margin))
        & (harmful_stability >= float(harmful_stability_threshold))
    )
    harmful_strength = torch.sigmoid(
        (gain - float(utility_margin)) / max(float(target_temperature), 1e-6)
    )
    harmful_strength = torch.where(stable_harmful, harmful_strength, torch.zeros_like(gain))
    target_keep = 1.0 - float(max_prune_target) * harmful_strength
    per_prompt = F.binary_cross_entropy_with_logits(prompt_logit, target_keep, reduction="none")
    loss = per_prompt[supervised].mean()
    harmful = stable_harmful
    helpful = supervised & (gain < -float(utility_margin))
    keep_probability = torch.sigmoid(prompt_logit.detach())
    return loss, state, {
        "p24_causal_prompt_teacher_loss": float(loss.detach().item()),
        "p24_causal_prompt_teacher_count": float(supervised.sum().item()),
        "p24_causal_prompt_teacher_updated": float(updated_count),
        "p24_causal_prompt_removal_gain_mean": float(gain[supervised].mean().item()),
        "p24_causal_prompt_harmful_ratio": float(harmful[supervised].float().mean().item()),
        "p24_causal_prompt_harmful_stability_mean": float(
            harmful_stability[supervised].mean().item()
        ),
        "p24_causal_prompt_target_keep_mean": float(target_keep[supervised].mean().item()),
        "p24_causal_prompt_harmful_keep_mean": float(
            keep_probability[harmful].mean().item() if bool(harmful.any()) else 1.0
        ),
        "p24_causal_prompt_helpful_keep_mean": float(
            keep_probability[helpful].mean().item() if bool(helpful.any()) else 1.0
        ),
    }


def _classification_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    training_cfg: dict[str, Any],
) -> torch.Tensor:
    mask = train_mask.to(device=logits.device, dtype=torch.bool)
    y = labels.to(device=logits.device, dtype=torch.long)
    selected = y[mask]
    if not bool(training_cfg.get("class_balanced_ce", False)) or selected.numel() == 0:
        return F.cross_entropy(logits[mask], selected)
    counts = torch.bincount(selected, minlength=int(num_classes)).to(device=logits.device, dtype=logits.dtype)
    mode = str(training_cfg.get("class_weight_mode", "inverse_sqrt"))
    if mode == "inverse":
        weights = 1.0 / counts.clamp_min(1.0)
    elif mode == "inverse_sqrt":
        weights = 1.0 / counts.clamp_min(1.0).sqrt()
    else:
        raise ValueError(f"Unsupported class_weight_mode={mode!r}")
    present = counts > 0
    weights = torch.where(present, weights, torch.zeros_like(weights))
    weights = weights * (present.to(dtype=weights.dtype).sum().clamp_min(1.0) / weights[present].sum().clamp_min(1e-12))
    return F.cross_entropy(logits[mask], selected, weight=weights)


def _p24_class_coverage_guard_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    temperature: float = 1.0,
    min_entropy_ratio: float = 0.75,
    max_dominant_ratio: float = 0.70,
    class_separation_weight: float = 1.0,
    marginal_balance_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Prevent a P24 candidate from inheriting a single-class P23 collapse."""

    selected_mask = mask.to(device=logits.device, dtype=torch.bool)
    selected_logits = logits[selected_mask]
    selected_labels = labels.to(device=logits.device, dtype=torch.long)[selected_mask]
    zero = logits.new_tensor(0.0)
    if selected_logits.numel() == 0 or int(num_classes) <= 1:
        return zero, {
            "p24_class_coverage_guard_loss": 0.0,
            "p24_class_separation_loss": 0.0,
            "p24_class_marginal_balance_loss": 0.0,
            "p24_prediction_coverage_ratio": 0.0,
            "p24_prediction_dominant_ratio": 0.0,
            "p24_prediction_marginal_entropy_ratio": 0.0,
            "p24_collapse_guard_strength": 0.0,
            "p24_class_mean_margin": 0.0,
        }

    resolved_temperature = max(float(temperature), 1e-6)
    scaled_logits = selected_logits / resolved_temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    marginal = probabilities.mean(dim=0).clamp_min(1e-8)
    uniform = torch.full_like(marginal, 1.0 / float(num_classes))
    marginal_balance = torch.sum(uniform * (uniform.log() - marginal.log()))

    class_means: list[torch.Tensor] = []
    present_ids: list[int] = []
    for class_id in range(int(num_classes)):
        class_mask = selected_labels == class_id
        if bool(class_mask.any()):
            class_means.append(scaled_logits[class_mask].mean(dim=0))
            present_ids.append(class_id)
    if class_means:
        class_mean_logits = torch.stack(class_means, dim=0)
        class_mean_targets = torch.tensor(present_ids, device=logits.device, dtype=torch.long)
        class_separation = F.cross_entropy(class_mean_logits, class_mean_targets)
        target_scores = class_mean_logits.gather(1, class_mean_targets.unsqueeze(1)).squeeze(1)
        other_scores = class_mean_logits.masked_fill(
            F.one_hot(class_mean_targets, num_classes=int(num_classes)).bool(),
            float("-inf"),
        ).max(dim=1).values
        class_mean_margin = (target_scores - other_scores).mean()
    else:
        class_separation = zero
        class_mean_margin = zero

    entropy = -(marginal * marginal.log()).sum()
    entropy_ratio = entropy / math.log(float(num_classes))
    hard_prediction = selected_logits.argmax(dim=-1)
    hard_counts = torch.bincount(hard_prediction, minlength=int(num_classes)).to(dtype=logits.dtype)
    hard_distribution = hard_counts / hard_counts.sum().clamp_min(1.0)
    hard_coverage = (hard_counts > 0).to(dtype=logits.dtype).mean()
    hard_dominant_ratio = hard_distribution.max()

    entropy_deficit = F.relu(logits.new_tensor(float(min_entropy_ratio)) - entropy_ratio)
    dominant_excess = F.relu(
        hard_dominant_ratio - logits.new_tensor(float(max_dominant_ratio))
    )
    entropy_scale = max(float(min_entropy_ratio), 1e-6)
    dominant_scale = max(1.0 - float(max_dominant_ratio), 1e-6)
    guard_strength = torch.maximum(
        entropy_deficit / entropy_scale,
        dominant_excess / dominant_scale,
    ).clamp(0.0, 1.0).detach()
    loss = (
        float(class_separation_weight) * class_separation
        + float(marginal_balance_weight) * guard_strength * marginal_balance
    )
    return loss, {
        "p24_class_coverage_guard_loss": float(loss.detach().item()),
        "p24_class_separation_loss": float(class_separation.detach().item()),
        "p24_class_marginal_balance_loss": float(marginal_balance.detach().item()),
        "p24_prediction_coverage_ratio": float(hard_coverage.detach().item()),
        "p24_prediction_dominant_ratio": float(hard_dominant_ratio.detach().item()),
        "p24_prediction_marginal_entropy_ratio": float(entropy_ratio.detach().item()),
        "p24_collapse_guard_strength": float(guard_strength.item()),
        "p24_class_mean_margin": float(class_mean_margin.detach().item()),
    }


def _p23_feature_harmful_suppression_loss(
    *,
    prompt_out: dict[str, Any],
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float,
    temperature: float,
    harm_quantile: float = 0.70,
    help_quantile: float = 0.70,
    min_signal: float = 0.005,
    help_weight: float = 0.25,
    detach_delta: bool = True,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    def zero_stats() -> dict[str, float]:
        return {
            "p23_feature_harm_loss": 0.0,
            "p23_feature_gate_supervision_loss": 0.0,
            "p23_feature_harm_count": 0.0,
            "p23_feature_help_count": 0.0,
            "p23_feature_harm_ratio": 0.0,
            "p23_feature_help_ratio": 0.0,
            "p23_feature_raw_harm_ratio": 0.0,
            "p23_feature_raw_help_ratio": 0.0,
            "p23_feature_gate_target_mean": 0.0,
            "p23_feature_gate_delta_corr": 0.0,
            "p23_edge_harm_ratio": 0.0,
            "p23_edge_help_ratio": 0.0,
            "p23_edge_gate_target_mean": 0.0,
            "p23_edge_gate_delta_corr": 0.0,
        }

    aux = prompt_out.get("aux", {})
    feature_gate = aux.get("p23_selective_feature_gate")
    edge_feature = aux.get("p23_selective_edge_feature_local")
    edge_node = aux.get("p23_selective_edge_node_index")
    edge_direction = aux.get("p23_selective_edge_direction")
    edge_gate = aux.get("p23_selective_edge_gate_probability", aux.get("p23_selective_edge_gate"))
    dominant_class = aux.get("p23_selective_feature_dominant_class")
    fallback = logits_prompt.new_tensor(0.0)
    if not (
        isinstance(feature_gate, torch.Tensor)
        and isinstance(edge_feature, torch.Tensor)
        and isinstance(edge_node, torch.Tensor)
    ):
        return fallback, fallback, zero_stats()
    if feature_gate.numel() == 0 or edge_feature.numel() == 0 or edge_node.numel() == 0:
        return fallback, fallback, zero_stats()

    y = labels.to(device=logits_prompt.device, dtype=torch.long)
    ce_prompt = F.cross_entropy(logits_prompt, y, reduction="none")
    ce_no_prompt = F.cross_entropy(logits_no_prompt.to(device=logits_prompt.device), y, reduction="none")
    delta = ce_prompt - ce_no_prompt
    if detach_delta:
        delta = delta.detach()

    edge_node = edge_node.to(device=logits_prompt.device, dtype=torch.long)
    edge_feature = edge_feature.to(device=logits_prompt.device, dtype=torch.long)
    train = train_mask.to(device=logits_prompt.device, dtype=torch.bool)
    valid = train[edge_node] & (edge_feature >= 0) & (edge_feature < feature_gate.numel())
    if isinstance(edge_direction, torch.Tensor) and edge_direction.numel() == edge_feature.numel():
        direction = edge_direction.to(device=logits_prompt.device, dtype=torch.long)
        valid = valid & (direction == 1)
    if not bool(valid.any()):
        return fallback, fallback, zero_stats()

    if isinstance(edge_gate, torch.Tensor) and edge_gate.numel() == edge_feature.numel():
        selected_delta = delta[edge_node[valid]]
        selected_gate = edge_gate.to(device=logits_prompt.device, dtype=logits_prompt.dtype)[valid].clamp(1e-6, 1.0 - 1e-6)
        harmful = F.relu(selected_delta - float(margin))
        helpful = F.relu(-selected_delta - float(margin))
        harm_q = min(max(float(harm_quantile), 0.0), 1.0)
        help_q = min(max(float(help_quantile), 0.0), 1.0)
        signal_floor = max(float(min_signal), 0.0)
        harm_threshold = torch.quantile(harmful.detach(), harm_q).clamp_min(signal_floor)
        help_threshold = torch.quantile(helpful.detach(), help_q).clamp_min(signal_floor)
        harmful_edge = (harmful >= harm_threshold) & (harmful >= helpful)
        helpful_edge = (helpful >= help_threshold) & (helpful > harmful)
        selected_edge = harmful_edge | helpful_edge
        utility_target = torch.sigmoid((-selected_delta.detach()) / max(float(temperature), 1e-6)).to(
            dtype=selected_gate.dtype
        )
        per_edge_harm = harmful.detach() * selected_gate
        per_edge_help = helpful.detach() * (1.0 - selected_gate)
        per_edge_gate_loss = F.binary_cross_entropy(selected_gate, utility_target, reduction="none")

        if class_balanced:
            selected_labels = y[edge_node[valid]]
            losses_harm: list[torch.Tensor] = []
            losses_help: list[torch.Tensor] = []
            losses_gate: list[torch.Tensor] = []
            for cls in torch.unique(selected_labels):
                cls_mask = selected_labels == cls
                cls_harm = harmful_edge & cls_mask
                cls_help = helpful_edge & cls_mask
                cls_selected = selected_edge & cls_mask
                if bool(cls_harm.any()):
                    losses_harm.append(per_edge_harm[cls_harm].mean())
                if bool(cls_help.any()):
                    losses_help.append(per_edge_help[cls_help].mean())
                if bool(cls_selected.any()):
                    losses_gate.append(per_edge_gate_loss[cls_selected].mean())
            harm_part = torch.stack(losses_harm).mean() if losses_harm else fallback
            help_part = torch.stack(losses_help).mean() if losses_help else fallback
            harm_loss = harm_part + max(float(help_weight), 0.0) * help_part
            gate_loss = torch.stack(losses_gate).mean() if losses_gate else fallback
        else:
            harm_part = per_edge_harm[harmful_edge].mean() if bool(harmful_edge.any()) else fallback
            help_part = per_edge_help[helpful_edge].mean() if bool(helpful_edge.any()) else fallback
            harm_loss = harm_part + max(float(help_weight), 0.0) * help_part
            gate_loss = per_edge_gate_loss[selected_edge].mean() if bool(selected_edge.any()) else fallback

        with torch.no_grad():
            gate_centered = selected_gate - selected_gate.mean()
            utility_centered = (-selected_delta) - (-selected_delta).mean()
            denom = gate_centered.norm() * utility_centered.norm()
            corr = (gate_centered * utility_centered).sum() / denom.clamp_min(1e-12)
        stats = {
            "p23_feature_harm_loss": float(harm_loss.detach().item()),
            "p23_feature_gate_supervision_loss": float(gate_loss.detach().item()),
            "p23_feature_harm_count": float(harmful_edge.sum().detach().item()),
            "p23_feature_help_count": float(helpful_edge.sum().detach().item()),
            "p23_feature_harm_ratio": float(harmful_edge.float().mean().detach().item()),
            "p23_feature_help_ratio": float(helpful_edge.float().mean().detach().item()),
            "p23_feature_raw_harm_ratio": float((harmful > 0).float().mean().detach().item()),
            "p23_feature_raw_help_ratio": float((helpful > 0).float().mean().detach().item()),
            "p23_feature_gate_target_mean": float(utility_target.mean().detach().item()),
            "p23_feature_gate_delta_corr": float(corr.detach().item()),
            "p23_edge_harm_ratio": float(harmful_edge.float().mean().detach().item()),
            "p23_edge_help_ratio": float(helpful_edge.float().mean().detach().item()),
            "p23_edge_gate_target_mean": float(utility_target.mean().detach().item()),
            "p23_edge_gate_delta_corr": float(corr.detach().item()),
        }
        return harm_loss, gate_loss, stats

    selected_feature = edge_feature[valid]
    selected_delta = delta[edge_node[valid]]
    num_features = int(feature_gate.numel())
    counts = feature_gate.new_zeros(num_features)
    counts.index_add_(0, selected_feature, torch.ones_like(selected_delta, dtype=feature_gate.dtype))
    harm_sum = feature_gate.new_zeros(num_features)
    help_sum = feature_gate.new_zeros(num_features)
    harmful = F.relu(selected_delta - float(margin))
    helpful = F.relu(-selected_delta - float(margin))
    harm_sum.index_add_(0, selected_feature, harmful.to(dtype=feature_gate.dtype))
    help_sum.index_add_(0, selected_feature, helpful.to(dtype=feature_gate.dtype))
    present = counts > 0
    harm = harm_sum / counts.clamp_min(1.0)
    help_value = help_sum / counts.clamp_min(1.0)
    utility = help_value - harm
    gate = feature_gate.to(device=logits_prompt.device, dtype=logits_prompt.dtype).clamp(1e-6, 1.0 - 1e-6)
    if not bool(present.any()):
        return fallback, fallback, zero_stats()

    harm_q = min(max(float(harm_quantile), 0.0), 1.0)
    help_q = min(max(float(help_quantile), 0.0), 1.0)
    signal_floor = max(float(min_signal), 0.0)
    harm_threshold = torch.quantile(harm[present].detach(), harm_q).clamp_min(signal_floor)
    help_threshold = torch.quantile(help_value[present].detach(), help_q).clamp_min(signal_floor)
    harmful_feature = present & (harm >= harm_threshold) & (harm >= help_value)
    helpful_feature = present & (help_value >= help_threshold) & (help_value > harm)
    selected_feature_mask = harmful_feature | helpful_feature

    target_gate = gate.detach().clone()
    utility_target = torch.sigmoid(utility.detach() / max(float(temperature), 1e-6)).to(dtype=gate.dtype)
    target_gate = torch.where(selected_feature_mask, utility_target, target_gate)
    per_feature_harm = harm.detach() * gate
    per_feature_help = help_value.detach() * (1.0 - gate)
    per_feature_gate_loss = F.binary_cross_entropy(gate, target_gate.to(dtype=gate.dtype), reduction="none")

    if class_balanced and isinstance(dominant_class, torch.Tensor) and dominant_class.numel() == num_features:
        dom = dominant_class.to(device=logits_prompt.device, dtype=torch.long)
        losses_harm: list[torch.Tensor] = []
        losses_help: list[torch.Tensor] = []
        losses_gate: list[torch.Tensor] = []
        for cls in torch.unique(dom[present]):
            cls_harm = harmful_feature & (dom == cls)
            cls_help = helpful_feature & (dom == cls)
            cls_selected = selected_feature_mask & (dom == cls)
            if bool(cls_harm.any()):
                losses_harm.append(per_feature_harm[cls_harm].mean())
            if bool(cls_help.any()):
                losses_help.append(per_feature_help[cls_help].mean())
            if bool(cls_selected.any()):
                losses_gate.append(per_feature_gate_loss[cls_selected].mean())
        harm_part = torch.stack(losses_harm).mean() if losses_harm else fallback
        help_part = torch.stack(losses_help).mean() if losses_help else fallback
        harm_loss = harm_part + max(float(help_weight), 0.0) * help_part
        gate_loss = torch.stack(losses_gate).mean() if losses_gate else fallback
    else:
        harm_part = per_feature_harm[harmful_feature].mean() if bool(harmful_feature.any()) else fallback
        help_part = per_feature_help[helpful_feature].mean() if bool(helpful_feature.any()) else fallback
        harm_loss = harm_part + max(float(help_weight), 0.0) * help_part
        gate_loss = per_feature_gate_loss[selected_feature_mask].mean() if bool(selected_feature_mask.any()) else fallback

    with torch.no_grad():
        gate_centered = gate[present] - gate[present].mean()
        utility_centered = utility[present] - utility[present].mean()
        denom = gate_centered.norm() * utility_centered.norm()
        corr = (gate_centered * utility_centered).sum() / denom.clamp_min(1e-12)
    stats = {
        "p23_feature_harm_loss": float(harm_loss.detach().item()),
        "p23_feature_gate_supervision_loss": float(gate_loss.detach().item()),
        "p23_feature_harm_count": float(harmful_feature.sum().detach().item()),
        "p23_feature_help_count": float(helpful_feature.sum().detach().item()),
        "p23_feature_harm_ratio": float(harmful_feature[present].float().mean().detach().item()),
        "p23_feature_help_ratio": float(helpful_feature[present].float().mean().detach().item()),
        "p23_feature_raw_harm_ratio": float((harm[present] > 0).float().mean().detach().item()),
        "p23_feature_raw_help_ratio": float((help_value[present] > 0).float().mean().detach().item()),
        "p23_feature_gate_target_mean": float(target_gate[present].mean().detach().item()),
        "p23_feature_gate_delta_corr": float(corr.detach().item()),
    }
    return harm_loss, gate_loss, stats


def _forward_prompt_graph(
    *,
    model: DualBranchFramework,
    prompt_graph_module: (
        PromptGraphModuleP1
        | SelectiveDiscreteFeaturePromptGraph
        | GraphiteStylePromptGraphAdapter
        | PromptGraphConstruction
        | P23V01PromptModule
        | None
    ),
    z: torch.Tensor,
    x_raw: torch.Tensor | None = None,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
    edge_scale_multiplier: float | torch.Tensor = 1.0,
    epoch: int | None = None,
    h_pre: torch.Tensor | None = None,
    no_prompt_logits: torch.Tensor | None = None,
    h_adp_no_prompt: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if h_pre is None:
        h_pre = model.encode_frozen(z, edge_index)
    if prompt_graph_module is None:
        prompt_out = _default_prompt_graph_out(z, edge_index)
    else:
        prompt_kwargs: dict[str, Any] = {
            "z": z,
            "h_pre": h_pre.detach(),
            "edge_index": edge_index,
            "train_mask": train_mask,
            "edge_scale_multiplier": edge_scale_multiplier,
            "no_prompt_logits": no_prompt_logits,
            "h_adp_no_prompt": h_adp_no_prompt,
        }
        if isinstance(prompt_graph_module, GraphiteStylePromptGraphAdapter):
            prompt_kwargs["x_raw"] = x_raw
        if isinstance(prompt_graph_module, PromptGraphConstruction):
            prompt_kwargs["x_raw"] = x_raw
            prompt_kwargs["labels"] = labels
            prompt_kwargs["epoch"] = epoch
        prompt_out = prompt_graph_module(**prompt_kwargs)
    forward_kwargs: dict[str, Any] = {
        "h_pre": h_pre,
        "adapted_x": prompt_out["adapted_x"],
        "adapted_edge_index": prompt_out["adapted_edge_index"],
        "adapted_edge_weight": prompt_out["adapted_edge_weight"],
        "return_aux": True,
    }
    if isinstance(model, PromptAwareGP2F):
        forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
        forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
    if (
        isinstance(prompt_graph_module, P23ProjectorSynchronizedPromptGraphAdapter)
        and prompt_graph_module.projector_receiver_enabled
    ):
        no_prompt_out = _forward_no_prompt_with_h_pre(
            model=model,
            z=z,
            edge_index=edge_index,
            h_pre=h_pre,
        )
        receiver_out = prompt_graph_module.apply_projector_receiver(
            h_base=no_prompt_out["h_adp"],
            prompt_out=prompt_out,
        )
        h_adp = receiver_out["h_adp"]
        alpha = no_prompt_out["alpha"]
        h_mix = alpha * no_prompt_out["h_pre"] + (1.0 - alpha) * h_adp
        model_out = dict(no_prompt_out)
        model_out["h_adp"] = h_adp
        model_out["h_mix"] = h_mix
        model_out["logits"] = model.classifier(h_mix)
        model_out["p23_receiver"] = receiver_out
        model_out["p23_no_receiver_logits"] = no_prompt_out["logits"].detach()
        model_out["p23_no_receiver_logits_train"] = no_prompt_out["logits"]
    else:
        model_out = model.forward_with_h_pre(z, edge_index, **forward_kwargs)
    if (
        isinstance(prompt_graph_module, PromptingAdaptation)
        and prompt_graph_module.counterfactual_utility_enabled
    ):
        aux = prompt_out.get("aux", {})
        static_prompt_weight = aux.get("p24_static_prompt_edge_weight")
        candidate_prompt_weight = aux.get("p24_candidate_prompt_edge_weight")
        if isinstance(static_prompt_weight, torch.Tensor) and isinstance(candidate_prompt_weight, torch.Tensor):
            original_edge_count = int(edge_index.size(1))
            original_weight = prompt_out["adapted_edge_weight"][:original_edge_count]
            static_kwargs = dict(forward_kwargs)
            static_kwargs["adapted_edge_weight"] = torch.cat([original_weight, static_prompt_weight], dim=0)
            candidate_kwargs = dict(forward_kwargs)
            candidate_kwargs["adapted_edge_weight"] = torch.cat([original_weight, candidate_prompt_weight], dim=0)
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    static_out = model.forward_with_h_pre(z, edge_index, **static_kwargs)
                    counterfactual_no_prompt = _forward_no_prompt_with_h_pre(
                        model=model,
                        z=z,
                        edge_index=edge_index,
                        h_pre=h_pre,
                    )
                candidate_out = model.forward_with_h_pre(z, edge_index, **candidate_kwargs)
            finally:
                model.train(was_training)
            if prompt_graph_module.counterfactual_router is not None:
                prompt_out = prompt_graph_module.apply_counterfactual_routing(
                    prompt_out=prompt_out,
                    static_logits=static_out["logits"],
                    candidate_logits=candidate_out["logits"],
                    original_edge_count=original_edge_count,
                    epoch=epoch,
                )
                deployed_kwargs = dict(forward_kwargs)
                deployed_kwargs["adapted_edge_weight"] = prompt_out["adapted_edge_weight"]
                model_out = model.forward_with_h_pre(z, edge_index, **deployed_kwargs)
            model_out["p24_static_logits"] = static_out["logits"].detach()
            model_out["p24_candidate_logits"] = candidate_out["logits"]
            model_out["p24_no_prompt_logits"] = counterfactual_no_prompt["logits"].detach()
    if isinstance(prompt_graph_module, P23V01PromptModule):
        no_receiver_logits = model_out["logits"]
        receiver_out = prompt_graph_module.apply_receiver(
            model_out["h_adp"],
            edge_scale=prompt_out.get("edge_scale", 1.0),
        )
        h_adp = receiver_out["h_adp"]
        alpha = model_out["alpha"]
        h_mix = alpha * model_out["h_pre"] + (1.0 - alpha) * h_adp
        model_out["h_adp"] = h_adp
        model_out["h_mix"] = h_mix
        model_out["logits"] = model.classifier(h_mix)
        model_out["p23_receiver"] = receiver_out
        model_out["p23_no_receiver_logits"] = no_receiver_logits.detach()
        prompt_out["aux"].update(prompt_graph_module.receiver_aux(receiver_out))
    model_out["h_pre_shared"] = h_pre
    return model_out, prompt_out


def _forward_no_prompt_with_h_pre(
    *,
    model: DualBranchFramework,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    h_pre: torch.Tensor,
) -> dict[str, Any]:
    prompt_out = _default_prompt_graph_out(z, edge_index)
    forward_kwargs: dict[str, Any] = {
        "h_pre": h_pre,
        "adapted_x": prompt_out["adapted_x"],
        "adapted_edge_index": prompt_out["adapted_edge_index"],
        "adapted_edge_weight": prompt_out["adapted_edge_weight"],
        "return_aux": True,
    }
    if isinstance(model, PromptAwareGP2F):
        forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
        forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
    return model.forward_with_h_pre(z, edge_index, **forward_kwargs)


def _needs_no_prompt_pool_evidence(
    prompt_graph_module: (
        PromptGraphModuleP1
        | SelectiveDiscreteFeaturePromptGraph
        | GraphiteStylePromptGraphAdapter
        | PromptGraphConstruction
        | P23V01PromptModule
        | None
    ),
) -> bool:
    if prompt_graph_module is None:
        return False
    if isinstance(prompt_graph_module, SelectiveDiscreteFeaturePromptGraph):
        return (
            float(getattr(prompt_graph_module, "uncertainty_weight", 0.0)) > 0.0
            or float(getattr(prompt_graph_module, "disagreement_weight", 0.0)) > 0.0
        )
    if isinstance(prompt_graph_module, P23V01PromptModule):
        return False
    return getattr(prompt_graph_module, "pool_strategy", "") == "utility_structural"


@torch.no_grad()
def _maybe_build_p23_v01_static_state(
    *,
    prompt_graph_module: torch.nn.Module | None,
    input_aligner: InputAligner,
    model: DualBranchFramework,
    x_raw: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(prompt_graph_module, P23V01PromptModule):
        return {}
    aligner_was_training = input_aligner.training
    model_was_training = model.training
    module_was_training = prompt_graph_module.training
    input_aligner.eval()
    model.eval()
    prompt_graph_module.eval()
    z_snapshot = input_aligner(x_raw)
    h_pre = model.encode_frozen(z_snapshot, edge_index)
    no_prompt_out = _forward_no_prompt_with_h_pre(
        model=model,
        z=z_snapshot,
        edge_index=edge_index,
        h_pre=h_pre,
    )
    state = prompt_graph_module.build_state(
        x_raw=x_raw,
        z_snapshot=z_snapshot.detach(),
        edge_index=edge_index,
        train_mask=train_mask,
        no_prompt_logits=no_prompt_out["logits"].detach(),
        h_pre_snapshot=h_pre.detach(),
        h_adp0_snapshot=no_prompt_out["h_adp"].detach(),
    )
    input_aligner.train(aligner_was_training)
    model.train(model_was_training)
    prompt_graph_module.train(module_was_training)
    return {
        "p23_static_pool_size": float(state.pool_mask.sum().item()),
        "p23_static_pool_ratio": float(state.pool_mask.float().mean().item()),
        "p23_static_feature_prompt_count": float(state.prompt_x.size(0)),
        "p23_static_prompt_edge_count": float(state.prompt_edge_index.size(1)),
        "p23_static_graph_risk": float(state.graph_risk.detach().item()),
    }


def _forward_prompt_adapter(
    *,
    model: DualBranchFramework,
    prompt_adapter_module: HeterophilyAwarePromptAdapter | ClassConditionedPatternPromptRouter | P21LiteAdaptiveFilter | P21V2HeteroFilter | P22ClassPatternEnrichmentBank,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    update_mask: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    compat_support_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    h_pre = model.encode_frozen(z, edge_index)
    no_prompt_out = _forward_no_prompt_with_h_pre(
        model=model,
        z=z,
        edge_index=edge_index,
        h_pre=h_pre,
    )
    adapter_kwargs: dict[str, Any] = {
        "z": z,
        "edge_index": edge_index,
        "h_adp": no_prompt_out["h_adp"],
        "update_mask": update_mask,
        "support_mask": support_mask,
        "compat_support_mask": compat_support_mask,
        "labels": labels,
    }
    if bool(getattr(prompt_adapter_module, "consumes_base_logits", False)):
        adapter_kwargs["base_logits"] = no_prompt_out["logits"].detach()
        adapter_kwargs["h_pre"] = h_pre.detach()
        adapter_kwargs["h_adp_base"] = no_prompt_out["h_adp"].detach()
    adapter_out = prompt_adapter_module(**adapter_kwargs)
    h_adp = adapter_out["h_adp"]
    alpha = model.alpha
    h_mix = alpha * h_pre + (1.0 - alpha) * h_adp
    adapter_logits = adapter_out.get("logits")
    logits = adapter_logits if isinstance(adapter_logits, torch.Tensor) else model.classifier(h_mix)
    model_out = {
        "logits": logits,
        "h_pre": h_pre,
        "h_adp": h_adp,
        "h_mix": h_mix,
        "alpha": alpha,
        "h_adp_full": h_adp,
        "h_pre_shared": h_pre,
        "prompt_adapter": adapter_out,
        "no_prompt_logits": no_prompt_out["logits"],
        "h_adp_no_prompt": no_prompt_out["h_adp"],
        "h_mix_no_prompt": no_prompt_out["h_mix"],
    }
    return model_out, adapter_out, no_prompt_out


def _mean_float_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats_list:
        return {}
    keys: set[str] = set()
    for stats in stats_list:
        keys.update(stats.keys())
    averaged: dict[str, Any] = {}
    for key in keys:
        values: list[float] = []
        for stats in stats_list:
            value = stats.get(key)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values and len(values) == len(stats_list):
            averaged[key] = float(sum(values) / len(values))
    return averaged


def _prompt_adapter_episode_consistency_loss(
    adapter_outs: list[dict[str, torch.Tensor]],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if len(adapter_outs) < 2:
        zero = torch.tensor(0.0, device=mask.device)
        return zero, zero, {
            "prompt_adapter_gate_consistency_loss": 0.0,
            "prompt_adapter_delta_consistency_loss": 0.0,
            "prompt_adapter_episode_count": float(len(adapter_outs)),
        }
    gate_values = [out.get("gate") for out in adapter_outs]
    delta_values = [out.get("delta") for out in adapter_outs]
    if not all(isinstance(value, torch.Tensor) for value in gate_values + delta_values):
        zero = torch.tensor(0.0, device=mask.device)
        return zero, zero, {
            "prompt_adapter_gate_consistency_loss": 0.0,
            "prompt_adapter_delta_consistency_loss": 0.0,
            "prompt_adapter_episode_count": float(len(adapter_outs)),
        }
    first_gate = gate_values[0]
    assert isinstance(first_gate, torch.Tensor)
    mask = mask.to(device=first_gate.device, dtype=torch.bool)
    if int(mask.sum().item()) == 0:
        zero = first_gate.new_tensor(0.0)
        return zero, zero, {
            "prompt_adapter_gate_consistency_loss": 0.0,
            "prompt_adapter_delta_consistency_loss": 0.0,
            "prompt_adapter_episode_count": float(len(adapter_outs)),
        }

    gate_stack = torch.stack([value[mask] for value in gate_values if isinstance(value, torch.Tensor)])
    gate_loss = gate_stack.var(dim=0, unbiased=False).mean()

    delta_stack = torch.stack([value[mask] for value in delta_values if isinstance(value, torch.Tensor)])
    pair_losses: list[torch.Tensor] = []
    for left, right in zip(delta_stack[:-1], delta_stack[1:]):
        left_norm = left.norm(dim=-1)
        right_norm = right.norm(dim=-1)
        valid = (left_norm > 1e-8) & (right_norm > 1e-8)
        if bool(valid.any()):
            cosine = F.cosine_similarity(left[valid], right[valid], dim=-1, eps=1e-12)
            pair_losses.append((1.0 - cosine).mean())
    delta_loss = torch.stack(pair_losses).mean() if pair_losses else gate_loss.new_tensor(0.0)
    return gate_loss, delta_loss, {
        "prompt_adapter_gate_consistency_loss": float(gate_loss.detach().item()),
        "prompt_adapter_delta_consistency_loss": float(delta_loss.detach().item()),
        "prompt_adapter_episode_count": float(len(adapter_outs)),
    }


def _adapter_mask(
    strategy: str,
    *,
    train_mask: torch.Tensor,
    support_mask: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    strategy = str(strategy)
    if strategy == "all":
        return torch.ones_like(train_mask, dtype=torch.bool)
    if strategy in {"candidate_pool", "pool"}:
        if candidate_mask is None:
            return torch.ones_like(train_mask, dtype=torch.bool)
        return candidate_mask.to(device=train_mask.device, dtype=torch.bool)
    if strategy == "train":
        return train_mask.bool()
    if strategy == "support":
        return support_mask.bool()
    if strategy == "query":
        return query_mask.bool()
    if strategy == "none":
        return torch.zeros_like(train_mask, dtype=torch.bool)
    raise ValueError(f"Unsupported prompt_adapter mask strategy: {strategy}")


def _prompt_adapter_diagnostics(adapter_out: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    if adapter_out is None:
        return {
            "prompt_adapter_enabled": 0.0,
            "prompt_adapter_update_norm": 0.0,
            "prompt_adapter_update_max_norm": 0.0,
            "prompt_adapter_raw_delta_norm": 0.0,
            "prompt_adapter_delta_norm": 0.0,
            "prompt_adapter_gate_mean": 0.0,
            "prompt_adapter_gate_min": 0.0,
            "prompt_adapter_gate_max": 0.0,
            "prompt_adapter_raw_gate_mean": 0.0,
            "prompt_adapter_raw_gate_min": 0.0,
            "prompt_adapter_raw_gate_max": 0.0,
            "prompt_adapter_update_mask_ratio": 0.0,
            "prompt_adapter_clip_ratio": 0.0,
            "high_frequency_norm": 0.0,
            "low_frequency_norm": 0.0,
            "support_context_enabled": 0.0,
            "support_context_available": 0.0,
            "support_context_coverage": 0.0,
            "support_context_count": 0.0,
            "support_similarity_margin": 0.0,
            "support_similarity_entropy": 0.0,
            "support_reliability_mean": 0.0,
            "support_reliability_min": 0.0,
            "support_reliability_max": 0.0,
            "support_topk_mean_score": 0.0,
        }

    def scalar(name: str) -> float:
        value = adapter_out.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().item())
        return 0.0

    def norm(name: str) -> float:
        value = adapter_out.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.detach().norm(dim=-1).mean().item()) if value.ndim >= 2 else float(value.detach().abs().mean().item())
        return 0.0

    def vector(name: str) -> list[float]:
        value = adapter_out.get(name)
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            return [float(item) for item in value.detach().cpu().tolist()]
        return []

    out = {
        "prompt_adapter_enabled": 1.0,
        "prompt_adapter_update_norm": scalar("prompt_update_norm"),
        "prompt_adapter_update_max_norm": scalar("prompt_update_max_norm"),
        "prompt_adapter_raw_delta_norm": scalar("prompt_raw_delta_norm"),
        "prompt_adapter_delta_norm": scalar("prompt_delta_norm"),
        "prompt_adapter_gate_mean": scalar("prompt_gate_mean"),
        "prompt_adapter_gate_min": scalar("prompt_gate_min"),
        "prompt_adapter_gate_max": scalar("prompt_gate_max"),
        "prompt_adapter_raw_gate_mean": scalar("prompt_raw_gate_mean"),
        "prompt_adapter_raw_gate_min": scalar("prompt_raw_gate_min"),
        "prompt_adapter_raw_gate_max": scalar("prompt_raw_gate_max"),
        "prompt_adapter_update_mask_ratio": scalar("prompt_update_mask_ratio"),
        "prompt_adapter_clip_ratio": scalar("prompt_update_clip_ratio"),
        "high_frequency_norm": scalar("high_frequency_norm"),
        "low_frequency_norm": scalar("low_frequency_norm"),
        "support_context_enabled": scalar("support_context_enabled"),
        "support_context_available": scalar("support_context_available"),
        "support_context_coverage": scalar("support_context_coverage"),
        "support_context_count": scalar("support_context_count"),
        "support_similarity_margin": scalar("support_similarity_margin"),
        "support_similarity_entropy": scalar("support_similarity_entropy"),
        "support_reliability_mean": scalar("support_reliability_mean"),
        "support_reliability_min": scalar("support_reliability_min"),
        "support_reliability_max": scalar("support_reliability_max"),
        "support_topk_mean_score": scalar("support_topk_mean_score"),
        "p21_filter_enabled": scalar("p21_filter_enabled"),
        "p21_beta": scalar("p21_beta"),
        "p21_gate_mean": scalar("p21_gate_mean"),
        "p21_gate_min": scalar("p21_gate_min"),
        "p21_gate_max": scalar("p21_gate_max"),
        "p21_channel_delta_reject_norm": scalar("p21_channel_delta_reject_norm"),
        "p21_channel_delta_ego_norm": scalar("p21_channel_delta_ego_norm"),
        "p21_channel_delta_low_norm": scalar("p21_channel_delta_low_norm"),
        "p21_channel_delta_two_norm": scalar("p21_channel_delta_two_norm"),
        "p21_channel_delta_high_norm": scalar("p21_channel_delta_high_norm"),
        "p21_channel_delta_compat_norm": scalar("p21_channel_delta_compat_norm"),
        "p21_channel_delta_role_norm": scalar("p21_channel_delta_role_norm"),
        "p21_alpha_entropy": scalar("p21_alpha_entropy"),
        "p21_alpha_reject_mean": scalar("p21_alpha_reject_mean"),
        "p21_alpha_ego_mean": scalar("p21_alpha_ego_mean"),
        "p21_alpha_low_mean": scalar("p21_alpha_low_mean"),
        "p21_alpha_two_mean": scalar("p21_alpha_two_mean"),
        "p21_alpha_high_mean": scalar("p21_alpha_high_mean"),
        "p21_alpha_compat_mean": scalar("p21_alpha_compat_mean"),
        "p21_alpha_role_mean": scalar("p21_alpha_role_mean"),
        "p21_alpha_global_reject": scalar("p21_alpha_global_reject"),
        "p21_alpha_global_ego": scalar("p21_alpha_global_ego"),
        "p21_alpha_global_low": scalar("p21_alpha_global_low"),
        "p21_alpha_global_two": scalar("p21_alpha_global_two"),
        "p21_alpha_global_high": scalar("p21_alpha_global_high"),
        "p21_alpha_global_compat": scalar("p21_alpha_global_compat"),
        "p21_alpha_global_role": scalar("p21_alpha_global_role"),
        "p21_channel_reject_norm": scalar("p21_channel_reject_norm"),
        "p21_channel_ego_norm": scalar("p21_channel_ego_norm"),
        "p21_channel_low_norm": scalar("p21_channel_low_norm"),
        "p21_channel_two_norm": scalar("p21_channel_two_norm"),
        "p21_channel_high_norm": scalar("p21_channel_high_norm"),
        "p21_channel_compat_norm": scalar("p21_channel_compat_norm"),
        "p21_channel_role_norm": scalar("p21_channel_role_norm"),
        "p21_v2_filter_enabled": scalar("p21_v2_filter_enabled"),
        "p21_v2_compat_class_coverage": scalar("p21_v2_compat_class_coverage"),
        "p21_v2_compat_proto_coverage": scalar("p21_v2_compat_proto_coverage"),
        "p21_v2_neighbor_prediction_entropy": scalar("p21_v2_neighbor_prediction_entropy"),
        "p21_ego_low_discrepancy": scalar("p21_ego_low_discrepancy"),
        "p21_low_two_discrepancy": scalar("p21_low_two_discrepancy"),
        "p21_no_prompt_entropy": scalar("p21_no_prompt_entropy"),
        "p21_no_prompt_margin": scalar("p21_no_prompt_margin"),
        "p22_pattern_scale": scalar("pattern_scale"),
        "p22_pattern_reg": scalar("pattern_reg"),
        "p22_basis_usage_loss": scalar("basis_usage_loss"),
        "p22_pattern_usage_entropy": scalar("pattern_usage_entropy"),
        "p22_pattern_max_prob_mean": scalar("pattern_max_prob_mean"),
        "p22_basis_usage_entropy": scalar("basis_usage_entropy"),
        "p22_class_pattern_kl_to_global": scalar("class_pattern_kl_to_global"),
        "p22_logit_bias_norm": norm("logit_bias"),
        "p22_enrichment_evidence_norm": norm("enrichment_evidence"),
        "p22_basis_evidence_norm": norm("basis_pattern_evidence"),
        "p22_final_pattern_evidence_norm": norm("pattern_evidence"),
        "p22_gate_mean": scalar("prompt_gate_mean"),
        "p22_gate_std": scalar("p22_gate_std"),
        "p22_gate_open_ratio": scalar("p22_gate_open_ratio"),
        "transition_C_row_entropy": scalar("transition_C_row_entropy"),
        "transition_C_diag_mean": scalar("transition_C_diag_mean"),
        "transition_C_offdiag_mean": scalar("transition_C_offdiag_mean"),
        "transition_C_max_mean": scalar("transition_C_max_mean"),
        "transition_support_edge_count": scalar("transition_support_edge_count"),
        "transition_support_nonzero_row_ratio": scalar("transition_support_nonzero_row_ratio"),
        "transition_support_class_pair_coverage": scalar("transition_support_class_pair_coverage"),
        "p22_pattern_usage_distribution": vector("pattern_usage_mean"),
        "p22_basis_usage_distribution": vector("basis_usage"),
    }
    basis_weight = adapter_out.get("pattern_basis_weight")
    if isinstance(basis_weight, torch.Tensor):
        out["p22_pattern_basis_weight"] = [
            [float(v) for v in row] for row in basis_weight.detach().cpu().tolist()
        ]
    topk_index = adapter_out.get("class_pattern_topk_index")
    topk_value = adapter_out.get("class_pattern_topk_value")
    if isinstance(topk_index, torch.Tensor) and isinstance(topk_value, torch.Tensor):
        out["p22_class_pattern_topk"] = {
            str(class_id): [
                {"pattern": int(idx), "value": float(val)}
                for idx, val in zip(topk_index[class_id].detach().cpu().tolist(), topk_value[class_id].detach().cpu().tolist())
            ]
            for class_id in range(int(topk_index.size(0)))
        }
    return out


def _prompt_adapter_delta_stats(
    *,
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return {
            f"{prefix}_mean_delta_ce": 0.0,
            f"{prefix}_positive_delta_ratio": 0.0,
            f"{prefix}_mean_ce_no_prompt": 0.0,
            f"{prefix}_mean_ce_prompt": 0.0,
            f"{prefix}_count": 0.0,
            f"{prefix}_delta_ce_by_class": {},
        }
    y = labels.to(device=logits_prompt.device)[idx]
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt.detach()[idx], y, reduction="none")
    delta = ce_no - ce_prompt
    by_class: dict[str, dict[str, float]] = {}
    for class_id in torch.unique(y.detach()).tolist():
        class_mask = y == int(class_id)
        by_class[str(int(class_id))] = {
            "count": float(class_mask.sum().item()),
            "mean_delta_ce": float(delta[class_mask].mean().item()),
            "positive_delta_ratio": float((delta[class_mask] > 0.0).float().mean().item()),
        }
    return {
        f"{prefix}_mean_delta_ce": float(delta.mean().item()),
        f"{prefix}_positive_delta_ratio": float((delta > 0.0).float().mean().item()),
        f"{prefix}_mean_ce_no_prompt": float(ce_no.mean().item()),
        f"{prefix}_mean_ce_prompt": float(ce_prompt.mean().item()),
        f"{prefix}_count": float(idx.numel()),
        f"{prefix}_delta_ce_by_class": by_class,
    }


def _safe_metric_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


@torch.no_grad()
def _p22_single_basis_delta_stats(
    *,
    adapter_out: dict[str, Any],
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    masks: dict[str, torch.Tensor],
    scale_grid: list[float],
) -> dict[str, float]:
    basis_evidence = adapter_out.get("basis_evidence")
    if not isinstance(basis_evidence, torch.Tensor) or basis_evidence.ndim != 3:
        return {}
    if not scale_grid:
        scale_grid = [0.05]
    basis_names_raw = adapter_out.get("basis_names")
    if isinstance(basis_names_raw, list) and len(basis_names_raw) == int(basis_evidence.size(1)):
        basis_names = [_safe_metric_name(str(name)) for name in basis_names_raw]
    else:
        basis_names = [f"basis_{idx}" for idx in range(int(basis_evidence.size(1)))]
    logits_base = base_logits.detach().to(device=basis_evidence.device, dtype=basis_evidence.dtype)
    y = labels.to(device=basis_evidence.device, dtype=torch.long)
    out: dict[str, float] = {}
    for split_name, split_mask in masks.items():
        mask = split_mask.to(device=basis_evidence.device, dtype=torch.bool)
        idx = torch.where(mask)[0]
        for basis_idx, basis_name in enumerate(basis_names):
            key_prefix = f"basis_delta_ce_{basis_name}_{split_name}"
            if idx.numel() == 0:
                out[f"{key_prefix}_best"] = 0.0
                out[f"{key_prefix}_best_scale"] = 0.0
                out[f"{key_prefix}_positive_ratio"] = 0.0
                if split_name == "val":
                    out[f"basis_delta_ce_{basis_name}"] = 0.0
                    out[f"basis_delta_scale_{basis_name}"] = 0.0
                continue
            base_ce = F.cross_entropy(logits_base[idx], y[idx], reduction="none")
            best_delta: torch.Tensor | None = None
            best_positive_ratio = 0.0
            best_scale = 0.0
            for scale in scale_grid:
                scale_value = float(scale)
                logits_basis = logits_base + scale_value * basis_evidence[:, basis_idx, :]
                basis_ce = F.cross_entropy(logits_basis[idx], y[idx], reduction="none")
                delta = base_ce - basis_ce
                mean_delta = delta.mean()
                if best_delta is None or float(mean_delta.item()) > float(best_delta.item()):
                    best_delta = mean_delta
                    best_positive_ratio = float((delta > 0.0).to(dtype=basis_evidence.dtype).mean().item())
                    best_scale = scale_value
            assert best_delta is not None
            out[f"{key_prefix}_best"] = float(best_delta.item())
            out[f"{key_prefix}_best_scale"] = best_scale
            out[f"{key_prefix}_positive_ratio"] = best_positive_ratio
            if split_name == "val":
                out[f"basis_delta_ce_{basis_name}"] = float(best_delta.item())
                out[f"basis_delta_scale_{basis_name}"] = best_scale
                out[f"basis_delta_positive_ratio_{basis_name}"] = best_positive_ratio
    return out


def _p22_pattern_only_metrics(
    *,
    adapter_out: dict[str, torch.Tensor] | None,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    prefix: str = "p22_pattern_only",
) -> dict[str, float]:
    if adapter_out is None or not isinstance(adapter_out.get("pattern_evidence"), torch.Tensor):
        return {f"{prefix}_acc": 0.0, f"{prefix}_macro_f1": 0.0}
    logits = adapter_out["pattern_evidence"]
    metrics = split_metrics(logits.detach(), labels.to(device=logits.device), mask.to(device=logits.device), num_classes=num_classes)
    return {f"{prefix}_acc": metrics["acc"], f"{prefix}_macro_f1": metrics["macro_f1"]}


def _p22_basis_teacher_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    basis_evidence = adapter_out.get("basis_evidence")
    student_basis = adapter_out.get("student_basis")
    if not isinstance(basis_evidence, torch.Tensor) or not isinstance(student_basis, torch.Tensor):
        zero = base_logits.new_tensor(0.0)
        return zero, {
            "p22_basis_teacher_loss": 0.0,
            "p22_basis_teacher_count": 0.0,
            "p22_basis_teacher_entropy": 0.0,
            "p22_basis_teacher_student_kl": 0.0,
            "p22_basis_teacher_agreement": 0.0,
        }
    mask = mask.to(device=base_logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        zero = base_logits.new_tensor(0.0)
        return zero, {
            "p22_basis_teacher_loss": 0.0,
            "p22_basis_teacher_count": 0.0,
            "p22_basis_teacher_entropy": 0.0,
            "p22_basis_teacher_student_kl": 0.0,
            "p22_basis_teacher_agreement": 0.0,
        }
    y = labels.to(device=base_logits.device, dtype=torch.long)
    base = base_logits.detach()
    with torch.no_grad():
        ce_no = F.cross_entropy(base[idx], y[idx], reduction="none")
        deltas: list[torch.Tensor] = []
        for basis_idx in range(int(basis_evidence.size(1))):
            logits_b = base + float(scale) * basis_evidence[:, basis_idx, :].detach()
            ce_b = F.cross_entropy(logits_b[idx], y[idx], reduction="none")
            deltas.append(ce_no - ce_b)
        basis_delta = torch.stack(deltas, dim=-1)
        centered = basis_delta - basis_delta.mean(dim=-1, keepdim=True)
        std = centered.std(dim=-1, keepdim=True).clamp_min(1e-8)
        teacher = torch.softmax(centered / std / max(float(temperature), 1e-6), dim=-1)
    student = student_basis[idx].clamp_min(1e-12)
    loss = -(teacher * student.log()).sum(dim=-1).mean()
    teacher_entropy = -(teacher * teacher.clamp_min(1e-12).log()).sum(dim=-1).mean()
    if teacher.size(-1) > 1:
        teacher_entropy = teacher_entropy / math.log(float(teacher.size(-1)))
    student_kl = (teacher * (teacher.clamp_min(1e-12).log() - student.log())).sum(dim=-1).mean()
    agreement = (teacher.argmax(dim=-1) == student.argmax(dim=-1)).to(dtype=base_logits.dtype).mean()
    return loss, {
        "p22_basis_teacher_loss": float(loss.detach().item()),
        "p22_basis_teacher_count": float(idx.numel()),
        "p22_basis_teacher_entropy": float(teacher_entropy.detach().item()),
        "p22_basis_teacher_student_kl": float(student_kl.detach().item()),
        "p22_basis_teacher_agreement": float(agreement.detach().item()),
    }


def _p22_crossfit_masks(
    *,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_folds: int,
    fold_idx: int,
    seed: int,
    epoch: int,
    resample_each_epoch: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_mask = train_mask.to(device=labels.device, dtype=torch.bool)
    train_idx = torch.where(train_mask)[0]
    if train_idx.numel() == 0:
        return train_mask, train_mask
    k = max(2, min(int(num_folds), int(train_idx.numel())))
    gen = torch.Generator(device="cpu")
    epoch_offset = int(epoch) if resample_each_epoch else 0
    gen.manual_seed(int(seed) * 1000003 + epoch_offset * 9176 + 31)
    folds: list[list[int]] = [[] for _ in range(k)]
    labels_cpu = labels.detach().cpu()
    train_idx_cpu = train_idx.detach().cpu()
    for class_id in torch.unique(labels_cpu[train_idx_cpu]).tolist():
        class_idx = train_idx_cpu[labels_cpu[train_idx_cpu] == int(class_id)]
        perm = class_idx[torch.randperm(class_idx.numel(), generator=gen)]
        for pos, node in enumerate(perm.tolist()):
            folds[pos % k].append(int(node))
    heldout = folds[int(fold_idx) % k]
    query_mask = torch.zeros_like(train_mask)
    if heldout:
        query_mask[torch.tensor(heldout, dtype=torch.long, device=labels.device)] = True
    if int(query_mask.sum().item()) == 0:
        query_mask[train_idx[int(fold_idx) % int(train_idx.numel())]] = True
    support_mask = train_mask & ~query_mask
    if int(support_mask.sum().item()) == 0:
        support_mask = train_mask
    return support_mask, query_mask


def _p22_deployment_losses(
    *,
    adapter_out: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    gate_margin: float,
    gate_target_mode: str,
    gate_positive_margin: float,
    gate_negative_margin: float,
    gate_ignore_neutral: bool,
    anti_harm_margin: float,
    gain_cap: float,
    gate_budget_max: float,
    gate_budget_warmup_epochs: int,
    epoch: int,
    gate_harm_negative_margin: float,
    gate_use_crossfit_stability: bool = False,
    gate_stability_helpful_rate: torch.Tensor | None = None,
    gate_stability_harmful_rate: torch.Tensor | None = None,
    gate_stability_seen: torch.Tensor | None = None,
    gate_helpful_stability_threshold: float = 0.70,
    gate_harmful_stability_threshold: float = 0.50,
    gate_stability_min_seen: int = 2,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    logits = adapter_out.get("logits")
    ungated_logits = adapter_out.get("ungated_logits")
    gate = adapter_out.get("effective_reliability_gate", adapter_out.get("gate"))
    gate_logit = adapter_out.get("reliability_gate_logit")
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(ungated_logits, torch.Tensor)
        or not isinstance(gate, torch.Tensor)
        or not isinstance(gate_logit, torch.Tensor)
    ):
        zero = base_logits.new_tensor(0.0)
        return (
            {"deployment": zero, "gate": zero, "anti_harm": zero, "gain_reward": zero, "gate_budget": zero, "gate_harm": zero},
            {
                "p22_deployment_loss": 0.0,
                "p22_gate_bce_loss": 0.0,
                "p22_anti_harm_loss": 0.0,
                "p22_gain_reward_loss": 0.0,
                "p22_train_delta_ce": 0.0,
                "p22_train_positive_delta_ratio": 0.0,
                "p22_harmful_delta_ratio": 0.0,
                "p22_large_harm_ratio": 0.0,
                "p22_gate_target_mean": 0.0,
                "p22_gate_target_positive_ratio": 0.0,
                "p22_gate_target_negative_ratio": 0.0,
                "p22_gate_target_ignore_ratio": 0.0,
                "p22_gate_target_valid_ratio": 0.0,
                "p22_gate_target_margin_pos": float(gate_positive_margin),
                "p22_gate_target_margin_neg": float(gate_negative_margin),
                "p22_gate_budget_loss": 0.0,
                "p22_gate_harm_loss": 0.0,
                "p22_gate_stability_enabled": 0.0,
                "p22_gate_stability_seen_mean": 0.0,
                "p22_gate_stability_enough_ratio": 0.0,
                "p22_gate_prompt_helpful_mean": 0.0,
                "p22_gate_prompt_harmful_mean": 0.0,
                "p22_gate_base_correct_mean": 0.0,
                "p22_gate_base_wrong_mean": 0.0,
                "p22_crossfit_delta_ce": 0.0,
                "p22_crossfit_positive_delta_ratio": 0.0,
                "p22_crossfit_harmful_ratio": 0.0,
            },
        )
    mask = mask.to(device=logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        zero = logits.new_tensor(0.0)
        return (
            {"deployment": zero, "gate": zero, "anti_harm": zero, "gain_reward": zero, "gate_budget": zero, "gate_harm": zero},
            {
                "p22_deployment_loss": 0.0,
                "p22_gate_bce_loss": 0.0,
                "p22_anti_harm_loss": 0.0,
                "p22_gain_reward_loss": 0.0,
                "p22_train_delta_ce": 0.0,
                "p22_train_positive_delta_ratio": 0.0,
                "p22_harmful_delta_ratio": 0.0,
                "p22_large_harm_ratio": 0.0,
                "p22_gate_target_mean": 0.0,
                "p22_gate_target_positive_ratio": 0.0,
                "p22_gate_target_negative_ratio": 0.0,
                "p22_gate_target_ignore_ratio": 0.0,
                "p22_gate_target_valid_ratio": 0.0,
                "p22_gate_target_margin_pos": float(gate_positive_margin),
                "p22_gate_target_margin_neg": float(gate_negative_margin),
                "p22_gate_budget_loss": 0.0,
                "p22_gate_harm_loss": 0.0,
                "p22_gate_stability_enabled": 0.0,
                "p22_gate_stability_seen_mean": 0.0,
                "p22_gate_stability_enough_ratio": 0.0,
                "p22_gate_prompt_helpful_mean": 0.0,
                "p22_gate_prompt_harmful_mean": 0.0,
                "p22_gate_base_correct_mean": 0.0,
                "p22_gate_base_wrong_mean": 0.0,
                "p22_crossfit_delta_ce": 0.0,
                "p22_crossfit_positive_delta_ratio": 0.0,
                "p22_crossfit_harmful_ratio": 0.0,
            },
        )
    y = labels.to(device=logits.device, dtype=torch.long)
    base = base_logits.detach().to(device=logits.device, dtype=logits.dtype)
    base_ce = F.cross_entropy(base[idx], y[idx], reduction="none")
    prompt_ce = F.cross_entropy(logits[idx], y[idx], reduction="none")
    delta = base_ce - prompt_ce
    with torch.no_grad():
        ungated_ce = F.cross_entropy(ungated_logits.detach()[idx], y[idx], reduction="none")
        candidate_delta = base_ce - ungated_ce
        if gate_target_mode == "tri_state":
            use_stability = (
                bool(gate_use_crossfit_stability)
                and isinstance(gate_stability_helpful_rate, torch.Tensor)
                and isinstance(gate_stability_harmful_rate, torch.Tensor)
                and isinstance(gate_stability_seen, torch.Tensor)
            )
            if use_stability:
                helpful_rate = gate_stability_helpful_rate.to(device=logits.device, dtype=logits.dtype)[idx]
                harmful_rate = gate_stability_harmful_rate.to(device=logits.device, dtype=logits.dtype)[idx]
                seen = gate_stability_seen.to(device=logits.device, dtype=logits.dtype)[idx]
                enough_seen = seen >= float(gate_stability_min_seen)
                positive = enough_seen & (helpful_rate > float(gate_helpful_stability_threshold))
                negative = enough_seen & (harmful_rate > float(gate_harmful_stability_threshold))
            else:
                seen = candidate_delta.new_ones(candidate_delta.numel())
                enough_seen = torch.ones_like(candidate_delta, dtype=torch.bool)
                positive = candidate_delta > float(gate_positive_margin)
                negative = candidate_delta < float(gate_negative_margin)
            valid = positive | negative
            if not gate_ignore_neutral:
                valid = torch.ones_like(positive, dtype=torch.bool)
                negative = ~positive
            target = positive.to(dtype=logits.dtype)
        elif gate_target_mode == "binary":
            positive = candidate_delta > float(gate_margin)
            negative = ~positive
            valid = torch.ones_like(positive, dtype=torch.bool)
            target = positive.to(dtype=logits.dtype)
            seen = candidate_delta.new_ones(candidate_delta.numel())
            enough_seen = torch.ones_like(candidate_delta, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported p22_gate_target_mode={gate_target_mode!r}")
    if bool(valid.any()):
        raw_loss = F.binary_cross_entropy_with_logits(gate_logit[idx][valid], target[valid], reduction="none")
        pos_count = target[valid].sum().clamp_min(1.0)
        neg_count = (1.0 - target[valid]).sum().clamp_min(1.0)
        weights = torch.where(target[valid] > 0.5, neg_count / pos_count, torch.ones_like(target[valid]))
        gate_loss = (raw_loss * weights).mean()
    else:
        gate_loss = logits.new_tensor(0.0)
    deployment = prompt_ce.mean()
    anti_harm = F.relu(float(anti_harm_margin) - delta).mean()
    gain = -delta.clamp_min(0.0).clamp_max(float(gain_cap)).mean()
    if int(epoch) > int(gate_budget_warmup_epochs):
        gate_budget = F.relu(gate[idx].mean() - float(gate_budget_max)).pow(2)
    else:
        gate_budget = logits.new_tensor(0.0)
    harmful_for_gate = candidate_delta.detach() < float(gate_harm_negative_margin)
    gate_harm = gate[idx][harmful_for_gate].mean() if bool(harmful_for_gate.any()) else logits.new_tensor(0.0)
    base_correct = base[idx].argmax(dim=-1) == y[idx]
    harmful = delta < 0.0
    large_harm = delta < -abs(float(anti_harm_margin) if anti_harm_margin != 0 else float(gate_margin))

    def masked_mean(values: torch.Tensor, item_mask: torch.Tensor) -> float:
        if not bool(item_mask.any()):
            return 0.0
        return float(values[item_mask].detach().mean().item())

    stats = {
        "p22_deployment_loss": float(deployment.detach().item()),
        "p22_gate_bce_loss": float(gate_loss.detach().item()),
        "p22_anti_harm_loss": float(anti_harm.detach().item()),
        "p22_gain_reward_loss": float(gain.detach().item()),
        "p22_train_delta_ce": float(delta.detach().mean().item()),
        "p22_train_positive_delta_ratio": float((delta > 0.0).to(dtype=logits.dtype).mean().item()),
        "p22_harmful_delta_ratio": float(harmful.to(dtype=logits.dtype).mean().item()),
        "p22_large_harm_ratio": float(large_harm.to(dtype=logits.dtype).mean().item()),
        "p22_gate_target_mean": float(target.detach().mean().item()),
        "p22_gate_target_positive_ratio": float(positive.to(dtype=logits.dtype).mean().item()),
        "p22_gate_target_negative_ratio": float(negative.to(dtype=logits.dtype).mean().item()),
        "p22_gate_target_ignore_ratio": float((~valid).to(dtype=logits.dtype).mean().item()),
        "p22_gate_target_valid_ratio": float(valid.to(dtype=logits.dtype).mean().item()),
        "p22_gate_target_margin_pos": float(gate_positive_margin),
        "p22_gate_target_margin_neg": float(gate_negative_margin),
        "p22_gate_budget_loss": float(gate_budget.detach().item()),
        "p22_gate_harm_loss": float(gate_harm.detach().item()),
        "p22_gate_stability_enabled": float(bool(gate_use_crossfit_stability)),
        "p22_gate_stability_seen_mean": float(seen.detach().mean().item()),
        "p22_gate_stability_enough_ratio": float(enough_seen.to(dtype=logits.dtype).mean().item()),
        "p22_gate_prompt_helpful_mean": masked_mean(gate[idx], positive),
        "p22_gate_prompt_harmful_mean": masked_mean(gate[idx], candidate_delta <= 0.0),
        "p22_gate_base_correct_mean": masked_mean(gate[idx], base_correct),
        "p22_gate_base_wrong_mean": masked_mean(gate[idx], ~base_correct),
        "p22_crossfit_delta_ce": float(delta.detach().mean().item()),
        "p22_crossfit_positive_delta_ratio": float((delta > 0.0).to(dtype=logits.dtype).mean().item()),
        "p22_crossfit_harmful_ratio": float(harmful.to(dtype=logits.dtype).mean().item()),
    }
    return {
        "deployment": deployment,
        "gate": gate_loss,
        "anti_harm": anti_harm,
        "gain_reward": gain,
        "gate_budget": gate_budget,
        "gate_harm": gate_harm,
    }, stats


@torch.no_grad()
def _p22_ungated_candidate_delta(
    *,
    adapter_out: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ungated_logits = adapter_out.get("ungated_logits")
    if not isinstance(ungated_logits, torch.Tensor):
        return base_logits.new_empty(0, dtype=torch.long), base_logits.new_empty(0)
    mask = mask.to(device=ungated_logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return idx, ungated_logits.new_empty(0)
    y = labels.to(device=ungated_logits.device, dtype=torch.long)
    base = base_logits.detach().to(device=ungated_logits.device, dtype=ungated_logits.dtype)
    base_ce = F.cross_entropy(base[idx], y[idx], reduction="none")
    ungated_ce = F.cross_entropy(ungated_logits.detach()[idx], y[idx], reduction="none")
    return idx, base_ce - ungated_ce


def _p22_grad_diagnostics(
    prompt_adapter_module: torch.nn.Module | None,
    *,
    token_before: torch.Tensor | None = None,
) -> dict[str, float]:
    if prompt_adapter_module is None or not isinstance(prompt_adapter_module, P22ClassPatternEnrichmentBank):
        return {
            "p22_pattern_token_grad_norm": 0.0,
            "p22_pattern_encoder_grad_norm": 0.0,
            "p22_pattern_basis_grad_norm": 0.0,
            "p22_reliability_gate_grad_norm": 0.0,
            "p22_pattern_token_update_norm": 0.0,
        }

    def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
        grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        return float(torch.stack(grads).norm().item())

    token_params = [prompt_adapter_module.pattern_tokens]
    encoder_params = [p for p in prompt_adapter_module.pattern_encoder.parameters()]
    basis_params = [prompt_adapter_module.pattern_basis_logits]
    reliability_module = getattr(prompt_adapter_module, "reliability_gate", None)
    reliability_params = [p for p in reliability_module.parameters()] if reliability_module is not None else []
    update_norm = 0.0
    if token_before is not None:
        update_norm = float((prompt_adapter_module.pattern_tokens.detach() - token_before.to(prompt_adapter_module.pattern_tokens.device)).norm().item())
    return {
        "p22_pattern_token_grad_norm": grad_norm(token_params),
        "p22_pattern_encoder_grad_norm": grad_norm(encoder_params),
        "p22_pattern_basis_grad_norm": grad_norm(basis_params),
        "p22_reliability_gate_grad_norm": grad_norm(reliability_params),
        "p22_pattern_token_update_norm": update_norm,
    }


def _prompt_adapter_candidate_pool_delta_stats(
    *,
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    split_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    prefix: str,
) -> dict[str, Any]:
    if candidate_mask is None:
        return {}
    split = split_mask.to(device=logits_prompt.device, dtype=torch.bool)
    candidate = candidate_mask.to(device=logits_prompt.device, dtype=torch.bool)
    inside = split & candidate
    outside = split & ~candidate
    out: dict[str, Any] = {}
    out.update(
        _prompt_adapter_delta_stats(
            logits_prompt=logits_prompt,
            logits_no_prompt=logits_no_prompt,
            labels=labels,
            mask=inside,
            prefix=f"{prefix}_candidate_pool",
        )
    )
    out.update(
        _prompt_adapter_delta_stats(
            logits_prompt=logits_prompt,
            logits_no_prompt=logits_no_prompt,
            labels=labels,
            mask=outside,
            prefix=f"{prefix}_outside_candidate_pool",
        )
    )
    return out


def _prompt_router_diagnostics(adapter_out: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    """Router-specific usage diagnostics for the p20 pattern prompt router.

    Returns no router keys when ``adapter_out`` is from a non-router module
    (so the shared adapter diagnostics remain unaffected).
    """
    if adapter_out is None or "pattern_weights" not in adapter_out:
        return {}

    def scalar(name: str) -> float:
        value = adapter_out.get(name)
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().item())
        return 0.0

    q = adapter_out.get("pattern_weights")
    a = adapter_out.get("easy_prob")
    p = adapter_out.get("soft_class_prob")
    out: dict[str, Any] = {
        "prompt_router_enabled": 1.0,
        "class_prompt_usage_ratio": scalar("class_prompt_usage"),
        "heterophily_prompt_usage_ratio": scalar("hetero_prompt_usage"),
        "pattern_usage_entropy": scalar("pattern_usage_entropy"),
        "prompt_message_norm": scalar("prompt_delta_norm"),
        "gate_mean": scalar("prompt_gate_mean"),
    }
    usage = adapter_out.get("pattern_usage_mean")
    if isinstance(usage, torch.Tensor):
        out["pattern_usage_mean"] = [float(v) for v in usage.detach().tolist()]
    # pattern_usage_by_class: average pattern distribution per predicted class.
    if isinstance(q, torch.Tensor) and isinstance(p, torch.Tensor) and q.numel() > 0:
        pred_class = p.detach().argmax(dim=-1)
        by_class: dict[str, list[float]] = {}
        for class_id in torch.unique(pred_class).tolist():
            class_mask = pred_class == int(class_id)
            by_class[str(int(class_id))] = [float(v) for v in q.detach()[class_mask].mean(dim=0).tolist()]
        out["pattern_usage_by_class"] = by_class
    return out


def _prompt_router_delta_breakdown(
    *,
    adapter_out: dict[str, torch.Tensor] | None,
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    """delta_CE broken down by routed pattern, class-vs-hetero prompt, and gate-delta corr."""
    if adapter_out is None or "pattern_weights" not in adapter_out:
        return {}
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return {
            f"{prefix}_delta_ce_by_pattern": {},
            f"{prefix}_delta_ce_class_prompt": 0.0,
            f"{prefix}_delta_ce_hetero_prompt": 0.0,
            f"{prefix}_gate_delta_corr": 0.0,
        }
    y = labels.to(device=logits_prompt.device)[idx]
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt.detach()[idx], y, reduction="none")
    delta = ce_no - ce_prompt

    out: dict[str, Any] = {}
    q = adapter_out["pattern_weights"].detach()[idx]
    routed_pattern = q.argmax(dim=-1)
    by_pattern: dict[str, dict[str, float]] = {}
    for pattern_id in torch.unique(routed_pattern).tolist():
        pattern_mask = routed_pattern == int(pattern_id)
        name = PATTERN_NAMES[int(pattern_id)] if int(pattern_id) < len(PATTERN_NAMES) else str(int(pattern_id))
        by_pattern[name] = {
            "count": float(pattern_mask.sum().item()),
            "mean_delta_ce": float(delta[pattern_mask].mean().item()),
            "positive_delta_ratio": float((delta[pattern_mask] > 0.0).float().mean().item()),
        }
    out[f"{prefix}_delta_ce_by_pattern"] = by_pattern

    a = adapter_out.get("easy_prob")
    if isinstance(a, torch.Tensor):
        a_sel = a.detach()[idx]
        class_routed = a_sel >= 0.5
        hetero_routed = ~class_routed
        out[f"{prefix}_delta_ce_class_prompt"] = (
            float(delta[class_routed].mean().item()) if bool(class_routed.any()) else 0.0
        )
        out[f"{prefix}_delta_ce_hetero_prompt"] = (
            float(delta[hetero_routed].mean().item()) if bool(hetero_routed.any()) else 0.0
        )

    gate = adapter_out.get("gate")
    if isinstance(gate, torch.Tensor):
        gate_sel = gate.detach()[idx]
        if gate_sel.numel() >= 2 and float(gate_sel.std().item()) > 1e-8 and float(delta.std().item()) > 1e-8:
            stacked = torch.stack([gate_sel, delta], dim=0)
            corr = torch.corrcoef(stacked)[0, 1]
            out[f"{prefix}_gate_delta_corr"] = float(corr.item())
        else:
            out[f"{prefix}_gate_delta_corr"] = 0.0
    return out


def prompt_router_pattern_supervision_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    no_prompt_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.05,
    probe_norm: float = 0.08,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Directly supervise the PatternRouter ``q_i`` with per-expert utility.

    For each (query) node we probe every heterophily expert *in isolation* at a
    fixed magnitude ``probe_norm`` and measure how much it reduces CE versus the
    no-prompt branch. The per-pattern utilities form a stop-gradient soft target
    (softmax over patterns); ``q_i`` is trained to match it via cross-entropy.

    Only training (query) labels are used (the teacher never touches val/test).
    The reject pattern (k=0) has utility 0 by construction, so when no expert
    helps the target concentrates on reject, teaching the router to abstain.
    """
    empty = {
        "prompt_router_pattern_supervision_loss": 0.0,
        "prompt_router_pattern_supervision_target_entropy": 0.0,
        "prompt_router_pattern_routing_agreement": 0.0,
        "prompt_router_pattern_supervision_count": 0.0,
    }
    pattern_messages = adapter_out.get("pattern_messages")
    q = adapter_out.get("pattern_weights")
    if not isinstance(pattern_messages, torch.Tensor) or not isinstance(q, torch.Tensor):
        return no_prompt_logits.new_tensor(0.0), dict(empty)
    mask = mask.to(device=no_prompt_logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return no_prompt_logits.new_tensor(0.0), dict(empty)

    y = labels.to(device=no_prompt_logits.device)[idx]
    num_patterns = int(pattern_messages.size(1))
    with torch.no_grad():
        alpha = model.alpha.detach()
        h_pre_sel = h_pre.detach()[idx]
        h_base_sel = h_adp_base.detach()[idx]
        msgs = pattern_messages.detach()[idx]  # [M, K, H]
        norms = msgs.norm(dim=-1, keepdim=True)
        probe = msgs / norms.clamp_min(1e-12) * float(probe_norm)  # fixed-strength what-if
        ce_no = F.cross_entropy(no_prompt_logits.detach()[idx], y, reduction="none")
        utilities = []
        for k in range(num_patterns):
            h_adp_k = h_base_sel + probe[:, k, :]
            h_mix_k = alpha * h_pre_sel + (1.0 - alpha) * h_adp_k
            logits_k = model.classifier(h_mix_k)
            ce_k = F.cross_entropy(logits_k, y, reduction="none")
            utilities.append(ce_no - ce_k)
        utility = torch.stack(utilities, dim=-1)  # [M, K]
        # Per-node ΔCE differences across patterns are tiny in absolute terms;
        # standardise per node so the target reflects the *ranking* of patterns
        # rather than their (vanishing) absolute scale. Otherwise softmax collapses
        # to uniform and the router has nothing to specialise toward.
        util_centered = utility - utility.mean(dim=-1, keepdim=True)
        util_std = util_centered.std(dim=-1, keepdim=True).clamp_min(1e-8)
        util_z = util_centered / util_std
        temp = max(float(temperature), 1e-6)
        target = torch.softmax(util_z / temp, dim=-1)  # stop-gradient soft target

    q_sel = q[idx].clamp_min(1e-12)
    per_node = -(target * q_sel.log()).sum(dim=-1)
    if class_balanced:
        per_class = []
        for class_id in torch.unique(y.detach()).tolist():
            class_mask = y == int(class_id)
            if bool(class_mask.any()):
                per_class.append(per_node[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else per_node.mean()
    else:
        loss = per_node.mean()

    target_entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1)
    if num_patterns > 1:
        target_entropy = target_entropy / math.log(float(num_patterns))
    agreement = (q[idx].argmax(dim=-1) == target.argmax(dim=-1)).float().mean()
    stats = {
        "prompt_router_pattern_supervision_loss": float(loss.detach().item()),
        "prompt_router_pattern_supervision_target_entropy": float(target_entropy.mean().item()),
        "prompt_router_pattern_routing_agreement": float(agreement.item()),
        "prompt_router_pattern_supervision_count": float(idx.numel()),
    }
    return loss, stats


def prompt_router_pattern_utility_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    no_prompt_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.5,
    probe_norm: float = 0.08,
    margin: float = 0.001,
    anti_harm_weight: float = 0.5,
    min_teacher_delta: float = 1e-5,
    helpful_fraction: float = 0.3,
    unhelpful_node_weight: float = 0.1,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train pattern experts with direct per-pattern utility.

    ``prompt_router_pattern_supervision_loss`` teaches the router which pattern
    to choose from a stop-gradient teacher. This loss is complementary: it
    probes each non-reject expert with gradients enabled, so the expert/message
    parameters themselves learn to reduce CE on candidate-pool query nodes.
    """
    empty = {
        "prompt_router_pattern_utility_loss": 0.0,
        "prompt_router_pattern_utility_mean_delta_ce": 0.0,
        "prompt_router_pattern_utility_positive_ratio": 0.0,
        "prompt_router_pattern_utility_harmful_ratio": 0.0,
        "prompt_router_pattern_utility_helpful_node_ratio": 0.0,
        "prompt_router_pattern_utility_count": 0.0,
        "prompt_router_pattern_utility_target_entropy": 0.0,
    }
    pattern_messages = adapter_out.get("pattern_messages")
    if not isinstance(pattern_messages, torch.Tensor) or pattern_messages.numel() == 0:
        return no_prompt_logits.new_tensor(0.0), dict(empty)
    mask = mask.to(device=no_prompt_logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    num_patterns = int(pattern_messages.size(1))
    if idx.numel() == 0 or num_patterns <= 1:
        return no_prompt_logits.new_tensor(0.0), dict(empty)

    y = labels.to(device=no_prompt_logits.device)[idx]
    alpha = model.alpha.detach()
    h_pre_sel = h_pre.detach()[idx]
    h_base_sel = h_adp_base.detach()[idx]
    msgs = pattern_messages[idx]  # [M, K, H], keep gradients for experts.
    norms = msgs.norm(dim=-1, keepdim=True)
    probe = msgs / norms.clamp_min(1e-12) * float(probe_norm)
    ce_no = F.cross_entropy(no_prompt_logits.detach()[idx], y, reduction="none")
    deltas: list[torch.Tensor] = []
    for k in range(1, num_patterns):
        h_adp_k = h_base_sel + probe[:, k, :]
        h_mix_k = alpha * h_pre_sel + (1.0 - alpha) * h_adp_k
        logits_k = model.classifier(h_mix_k)
        ce_k = F.cross_entropy(logits_k, y, reduction="none")
        deltas.append(ce_no - ce_k)
    delta = torch.stack(deltas, dim=-1)  # [M, K-1]

    with torch.no_grad():
        util_centered = delta.detach() - delta.detach().mean(dim=-1, keepdim=True)
        util_std = util_centered.std(dim=-1, keepdim=True).clamp_min(1e-8)
        util_z = util_centered / util_std
        target = torch.softmax(util_z / max(float(temperature), 1e-6), dim=-1)
        max_delta = delta.detach().max(dim=-1).values
        helpful_node = max_delta > float(min_teacher_delta)
        fraction = min(max(float(helpful_fraction), 0.0), 1.0)
        if 0.0 < fraction < 1.0 and max_delta.numel() > 1:
            threshold = torch.quantile(max_delta, 1.0 - fraction)
            helpful_node = helpful_node & (max_delta >= threshold)

    margin_value = float(margin)
    per_pattern = F.relu(margin_value - delta)
    anti_harm = F.relu(-delta).pow(2)
    helpful_loss = (target * (per_pattern + float(anti_harm_weight) * anti_harm)).sum(dim=-1)
    unhelpful_loss = float(unhelpful_node_weight) * anti_harm.mean(dim=-1)
    weighted = torch.where(helpful_node, helpful_loss, unhelpful_loss)

    if class_balanced:
        per_class = []
        for class_id in torch.unique(y.detach()).tolist():
            class_mask = y == int(class_id)
            if bool(class_mask.any()):
                per_class.append(weighted[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else weighted.mean()
    else:
        loss = weighted.mean()

    target_entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1)
    if target.size(1) > 1:
        target_entropy = target_entropy / math.log(float(target.size(1)))
    stats = {
        "prompt_router_pattern_utility_loss": float(loss.detach().item()),
        "prompt_router_pattern_utility_mean_delta_ce": float(delta.detach().mean().item()),
        "prompt_router_pattern_utility_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
        "prompt_router_pattern_utility_harmful_ratio": float((delta.detach() < 0.0).float().mean().item()),
        "prompt_router_pattern_utility_helpful_node_ratio": float(helpful_node.float().mean().item()),
        "prompt_router_pattern_utility_count": float(idx.numel() * (num_patterns - 1)),
        "prompt_router_pattern_utility_target_entropy": float(target_entropy.detach().mean().item()),
    }
    return loss, stats


def prompt_router_class_pattern_reliability_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    no_prompt_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.5,
    probe_norm: float = 0.08,
    positive_margin: float = 1e-5,
    harmful_margin: float = 0.0,
    min_class_count: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Router-level class x pattern reliability supervision.

    Pattern utility is noisy per node. This aggregates the teacher by true
    training class and teaches q_i to avoid patterns that are class-level
    harmful, while concentrating on useful class-pattern combinations when one
    exists. Gradients flow to the router q_i, not through the teacher probes.
    """
    empty = {
        "prompt_router_class_pattern_reliability_loss": 0.0,
        "prompt_router_class_pattern_reliability_count": 0.0,
        "prompt_router_class_pattern_reliable_pair_ratio": 0.0,
        "prompt_router_class_pattern_harmful_pair_ratio": 0.0,
        "prompt_router_class_pattern_target_entropy": 0.0,
        "prompt_router_class_pattern_nonreject_target_mass": 0.0,
    }
    pattern_messages = adapter_out.get("pattern_messages")
    q = adapter_out.get("pattern_weights")
    if not isinstance(pattern_messages, torch.Tensor) or not isinstance(q, torch.Tensor):
        return no_prompt_logits.new_tensor(0.0), dict(empty)
    mask = mask.to(device=no_prompt_logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    num_patterns = int(pattern_messages.size(1))
    if idx.numel() == 0 or num_patterns <= 1:
        return no_prompt_logits.new_tensor(0.0), dict(empty)

    y = labels.to(device=no_prompt_logits.device)[idx]
    with torch.no_grad():
        alpha = model.alpha.detach()
        h_pre_sel = h_pre.detach()[idx]
        h_base_sel = h_adp_base.detach()[idx]
        msgs = pattern_messages.detach()[idx]
        norms = msgs.norm(dim=-1, keepdim=True)
        probe = msgs / norms.clamp_min(1e-12) * float(probe_norm)
        ce_no = F.cross_entropy(no_prompt_logits.detach()[idx], y, reduction="none")
        utilities: list[torch.Tensor] = []
        for k in range(num_patterns):
            h_adp_k = h_base_sel + probe[:, k, :]
            h_mix_k = alpha * h_pre_sel + (1.0 - alpha) * h_adp_k
            logits_k = model.classifier(h_mix_k)
            ce_k = F.cross_entropy(logits_k, y, reduction="none")
            utilities.append(ce_no - ce_k)
        utility = torch.stack(utilities, dim=-1)  # [M, K]

    q_sel = q[idx].clamp_min(1e-12)
    losses: list[torch.Tensor] = []
    target_entropies: list[torch.Tensor] = []
    nonreject_masses: list[torch.Tensor] = []
    reliable_pairs = 0
    harmful_pairs = 0
    total_pairs = 0
    supervised_nodes = 0
    for class_id in torch.unique(y.detach()).tolist():
        class_mask = y == int(class_id)
        class_count = int(class_mask.sum().item())
        if class_count < int(min_class_count):
            continue
        class_utility = utility[class_mask].mean(dim=0)  # [K]
        nonreject_utility = class_utility[1:]
        reliable_pairs += int((nonreject_utility > float(positive_margin)).sum().item())
        harmful = nonreject_utility < -float(harmful_margin)
        harmful_pairs += int(harmful.sum().item())
        total_pairs += int(nonreject_utility.numel())

        if bool((nonreject_utility > float(positive_margin)).any()):
            centered = class_utility - class_utility.mean()
            scaled = centered / centered.std().clamp_min(1e-8)
            target = torch.softmax(scaled / max(float(temperature), 1e-6), dim=-1)
        else:
            target = torch.zeros_like(class_utility)
            target[0] = 1.0
        q_class = q_sel[class_mask]
        ce = -(target.detach().unsqueeze(0) * q_class.log()).sum(dim=-1).mean()
        harmful_penalty = q_class[:, 1:][:, harmful].mean() if bool(harmful.any()) else ce.new_tensor(0.0)
        losses.append(ce + harmful_penalty)
        target_entropy = -(target * target.clamp_min(1e-12).log()).sum()
        if num_patterns > 1:
            target_entropy = target_entropy / math.log(float(num_patterns))
        target_entropies.append(target_entropy.detach())
        nonreject_masses.append(target[1:].sum().detach())
        supervised_nodes += class_count

    if not losses:
        return no_prompt_logits.new_tensor(0.0), dict(empty)
    loss = torch.stack(losses).mean()
    total = max(1, total_pairs)
    stats = {
        "prompt_router_class_pattern_reliability_loss": float(loss.detach().item()),
        "prompt_router_class_pattern_reliability_count": float(supervised_nodes),
        "prompt_router_class_pattern_reliable_pair_ratio": float(reliable_pairs / total),
        "prompt_router_class_pattern_harmful_pair_ratio": float(harmful_pairs / total),
        "prompt_router_class_pattern_target_entropy": float(torch.stack(target_entropies).mean().item()),
        "prompt_router_class_pattern_nonreject_target_mass": float(torch.stack(nonreject_masses).mean().item()),
    }
    return loss, stats


def prompt_router_expert_utility_supervision_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    no_prompt_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    prefix: str = "prompt_router_expert",
    temperature: float = 0.5,
    margin: float = 1e-5,
    target_mode: str = "soft",
    gain_temperature: float | None = None,
    probe_norm: float | None = None,
    class_balanced: bool = True,
    gate_weight: float = 0.0,
    gate_target_mode: str = "binary",
    gate_target_temperature: float | None = None,
    oracle_scale_grid: list[float] | tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise PatternRouter from actual heterophily expert utility.

    This is the minimal P20 utility-supervised teacher.  For each labelled
    train-query node, it evaluates every bounded expert message in isolation:

    ``delta_k = CE_no_prompt - CE_expert_k``.

    Pattern 0 is reject/no-correction.  If no non-reject expert beats
    ``margin``, the target is reject.  Otherwise the router is trained to match
    either a soft utility target or the best expert.  Val/test callers should
    use this function only under ``torch.no_grad()`` and ignore the returned
    loss; labels there are diagnostics only.
    """
    empty = {
        f"{prefix}_loss": 0.0,
        f"{prefix}_count": 0.0,
        f"{prefix}_oracle_best_expert_gain": 0.0,
        f"{prefix}_oracle_positive_ratio": 0.0,
        f"{prefix}_no_prompt_acc": 0.0,
        f"{prefix}_no_prompt_macro_f1": 0.0,
        f"{prefix}_oracle_best_expert_acc": 0.0,
        f"{prefix}_oracle_best_expert_macro_f1": 0.0,
        f"{prefix}_oracle_best_expert_acc_lift_vs_no_prompt": 0.0,
        f"{prefix}_oracle_best_expert_macro_f1_lift_vs_no_prompt": 0.0,
        f"{prefix}_oracle_scaled_best_scale": 0.0,
        f"{prefix}_oracle_scaled_best_gain": 0.0,
        f"{prefix}_oracle_scaled_best_acc": 0.0,
        f"{prefix}_oracle_scaled_best_macro_f1": 0.0,
        f"{prefix}_oracle_scaled_best_acc_lift_vs_no_prompt": 0.0,
        f"{prefix}_oracle_scaled_best_macro_f1_lift_vs_no_prompt": 0.0,
        f"{prefix}_router_accuracy_to_best_expert": 0.0,
        f"{prefix}_router_soft_target_kl": 0.0,
        f"{prefix}_target_entropy": 0.0,
        f"{prefix}_no_correction_ratio": 0.0,
        f"{prefix}_learned_weighted_delta_ce": 0.0,
        f"{prefix}_learned_best_weight_mean": 0.0,
        f"{prefix}_router_loss": 0.0,
        f"{prefix}_gate_supervision_loss": 0.0,
        f"{prefix}_gate_target_mean": 0.0,
        f"{prefix}_gate_target_std": 0.0,
        f"{prefix}_gate_mean": 0.0,
        f"{prefix}_gate_accuracy_to_oracle": 0.0,
    }
    pattern_messages = adapter_out.get("pattern_messages")
    q = adapter_out.get("pattern_weights")
    if not isinstance(pattern_messages, torch.Tensor) or not isinstance(q, torch.Tensor):
        return no_prompt_logits.new_tensor(0.0), dict(empty)
    mask = mask.to(device=no_prompt_logits.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    num_patterns = int(pattern_messages.size(1))
    if idx.numel() == 0 or num_patterns == 0:
        return no_prompt_logits.new_tensor(0.0), dict(empty)

    y = labels.to(device=no_prompt_logits.device, dtype=torch.long)[idx]
    alpha = model.alpha.detach()
    h_pre_sel = h_pre.detach()[idx]
    h_base_sel = h_adp_base.detach()[idx]
    msgs = pattern_messages.detach()[idx]
    if probe_norm is not None and float(probe_norm) > 0.0:
        msgs = msgs / msgs.norm(dim=-1, keepdim=True).clamp_min(1e-12) * float(probe_norm)
    ce_no = F.cross_entropy(no_prompt_logits.detach()[idx], y, reduction="none")
    deltas: list[torch.Tensor] = []
    logits_by_pattern: list[torch.Tensor] = []
    for k in range(num_patterns):
        h_adp_k = h_base_sel + msgs[:, k, :]
        h_mix_k = alpha * h_pre_sel + (1.0 - alpha) * h_adp_k
        logits_k = model.classifier(h_mix_k)
        ce_k = F.cross_entropy(logits_k, y, reduction="none")
        deltas.append(ce_no - ce_k)
        logits_by_pattern.append(logits_k)
    utility = torch.stack(deltas, dim=-1)  # [M, K], includes reject.
    logits_stack = torch.stack(logits_by_pattern, dim=1)  # [M, K, C]

    with torch.no_grad():
        nonreject = utility[:, 1:] if num_patterns > 1 else utility[:, :0]
        if nonreject.numel() > 0:
            nonreject_best_delta, nonreject_best = nonreject.max(dim=-1)
            best_idx = nonreject_best + 1
            reject = nonreject_best_delta <= float(margin)
            best_idx = torch.where(reject, torch.zeros_like(best_idx), best_idx)
        else:
            nonreject_best_delta = utility.new_zeros(idx.numel())
            best_idx = torch.zeros(idx.numel(), dtype=torch.long, device=utility.device)
            reject = torch.ones(idx.numel(), dtype=torch.bool, device=utility.device)
        best_delta = utility.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        target = torch.zeros_like(utility)
        target[reject, 0] = 1.0
        helpful = ~reject
        if bool(helpful.any()):
            mode = str(target_mode)
            if mode == "hard":
                target[helpful, :] = 0.0
                target[helpful, best_idx[helpful]] = 1.0
            elif mode in {"margin_softmax", "utility_margin_softmax"}:
                gain_temp = max(float(gain_temperature if gain_temperature is not None else temperature), 1e-6)
                scores = utility[helpful].clone()
                scores[:, 0] = 0.0
                if scores.size(1) > 1:
                    scores[:, 1:] = (scores[:, 1:] - float(margin)) / gain_temp
                target[helpful] = torch.softmax(scores, dim=-1)
            else:
                util_h = utility[helpful]
                centered = util_h - util_h.mean(dim=-1, keepdim=True)
                scaled = centered / centered.std(dim=-1, keepdim=True).clamp_min(1e-8)
                target[helpful] = torch.softmax(scaled / max(float(temperature), 1e-6), dim=-1)

        oracle_logits = logits_stack[torch.arange(idx.numel(), device=utility.device), best_idx]
        oracle_pred = oracle_logits.argmax(dim=-1)
        oracle_acc = (oracle_pred == y).float().mean()
        no_prompt_pred = no_prompt_logits.detach()[idx].argmax(dim=-1)
        no_prompt_acc = (no_prompt_pred == y).float().mean()
        oracle_macro_scores: list[torch.Tensor] = []
        no_prompt_macro_scores: list[torch.Tensor] = []
        for class_id in range(int(num_classes)):
            true_c = y == int(class_id)
            if not bool(true_c.any()):
                continue
            pred_c = oracle_pred == int(class_id)
            tp = (true_c & pred_c).sum().float()
            precision = tp / pred_c.sum().clamp_min(1).float()
            recall = tp / true_c.sum().clamp_min(1).float()
            denom = precision + recall
            oracle_macro_scores.append(torch.where(denom > 0, 2.0 * precision * recall / denom, denom))

            no_prompt_pred_c = no_prompt_pred == int(class_id)
            no_prompt_tp = (true_c & no_prompt_pred_c).sum().float()
            no_prompt_precision = no_prompt_tp / no_prompt_pred_c.sum().clamp_min(1).float()
            no_prompt_recall = no_prompt_tp / true_c.sum().clamp_min(1).float()
            no_prompt_denom = no_prompt_precision + no_prompt_recall
            no_prompt_macro_scores.append(
                torch.where(
                    no_prompt_denom > 0,
                    2.0 * no_prompt_precision * no_prompt_recall / no_prompt_denom,
                    no_prompt_denom,
                )
            )
        oracle_macro = torch.stack(oracle_macro_scores).mean() if oracle_macro_scores else oracle_acc.new_tensor(0.0)
        no_prompt_macro = (
            torch.stack(no_prompt_macro_scores).mean() if no_prompt_macro_scores else no_prompt_acc.new_tensor(0.0)
        )
        scaled_best = {
            "scale": utility.new_tensor(1.0),
            "gain": best_delta.detach().mean(),
            "acc": oracle_acc.detach(),
            "macro": oracle_macro.detach(),
        }
        scaled_records: dict[float, dict[str, torch.Tensor]] = {}
        if oracle_scale_grid:
            for raw_scale in oracle_scale_grid:
                scale_value = float(raw_scale)
                if scale_value <= 0.0:
                    continue
                scaled_deltas: list[torch.Tensor] = []
                scaled_logits_by_pattern: list[torch.Tensor] = []
                for k in range(num_patterns):
                    h_adp_k = h_base_sel + scale_value * msgs[:, k, :]
                    h_mix_k = alpha * h_pre_sel + (1.0 - alpha) * h_adp_k
                    logits_k = model.classifier(h_mix_k)
                    ce_k = F.cross_entropy(logits_k, y, reduction="none")
                    scaled_deltas.append(ce_no - ce_k)
                    scaled_logits_by_pattern.append(logits_k)
                scaled_utility = torch.stack(scaled_deltas, dim=-1)
                scaled_logits_stack = torch.stack(scaled_logits_by_pattern, dim=1)
                scaled_nonreject = scaled_utility[:, 1:] if num_patterns > 1 else scaled_utility[:, :0]
                if scaled_nonreject.numel() > 0:
                    scaled_nonreject_best_delta, scaled_nonreject_best = scaled_nonreject.max(dim=-1)
                    scaled_best_idx = scaled_nonreject_best + 1
                    scaled_reject = scaled_nonreject_best_delta <= float(margin)
                    scaled_best_idx = torch.where(
                        scaled_reject, torch.zeros_like(scaled_best_idx), scaled_best_idx
                    )
                else:
                    scaled_best_idx = torch.zeros(idx.numel(), dtype=torch.long, device=utility.device)
                scaled_best_delta = scaled_utility.gather(1, scaled_best_idx.unsqueeze(-1)).squeeze(-1)
                scaled_oracle_logits = scaled_logits_stack[
                    torch.arange(idx.numel(), device=utility.device), scaled_best_idx
                ]
                scaled_oracle_pred = scaled_oracle_logits.argmax(dim=-1)
                scaled_oracle_acc = (scaled_oracle_pred == y).float().mean()
                scaled_macro_scores: list[torch.Tensor] = []
                for class_id in range(int(num_classes)):
                    true_c = y == int(class_id)
                    if not bool(true_c.any()):
                        continue
                    pred_c = scaled_oracle_pred == int(class_id)
                    tp = (true_c & pred_c).sum().float()
                    precision = tp / pred_c.sum().clamp_min(1).float()
                    recall = tp / true_c.sum().clamp_min(1).float()
                    denom = precision + recall
                    scaled_macro_scores.append(torch.where(denom > 0, 2.0 * precision * recall / denom, denom))
                scaled_oracle_macro = (
                    torch.stack(scaled_macro_scores).mean()
                    if scaled_macro_scores
                    else scaled_oracle_acc.new_tensor(0.0)
                )
                scaled_gain = scaled_best_delta.detach().mean()
                scaled_records[scale_value] = {
                    "gain": scaled_gain,
                    "acc": scaled_oracle_acc.detach(),
                    "macro": scaled_oracle_macro.detach(),
                }
                current_lift = scaled_oracle_acc.detach() - no_prompt_acc.detach()
                best_lift = scaled_best["acc"] - no_prompt_acc.detach()
                if bool((current_lift > best_lift).item()) or (
                    bool((current_lift == best_lift).item()) and bool((scaled_gain > scaled_best["gain"]).item())
                ):
                    scaled_best = {
                        "scale": utility.new_tensor(scale_value),
                        "gain": scaled_gain,
                        "acc": scaled_oracle_acc.detach(),
                        "macro": scaled_oracle_macro.detach(),
                    }

    q_sel = q[idx].clamp_min(1e-12)
    per_node = -(target.detach() * q_sel.log()).sum(dim=-1)
    if class_balanced:
        per_class = []
        for class_id in torch.unique(y.detach()).tolist():
            class_mask = y == int(class_id)
            if bool(class_mask.any()):
                per_class.append(per_node[class_mask].mean())
        router_loss = torch.stack(per_class).mean() if per_class else per_node.mean()
    else:
        router_loss = per_node.mean()

    gate = adapter_out.get("gate")
    gate_loss = q_sel.new_tensor(0.0)
    gate_mode = str(gate_target_mode)
    if gate_mode in {"margin_sigmoid", "utility_margin_sigmoid"}:
        gate_temp = max(float(gate_target_temperature if gate_target_temperature is not None else margin), 1e-8)
        gate_target = torch.sigmoid((nonreject_best_delta.detach() - float(margin)) / gate_temp)
    else:
        gate_target = (best_idx != 0).detach().float()
    gate_mean = q_sel.new_tensor(0.0)
    gate_acc = q_sel.new_tensor(0.0)
    if isinstance(gate, torch.Tensor) and gate.numel() > 0:
        gate_values = gate.to(device=no_prompt_logits.device)[idx].clamp(1e-6, 1.0 - 1e-6)
        gate_losses = F.binary_cross_entropy(gate_values, gate_target, reduction="none")
        if class_balanced:
            per_class_gate = []
            for class_id in torch.unique(y.detach()).tolist():
                class_mask = y == int(class_id)
                if bool(class_mask.any()):
                    per_class_gate.append(gate_losses[class_mask].mean())
            gate_loss = torch.stack(per_class_gate).mean() if per_class_gate else gate_losses.mean()
        else:
            gate_loss = gate_losses.mean()
        gate_mean = gate_values.detach().mean()
        gate_acc = ((gate_values.detach() >= 0.5) == (gate_target.detach() >= 0.5)).float().mean()
    loss = router_loss + float(gate_weight) * gate_loss

    target_entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1)
    if num_patterns > 1:
        target_entropy = target_entropy / math.log(float(num_patterns))
    q_argmax = q[idx].argmax(dim=-1)
    weighted_delta = (q[idx].detach() * utility.detach()).sum(dim=-1)
    best_q = q[idx].detach().gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    soft_kl = (target.detach() * (target.detach().clamp_min(1e-12).log() - q_sel.log())).sum(dim=-1)

    stats = {
        **empty,
        f"{prefix}_loss": float(loss.detach().item()),
        f"{prefix}_count": float(idx.numel()),
        f"{prefix}_oracle_best_expert_gain": float(best_delta.detach().mean().item()),
        f"{prefix}_oracle_positive_ratio": float((best_delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_no_prompt_acc": float(no_prompt_acc.detach().item()),
        f"{prefix}_no_prompt_macro_f1": float(no_prompt_macro.detach().item()),
        f"{prefix}_oracle_best_expert_acc": float(oracle_acc.detach().item()),
        f"{prefix}_oracle_best_expert_macro_f1": float(oracle_macro.detach().item()),
        f"{prefix}_oracle_best_expert_acc_lift_vs_no_prompt": float((oracle_acc - no_prompt_acc).detach().item()),
        f"{prefix}_oracle_best_expert_macro_f1_lift_vs_no_prompt": float(
            (oracle_macro - no_prompt_macro).detach().item()
        ),
        f"{prefix}_oracle_scaled_best_scale": float(scaled_best["scale"].detach().item()),
        f"{prefix}_oracle_scaled_best_gain": float(scaled_best["gain"].detach().item()),
        f"{prefix}_oracle_scaled_best_acc": float(scaled_best["acc"].detach().item()),
        f"{prefix}_oracle_scaled_best_macro_f1": float(scaled_best["macro"].detach().item()),
        f"{prefix}_oracle_scaled_best_acc_lift_vs_no_prompt": float(
            (scaled_best["acc"] - no_prompt_acc.detach()).item()
        ),
        f"{prefix}_oracle_scaled_best_macro_f1_lift_vs_no_prompt": float(
            (scaled_best["macro"] - no_prompt_macro.detach()).item()
        ),
        f"{prefix}_router_accuracy_to_best_expert": float((q_argmax == best_idx).float().mean().item()),
        f"{prefix}_router_soft_target_kl": float(soft_kl.detach().mean().item()),
        f"{prefix}_target_entropy": float(target_entropy.detach().mean().item()),
        f"{prefix}_no_correction_ratio": float((best_idx == 0).float().mean().item()),
        f"{prefix}_learned_weighted_delta_ce": float(weighted_delta.mean().item()),
        f"{prefix}_learned_best_weight_mean": float(best_q.mean().item()),
        f"{prefix}_router_loss": float(router_loss.detach().item()),
        f"{prefix}_gate_supervision_loss": float(gate_loss.detach().item()),
        f"{prefix}_gate_target_mean": float(gate_target.detach().mean().item()),
        f"{prefix}_gate_target_std": float(
            gate_target.detach().std(unbiased=False).item() if gate_target.numel() > 1 else 0.0
        ),
        f"{prefix}_gate_mean": float(gate_mean.item()),
        f"{prefix}_gate_accuracy_to_oracle": float(gate_acc.item()),
    }
    for scale_value, record in scaled_records.items():
        label = _scale_label(scale_value)
        stats[f"{prefix}_oracle_scale_{label}_gain"] = float(record["gain"].detach().item())
        stats[f"{prefix}_oracle_scale_{label}_acc_lift_vs_no_prompt"] = float(
            (record["acc"] - no_prompt_acc.detach()).item()
        )
        stats[f"{prefix}_oracle_scale_{label}_macro_f1_lift_vs_no_prompt"] = float(
            (record["macro"] - no_prompt_macro.detach()).item()
        )
    for k in range(num_patterns):
        name = PATTERN_NAMES[k] if k < len(PATTERN_NAMES) else f"pattern_{k}"
        stats[f"{prefix}_mean_delta_{name}"] = float(utility[:, k].detach().mean().item())
        stats[f"{prefix}_best_ratio_{name}"] = float((best_idx == k).float().mean().item())
        stats[f"{prefix}_weight_mean_{name}"] = float(q[idx, k].detach().mean().item())
    return loss, stats


def prompt_router_deployment_utility_loss(
    *,
    logits_prompt: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "prompt_router_deployment",
    margin: float = 0.0,
    anti_harm_weight: float = 0.0,
    anti_harm_margin: float = 0.0,
    gain_reward_weight: float = 0.0,
    gain_reward_cap: float | None = None,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the actually deployed prompt path to improve CE over no-prompt."""
    empty = {
        f"{prefix}_loss": 0.0,
        f"{prefix}_count": 0.0,
        f"{prefix}_mean_delta_ce": 0.0,
        f"{prefix}_positive_delta_ratio": 0.0,
        f"{prefix}_harmful_delta_ratio": 0.0,
        f"{prefix}_margin_satisfied_ratio": 0.0,
        f"{prefix}_prompt_ce": 0.0,
        f"{prefix}_no_prompt_ce": 0.0,
        f"{prefix}_anti_harm_loss": 0.0,
        f"{prefix}_gain_reward": 0.0,
    }
    mask = mask.to(device=logits_prompt.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return logits_prompt.new_tensor(0.0), dict(empty)

    y = labels.to(device=logits_prompt.device, dtype=torch.long)[idx]
    ce_prompt = F.cross_entropy(logits_prompt[idx], y, reduction="none")
    ce_no = F.cross_entropy(logits_no_prompt.detach()[idx], y, reduction="none")
    delta = ce_no - ce_prompt
    margin_loss = F.relu(float(margin) - delta)
    anti_harm = F.relu(float(anti_harm_margin) - delta)
    reward = delta.clamp_min(0.0)
    if gain_reward_cap is not None and float(gain_reward_cap) > 0.0:
        reward = reward.clamp_max(float(gain_reward_cap))
    per_node = margin_loss + float(anti_harm_weight) * anti_harm - float(gain_reward_weight) * reward

    if class_balanced:
        per_class = []
        per_class_anti_harm = []
        per_class_reward = []
        for class_id in torch.unique(y.detach()).tolist():
            class_mask = y == int(class_id)
            if bool(class_mask.any()):
                per_class.append(per_node[class_mask].mean())
                per_class_anti_harm.append(anti_harm[class_mask].mean())
                per_class_reward.append(reward[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else per_node.mean()
        anti_harm_loss = (
            torch.stack(per_class_anti_harm).mean() if per_class_anti_harm else anti_harm.mean()
        )
        gain_reward = torch.stack(per_class_reward).mean() if per_class_reward else reward.mean()
    else:
        loss = per_node.mean()
        anti_harm_loss = anti_harm.mean()
        gain_reward = reward.mean()

    stats = {
        **empty,
        f"{prefix}_loss": float(loss.detach().item()),
        f"{prefix}_count": float(idx.numel()),
        f"{prefix}_mean_delta_ce": float(delta.detach().mean().item()),
        f"{prefix}_positive_delta_ratio": float((delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_harmful_delta_ratio": float((delta.detach() < 0.0).float().mean().item()),
        f"{prefix}_margin_satisfied_ratio": float((delta.detach() >= float(margin)).float().mean().item()),
        f"{prefix}_prompt_ce": float(ce_prompt.detach().mean().item()),
        f"{prefix}_no_prompt_ce": float(ce_no.detach().mean().item()),
        f"{prefix}_anti_harm_loss": float(anti_harm_loss.detach().item()),
        f"{prefix}_gain_reward": float(gain_reward.detach().item()),
    }
    return loss, stats


def p21_channel_utility_supervision_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "p21_channel_utility",
    temperature: float = 0.10,
    margin: float = 0.0,
    min_teacher_delta: float = 0.0,
    gate_source: str = "actual",
    target_mode: str = "reject_margin_softmax",
    gate_temperature: float = 0.02,
    gate_margin: float = 0.001,
    gate_target_mode: str = "soft",
    class_balanced: bool = True,
    num_classes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Supervise P21 reject-aware router and gate from channel CE probes."""
    raw_channel_names = adapter_out.get("channel_names", CHANNEL_NAMES)
    channel_names = tuple(str(name) for name in raw_channel_names) if isinstance(raw_channel_names, (list, tuple)) else CHANNEL_NAMES
    if len(channel_names) == 0 or channel_names[0] != "reject":
        channel_names = CHANNEL_NAMES
    empty = {
        f"{prefix}_loss": 0.0,
        f"{prefix}_gate_loss": 0.0,
        f"{prefix}_count": 0.0,
        f"{prefix}_mean_oracle_delta_ce": 0.0,
        f"{prefix}_best_channel_delta_ce": 0.0,
        f"{prefix}_positive_oracle_ratio": 0.0,
        f"{prefix}_best_channel_positive_ratio": 0.0,
        f"{prefix}_routed_delta_ce": 0.0,
        f"{prefix}_routed_positive_ratio": 0.0,
        f"{prefix}_routing_agreement": 0.0,
        f"{prefix}_router_agreement_to_oracle": 0.0,
        f"{prefix}_teacher_entropy": 0.0,
        f"{prefix}_gate_target_mean": 0.0,
        f"{prefix}_gate_target_std": 0.0,
        f"{prefix}_gate_mean": 0.0,
        f"{prefix}_gate_accuracy_to_oracle": 0.0,
        f"{prefix}_best_channel_acc": 0.0,
        f"{prefix}_best_channel_macro_f1": 0.0,
        f"{prefix}_best_channel_acc_lift_vs_no_prompt": 0.0,
        f"{prefix}_best_channel_macro_f1_lift_vs_no_prompt": 0.0,
    }
    for name in channel_names:
        empty[f"{prefix}_{name}_mean_delta_ce"] = 0.0
        empty[f"{prefix}_{name}_best_ratio"] = 0.0
        empty[f"{prefix}_{name}_alpha_mean"] = 0.0

    alpha = adapter_out.get("alpha")
    channel_deltas = adapter_out.get("channel_deltas")
    gate = adapter_out.get("gate")
    if not isinstance(alpha, torch.Tensor) or not isinstance(channel_deltas, torch.Tensor):
        ref = logits_no_prompt if isinstance(logits_no_prompt, torch.Tensor) else h_adp_base
        return ref.new_tensor(0.0), ref.new_tensor(0.0), dict(empty)
    if alpha.size(1) != channel_deltas.size(1):
        return alpha.new_tensor(0.0), alpha.new_tensor(0.0), dict(empty)
    if len(channel_names) != int(channel_deltas.size(1)):
        channel_names = tuple(f"channel_{idx}" for idx in range(int(channel_deltas.size(1))))
        channel_names = ("reject", *channel_names[1:])

    mask = mask.to(device=alpha.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return alpha.new_tensor(0.0), alpha.new_tensor(0.0), dict(empty)

    h_pre = h_pre.to(device=alpha.device)
    h_adp_base = h_adp_base.to(device=alpha.device)
    labels = labels.to(device=alpha.device, dtype=torch.long)
    logits_no_prompt = logits_no_prompt.to(device=alpha.device)
    if isinstance(gate, torch.Tensor) and gate.shape[:1] == alpha.shape[:1]:
        gate_vec = gate.to(device=alpha.device, dtype=h_adp_base.dtype)
    else:
        gate_vec = h_adp_base.new_ones(alpha.size(0))
    if gate_source == "unit":
        gate_vec = torch.ones_like(gate_vec)
    elif gate_source == "max":
        gate_max_tensor = adapter_out.get("gate_max")
        gate_max_value = float(gate_max_tensor.detach().item()) if isinstance(gate_max_tensor, torch.Tensor) else 1.0
        gate_vec = torch.full_like(gate_vec, gate_max_value)

    classifier = model.classifier
    mix_alpha = model.alpha
    with torch.no_grad():
        ce_no = F.cross_entropy(logits_no_prompt[idx], labels[idx], reduction="none")
        channel_ce: list[torch.Tensor] = []
        channel_logits: list[torch.Tensor] = []
        for channel_idx in range(channel_deltas.size(1)):
            h_channel = h_adp_base + gate_vec.unsqueeze(-1) * channel_deltas[:, channel_idx, :]
            logits_channel = classifier(mix_alpha * h_pre + (1.0 - mix_alpha) * h_channel)
            channel_logits.append(logits_channel[idx])
            channel_ce.append(F.cross_entropy(logits_channel[idx], labels[idx], reduction="none"))
        ce_channels = torch.stack(channel_ce, dim=-1)
        logits_channels = torch.stack(channel_logits, dim=1)
        utility = ce_no.unsqueeze(-1) - ce_channels
        utility[:, 0] = 0.0
        nonreject_utility = utility[:, 1:]
        best_nonreject_delta, best_nonreject_offset = nonreject_utility.max(dim=-1)
        threshold = max(float(margin), float(min_teacher_delta))
        helpful = best_nonreject_delta >= threshold
        best_idx = best_nonreject_offset + 1
        best_idx = torch.where(helpful, best_idx, torch.zeros_like(best_idx))
        best_delta = utility.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        if target_mode == "hard_reject_or_best":
            teacher = torch.zeros_like(utility)
            teacher.scatter_(1, best_idx.unsqueeze(-1), 1.0)
        else:
            teacher_logits = torch.cat(
                [
                    utility.new_zeros(utility.size(0), 1),
                    (nonreject_utility - float(margin)) / max(float(temperature), 1e-6),
                ],
                dim=-1,
            )
            teacher = torch.softmax(teacher_logits, dim=-1)
            reject_teacher = torch.zeros_like(teacher)
            reject_teacher[:, 0] = 1.0
            teacher = torch.where(helpful.unsqueeze(-1), teacher, reject_teacher)

        gate_target_arg = (best_nonreject_delta - float(gate_margin)) / max(float(gate_temperature), 1e-6)
        if gate_target_mode == "binary":
            gate_target = (best_nonreject_delta > float(gate_margin)).to(dtype=alpha.dtype)
        else:
            soft_gate_target = torch.sigmoid(gate_target_arg).to(dtype=alpha.dtype)
            # Reject-aware gate: nodes without enough non-reject utility must
            # explicitly learn no-update instead of a soft half-open gate.
            gate_target = torch.where(
                best_nonreject_delta > float(gate_margin),
                soft_gate_target,
                torch.zeros_like(soft_gate_target),
            )

        selected_logits = logits_channels[torch.arange(idx.numel(), device=alpha.device), best_idx]
        y = labels[idx]
        no_prompt_pred = logits_no_prompt[idx].argmax(dim=-1)
        oracle_pred = selected_logits.argmax(dim=-1)

        def acc_and_macro_f1(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            acc = (pred == y).to(dtype=alpha.dtype).mean()
            class_count = int(num_classes) if num_classes is not None else int(logits_no_prompt.size(-1))
            f1_values = []
            for class_id in range(class_count):
                cls = y == class_id
                pred_cls = pred == class_id
                tp = (cls & pred_cls).to(dtype=alpha.dtype).sum()
                fp = (~cls & pred_cls).to(dtype=alpha.dtype).sum()
                fn = (cls & ~pred_cls).to(dtype=alpha.dtype).sum()
                denom = (2.0 * tp + fp + fn).clamp_min(1e-12)
                f1_values.append((2.0 * tp) / denom)
            return acc, torch.stack(f1_values).mean()

        no_prompt_acc, no_prompt_macro = acc_and_macro_f1(no_prompt_pred)
        oracle_acc, oracle_macro = acc_and_macro_f1(oracle_pred)

    log_alpha = alpha[idx].clamp_min(1e-8).log()
    per_node_loss = F.kl_div(log_alpha, teacher, reduction="none").sum(dim=-1)
    routed_delta = (alpha[idx].detach() * utility).sum(dim=-1)
    gate_max_tensor = adapter_out.get("gate_max")
    gate_max_value = float(gate_max_tensor.detach().item()) if isinstance(gate_max_tensor, torch.Tensor) else 1.0
    gate_prob = (gate[idx] / max(gate_max_value, 1e-12)).clamp(1e-6, 1.0 - 1e-6)
    gate_loss_per_node = F.binary_cross_entropy(gate_prob, gate_target, reduction="none")
    if class_balanced:
        per_class = []
        gate_per_class = []
        for class_id in torch.unique(labels[idx].detach()).tolist():
            class_mask = labels[idx] == int(class_id)
            if bool(class_mask.any()):
                per_class.append(per_node_loss[class_mask].mean())
                gate_per_class.append(gate_loss_per_node[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else per_node_loss.mean()
        gate_loss = torch.stack(gate_per_class).mean() if gate_per_class else gate_loss_per_node.mean()
    else:
        loss = per_node_loss.mean()
        gate_loss = gate_loss_per_node.mean()

    teacher_entropy = -(teacher * teacher.clamp_min(1e-12).log()).sum(dim=-1) / math.log(float(len(channel_names)))
    oracle_binary = (best_idx != 0).to(dtype=alpha.dtype)
    gate_binary = (gate_prob.detach() >= 0.5).to(dtype=alpha.dtype)
    stats = {
        **empty,
        f"{prefix}_loss": float(loss.detach().item()),
        f"{prefix}_gate_loss": float(gate_loss.detach().item()),
        f"{prefix}_gate_supervision_loss": float(gate_loss.detach().item()),
        f"{prefix}_count": float(idx.numel()),
        f"{prefix}_mean_oracle_delta_ce": float(best_delta.detach().mean().item()),
        f"{prefix}_best_channel_delta_ce": float(best_delta.detach().mean().item()),
        f"{prefix}_positive_oracle_ratio": float((best_delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_best_channel_positive_ratio": float((best_delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_routed_delta_ce": float(routed_delta.detach().mean().item()),
        f"{prefix}_routed_positive_ratio": float((routed_delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_routing_agreement": float((alpha[idx].detach().argmax(dim=-1) == best_idx).float().mean().item()),
        f"{prefix}_router_agreement_to_oracle": float(
            (alpha[idx].detach().argmax(dim=-1) == best_idx).float().mean().item()
        ),
        f"{prefix}_teacher_entropy": float(teacher_entropy.detach().mean().item()),
        f"{prefix}_gate_target_mean": float(gate_target.detach().mean().item()),
        f"{prefix}_gate_target_std": float(gate_target.detach().std(unbiased=False).item()),
        f"{prefix}_gate_mean": float(gate_prob.detach().mean().item()),
        f"{prefix}_gate_accuracy_to_oracle": float((gate_binary == oracle_binary).float().mean().item()),
        f"{prefix}_best_channel_acc": float(oracle_acc.detach().item()),
        f"{prefix}_best_channel_macro_f1": float(oracle_macro.detach().item()),
        f"{prefix}_best_channel_acc_lift_vs_no_prompt": float((oracle_acc - no_prompt_acc).detach().item()),
        f"{prefix}_best_channel_macro_f1_lift_vs_no_prompt": float(
            (oracle_macro - no_prompt_macro).detach().item()
        ),
    }
    for channel_idx, name in enumerate(channel_names):
        stats[f"{prefix}_{name}_mean_delta_ce"] = float(utility[:, channel_idx].detach().mean().item())
        stats[f"{prefix}_{name}_best_ratio"] = float((best_idx == channel_idx).float().mean().item())
        stats[f"{prefix}_{name}_alpha_mean"] = float(alpha[idx, channel_idx].detach().mean().item())
    is_v2_channel_bank = "compat" in channel_names or "role" in channel_names
    if is_v2_channel_bank and prefix.startswith("p21_oracle_"):
        v2_prefix = prefix.replace("p21_oracle", "p21_v2_oracle", 1)
        for field in (
            "loss",
            "gate_loss",
            "gate_supervision_loss",
            "count",
            "mean_oracle_delta_ce",
            "best_channel_delta_ce",
            "positive_oracle_ratio",
            "best_channel_positive_ratio",
            "routed_delta_ce",
            "routed_positive_ratio",
            "routing_agreement",
            "router_agreement_to_oracle",
            "teacher_entropy",
            "gate_target_mean",
            "gate_target_std",
            "gate_mean",
            "gate_accuracy_to_oracle",
            "best_channel_acc",
            "best_channel_macro_f1",
            "best_channel_acc_lift_vs_no_prompt",
            "best_channel_macro_f1_lift_vs_no_prompt",
        ):
            stats[f"{v2_prefix}_{field}"] = stats[f"{prefix}_{field}"]
        for name in channel_names:
            stats[f"{v2_prefix}_{name}_mean_delta_ce"] = stats[f"{prefix}_{name}_mean_delta_ce"]
            stats[f"{v2_prefix}_{name}_best_ratio"] = stats[f"{prefix}_{name}_best_ratio"]
            stats[f"{v2_prefix}_{name}_alpha_mean"] = stats[f"{prefix}_{name}_alpha_mean"]
    if prefix == "p21_channel_utility":
        stats.update(
            {
                "p21_oracle_best_channel_delta_ce": stats[f"{prefix}_best_channel_delta_ce"],
                "p21_oracle_best_channel_positive_ratio": stats[f"{prefix}_best_channel_positive_ratio"],
                "p21_oracle_best_channel_acc": stats[f"{prefix}_best_channel_acc"],
                "p21_oracle_best_channel_macro_f1": stats[f"{prefix}_best_channel_macro_f1"],
                "p21_oracle_best_channel_acc_lift_vs_no_prompt": stats[
                    f"{prefix}_best_channel_acc_lift_vs_no_prompt"
                ],
                "p21_oracle_best_channel_macro_f1_lift_vs_no_prompt": stats[
                    f"{prefix}_best_channel_macro_f1_lift_vs_no_prompt"
                ],
                "p21_routed_delta_ce": stats[f"{prefix}_routed_delta_ce"],
                "p21_routed_positive_ratio": stats[f"{prefix}_routed_positive_ratio"],
                "p21_router_agreement_to_oracle": stats[f"{prefix}_router_agreement_to_oracle"],
                "p21_teacher_entropy": stats[f"{prefix}_teacher_entropy"],
                "p21_gate_supervision_loss": stats[f"{prefix}_gate_supervision_loss"],
                "p21_gate_target_mean": stats[f"{prefix}_gate_target_mean"],
                "p21_gate_target_std": stats[f"{prefix}_gate_target_std"],
                "p21_gate_accuracy_to_oracle": stats[f"{prefix}_gate_accuracy_to_oracle"],
            }
        )
        for name in channel_names:
            stats[f"p21_{name}_best_ratio"] = stats[f"{prefix}_{name}_best_ratio"]
            stats[f"p21_{name}_mean_delta_ce"] = stats[f"{prefix}_{name}_mean_delta_ce"]
            stats[f"p21_{name}_alpha_mean"] = stats[f"{prefix}_{name}_alpha_mean"]
        if is_v2_channel_bank:
            stats.update(
                {
                    "p21_v2_oracle_best_channel_delta_ce": stats[f"{prefix}_best_channel_delta_ce"],
                    "p21_v2_oracle_best_channel_positive_ratio": stats[f"{prefix}_best_channel_positive_ratio"],
                    "p21_v2_oracle_best_channel_acc": stats[f"{prefix}_best_channel_acc"],
                    "p21_v2_oracle_best_channel_macro_f1": stats[f"{prefix}_best_channel_macro_f1"],
                    "p21_v2_oracle_best_channel_acc_lift_vs_no_prompt": stats[
                        f"{prefix}_best_channel_acc_lift_vs_no_prompt"
                    ],
                    "p21_v2_oracle_best_channel_macro_f1_lift_vs_no_prompt": stats[
                        f"{prefix}_best_channel_macro_f1_lift_vs_no_prompt"
                    ],
                    "p21_v2_routed_delta_ce": stats[f"{prefix}_routed_delta_ce"],
                    "p21_v2_routed_positive_ratio": stats[f"{prefix}_routed_positive_ratio"],
                    "p21_v2_router_agreement_to_oracle": stats[f"{prefix}_router_agreement_to_oracle"],
                    "p21_v2_gate_target_mean": stats[f"{prefix}_gate_target_mean"],
                    "p21_v2_gate_target_std": stats[f"{prefix}_gate_target_std"],
                    "p21_v2_gate_mean": stats[f"{prefix}_gate_mean"],
                    "p21_v2_gate_accuracy_to_oracle": stats[f"{prefix}_gate_accuracy_to_oracle"],
                }
            )
            for name in channel_names:
                stats[f"p21_v2_{name}_best_ratio"] = stats[f"{prefix}_{name}_best_ratio"]
                stats[f"p21_v2_{name}_mean_delta_ce"] = stats[f"{prefix}_{name}_mean_delta_ce"]
                stats[f"p21_v2_{name}_alpha_mean"] = stats[f"{prefix}_{name}_alpha_mean"]
    return loss, gate_loss, stats


def _p21_channel_names(adapter_out: dict[str, Any]) -> tuple[str, ...]:
    raw_channel_names = adapter_out.get("channel_names", CHANNEL_NAMES)
    channel_names = (
        tuple(str(name) for name in raw_channel_names)
        if isinstance(raw_channel_names, (list, tuple))
        else CHANNEL_NAMES
    )
    if len(channel_names) == 0 or channel_names[0] != "reject":
        return CHANNEL_NAMES
    return channel_names


def _p21_acc_and_macro_f1(
    *,
    pred: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    acc = (pred == labels).to(dtype=dtype).mean()
    class_count = int(num_classes) if num_classes is not None else int(pred.max().item() + 1 if pred.numel() else 1)
    f1_values = []
    for class_id in range(class_count):
        cls = labels == class_id
        pred_cls = pred == class_id
        tp = (cls & pred_cls).to(dtype=dtype).sum()
        fp = (~cls & pred_cls).to(dtype=dtype).sum()
        fn = (cls & ~pred_cls).to(dtype=dtype).sum()
        denom = (2.0 * tp + fp + fn).clamp_min(1e-12)
        f1_values.append((2.0 * tp) / denom)
    return acc, torch.stack(f1_values).mean()


def p21_channel_ungated_oracle_diagnostics(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "p21_v2_ungated_oracle",
    scale_grid: list[float] | tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    num_classes: int | None = None,
) -> dict[str, float]:
    """Evaluate true scale-grid expert upper bound without actual gate."""
    channel_names = _p21_channel_names(adapter_out)
    channel_deltas = adapter_out.get("channel_deltas")
    if not isinstance(channel_deltas, torch.Tensor):
        return {f"{prefix}_count": 0.0}
    if len(channel_names) != int(channel_deltas.size(1)):
        channel_names = tuple(f"channel_{idx}" for idx in range(int(channel_deltas.size(1))))
        channel_names = ("reject", *channel_names[1:])

    mask = mask.to(device=channel_deltas.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    empty = {
        f"{prefix}_count": 0.0,
        f"{prefix}_best_channel_delta_ce": 0.0,
        f"{prefix}_best_channel_positive_ratio": 0.0,
        f"{prefix}_best_channel_acc": 0.0,
        f"{prefix}_best_channel_macro_f1": 0.0,
        f"{prefix}_best_channel_acc_lift_vs_no_prompt": 0.0,
        f"{prefix}_best_channel_macro_f1_lift_vs_no_prompt": 0.0,
        f"{prefix}_best_scale": 0.0,
    }
    for name in channel_names:
        empty[f"{prefix}_{name}_best_ratio"] = 0.0
        empty[f"{prefix}_{name}_mean_delta_ce"] = 0.0
    if idx.numel() == 0:
        return dict(empty)

    h_pre = h_pre.to(device=channel_deltas.device)
    h_adp_base = h_adp_base.to(device=channel_deltas.device)
    labels = labels.to(device=channel_deltas.device, dtype=torch.long)
    logits_no_prompt = logits_no_prompt.to(device=channel_deltas.device)
    scale_values = [float(v) for v in scale_grid if float(v) > 0.0]
    if not scale_values:
        scale_values = [1.0]

    classifier = model.classifier
    mix_alpha = model.alpha
    with torch.no_grad():
        ce_no = F.cross_entropy(logits_no_prompt[idx], labels[idx], reduction="none")
        candidate_delta: list[torch.Tensor] = [ce_no.new_zeros(idx.numel())]
        candidate_logits: list[torch.Tensor] = [logits_no_prompt[idx]]
        candidate_channel_idx: list[int] = [0]
        candidate_scale: list[float] = [0.0]
        channel_mean_delta = ce_no.new_zeros(channel_deltas.size(1))
        for channel_idx in range(1, int(channel_deltas.size(1))):
            channel_scale_deltas = []
            for scale in scale_values:
                h_channel = h_adp_base + float(scale) * channel_deltas[:, channel_idx, :]
                logits_channel = classifier(mix_alpha * h_pre + (1.0 - mix_alpha) * h_channel)
                ce_channel = F.cross_entropy(logits_channel[idx], labels[idx], reduction="none")
                delta = ce_no - ce_channel
                candidate_delta.append(delta)
                candidate_logits.append(logits_channel[idx])
                candidate_channel_idx.append(channel_idx)
                candidate_scale.append(float(scale))
                channel_scale_deltas.append(delta)
            channel_mean_delta[channel_idx] = torch.stack(channel_scale_deltas, dim=-1).max(dim=-1).values.mean()
        delta_matrix = torch.stack(candidate_delta, dim=-1)
        best_delta, best_candidate = delta_matrix.max(dim=-1)
        logits_matrix = torch.stack(candidate_logits, dim=1)
        selected_logits = logits_matrix[torch.arange(idx.numel(), device=idx.device), best_candidate]
        best_channel = torch.tensor(candidate_channel_idx, device=idx.device, dtype=torch.long)[best_candidate]
        best_scale = torch.tensor(candidate_scale, device=idx.device, dtype=ce_no.dtype)[best_candidate]
        y = labels[idx]
        no_prompt_pred = logits_no_prompt[idx].argmax(dim=-1)
        oracle_pred = selected_logits.argmax(dim=-1)
        no_prompt_acc, no_prompt_macro = _p21_acc_and_macro_f1(
            pred=no_prompt_pred,
            labels=y,
            num_classes=num_classes,
            dtype=ce_no.dtype,
        )
        oracle_acc, oracle_macro = _p21_acc_and_macro_f1(
            pred=oracle_pred,
            labels=y,
            num_classes=num_classes,
            dtype=ce_no.dtype,
        )

    stats = {
        **empty,
        f"{prefix}_count": float(idx.numel()),
        f"{prefix}_best_channel_delta_ce": float(best_delta.detach().mean().item()),
        f"{prefix}_best_channel_positive_ratio": float((best_delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_best_channel_acc": float(oracle_acc.detach().item()),
        f"{prefix}_best_channel_macro_f1": float(oracle_macro.detach().item()),
        f"{prefix}_best_channel_acc_lift_vs_no_prompt": float((oracle_acc - no_prompt_acc).detach().item()),
        f"{prefix}_best_channel_macro_f1_lift_vs_no_prompt": float((oracle_macro - no_prompt_macro).detach().item()),
        f"{prefix}_best_scale": float(best_scale.detach().mean().item()),
    }
    for channel_idx, name in enumerate(channel_names):
        stats[f"{prefix}_{name}_best_ratio"] = float((best_channel == channel_idx).float().mean().item())
        stats[f"{prefix}_{name}_mean_delta_ce"] = float(channel_mean_delta[channel_idx].detach().item())
    return stats


def p21_channel_expert_utility_loss(
    *,
    adapter_out: dict[str, torch.Tensor],
    model: DualBranchFramework,
    h_pre: torch.Tensor,
    h_adp_base: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "p21_channel_expert",
    probe_scale: float = 1.0,
    temperature: float = 0.10,
    margin: float = 0.001,
    anti_harm_weight: float = 0.5,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Directly train non-reject expert deltas to reduce labeled-query CE."""
    channel_names = _p21_channel_names(adapter_out)
    channel_deltas = adapter_out.get("channel_deltas")
    ref = h_adp_base if isinstance(h_adp_base, torch.Tensor) else logits_no_prompt
    empty = {
        f"{prefix}_loss": 0.0,
        f"{prefix}_count": 0.0,
        f"{prefix}_mean_delta_ce": 0.0,
        f"{prefix}_positive_ratio": 0.0,
        f"{prefix}_best_delta_ce": 0.0,
        f"{prefix}_anti_harm_loss": 0.0,
    }
    for name in channel_names[1:]:
        empty[f"{prefix}_best_channel_ratio_{name}"] = 0.0
        empty[f"{prefix}_{name}_mean_delta_ce"] = 0.0
    if not isinstance(channel_deltas, torch.Tensor) or channel_deltas.size(1) <= 1:
        return ref.new_tensor(0.0), dict(empty)
    if len(channel_names) != int(channel_deltas.size(1)):
        channel_names = tuple(f"channel_{idx}" for idx in range(int(channel_deltas.size(1))))
        channel_names = ("reject", *channel_names[1:])

    mask = mask.to(device=channel_deltas.device, dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return channel_deltas.new_tensor(0.0), dict(empty)
    h_pre_d = h_pre.detach().to(device=channel_deltas.device)
    h_adp_base_d = h_adp_base.detach().to(device=channel_deltas.device)
    logits_no_prompt_d = logits_no_prompt.detach().to(device=channel_deltas.device)
    labels = labels.to(device=channel_deltas.device, dtype=torch.long)
    mix_alpha = model.alpha.detach()
    classifier = model.classifier
    ce_channels = []
    logits_channels = []
    for channel_idx in range(1, int(channel_deltas.size(1))):
        h_channel = h_adp_base_d + float(probe_scale) * channel_deltas[:, channel_idx, :]
        logits_channel = classifier(mix_alpha * h_pre_d + (1.0 - mix_alpha) * h_channel)
        logits_channels.append(logits_channel[idx])
        ce_channels.append(F.cross_entropy(logits_channel[idx], labels[idx], reduction="none"))
    ce_stack = torch.stack(ce_channels, dim=-1)
    with torch.no_grad():
        weights = torch.softmax(-ce_stack / max(float(temperature), 1e-6), dim=-1)
        ce_no = F.cross_entropy(logits_no_prompt_d[idx], labels[idx], reduction="none")
        delta = ce_no.unsqueeze(-1) - ce_stack.detach()
        best_delta, best_offset = delta.max(dim=-1)
    expert_per_node = (weights * ce_stack).sum(dim=-1)
    anti_harm = F.relu(float(margin) - (ce_no.unsqueeze(-1) - ce_stack))
    per_node = expert_per_node + float(anti_harm_weight) * anti_harm.mean(dim=-1)
    if class_balanced:
        per_class = []
        per_class_anti = []
        for class_id in torch.unique(labels[idx].detach()).tolist():
            class_mask = labels[idx] == int(class_id)
            if bool(class_mask.any()):
                per_class.append(per_node[class_mask].mean())
                per_class_anti.append(anti_harm[class_mask].mean())
        loss = torch.stack(per_class).mean() if per_class else per_node.mean()
        anti_harm_loss = torch.stack(per_class_anti).mean() if per_class_anti else anti_harm.mean()
    else:
        loss = per_node.mean()
        anti_harm_loss = anti_harm.mean()

    stats = {
        **empty,
        f"{prefix}_loss": float(loss.detach().item()),
        f"{prefix}_count": float(idx.numel()),
        f"{prefix}_mean_delta_ce": float(delta.detach().mean().item()),
        f"{prefix}_positive_ratio": float((delta.detach() > 0.0).float().mean().item()),
        f"{prefix}_best_delta_ce": float(best_delta.detach().mean().item()),
        f"{prefix}_anti_harm_loss": float(anti_harm_loss.detach().item()),
    }
    for offset, name in enumerate(channel_names[1:]):
        stats[f"{prefix}_best_channel_ratio_{name}"] = float((best_offset == offset).float().mean().item())
        stats[f"{prefix}_{name}_mean_delta_ce"] = float(delta[:, offset].detach().mean().item())
    return loss, stats


def _forward_prompt_graph_with_message_scale(
    *,
    model: DualBranchFramework,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    h_pre: torch.Tensor,
    prompt_out: dict[str, Any],
    message_scale: float,
) -> dict[str, Any]:
    old_scale: float | None = None
    if isinstance(model, PromptAwareGP2F):
        old_scale = float(model.prompt_message_scale)
        model.prompt_message_scale = float(message_scale)
    try:
        forward_kwargs: dict[str, Any] = {
            "h_pre": h_pre,
            "adapted_x": prompt_out["adapted_x"],
            "adapted_edge_index": prompt_out["adapted_edge_index"],
            "adapted_edge_weight": prompt_out["adapted_edge_weight"],
            "return_aux": True,
        }
        if isinstance(model, PromptAwareGP2F):
            forward_kwargs["adapted_edge_type"] = prompt_out.get("adapted_edge_type")
            forward_kwargs["prompt_update_mask"] = prompt_out.get("pool_mask")
        return model.forward_with_h_pre(z, edge_index, **forward_kwargs)
    finally:
        if old_scale is not None and isinstance(model, PromptAwareGP2F):
            model.prompt_message_scale = old_scale


def _acceptance_supervision_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    positive_margin: float = 0.0,
    negative_margin: float = 0.0,
    balance_targets: bool = True,
    signal: str = "ce_delta",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train-only supervision for whether a pool node should receive prompt.

    Targets are built only from train-pool nodes by comparing prompt-on and
    prompt-off predictions. Validation/test labels are never inspected here.
    """

    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    gate_logits = aux.get("pool_acceptance_logit")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else torch.tensor(0.0)
    empty_stats = {
        "prompt_acceptance_supervision": 0.0,
        "acceptance_supervised_count": 0.0,
        "acceptance_positive_count": 0.0,
        "acceptance_negative_count": 0.0,
        "acceptance_ignored_count": 0.0,
        "acceptance_target_mean": 0.0,
        "acceptance_score_delta_mean": 0.0,
        "ce_delta_positive_ratio_train": 0.0,
        "ce_delta_negative_ratio_train": 0.0,
        "prompt_helpful_acceptance_precision_train": 0.0,
        "prompt_harmful_rejection_precision_train": 0.0,
    }
    if not (isinstance(pool_mask, torch.Tensor) and isinstance(gate_logits, torch.Tensor) and gate_logits.numel() > 0):
        return fallback, empty_stats
    pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != gate_logits.numel():
        raise ValueError("pool_acceptance_logit rows must match pool_mask true count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, empty_stats

    pool_idx_train = pool_idx[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx_train]
    on = logits_on.detach()[pool_idx_train]
    off = logits_off.detach()[pool_idx_train]
    on_pred = on.argmax(dim=-1)
    off_pred = off.argmax(dim=-1)
    on_correct = on_pred == y
    off_correct = off_pred == y

    positive_margin = max(float(positive_margin), 0.0)
    negative_margin = max(float(negative_margin), 0.0)
    if signal == "true_logit_delta":
        score_delta = on.gather(1, y.view(-1, 1)).squeeze(1) - off.gather(1, y.view(-1, 1)).squeeze(1)
    elif signal == "ce_delta":
        off_ce = F.cross_entropy(off, y, reduction="none")
        on_ce = F.cross_entropy(on, y, reduction="none")
        score_delta = off_ce - on_ce
    else:
        raise ValueError(
            f"Unsupported acceptance supervision signal={signal!r}; expected 'ce_delta' or 'true_logit_delta'"
        )
    positive = (on_correct & ~off_correct) | (score_delta > positive_margin)
    negative = (off_correct & ~on_correct) | (score_delta < -negative_margin)
    valid = positive ^ negative
    ignored = ~valid
    train_pool_count = max(1, int(train_pool.sum().item()))
    if int(valid.sum().item()) == 0:
        stats = dict(empty_stats)
        stats["acceptance_ignored_count"] = float(ignored.sum().item())
        stats["ce_delta_positive_ratio_train"] = float(positive.sum().item() / train_pool_count)
        stats["ce_delta_negative_ratio_train"] = float(negative.sum().item() / train_pool_count)
        return fallback, stats

    targets = positive[valid].to(dtype=gate_logits.dtype)
    selected_logits = gate_logits[train_pool][valid]
    if balance_targets and bool((targets > 0.5).any()) and bool((targets < 0.5).any()):
        pos_count = targets.sum().clamp_min(1.0)
        neg_count = (1.0 - targets).sum().clamp_min(1.0)
        pos_weight = neg_count / pos_count
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets, pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets)
    selected_gate = torch.sigmoid(selected_logits)
    accepted = selected_gate >= 0.5
    rejected = ~accepted
    helpful = targets > 0.5
    harmful = targets < 0.5
    helpful_acceptance_precision = (
        float((helpful & accepted).float().sum().item() / max(1, int(accepted.sum().item())))
        if int(accepted.sum().item()) > 0
        else 0.0
    )
    harmful_rejection_precision = (
        float((harmful & rejected).float().sum().item() / max(1, int(rejected.sum().item())))
        if int(rejected.sum().item()) > 0
        else 0.0
    )
    stats = {
        "prompt_acceptance_supervision": float(loss.detach().item()),
        "acceptance_supervised_count": float(valid.sum().item()),
        "acceptance_positive_count": float((positive & valid).sum().item()),
        "acceptance_negative_count": float((negative & valid).sum().item()),
        "acceptance_ignored_count": float(ignored.sum().item()),
        "acceptance_target_mean": float(targets.mean().detach().item()),
        "acceptance_score_delta_mean": float(score_delta[valid].mean().detach().item()),
        "ce_delta_positive_ratio_train": float(positive.sum().item() / train_pool_count),
        "ce_delta_negative_ratio_train": float(negative.sum().item() / train_pool_count),
        "prompt_helpful_acceptance_precision_train": helpful_acceptance_precision,
        "prompt_harmful_rejection_precision_train": harmful_rejection_precision,
    }
    return loss, stats


def _benefit_supervision_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float,
    balance_targets: bool,
    label_strategy: str = "margin",
    quantile: float = 0.20,
    eps: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    benefit_logits = aux.get("benefit_gate_logit")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else torch.tensor(0.0)
    empty_stats = {
        "prompt_benefit_supervision": 0.0,
        "benefit_supervised_count": 0.0,
        "benefit_positive_count": 0.0,
        "benefit_negative_count": 0.0,
        "benefit_ignored_count": 0.0,
        "mean_delta_ce_train_pool": 0.0,
        "positive_delta_ratio_train_pool": 0.0,
        "benefit_weight_delta_corr_train": 0.0,
        "benefit_label_strategy_quantile": 0.0,
        "benefit_quantile_fraction": 0.0,
        "benefit_positive_threshold": 0.0,
        "benefit_negative_threshold": 0.0,
        "benefit_delta_eps": 0.0,
    }
    if not (
        isinstance(pool_mask, torch.Tensor)
        and isinstance(benefit_logits, torch.Tensor)
        and benefit_logits.numel() > 0
    ):
        return fallback, empty_stats
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != benefit_logits.size(0):
        raise ValueError("benefit_gate_logit rows must match pool index count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, empty_stats

    pool_idx_train = pool_idx[train_pool]
    y = labels.to(device=pool_mask.device)[pool_idx_train]
    on = logits_on.detach()[pool_idx_train]
    off = logits_off.detach()[pool_idx_train]
    off_ce = F.cross_entropy(off, y, reduction="none")
    on_ce = F.cross_entropy(on, y, reduction="none")
    delta_ce = off_ce - on_ce
    margin = max(float(margin), 0.0)
    eps = max(float(eps), 0.0)
    strategy = str(label_strategy).lower()
    stats = dict(empty_stats)
    stats["benefit_label_strategy_quantile"] = float(strategy in {"quantile", "hybrid_quantile_margin"})
    stats["benefit_quantile_fraction"] = float(quantile)
    stats["benefit_delta_eps"] = float(eps)
    if strategy == "margin":
        positive = delta_ce > margin
        negative = delta_ce < -margin
        positive_threshold = delta_ce.new_tensor(margin)
        negative_threshold = delta_ce.new_tensor(-margin)
    elif strategy in {"quantile", "hybrid_quantile_margin"}:
        fraction = min(max(float(quantile), 0.0), 0.5)
        if delta_ce.numel() < 2 or fraction <= 0.0:
            positive = delta_ce > margin
            negative = delta_ce < -margin
            positive_threshold = delta_ce.new_tensor(margin)
            negative_threshold = delta_ce.new_tensor(-margin)
        else:
            k = min(max(1, int(math.ceil(float(delta_ce.numel()) * fraction))), delta_ce.numel() // 2)
            sorted_delta, sorted_idx = torch.sort(delta_ce)
            negative_threshold = sorted_delta[k - 1]
            positive_threshold = sorted_delta[-k]
            positive = torch.zeros_like(delta_ce, dtype=torch.bool)
            negative = torch.zeros_like(delta_ce, dtype=torch.bool)
            negative[sorted_idx[:k]] = True
            positive[sorted_idx[-k:]] = True
        if strategy == "hybrid_quantile_margin":
            positive = positive & (delta_ce > eps)
            negative = negative & (delta_ce < -eps)
    else:
        raise ValueError(
            f"Unsupported benefit supervision label_strategy={label_strategy!r}; "
            "expected 'margin', 'quantile', or 'hybrid_quantile_margin'"
        )
    valid = positive ^ negative
    ignored = ~valid
    train_count = max(1, int(train_pool.sum().item()))
    stats["mean_delta_ce_train_pool"] = float(delta_ce.mean().detach().item())
    stats["positive_delta_ratio_train_pool"] = float(positive.float().mean().detach().item())
    stats["benefit_ignored_count"] = float(ignored.sum().item())
    stats["benefit_positive_threshold"] = float(positive_threshold.detach().item())
    stats["benefit_negative_threshold"] = float(negative_threshold.detach().item())
    if int(valid.sum().item()) == 0:
        return fallback, stats

    selected_logits = benefit_logits[train_pool][valid].reshape(-1)
    targets = positive[valid].to(dtype=selected_logits.dtype).unsqueeze(-1).expand(-1, benefit_logits.size(1)).reshape(-1)
    if balance_targets and bool((targets > 0.5).any()) and bool((targets < 0.5).any()):
        pos_count = targets.sum().clamp_min(1.0)
        neg_count = (1.0 - targets).sum().clamp_min(1.0)
        pos_weight = neg_count / pos_count
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets, pos_weight=pos_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets)

    benefit_weight = torch.sigmoid(benefit_logits[train_pool]).mean(dim=-1)
    valid_weight = benefit_weight[valid]
    valid_delta = delta_ce[valid]
    if valid_weight.numel() > 1 and float(valid_weight.std(unbiased=False).item()) > 1e-12:
        centered_weight = valid_weight - valid_weight.mean()
        centered_delta = valid_delta - valid_delta.mean()
        corr = (centered_weight * centered_delta).mean() / (
            centered_weight.pow(2).mean().sqrt() * centered_delta.pow(2).mean().sqrt()
        ).clamp_min(1e-12)
        stats["benefit_weight_delta_corr_train"] = float(corr.detach().item())
    stats.update(
        {
            "prompt_benefit_supervision": float(loss.detach().item()),
            "benefit_supervised_count": float(valid.sum().item()),
            "benefit_positive_count": float(positive[valid].sum().item()),
            "benefit_negative_count": float(negative[valid].sum().item()),
            "benefit_ignored_count": float(ignored.sum().item()),
            "positive_delta_ratio_train_pool": float(positive.sum().item() / train_count),
        }
    )
    return loss, stats


def _prompt_correction_losses(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    eps: float,
    target: str = "delta_prob_to_label",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty_stats = {
        "prompt_correction_loss": 0.0,
        "prompt_anti_harm_loss": 0.0,
        "prompt_correction_supervised_count": 0.0,
        "prompt_correction_harmful_count": 0.0,
        "prompt_correction_target_norm": 0.0,
        "prompt_delta_logit_norm": 0.0,
        "prompt_delta_logit_label_alignment": 0.0,
        "harmful_prompt_delta_ratio": 0.0,
    }
    if not isinstance(pool_mask, torch.Tensor):
        return fallback, fallback, empty_stats
    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() == 0:
        return fallback, fallback, empty_stats
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, fallback, empty_stats

    pool_idx_train = pool_idx[train_pool]
    y = labels.to(device=logits_on.device)[pool_idx_train]
    on = logits_on[pool_idx_train]
    off = logits_off.detach()[pool_idx_train]
    off_ce = F.cross_entropy(off, y, reduction="none")
    on_ce = F.cross_entropy(on.detach(), y, reduction="none")
    delta_ce = off_ce - on_ce
    eps = max(float(eps), 0.0)
    positive = delta_ce > eps
    harmful = delta_ce < -eps

    delta_logits = on - off
    off_prob = F.softmax(off, dim=-1)
    if target == "delta_prob_to_label":
        target_delta = F.one_hot(y, num_classes=logits_on.size(-1)).to(dtype=logits_on.dtype) - off_prob
    else:
        raise ValueError(
            f"Unsupported prompt_correction_target={target!r}; expected 'delta_prob_to_label'"
        )

    weight = torch.ones_like(delta_ce, dtype=logits_on.dtype)
    benefit_gate = aux.get("benefit_gate")
    if isinstance(benefit_gate, torch.Tensor) and benefit_gate.numel() > 0:
        benefit_gate = benefit_gate.to(device=logits_on.device, dtype=logits_on.dtype)
        if benefit_gate.size(0) == pool_idx.numel():
            weight = weight * benefit_gate[train_pool].mean(dim=-1).detach()
    receive_gate = aux.get("effective_receive_gate")
    if isinstance(receive_gate, torch.Tensor) and receive_gate.numel() > 0:
        receive_gate = receive_gate.to(device=logits_on.device, dtype=logits_on.dtype)
        if receive_gate.size(0) == pool_idx.numel():
            weight = weight * receive_gate[train_pool].detach()

    if int(positive.sum().item()) > 0:
        pos_weight = weight[positive].clamp_min(1e-6)
        pos_loss = (delta_logits[positive] - target_delta[positive]).pow(2).mean(dim=-1)
        correction_loss = (pos_loss * pos_weight).sum() / pos_weight.sum().clamp_min(1e-6)
        target_norm = target_delta[positive].norm(dim=-1).mean()
        delta_norm = delta_logits[positive].norm(dim=-1).mean()
        label_alignment = delta_logits[positive, y[positive]].mean()
    else:
        correction_loss = fallback
        target_norm = fallback
        delta_norm = fallback
        label_alignment = fallback

    if int(harmful.sum().item()) > 0:
        harmful_delta = delta_logits[harmful]
        harmful_y = y[harmful]
        true_class_increase = harmful_delta.gather(1, harmful_y.view(-1, 1)).squeeze(-1)
        wrong_mask = torch.ones_like(harmful_delta, dtype=torch.bool)
        wrong_mask.scatter_(1, harmful_y.view(-1, 1), False)
        wrong_class_increase = harmful_delta.masked_fill(~wrong_mask, float("-inf")).max(dim=-1).values
        harmful_score = F.relu(wrong_class_increase - true_class_increase)
        harm_weight = weight[harmful].clamp_min(1e-6)
        anti_harm_loss = (harmful_score * harm_weight).sum() / harm_weight.sum().clamp_min(1e-6)
        harmful_prompt_delta_ratio = float((harmful_score > 0.0).float().mean().detach().item())
    else:
        anti_harm_loss = fallback
        harmful_prompt_delta_ratio = 0.0

    stats = {
        "prompt_correction_loss": float(correction_loss.detach().item()),
        "prompt_anti_harm_loss": float(anti_harm_loss.detach().item()),
        "prompt_correction_supervised_count": float(positive.sum().item()),
        "prompt_correction_harmful_count": float(harmful.sum().item()),
        "prompt_correction_target_norm": float(target_norm.detach().item()),
        "prompt_delta_logit_norm": float(delta_norm.detach().item()),
        "prompt_delta_logit_label_alignment": float(label_alignment.detach().item()),
        "harmful_prompt_delta_ratio": harmful_prompt_delta_ratio,
    }
    return correction_loss, anti_harm_loss, stats


def _prompt_message_help_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float,
    class_balanced: bool = True,
    anti_harm_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty_stats: dict[str, Any] = {
        "prompt_message_help_loss": 0.0,
        "class_balanced_mean_delta_ce": 0.0,
        "class_balanced_positive_delta_ratio": 0.0,
        "message_help_supervised_count": 0.0,
        "message_help_class_count": 0.0,
        "message_help_margin": float(margin),
        "message_help_class_balanced": float(class_balanced),
        "prompt_class_anti_harm_loss": 0.0,
        "prompt_class_anti_harm_floor": float(anti_harm_floor),
        "delta_ce_by_class_train_pool": {},
    }
    if not isinstance(pool_mask, torch.Tensor):
        return fallback, fallback, empty_stats

    train_pool_mask = train_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    train_pool_idx = torch.where(train_pool_mask)[0]
    if train_pool_idx.numel() == 0:
        return fallback, fallback, empty_stats

    y = labels.to(device=logits_on.device)[train_pool_idx]
    ce_off = F.cross_entropy(logits_off.detach()[train_pool_idx], y, reduction="none")
    ce_on = F.cross_entropy(logits_on[train_pool_idx], y, reduction="none")
    delta_ce = ce_off - ce_on
    per_node_loss = F.relu(float(margin) - delta_ce)
    positive = delta_ce > 0.0

    class_losses: list[torch.Tensor] = []
    class_mean_delta: list[torch.Tensor] = []
    class_positive_ratio: list[torch.Tensor] = []
    class_anti_harm_losses: list[torch.Tensor] = []
    by_class: dict[str, dict[str, float]] = {}
    for class_id in torch.unique(y.detach()).tolist():
        class_mask = y == int(class_id)
        if not bool(class_mask.any()):
            continue
        class_losses.append(per_node_loss[class_mask].mean())
        mean_delta = delta_ce[class_mask].mean()
        class_mean_delta.append(mean_delta.detach())
        class_anti_harm_losses.append(F.relu(float(anti_harm_floor) - mean_delta))
        class_positive_ratio.append(positive[class_mask].to(dtype=logits_on.dtype).detach().mean())
        by_class[str(int(class_id))] = {
            "count": float(class_mask.sum().item()),
            "mean_delta_ce": float(delta_ce[class_mask].detach().mean().item()),
            "positive_delta_ratio": float(positive[class_mask].float().detach().mean().item()),
            "mean_ce_no_prompt": float(ce_off[class_mask].detach().mean().item()),
            "mean_ce_prompt": float(ce_on[class_mask].detach().mean().item()),
        }

    if class_balanced and class_losses:
        loss = torch.stack(class_losses).mean()
        balanced_delta = torch.stack(class_mean_delta).mean()
        balanced_positive = torch.stack(class_positive_ratio).mean()
        class_count = float(len(class_losses))
        anti_harm_loss = torch.stack(class_anti_harm_losses).mean()
    else:
        loss = per_node_loss.mean()
        balanced_delta = delta_ce.detach().mean()
        balanced_positive = positive.to(dtype=logits_on.dtype).detach().mean()
        class_count = float(len(class_losses))
        anti_harm_loss = F.relu(float(anti_harm_floor) - delta_ce.mean())

    stats: dict[str, Any] = {
        "prompt_message_help_loss": float(loss.detach().item()),
        "class_balanced_mean_delta_ce": float(balanced_delta.item()),
        "class_balanced_positive_delta_ratio": float(balanced_positive.item()),
        "message_help_supervised_count": float(train_pool_idx.numel()),
        "message_help_class_count": class_count,
        "message_help_margin": float(margin),
        "message_help_class_balanced": float(class_balanced),
        "prompt_class_anti_harm_loss": float(anti_harm_loss.detach().item()),
        "prompt_class_anti_harm_floor": float(anti_harm_floor),
        "delta_ce_by_class_train_pool": by_class,
    }
    return loss, anti_harm_loss, stats


def _edge_utility_supervision_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    aux = prompt_out.get("aux", {})
    edge_logits = aux.get("edge_utility_logit")
    pool_idx = aux.get("pool_idx")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty = {
        "edge_utility_supervision_loss": 0.0,
        "edge_utility_supervised_count": 0.0,
        "edge_utility_positive_count": 0.0,
        "edge_utility_negative_count": 0.0,
        "edge_utility_ignored_count": 0.0,
        "edge_utility_target_mean": 0.0,
        "edge_utility_delta_ce_mean": 0.0,
        "edge_utility_delta_ce_positive_ratio": 0.0,
        "edge_utility_delta_corr_train": 0.0,
    }
    if not (isinstance(edge_logits, torch.Tensor) and isinstance(pool_idx, torch.Tensor)):
        return fallback, empty
    if edge_logits.numel() == 0 or pool_idx.numel() == 0:
        return fallback, empty

    train_pool = train_mask.to(device=pool_idx.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) == 0:
        return fallback, empty

    pool_train_idx = pool_idx[train_pool]
    y = labels.to(device=logits_on.device)[pool_train_idx]
    ce_off = F.cross_entropy(logits_off.detach()[pool_train_idx], y, reduction="none")
    ce_on = F.cross_entropy(logits_on.detach()[pool_train_idx], y, reduction="none")
    delta_ce = ce_off - ce_on
    positive = delta_ce > float(margin)
    negative = delta_ce < -float(margin)
    valid = positive | negative
    ignored = ~valid
    stats = {
        **empty,
        "edge_utility_ignored_count": float(ignored.sum().item()),
        "edge_utility_delta_ce_mean": float(delta_ce.detach().mean().item()),
        "edge_utility_delta_ce_positive_ratio": float((delta_ce > 0.0).float().mean().detach().item()),
    }
    if int(valid.sum().item()) == 0:
        return fallback, stats

    selected_logits = edge_logits[train_pool][valid].reshape(-1)
    targets = positive[valid].to(dtype=selected_logits.dtype).unsqueeze(-1).expand(-1, edge_logits.size(1)).reshape(-1)
    if bool((targets > 0.5).any()) and bool((targets < 0.5).any()):
        pos_count = targets.sum().clamp_min(1.0)
        neg_count = (1.0 - targets).sum().clamp_min(1.0)
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets, pos_weight=neg_count / pos_count)
    else:
        loss = F.binary_cross_entropy_with_logits(selected_logits, targets)

    edge_utility = torch.sigmoid(edge_logits[train_pool]).mean(dim=-1)
    valid_utility = edge_utility[valid]
    valid_delta = delta_ce[valid]
    corr = logits_on.new_tensor(0.0)
    if valid_utility.numel() > 1 and float(valid_utility.std(unbiased=False).detach().item()) > 1e-12:
        centered_u = valid_utility - valid_utility.mean()
        centered_delta = valid_delta - valid_delta.mean()
        corr = (centered_u * centered_delta).mean() / (
            centered_u.pow(2).mean().sqrt() * centered_delta.pow(2).mean().sqrt()
        ).clamp_min(1e-12)

    stats.update(
        {
            "edge_utility_supervision_loss": float(loss.detach().item()),
            "edge_utility_supervised_count": float(valid.sum().item()),
            "edge_utility_positive_count": float(positive[valid].sum().item()),
            "edge_utility_negative_count": float(negative[valid].sum().item()),
            "edge_utility_ignored_count": float(ignored.sum().item()),
            "edge_utility_target_mean": float(targets.detach().mean().item()),
            "edge_utility_delta_corr_train": float(corr.detach().item()),
        }
    )
    return loss, stats


def _correction_alignment_losses(
    *,
    model: DualBranchFramework,
    prompt_out: dict[str, Any],
    h_on: torch.Tensor,
    h_off: torch.Tensor,
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pool_mask = prompt_out.get("pool_mask")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else h_on.new_tensor(0.0)
    empty = {
        "correction_alignment_loss": 0.0,
        "correction_alignment_anti_harm_loss": 0.0,
        "correction_alignment_node_count": 0.0,
        "correction_alignment_harmful_count": 0.0,
        "correction_alignment_cosine_mean": 0.0,
        "correction_alignment_delta_h_norm": 0.0,
        "correction_alignment_delta_ce_mean": 0.0,
    }
    if not isinstance(pool_mask, torch.Tensor):
        return fallback, fallback, empty
    eligible = train_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    idx = torch.where(eligible)[0]
    if idx.numel() == 0:
        return fallback, fallback, empty
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        return fallback, fallback, empty

    y = labels.to(device=h_on.device)[idx]
    ce_off = F.cross_entropy(logits_off.detach()[idx], y, reduction="none")
    ce_on = F.cross_entropy(logits_on.detach()[idx], y, reduction="none")
    delta_ce = ce_off - ce_on
    helpful = delta_ce > float(margin)
    harmful = delta_ce < -float(margin)
    delta_h = h_on[idx] - h_off.detach()[idx]
    delta_norm = delta_h.norm(dim=-1)
    class_direction = classifier.weight.detach().to(device=h_on.device, dtype=h_on.dtype)[y]
    cosine = F.cosine_similarity(delta_h, class_direction, dim=-1, eps=1e-12)

    if int(helpful.sum().item()) > 0:
        alignment_loss = (1.0 - cosine[helpful]).mean()
        cosine_mean = cosine[helpful].detach().mean()
    else:
        alignment_loss = fallback
        cosine_mean = fallback
    if int(harmful.sum().item()) > 0:
        anti_harm_loss = delta_norm[harmful].mean()
    else:
        anti_harm_loss = fallback

    stats = {
        "correction_alignment_loss": float(alignment_loss.detach().item()),
        "correction_alignment_anti_harm_loss": float(anti_harm_loss.detach().item()),
        "correction_alignment_node_count": float(helpful.sum().item()),
        "correction_alignment_harmful_count": float(harmful.sum().item()),
        "correction_alignment_cosine_mean": float(cosine_mean.detach().item()),
        "correction_alignment_delta_h_norm": float(delta_norm.detach().mean().item()),
        "correction_alignment_delta_ce_mean": float(delta_ce.detach().mean().item()),
    }
    return alignment_loss, anti_harm_loss, stats


def _query_proto_alignment_loss(
    *,
    h_on: torch.Tensor,
    h_off: torch.Tensor,
    labels: torch.Tensor,
    support_mask: torch.Tensor,
    query_mask: torch.Tensor,
    pool_mask: torch.Tensor,
    margin: float = 0.005,
    normalize: bool = True,
    class_balanced: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    fallback = h_on.new_tensor(0.0)
    support_mask = support_mask.to(device=h_on.device, dtype=torch.bool)
    query_mask = query_mask.to(device=h_on.device, dtype=torch.bool)
    pool_mask = pool_mask.to(device=h_on.device, dtype=torch.bool)
    query_pool = query_mask & pool_mask
    empty_stats: dict[str, Any] = {
        "query_proto_alignment_loss": 0.0,
        "query_proto_supervised_count": float(query_pool.sum().item()),
        "query_proto_class_count": 0.0,
        "query_proto_mean_delta_dist": 0.0,
        "query_proto_positive_ratio": 0.0,
        "query_proto_margin": float(margin),
        "query_proto_class_balanced": float(class_balanced),
        "query_proto_by_class": {},
    }
    if int(query_pool.sum().item()) == 0 or int(support_mask.sum().item()) == 0:
        return fallback, empty_stats

    labels = labels.to(device=h_on.device)
    feat_on = F.normalize(h_on, dim=-1, eps=1e-12) if normalize else h_on
    feat_off = F.normalize(h_off.detach(), dim=-1, eps=1e-12) if normalize else h_off.detach()
    support_feat = F.normalize(h_off.detach(), dim=-1, eps=1e-12) if normalize else h_off.detach()

    class_losses: list[torch.Tensor] = []
    class_delta: list[torch.Tensor] = []
    class_positive: list[torch.Tensor] = []
    by_class: dict[str, dict[str, float]] = {}
    for class_id in torch.unique(labels[support_mask | query_pool].detach()).tolist():
        class_id = int(class_id)
        support_class = support_mask & (labels == class_id)
        query_class = query_pool & (labels == class_id)
        if not bool(support_class.any()) or not bool(query_class.any()):
            continue
        proto = support_feat[support_class].mean(dim=0)
        proto = F.normalize(proto, dim=0, eps=1e-12) if normalize else proto
        dist_off = 1.0 - F.cosine_similarity(feat_off[query_class], proto.unsqueeze(0), dim=-1, eps=1e-12)
        dist_on = 1.0 - F.cosine_similarity(feat_on[query_class], proto.unsqueeze(0), dim=-1, eps=1e-12)
        delta_dist = dist_off - dist_on
        loss = F.relu(float(margin) - delta_dist)
        class_losses.append(loss.mean())
        class_delta.append(delta_dist.detach().mean())
        class_positive.append((delta_dist > 0.0).to(dtype=h_on.dtype).detach().mean())
        by_class[str(class_id)] = {
            "count": float(query_class.sum().item()),
            "mean_delta_dist": float(delta_dist.detach().mean().item()),
            "positive_delta_ratio": float((delta_dist > 0.0).float().detach().mean().item()),
        }

    if not class_losses:
        return fallback, empty_stats
    if class_balanced:
        loss = torch.stack(class_losses).mean()
        mean_delta = torch.stack(class_delta).mean()
        positive_ratio = torch.stack(class_positive).mean()
    else:
        loss = torch.stack(class_losses).mean()
        mean_delta = torch.stack(class_delta).mean()
        positive_ratio = torch.stack(class_positive).mean()
    stats = {
        "query_proto_alignment_loss": float(loss.detach().item()),
        "query_proto_supervised_count": float(query_pool.sum().item()),
        "query_proto_class_count": float(len(class_losses)),
        "query_proto_mean_delta_dist": float(mean_delta.detach().item()),
        "query_proto_positive_ratio": float(positive_ratio.detach().item()),
        "query_proto_margin": float(margin),
        "query_proto_class_balanced": float(class_balanced),
        "query_proto_by_class": by_class,
    }
    return loss, stats


def _utility_receive_gate_loss(
    *,
    prompt_out: dict[str, Any],
    logits_on: torch.Tensor,
    logits_off: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    quantile: float = 0.20,
    eps: float = 1e-4,
    class_balanced: bool = True,
    label_strategy: str = "quantile",
) -> tuple[torch.Tensor, dict[str, Any]]:
    aux = prompt_out.get("aux", {})
    pool_mask = prompt_out.get("pool_mask")
    gate = aux.get("utility_receive_gate")
    edge_scale = prompt_out.get("edge_scale")
    fallback = edge_scale.new_tensor(0.0) if isinstance(edge_scale, torch.Tensor) else logits_on.new_tensor(0.0)
    empty_stats: dict[str, Any] = {
        "utility_receive_gate_loss": 0.0,
        "utility_gate_supervised_count": 0.0,
        "utility_gate_positive_count": 0.0,
        "utility_gate_negative_count": 0.0,
        "utility_gate_ignored_count": 0.0,
        "utility_gate_target_mean": 0.0,
        "utility_gate_delta_corr_train": 0.0,
        "utility_gate_positive_threshold_mean": 0.0,
        "utility_gate_negative_threshold_mean": 0.0,
        "utility_gate_quantile": float(quantile),
        "utility_gate_eps": float(eps),
        "utility_gate_class_balanced": float(class_balanced),
        "utility_gate_label_strategy": label_strategy,
    }
    if not (
        isinstance(pool_mask, torch.Tensor)
        and isinstance(gate, torch.Tensor)
        and gate.numel() > 0
    ):
        return fallback, empty_stats

    pool_idx = aux.get("pool_idx")
    if not isinstance(pool_idx, torch.Tensor):
        pool_idx = torch.where(pool_mask.bool())[0]
    if pool_idx.numel() != gate.numel():
        raise ValueError("utility_receive_gate rows must match pool index count")
    train_pool = train_mask.to(device=pool_mask.device, dtype=torch.bool)[pool_idx]
    if int(train_pool.sum().item()) < 2:
        return fallback, empty_stats

    y_all = labels.to(device=logits_on.device)[pool_idx]
    train_pool_idx = pool_idx[train_pool]
    y = y_all[train_pool]
    ce_off = F.cross_entropy(logits_off.detach()[train_pool_idx], y, reduction="none")
    ce_on = F.cross_entropy(logits_on.detach()[train_pool_idx], y, reduction="none")
    delta_ce = ce_off - ce_on
    gate_train = gate[train_pool].clamp(min=1e-6, max=1.0 - 1e-6)
    fraction = min(max(float(quantile), 0.0), 0.5)
    eps = max(float(eps), 0.0)

    selected_masks: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    positive_thresholds: list[torch.Tensor] = []
    negative_thresholds: list[torch.Tensor] = []
    classes = torch.unique(y.detach()).tolist() if class_balanced else [-1]
    for class_id in classes:
        class_mask = torch.ones_like(y, dtype=torch.bool) if int(class_id) == -1 else y == int(class_id)
        rows = torch.where(class_mask)[0]
        if rows.numel() < 1:
            continue
        selected = torch.zeros_like(y, dtype=torch.bool)
        target = torch.zeros_like(delta_ce)
        class_delta = delta_ce[rows]
        if label_strategy == "margin":
            pos_rows = rows[class_delta > eps]
            neg_rows = rows[class_delta < -eps]
            if bool(pos_rows.numel() > 0):
                selected[pos_rows] = True
                target[pos_rows] = 1.0
            if bool(neg_rows.numel() > 0):
                selected[neg_rows] = True
                target[neg_rows] = 0.0
            if bool((class_delta > eps).any()):
                positive_thresholds.append(class_delta[class_delta > eps].min().detach())
            if bool((class_delta < -eps).any()):
                negative_thresholds.append(class_delta[class_delta < -eps].max().detach())
        elif label_strategy == "quantile":
            if rows.numel() < 2 or fraction <= 0.0:
                continue
            k = min(max(1, int(math.ceil(float(rows.numel()) * fraction))), int(rows.numel()) // 2)
            if k <= 0:
                continue
            sorted_delta, order = torch.sort(class_delta)
            neg_rows = rows[order[:k]]
            pos_rows = rows[order[-k:]]
            pos_keep = delta_ce[pos_rows] > eps
            neg_keep = delta_ce[neg_rows] < -eps
            if bool(pos_keep.any()):
                selected[pos_rows[pos_keep]] = True
                target[pos_rows[pos_keep]] = 1.0
            if bool(neg_keep.any()):
                selected[neg_rows[neg_keep]] = True
                target[neg_rows[neg_keep]] = 0.0
            negative_thresholds.append(sorted_delta[k - 1].detach())
            positive_thresholds.append(sorted_delta[-k].detach())
        else:
            raise ValueError("utility receive gate label_strategy must be 'quantile' or 'margin'")
        if bool(selected.any()):
            selected_masks.append(selected)
            target_values.append(target)

    if not selected_masks:
        return fallback, empty_stats
    supervised = torch.stack(selected_masks, dim=0).any(dim=0)
    target = torch.stack(target_values, dim=0).sum(dim=0).clamp(max=1.0)
    if int(supervised.sum().item()) == 0:
        return fallback, empty_stats

    loss = F.binary_cross_entropy(gate_train[supervised], target[supervised])
    positive = (target[supervised] > 0.5)
    negative = ~positive
    stats = {
        "utility_receive_gate_loss": float(loss.detach().item()),
        "utility_gate_supervised_count": float(supervised.sum().item()),
        "utility_gate_positive_count": float(positive.sum().item()),
        "utility_gate_negative_count": float(negative.sum().item()),
        "utility_gate_ignored_count": float((~supervised).sum().item()),
        "utility_gate_target_mean": float(target[supervised].detach().mean().item()),
        "utility_gate_delta_corr_train": _pearson_corr(gate_train.detach(), delta_ce.detach()),
        "utility_gate_positive_threshold_mean": (
            float(torch.stack(positive_thresholds).mean().item()) if positive_thresholds else 0.0
        ),
        "utility_gate_negative_threshold_mean": (
            float(torch.stack(negative_thresholds).mean().item()) if negative_thresholds else 0.0
        ),
        "utility_gate_quantile": float(quantile),
        "utility_gate_eps": float(eps),
        "utility_gate_class_balanced": float(class_balanced),
        "utility_gate_label_strategy": label_strategy,
    }
    return loss, stats


def _edge_scale_multiplier(epoch: int, prompt_graph_cfg: dict[str, Any]) -> float:
    warmup_epochs = int(prompt_graph_cfg.get("edge_scale_warmup_epochs", 0))
    start = float(prompt_graph_cfg.get("edge_scale_warmup_start", 1.0 if warmup_epochs <= 0 else 0.0))
    start = min(max(start, 0.0), 1.0)
    if warmup_epochs <= 0:
        return 1.0
    progress = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    return start + (1.0 - start) * progress


def _message_scale_grid(config: dict[str, Any]) -> list[float]:
    raw = config.get("prompt_aware", {}).get("message_scale_grid")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    else:
        values = [float(value) for value in raw]
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _scale_label(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _float_grid(raw: Any, *, default: list[float] | None = None) -> list[float]:
    if raw is None:
        values = list(default or [])
    elif isinstance(raw, str):
        values = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]
    else:
        values = [float(raw)]
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _candidate_monitor_value(result: dict[str, Any], metric: str) -> float:
    if metric in result.get("best", {}):
        return float(result["best"].get(metric, 0.0))
    if metric in result.get("final", {}):
        return float(result["final"].get(metric, 0.0))
    return float(result["best"].get("val_acc", 0.0))


def _select_message_scale_candidate(
    candidates: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("message_scale_grid selection requires at least one candidate")
    minimize = metric in {"train_loss", "total", "cls", "edge_l1", "prompt_balance"}
    ordered = sorted(
        candidates,
        key=lambda item: (
            _candidate_monitor_value(item, metric),
            -float(item.get("candidate_message_scale", 0.0)),
        ),
        reverse=not minimize,
    )
    return ordered[0]


def _compact_scale_candidate(result: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "message_scale": float(result.get("candidate_message_scale", result["best"].get("prompt_message_scale", 0.0))),
        "selection_metric": metric,
        "selection_value": _candidate_monitor_value(result, metric),
        "best_epoch": result["best"].get("best_epoch", 0.0),
        "best_val_acc": result["best"].get("val_acc", 0.0),
        "best_val_macro_f1": result["best"].get("val_macro_f1", 0.0),
        "best_test_acc": result["best"].get("test_acc", 0.0),
        "best_test_macro_f1": result["best"].get("test_macro_f1", 0.0),
        "final_test_acc": result["final"].get("test_acc", 0.0),
        "final_test_macro_f1": result["final"].get("test_macro_f1", 0.0),
        "run_dir": result.get("run_dir", ""),
    }


@torch.no_grad()
def _init_equivalence(
    *,
    model: DualBranchFramework,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
    prompt_adapter_module: HeterophilyAwarePromptAdapter | P22ClassPatternEnrichmentBank | None = None,
) -> dict[str, float]:
    model.eval()
    input_aligner.eval()
    if prompt_graph_module is not None:
        prompt_graph_module.eval()
    if prompt_adapter_module is not None:
        prompt_adapter_module.eval()
    z = input_aligner(x)
    if prompt_adapter_module is not None:
        h_pre = model.encode_frozen(z, edge_index)
        baseline = _forward_no_prompt_with_h_pre(
            model=model,
            z=z,
            edge_index=edge_index,
            h_pre=h_pre,
        )
        prompted, adapter_out, _ = _forward_prompt_adapter(
            model=model,
            prompt_adapter_module=prompt_adapter_module,
            z=z,
            edge_index=edge_index,
            update_mask=torch.ones_like(train_mask, dtype=torch.bool),
            support_mask=train_mask,
            compat_support_mask=train_mask,
            labels=labels,
        )
        return {
            "init_original_x_delta": 0.0,
            "init_logit_delta": float((prompted["logits"] - baseline["logits"]).abs().max().item()),
            "init_logit_delta_full_edge_scale": float((prompted["logits"] - baseline["logits"]).abs().max().item()),
            "init_prompt_adapter_update_norm": float(adapter_out["prompt_update_norm"].detach().item()),
        }
    baseline = model(z, edge_index, return_aux=True)
    h_pre = None
    no_prompt_out = None
    if _needs_no_prompt_pool_evidence(prompt_graph_module):
        h_pre = model.encode_frozen(z, edge_index)
        no_prompt_out = _forward_no_prompt_with_h_pre(
            model=model,
            z=z,
            edge_index=edge_index,
            h_pre=h_pre,
        )
    prompted_zero, prompt_out_zero = _forward_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        z=z,
        x_raw=x,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        edge_scale_multiplier=0.0,
        h_pre=h_pre,
        no_prompt_logits=None if no_prompt_out is None else no_prompt_out["logits"],
        h_adp_no_prompt=None if no_prompt_out is None else no_prompt_out["h_adp"],
    )
    prompted_full, _ = _forward_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        z=z,
        x_raw=x,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        edge_scale_multiplier=1.0,
        h_pre=h_pre,
        no_prompt_logits=None if no_prompt_out is None else no_prompt_out["logits"],
        h_adp_no_prompt=None if no_prompt_out is None else no_prompt_out["h_adp"],
    )
    return {
        "init_original_x_delta": float((prompt_out_zero["adapted_x"][: z.size(0)] - z).abs().max().item()),
        "init_logit_delta": float((prompted_zero["logits"] - baseline["logits"]).abs().max().item()),
        "init_logit_delta_full_edge_scale": float((prompted_full["logits"] - baseline["logits"]).abs().max().item()),
    }


@torch.no_grad()
def evaluate_prompt_graph(
    *,
    model: DualBranchFramework,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    edge_scale_multiplier: float = 1.0,
    epoch: int | None = None,
    prompt_adapter_module: HeterophilyAwarePromptAdapter | ClassConditionedPatternPromptRouter | P21LiteAdaptiveFilter | P21V2HeteroFilter | P22ClassPatternEnrichmentBank | None = None,
) -> dict[str, Any]:
    model.eval()
    input_aligner.eval()
    if prompt_graph_module is not None:
        prompt_graph_module.eval()
    z = input_aligner(x)
    if prompt_adapter_module is not None:
        prompt_adapter_module.eval()
        prompt_adapter_cfg = dict(getattr(prompt_adapter_module, "config", {}) or {})
        p21_eval_cfg = _resolve_p21_utility_cfg(prompt_adapter_cfg)
        update_mask = torch.ones_like(train_mask, dtype=torch.bool)
        pool_stats: dict[str, Any] = {}
        if bool(prompt_adapter_cfg.get("use_candidate_pool", False)):
            h_pre_for_pool = model.encode_frozen(z, edge_index)
            no_prompt_for_pool = _forward_no_prompt_with_h_pre(
                model=model,
                z=z,
                edge_index=edge_index,
                h_pre=h_pre_for_pool,
            )
            update_mask, pool_stats = _adapter_candidate_pool(
                z=z,
                edge_index=edge_index,
                train_mask=train_mask,
                prompt_adapter_cfg=prompt_adapter_cfg,
                no_prompt_logits=no_prompt_for_pool["logits"],
                h_pre=h_pre_for_pool,
                h_adp_no_prompt=no_prompt_for_pool["h_adp"],
            )
        model_out, adapter_out, no_prompt_out = _forward_prompt_adapter(
            model=model,
            prompt_adapter_module=prompt_adapter_module,
            z=z,
            edge_index=edge_index,
            update_mask=update_mask,
            support_mask=train_mask,
            compat_support_mask=train_mask,
            labels=labels,
        )
        logits = model_out["logits"]
        branch_cosine = torch.nn.functional.cosine_similarity(model_out["h_pre"], model_out["h_adp"], dim=-1).mean()
        train = split_metrics(logits, labels, train_mask, num_classes=num_classes)
        val = split_metrics(logits, labels, val_mask, num_classes=num_classes)
        test = split_metrics(logits, labels, test_mask, num_classes=num_classes)
        adapter_diag = _prompt_adapter_diagnostics(adapter_out)
        adapter_diag.update(pool_stats)
        train_delta_stats = _prompt_adapter_delta_stats(
            logits_prompt=logits,
            logits_no_prompt=no_prompt_out["logits"],
            labels=labels,
            mask=train_mask,
            prefix="adapter_train",
        )
        val_delta_stats = _prompt_adapter_delta_stats(
            logits_prompt=logits,
            logits_no_prompt=no_prompt_out["logits"],
            labels=labels,
            mask=val_mask,
            prefix="adapter_val",
        )
        test_delta_stats = _prompt_adapter_delta_stats(
            logits_prompt=logits,
            logits_no_prompt=no_prompt_out["logits"],
            labels=labels,
            mask=test_mask,
            prefix="adapter_test",
        )
        adapter_diag.update(train_delta_stats)
        adapter_diag.update(val_delta_stats)
        adapter_diag.update(test_delta_stats)
        adapter_diag.update(
            {
                "p22_train_delta_ce": train_delta_stats.get("adapter_train_mean_delta_ce", 0.0),
                "p22_train_positive_delta_ratio": train_delta_stats.get("adapter_train_positive_delta_ratio", 0.0),
                "p22_val_delta_ce": val_delta_stats.get("adapter_val_mean_delta_ce", 0.0),
                "p22_val_positive_delta_ratio": val_delta_stats.get("adapter_val_positive_delta_ratio", 0.0),
                "p22_test_delta_ce": test_delta_stats.get("adapter_test_mean_delta_ce", 0.0),
                "p22_test_positive_delta_ratio": test_delta_stats.get("adapter_test_positive_delta_ratio", 0.0),
            }
        )
        if bool(prompt_adapter_cfg.get("log_single_basis_delta_ce", False)):
            raw_scale_grid = prompt_adapter_cfg.get("basis_delta_scale_grid", [0.01, 0.03, 0.05, 0.10, 0.20])
            if isinstance(raw_scale_grid, str):
                scale_grid = [float(item.strip()) for item in raw_scale_grid.split(",") if item.strip()]
            elif isinstance(raw_scale_grid, list):
                scale_grid = [float(item) for item in raw_scale_grid]
            else:
                scale_grid = [float(raw_scale_grid)]
            adapter_diag.update(
                _p22_single_basis_delta_stats(
                    adapter_out=adapter_out,
                    base_logits=no_prompt_out["logits"],
                    labels=labels,
                    masks={"train": train_mask, "val": val_mask, "test": test_mask},
                    scale_grid=scale_grid,
                )
            )
        adapter_diag.update(
            _p22_pattern_only_metrics(
                adapter_out=adapter_out,
                labels=labels,
                mask=train_mask,
                num_classes=num_classes,
                prefix="p22_pattern_only_train",
            )
        )
        adapter_diag.update(
            _p22_pattern_only_metrics(
                adapter_out=adapter_out,
                labels=labels,
                mask=val_mask,
                num_classes=num_classes,
                prefix="p22_pattern_only_val",
            )
        )
        adapter_diag.update(
            _p22_pattern_only_metrics(
                adapter_out=adapter_out,
                labels=labels,
                mask=test_mask,
                num_classes=num_classes,
                prefix="p22_pattern_only_test",
            )
        )
        for eval_mask, eval_prefix in (
            (train_mask, "adapter_train"),
            (val_mask, "adapter_val"),
            (test_mask, "adapter_test"),
        ):
            adapter_diag.update(
                _prompt_adapter_candidate_pool_delta_stats(
                    logits_prompt=logits,
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=labels,
                    split_mask=eval_mask,
                    candidate_mask=update_mask,
                    prefix=eval_prefix,
                )
            )
        adapter_diag.update(_prompt_router_diagnostics(adapter_out))
        for eval_mask, eval_prefix in (
            (train_mask, "adapter_train"),
            (val_mask, "adapter_val"),
            (test_mask, "adapter_test"),
        ):
            adapter_diag.update(
                _prompt_router_delta_breakdown(
                    adapter_out=adapter_out,
                    logits_prompt=logits,
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=labels,
                    mask=eval_mask,
                    prefix=eval_prefix,
                )
            )
        if "channel_deltas" in adapter_out:
            for eval_mask, eval_prefix in (
                (train_mask, "p21_oracle_train"),
                (val_mask, "p21_oracle_val"),
                (test_mask, "p21_oracle_test"),
            ):
                _, _, oracle_stats = p21_channel_utility_supervision_loss(
                    adapter_out=adapter_out,
                    model=model,
                    h_pre=model_out["h_pre"],
                    h_adp_base=no_prompt_out["h_adp"],
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=labels,
                    mask=eval_mask,
                    prefix=eval_prefix,
                    temperature=float(p21_eval_cfg["p21_channel_utility_temperature"]),
                    margin=float(p21_eval_cfg["p21_channel_utility_margin"]),
                    min_teacher_delta=float(p21_eval_cfg["p21_channel_utility_min_teacher_delta"]),
                    target_mode=str(p21_eval_cfg["p21_channel_utility_target_mode"]),
                    gate_source=str(p21_eval_cfg["p21_oracle_gate_source"]),
                    gate_temperature=float(p21_eval_cfg["p21_gate_utility_temperature"]),
                    gate_margin=float(p21_eval_cfg["p21_gate_utility_margin"]),
                    gate_target_mode=str(p21_eval_cfg["p21_gate_target_mode"]),
                    class_balanced=False,
                    num_classes=num_classes,
                )
                adapter_diag.update(oracle_stats)
            oracle_scale_grid = _float_grid(
                p21_eval_cfg.get("p21_oracle_scale_grid"),
                default=[0.25, 0.5, 1.0, 2.0, 4.0],
            )
            for eval_mask, eval_prefix in (
                (train_mask, "p21_v2_ungated_oracle_train"),
                (val_mask, "p21_v2_ungated_oracle_val"),
                (test_mask, "p21_v2_ungated_oracle_test"),
            ):
                adapter_diag.update(
                    p21_channel_ungated_oracle_diagnostics(
                        adapter_out=adapter_out,
                        model=model,
                        h_pre=model_out["h_pre"],
                        h_adp_base=no_prompt_out["h_adp"],
                        logits_no_prompt=no_prompt_out["logits"],
                        labels=labels,
                        mask=eval_mask,
                        prefix=eval_prefix,
                        scale_grid=oracle_scale_grid,
                        num_classes=num_classes,
                    )
                )
        if "pattern_messages" in adapter_out:
            oracle_scale_grid = _float_grid(
                prompt_adapter_cfg.get("prompt_router_expert_oracle_scale_grid"),
                default=[0.5, 1.0, 2.0, 4.0],
            )
            for eval_mask, eval_prefix in (
                (train_mask, "prompt_router_expert_train"),
                (val_mask, "prompt_router_expert_val"),
                (test_mask, "prompt_router_expert_test"),
            ):
                _, expert_stats = prompt_router_expert_utility_supervision_loss(
                    adapter_out=adapter_out,
                    model=model,
                    h_pre=model_out["h_pre"],
                    h_adp_base=no_prompt_out["h_adp"],
                    no_prompt_logits=no_prompt_out["logits"],
                    labels=labels,
                    mask=eval_mask,
                    num_classes=num_classes,
                    prefix=eval_prefix,
                    temperature=float(prompt_adapter_cfg.get("prompt_router_expert_utility_temperature", 0.5)),
                    margin=float(prompt_adapter_cfg.get("prompt_router_expert_utility_margin", 0.0005)),
                    target_mode=str(prompt_adapter_cfg.get("prompt_router_expert_utility_target", "soft")),
                    gain_temperature=prompt_adapter_cfg.get("prompt_router_expert_utility_gain_temperature"),
                    probe_norm=prompt_adapter_cfg.get("prompt_router_expert_utility_probe_norm"),
                    class_balanced=False,
                    gate_weight=float(prompt_adapter_cfg.get("prompt_router_expert_utility_gate_weight", 0.0)),
                    gate_target_mode=str(prompt_adapter_cfg.get("prompt_router_expert_utility_gate_target", "binary")),
                    gate_target_temperature=prompt_adapter_cfg.get("prompt_router_expert_utility_gate_temperature"),
                    oracle_scale_grid=oracle_scale_grid,
                )
                adapter_diag.update(expert_stats)
            for eval_mask, eval_prefix in (
                (train_mask & update_mask, "prompt_router_expert_train_pool"),
                (val_mask & update_mask, "prompt_router_expert_val_pool"),
                (test_mask & update_mask, "prompt_router_expert_test_pool"),
            ):
                _, expert_stats = prompt_router_expert_utility_supervision_loss(
                    adapter_out=adapter_out,
                    model=model,
                    h_pre=model_out["h_pre"],
                    h_adp_base=no_prompt_out["h_adp"],
                    no_prompt_logits=no_prompt_out["logits"],
                    labels=labels,
                    mask=eval_mask,
                    num_classes=num_classes,
                    prefix=eval_prefix,
                    temperature=float(prompt_adapter_cfg.get("prompt_router_expert_utility_temperature", 0.5)),
                    margin=float(prompt_adapter_cfg.get("prompt_router_expert_utility_margin", 0.0005)),
                    target_mode=str(prompt_adapter_cfg.get("prompt_router_expert_utility_target", "soft")),
                    gain_temperature=prompt_adapter_cfg.get("prompt_router_expert_utility_gain_temperature"),
                    probe_norm=prompt_adapter_cfg.get("prompt_router_expert_utility_probe_norm"),
                    class_balanced=False,
                    gate_weight=float(prompt_adapter_cfg.get("prompt_router_expert_utility_gate_weight", 0.0)),
                    gate_target_mode=str(prompt_adapter_cfg.get("prompt_router_expert_utility_gate_target", "binary")),
                    gate_target_temperature=prompt_adapter_cfg.get("prompt_router_expert_utility_gate_temperature"),
                    oracle_scale_grid=oracle_scale_grid,
                )
                adapter_diag.update(expert_stats)
        return {
            "train_acc": train["acc"],
            "train_macro_f1": train["macro_f1"],
            "train_auroc": train.get("auroc", 0.0),
            "train_auprc": train.get("auprc", 0.0),
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "val_auroc": val.get("auroc", 0.0),
            "val_auprc": val.get("auprc", 0.0),
            "test_acc": test["acc"],
            "test_macro_f1": test["macro_f1"],
            "test_auroc": test.get("auroc", 0.0),
            "test_auprc": test.get("auprc", 0.0),
            "alpha": float(model_out["alpha"].detach().item()),
            "branch_cosine": float(branch_cosine.detach().item()),
            "pool_ratio": float(update_mask.float().mean().item()),
            "train_pool_ratio": float((update_mask & train_mask.bool()).float().mean().item()),
            "prompt_node_count": 0,
            "prompt_edge_count": 0,
            "edge_scale": 0.0,
            "raw_edge_scale": 0.0,
            "edge_scale_multiplier": 0.0,
            "mean_prompt_edge_weight": 0.0,
            **adapter_diag,
        }
    h_pre = None
    no_prompt_out = None
    if _needs_no_prompt_pool_evidence(prompt_graph_module):
        h_pre = model.encode_frozen(z, edge_index)
        no_prompt_out = _forward_no_prompt_with_h_pre(
            model=model,
            z=z,
            edge_index=edge_index,
            h_pre=h_pre,
        )
    model_out, prompt_out = _forward_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        z=z,
        x_raw=x,
        edge_index=edge_index,
        train_mask=train_mask,
        labels=labels,
        edge_scale_multiplier=edge_scale_multiplier,
        epoch=epoch,
        h_pre=h_pre,
        no_prompt_logits=None if no_prompt_out is None else no_prompt_out["logits"],
        h_adp_no_prompt=None if no_prompt_out is None else no_prompt_out["h_adp"],
    )
    logits = model_out["logits"]
    _attach_p23_ce_delta_diagnostics(
        model_out=model_out,
        prompt_out=prompt_out,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    branch_cosine = torch.nn.functional.cosine_similarity(model_out["h_pre"], model_out["h_adp"], dim=-1).mean()
    train = split_metrics(logits, labels, train_mask, num_classes=num_classes)
    val = split_metrics(logits, labels, val_mask, num_classes=num_classes)
    test = split_metrics(logits, labels, test_mask, num_classes=num_classes)
    no_receiver_metrics: dict[str, float] = {}
    no_receiver_logits = model_out.get("p23_no_receiver_logits")
    if isinstance(no_receiver_logits, torch.Tensor):
        no_receiver_val = split_metrics(
            no_receiver_logits, labels, val_mask, num_classes=num_classes
        )
        no_receiver_test = split_metrics(
            no_receiver_logits, labels, test_mask, num_classes=num_classes
        )
        no_receiver_metrics = {
            "p23_no_receiver_val_acc": float(no_receiver_val["acc"]),
            "p23_no_receiver_val_macro_f1": float(no_receiver_val["macro_f1"]),
            "p23_no_receiver_test_acc": float(no_receiver_test["acc"]),
            "p23_no_receiver_test_macro_f1": float(no_receiver_test["macro_f1"]),
            "p23_receiver_test_acc_delta": float(test["acc"] - no_receiver_test["acc"]),
            "p23_receiver_test_macro_f1_delta": float(
                test["macro_f1"] - no_receiver_test["macro_f1"]
            ),
        }
    counterfactual_diag: dict[str, float] = {}
    static_logits = model_out.get("p24_static_logits")
    candidate_logits = model_out.get("p24_candidate_logits")
    if isinstance(static_logits, torch.Tensor) and isinstance(candidate_logits, torch.Tensor):
        for split_name, split_mask in (
            ("train", train_mask),
            ("val", val_mask),
            ("test", test_mask),
        ):
            mask = split_mask.to(device=logits.device, dtype=torch.bool)
            if not bool(mask.any()):
                continue
            split_labels = labels.to(device=logits.device, dtype=torch.long)[mask]
            static_ce = F.cross_entropy(static_logits[mask], split_labels)
            candidate_ce = F.cross_entropy(candidate_logits[mask], split_labels)
            deployed_ce = F.cross_entropy(logits[mask], split_labels)
            static_prediction = static_logits[mask].argmax(dim=-1)
            static_metrics = split_metrics(
                static_logits,
                labels,
                split_mask,
                num_classes=num_classes,
            )
            candidate_metrics = split_metrics(
                candidate_logits,
                labels,
                split_mask,
                num_classes=num_classes,
            )
            coverage_diagnostics: dict[str, float] = {}
            for branch_name, branch_logits in (
                ("static", static_logits),
                ("candidate", candidate_logits),
                ("deployed", logits),
            ):
                _, branch_coverage = _p24_class_coverage_guard_loss(
                    logits=branch_logits,
                    labels=labels,
                    mask=split_mask,
                    num_classes=num_classes,
                )
                coverage_diagnostics.update(
                    {
                        f"p24_{split_name}_{branch_name}_prediction_coverage_ratio": branch_coverage[
                            "p24_prediction_coverage_ratio"
                        ],
                        f"p24_{split_name}_{branch_name}_prediction_dominant_ratio": branch_coverage[
                            "p24_prediction_dominant_ratio"
                        ],
                        f"p24_{split_name}_{branch_name}_prediction_entropy_ratio": branch_coverage[
                            "p24_prediction_marginal_entropy_ratio"
                        ],
                    }
                )
            counterfactual_diag.update(
                {
                    f"p24_{split_name}_static_ce": float(static_ce.item()),
                    f"p24_{split_name}_candidate_ce": float(candidate_ce.item()),
                    f"p24_{split_name}_deployed_ce": float(deployed_ce.item()),
                    f"p24_{split_name}_candidate_gain": float((static_ce - candidate_ce).item()),
                    f"p24_{split_name}_deployed_gain": float((static_ce - deployed_ce).item()),
                    f"p24_{split_name}_candidate_change_ratio": float(
                        (candidate_logits[mask].argmax(dim=-1) != static_prediction).float().mean().item()
                    ),
                    f"p24_{split_name}_deployed_change_ratio": float(
                        (logits[mask].argmax(dim=-1) != static_prediction).float().mean().item()
                    ),
                    f"p24_{split_name}_static_acc": static_metrics["acc"],
                    f"p24_{split_name}_static_macro_f1": static_metrics["macro_f1"],
                    f"p24_{split_name}_candidate_acc": candidate_metrics["acc"],
                    f"p24_{split_name}_candidate_macro_f1": candidate_metrics["macro_f1"],
                    **coverage_diagnostics,
                }
            )
    return {
        "train_acc": train["acc"],
        "train_macro_f1": train["macro_f1"],
        "train_auroc": train.get("auroc", 0.0),
        "train_auprc": train.get("auprc", 0.0),
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "val_auroc": val.get("auroc", 0.0),
        "val_auprc": val.get("auprc", 0.0),
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_auroc": test.get("auroc", 0.0),
        "test_auprc": test.get("auprc", 0.0),
        "alpha": float(model_out["alpha"].detach().item()),
        "branch_cosine": float(branch_cosine.detach().item()),
        **no_receiver_metrics,
        **counterfactual_diag,
        **_prompt_aware_diagnostics(model_out),
        **_prompt_graph_diagnostics(
            prompt_out,
            z=z,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            labels=labels,
            num_classes=num_classes,
        ),
    }


def _empty_prompt_message_utility(scale: float, reason: str) -> dict[str, Any]:
    return {
        "diagnostic": "prompt_message_utility",
        "message_scale": float(scale),
        "model_state": "final_after_training",
        "reason": reason,
        "train_pool_count": 0,
        "val_pool_count": 0,
        "test_pool_count": 0,
        "mean_delta_ce": 0.0,
        "median_delta_ce": 0.0,
        "positive_delta_ratio": 0.0,
        "mean_ce_no_prompt": 0.0,
        "mean_ce_prompt": 0.0,
        "mean_delta_ce_train_pool": 0.0,
        "positive_delta_ratio_train_pool": 0.0,
        "mean_delta_ce_val_pool": 0.0,
        "positive_delta_ratio_val_pool": 0.0,
        "mean_delta_ce_test_pool": 0.0,
        "positive_delta_ratio_test_pool": 0.0,
        "delta_ce_by_class": {},
        "delta_ce_by_prompt_slot": {},
        "delta_ce_by_acceptance_gate": {},
        "delta_ce_by_structural_score": {},
        "acceptance_delta_correlation": 0.0,
        "structural_score_delta_correlation": 0.0,
        "node_records": [],
    }


def _group_delta_stats(
    *,
    keys: torch.Tensor,
    delta_ce: torch.Tensor,
    positive: torch.Tensor,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if keys.numel() == 0:
        return out
    for key in torch.unique(keys.detach()).tolist():
        key_int = int(key)
        mask = keys == key_int
        if not bool(mask.any()):
            continue
        out[str(key_int)] = {
            "count": int(mask.sum().item()),
            "mean_delta_ce": float(delta_ce[mask].mean().item()),
            "positive_delta_ratio": float(positive[mask].float().mean().item()),
        }
    return out


def _bin_delta_stats(
    *,
    values: torch.Tensor,
    delta_ce: torch.Tensor,
    positive: torch.Tensor,
    prefix: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if values.numel() == 0:
        return out
    if values.numel() < 3 or float((values.max() - values.min()).abs().item()) < 1e-12:
        mask = torch.ones_like(values, dtype=torch.bool)
        out[f"{prefix}_all"] = {
            "count": int(mask.sum().item()),
            "value_min": float(values.min().item()),
            "value_max": float(values.max().item()),
            "value_mean": float(values.mean().item()),
            "mean_delta_ce": float(delta_ce[mask].mean().item()),
            "positive_delta_ratio": float(positive[mask].float().mean().item()),
        }
        return out
    q1 = torch.quantile(values.float(), 1.0 / 3.0)
    q2 = torch.quantile(values.float(), 2.0 / 3.0)
    bins = {
        f"{prefix}_low": values <= q1,
        f"{prefix}_mid": (values > q1) & (values <= q2),
        f"{prefix}_high": values > q2,
    }
    for name, mask in bins.items():
        if not bool(mask.any()):
            continue
        out[name] = {
            "count": int(mask.sum().item()),
            "value_min": float(values[mask].min().item()),
            "value_max": float(values[mask].max().item()),
            "value_mean": float(values[mask].mean().item()),
            "mean_delta_ce": float(delta_ce[mask].mean().item()),
            "positive_delta_ratio": float(positive[mask].float().mean().item()),
        }
    return out


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    x_centered = x.float() - x.float().mean()
    y_centered = y.float() - y.float().mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x_centered * y_centered).sum().div(denom).item())


def _pool_split_delta_stats(
    *,
    split_name: str,
    split_mask: torch.Tensor | None,
    pool_mask: torch.Tensor,
    logits_no_prompt: torch.Tensor,
    logits_prompt: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    prefix = f"{split_name}_pool"
    if split_mask is None:
        return {
            f"{prefix}_count": 0,
            f"mean_delta_ce_{prefix}": 0.0,
            f"median_delta_ce_{prefix}": 0.0,
            f"positive_delta_ratio_{prefix}": 0.0,
            f"mean_ce_no_prompt_{prefix}": 0.0,
            f"mean_ce_prompt_{prefix}": 0.0,
            f"delta_ce_by_class_{prefix}": {},
        }
    mask = split_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return {
            f"{prefix}_count": 0,
            f"mean_delta_ce_{prefix}": 0.0,
            f"median_delta_ce_{prefix}": 0.0,
            f"positive_delta_ratio_{prefix}": 0.0,
            f"mean_ce_no_prompt_{prefix}": 0.0,
            f"mean_ce_prompt_{prefix}": 0.0,
            f"delta_ce_by_class_{prefix}": {},
        }
    y = labels.to(device=idx.device)[idx]
    ce_no_prompt = F.cross_entropy(logits_no_prompt[idx], y, reduction="none")
    ce_prompt = F.cross_entropy(logits_prompt[idx], y, reduction="none")
    delta_ce = ce_no_prompt - ce_prompt
    positive = delta_ce > 0.0
    return {
        f"{prefix}_count": int(idx.numel()),
        f"mean_delta_ce_{prefix}": float(delta_ce.mean().item()),
        f"median_delta_ce_{prefix}": float(delta_ce.median().item()),
        f"positive_delta_ratio_{prefix}": float(positive.float().mean().item()),
        f"mean_ce_no_prompt_{prefix}": float(ce_no_prompt.mean().item()),
        f"mean_ce_prompt_{prefix}": float(ce_prompt.mean().item()),
        f"delta_ce_by_class_{prefix}": _group_delta_stats(keys=y, delta_ce=delta_ce, positive=positive),
    }


@torch.no_grad()
def diagnose_prompt_message_utility(
    *,
    model: DualBranchFramework,
    prompt_graph_module: PromptGraphModuleP1 | None,
    input_aligner: InputAligner,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    test_mask: torch.Tensor | None = None,
    message_scale: float,
) -> dict[str, Any]:
    """Measure whether prompt-on logits reduce CE on train-pool nodes.

    This diagnostic intentionally uses a fixed positive message scale. It does
    not let validation select scale=0 before measuring prompt utility.
    """

    if prompt_graph_module is None:
        return _empty_prompt_message_utility(message_scale, "prompt_graph_module_disabled")
    model.eval()
    input_aligner.eval()
    prompt_graph_module.eval()

    old_scale: float | None = None
    if isinstance(model, PromptAwareGP2F):
        old_scale = float(model.prompt_message_scale)
        model.prompt_message_scale = float(message_scale)

    try:
        z = input_aligner(x)
        h_pre = None
        model_no_prompt = None
        if _needs_no_prompt_pool_evidence(prompt_graph_module):
            h_pre = model.encode_frozen(z, edge_index)
            model_no_prompt = _forward_no_prompt_with_h_pre(
                model=model,
                z=z,
                edge_index=edge_index,
                h_pre=h_pre,
            )
        model_prompt, prompt_out = _forward_prompt_graph(
            model=model,
            prompt_graph_module=prompt_graph_module,
            z=z,
            x_raw=x,
            edge_index=edge_index,
            train_mask=train_mask,
            labels=labels,
            edge_scale_multiplier=1.0,
            h_pre=h_pre,
            no_prompt_logits=None if model_no_prompt is None else model_no_prompt["logits"],
            h_adp_no_prompt=None if model_no_prompt is None else model_no_prompt["h_adp"],
        )
        if model_no_prompt is None:
            model_no_prompt = _forward_no_prompt_with_h_pre(
                model=model,
                z=z,
                edge_index=edge_index,
                h_pre=model_prompt["h_pre_shared"],
            )
    finally:
        if old_scale is not None:
            model.prompt_message_scale = old_scale

    pool_mask = prompt_out.get("pool_mask")
    if not isinstance(pool_mask, torch.Tensor):
        return _empty_prompt_message_utility(message_scale, "missing_pool_mask")
    train_pool_mask = train_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()
    train_pool_idx = torch.where(train_pool_mask)[0]
    if train_pool_idx.numel() == 0:
        return _empty_prompt_message_utility(message_scale, "empty_train_pool")

    y = labels.to(device=train_pool_idx.device)[train_pool_idx]
    ce_no_prompt = F.cross_entropy(model_no_prompt["logits"][train_pool_idx], y, reduction="none")
    ce_prompt = F.cross_entropy(model_prompt["logits"][train_pool_idx], y, reduction="none")
    delta_ce = ce_no_prompt - ce_prompt
    positive = delta_ce > 0.0

    aux = prompt_out.get("aux", {})
    pool_idx = aux.get("pool_idx")
    pool_acceptance = aux.get("pool_acceptance_gate")
    structural_score = aux.get("structural_score")
    routing_full_prob = aux.get("routing_full_prob")
    top_prompt_ids = aux.get("top_prompt_ids")

    pool_position = torch.full((pool_mask.numel(),), -1, dtype=torch.long, device=pool_mask.device)
    if isinstance(pool_idx, torch.Tensor) and pool_idx.numel() > 0:
        pool_position[pool_idx] = torch.arange(pool_idx.numel(), dtype=torch.long, device=pool_idx.device)
    train_pool_pos = pool_position[train_pool_idx]
    valid_pool_pos = train_pool_pos >= 0

    if isinstance(pool_acceptance, torch.Tensor) and pool_acceptance.numel() > 0 and bool(valid_pool_pos.all()):
        acceptance = pool_acceptance[train_pool_pos]
    else:
        acceptance = delta_ce.new_ones(delta_ce.numel())
    if isinstance(structural_score, torch.Tensor) and structural_score.numel() >= pool_mask.numel():
        train_structural_score = structural_score[train_pool_idx]
    else:
        train_structural_score = delta_ce.new_zeros(delta_ce.numel())
    if isinstance(top_prompt_ids, torch.Tensor) and top_prompt_ids.numel() > 0 and bool(valid_pool_pos.all()):
        primary_prompt_slot = top_prompt_ids[train_pool_pos, 0].to(dtype=torch.long)
    elif isinstance(routing_full_prob, torch.Tensor) and routing_full_prob.numel() > 0 and bool(valid_pool_pos.all()):
        primary_prompt_slot = routing_full_prob[train_pool_pos].argmax(dim=-1).to(dtype=torch.long)
    else:
        primary_prompt_slot = torch.full_like(train_pool_idx, -1)

    result = {
        "diagnostic": "prompt_message_utility",
        "message_scale": float(message_scale),
        "model_state": "final_after_training",
        "reason": "",
        "train_pool_count": int(train_pool_idx.numel()),
        "val_pool_count": int(((val_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()).sum().item())) if isinstance(val_mask, torch.Tensor) else 0,
        "test_pool_count": int(((test_mask.to(device=pool_mask.device, dtype=torch.bool) & pool_mask.bool()).sum().item())) if isinstance(test_mask, torch.Tensor) else 0,
        "mean_delta_ce": float(delta_ce.mean().item()),
        "median_delta_ce": float(delta_ce.median().item()),
        "positive_delta_ratio": float(positive.float().mean().item()),
        "mean_ce_no_prompt": float(ce_no_prompt.mean().item()),
        "mean_ce_prompt": float(ce_prompt.mean().item()),
        "delta_ce_by_class": _group_delta_stats(keys=y, delta_ce=delta_ce, positive=positive),
        "delta_ce_by_prompt_slot": _group_delta_stats(
            keys=primary_prompt_slot,
            delta_ce=delta_ce,
            positive=positive,
        ),
        "delta_ce_by_acceptance_gate": _bin_delta_stats(
            values=acceptance,
            delta_ce=delta_ce,
            positive=positive,
            prefix="acceptance",
        ),
        "delta_ce_by_structural_score": _bin_delta_stats(
            values=train_structural_score,
            delta_ce=delta_ce,
            positive=positive,
            prefix="structural_score",
        ),
        "acceptance_delta_correlation": _pearson_corr(acceptance, delta_ce),
        "structural_score_delta_correlation": _pearson_corr(train_structural_score, delta_ce),
        "node_records": [],
    }
    result.update(
        _pool_split_delta_stats(
            split_name="train",
            split_mask=train_mask,
            pool_mask=pool_mask,
            logits_no_prompt=model_no_prompt["logits"],
            logits_prompt=model_prompt["logits"],
            labels=labels,
        )
    )
    result.update(
        _pool_split_delta_stats(
            split_name="val",
            split_mask=val_mask,
            pool_mask=pool_mask,
            logits_no_prompt=model_no_prompt["logits"],
            logits_prompt=model_prompt["logits"],
            labels=labels,
        )
    )
    result.update(
        _pool_split_delta_stats(
            split_name="test",
            split_mask=test_mask,
            pool_mask=pool_mask,
            logits_no_prompt=model_no_prompt["logits"],
            logits_prompt=model_prompt["logits"],
            labels=labels,
        )
    )
    for row, node_id in enumerate(train_pool_idx.detach().cpu().tolist()):
        result["node_records"].append(
            {
                "node_id": int(node_id),
                "label": int(y[row].detach().cpu().item()),
                "ce_no_prompt": float(ce_no_prompt[row].detach().cpu().item()),
                "ce_prompt": float(ce_prompt[row].detach().cpu().item()),
                "delta_ce": float(delta_ce[row].detach().cpu().item()),
                "benefited": bool(positive[row].detach().cpu().item()),
                "primary_prompt_slot": int(primary_prompt_slot[row].detach().cpu().item()),
                "acceptance_gate": float(acceptance[row].detach().cpu().item()),
                "structural_score": float(train_structural_score[row].detach().cpu().item()),
            }
        )
    return result


def _write_prompt_message_utility_csv(path: Path, utility: dict[str, Any]) -> None:
    records = utility.get("node_records", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "node_id",
        "label",
        "ce_no_prompt",
        "ce_prompt",
        "delta_ce",
        "benefited",
        "primary_prompt_slot",
        "acceptance_gate",
        "structural_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def run_single(
    config: dict[str, Any],
    *,
    repo_root: Path,
    run_index: int = 0,
    total_runs: int = 1,
    run_group_dir: Path | None = None,
) -> dict[str, Any]:
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    pretrained_cfg = config.get("pretrained", {})
    model_cfg = config.get("model", {})
    prompt_graph_cfg = config.get("prompt_graph", {})
    prompt_aware_cfg = config.get("prompt_aware", {})
    prompt_adapter_cfg = config.get("prompt_adapter", {})
    training_cfg = config.get("training", {})
    loss_cfg_raw = config.get("loss", {})

    seed = int(experiment_cfg.get("seed", 0))
    split_seed = int(experiment_cfg.get("split_seed", seed))
    trial_index = int(experiment_cfg.get("trial_index", 0))
    if bool(experiment_cfg.get("reset_seed", True)):
        set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dataset = str(experiment_cfg.get("target_dataset", "Cora"))
    variant = _prompt_variant(config)

    data_root = _resolve_path(data_cfg.get("root", "datasets"), base_dir=repo_root)
    loaded = load_node_dataset(
        target_dataset,
        data_root,
        download_if_missing=bool(data_cfg.get("download_if_missing", False)),
        normalize_features=(
            None
            if data_cfg.get("normalize_features") is None
            else bool(data_cfg["normalize_features"])
        ),
    )
    graph = loaded.data.to(device)
    gp2f_aux_loss_cfg = GP2FLossConfig(
        use_cls=False,
        use_original_contrastive=bool(loss_cfg_raw.get("use_original_contrastive", False)),
        use_original_topology_fusion=bool(loss_cfg_raw.get("use_original_topology_fusion", False)),
        lambda_ctr=float(loss_cfg_raw.get("lambda_ctr", 0.0)),
        lambda_fus=float(loss_cfg_raw.get("lambda_fus", 0.0)),
        tau_ctr=float(loss_cfg_raw.get("tau_ctr", 0.5)),
        tau_fus=float(loss_cfg_raw.get("tau_fus", 0.05)),
        topology_percentile=float(loss_cfg_raw.get("topology_percentile", 70.0)),
        topology_threshold_mode=str(loss_cfg_raw.get("topology_threshold_mode", "exact")),
        topology_threshold_sample_size=int(
            loss_cfg_raw.get("topology_threshold_sample_size", 5000)
        ),
        topology_threshold_large_graph_nodes=int(
            loss_cfg_raw.get("topology_threshold_large_graph_nodes", 1000)
        ),
        dense_contrastive_max_nodes=int(loss_cfg_raw.get("dense_contrastive_max_nodes", 5000)),
        contrastive_sample_size=int(loss_cfg_raw.get("contrastive_sample_size", 2048)),
        dense_topology_max_nodes=int(loss_cfg_raw.get("dense_topology_max_nodes", 5000)),
        topology_sample_size=int(loss_cfg_raw.get("topology_sample_size", 20000)),
    )
    effective_shots, effective_shot_ratio, shot_setting_mode = _resolve_shot_setting(
        data_cfg,
        experiment_cfg,
        target_dataset,
    )
    split_protocol = str(data_cfg.get("split_protocol", "generated"))
    if split_protocol == "gp2f_precomputed":
        if effective_shot_ratio is not None:
            raise ValueError("gp2f_precomputed currently supports fixed-shot splits only")
        split = load_gp2f_precomputed_split(
            graph.y,
            sample_data_root=_resolve_path(
                data_cfg.get("gp2f_sample_data_root", "splits/gp2f"),
                base_dir=repo_root,
            ),
            dataset=str(data_cfg.get("gp2f_dataset_name", target_dataset)),
            shots=effective_shots,
            trial_index=trial_index,
            seed=split_seed,
            unused_as_validation=bool(data_cfg.get("gp2f_unused_as_validation", True)),
        )
    elif split_protocol == "generated":
        split = build_few_shot_split(
            graph.y,
            shots=effective_shots,
            shot_ratio=effective_shot_ratio,
            val_per_class=int(data_cfg.get("val_per_class", experiment_cfg.get("val_per_class", 30))),
            seed=split_seed,
            test_ratio=(
                None
                if data_cfg.get("gp2f_test_ratio") is None
                else float(data_cfg["gp2f_test_ratio"])
            ),
        )
    else:
        raise ValueError(
            f"Unsupported data.split_protocol={split_protocol!r}; "
            "expected 'generated' or 'gp2f_precomputed'"
        )

    checkpoint_path = _resolve_path(
        pretrained_cfg.get("checkpoint_path", "checkpoints/cora_grace.pt"),
        base_dir=repo_root,
    )
    backbone = load_pretrained_backbone(
        checkpoint_path,
        device=device,
        freeze=bool(pretrained_cfg.get("freeze_backbone", True)),
        graphmae_edge_weight_mode=str(
            pretrained_cfg.get("graphmae_edge_weight_mode", "attention_prior")
        ),
        graphmae_edge_weight_strength=float(
            pretrained_cfg.get("graphmae_edge_weight_strength", 1.0)
        ),
    )
    if "dropout" in model_cfg and model_cfg["dropout"] is not None:
        setattr(backbone, "dropout", float(model_cfg["dropout"]))
    source_dim, hidden_dim = infer_backbone_dimensions(backbone)
    input_aligner = InputAligner(
        loaded.num_features,
        source_dim,
        style=str(model_cfg.get("input_aligner", "linear")),
    ).to(device)
    model_kwargs = {
        "hidden_dim": hidden_dim,
        "num_classes": loaded.num_classes,
        "adapter_bottleneck_dim": int(model_cfg.get("adapter_bottleneck_dim", 16)),
        "adapter_beta_init": float(model_cfg.get("adapter_beta_init", 0.01)),
        "adapter_alpha_init": float(model_cfg.get("adapter_alpha_init", 0.1)),
        "adapter_style": str(model_cfg.get("adapter_style", "stable_zero_init")),
        "alpha_init": float(model_cfg.get("alpha_init", 0.5)),
        "fusion_alpha_style": str(model_cfg.get("fusion_alpha_style", "sigmoid")),
        "use_adapters": bool(model_cfg.get("use_adapters", True)),
        "fusion_mode": str(model_cfg.get("fusion_mode", "dual_branch")),
    }
    if bool(prompt_aware_cfg.get("enabled", False)):
        prompt_aware_cfg = dict(prompt_aware_cfg)
        prompt_aware_cfg.setdefault("num_prompt_nodes", int(prompt_graph_cfg.get("num_prompt_nodes", 16)))
        prompt_aware_cfg.setdefault("prompt_slot_head_count", int(prompt_graph_cfg.get("num_prompt_nodes", 16)))
        model = PromptAwareGP2F(backbone, prompt_aware_config=prompt_aware_cfg, **model_kwargs).to(device)
    else:
        model = DualBranchFramework(backbone, **model_kwargs).to(device)
    prompt_graph_module = _build_prompt_graph_module(
        variant=variant,
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        num_classes=loaded.num_classes,
        prompt_graph_cfg=prompt_graph_cfg,
        device=device,
    )
    prompt_adapter_module = _build_prompt_adapter_module(
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        num_classes=loaded.num_classes,
        prompt_adapter_cfg=prompt_adapter_cfg,
        device=device,
    )

    base_checkpoint_path = _resolve_base_checkpoint(
        training_cfg,
        repo_root=repo_root,
        seed=seed,
        split_seed=split_seed,
        trial_index=trial_index,
    )
    if variant == "p14_freeze_prompt_adapter" and base_checkpoint_path is None:
        print(
            "Warning: p14_freeze_prompt_adapter is running without a base checkpoint; "
            "the frozen classifier is randomly initialized, so this run is only a code-path smoke test."
        )
    if base_checkpoint_path is not None:
        _load_base_checkpoint(
            base_checkpoint_path,
            model=model,
            input_aligner=input_aligner,
            prompt_graph_module=prompt_graph_module,
            prompt_adapter_module=prompt_adapter_module,
            device=device,
            load_prompt=bool(training_cfg.get("load_prompt_from_base_checkpoint", False)),
        )
    enable_test_label_cheat = bool(training_cfg.get("enable_test_label_cheat", False))
    test_label_cheat_fraction = min(
        max(float(training_cfg.get("test_label_cheat_fraction", 0.10)), 0.0),
        1.0,
    )
    label_train_mask = split.train_mask.bool()
    if enable_test_label_cheat:
        # Oracle diagnostic only: deliberately leak test labels into the
        # supervised training pool while keeping eval splits unchanged. Leak a
        # deterministic subset per seed so oracle runs remain reproducible.
        test_idx = torch.where(split.test_mask.bool())[0]
        leak_count = int(test_idx.numel() * test_label_cheat_fraction)
        if test_label_cheat_fraction > 0.0 and test_idx.numel() > 0:
            leak_count = max(1, leak_count)
        leak_count = min(leak_count, int(test_idx.numel()))
        if leak_count > 0:
            generator = torch.Generator(device=test_idx.device)
            generator.manual_seed(int(seed) * 1000003 + 9176)
            selected = test_idx[torch.randperm(test_idx.numel(), device=test_idx.device, generator=generator)[:leak_count]]
            test_label_cheat_mask = torch.zeros_like(label_train_mask, dtype=torch.bool)
            test_label_cheat_mask[selected] = True
            label_train_mask = label_train_mask | test_label_cheat_mask
    cheat_test_label_count = int((label_train_mask & split.test_mask.bool()).sum().item())
    actual_test_label_cheat_fraction = cheat_test_label_count / max(1, int(split.test_mask.bool().sum().item()))

    init_train_mask = label_train_mask
    if bool(prompt_graph_cfg.get("support_only_prompt_graph", False)) and bool(
        prompt_graph_cfg.get("support_query_split", {}).get(
            "enabled", prompt_graph_cfg.get("support_query_split_enabled", False)
        )
    ):
        init_train_mask, _, _ = _support_query_masks_for_epoch(
            graph.y,
            label_train_mask,
            prompt_graph_cfg,
            seed=seed,
            epoch=0,
        )
    p23_static_init_stats = _maybe_build_p23_v01_static_state(
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        model=model,
        x_raw=graph.x,
        edge_index=graph.edge_index,
        train_mask=init_train_mask,
    )
    class_key_init_stats = _maybe_initialize_class_keys(
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        model=model,
        x=graph.x,
        edge_index=graph.edge_index,
        labels=graph.y,
        train_mask=init_train_mask,
        prompt_graph_cfg=prompt_graph_cfg,
    )
    pattern_key_init_stats = _maybe_initialize_pattern_keys(
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        model=model,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=init_train_mask,
        prompt_graph_cfg=prompt_graph_cfg,
    )
    freeze_base_model = bool(training_cfg.get("freeze_base_model", False))
    if freeze_base_model:
        _set_module_trainable(input_aligner, False)
        _set_module_trainable(model, False)
    p24_train_classifier = bool(
        isinstance(prompt_graph_module, PromptingAdaptation)
        and training_cfg.get("train_classifier_for_p24", False)
    )
    if p24_train_classifier:
        _set_module_trainable(model.classifier, True)
    p24_collapse_recovery_cfg = dict(training_cfg.get("p24_collapse_recovery", {}))
    if isinstance(prompt_graph_module, PromptingAdaptation):
        if bool(p24_collapse_recovery_cfg.get("train_input_aligner", False)):
            _set_module_trainable(input_aligner, True)
        if bool(p24_collapse_recovery_cfg.get("train_adapters", False)):
            _set_module_trainable(model.adapters, True)
        if bool(p24_collapse_recovery_cfg.get("train_fusion_alpha", False)):
            model.alpha_logit.requires_grad = True
    p23_freeze_input_aligner = (
        isinstance(prompt_graph_module, (P23V01PromptModule, P23ProjectorSynchronizedPromptGraphAdapter))
        and bool(training_cfg.get("freeze_input_aligner_for_p23", False))
    )
    if p23_freeze_input_aligner:
        _set_module_trainable(input_aligner, False)
    if isinstance(model, PromptAwareGP2F):
        _set_prompt_aware_trainable(
            model,
            bool(training_cfg.get("train_prompt_aware_module", True)),
        )
    prompt_graph_trainable_scope = str(
        training_cfg.get("prompt_graph_trainable_scope", "all")
    ).lower()
    if prompt_graph_module is not None:
        _set_prompt_graph_trainable_scope(
            prompt_graph_module,
            trainable=bool(training_cfg.get("train_prompt_graph_module", True)),
            scope=prompt_graph_trainable_scope,
        )
    if prompt_adapter_module is not None:
        _set_module_trainable(prompt_adapter_module, bool(training_cfg.get("train_prompt_adapter", True)))
    if not model.use_adapters:
        _set_module_trainable(model.adapters, False)
    if model.fusion_mode != "dual_branch":
        if model.raw_alpha is not None:
            model.raw_alpha.requires_grad = False
        if model.alpha_logit is not None:
            model.alpha_logit.requires_grad = False

    trainable_summary = _trainable_parameter_summary(
        input_aligner=input_aligner,
        model=model,
        prompt_graph_module=prompt_graph_module,
        prompt_adapter_module=prompt_adapter_module,
    )
    optimizer_groups, trainable_params, optimizer_summary = _optimizer_groups(
        input_aligner=input_aligner,
        model=model,
        prompt_graph_module=prompt_graph_module,
        prompt_adapter_module=prompt_adapter_module,
        training_cfg=training_cfg,
    )
    if not trainable_params:
        raise RuntimeError("No trainable parameters are enabled for this run")
    optimizer = torch.optim.Adam(optimizer_groups)

    output_root = _resolve_path(training_cfg.get("output_dir", "outputs/gp2f_prompt_p1"), base_dir=repo_root)
    if run_group_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_group_dir = output_root / loaded.name / timestamp
    if "trial_index" in experiment_cfg:
        run_dir = run_group_dir / f"seed_{split_seed}" / f"trial_{trial_index}"
    else:
        run_dir = run_group_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config)

    init_eq = _init_equivalence(
        model=model,
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        train_mask=label_train_mask,
        labels=graph.y,
        prompt_adapter_module=prompt_adapter_module,
    )

    epochs = int(training_cfg.get("epochs", 200))
    log_every = int(training_cfg.get("log_every", 20))
    eval_every = max(1, int(training_cfg.get("eval_every", 1)))
    monitor = str(training_cfg.get("early_stop_metric", "val_acc"))
    val_present_classes = sorted(
        int(value)
        for value in torch.unique(graph.y[split.val_mask.bool()]).detach().cpu().tolist()
    )
    val_missing_classes = [
        class_id for class_id in range(loaded.num_classes) if class_id not in val_present_classes
    ]
    val_class_coverage = len(val_present_classes) / max(1, loaded.num_classes)
    if monitor.startswith("val_") and val_missing_classes:
        print(
            "Warning: validation-based checkpoint selection is unreliable because "
            f"the validation split misses classes {val_missing_classes}; "
            "use --early_stop_metric train_loss to preserve the GP2F 90/10 protocol.",
            file=sys.stderr,
        )
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_epochs = int(training_cfg.get("early_stop_min_epochs", 0))
    keep_checkpoint = bool(training_cfg.get("keep_best_checkpoint", True))
    lambda_edge_l1 = float(prompt_graph_cfg.get("lambda_edge_l1", 0.0))
    lambda_prompt_balance = float(prompt_graph_cfg.get("lambda_prompt_balance", 0.0))
    lambda_p23_feature_harm = float(prompt_graph_cfg.get("lambda_p23_feature_harm", 0.0))
    lambda_p23_feature_gate_supervision = float(prompt_graph_cfg.get("lambda_p23_feature_gate_supervision", 0.0))
    p23_feature_harm_margin = float(prompt_graph_cfg.get("p23_feature_harm_margin", 0.02))
    p23_feature_harm_temperature = float(prompt_graph_cfg.get("p23_feature_harm_temperature", 0.15))
    p23_feature_harm_quantile = float(prompt_graph_cfg.get("p23_feature_harm_quantile", 0.70))
    p23_feature_help_quantile = float(prompt_graph_cfg.get("p23_feature_help_quantile", 0.70))
    p23_feature_min_signal = float(prompt_graph_cfg.get("p23_feature_min_signal", 0.005))
    p23_feature_help_weight = float(prompt_graph_cfg.get("p23_feature_help_weight", 0.25))
    p23_feature_harm_warmup_epochs = int(prompt_graph_cfg.get("p23_feature_harm_warmup_epochs", 0))
    p23_feature_harm_detach_delta = bool(prompt_graph_cfg.get("p23_feature_harm_detach_delta", True))
    p23_feature_harm_class_balanced = bool(prompt_graph_cfg.get("p23_feature_harm_class_balanced", True))
    lambda_p23_prompt_residual_delta_norm = float(prompt_graph_cfg.get("lambda_p23_prompt_residual_delta_norm", 0.0))
    lambda_p23_prompt_residual_delta_align = float(prompt_graph_cfg.get("lambda_p23_prompt_residual_delta_align", 0.0))
    lambda_p23_prompt_residual_gate_budget = float(prompt_graph_cfg.get("lambda_p23_prompt_residual_gate_budget", 0.0))
    lambda_p24_keep_static = float(prompt_graph_cfg.get("lambda_p24_keep_static", 0.0))
    lambda_p24_hub_suppression = float(prompt_graph_cfg.get("lambda_p24_hub_suppression", 0.0))
    lambda_p24_prompt_homophily = float(prompt_graph_cfg.get("lambda_p24_prompt_homophily", 0.0))
    lambda_p24_edge_budget = float(prompt_graph_cfg.get("lambda_p24_edge_budget", 0.0))
    lambda_p24_receiver_budget = float(
        prompt_graph_cfg.get("lambda_p24_receiver_budget", 0.0)
    )
    lambda_p24_gumbel_keep_ratio = float(prompt_graph_cfg.get("lambda_p24_gumbel_keep_ratio", 0.0))
    lambda_p24_gumbel_entropy = float(prompt_graph_cfg.get("lambda_p24_gumbel_entropy", 0.0))
    p24_gumbel_entropy_warmup_epochs = int(
        prompt_graph_cfg.get("p24_gumbel_entropy_warmup_epochs", 0)
    )
    lambda_p24_candidate_cls = float(prompt_graph_cfg.get("lambda_p24_candidate_cls", 0.0))
    lambda_p24_candidate_anti_harm = float(prompt_graph_cfg.get("lambda_p24_candidate_anti_harm", 0.0))
    lambda_p24_relation_prior = float(prompt_graph_cfg.get("lambda_p24_relation_prior", 0.0))
    lambda_p24_relation_preservation = float(
        prompt_graph_cfg.get("lambda_p24_relation_preservation", 0.0)
    )
    lambda_p24_relation_homophily_collapse = float(
        prompt_graph_cfg.get("lambda_p24_relation_homophily_collapse", 0.0)
    )
    lambda_p24_relation_uncertain_keep = float(
        prompt_graph_cfg.get("lambda_p24_relation_uncertain_keep", 0.0)
    )
    p24_candidate_anti_harm_margin = float(prompt_graph_cfg.get("p24_candidate_anti_harm_margin", 0.001))
    p24_candidate_loss_mask = str(prompt_graph_cfg.get("p24_candidate_loss_mask", "query")).lower()
    if p24_candidate_loss_mask not in {"support", "query", "all_train"}:
        raise ValueError("p24_candidate_loss_mask must be support, query, or all_train")
    lambda_p24_route_utility = float(prompt_graph_cfg.get("lambda_p24_route_utility", 0.0))
    p24_route_utility_warmup_epochs = int(prompt_graph_cfg.get("p24_route_utility_warmup_epochs", 0))
    p24_route_positive_margin = float(prompt_graph_cfg.get("p24_route_positive_margin", 0.005))
    p24_route_negative_margin = float(prompt_graph_cfg.get("p24_route_negative_margin", 0.005))
    p24_route_adapt_positive_margin = float(
        prompt_graph_cfg.get("p24_route_adapt_positive_margin", p24_route_positive_margin)
    )
    p24_route_adapt_negative_margin = float(
        prompt_graph_cfg.get("p24_route_adapt_negative_margin", p24_route_negative_margin)
    )
    p24_route_no_prompt_positive_margin = float(
        prompt_graph_cfg.get("p24_route_no_prompt_positive_margin", p24_route_positive_margin)
    )
    p24_route_no_prompt_negative_margin = float(
        prompt_graph_cfg.get("p24_route_no_prompt_negative_margin", p24_route_negative_margin)
    )
    p24_route_class_balanced = bool(prompt_graph_cfg.get("p24_route_class_balanced", True))
    p24_route_supervise_no_prompt = bool(prompt_graph_cfg.get("p24_route_supervise_no_prompt", True))
    p24_route_target_mode = str(prompt_graph_cfg.get("p24_route_target_mode", "hard"))
    p24_route_gain_scale = float(prompt_graph_cfg.get("p24_route_gain_scale", 0.01))
    p24_class_coverage_cfg = dict(prompt_graph_cfg.get("p24_class_coverage_guard", {}))
    p24_class_coverage_enabled = bool(p24_class_coverage_cfg.get("enabled", False))
    lambda_p24_class_coverage_guard = float(
        prompt_graph_cfg.get("lambda_p24_class_coverage_guard", 0.0)
    )
    p24_class_coverage_warmup_epochs = int(p24_class_coverage_cfg.get("warmup_epochs", 0))
    p24_class_coverage_loss_mask = str(
        p24_class_coverage_cfg.get("loss_mask", "all_train")
    ).lower()
    if p24_class_coverage_loss_mask not in {"support", "query", "all_train"}:
        raise ValueError("p24_class_coverage_guard.loss_mask must be support, query, or all_train")
    lambda_p24_prompt_utility = float(prompt_graph_cfg.get("lambda_p24_prompt_utility", 0.0))
    p24_prompt_utility_warmup_epochs = int(
        prompt_graph_cfg.get("p24_prompt_utility_warmup_epochs", p24_route_utility_warmup_epochs)
    )
    p24_prompt_utility_shrinkage_tau = float(
        prompt_graph_cfg.get("p24_prompt_utility_shrinkage_tau", 10.0)
    )
    p24_prompt_utility_gain_scale = float(
        prompt_graph_cfg.get("p24_prompt_utility_gain_scale", p24_route_gain_scale)
    )
    p24_prompt_utility_target_log_scale = float(
        prompt_graph_cfg.get("p24_prompt_utility_target_log_scale", 0.10)
    )
    p24_prompt_utility_hub_quantile = float(
        prompt_graph_cfg.get("p24_prompt_utility_hub_quantile", 0.80)
    )
    p24_denoising_cfg = dict(prompt_graph_cfg.get("prompt_edge_denoising", {}))
    p24_counterfactual_cfg = dict(p24_denoising_cfg.get("counterfactual_router", {}))
    p24_counterfactual_enabled = bool(p24_counterfactual_cfg.get("enabled", False))
    p24_counterfactual_ema_decay = float(p24_counterfactual_cfg.get("gain_ema", 0.90))
    p24_counterfactual_gain_cap = float(p24_counterfactual_cfg.get("gain_cap", 0.02))
    p24_counterfactual_gain_margin = float(p24_counterfactual_cfg.get("gain_margin", 0.0005))
    p24_counterfactual_harm_weight = float(p24_counterfactual_cfg.get("harm_weight", 2.0))
    p24_counterfactual_rank_weight = float(p24_counterfactual_cfg.get("rank_weight", 0.10))
    p24_counterfactual_budget_weight = float(p24_counterfactual_cfg.get("budget_weight", 0.02))
    p24_counterfactual_positive_adapt_floor = float(
        p24_counterfactual_cfg.get("positive_adapt_floor", 0.05)
    )
    p24_causal_teacher_cfg = dict(p24_denoising_cfg.get("causal_prompt_teacher", {}))
    p24_causal_teacher_enabled = bool(p24_causal_teacher_cfg.get("enabled", False))
    lambda_p24_causal_prompt_teacher = float(
        prompt_graph_cfg.get("lambda_p24_causal_prompt_teacher", 0.0)
    )
    p24_causal_teacher_state: dict[str, torch.Tensor] | None = None
    p24_gain_ema = torch.full(
        (int(graph.y.numel()),),
        float("nan"),
        dtype=torch.float32,
        device=graph.y.device,
    )
    lambda_prompt_role_diversity = float(prompt_graph_cfg.get("lambda_prompt_role_diversity", 0.0))
    lambda_prompt_acceptance = float(prompt_graph_cfg.get("lambda_prompt_acceptance", 0.0))
    lambda_prompt_acceptance_budget = float(prompt_graph_cfg.get("lambda_prompt_acceptance_budget", 0.0))
    acceptance_budget_min_raw = prompt_graph_cfg.get("acceptance_budget_min")
    acceptance_budget_max_raw = prompt_graph_cfg.get("acceptance_budget_max")
    acceptance_budget_min = None if acceptance_budget_min_raw is None else float(acceptance_budget_min_raw)
    acceptance_budget_max = None if acceptance_budget_max_raw is None else float(acceptance_budget_max_raw)
    lambda_prompt_acceptance_supervision = float(prompt_graph_cfg.get("lambda_prompt_acceptance_supervision", 0.0))
    acceptance_supervision_positive_margin = float(prompt_graph_cfg.get("acceptance_supervision_positive_margin", 0.0))
    acceptance_supervision_negative_margin = float(prompt_graph_cfg.get("acceptance_supervision_negative_margin", 0.0))
    acceptance_supervision_balance_targets = bool(prompt_graph_cfg.get("acceptance_supervision_balance_targets", True))
    acceptance_supervision_signal = str(prompt_graph_cfg.get("acceptance_supervision_signal", "ce_delta"))
    lambda_prompt_usage_consistency = float(prompt_graph_cfg.get("lambda_prompt_usage_consistency", 0.0))
    usage_consistency_margin = float(prompt_graph_cfg.get("usage_consistency_margin", 0.25))
    usage_consistency_negative_weight = float(prompt_graph_cfg.get("usage_consistency_negative_weight", 0.05))
    lambda_prompt_view_entropy = float(prompt_graph_cfg.get("lambda_prompt_view_entropy", 0.0))
    lambda_view_prior = float(prompt_graph_cfg.get("lambda_view_prior", 0.0))
    view_prior = prompt_graph_cfg.get("view_prior", None)
    lambda_class_route = float(prompt_graph_cfg.get("lambda_class_route", 0.0))
    lambda_key_proto = float(prompt_graph_cfg.get("lambda_key_proto", 0.0))
    lambda_prompt_benefit_supervision = float(prompt_graph_cfg.get("lambda_prompt_benefit_supervision", 0.0))
    benefit_supervision_warmup_epochs = int(prompt_graph_cfg.get("benefit_supervision_warmup_epochs", 0))
    benefit_delta_margin = float(prompt_graph_cfg.get("benefit_delta_margin", 0.0))
    benefit_supervision_balance_targets = bool(prompt_graph_cfg.get("benefit_supervision_balance_targets", True))
    benefit_supervision_label_strategy = str(prompt_graph_cfg.get("benefit_supervision_label_strategy", "margin"))
    benefit_supervision_quantile = float(prompt_graph_cfg.get("benefit_supervision_quantile", 0.20))
    benefit_delta_eps = float(prompt_graph_cfg.get("benefit_delta_eps", 0.0))
    benefit_supervision_quantile_warmup_epochs = int(
        prompt_graph_cfg.get("benefit_supervision_quantile_warmup_epochs", 0)
    )
    benefit_supervision_probe_scale_raw = prompt_graph_cfg.get("benefit_supervision_probe_scale")
    benefit_supervision_probe_scale = (
        None if benefit_supervision_probe_scale_raw is None else float(benefit_supervision_probe_scale_raw)
    )
    lambda_prompt_correction = float(prompt_graph_cfg.get("lambda_prompt_correction", 0.0))
    lambda_prompt_anti_harm = float(prompt_graph_cfg.get("lambda_prompt_anti_harm", 0.0))
    prompt_correction_warmup_epochs = int(prompt_graph_cfg.get("prompt_correction_warmup_epochs", 0))
    prompt_correction_eps = float(prompt_graph_cfg.get("prompt_correction_eps", 0.0))
    prompt_correction_target = str(prompt_graph_cfg.get("prompt_correction_target", "delta_prob_to_label"))
    lambda_prompt_message_help = float(prompt_graph_cfg.get("lambda_prompt_message_help", 0.0))
    prompt_message_help_margin = float(prompt_graph_cfg.get("prompt_message_help_margin", 0.0))
    prompt_message_help_warmup_epochs = int(prompt_graph_cfg.get("prompt_message_help_warmup_epochs", 0))
    prompt_message_help_class_balanced = bool(prompt_graph_cfg.get("prompt_message_help_class_balanced", True))
    lambda_prompt_message_help_query = float(prompt_graph_cfg.get("lambda_prompt_message_help_query", 0.0))
    prompt_message_help_query_warmup_epochs = int(
        prompt_graph_cfg.get("prompt_message_help_query_warmup_epochs", prompt_message_help_warmup_epochs)
    )
    lambda_prompt_class_anti_harm = float(prompt_graph_cfg.get("lambda_prompt_class_anti_harm", 0.0))
    prompt_class_anti_harm_floor = float(prompt_graph_cfg.get("prompt_class_anti_harm_floor", 0.0))
    lambda_utility_receive_gate = float(prompt_graph_cfg.get("lambda_utility_receive_gate", 0.0))
    lambda_utility_receive_gate_query = float(prompt_graph_cfg.get("lambda_utility_receive_gate_query", 0.0))
    lambda_receive_gate_budget = float(prompt_graph_cfg.get("lambda_receive_gate_budget", 0.0))
    receive_gate_budget_min = prompt_graph_cfg.get("receive_gate_budget_min")
    receive_gate_budget_max = prompt_graph_cfg.get("receive_gate_budget_max")
    utility_receive_gate_warmup_epochs = int(prompt_graph_cfg.get("utility_receive_gate_warmup_epochs", 0))
    utility_receive_gate_query_warmup_epochs = int(
        prompt_graph_cfg.get("utility_receive_gate_query_warmup_epochs", utility_receive_gate_warmup_epochs)
    )
    utility_receive_gate_quantile = float(prompt_graph_cfg.get("utility_receive_gate_quantile", 0.20))
    utility_receive_gate_eps = float(prompt_graph_cfg.get("utility_receive_gate_eps", 1e-4))
    utility_receive_gate_class_balanced = bool(prompt_graph_cfg.get("utility_receive_gate_class_balanced", True))
    utility_receive_gate_label_strategy = str(prompt_graph_cfg.get("utility_receive_gate_label_strategy", "quantile"))
    lambda_query_proto_alignment = float(prompt_graph_cfg.get("lambda_query_proto_alignment", 0.0))
    query_proto_alignment_warmup_epochs = int(prompt_graph_cfg.get("query_proto_alignment_warmup_epochs", 0))
    query_proto_margin = float(prompt_graph_cfg.get("query_proto_margin", 0.005))
    query_proto_class_balanced = bool(prompt_graph_cfg.get("query_proto_class_balanced", True))
    lambda_edge_utility_supervision = float(prompt_graph_cfg.get("lambda_edge_utility_supervision", 0.0))
    edge_utility_margin = float(prompt_graph_cfg.get("edge_utility_margin", 0.0))
    edge_utility_warmup_epochs = int(prompt_graph_cfg.get("edge_utility_warmup_epochs", 0))
    lambda_correction_alignment = float(prompt_graph_cfg.get("lambda_correction_alignment", 0.0))
    lambda_correction_anti_harm = float(prompt_graph_cfg.get("lambda_correction_anti_harm", 0.0))
    correction_alignment_margin = float(prompt_graph_cfg.get("correction_alignment_margin", 0.0))
    correction_alignment_warmup_epochs = int(prompt_graph_cfg.get("correction_alignment_warmup_epochs", 0))
    lambda_p23_norm = float(prompt_graph_cfg.get("lambda_p23_norm", prompt_graph_cfg.get("norm_weight", 0.0)))
    lambda_p23_hub_budget = float(
        prompt_graph_cfg.get("lambda_p23_hub_budget", prompt_graph_cfg.get("hub_budget_weight", 0.0))
    )
    lambda_p23_node_gate_utility = float(prompt_graph_cfg.get("lambda_p23_node_gate_utility", 0.0))
    p23_node_gate_positive_margin = float(prompt_graph_cfg.get("p23_node_gate_positive_margin", 0.001))
    p23_node_gate_negative_margin = float(prompt_graph_cfg.get("p23_node_gate_negative_margin", -0.001))
    p23_node_gate_utility_warmup_epochs = int(prompt_graph_cfg.get("p23_node_gate_utility_warmup_epochs", 0))
    p23_node_gate_utility_class_balanced = bool(prompt_graph_cfg.get("p23_node_gate_utility_class_balanced", True))
    lambda_prompt_adapter_update_norm = float(training_cfg.get("lambda_prompt_adapter_update_norm", 0.0))
    lambda_prompt_adapter_gate_budget = float(training_cfg.get("lambda_prompt_adapter_gate_budget", 0.0))
    lambda_prompt_adapter_message_help = float(training_cfg.get("lambda_prompt_adapter_message_help", 0.0))
    prompt_adapter_message_help_margin = float(training_cfg.get("prompt_adapter_message_help_margin", 0.0))
    prompt_adapter_message_help_anti_harm_weight = float(
        training_cfg.get("prompt_adapter_message_help_anti_harm_weight", 0.0)
    )
    prompt_adapter_message_help_anti_harm_margin = float(
        training_cfg.get("prompt_adapter_message_help_anti_harm_margin", 0.0)
    )
    prompt_adapter_message_help_class_balanced = bool(
        training_cfg.get("prompt_adapter_message_help_class_balanced", True)
    )
    lambda_prompt_adapter_utility_gate = float(training_cfg.get("lambda_prompt_adapter_utility_gate", 0.0))
    prompt_adapter_utility_gate_temperature = float(training_cfg.get("prompt_adapter_utility_gate_temperature", 0.02))
    prompt_adapter_utility_gate_margin = float(training_cfg.get("prompt_adapter_utility_gate_margin", 0.0))
    prompt_adapter_utility_gate_class_balanced = bool(
        training_cfg.get("prompt_adapter_utility_gate_class_balanced", True)
    )
    prompt_adapter_utility_gate_source = str(training_cfg.get("prompt_adapter_utility_gate_source", "raw_gate"))
    prompt_adapter_episode_count = max(1, int(training_cfg.get("prompt_adapter_episode_count_per_epoch", 1)))
    lambda_prompt_adapter_gate_consistency = float(training_cfg.get("lambda_prompt_adapter_gate_consistency", 0.0))
    lambda_prompt_adapter_delta_consistency = float(training_cfg.get("lambda_prompt_adapter_delta_consistency", 0.0))
    lambda_prompt_router_pattern_balance = float(training_cfg.get("lambda_prompt_router_pattern_balance", 0.0))
    prompt_router_pattern_balance_entropy_floor = float(
        training_cfg.get("prompt_router_pattern_balance_entropy_floor", 0.5)
    )
    lambda_prompt_router_pattern_supervision = float(
        training_cfg.get("lambda_prompt_router_pattern_supervision", 0.0)
    )
    lambda_prompt_router_pattern_utility = float(
        training_cfg.get("lambda_prompt_router_pattern_utility", 0.0)
    )
    lambda_prompt_router_class_pattern_reliability = float(
        training_cfg.get("lambda_prompt_router_class_pattern_reliability", 0.0)
    )
    lambda_p22_pattern_only = float(training_cfg.get("lambda_p22_pattern_only", 0.0))
    lambda_p22_pattern_reg = float(training_cfg.get("lambda_p22_pattern_reg", 0.0))
    lambda_p22_basis_teacher = float(training_cfg.get("lambda_p22_basis_teacher", 0.0))
    lambda_p22_basis_usage = float(training_cfg.get("lambda_p22_basis_usage", 0.0))
    lambda_p22_deployment = float(training_cfg.get("lambda_p22_deployment", 0.0))
    lambda_p22_gate = float(training_cfg.get("lambda_p22_gate", 0.0))
    lambda_p22_anti_harm = float(training_cfg.get("lambda_p22_anti_harm", 0.0))
    lambda_p22_gain_reward = float(training_cfg.get("lambda_p22_gain_reward", 0.0))
    lambda_p22_gate_budget = float(training_cfg.get("lambda_p22_gate_budget", 0.0))
    lambda_p22_gate_harm = float(training_cfg.get("lambda_p22_gate_harm", 0.0))
    lambda_p22_scale_reg = float(training_cfg.get("lambda_p22_scale_reg", 0.0))
    p22_stage1_epochs = int(training_cfg.get("p22_stage1_epochs", 0))
    p22_stage1_pattern_only = bool(training_cfg.get("p22_stage1_pattern_only", False))
    p22_support_source = str(training_cfg.get("p22_support_source", "episode"))
    p22_loss_source = str(training_cfg.get("p22_loss_source", "episode"))
    p22_crossfit_enabled = bool(training_cfg.get("p22_crossfit_enabled", False))
    p22_crossfit_num_folds = int(training_cfg.get("p22_crossfit_num_folds", 5))
    p22_crossfit_resample_each_epoch = bool(training_cfg.get("p22_crossfit_resample_each_epoch", True))
    p22_freeze_pattern_after_epoch = int(training_cfg.get("p22_freeze_pattern_after_epoch", 0))
    p22_gate_margin = float(training_cfg.get("p22_gate_margin", 0.0005))
    p22_gate_target_mode = str(training_cfg.get("p22_gate_target_mode", "binary"))
    p22_gate_positive_margin = float(training_cfg.get("p22_gate_positive_margin", 0.010))
    p22_gate_negative_margin = float(training_cfg.get("p22_gate_negative_margin", -0.005))
    p22_gate_ignore_neutral = bool(training_cfg.get("p22_gate_ignore_neutral", True))
    p22_gate_target_source = str(training_cfg.get("p22_gate_target_source", "ungated_delta"))
    if p22_gate_target_source != "ungated_delta":
        raise ValueError(f"Unsupported p22_gate_target_source={p22_gate_target_source!r}")
    p22_gate_use_crossfit_stability = bool(training_cfg.get("p22_gate_use_crossfit_stability", False))
    p22_gate_helpful_stability_threshold = float(training_cfg.get("p22_gate_helpful_stability_threshold", 0.70))
    p22_gate_harmful_stability_threshold = float(training_cfg.get("p22_gate_harmful_stability_threshold", 0.50))
    p22_gate_stability_min_seen = int(training_cfg.get("p22_gate_stability_min_seen", 2))
    p22_anti_harm_margin = float(training_cfg.get("p22_anti_harm_margin", 0.0))
    p22_gate_budget_max = float(training_cfg.get("p22_gate_budget_max", 0.35))
    p22_gate_budget_warmup_epochs = int(training_cfg.get("p22_gate_budget_warmup_epochs", 30))
    p22_gate_harm_negative_margin = float(training_cfg.get("p22_gate_harm_negative_margin", -0.005))
    p22_safe_checkpoint_enabled = bool(training_cfg.get("p22_safe_checkpoint_enabled", False))
    p22_safe_checkpoint_metric = str(training_cfg.get("p22_safe_checkpoint_metric", "val_acc_plus_val_delta_ce"))
    p22_safe_checkpoint_min_val_delta_ce = float(training_cfg.get("p22_safe_checkpoint_min_val_delta_ce", -0.0005))
    p22_safe_checkpoint_delta_weight = float(training_cfg.get("p22_safe_checkpoint_delta_weight", 0.5))
    p22_safe_checkpoint_delta_cap = float(training_cfg.get("p22_safe_checkpoint_delta_cap", 0.01))
    p22_gain_reward_cap = float(training_cfg.get("p22_gain_reward_cap", 0.02))
    p22_basis_teacher_temperature = float(prompt_adapter_cfg.get("basis_teacher_temperature", 0.5))
    p22_basis_teacher_scale = float(prompt_adapter_cfg.get("basis_teacher_scale", 1.0))
    lambda_prompt_router_deployment_utility = float(
        training_cfg.get("lambda_prompt_router_deployment_utility", 0.0)
    )
    lambda_p21_channel_utility = float(training_cfg.get("lambda_p21_channel_utility", 0.0))
    lambda_p21_gate_utility = float(training_cfg.get("lambda_p21_gate_utility", 0.0))
    expert_warmup_epochs = int(training_cfg.get("expert_warmup_epochs", 0))
    lambda_p21_channel_expert_utility = float(training_cfg.get("lambda_p21_channel_expert_utility", 0.0))
    lambda_p21_channel_expert_utility_after_warmup = float(
        training_cfg.get("lambda_p21_channel_expert_utility_after_warmup", min(lambda_p21_channel_expert_utility, 0.05))
    )
    p21_channel_expert_probe_scale = float(training_cfg.get("p21_channel_expert_probe_scale", 1.0))
    p21_channel_expert_temperature = float(training_cfg.get("p21_channel_expert_temperature", 0.10))
    p21_channel_expert_margin = float(training_cfg.get("p21_channel_expert_margin", 0.001))
    p21_channel_expert_anti_harm_weight = float(training_cfg.get("p21_channel_expert_anti_harm_weight", 0.5))
    p21_channel_expert_class_balanced = bool(training_cfg.get("p21_channel_expert_class_balanced", True))
    lambda_prompt_router_expert_utility_supervision = float(
        training_cfg.get("lambda_prompt_router_expert_utility_supervision", 0.0)
    )
    prompt_router_pattern_supervision_temperature = float(
        training_cfg.get("prompt_router_pattern_supervision_temperature", 0.05)
    )
    prompt_router_pattern_utility_temperature = float(
        training_cfg.get("prompt_router_pattern_utility_temperature", prompt_router_pattern_supervision_temperature)
    )
    prompt_router_pattern_supervision_probe_norm = float(
        training_cfg.get(
            "prompt_router_pattern_supervision_probe_norm",
            prompt_adapter_cfg.get("max_update_norm", 0.08),
        )
    )
    prompt_router_pattern_utility_probe_norm = float(
        training_cfg.get("prompt_router_pattern_utility_probe_norm", prompt_router_pattern_supervision_probe_norm)
    )
    prompt_router_pattern_utility_margin = float(training_cfg.get("prompt_router_pattern_utility_margin", 0.0))
    prompt_router_pattern_utility_anti_harm_weight = float(
        training_cfg.get("prompt_router_pattern_utility_anti_harm_weight", 0.0)
    )
    prompt_router_pattern_utility_min_teacher_delta = float(
        training_cfg.get("prompt_router_pattern_utility_min_teacher_delta", 0.0)
    )
    prompt_router_pattern_utility_helpful_fraction = float(
        training_cfg.get("prompt_router_pattern_utility_helpful_fraction", 1.0)
    )
    prompt_router_pattern_utility_unhelpful_node_weight = float(
        training_cfg.get("prompt_router_pattern_utility_unhelpful_node_weight", 0.0)
    )
    p21_utility_cfg = _resolve_p21_utility_cfg(prompt_adapter_cfg, training_cfg)
    p21_channel_utility_temperature = float(p21_utility_cfg["p21_channel_utility_temperature"])
    p21_channel_utility_margin = float(p21_utility_cfg["p21_channel_utility_margin"])
    p21_channel_utility_min_teacher_delta = float(p21_utility_cfg["p21_channel_utility_min_teacher_delta"])
    p21_channel_utility_target_mode = str(p21_utility_cfg["p21_channel_utility_target_mode"])
    p21_channel_utility_gate_source = str(p21_utility_cfg["p21_channel_utility_gate_source"])
    p21_channel_utility_gate_source_warmup = str(
        p21_utility_cfg["p21_channel_utility_gate_source_warmup"]
    )
    p21_channel_utility_actual_gate_start_epoch = int(p21_utility_cfg["p21_channel_utility_actual_gate_start_epoch"])
    p21_channel_utility_class_balanced = bool(
        training_cfg.get("p21_channel_utility_class_balanced", True)
    )
    p21_gate_utility_temperature = float(p21_utility_cfg["p21_gate_utility_temperature"])
    p21_gate_utility_margin = float(p21_utility_cfg["p21_gate_utility_margin"])
    p21_gate_target_mode = str(p21_utility_cfg["p21_gate_target_mode"])
    prompt_router_class_pattern_reliability_temperature = float(
        training_cfg.get(
            "prompt_router_class_pattern_reliability_temperature",
            prompt_router_pattern_utility_temperature,
        )
    )
    prompt_router_class_pattern_reliability_probe_norm = float(
        training_cfg.get(
            "prompt_router_class_pattern_reliability_probe_norm",
            prompt_router_pattern_utility_probe_norm,
        )
    )
    prompt_router_class_pattern_reliability_positive_margin = float(
        training_cfg.get("prompt_router_class_pattern_reliability_positive_margin", 1e-5)
    )
    prompt_router_class_pattern_reliability_harmful_margin = float(
        training_cfg.get("prompt_router_class_pattern_reliability_harmful_margin", 0.0)
    )
    prompt_router_class_pattern_reliability_min_class_count = int(
        training_cfg.get("prompt_router_class_pattern_reliability_min_class_count", 2)
    )
    prompt_router_expert_utility_temperature = float(
        training_cfg.get("prompt_router_expert_utility_temperature", prompt_router_pattern_utility_temperature)
    )
    prompt_router_expert_utility_margin = float(
        training_cfg.get("prompt_router_expert_utility_margin", prompt_router_pattern_utility_min_teacher_delta)
    )
    prompt_router_expert_utility_target = str(training_cfg.get("prompt_router_expert_utility_target", "soft"))
    prompt_router_expert_utility_gain_temperature_raw = training_cfg.get(
        "prompt_router_expert_utility_gain_temperature"
    )
    prompt_router_expert_utility_gain_temperature = (
        None
        if prompt_router_expert_utility_gain_temperature_raw is None
        else float(prompt_router_expert_utility_gain_temperature_raw)
    )
    prompt_router_expert_utility_probe_norm = training_cfg.get("prompt_router_expert_utility_probe_norm")
    prompt_router_expert_utility_probe_norm = (
        None if prompt_router_expert_utility_probe_norm is None else float(prompt_router_expert_utility_probe_norm)
    )
    prompt_router_expert_utility_class_balanced = bool(
        training_cfg.get("prompt_router_expert_utility_class_balanced", True)
    )
    prompt_router_expert_utility_gate_weight = float(
        training_cfg.get("prompt_router_expert_utility_gate_weight", 0.0)
    )
    prompt_router_expert_utility_gate_target = str(
        training_cfg.get("prompt_router_expert_utility_gate_target", "binary")
    )
    prompt_router_expert_utility_gate_temperature_raw = training_cfg.get(
        "prompt_router_expert_utility_gate_temperature"
    )
    prompt_router_expert_utility_gate_temperature = (
        None
        if prompt_router_expert_utility_gate_temperature_raw is None
        else float(prompt_router_expert_utility_gate_temperature_raw)
    )
    prompt_router_deployment_utility_margin = float(
        training_cfg.get("prompt_router_deployment_utility_margin", prompt_router_expert_utility_margin)
    )
    prompt_router_deployment_utility_anti_harm_weight = float(
        training_cfg.get("prompt_router_deployment_utility_anti_harm_weight", 1.0)
    )
    prompt_router_deployment_utility_anti_harm_margin = float(
        training_cfg.get("prompt_router_deployment_utility_anti_harm_margin", 0.0)
    )
    prompt_router_deployment_utility_gain_reward_weight = float(
        training_cfg.get("prompt_router_deployment_utility_gain_reward_weight", 0.0)
    )
    prompt_router_deployment_utility_gain_reward_cap_raw = training_cfg.get(
        "prompt_router_deployment_utility_gain_reward_cap"
    )
    prompt_router_deployment_utility_gain_reward_cap = (
        None
        if prompt_router_deployment_utility_gain_reward_cap_raw is None
        else float(prompt_router_deployment_utility_gain_reward_cap_raw)
    )
    prompt_router_deployment_utility_class_balanced = bool(
        training_cfg.get("prompt_router_deployment_utility_class_balanced", True)
    )
    prompt_router_pattern_supervision_class_balanced = bool(
        training_cfg.get("prompt_router_pattern_supervision_class_balanced", True)
    )
    prompt_router_pattern_utility_class_balanced = bool(
        training_cfg.get("prompt_router_pattern_utility_class_balanced", True)
    )
    prompt_adapter_update_mask_strategy = str(training_cfg.get("prompt_adapter_update_mask", "all"))
    prompt_adapter_loss_mask_strategy = str(training_cfg.get("prompt_adapter_loss_mask", "query"))
    prompt_adapter_gate_budget = float(prompt_adapter_cfg.get("gate_budget", 0.35))
    support_query_enabled = bool(
        prompt_graph_cfg.get("support_query_split", {}).get(
            "enabled", prompt_graph_cfg.get("support_query_split_enabled", False)
        )
    )
    support_only_prompt_graph = bool(prompt_graph_cfg.get("support_only_prompt_graph", False))
    p24_schedule_cfg = dict(prompt_graph_cfg.get("p24_training_schedule", {}))
    p24_route_start_epoch = int(
        p24_schedule_cfg.get(
            "route_start_epoch",
            p24_counterfactual_cfg.get("route_start_epoch", 0),
        )
    )
    p24_reset_early_stop = bool(
        p24_schedule_cfg.get("reset_early_stop_at_route_start", False)
    )
    p24_min_route_training_epochs = int(
        p24_schedule_cfg.get("min_route_training_epochs", 0)
    )
    p24_candidate_lr_scale_after_warmup = float(
        p24_schedule_cfg.get("candidate_lr_scale_after_warmup", 1.0)
    )
    if p24_candidate_lr_scale_after_warmup < 0.0:
        raise ValueError("candidate_lr_scale_after_warmup must be non-negative")
    p24_early_stop_min_epoch = (
        max(min_epochs, p24_route_start_epoch + p24_min_route_training_epochs)
        if p24_counterfactual_enabled
        else min_epochs
    )
    edge_scale_warmup_epochs = int(prompt_graph_cfg.get("edge_scale_warmup_epochs", 0))
    edge_scale_warmup_start = float(prompt_graph_cfg.get("edge_scale_warmup_start", 1.0 if edge_scale_warmup_epochs <= 0 else 0.0))
    best_checkpoint_path = run_dir / "best_model.pt"
    identity_checkpoint_path = run_dir / "identity_model.pt"
    candidate_checkpoint_path = run_dir / "best_candidate_model.pt"
    deployed_checkpoint_path = run_dir / "best_deployed_model.pt"
    safe_checkpoint_path = run_dir / "safe_model.pt"
    best_val = -1.0
    best_metrics: dict[str, Any] = {}
    best_by_val_acc: dict[str, Any] = {}
    best_by_train_loss: dict[str, Any] = {}
    oracle_best_by_test_acc: dict[str, Any] = {}
    identity_metrics: dict[str, Any] = {}
    best_candidate_val = -1.0
    best_candidate_metrics: dict[str, Any] = {}
    best_deployed_val = -1.0
    best_deployed_metrics: dict[str, Any] = {}
    best_safe_score = float("-inf")
    safe_metrics: dict[str, Any] = {}
    p22_stability_seen = graph.x.new_zeros(graph.num_nodes, dtype=torch.float32)
    p22_stability_helpful = graph.x.new_zeros(graph.num_nodes, dtype=torch.float32)
    p22_stability_harmful = graph.x.new_zeros(graph.num_nodes, dtype=torch.float32)
    loss_curve: list[dict[str, Any]] = []
    prompt_curve: list[dict[str, Any]] = []
    epochs_no_improve = 0
    stopped_epoch = epochs
    early_stopped = False

    if "trial_index" in experiment_cfg:
        run_label = f"{loaded.name} split_seed={split_seed} trial={trial_index} {variant}"
    else:
        run_label = f"{loaded.name} seed={seed} {variant}"
    if total_runs > 1:
        run_label = f"{run_label} ({run_index + 1}/{total_runs})"
    progress = tqdm(range(1, epochs + 1), desc=run_label, unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        if (
            p24_counterfactual_enabled
            and p24_reset_early_stop
            and epoch == p24_route_start_epoch + 1
        ):
            epochs_no_improve = 0
        model.train()
        input_aligner.train()
        if prompt_graph_module is not None:
            prompt_graph_module.train()
        if prompt_adapter_module is not None:
            prompt_adapter_module.train()
            if hasattr(prompt_adapter_module, "set_epoch"):
                prompt_adapter_module.set_epoch(epoch)
            if (
                variant in {
                    "p22_reliability_calibrated_basis_bank",
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                }
                and isinstance(prompt_adapter_module, P22ClassPatternEnrichmentBank)
                and p22_freeze_pattern_after_epoch > 0
            ):
                freeze_pattern = epoch > p22_freeze_pattern_after_epoch
                prompt_adapter_module.pattern_tokens.requires_grad_(not freeze_pattern)
                prompt_adapter_module.pattern_basis_logits.requires_grad_(not freeze_pattern)
                for param in prompt_adapter_module.pattern_encoder.parameters():
                    param.requires_grad_(not freeze_pattern)
                prompt_adapter_module.raw_pattern_scale.requires_grad_(True)
                if getattr(prompt_adapter_module, "reliability_gate", None) is not None:
                    for param in prompt_adapter_module.reliability_gate.parameters():
                        param.requires_grad_(True)
        optimizer.zero_grad()

        support_mask, query_mask, support_query_stats = _support_query_masks_for_epoch(
            graph.y,
            label_train_mask,
            prompt_graph_cfg,
            seed=seed,
            epoch=epoch,
        )
        prompt_supervision_mask = support_mask if support_query_enabled else label_train_mask
        prompt_query_mask = query_mask if support_query_enabled else label_train_mask
        if p24_candidate_loss_mask == "support":
            p24_candidate_supervision_mask = prompt_supervision_mask
        elif p24_candidate_loss_mask == "all_train":
            p24_candidate_supervision_mask = label_train_mask
        else:
            p24_candidate_supervision_mask = prompt_query_mask
        prompt_graph_train_mask = (
            prompt_supervision_mask if support_query_enabled and support_only_prompt_graph else label_train_mask
        )
        z = input_aligner(graph.x)
        current_edge_scale_multiplier = _edge_scale_multiplier(epoch, prompt_graph_cfg)
        adapter_candidate_pool_mask: torch.Tensor | None = None
        adapter_candidate_pool_stats: dict[str, Any] = {}
        if prompt_adapter_module is not None and bool(prompt_adapter_cfg.get("use_candidate_pool", False)):
            with torch.no_grad():
                h_pre_for_pool = model.encode_frozen(z, graph.edge_index)
                no_prompt_for_pool = _forward_no_prompt_with_h_pre(
                    model=model,
                    z=z,
                    edge_index=graph.edge_index,
                    h_pre=h_pre_for_pool,
                )
                adapter_candidate_pool_mask, adapter_candidate_pool_stats = _adapter_candidate_pool(
                    z=z,
                    edge_index=graph.edge_index,
                    train_mask=label_train_mask,
                    prompt_adapter_cfg=prompt_adapter_cfg,
                    no_prompt_logits=no_prompt_for_pool["logits"],
                    h_pre=h_pre_for_pool,
                    h_adp_no_prompt=no_prompt_for_pool["h_adp"],
                )
        adapter_out: dict[str, torch.Tensor] | None = None
        adapter_train_stats: dict[str, Any] = {}
        prompt_adapter_update_norm = z.new_tensor(0.0)
        prompt_adapter_budget = z.new_tensor(0.0)
        prompt_adapter_message_help = z.new_tensor(0.0)
        prompt_adapter_message_help_stats = {
            "prompt_adapter_message_help_loss": 0.0,
            "prompt_adapter_message_help_mean_delta_ce": 0.0,
            "prompt_adapter_message_help_positive_ratio": 0.0,
            "prompt_adapter_message_help_count": 0.0,
            "prompt_adapter_message_help_anti_harm_loss": 0.0,
        }
        prompt_adapter_utility_gate = z.new_tensor(0.0)
        prompt_adapter_utility_gate_stats = {
            "prompt_adapter_utility_gate_loss": 0.0,
            "prompt_adapter_utility_gate_target_mean": 0.0,
            "prompt_adapter_utility_gate_count": 0.0,
            "prompt_adapter_utility_gate_delta_mean": 0.0,
            "prompt_adapter_utility_gate_positive_ratio": 0.0,
        }
        prompt_adapter_gate_consistency = z.new_tensor(0.0)
        prompt_adapter_delta_consistency = z.new_tensor(0.0)
        prompt_adapter_pattern_balance = z.new_tensor(0.0)
        prompt_adapter_pattern_supervision = z.new_tensor(0.0)
        prompt_adapter_pattern_utility = z.new_tensor(0.0)
        prompt_adapter_class_pattern_reliability = z.new_tensor(0.0)
        p22_pattern_only = z.new_tensor(0.0)
        p22_pattern_reg = z.new_tensor(0.0)
        p22_basis_teacher = z.new_tensor(0.0)
        p22_basis_usage = z.new_tensor(0.0)
        p22_deployment = z.new_tensor(0.0)
        p22_gate = z.new_tensor(0.0)
        p22_anti_harm = z.new_tensor(0.0)
        p22_gain_reward = z.new_tensor(0.0)
        p22_gate_budget = z.new_tensor(0.0)
        p22_gate_harm = z.new_tensor(0.0)
        p22_scale_reg = z.new_tensor(0.0)
        p22_basis_teacher_stats = {
            "p22_basis_teacher_loss": 0.0,
            "p22_basis_teacher_count": 0.0,
            "p22_basis_teacher_entropy": 0.0,
            "p22_basis_teacher_student_kl": 0.0,
            "p22_basis_teacher_agreement": 0.0,
        }
        p22_pattern_only_stats = {
            "p22_pattern_only_acc": 0.0,
            "p22_pattern_only_macro_f1": 0.0,
        }
        p22_deployment_stats = {
            "p22_deployment_loss": 0.0,
            "p22_gate_bce_loss": 0.0,
            "p22_anti_harm_loss": 0.0,
            "p22_gain_reward_loss": 0.0,
            "p22_train_delta_ce": 0.0,
            "p22_train_positive_delta_ratio": 0.0,
            "p22_harmful_delta_ratio": 0.0,
            "p22_large_harm_ratio": 0.0,
            "p22_gate_target_mean": 0.0,
            "p22_gate_target_positive_ratio": 0.0,
            "p22_gate_target_negative_ratio": 0.0,
            "p22_gate_target_ignore_ratio": 0.0,
            "p22_gate_target_valid_ratio": 0.0,
            "p22_gate_target_margin_pos": 0.0,
            "p22_gate_target_margin_neg": 0.0,
            "p22_gate_budget_loss": 0.0,
            "p22_gate_harm_loss": 0.0,
            "p22_gate_stability_enabled": 0.0,
            "p22_gate_stability_seen_mean": 0.0,
            "p22_gate_stability_enough_ratio": 0.0,
            "p22_gate_prompt_helpful_mean": 0.0,
            "p22_gate_prompt_harmful_mean": 0.0,
            "p22_gate_base_correct_mean": 0.0,
            "p22_gate_base_wrong_mean": 0.0,
            "p22_crossfit_delta_ce": 0.0,
            "p22_crossfit_positive_delta_ratio": 0.0,
            "p22_crossfit_harmful_ratio": 0.0,
        }
        prompt_adapter_deployment_utility = z.new_tensor(0.0)
        prompt_adapter_expert_utility_supervision = z.new_tensor(0.0)
        prompt_adapter_pattern_supervision_stats = {
            "prompt_router_pattern_supervision_loss": 0.0,
            "prompt_router_pattern_supervision_target_entropy": 0.0,
            "prompt_router_pattern_routing_agreement": 0.0,
            "prompt_router_pattern_supervision_count": 0.0,
        }
        prompt_adapter_pattern_utility_stats = {
            "prompt_router_pattern_utility_loss": 0.0,
            "prompt_router_pattern_utility_mean_delta_ce": 0.0,
            "prompt_router_pattern_utility_positive_ratio": 0.0,
            "prompt_router_pattern_utility_harmful_ratio": 0.0,
            "prompt_router_pattern_utility_helpful_node_ratio": 0.0,
            "prompt_router_pattern_utility_count": 0.0,
            "prompt_router_pattern_utility_target_entropy": 0.0,
        }
        prompt_adapter_class_pattern_reliability_stats = {
            "prompt_router_class_pattern_reliability_loss": 0.0,
            "prompt_router_class_pattern_reliability_count": 0.0,
            "prompt_router_class_pattern_reliable_pair_ratio": 0.0,
            "prompt_router_class_pattern_harmful_pair_ratio": 0.0,
            "prompt_router_class_pattern_target_entropy": 0.0,
            "prompt_router_class_pattern_nonreject_target_mass": 0.0,
        }
        prompt_adapter_deployment_utility_stats = {
            "prompt_router_deployment_loss": 0.0,
            "prompt_router_deployment_count": 0.0,
            "prompt_router_deployment_mean_delta_ce": 0.0,
            "prompt_router_deployment_positive_delta_ratio": 0.0,
            "prompt_router_deployment_harmful_delta_ratio": 0.0,
            "prompt_router_deployment_margin_satisfied_ratio": 0.0,
            "prompt_router_deployment_prompt_ce": 0.0,
            "prompt_router_deployment_no_prompt_ce": 0.0,
            "prompt_router_deployment_anti_harm_loss": 0.0,
            "prompt_router_deployment_gain_reward": 0.0,
        }
        p21_channel_utility = z.new_tensor(0.0)
        p21_gate_utility = z.new_tensor(0.0)
        p21_channel_expert_utility = z.new_tensor(0.0)
        p21_channel_utility_stats = {
            "p21_channel_utility_loss": 0.0,
            "p21_channel_utility_gate_loss": 0.0,
            "p21_channel_utility_count": 0.0,
            "p21_channel_utility_mean_oracle_delta_ce": 0.0,
            "p21_channel_utility_best_channel_delta_ce": 0.0,
            "p21_channel_utility_positive_oracle_ratio": 0.0,
            "p21_channel_utility_best_channel_positive_ratio": 0.0,
            "p21_channel_utility_routed_delta_ce": 0.0,
            "p21_channel_utility_routed_positive_ratio": 0.0,
            "p21_channel_utility_routing_agreement": 0.0,
            "p21_channel_utility_router_agreement_to_oracle": 0.0,
            "p21_channel_utility_teacher_entropy": 0.0,
            "p21_channel_utility_gate_target_mean": 0.0,
            "p21_channel_utility_gate_target_std": 0.0,
            "p21_channel_utility_gate_mean": 0.0,
            "p21_channel_utility_gate_accuracy_to_oracle": 0.0,
            "p21_channel_utility_best_channel_acc": 0.0,
            "p21_channel_utility_best_channel_macro_f1": 0.0,
            "p21_channel_utility_best_channel_acc_lift_vs_no_prompt": 0.0,
            "p21_channel_utility_best_channel_macro_f1_lift_vs_no_prompt": 0.0,
        }
        for channel_name in P21_V2_CHANNEL_NAMES:
            p21_channel_utility_stats[f"p21_channel_utility_{channel_name}_mean_delta_ce"] = 0.0
            p21_channel_utility_stats[f"p21_channel_utility_{channel_name}_best_ratio"] = 0.0
            p21_channel_utility_stats[f"p21_channel_utility_{channel_name}_alpha_mean"] = 0.0
            p21_channel_utility_stats[f"p21_{channel_name}_mean_delta_ce"] = 0.0
            p21_channel_utility_stats[f"p21_{channel_name}_best_ratio"] = 0.0
            p21_channel_utility_stats[f"p21_{channel_name}_alpha_mean"] = 0.0
            p21_channel_utility_stats[f"p21_v2_{channel_name}_mean_delta_ce"] = 0.0
            p21_channel_utility_stats[f"p21_v2_{channel_name}_best_ratio"] = 0.0
            p21_channel_utility_stats[f"p21_v2_{channel_name}_alpha_mean"] = 0.0
        p21_channel_expert_stats = {
            "p21_channel_expert_loss": 0.0,
            "p21_channel_expert_count": 0.0,
            "p21_channel_expert_mean_delta_ce": 0.0,
            "p21_channel_expert_positive_ratio": 0.0,
            "p21_channel_expert_best_delta_ce": 0.0,
            "p21_channel_expert_anti_harm_loss": 0.0,
        }
        for channel_name in P21_V2_CHANNEL_NAMES[1:]:
            p21_channel_expert_stats[f"p21_channel_expert_best_channel_ratio_{channel_name}"] = 0.0
            p21_channel_expert_stats[f"p21_channel_expert_{channel_name}_mean_delta_ce"] = 0.0
        prompt_adapter_expert_utility_stats = {
            "prompt_router_expert_loss": 0.0,
            "prompt_router_expert_count": 0.0,
            "prompt_router_expert_oracle_best_expert_gain": 0.0,
            "prompt_router_expert_oracle_positive_ratio": 0.0,
            "prompt_router_expert_no_prompt_acc": 0.0,
            "prompt_router_expert_no_prompt_macro_f1": 0.0,
            "prompt_router_expert_oracle_best_expert_acc": 0.0,
            "prompt_router_expert_oracle_best_expert_macro_f1": 0.0,
            "prompt_router_expert_oracle_best_expert_acc_lift_vs_no_prompt": 0.0,
            "prompt_router_expert_oracle_best_expert_macro_f1_lift_vs_no_prompt": 0.0,
            "prompt_router_expert_router_accuracy_to_best_expert": 0.0,
            "prompt_router_expert_router_soft_target_kl": 0.0,
            "prompt_router_expert_target_entropy": 0.0,
            "prompt_router_expert_no_correction_ratio": 0.0,
            "prompt_router_expert_learned_weighted_delta_ce": 0.0,
            "prompt_router_expert_learned_best_weight_mean": 0.0,
            "prompt_router_expert_router_loss": 0.0,
            "prompt_router_expert_gate_supervision_loss": 0.0,
            "prompt_router_expert_gate_target_mean": 0.0,
            "prompt_router_expert_gate_mean": 0.0,
            "prompt_router_expert_gate_accuracy_to_oracle": 0.0,
        }
        prompt_adapter_consistency_stats = {
            "prompt_adapter_gate_consistency_loss": 0.0,
            "prompt_adapter_delta_consistency_loss": 0.0,
            "prompt_adapter_episode_count": 1.0,
        }
        effective_lambda_p21_channel_expert = 0.0
        effective_lambda_p21_channel_utility = 0.0
        effective_lambda_p21_gate_utility = 0.0
        effective_lambda_deployment_utility = 0.0
        in_expert_warmup = False
        effective_p21_gate_source = p21_channel_utility_gate_source
        no_prompt_out: dict[str, Any] | None = None
        if prompt_adapter_module is not None:
            prompt_out = _prompt_out_with_pool(
                z,
                graph.edge_index,
                adapter_candidate_pool_mask,
                adapter_candidate_pool_stats,
            )
            episode_count = prompt_adapter_episode_count if support_query_enabled else 1
            cls_losses: list[torch.Tensor] = []
            update_losses: list[torch.Tensor] = []
            budget_losses: list[torch.Tensor] = []
            message_help_losses: list[torch.Tensor] = []
            utility_gate_losses: list[torch.Tensor] = []
            pattern_balance_losses: list[torch.Tensor] = []
            pattern_supervision_losses: list[torch.Tensor] = []
            pattern_utility_losses: list[torch.Tensor] = []
            class_pattern_reliability_losses: list[torch.Tensor] = []
            deployment_utility_losses: list[torch.Tensor] = []
            p22_pattern_only_losses: list[torch.Tensor] = []
            p22_pattern_reg_losses: list[torch.Tensor] = []
            p22_basis_teacher_losses: list[torch.Tensor] = []
            p22_basis_usage_losses: list[torch.Tensor] = []
            p22_deployment_losses: list[torch.Tensor] = []
            p22_gate_losses: list[torch.Tensor] = []
            p22_anti_harm_losses: list[torch.Tensor] = []
            p22_gain_reward_losses: list[torch.Tensor] = []
            p22_gate_budget_losses: list[torch.Tensor] = []
            p22_gate_harm_losses: list[torch.Tensor] = []
            p21_channel_expert_losses: list[torch.Tensor] = []
            p21_channel_utility_losses: list[torch.Tensor] = []
            p21_gate_utility_losses: list[torch.Tensor] = []
            expert_utility_losses: list[torch.Tensor] = []
            pattern_supervision_stats_list: list[dict[str, Any]] = []
            pattern_utility_stats_list: list[dict[str, Any]] = []
            class_pattern_reliability_stats_list: list[dict[str, Any]] = []
            deployment_utility_stats_list: list[dict[str, Any]] = []
            p21_channel_expert_stats_list: list[dict[str, Any]] = []
            p21_channel_utility_stats_list: list[dict[str, Any]] = []
            expert_utility_stats_list: list[dict[str, Any]] = []
            adapter_outs: list[dict[str, torch.Tensor]] = []
            adapter_stats_list: list[dict[str, Any]] = []
            message_help_stats_list: list[dict[str, Any]] = []
            utility_gate_stats_list: list[dict[str, Any]] = []
            support_query_stats_list: list[dict[str, Any]] = []
            p22_basis_teacher_stats_list: list[dict[str, Any]] = []
            p22_pattern_only_stats_list: list[dict[str, Any]] = []
            p22_deployment_stats_list: list[dict[str, Any]] = []
            in_expert_warmup = epoch <= expert_warmup_epochs
            effective_lambda_p21_channel_expert = (
                lambda_p21_channel_expert_utility
                if in_expert_warmup
                else lambda_p21_channel_expert_utility_after_warmup
            )
            effective_lambda_p21_channel_utility = 0.0 if in_expert_warmup else lambda_p21_channel_utility
            effective_lambda_p21_gate_utility = 0.0 if in_expert_warmup else lambda_p21_gate_utility
            effective_lambda_deployment_utility = (
                0.0 if in_expert_warmup else lambda_prompt_router_deployment_utility
            )
            effective_p21_gate_source = p21_channel_utility_gate_source
            if p21_channel_utility_actual_gate_start_epoch > 0 and epoch < p21_channel_utility_actual_gate_start_epoch:
                effective_p21_gate_source = p21_channel_utility_gate_source_warmup
            p22_is_reliability = variant in {
                "p22_reliability_calibrated_basis_bank",
                "p22_v031_conservative_reliability_basis_bank",
                "p22_v04_minimal_transition_basis",
            }
            effective_episode_count = (
                max(2, int(p22_crossfit_num_folds))
                if p22_is_reliability and p22_crossfit_enabled
                else episode_count
            )
            for episode_idx in range(effective_episode_count):
                if p22_is_reliability and p22_crossfit_enabled:
                    episode_support_mask, episode_query_mask = _p22_crossfit_masks(
                        labels=graph.y,
                        train_mask=label_train_mask,
                        num_folds=p22_crossfit_num_folds,
                        fold_idx=episode_idx,
                        seed=seed,
                        epoch=epoch,
                        resample_each_epoch=p22_crossfit_resample_each_epoch,
                    )
                    episode_support_query_stats = {
                        "support_count": float(episode_support_mask.sum().item()),
                        "query_count": float(episode_query_mask.sum().item()),
                        "support_query_enabled": 1.0,
                        "p22_crossfit_enabled": 1.0,
                    }
                elif episode_idx == 0:
                    episode_support_mask = support_mask
                    episode_query_mask = prompt_query_mask
                    episode_support_query_stats = support_query_stats
                else:
                    episode_support_mask, episode_query_mask, episode_support_query_stats = _support_query_masks_for_epoch(
                        graph.y,
                        label_train_mask,
                        prompt_graph_cfg,
                        seed=seed,
                        epoch=epoch * 1009 + episode_idx,
                    )
                if variant in {
                    "p22_class_pattern_enrichment_bank",
                    "p22_reliability_calibrated_basis_bank",
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                }:
                    if p22_is_reliability and p22_crossfit_enabled:
                        pass
                    elif p22_support_source == "full_train":
                        episode_support_mask = label_train_mask.bool()
                    elif p22_support_source not in {"episode", "support"}:
                        raise ValueError(f"Unsupported p22_support_source={p22_support_source!r}")
                episode_loss_query_mask = episode_query_mask if support_query_enabled else label_train_mask
                if p22_is_reliability and p22_crossfit_enabled:
                    episode_loss_query_mask = episode_query_mask
                if variant in {
                    "p22_class_pattern_enrichment_bank",
                    "p22_reliability_calibrated_basis_bank",
                    "p22_v031_conservative_reliability_basis_bank",
                    "p22_v04_minimal_transition_basis",
                }:
                    if p22_is_reliability and p22_crossfit_enabled:
                        pass
                    elif p22_loss_source == "train":
                        episode_loss_query_mask = label_train_mask.bool()
                    elif p22_loss_source not in {"episode", "query"}:
                        raise ValueError(f"Unsupported p22_loss_source={p22_loss_source!r}")
                episode_update_mask = _adapter_mask(
                    prompt_adapter_update_mask_strategy,
                    train_mask=label_train_mask,
                    support_mask=episode_support_mask,
                    query_mask=episode_loss_query_mask,
                    candidate_mask=adapter_candidate_pool_mask,
                )
                episode_loss_mask = _adapter_mask(
                    prompt_adapter_loss_mask_strategy,
                    train_mask=label_train_mask,
                    support_mask=episode_support_mask,
                    query_mask=episode_loss_query_mask,
                    candidate_mask=adapter_candidate_pool_mask,
                )
                if int(episode_loss_mask.sum().item()) == 0:
                    episode_loss_mask = label_train_mask.bool()
                episode_adapter_supervision_mask = episode_loss_mask & episode_update_mask
                if int(episode_adapter_supervision_mask.sum().item()) == 0:
                    episode_adapter_supervision_mask = episode_loss_mask
                episode_model_out, episode_adapter_out, episode_no_prompt_out = _forward_prompt_adapter(
                    model=model,
                    prompt_adapter_module=prompt_adapter_module,
                    z=z,
                    edge_index=graph.edge_index,
                    update_mask=episode_update_mask,
                    support_mask=episode_support_mask,
                    compat_support_mask=label_train_mask,
                    labels=graph.y,
                )
                cls_losses.append(
                    F.cross_entropy(
                        episode_model_out["logits"][episode_loss_mask],
                        graph.y[episode_loss_mask],
                    )
                )
                if "pattern_evidence" in episode_adapter_out:
                    p22_logits = episode_adapter_out["pattern_evidence"]
                    p22_pattern_only_losses.append(F.cross_entropy(p22_logits[episode_loss_mask], graph.y[episode_loss_mask]))
                    p22_reg_value = episode_adapter_out.get("pattern_reg")
                    if isinstance(p22_reg_value, torch.Tensor):
                        p22_pattern_reg_losses.append(p22_reg_value)
                    else:
                        p22_pattern_reg_losses.append(z.new_tensor(0.0))
                    p22_usage_value = episode_adapter_out.get("basis_usage_loss")
                    if isinstance(p22_usage_value, torch.Tensor):
                        p22_basis_usage_losses.append(p22_usage_value)
                    else:
                        p22_basis_usage_losses.append(z.new_tensor(0.0))
                    if bool(prompt_adapter_cfg.get("use_basis_teacher", True)):
                        episode_p22_basis_teacher, episode_p22_basis_teacher_stats = _p22_basis_teacher_loss(
                            adapter_out=episode_adapter_out,
                            base_logits=episode_no_prompt_out["logits"],
                            labels=graph.y,
                            mask=episode_loss_mask,
                            temperature=p22_basis_teacher_temperature,
                            scale=p22_basis_teacher_scale,
                        )
                    else:
                        episode_p22_basis_teacher = z.new_tensor(0.0)
                        episode_p22_basis_teacher_stats = dict(p22_basis_teacher_stats)
                    p22_basis_teacher_losses.append(episode_p22_basis_teacher)
                    p22_stability_helpful_rate = None
                    p22_stability_harmful_rate = None
                    if p22_is_reliability and p22_gate_use_crossfit_stability:
                        stability_idx, stability_delta = _p22_ungated_candidate_delta(
                            adapter_out=episode_adapter_out,
                            base_logits=episode_no_prompt_out["logits"],
                            labels=graph.y,
                            mask=episode_loss_mask,
                        )
                        if stability_idx.numel() > 0:
                            stability_idx = stability_idx.to(device=p22_stability_seen.device)
                            stability_delta = stability_delta.to(device=p22_stability_seen.device)
                            p22_stability_seen[stability_idx] += 1.0
                            p22_stability_helpful[stability_idx] += (
                                stability_delta > p22_gate_positive_margin
                            ).to(dtype=p22_stability_helpful.dtype)
                            p22_stability_harmful[stability_idx] += (
                                stability_delta < p22_gate_negative_margin
                            ).to(dtype=p22_stability_harmful.dtype)
                        p22_stability_helpful_rate = p22_stability_helpful / p22_stability_seen.clamp_min(1.0)
                        p22_stability_harmful_rate = p22_stability_harmful / p22_stability_seen.clamp_min(1.0)
                    episode_p22_deploy_losses, episode_p22_deploy_stats = _p22_deployment_losses(
                        adapter_out=episode_adapter_out,
                        base_logits=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_loss_mask,
                        gate_margin=p22_gate_margin,
                        gate_target_mode=p22_gate_target_mode,
                        gate_positive_margin=p22_gate_positive_margin,
                        gate_negative_margin=p22_gate_negative_margin,
                        gate_ignore_neutral=p22_gate_ignore_neutral,
                        anti_harm_margin=p22_anti_harm_margin,
                        gain_cap=p22_gain_reward_cap,
                        gate_budget_max=p22_gate_budget_max,
                        gate_budget_warmup_epochs=p22_gate_budget_warmup_epochs,
                        epoch=epoch,
                        gate_harm_negative_margin=p22_gate_harm_negative_margin,
                        gate_use_crossfit_stability=p22_is_reliability and p22_gate_use_crossfit_stability,
                        gate_stability_helpful_rate=p22_stability_helpful_rate,
                        gate_stability_harmful_rate=p22_stability_harmful_rate,
                        gate_stability_seen=p22_stability_seen,
                        gate_helpful_stability_threshold=p22_gate_helpful_stability_threshold,
                        gate_harmful_stability_threshold=p22_gate_harmful_stability_threshold,
                        gate_stability_min_seen=p22_gate_stability_min_seen,
                    )
                    p22_deployment_losses.append(episode_p22_deploy_losses["deployment"])
                    p22_gate_losses.append(episode_p22_deploy_losses["gate"])
                    p22_anti_harm_losses.append(episode_p22_deploy_losses["anti_harm"])
                    p22_gain_reward_losses.append(episode_p22_deploy_losses["gain_reward"])
                    p22_gate_budget_losses.append(episode_p22_deploy_losses["gate_budget"])
                    p22_gate_harm_losses.append(episode_p22_deploy_losses["gate_harm"])
                    p22_pattern_only_stats_list.append(
                        _p22_pattern_only_metrics(
                            adapter_out=episode_adapter_out,
                            labels=graph.y,
                            mask=episode_loss_mask,
                            num_classes=loaded.num_classes,
                        )
                    )
                    p22_basis_teacher_stats_list.append(episode_p22_basis_teacher_stats)
                    p22_deployment_stats_list.append(episode_p22_deploy_stats)
                else:
                    p22_pattern_only_losses.append(z.new_tensor(0.0))
                    p22_pattern_reg_losses.append(z.new_tensor(0.0))
                    p22_basis_teacher_losses.append(z.new_tensor(0.0))
                    p22_basis_usage_losses.append(z.new_tensor(0.0))
                    p22_deployment_losses.append(z.new_tensor(0.0))
                    p22_gate_losses.append(z.new_tensor(0.0))
                    p22_anti_harm_losses.append(z.new_tensor(0.0))
                    p22_gain_reward_losses.append(z.new_tensor(0.0))
                    p22_gate_budget_losses.append(z.new_tensor(0.0))
                    p22_gate_harm_losses.append(z.new_tensor(0.0))
                    p22_basis_teacher_stats_list.append(dict(p22_basis_teacher_stats))
                    p22_pattern_only_stats_list.append(dict(p22_pattern_only_stats))
                    p22_deployment_stats_list.append(dict(p22_deployment_stats))
                update_losses.append(prompt_adapter_update_norm_loss(episode_adapter_out, episode_update_mask))
                budget_losses.append(
                    prompt_adapter_gate_budget_loss(
                        episode_adapter_out,
                        max_gate=prompt_adapter_gate_budget,
                        mask=episode_update_mask,
                    )
                )
                if lambda_prompt_adapter_message_help > 0.0:
                    episode_message_help, episode_message_help_stats = prompt_adapter_message_help_loss(
                        logits_prompt=episode_model_out["logits"],
                        logits_no_prompt=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_adapter_supervision_mask,
                        margin=prompt_adapter_message_help_margin,
                        anti_harm_weight=prompt_adapter_message_help_anti_harm_weight,
                        anti_harm_margin=prompt_adapter_message_help_anti_harm_margin,
                        class_balanced=prompt_adapter_message_help_class_balanced,
                    )
                else:
                    episode_message_help = z.new_tensor(0.0)
                    episode_message_help_stats = dict(prompt_adapter_message_help_stats)
                if lambda_prompt_adapter_utility_gate > 0.0:
                    episode_utility_gate, episode_utility_gate_stats = prompt_adapter_utility_gate_loss(
                        adapter_out=episode_adapter_out,
                        logits_prompt=episode_model_out["logits"],
                        logits_no_prompt=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_adapter_supervision_mask,
                        temperature=prompt_adapter_utility_gate_temperature,
                        margin=prompt_adapter_utility_gate_margin,
                        class_balanced=prompt_adapter_utility_gate_class_balanced,
                        gate_source=prompt_adapter_utility_gate_source,
                    )
                else:
                    episode_utility_gate = z.new_tensor(0.0)
                    episode_utility_gate_stats = dict(prompt_adapter_utility_gate_stats)
                if lambda_prompt_router_pattern_balance > 0.0:
                    episode_pattern_balance = prompt_router_pattern_balance_loss(
                        episode_adapter_out,
                        episode_loss_mask,
                        entropy_floor=prompt_router_pattern_balance_entropy_floor,
                    )
                else:
                    episode_pattern_balance = z.new_tensor(0.0)
                if lambda_prompt_router_pattern_supervision > 0.0 and "pattern_messages" in episode_adapter_out:
                    episode_pattern_supervision, episode_pattern_supervision_stats = (
                        prompt_router_pattern_supervision_loss(
                            adapter_out=episode_adapter_out,
                            model=model,
                            h_pre=episode_no_prompt_out["h_pre"],
                            h_adp_base=episode_no_prompt_out["h_adp"],
                            no_prompt_logits=episode_no_prompt_out["logits"],
                            labels=graph.y,
                            mask=episode_adapter_supervision_mask,
                            temperature=prompt_router_pattern_supervision_temperature,
                            probe_norm=prompt_router_pattern_supervision_probe_norm,
                            class_balanced=prompt_router_pattern_supervision_class_balanced,
                        )
                    )
                else:
                    episode_pattern_supervision = z.new_tensor(0.0)
                    episode_pattern_supervision_stats = dict(prompt_adapter_pattern_supervision_stats)
                if lambda_prompt_router_pattern_utility > 0.0 and "pattern_messages" in episode_adapter_out:
                    episode_pattern_utility, episode_pattern_utility_stats = prompt_router_pattern_utility_loss(
                        adapter_out=episode_adapter_out,
                        model=model,
                        h_pre=episode_no_prompt_out["h_pre"],
                        h_adp_base=episode_no_prompt_out["h_adp"],
                        no_prompt_logits=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_adapter_supervision_mask,
                        temperature=prompt_router_pattern_utility_temperature,
                        probe_norm=prompt_router_pattern_utility_probe_norm,
                        margin=prompt_router_pattern_utility_margin,
                        anti_harm_weight=prompt_router_pattern_utility_anti_harm_weight,
                        min_teacher_delta=prompt_router_pattern_utility_min_teacher_delta,
                        helpful_fraction=prompt_router_pattern_utility_helpful_fraction,
                        unhelpful_node_weight=prompt_router_pattern_utility_unhelpful_node_weight,
                        class_balanced=prompt_router_pattern_utility_class_balanced,
                    )
                else:
                    episode_pattern_utility = z.new_tensor(0.0)
                    episode_pattern_utility_stats = dict(prompt_adapter_pattern_utility_stats)
                if (
                    lambda_prompt_router_class_pattern_reliability > 0.0
                    and "pattern_messages" in episode_adapter_out
                ):
                    (
                        episode_class_pattern_reliability,
                        episode_class_pattern_reliability_stats,
                    ) = prompt_router_class_pattern_reliability_loss(
                        adapter_out=episode_adapter_out,
                        model=model,
                        h_pre=episode_no_prompt_out["h_pre"],
                        h_adp_base=episode_no_prompt_out["h_adp"],
                        no_prompt_logits=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_adapter_supervision_mask,
                        temperature=prompt_router_class_pattern_reliability_temperature,
                        probe_norm=prompt_router_class_pattern_reliability_probe_norm,
                        positive_margin=prompt_router_class_pattern_reliability_positive_margin,
                        harmful_margin=prompt_router_class_pattern_reliability_harmful_margin,
                        min_class_count=prompt_router_class_pattern_reliability_min_class_count,
                    )
                else:
                    episode_class_pattern_reliability = z.new_tensor(0.0)
                    episode_class_pattern_reliability_stats = dict(prompt_adapter_class_pattern_reliability_stats)
                if effective_lambda_deployment_utility > 0.0:
                    episode_deployment_utility, episode_deployment_utility_stats = (
                        prompt_router_deployment_utility_loss(
                            logits_prompt=episode_model_out["logits"],
                            logits_no_prompt=episode_no_prompt_out["logits"],
                            labels=graph.y,
                            mask=episode_adapter_supervision_mask,
                            prefix="prompt_router_deployment",
                            margin=prompt_router_deployment_utility_margin,
                            anti_harm_weight=prompt_router_deployment_utility_anti_harm_weight,
                            anti_harm_margin=prompt_router_deployment_utility_anti_harm_margin,
                            gain_reward_weight=prompt_router_deployment_utility_gain_reward_weight,
                            gain_reward_cap=prompt_router_deployment_utility_gain_reward_cap,
                            class_balanced=prompt_router_deployment_utility_class_balanced,
                        )
                    )
                else:
                    episode_deployment_utility = z.new_tensor(0.0)
                    episode_deployment_utility_stats = dict(prompt_adapter_deployment_utility_stats)
                if effective_lambda_p21_channel_expert > 0.0 and "channel_deltas" in episode_adapter_out:
                    episode_p21_channel_expert, episode_p21_channel_expert_stats = p21_channel_expert_utility_loss(
                        adapter_out=episode_adapter_out,
                        model=model,
                        h_pre=episode_no_prompt_out["h_pre"],
                        h_adp_base=episode_no_prompt_out["h_adp"],
                        logits_no_prompt=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_adapter_supervision_mask,
                        probe_scale=p21_channel_expert_probe_scale,
                        temperature=p21_channel_expert_temperature,
                        margin=p21_channel_expert_margin,
                        anti_harm_weight=p21_channel_expert_anti_harm_weight,
                        class_balanced=p21_channel_expert_class_balanced,
                    )
                else:
                    episode_p21_channel_expert = z.new_tensor(0.0)
                    episode_p21_channel_expert_stats = dict(p21_channel_expert_stats)
                if effective_lambda_p21_channel_utility > 0.0 and "channel_deltas" in episode_adapter_out:
                    episode_p21_channel_utility, episode_p21_gate_utility, episode_p21_channel_utility_stats = (
                        p21_channel_utility_supervision_loss(
                            adapter_out=episode_adapter_out,
                            model=model,
                            h_pre=episode_no_prompt_out["h_pre"],
                            h_adp_base=episode_no_prompt_out["h_adp"],
                            logits_no_prompt=episode_no_prompt_out["logits"],
                            labels=graph.y,
                            mask=episode_adapter_supervision_mask,
                            temperature=p21_channel_utility_temperature,
                            margin=p21_channel_utility_margin,
                            min_teacher_delta=p21_channel_utility_min_teacher_delta,
                            target_mode=p21_channel_utility_target_mode,
                            gate_source=effective_p21_gate_source,
                            gate_temperature=p21_gate_utility_temperature,
                            gate_margin=p21_gate_utility_margin,
                            gate_target_mode=p21_gate_target_mode,
                            class_balanced=p21_channel_utility_class_balanced,
                            num_classes=loaded.num_classes,
                        )
                    )
                else:
                    episode_p21_channel_utility = z.new_tensor(0.0)
                    episode_p21_gate_utility = z.new_tensor(0.0)
                    episode_p21_channel_utility_stats = dict(p21_channel_utility_stats)
                if (
                    lambda_prompt_router_expert_utility_supervision > 0.0
                    and "pattern_messages" in episode_adapter_out
                ):
                    episode_expert_utility, episode_expert_utility_stats = (
                        prompt_router_expert_utility_supervision_loss(
                            adapter_out=episode_adapter_out,
                            model=model,
                            h_pre=episode_no_prompt_out["h_pre"],
                            h_adp_base=episode_no_prompt_out["h_adp"],
                            no_prompt_logits=episode_no_prompt_out["logits"],
                            labels=graph.y,
                            mask=episode_adapter_supervision_mask,
                            num_classes=loaded.num_classes,
                            prefix="prompt_router_expert",
                            temperature=prompt_router_expert_utility_temperature,
                            margin=prompt_router_expert_utility_margin,
                            target_mode=prompt_router_expert_utility_target,
                            gain_temperature=prompt_router_expert_utility_gain_temperature,
                            probe_norm=prompt_router_expert_utility_probe_norm,
                            class_balanced=prompt_router_expert_utility_class_balanced,
                            gate_weight=prompt_router_expert_utility_gate_weight,
                            gate_target_mode=prompt_router_expert_utility_gate_target,
                            gate_target_temperature=prompt_router_expert_utility_gate_temperature,
                        )
                    )
                else:
                    episode_expert_utility = z.new_tensor(0.0)
                    episode_expert_utility_stats = dict(prompt_adapter_expert_utility_stats)
                message_help_losses.append(episode_message_help)
                utility_gate_losses.append(episode_utility_gate)
                pattern_balance_losses.append(episode_pattern_balance)
                pattern_supervision_losses.append(episode_pattern_supervision)
                pattern_utility_losses.append(episode_pattern_utility)
                class_pattern_reliability_losses.append(episode_class_pattern_reliability)
                deployment_utility_losses.append(episode_deployment_utility)
                p21_channel_expert_losses.append(episode_p21_channel_expert)
                p21_channel_utility_losses.append(episode_p21_channel_utility)
                p21_gate_utility_losses.append(episode_p21_gate_utility)
                expert_utility_losses.append(episode_expert_utility)
                pattern_supervision_stats_list.append(episode_pattern_supervision_stats)
                pattern_utility_stats_list.append(episode_pattern_utility_stats)
                class_pattern_reliability_stats_list.append(episode_class_pattern_reliability_stats)
                deployment_utility_stats_list.append(episode_deployment_utility_stats)
                p21_channel_expert_stats_list.append(episode_p21_channel_expert_stats)
                p21_channel_utility_stats_list.append(episode_p21_channel_utility_stats)
                expert_utility_stats_list.append(episode_expert_utility_stats)
                adapter_outs.append(episode_adapter_out)
                episode_adapter_stats = _prompt_adapter_diagnostics(episode_adapter_out)
                episode_adapter_stats.update(_prompt_router_diagnostics(episode_adapter_out))
                episode_adapter_stats.update(
                    _prompt_adapter_delta_stats(
                        logits_prompt=episode_model_out["logits"],
                        logits_no_prompt=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        mask=episode_loss_mask,
                        prefix="adapter_query",
                    )
                )
                episode_adapter_stats.update(
                    _prompt_adapter_candidate_pool_delta_stats(
                        logits_prompt=episode_model_out["logits"],
                        logits_no_prompt=episode_no_prompt_out["logits"],
                        labels=graph.y,
                        split_mask=episode_loss_mask,
                        candidate_mask=adapter_candidate_pool_mask,
                        prefix="adapter_query",
                    )
                )
                adapter_stats_list.append(episode_adapter_stats)
                message_help_stats_list.append(episode_message_help_stats)
                utility_gate_stats_list.append(episode_utility_gate_stats)
                support_query_stats_list.append(episode_support_query_stats)
                model_out = episode_model_out
                adapter_out = episode_adapter_out
                no_prompt_out = episode_no_prompt_out

            cls_loss = torch.stack(cls_losses).mean()
            prompt_adapter_update_norm = torch.stack(update_losses).mean()
            prompt_adapter_budget = torch.stack(budget_losses).mean()
            prompt_adapter_message_help = torch.stack(message_help_losses).mean()
            prompt_adapter_utility_gate = torch.stack(utility_gate_losses).mean()
            prompt_adapter_pattern_balance = torch.stack(pattern_balance_losses).mean()
            prompt_adapter_pattern_supervision = torch.stack(pattern_supervision_losses).mean()
            prompt_adapter_pattern_utility = torch.stack(pattern_utility_losses).mean()
            prompt_adapter_class_pattern_reliability = torch.stack(class_pattern_reliability_losses).mean()
            p22_pattern_only = torch.stack(p22_pattern_only_losses).mean()
            p22_pattern_reg = torch.stack(p22_pattern_reg_losses).mean()
            p22_basis_teacher = torch.stack(p22_basis_teacher_losses).mean()
            p22_basis_usage = torch.stack(p22_basis_usage_losses).mean()
            p22_deployment = torch.stack(p22_deployment_losses).mean()
            p22_gate = torch.stack(p22_gate_losses).mean()
            p22_anti_harm = torch.stack(p22_anti_harm_losses).mean()
            p22_gain_reward = torch.stack(p22_gain_reward_losses).mean()
            p22_gate_budget = torch.stack(p22_gate_budget_losses).mean()
            p22_gate_harm = torch.stack(p22_gate_harm_losses).mean()
            if isinstance(prompt_adapter_module, P22ClassPatternEnrichmentBank):
                p22_scale_reg = torch.sigmoid(prompt_adapter_module.raw_pattern_scale).pow(2)
            else:
                p22_scale_reg = z.new_tensor(0.0)
            prompt_adapter_deployment_utility = torch.stack(deployment_utility_losses).mean()
            p21_channel_expert_utility = torch.stack(p21_channel_expert_losses).mean()
            p21_channel_utility = torch.stack(p21_channel_utility_losses).mean()
            p21_gate_utility = torch.stack(p21_gate_utility_losses).mean()
            prompt_adapter_expert_utility_supervision = torch.stack(expert_utility_losses).mean()
            prompt_adapter_pattern_supervision_stats = _mean_float_stats(pattern_supervision_stats_list)
            prompt_adapter_pattern_utility_stats = _mean_float_stats(pattern_utility_stats_list)
            prompt_adapter_class_pattern_reliability_stats = _mean_float_stats(class_pattern_reliability_stats_list)
            p22_basis_teacher_stats = _mean_float_stats(p22_basis_teacher_stats_list)
            p22_pattern_only_stats = _mean_float_stats(p22_pattern_only_stats_list)
            p22_deployment_stats = _mean_float_stats(p22_deployment_stats_list)
            prompt_adapter_deployment_utility_stats = _mean_float_stats(deployment_utility_stats_list)
            p21_channel_expert_stats = _mean_float_stats(p21_channel_expert_stats_list)
            p21_channel_utility_stats = _mean_float_stats(p21_channel_utility_stats_list)
            prompt_adapter_expert_utility_stats = _mean_float_stats(expert_utility_stats_list)
            consistency_mask = label_train_mask.bool()
            prompt_adapter_gate_consistency, prompt_adapter_delta_consistency, prompt_adapter_consistency_stats = (
                _prompt_adapter_episode_consistency_loss(adapter_outs, consistency_mask)
            )
            adapter_train_stats = _mean_float_stats(adapter_stats_list)
            adapter_train_stats.update(adapter_candidate_pool_stats)
            prompt_adapter_message_help_stats = _mean_float_stats(message_help_stats_list)
            prompt_adapter_utility_gate_stats = _mean_float_stats(utility_gate_stats_list)
            support_query_stats = _mean_float_stats(support_query_stats_list)
            support_query_stats.setdefault("enabled", float(support_query_enabled))
            support_query_stats.setdefault("support_count", float(support_mask.sum().item()))
            support_query_stats.setdefault("query_count", float(prompt_query_mask.sum().item()))
            support_query_stats.setdefault(
                "train_support_ratio",
                float(support_mask.float().sum().item() / max(1, int(label_train_mask.bool().sum().item()))),
            )
            adapter_train_stats.update(prompt_adapter_consistency_stats)
            adapter_train_stats.update(prompt_adapter_pattern_supervision_stats)
            adapter_train_stats.update(prompt_adapter_pattern_utility_stats)
            adapter_train_stats.update(prompt_adapter_class_pattern_reliability_stats)
            adapter_train_stats.update(p22_basis_teacher_stats)
            adapter_train_stats.update(p22_pattern_only_stats)
            adapter_train_stats.update(p22_deployment_stats)
            adapter_train_stats.update(prompt_adapter_deployment_utility_stats)
            adapter_train_stats.update(p21_channel_expert_stats)
            adapter_train_stats.update(p21_channel_utility_stats)
            adapter_train_stats.update(prompt_adapter_expert_utility_stats)
            assert adapter_out is not None and no_prompt_out is not None
            adapter_train_stats.update(
                _prompt_adapter_delta_stats(
                    logits_prompt=model_out["logits"],
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=graph.y,
                    mask=label_train_mask,
                    prefix="adapter_train",
                )
            )
            adapter_train_stats.update(
                _prompt_adapter_delta_stats(
                    logits_prompt=model_out["logits"],
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=graph.y,
                    mask=prompt_query_mask,
                    prefix="adapter_query",
                )
            )
            adapter_train_stats.update(_prompt_router_diagnostics(adapter_out))
            adapter_train_stats.update(
                _prompt_router_delta_breakdown(
                    adapter_out=adapter_out,
                    logits_prompt=model_out["logits"],
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=graph.y,
                    mask=prompt_query_mask,
                    prefix="adapter_query",
                )
            )
        else:
            pool_needs_no_prompt = _needs_no_prompt_pool_evidence(prompt_graph_module)
            h_pre_for_prompt: torch.Tensor | None = None
            no_prompt_out = None
            if pool_needs_no_prompt:
                h_pre_for_prompt = model.encode_frozen(z, graph.edge_index)
                with torch.no_grad():
                    no_prompt_out = _forward_no_prompt_with_h_pre(
                        model=model,
                        z=z,
                        edge_index=graph.edge_index,
                        h_pre=h_pre_for_prompt,
                    )
            model_out, prompt_out = _forward_prompt_graph(
                model=model,
                prompt_graph_module=prompt_graph_module,
                z=z,
                x_raw=graph.x,
                edge_index=graph.edge_index,
                train_mask=prompt_graph_train_mask,
                labels=graph.y,
                edge_scale_multiplier=current_edge_scale_multiplier,
                epoch=epoch,
                h_pre=h_pre_for_prompt,
                no_prompt_logits=None if no_prompt_out is None else no_prompt_out["logits"],
                h_adp_no_prompt=None if no_prompt_out is None else no_prompt_out["h_adp"],
            )
            cls_loss = _classification_loss(
                logits=model_out["logits"],
                labels=graph.y,
                train_mask=label_train_mask,
                num_classes=loaded.num_classes,
                training_cfg=training_cfg,
            )
        _attach_p23_ce_delta_diagnostics(
            model_out=model_out,
            prompt_out=prompt_out,
            labels=graph.y,
            train_mask=label_train_mask,
            val_mask=split.val_mask,
            test_mask=split.test_mask,
        )
        gp2f_aux_loss = compute_gp2f_loss(
            logits=model_out["logits"],
            labels=graph.y,
            train_mask=label_train_mask,
            h_pre=model_out["h_pre"],
            h_adp=model_out["h_adp"],
            h_mix=model_out["h_mix"],
            alpha=model_out.get("alpha"),
            edge_index=graph.edge_index,
            cfg=gp2f_aux_loss_cfg,
        )
        edge_l1 = prompt_edge_l1_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        prompt_balance = prompt_balance_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        p23_norm = (
            prompt_out.get("aux", {}).get("p23_norm_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p23_hub_budget = (
            prompt_out.get("aux", {}).get("p23_hub_budget_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p23_prompt_residual_delta_norm = (
            prompt_out.get("aux", {}).get("p23_prompt_residual_delta_norm_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p23_prompt_residual_delta_align = (
            prompt_out.get("aux", {}).get("p23_prompt_residual_delta_align_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p23_prompt_residual_gate_budget = (
            prompt_out.get("aux", {}).get("p23_prompt_residual_gate_budget_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_keep_static = (
            prompt_out.get("aux", {}).get("p24_keep_static_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_hub_suppression = (
            prompt_out.get("aux", {}).get("p24_hub_suppression_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_prompt_homophily = (
            prompt_out.get("aux", {}).get("p24_prompt_homophily_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_edge_budget = (
            prompt_out.get("aux", {}).get("p24_edge_budget_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_receiver_budget = (
            prompt_out.get("aux", {}).get("p24_receiver_budget_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_gumbel_keep_ratio = (
            prompt_out.get("aux", {}).get("p24_gumbel_keep_ratio_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_gumbel_entropy = (
            prompt_out.get("aux", {}).get("p24_gumbel_entropy_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None and epoch > p24_gumbel_entropy_warmup_epochs
            else z.new_tensor(0.0)
        )
        p24_relation_prior = (
            prompt_out.get("aux", {}).get("p24_relation_prior_loss", z.new_tensor(0.0))
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_relation_preservation = (
            prompt_out.get("aux", {}).get(
                "p24_relation_preservation_loss", z.new_tensor(0.0)
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_relation_homophily_collapse = (
            prompt_out.get("aux", {}).get(
                "p24_relation_homophily_collapse_loss", z.new_tensor(0.0)
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_relation_uncertain_keep = (
            prompt_out.get("aux", {}).get(
                "p24_relation_uncertain_keep_loss", z.new_tensor(0.0)
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        p24_candidate_logits = model_out.get("p24_candidate_logits")
        p24_candidate_cls = (
            _classification_loss(
                logits=p24_candidate_logits,
                labels=graph.y,
                train_mask=p24_candidate_supervision_mask,
                num_classes=loaded.num_classes,
                training_cfg=training_cfg,
            )
            if isinstance(p24_candidate_logits, torch.Tensor) and bool(p24_candidate_supervision_mask.any())
            else z.new_tensor(0.0)
        )
        if isinstance(p24_candidate_logits, torch.Tensor) and bool(p24_candidate_supervision_mask.any()):
            p24_candidate_anti_harm, p24_candidate_anti_harm_stats = _p24_candidate_anti_harm_loss(
                model_out=model_out,
                labels=graph.y,
                train_mask=p24_candidate_supervision_mask,
                margin=p24_candidate_anti_harm_margin,
                class_balanced=p24_route_class_balanced,
            )
        else:
            p24_candidate_anti_harm = z.new_tensor(0.0)
            p24_candidate_anti_harm_stats = {
                "p24_candidate_anti_harm_loss": 0.0,
                "p24_candidate_gain_mean": 0.0,
                "p24_candidate_positive_ratio": 0.0,
            }
        if p24_class_coverage_loss_mask == "support":
            p24_class_coverage_mask = prompt_support_mask
        elif p24_class_coverage_loss_mask == "query":
            p24_class_coverage_mask = prompt_query_mask
        else:
            p24_class_coverage_mask = label_train_mask
        if (
            p24_class_coverage_enabled
            and epoch > p24_class_coverage_warmup_epochs
            and isinstance(p24_candidate_logits, torch.Tensor)
            and bool(p24_class_coverage_mask.any())
        ):
            p24_class_coverage_guard, p24_class_coverage_stats = (
                _p24_class_coverage_guard_loss(
                    logits=p24_candidate_logits,
                    labels=graph.y,
                    mask=p24_class_coverage_mask,
                    num_classes=loaded.num_classes,
                    temperature=float(p24_class_coverage_cfg.get("temperature", 1.0)),
                    min_entropy_ratio=float(
                        p24_class_coverage_cfg.get("min_entropy_ratio", 0.75)
                    ),
                    max_dominant_ratio=float(
                        p24_class_coverage_cfg.get("max_dominant_ratio", 0.70)
                    ),
                    class_separation_weight=float(
                        p24_class_coverage_cfg.get("class_separation_weight", 1.0)
                    ),
                    marginal_balance_weight=float(
                        p24_class_coverage_cfg.get("marginal_balance_weight", 0.25)
                    ),
                )
            )
        else:
            p24_class_coverage_guard = z.new_tensor(0.0)
            p24_class_coverage_stats = {
                "p24_class_coverage_guard_loss": 0.0,
                "p24_class_separation_loss": 0.0,
                "p24_class_marginal_balance_loss": 0.0,
                "p24_prediction_coverage_ratio": 0.0,
                "p24_prediction_dominant_ratio": 0.0,
                "p24_prediction_marginal_entropy_ratio": 0.0,
                "p24_collapse_guard_strength": 0.0,
                "p24_class_mean_margin": 0.0,
            }
        if (
            isinstance(prompt_graph_module, PromptingAdaptation)
            and lambda_p24_route_utility > 0.0
            and epoch > p24_route_utility_warmup_epochs
        ):
            if p24_counterfactual_enabled:
                crossfit_learner = prompt_graph_module.crossfit_utility_learner
                crossfit_aux = prompt_out.get("aux", {})
                crossfit_tensors = (
                    model_out.get("p24_static_logits"),
                    model_out.get("p24_candidate_logits"),
                    crossfit_aux.get("p24_counterfactual_predicted_gain"),
                    crossfit_aux.get("p24_route_adapt"),
                )
                if crossfit_learner is not None and all(
                    isinstance(value, torch.Tensor) for value in crossfit_tensors
                ):
                    p24_route_utility, p24_route_utility_stats = crossfit_learner.loss(
                        static_logits=crossfit_tensors[0],
                        candidate_logits=crossfit_tensors[1],
                        predicted_gain=crossfit_tensors[2],
                        adapt_ratio=crossfit_tensors[3],
                        labels=graph.y,
                        train_mask=label_train_mask,
                        query_mask=prompt_query_mask,
                    )
                else:
                    p24_route_utility, p24_route_utility_stats = _p24_counterfactual_route_loss(
                        model_out=model_out,
                        prompt_out=prompt_out,
                        labels=graph.y,
                        query_mask=prompt_query_mask,
                        gain_ema=p24_gain_ema,
                        ema_decay=p24_counterfactual_ema_decay,
                        gain_cap=p24_counterfactual_gain_cap,
                        gain_margin=p24_counterfactual_gain_margin,
                        harm_weight=p24_counterfactual_harm_weight,
                        rank_weight=p24_counterfactual_rank_weight,
                        budget_weight=p24_counterfactual_budget_weight,
                        positive_adapt_floor=p24_counterfactual_positive_adapt_floor,
                        class_balanced=p24_route_class_balanced,
                    )
            else:
                p24_route_utility, p24_route_utility_stats = _p24_route_utility_loss(
                    model_out=model_out,
                    prompt_out=prompt_out,
                    labels=graph.y,
                    train_mask=prompt_query_mask,
                    adapt_positive_margin=p24_route_adapt_positive_margin,
                    adapt_negative_margin=p24_route_adapt_negative_margin,
                    no_prompt_positive_margin=p24_route_no_prompt_positive_margin,
                    no_prompt_negative_margin=p24_route_no_prompt_negative_margin,
                    class_balanced=p24_route_class_balanced,
                    supervise_no_prompt=p24_route_supervise_no_prompt,
                    target_mode=p24_route_target_mode,
                    gain_scale=p24_route_gain_scale,
                )
        else:
            p24_route_utility = z.new_tensor(0.0)
            p24_route_utility_stats = {
                "p24_route_utility_loss": 0.0,
                "p24_route_utility_count": 0.0,
                "p24_route_adapt_positive_ratio": 0.0,
                "p24_route_no_prompt_positive_ratio": 0.0,
                "p24_candidate_gain_mean": 0.0,
                "p24_no_prompt_gain_mean": 0.0,
                "p24_route_target_mean": 0.0,
            }
        if (
            isinstance(prompt_graph_module, PromptingAdaptation)
            and lambda_p24_prompt_utility > 0.0
            and epoch > p24_prompt_utility_warmup_epochs
        ):
            p24_prompt_utility, p24_prompt_utility_stats = _p24_prompt_utility_loss(
                model_out=model_out,
                prompt_out=prompt_out,
                labels=graph.y,
                query_mask=prompt_query_mask,
                shrinkage_tau=p24_prompt_utility_shrinkage_tau,
                gain_scale=p24_prompt_utility_gain_scale,
                target_log_scale=p24_prompt_utility_target_log_scale,
                hub_quantile=p24_prompt_utility_hub_quantile,
            )
        else:
            p24_prompt_utility = z.new_tensor(0.0)
            p24_prompt_utility_stats = {
                "p24_prompt_utility_loss": 0.0,
                "p24_prompt_utility_count": 0.0,
                "p24_prompt_utility_mean": 0.0,
                "p24_prompt_utility_reliability_mean": 0.0,
                "p24_noisy_hub_prompt_ratio": 0.0,
                "p24_noisy_hub_prompt_gate_mean": 1.0,
                "p24_reliable_prompt_gate_mean": 1.0,
            }
        if (
            p24_causal_teacher_enabled
            and lambda_p24_causal_prompt_teacher > 0.0
            and isinstance(prompt_graph_module, PromptingAdaptation)
        ):
            (
                p24_causal_prompt_teacher,
                p24_causal_teacher_state,
                p24_causal_prompt_teacher_stats,
            ) = _p24_causal_prompt_teacher_loss(
                model=model,
                model_out=model_out,
                prompt_out=prompt_out,
                z=z,
                edge_index=graph.edge_index,
                labels=graph.y,
                query_mask=prompt_query_mask,
                state=p24_causal_teacher_state,
                epoch=epoch,
                update_interval=int(p24_causal_teacher_cfg.get("update_interval", 5)),
                warmup_epochs=int(p24_causal_teacher_cfg.get("warmup_epochs", 10)),
                max_prompts_per_update=int(p24_causal_teacher_cfg.get("max_prompts_per_update", 8)),
                candidate_pool_size=int(p24_causal_teacher_cfg.get("candidate_pool_size", 32)),
                probe_strength=float(p24_causal_teacher_cfg.get("probe_strength", 0.50)),
                ema_decay=float(p24_causal_teacher_cfg.get("ema_decay", 0.90)),
                utility_margin=float(p24_causal_teacher_cfg.get("utility_margin", 0.0005)),
                target_temperature=float(p24_causal_teacher_cfg.get("target_temperature", 0.002)),
                max_prune_target=float(p24_causal_teacher_cfg.get("max_prune_target", 0.40)),
                min_observations=int(p24_causal_teacher_cfg.get("min_observations", 2)),
                harmful_stability_threshold=float(
                    p24_causal_teacher_cfg.get("harmful_stability_threshold", 0.70)
                ),
            )
        else:
            p24_causal_prompt_teacher = z.new_tensor(0.0)
            p24_causal_prompt_teacher_stats = {
                "p24_causal_prompt_teacher_loss": 0.0,
                "p24_causal_prompt_teacher_count": 0.0,
                "p24_causal_prompt_teacher_updated": 0.0,
                "p24_causal_prompt_removal_gain_mean": 0.0,
                "p24_causal_prompt_harmful_ratio": 0.0,
                "p24_causal_prompt_harmful_stability_mean": 0.0,
                "p24_causal_prompt_target_keep_mean": 1.0,
                "p24_causal_prompt_harmful_keep_mean": 1.0,
                "p24_causal_prompt_helpful_keep_mean": 1.0,
            }
        if (
            isinstance(prompt_graph_module, P23V01PromptModule)
            and lambda_p23_node_gate_utility > 0.0
            and epoch > p23_node_gate_utility_warmup_epochs
        ):
            p23_node_gate_utility, p23_node_gate_utility_stats = _p23_node_gate_utility_loss(
                model_out=model_out,
                prompt_out=prompt_out,
                labels=graph.y,
                train_mask=label_train_mask,
                positive_margin=p23_node_gate_positive_margin,
                negative_margin=p23_node_gate_negative_margin,
                class_balanced=p23_node_gate_utility_class_balanced,
            )
        else:
            p23_node_gate_utility = z.new_tensor(0.0)
            p23_node_gate_utility_stats = {
                "p23_node_gate_utility_loss": 0.0,
                "p23_node_gate_utility_count": 0.0,
                "p23_node_gate_positive_count": 0.0,
                "p23_node_gate_negative_count": 0.0,
                "p23_node_gate_target_mean": 0.0,
                "p23_node_gate_delta_corr_train": float(prompt_out.get("aux", {}).get("p23_node_gate_delta_corr_train", z.new_tensor(0.0)).detach().item())
                if isinstance(prompt_out.get("aux", {}).get("p23_node_gate_delta_corr_train"), torch.Tensor)
                else 0.0,
            }
        legacy_prompt_graph = isinstance(prompt_graph_module, PromptGraphModuleP1)
        prompt_role_diversity = (
            prompt_role_diversity_loss(prompt_graph_module) if legacy_prompt_graph else z.new_tensor(0.0)
        )
        prompt_acceptance = prompt_acceptance_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        prompt_acceptance_budget = (
            prompt_acceptance_budget_loss(
                prompt_out,
                min_acceptance=acceptance_budget_min,
                max_acceptance=acceptance_budget_max,
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        needs_no_prompt_delta = (
            prompt_graph_module is not None
            and (
                lambda_prompt_acceptance_supervision > 0.0
                or (
                    lambda_prompt_benefit_supervision > 0.0
                    and epoch > benefit_supervision_warmup_epochs
                )
                or (
                    (lambda_prompt_correction > 0.0 or lambda_prompt_anti_harm > 0.0)
                    and epoch > prompt_correction_warmup_epochs
                )
                or (
                    lambda_prompt_message_help > 0.0
                    and epoch > prompt_message_help_warmup_epochs
                )
                or (
                    lambda_prompt_message_help_query > 0.0
                    and epoch > prompt_message_help_query_warmup_epochs
                )
                or (
                    lambda_utility_receive_gate > 0.0
                    and epoch > utility_receive_gate_warmup_epochs
                )
                or (
                    lambda_utility_receive_gate_query > 0.0
                    and epoch > utility_receive_gate_query_warmup_epochs
                )
                or (
                    lambda_query_proto_alignment > 0.0
                    and epoch > query_proto_alignment_warmup_epochs
                )
                or (
                    lambda_edge_utility_supervision > 0.0
                    and epoch > edge_utility_warmup_epochs
                )
                or (
                    (lambda_correction_alignment > 0.0 or lambda_correction_anti_harm > 0.0)
                    and epoch > correction_alignment_warmup_epochs
                )
                or (
                    isinstance(prompt_graph_module, PromptGraphConstruction)
                    and (
                        lambda_p23_feature_harm > 0.0
                        or lambda_p23_feature_gate_supervision > 0.0
                    )
                    and epoch > p23_feature_harm_warmup_epochs
                )
            )
        )
        if needs_no_prompt_delta and no_prompt_out is None:
            with torch.no_grad():
                no_prompt_out = _forward_no_prompt_with_h_pre(
                    model=model,
                    z=z,
                    edge_index=graph.edge_index,
                    h_pre=model_out["h_pre_shared"],
                )
        if prompt_graph_module is not None and lambda_prompt_acceptance_supervision > 0.0 and no_prompt_out is not None:
            prompt_acceptance_supervision, acceptance_supervision_stats = _acceptance_supervision_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=label_train_mask,
                positive_margin=acceptance_supervision_positive_margin,
                negative_margin=acceptance_supervision_negative_margin,
                balance_targets=acceptance_supervision_balance_targets,
                signal=acceptance_supervision_signal,
            )
        else:
            prompt_acceptance_supervision = z.new_tensor(0.0)
            acceptance_supervision_stats = {
                "prompt_acceptance_supervision": 0.0,
                "acceptance_supervised_count": 0.0,
                "acceptance_positive_count": 0.0,
                "acceptance_negative_count": 0.0,
                "acceptance_ignored_count": 0.0,
                "acceptance_target_mean": 0.0,
                "acceptance_score_delta_mean": 0.0,
            }
        if (
            prompt_graph_module is not None
            and lambda_prompt_benefit_supervision > 0.0
            and epoch > benefit_supervision_warmup_epochs
            and no_prompt_out is not None
        ):
            benefit_logits_on = model_out["logits"]
            if (
                benefit_supervision_probe_scale is not None
                and benefit_supervision_probe_scale > 0.0
                and isinstance(model, PromptAwareGP2F)
            ):
                with torch.no_grad():
                    probe_out = _forward_prompt_graph_with_message_scale(
                        model=model,
                        z=z,
                        edge_index=graph.edge_index,
                        h_pre=model_out["h_pre_shared"],
                        prompt_out=prompt_out,
                        message_scale=benefit_supervision_probe_scale,
                    )
                benefit_logits_on = probe_out["logits"]
            benefit_label_strategy_epoch = benefit_supervision_label_strategy
            if (
                benefit_supervision_label_strategy == "hybrid_quantile_margin"
                and epoch <= benefit_supervision_warmup_epochs + benefit_supervision_quantile_warmup_epochs
            ):
                benefit_label_strategy_epoch = "quantile"
            prompt_benefit_supervision, benefit_supervision_stats = _benefit_supervision_loss(
                prompt_out=prompt_out,
                logits_on=benefit_logits_on,
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=label_train_mask,
                margin=benefit_delta_margin,
                balance_targets=benefit_supervision_balance_targets,
                label_strategy=benefit_label_strategy_epoch,
                quantile=benefit_supervision_quantile,
                eps=benefit_delta_eps,
            )
        else:
            prompt_benefit_supervision = z.new_tensor(0.0)
            benefit_supervision_stats = {
                "prompt_benefit_supervision": 0.0,
                "benefit_supervised_count": 0.0,
                "benefit_positive_count": 0.0,
                "benefit_negative_count": 0.0,
                "benefit_ignored_count": 0.0,
                "mean_delta_ce_train_pool": 0.0,
                "positive_delta_ratio_train_pool": 0.0,
                "benefit_weight_delta_corr_train": 0.0,
                "benefit_label_strategy_quantile": 0.0,
                "benefit_quantile_fraction": 0.0,
                "benefit_positive_threshold": 0.0,
                "benefit_negative_threshold": 0.0,
                "benefit_delta_eps": 0.0,
            }
        if (
            prompt_graph_module is not None
            and (lambda_prompt_correction > 0.0 or lambda_prompt_anti_harm > 0.0)
            and epoch > prompt_correction_warmup_epochs
            and no_prompt_out is not None
        ):
            prompt_correction_loss, prompt_anti_harm_loss, prompt_correction_stats = _prompt_correction_losses(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=label_train_mask,
                eps=prompt_correction_eps,
                target=prompt_correction_target,
            )
        else:
            prompt_correction_loss = z.new_tensor(0.0)
            prompt_anti_harm_loss = z.new_tensor(0.0)
            prompt_correction_stats = {
                "prompt_correction_loss": 0.0,
                "prompt_anti_harm_loss": 0.0,
                "prompt_correction_supervised_count": 0.0,
                "prompt_correction_harmful_count": 0.0,
                "prompt_correction_target_norm": 0.0,
                "prompt_delta_logit_norm": 0.0,
                "prompt_delta_logit_label_alignment": 0.0,
                "harmful_prompt_delta_ratio": 0.0,
            }
        if (
            prompt_graph_module is not None
            and lambda_prompt_message_help > 0.0
            and epoch > prompt_message_help_warmup_epochs
            and no_prompt_out is not None
        ):
            prompt_message_help, prompt_class_anti_harm, prompt_message_help_stats = _prompt_message_help_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=label_train_mask,
                margin=prompt_message_help_margin,
                class_balanced=prompt_message_help_class_balanced,
                anti_harm_floor=prompt_class_anti_harm_floor,
            )
        else:
            prompt_message_help = z.new_tensor(0.0)
            prompt_class_anti_harm = z.new_tensor(0.0)
            prompt_message_help_stats = {
                "prompt_message_help_loss": 0.0,
                "class_balanced_mean_delta_ce": 0.0,
                "class_balanced_positive_delta_ratio": 0.0,
                "message_help_supervised_count": 0.0,
                "message_help_class_count": 0.0,
                "message_help_margin": prompt_message_help_margin,
                "message_help_class_balanced": float(prompt_message_help_class_balanced),
                "prompt_class_anti_harm_loss": 0.0,
                "prompt_class_anti_harm_floor": prompt_class_anti_harm_floor,
                "delta_ce_by_class_train_pool": {},
            }
        if (
            prompt_graph_module is not None
            and lambda_prompt_message_help_query > 0.0
            and epoch > prompt_message_help_query_warmup_epochs
            and no_prompt_out is not None
        ):
            (
                prompt_message_help_query,
                _prompt_message_help_query_anti_harm,
                prompt_message_help_query_stats,
            ) = _prompt_message_help_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=prompt_query_mask,
                margin=prompt_message_help_margin,
                class_balanced=prompt_message_help_class_balanced,
                anti_harm_floor=prompt_class_anti_harm_floor,
            )
            prompt_message_help_query_stats = {
                "prompt_message_help_query_loss": prompt_message_help_query_stats["prompt_message_help_loss"],
                "query_class_balanced_mean_delta_ce": prompt_message_help_query_stats[
                    "class_balanced_mean_delta_ce"
                ],
                "query_class_balanced_positive_delta_ratio": prompt_message_help_query_stats[
                    "class_balanced_positive_delta_ratio"
                ],
                "message_help_query_supervised_count": prompt_message_help_query_stats[
                    "message_help_supervised_count"
                ],
                "message_help_query_class_count": prompt_message_help_query_stats["message_help_class_count"],
                "delta_ce_by_class_query_pool": prompt_message_help_query_stats["delta_ce_by_class_train_pool"],
            }
        else:
            prompt_message_help_query = z.new_tensor(0.0)
            prompt_message_help_query_stats = {
                "prompt_message_help_query_loss": 0.0,
                "query_class_balanced_mean_delta_ce": 0.0,
                "query_class_balanced_positive_delta_ratio": 0.0,
                "message_help_query_supervised_count": 0.0,
                "message_help_query_class_count": 0.0,
                "delta_ce_by_class_query_pool": {},
            }
        if (
            prompt_graph_module is not None
            and lambda_utility_receive_gate > 0.0
            and epoch > utility_receive_gate_warmup_epochs
            and no_prompt_out is not None
        ):
            utility_receive_gate_loss, utility_receive_gate_stats = _utility_receive_gate_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=label_train_mask,
                quantile=utility_receive_gate_quantile,
                eps=utility_receive_gate_eps,
                class_balanced=utility_receive_gate_class_balanced,
                label_strategy=utility_receive_gate_label_strategy,
            )
        else:
            utility_receive_gate_loss = z.new_tensor(0.0)
            utility_receive_gate_stats = {
                "utility_receive_gate_loss": 0.0,
                "utility_gate_supervised_count": 0.0,
                "utility_gate_positive_count": 0.0,
                "utility_gate_negative_count": 0.0,
                "utility_gate_ignored_count": 0.0,
                "utility_gate_target_mean": 0.0,
                "utility_gate_delta_corr_train": 0.0,
                "utility_gate_positive_threshold_mean": 0.0,
                "utility_gate_negative_threshold_mean": 0.0,
                "utility_gate_quantile": utility_receive_gate_quantile,
                "utility_gate_eps": utility_receive_gate_eps,
                "utility_gate_class_balanced": float(utility_receive_gate_class_balanced),
                "utility_gate_label_strategy": utility_receive_gate_label_strategy,
            }
        if (
            prompt_graph_module is not None
            and lambda_utility_receive_gate_query > 0.0
            and epoch > utility_receive_gate_query_warmup_epochs
            and no_prompt_out is not None
        ):
            utility_receive_gate_query_loss, utility_receive_gate_query_stats = _utility_receive_gate_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=prompt_query_mask,
                quantile=utility_receive_gate_quantile,
                eps=utility_receive_gate_eps,
                class_balanced=utility_receive_gate_class_balanced,
                label_strategy=utility_receive_gate_label_strategy,
            )
            utility_receive_gate_query_stats = {
                "utility_receive_gate_query_loss": utility_receive_gate_query_stats["utility_receive_gate_loss"],
                "utility_gate_query_supervised_count": utility_receive_gate_query_stats[
                    "utility_gate_supervised_count"
                ],
                "utility_gate_query_positive_count": utility_receive_gate_query_stats[
                    "utility_gate_positive_count"
                ],
                "utility_gate_query_negative_count": utility_receive_gate_query_stats[
                    "utility_gate_negative_count"
                ],
                "utility_gate_query_ignored_count": utility_receive_gate_query_stats["utility_gate_ignored_count"],
                "utility_gate_query_target_mean": utility_receive_gate_query_stats["utility_gate_target_mean"],
                "utility_gate_query_delta_corr": utility_receive_gate_query_stats["utility_gate_delta_corr_train"],
                "utility_gate_query_label_strategy": utility_receive_gate_query_stats["utility_gate_label_strategy"],
            }
        else:
            utility_receive_gate_query_loss = z.new_tensor(0.0)
            utility_receive_gate_query_stats = {
                "utility_receive_gate_query_loss": 0.0,
                "utility_gate_query_supervised_count": 0.0,
                "utility_gate_query_positive_count": 0.0,
                "utility_gate_query_negative_count": 0.0,
                "utility_gate_query_ignored_count": 0.0,
                "utility_gate_query_target_mean": 0.0,
                "utility_gate_query_delta_corr": 0.0,
                "utility_gate_query_label_strategy": utility_receive_gate_label_strategy,
            }
        if (
            prompt_graph_module is not None
            and lambda_query_proto_alignment > 0.0
            and epoch > query_proto_alignment_warmup_epochs
            and no_prompt_out is not None
        ):
            query_proto_alignment, query_proto_alignment_stats = _query_proto_alignment_loss(
                h_on=model_out["h_mix"],
                h_off=no_prompt_out["h_mix"],
                labels=graph.y,
                support_mask=support_mask,
                query_mask=query_mask,
                pool_mask=prompt_out.get("pool_mask", torch.zeros_like(split.train_mask)),
                margin=query_proto_margin,
                class_balanced=query_proto_class_balanced,
            )
        else:
            query_proto_alignment = z.new_tensor(0.0)
            query_proto_alignment_stats = {
                "query_proto_alignment_loss": 0.0,
                "query_proto_supervised_count": 0.0,
                "query_proto_class_count": 0.0,
                "query_proto_mean_delta_dist": 0.0,
                "query_proto_positive_ratio": 0.0,
                "query_proto_margin": query_proto_margin,
                "query_proto_class_balanced": float(query_proto_class_balanced),
                "query_proto_by_class": {},
            }
        if (
            prompt_graph_module is not None
            and lambda_edge_utility_supervision > 0.0
            and epoch > edge_utility_warmup_epochs
            and no_prompt_out is not None
        ):
            edge_utility_supervision, edge_utility_stats = _edge_utility_supervision_loss(
                prompt_out=prompt_out,
                logits_on=model_out["logits"],
                logits_off=no_prompt_out["logits"],
                labels=graph.y,
                train_mask=prompt_supervision_mask,
                margin=edge_utility_margin,
            )
        else:
            edge_utility_supervision = z.new_tensor(0.0)
            edge_utility_stats = {
                "edge_utility_supervision_loss": 0.0,
                "edge_utility_supervised_count": 0.0,
                "edge_utility_positive_count": 0.0,
                "edge_utility_negative_count": 0.0,
                "edge_utility_ignored_count": 0.0,
                "edge_utility_target_mean": 0.0,
                "edge_utility_delta_ce_mean": 0.0,
                "edge_utility_delta_ce_positive_ratio": 0.0,
                "edge_utility_delta_corr_train": 0.0,
            }
        if (
            prompt_graph_module is not None
            and (lambda_correction_alignment > 0.0 or lambda_correction_anti_harm > 0.0)
            and epoch > correction_alignment_warmup_epochs
            and no_prompt_out is not None
        ):
            correction_alignment, correction_alignment_anti_harm, correction_alignment_stats = (
                _correction_alignment_losses(
                    model=model,
                    prompt_out=prompt_out,
                    h_on=model_out["h_adp"],
                    h_off=no_prompt_out["h_adp"],
                    logits_on=model_out["logits"],
                    logits_off=no_prompt_out["logits"],
                    labels=graph.y,
                    train_mask=prompt_supervision_mask,
                    margin=correction_alignment_margin,
                )
            )
        else:
            correction_alignment = z.new_tensor(0.0)
            correction_alignment_anti_harm = z.new_tensor(0.0)
            correction_alignment_stats = {
                "correction_alignment_loss": 0.0,
                "correction_alignment_anti_harm_loss": 0.0,
                "correction_alignment_node_count": 0.0,
                "correction_alignment_harmful_count": 0.0,
                "correction_alignment_cosine_mean": 0.0,
                "correction_alignment_delta_h_norm": 0.0,
                "correction_alignment_delta_ce_mean": 0.0,
            }
        prompt_usage_consistency = (
            prompt_usage_consistency_loss(
                prompt_out,
                graph.y,
                prompt_supervision_mask,
                margin=usage_consistency_margin,
                negative_weight=usage_consistency_negative_weight,
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        prompt_view_entropy = prompt_view_entropy_loss(prompt_out) if prompt_graph_module is not None else z.new_tensor(0.0)
        prompt_view_prior = (
            prompt_view_prior_loss(prompt_out, view_prior)
            if prompt_graph_module is not None and view_prior is not None
            else z.new_tensor(0.0)
        )
        class_route = (
            prompt_class_route_loss(prompt_out, graph.y, prompt_supervision_mask)
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        key_proto = (
            prompt_key_proto_loss(prompt_graph_module, prompt_out, graph.y, prompt_supervision_mask)
            if legacy_prompt_graph
            else z.new_tensor(0.0)
        )
        receive_gate_budget = (
            utility_receive_gate_budget_loss(
                prompt_out,
                min_receive=None if receive_gate_budget_min is None else float(receive_gate_budget_min),
                max_receive=None if receive_gate_budget_max is None else float(receive_gate_budget_max),
            )
            if prompt_graph_module is not None
            else z.new_tensor(0.0)
        )
        if (
            isinstance(prompt_graph_module, PromptGraphConstruction)
            and no_prompt_out is not None
            and epoch > p23_feature_harm_warmup_epochs
            and (lambda_p23_feature_harm > 0.0 or lambda_p23_feature_gate_supervision > 0.0)
        ):
            p23_feature_harm_loss, p23_feature_gate_supervision_loss, p23_feature_harm_stats = (
                _p23_feature_harmful_suppression_loss(
                    prompt_out=prompt_out,
                    logits_prompt=model_out["logits"],
                    logits_no_prompt=no_prompt_out["logits"],
                    labels=graph.y,
                    train_mask=label_train_mask,
                    margin=p23_feature_harm_margin,
                    temperature=p23_feature_harm_temperature,
                    harm_quantile=p23_feature_harm_quantile,
                    help_quantile=p23_feature_help_quantile,
                    min_signal=p23_feature_min_signal,
                    help_weight=p23_feature_help_weight,
                    detach_delta=p23_feature_harm_detach_delta,
                    class_balanced=p23_feature_harm_class_balanced,
                )
            )
        else:
            p23_feature_harm_loss = z.new_tensor(0.0)
            p23_feature_gate_supervision_loss = z.new_tensor(0.0)
            p23_feature_harm_stats = {
                "p23_feature_harm_loss": 0.0,
                "p23_feature_gate_supervision_loss": 0.0,
                "p23_feature_harm_count": 0.0,
                "p23_feature_help_count": 0.0,
                "p23_feature_harm_ratio": 0.0,
                "p23_feature_help_ratio": 0.0,
                "p23_feature_raw_harm_ratio": 0.0,
                "p23_feature_raw_help_ratio": 0.0,
                "p23_feature_gate_target_mean": 0.0,
                "p23_feature_gate_delta_corr": 0.0,
            }
        p22_warmup_loss = (
            lambda_p22_pattern_only * p22_pattern_only
            + lambda_p22_pattern_reg * p22_pattern_reg
            + lambda_p22_basis_teacher * p22_basis_teacher
            + lambda_p22_basis_usage * p22_basis_usage
        )
        p22_deployment_aux_loss = (
            lambda_p22_deployment * p22_deployment
            + lambda_p22_gate * p22_gate
            + lambda_p22_anti_harm * p22_anti_harm
            + lambda_p22_gain_reward * p22_gain_reward
            + lambda_p22_gate_budget * p22_gate_budget
            + lambda_p22_gate_harm * p22_gate_harm
            + lambda_p22_scale_reg * p22_scale_reg
        )
        p22_aux_loss = p22_warmup_loss + p22_deployment_aux_loss
        p23_projector_receiver_base_cls = z.new_tensor(0.0)
        p23_projector_receiver_cls_objective = cls_loss
        if (
            isinstance(prompt_graph_module, P23ProjectorSynchronizedPromptGraphAdapter)
            and prompt_graph_module.projector_receiver_enabled
            and prompt_graph_module.projector_receiver.base_cls_weight > 0.0
        ):
            base_logits_train = model_out.get("p23_no_receiver_logits_train")
            if isinstance(base_logits_train, torch.Tensor):
                p23_projector_receiver_base_cls = _classification_loss(
                    logits=base_logits_train,
                    labels=graph.y,
                    train_mask=label_train_mask,
                    num_classes=loaded.num_classes,
                    training_cfg=training_cfg,
                )
                base_weight = float(prompt_graph_module.projector_receiver.base_cls_weight)
                p23_projector_receiver_cls_objective = (
                    cls_loss + base_weight * p23_projector_receiver_base_cls
                ) / (1.0 + base_weight)
        full_loss = (
            p23_projector_receiver_cls_objective
            + gp2f_aux_loss.total
            + lambda_edge_l1 * edge_l1
            + lambda_prompt_balance * prompt_balance
            + lambda_prompt_role_diversity * prompt_role_diversity
            + lambda_prompt_acceptance * prompt_acceptance
            + lambda_prompt_acceptance_budget * prompt_acceptance_budget
            + lambda_prompt_acceptance_supervision * prompt_acceptance_supervision
            + lambda_prompt_usage_consistency * prompt_usage_consistency
            + lambda_prompt_view_entropy * prompt_view_entropy
            + lambda_view_prior * prompt_view_prior
            + lambda_class_route * class_route
            + lambda_key_proto * key_proto
            + lambda_prompt_benefit_supervision * prompt_benefit_supervision
            + lambda_prompt_correction * prompt_correction_loss
            + lambda_prompt_anti_harm * prompt_anti_harm_loss
            + lambda_prompt_message_help * prompt_message_help
            + lambda_prompt_message_help_query * prompt_message_help_query
            + lambda_prompt_class_anti_harm * prompt_class_anti_harm
            + lambda_utility_receive_gate * utility_receive_gate_loss
            + lambda_utility_receive_gate_query * utility_receive_gate_query_loss
            + lambda_receive_gate_budget * receive_gate_budget
            + lambda_query_proto_alignment * query_proto_alignment
            + lambda_edge_utility_supervision * edge_utility_supervision
            + lambda_correction_alignment * correction_alignment
            + lambda_correction_anti_harm * correction_alignment_anti_harm
            + lambda_p23_norm * p23_norm
            + lambda_p23_hub_budget * p23_hub_budget
            + lambda_p23_node_gate_utility * p23_node_gate_utility
            + lambda_p23_feature_harm * p23_feature_harm_loss
            + lambda_p23_feature_gate_supervision * p23_feature_gate_supervision_loss
            + lambda_p23_prompt_residual_delta_norm * p23_prompt_residual_delta_norm
            + lambda_p23_prompt_residual_delta_align * p23_prompt_residual_delta_align
            + lambda_p23_prompt_residual_gate_budget * p23_prompt_residual_gate_budget
            + lambda_p24_keep_static * p24_keep_static
            + lambda_p24_hub_suppression * p24_hub_suppression
            + lambda_p24_prompt_homophily * p24_prompt_homophily
            + lambda_p24_edge_budget * p24_edge_budget
            + lambda_p24_receiver_budget * p24_receiver_budget
            + lambda_p24_gumbel_keep_ratio * p24_gumbel_keep_ratio
            + lambda_p24_gumbel_entropy * p24_gumbel_entropy
            + lambda_p24_candidate_cls * p24_candidate_cls
            + lambda_p24_candidate_anti_harm * p24_candidate_anti_harm
            + lambda_p24_relation_prior * p24_relation_prior
            + lambda_p24_relation_preservation * p24_relation_preservation
            + lambda_p24_relation_homophily_collapse * p24_relation_homophily_collapse
            + lambda_p24_relation_uncertain_keep * p24_relation_uncertain_keep
            + lambda_p24_route_utility * p24_route_utility
            + lambda_p24_class_coverage_guard * p24_class_coverage_guard
            + lambda_p24_prompt_utility * p24_prompt_utility
            + lambda_p24_causal_prompt_teacher * p24_causal_prompt_teacher
            + lambda_prompt_adapter_update_norm * prompt_adapter_update_norm
            + lambda_prompt_adapter_gate_budget * prompt_adapter_budget
            + lambda_prompt_adapter_message_help * prompt_adapter_message_help
            + lambda_prompt_adapter_utility_gate * prompt_adapter_utility_gate
            + lambda_prompt_adapter_gate_consistency * prompt_adapter_gate_consistency
            + lambda_prompt_adapter_delta_consistency * prompt_adapter_delta_consistency
            + lambda_prompt_router_pattern_balance * prompt_adapter_pattern_balance
            + lambda_prompt_router_pattern_supervision * prompt_adapter_pattern_supervision
            + lambda_prompt_router_pattern_utility * prompt_adapter_pattern_utility
            + lambda_prompt_router_class_pattern_reliability * prompt_adapter_class_pattern_reliability
            + p22_aux_loss
            + effective_lambda_deployment_utility * prompt_adapter_deployment_utility
            + effective_lambda_p21_channel_expert * p21_channel_expert_utility
            + effective_lambda_p21_channel_utility * p21_channel_utility
            + effective_lambda_p21_gate_utility * p21_gate_utility
            + lambda_prompt_router_expert_utility_supervision * prompt_adapter_expert_utility_supervision
        )
        p22_stage1_active = (
            variant in {
                "p22_class_pattern_enrichment_bank",
                "p22_reliability_calibrated_basis_bank",
                "p22_v031_conservative_reliability_basis_bank",
                "p22_v04_minimal_transition_basis",
            }
            and p22_stage1_pattern_only
            and epoch <= p22_stage1_epochs
        )
        loss = p22_warmup_loss if p22_stage1_active else full_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        p22_token_before = (
            prompt_adapter_module.pattern_tokens.detach().clone()
            if isinstance(prompt_adapter_module, P22ClassPatternEnrichmentBank)
            else None
        )
        loss.backward()
        if (
            p24_counterfactual_enabled
            and epoch > p24_route_start_epoch
            and p24_candidate_lr_scale_after_warmup != 1.0
            and isinstance(prompt_graph_module, PromptingAdaptation)
        ):
            candidate_selector = (
                prompt_graph_module.hierarchical_selector
                if prompt_graph_module.hierarchical_selector is not None
                else (
                    prompt_graph_module.gumbel_selector
                    if prompt_graph_module.gumbel_selector is not None
                    else prompt_graph_module.relation_selector
                )
            )
            for parameter in candidate_selector.parameters() if candidate_selector is not None else ():
                if parameter.grad is not None:
                    parameter.grad.mul_(p24_candidate_lr_scale_after_warmup)
        p22_grad_stats = _p22_grad_diagnostics(prompt_adapter_module)
        torch.nn.utils.clip_grad_norm_(trainable_params, float(training_cfg.get("grad_clip", 1.0)))
        optimizer.step()
        p22_grad_stats.update(_p22_grad_diagnostics(prompt_adapter_module, token_before=p22_token_before))

        prompt_log = _prompt_graph_diagnostics(
            prompt_out,
            z=z,
            train_mask=label_train_mask,
        )
        prompt_log.update(p23_node_gate_utility_stats)
        prompt_log.update(p23_feature_harm_stats)
        prompt_log.update(p24_route_utility_stats)
        prompt_log.update(p24_causal_prompt_teacher_stats)
        prompt_log.update(p24_candidate_anti_harm_stats)
        prompt_log.update(p24_prompt_utility_stats)
        prompt_aware_log = _prompt_aware_diagnostics(model_out)
        log_item = {
            "epoch": float(epoch),
            "test_label_cheat_enabled": float(enable_test_label_cheat),
            "test_label_cheat_fraction": float(test_label_cheat_fraction),
            "actual_test_label_cheat_fraction": float(actual_test_label_cheat_fraction),
            "test_label_cheat_count": float(cheat_test_label_count),
            "label_train_count": float(label_train_mask.sum().item()),
            "total": float(loss.detach().item()),
            "full_loss": float(full_loss.detach().item()),
            "p22_aux_loss": float(p22_aux_loss.detach().item()),
            "gp2f_contrastive_loss": float(gp2f_aux_loss.contrastive.detach().item()),
            "gp2f_topology_fusion_loss": float(gp2f_aux_loss.topology_fusion.detach().item()),
            "gp2f_contrastive_mode": gp2f_aux_loss.contrastive_mode,
            "gp2f_topology_mode": gp2f_aux_loss.topology_mode,
            "p22_stage1_active": float(p22_stage1_active),
            "cls": float(cls_loss.detach().item()),
            "p23_projector_receiver_base_cls": float(
                p23_projector_receiver_base_cls.detach().item()
            ),
            "p23_projector_receiver_cls_objective": float(
                p23_projector_receiver_cls_objective.detach().item()
            ),
            "edge_l1": float(edge_l1.detach().item()),
            "prompt_balance": float(prompt_balance.detach().item()),
            "prompt_role_diversity": float(prompt_role_diversity.detach().item()),
            "prompt_acceptance": float(prompt_acceptance.detach().item()),
            "prompt_acceptance_budget": float(prompt_acceptance_budget.detach().item()),
            "prompt_acceptance_supervision": float(prompt_acceptance_supervision.detach().item()),
            "prompt_usage_consistency": float(prompt_usage_consistency.detach().item()),
            "prompt_view_entropy": float(prompt_view_entropy.detach().item()),
            "prompt_view_prior": float(prompt_view_prior.detach().item()),
            "class_route": float(class_route.detach().item()),
            "key_proto": float(key_proto.detach().item()),
            "prompt_benefit_supervision": float(prompt_benefit_supervision.detach().item()),
            "prompt_correction_loss": float(prompt_correction_loss.detach().item()),
            "prompt_anti_harm_loss": float(prompt_anti_harm_loss.detach().item()),
            "prompt_message_help_loss": float(prompt_message_help.detach().item()),
            "prompt_message_help_query_loss": float(prompt_message_help_query.detach().item()),
            "prompt_class_anti_harm_loss": float(prompt_class_anti_harm.detach().item()),
            "utility_receive_gate_loss": float(utility_receive_gate_loss.detach().item()),
            "utility_receive_gate_query_loss": float(utility_receive_gate_query_loss.detach().item()),
            "receive_gate_budget": float(receive_gate_budget.detach().item()),
            "query_proto_alignment_loss": float(query_proto_alignment.detach().item()),
            "edge_utility_supervision_loss": float(edge_utility_supervision.detach().item()),
            "correction_alignment_loss": float(correction_alignment.detach().item()),
            "correction_alignment_anti_harm_loss": float(correction_alignment_anti_harm.detach().item()),
            "p23_norm_loss": float(p23_norm.detach().item()),
            "p23_hub_budget_loss": float(p23_hub_budget.detach().item()),
            "p23_node_gate_utility_loss": float(p23_node_gate_utility.detach().item()),
            "p23_feature_harm_loss": float(p23_feature_harm_loss.detach().item()),
            "p23_feature_gate_supervision_loss": float(p23_feature_gate_supervision_loss.detach().item()),
            "p23_prompt_residual_delta_norm_loss": float(p23_prompt_residual_delta_norm.detach().item()),
            "p23_prompt_residual_delta_align_loss": float(p23_prompt_residual_delta_align.detach().item()),
            "p23_prompt_residual_gate_budget_loss": float(p23_prompt_residual_gate_budget.detach().item()),
            "p24_keep_static_loss": float(p24_keep_static.detach().item()),
            "p24_hub_suppression_loss": float(p24_hub_suppression.detach().item()),
            "p24_prompt_homophily_loss": float(p24_prompt_homophily.detach().item()),
            "p24_edge_budget_loss": float(p24_edge_budget.detach().item()),
            "p24_receiver_budget_loss": float(p24_receiver_budget.detach().item()),
            "p24_candidate_cls_loss": float(p24_candidate_cls.detach().item()),
            "p24_candidate_anti_harm_loss": float(p24_candidate_anti_harm.detach().item()),
            "p24_relation_prior_loss": float(p24_relation_prior.detach().item()),
            "p24_relation_preservation_loss": float(p24_relation_preservation.detach().item()),
            "p24_relation_homophily_collapse_loss": float(
                p24_relation_homophily_collapse.detach().item()
            ),
            "p24_relation_uncertain_keep_loss": float(
                p24_relation_uncertain_keep.detach().item()
            ),
            "p24_route_utility_loss": float(p24_route_utility.detach().item()),
            **p24_class_coverage_stats,
            "p24_prompt_utility_loss": float(p24_prompt_utility.detach().item()),
            "prompt_adapter_update_norm_loss": float(prompt_adapter_update_norm.detach().item()),
            "prompt_adapter_gate_budget_loss": float(prompt_adapter_budget.detach().item()),
            "prompt_adapter_message_help_loss": float(prompt_adapter_message_help.detach().item()),
            "prompt_adapter_utility_gate_loss": float(prompt_adapter_utility_gate.detach().item()),
            "prompt_adapter_gate_consistency_loss": float(prompt_adapter_gate_consistency.detach().item()),
            "prompt_adapter_delta_consistency_loss": float(prompt_adapter_delta_consistency.detach().item()),
            "prompt_adapter_pattern_balance_loss": float(prompt_adapter_pattern_balance.detach().item()),
            "prompt_router_pattern_supervision_loss": float(prompt_adapter_pattern_supervision.detach().item()),
            "prompt_router_pattern_utility_loss": float(prompt_adapter_pattern_utility.detach().item()),
            "prompt_router_class_pattern_reliability_loss": float(
                prompt_adapter_class_pattern_reliability.detach().item()
            ),
            "p22_pattern_only_loss": float(p22_pattern_only.detach().item()),
            "p22_pattern_reg_loss": float(p22_pattern_reg.detach().item()),
            "p22_basis_teacher_loss": float(p22_basis_teacher.detach().item()),
            "p22_basis_usage_loss": float(p22_basis_usage.detach().item()),
            "p22_deployment_loss": float(p22_deployment.detach().item()),
            "p22_gate_bce_loss": float(p22_gate.detach().item()),
            "p22_anti_harm_loss": float(p22_anti_harm.detach().item()),
            "p22_gain_reward_loss": float(p22_gain_reward.detach().item()),
            "p22_gate_budget_loss": float(p22_gate_budget.detach().item()),
            "p22_gate_harm_loss": float(p22_gate_harm.detach().item()),
            "p22_scale_reg_loss": float(p22_scale_reg.detach().item()),
            "prompt_router_deployment_utility_loss": float(prompt_adapter_deployment_utility.detach().item()),
            "p21_channel_expert_utility_loss": float(p21_channel_expert_utility.detach().item()),
            "p21_channel_utility_loss": float(p21_channel_utility.detach().item()),
            "p21_gate_utility_loss": float(p21_gate_utility.detach().item()),
            "prompt_router_expert_utility_supervision_loss": float(
                prompt_adapter_expert_utility_supervision.detach().item()
            ),
            "prompt_adapter_episode_count": float(prompt_adapter_episode_count),
            "lambda_edge_l1": lambda_edge_l1,
            "lambda_prompt_balance": lambda_prompt_balance,
            "lambda_p23_feature_harm": lambda_p23_feature_harm,
            "lambda_p23_feature_gate_supervision": lambda_p23_feature_gate_supervision,
            "p23_feature_harm_margin": p23_feature_harm_margin,
            "p23_feature_harm_temperature": p23_feature_harm_temperature,
            "p23_feature_harm_quantile": p23_feature_harm_quantile,
            "p23_feature_help_quantile": p23_feature_help_quantile,
            "p23_feature_min_signal": p23_feature_min_signal,
            "p23_feature_help_weight": p23_feature_help_weight,
            "p23_feature_harm_warmup_epochs": p23_feature_harm_warmup_epochs,
            "p23_feature_harm_detach_delta": p23_feature_harm_detach_delta,
            "p23_feature_harm_class_balanced": p23_feature_harm_class_balanced,
            "lambda_prompt_role_diversity": lambda_prompt_role_diversity,
            "lambda_prompt_acceptance": lambda_prompt_acceptance,
            "lambda_prompt_acceptance_budget": lambda_prompt_acceptance_budget,
            "lambda_prompt_acceptance_supervision": lambda_prompt_acceptance_supervision,
            "lambda_prompt_usage_consistency": lambda_prompt_usage_consistency,
            "lambda_prompt_view_entropy": lambda_prompt_view_entropy,
            "lambda_view_prior": lambda_view_prior,
            "lambda_class_route": lambda_class_route,
            "lambda_key_proto": lambda_key_proto,
            "lambda_prompt_benefit_supervision": lambda_prompt_benefit_supervision,
            "lambda_prompt_correction": lambda_prompt_correction,
            "lambda_prompt_anti_harm": lambda_prompt_anti_harm,
            "lambda_prompt_message_help": lambda_prompt_message_help,
            "lambda_prompt_message_help_query": lambda_prompt_message_help_query,
            "lambda_prompt_class_anti_harm": lambda_prompt_class_anti_harm,
            "lambda_utility_receive_gate": lambda_utility_receive_gate,
            "lambda_utility_receive_gate_query": lambda_utility_receive_gate_query,
            "lambda_receive_gate_budget": lambda_receive_gate_budget,
            "lambda_query_proto_alignment": lambda_query_proto_alignment,
            "lambda_edge_utility_supervision": lambda_edge_utility_supervision,
            "lambda_correction_alignment": lambda_correction_alignment,
            "lambda_correction_anti_harm": lambda_correction_anti_harm,
            "lambda_p23_norm": lambda_p23_norm,
            "lambda_p23_hub_budget": lambda_p23_hub_budget,
            "lambda_p23_node_gate_utility": lambda_p23_node_gate_utility,
            "lambda_p23_prompt_residual_delta_norm": lambda_p23_prompt_residual_delta_norm,
            "lambda_p23_prompt_residual_delta_align": lambda_p23_prompt_residual_delta_align,
            "lambda_p23_prompt_residual_gate_budget": lambda_p23_prompt_residual_gate_budget,
            "lambda_p24_keep_static": lambda_p24_keep_static,
            "lambda_p24_hub_suppression": lambda_p24_hub_suppression,
            "lambda_p24_prompt_homophily": lambda_p24_prompt_homophily,
            "lambda_p24_edge_budget": lambda_p24_edge_budget,
            "lambda_p24_receiver_budget": lambda_p24_receiver_budget,
            "lambda_p24_gumbel_keep_ratio": lambda_p24_gumbel_keep_ratio,
            "lambda_p24_gumbel_entropy": lambda_p24_gumbel_entropy,
            "p24_gumbel_entropy_warmup_epochs": float(p24_gumbel_entropy_warmup_epochs),
            "lambda_p24_candidate_cls": lambda_p24_candidate_cls,
            "lambda_p24_candidate_anti_harm": lambda_p24_candidate_anti_harm,
            "p24_candidate_anti_harm_margin": p24_candidate_anti_harm_margin,
            "p24_candidate_loss_mask": p24_candidate_loss_mask,
            "lambda_p24_route_utility": lambda_p24_route_utility,
            "p24_route_utility_warmup_epochs": float(p24_route_utility_warmup_epochs),
            "p24_route_positive_margin": p24_route_positive_margin,
            "p24_route_negative_margin": p24_route_negative_margin,
            "p24_route_adapt_positive_margin": p24_route_adapt_positive_margin,
            "p24_route_adapt_negative_margin": p24_route_adapt_negative_margin,
            "p24_route_no_prompt_positive_margin": p24_route_no_prompt_positive_margin,
            "p24_route_no_prompt_negative_margin": p24_route_no_prompt_negative_margin,
            "p24_route_class_balanced": float(p24_route_class_balanced),
            "p24_route_supervise_no_prompt": float(p24_route_supervise_no_prompt),
            "p24_route_target_mode": p24_route_target_mode,
            "p24_route_gain_scale": p24_route_gain_scale,
            "lambda_p24_class_coverage_guard": lambda_p24_class_coverage_guard,
            "p24_class_coverage_guard_enabled": float(p24_class_coverage_enabled),
            "p24_class_coverage_warmup_epochs": float(p24_class_coverage_warmup_epochs),
            "p24_class_coverage_loss_mask": p24_class_coverage_loss_mask,
            "p24_class_coverage_temperature": float(
                p24_class_coverage_cfg.get("temperature", 1.0)
            ),
            "p24_class_coverage_min_entropy_ratio": float(
                p24_class_coverage_cfg.get("min_entropy_ratio", 0.75)
            ),
            "p24_class_coverage_max_dominant_ratio": float(
                p24_class_coverage_cfg.get("max_dominant_ratio", 0.70)
            ),
            "lambda_p24_prompt_utility": lambda_p24_prompt_utility,
            "p24_prompt_utility_warmup_epochs": float(p24_prompt_utility_warmup_epochs),
            "p24_prompt_utility_shrinkage_tau": p24_prompt_utility_shrinkage_tau,
            "p24_prompt_utility_gain_scale": p24_prompt_utility_gain_scale,
            "p24_prompt_utility_target_log_scale": p24_prompt_utility_target_log_scale,
            "p24_prompt_utility_hub_quantile": p24_prompt_utility_hub_quantile,
            "p23_node_gate_utility_warmup_epochs": float(p23_node_gate_utility_warmup_epochs),
            "lambda_p23_feature_harm": lambda_p23_feature_harm,
            "lambda_p23_feature_gate_supervision": lambda_p23_feature_gate_supervision,
            "p23_feature_harm_margin": p23_feature_harm_margin,
            "p23_feature_harm_temperature": p23_feature_harm_temperature,
            "p23_feature_harm_quantile": p23_feature_harm_quantile,
            "p23_feature_help_quantile": p23_feature_help_quantile,
            "p23_feature_min_signal": p23_feature_min_signal,
            "p23_feature_help_weight": p23_feature_help_weight,
            "p23_feature_harm_warmup_epochs": float(p23_feature_harm_warmup_epochs),
            "p23_feature_harm_detach_delta": float(p23_feature_harm_detach_delta),
            "p23_feature_harm_class_balanced": float(p23_feature_harm_class_balanced),
            "lambda_gp2f_contrastive": gp2f_aux_loss_cfg.lambda_ctr,
            "lambda_gp2f_topology_fusion": gp2f_aux_loss_cfg.lambda_fus,
            "lambda_prompt_adapter_update_norm": lambda_prompt_adapter_update_norm,
            "lambda_prompt_adapter_gate_budget": lambda_prompt_adapter_gate_budget,
            "lambda_prompt_adapter_message_help": lambda_prompt_adapter_message_help,
            "lambda_prompt_adapter_utility_gate": lambda_prompt_adapter_utility_gate,
            "lambda_prompt_adapter_gate_consistency": lambda_prompt_adapter_gate_consistency,
            "lambda_prompt_adapter_delta_consistency": lambda_prompt_adapter_delta_consistency,
            "lambda_prompt_router_pattern_balance": lambda_prompt_router_pattern_balance,
            "lambda_prompt_router_pattern_supervision": lambda_prompt_router_pattern_supervision,
            "lambda_prompt_router_pattern_utility": lambda_prompt_router_pattern_utility,
            "lambda_prompt_router_class_pattern_reliability": lambda_prompt_router_class_pattern_reliability,
            "lambda_p22_pattern_only": lambda_p22_pattern_only,
            "lambda_p22_pattern_reg": lambda_p22_pattern_reg,
            "lambda_p22_basis_teacher": lambda_p22_basis_teacher,
            "lambda_p22_basis_usage": lambda_p22_basis_usage,
            "lambda_p22_deployment": lambda_p22_deployment,
            "lambda_p22_gate": lambda_p22_gate,
            "lambda_p22_anti_harm": lambda_p22_anti_harm,
            "lambda_p22_gain_reward": lambda_p22_gain_reward,
            "lambda_p22_gate_budget": lambda_p22_gate_budget,
            "lambda_p22_gate_harm": lambda_p22_gate_harm,
            "lambda_p22_scale_reg": lambda_p22_scale_reg,
            "p22_stage1_epochs": float(p22_stage1_epochs),
            "p22_support_source": p22_support_source,
            "p22_loss_source": p22_loss_source,
            "p22_crossfit_enabled": float(p22_crossfit_enabled),
            "p22_crossfit_num_folds": float(p22_crossfit_num_folds),
            "p22_freeze_pattern_after_epoch": float(p22_freeze_pattern_after_epoch),
            "p22_gate_margin": p22_gate_margin,
            "p22_gate_target_mode": p22_gate_target_mode,
            "p22_gate_target_source": p22_gate_target_source,
            "p22_gate_positive_margin": p22_gate_positive_margin,
            "p22_gate_negative_margin": p22_gate_negative_margin,
            "p22_gate_ignore_neutral": float(p22_gate_ignore_neutral),
            "p22_gate_use_crossfit_stability": float(p22_gate_use_crossfit_stability),
            "p22_gate_helpful_stability_threshold": p22_gate_helpful_stability_threshold,
            "p22_gate_harmful_stability_threshold": p22_gate_harmful_stability_threshold,
            "p22_gate_stability_min_seen": float(p22_gate_stability_min_seen),
            "p22_gate_budget_max": p22_gate_budget_max,
            "p22_gate_budget_warmup_epochs": float(p22_gate_budget_warmup_epochs),
            "p22_gate_harm_negative_margin": p22_gate_harm_negative_margin,
            "p22_safe_checkpoint_enabled": float(p22_safe_checkpoint_enabled),
            "p22_safe_checkpoint_metric": p22_safe_checkpoint_metric,
            "p22_safe_checkpoint_min_val_delta_ce": p22_safe_checkpoint_min_val_delta_ce,
            "p22_safe_checkpoint_delta_weight": p22_safe_checkpoint_delta_weight,
            "p22_safe_checkpoint_delta_cap": p22_safe_checkpoint_delta_cap,
            "p22_anti_harm_margin": p22_anti_harm_margin,
            "p22_gain_reward_cap": p22_gain_reward_cap,
            "lambda_prompt_router_deployment_utility": lambda_prompt_router_deployment_utility,
            "lambda_p21_channel_expert_utility": lambda_p21_channel_expert_utility,
            "lambda_p21_channel_expert_utility_after_warmup": lambda_p21_channel_expert_utility_after_warmup,
            "lambda_p21_channel_utility": lambda_p21_channel_utility,
            "lambda_p21_gate_utility": lambda_p21_gate_utility,
            "effective_lambda_prompt_router_deployment_utility": effective_lambda_deployment_utility,
            "effective_lambda_p21_channel_expert_utility": effective_lambda_p21_channel_expert,
            "effective_lambda_p21_channel_utility": effective_lambda_p21_channel_utility,
            "effective_lambda_p21_gate_utility": effective_lambda_p21_gate_utility,
            "p21_expert_warmup_active": float(in_expert_warmup),
            "p21_expert_warmup_epochs": float(expert_warmup_epochs),
            "lambda_prompt_router_expert_utility_supervision": lambda_prompt_router_expert_utility_supervision,
            "prompt_router_pattern_utility_margin": prompt_router_pattern_utility_margin,
            "prompt_router_pattern_utility_anti_harm_weight": prompt_router_pattern_utility_anti_harm_weight,
            "prompt_router_pattern_utility_min_teacher_delta": prompt_router_pattern_utility_min_teacher_delta,
            "prompt_router_pattern_utility_helpful_fraction": prompt_router_pattern_utility_helpful_fraction,
            "prompt_router_pattern_utility_unhelpful_node_weight": prompt_router_pattern_utility_unhelpful_node_weight,
            "prompt_router_class_pattern_reliability_positive_margin": (
                prompt_router_class_pattern_reliability_positive_margin
            ),
            "prompt_router_class_pattern_reliability_harmful_margin": (
                prompt_router_class_pattern_reliability_harmful_margin
            ),
            "prompt_router_deployment_utility_margin": prompt_router_deployment_utility_margin,
            "prompt_router_deployment_utility_anti_harm_weight": (
                prompt_router_deployment_utility_anti_harm_weight
            ),
            "prompt_router_deployment_utility_anti_harm_margin": (
                prompt_router_deployment_utility_anti_harm_margin
            ),
            "prompt_router_deployment_utility_gain_reward_weight": (
                prompt_router_deployment_utility_gain_reward_weight
            ),
            "prompt_router_deployment_utility_gain_reward_cap": (
                0.0
                if prompt_router_deployment_utility_gain_reward_cap is None
                else prompt_router_deployment_utility_gain_reward_cap
            ),
            "prompt_router_deployment_utility_class_balanced": float(
                prompt_router_deployment_utility_class_balanced
            ),
            "p21_channel_utility_temperature": p21_channel_utility_temperature,
            "p21_channel_utility_margin": p21_channel_utility_margin,
            "p21_channel_utility_min_teacher_delta": p21_channel_utility_min_teacher_delta,
            "p21_channel_utility_target_mode": p21_channel_utility_target_mode,
            "p21_channel_utility_gate_source": p21_channel_utility_gate_source,
            "p21_channel_utility_effective_gate_source": effective_p21_gate_source,
            "p21_channel_utility_gate_source_warmup": p21_channel_utility_gate_source_warmup,
            "p21_channel_utility_actual_gate_start_epoch": float(p21_channel_utility_actual_gate_start_epoch),
            "p21_channel_utility_class_balanced": float(p21_channel_utility_class_balanced),
            "p21_channel_expert_probe_scale": p21_channel_expert_probe_scale,
            "p21_channel_expert_temperature": p21_channel_expert_temperature,
            "p21_channel_expert_margin": p21_channel_expert_margin,
            "p21_channel_expert_anti_harm_weight": p21_channel_expert_anti_harm_weight,
            "p21_gate_utility_temperature": p21_gate_utility_temperature,
            "p21_gate_utility_margin": p21_gate_utility_margin,
            "p21_gate_target_mode": p21_gate_target_mode,
            "prompt_router_expert_utility_temperature": prompt_router_expert_utility_temperature,
            "prompt_router_expert_utility_gain_temperature": (
                0.0
                if prompt_router_expert_utility_gain_temperature is None
                else prompt_router_expert_utility_gain_temperature
            ),
            "prompt_router_expert_utility_margin": prompt_router_expert_utility_margin,
            "prompt_router_expert_utility_target": prompt_router_expert_utility_target,
            "prompt_router_expert_utility_gate_weight": prompt_router_expert_utility_gate_weight,
            "prompt_router_expert_utility_gate_target": prompt_router_expert_utility_gate_target,
            "prompt_router_expert_utility_gate_temperature": (
                0.0
                if prompt_router_expert_utility_gate_temperature is None
                else prompt_router_expert_utility_gate_temperature
            ),
            "prompt_adapter_message_help_margin": prompt_adapter_message_help_margin,
            "prompt_adapter_message_help_anti_harm_weight": prompt_adapter_message_help_anti_harm_weight,
            "prompt_adapter_message_help_anti_harm_margin": prompt_adapter_message_help_anti_harm_margin,
            "prompt_adapter_message_help_class_balanced": float(prompt_adapter_message_help_class_balanced),
            "prompt_adapter_utility_gate_temperature": prompt_adapter_utility_gate_temperature,
            "prompt_adapter_utility_gate_margin": prompt_adapter_utility_gate_margin,
            "prompt_adapter_utility_gate_class_balanced": float(prompt_adapter_utility_gate_class_balanced),
            "prompt_adapter_utility_gate_source": prompt_adapter_utility_gate_source,
            "prompt_adapter_update_mask_strategy": prompt_adapter_update_mask_strategy,
            "prompt_adapter_loss_mask_strategy": prompt_adapter_loss_mask_strategy,
            "prompt_adapter_gate_budget": prompt_adapter_gate_budget,
            "edge_utility_margin": edge_utility_margin,
            "edge_utility_warmup_epochs": edge_utility_warmup_epochs,
            "correction_alignment_margin": correction_alignment_margin,
            "correction_alignment_warmup_epochs": correction_alignment_warmup_epochs,
            "utility_receive_gate_warmup_epochs": utility_receive_gate_warmup_epochs,
            "utility_receive_gate_query_warmup_epochs": utility_receive_gate_query_warmup_epochs,
            "utility_receive_gate_quantile": utility_receive_gate_quantile,
            "utility_receive_gate_eps": utility_receive_gate_eps,
            "utility_receive_gate_class_balanced": float(utility_receive_gate_class_balanced),
            "utility_receive_gate_label_strategy": utility_receive_gate_label_strategy,
            "receive_gate_budget_min": 0.0 if receive_gate_budget_min is None else float(receive_gate_budget_min),
            "receive_gate_budget_max": 0.0 if receive_gate_budget_max is None else float(receive_gate_budget_max),
            "prompt_correction_warmup_epochs": prompt_correction_warmup_epochs,
            "prompt_correction_eps": prompt_correction_eps,
            "prompt_correction_target": prompt_correction_target,
            "prompt_message_help_warmup_epochs": prompt_message_help_warmup_epochs,
            "prompt_message_help_query_warmup_epochs": prompt_message_help_query_warmup_epochs,
            "prompt_message_help_margin": prompt_message_help_margin,
            "prompt_message_help_class_balanced": float(prompt_message_help_class_balanced),
            "prompt_class_anti_harm_floor": prompt_class_anti_harm_floor,
            "support_query_enabled": float(support_query_stats.get("enabled", False)),
            "support_only_prompt_graph": float(support_only_prompt_graph),
            "support_count": float(support_query_stats.get("support_count", 0)),
            "query_count": float(support_query_stats.get("query_count", 0)),
            "train_support_ratio": float(support_query_stats.get("train_support_ratio", 1.0)),
            "support_query_fold_index": float(support_query_stats.get("fold_index", -1)),
            "support_query_num_folds": float(support_query_stats.get("num_folds", 0)),
            "query_proto_alignment_warmup_epochs": query_proto_alignment_warmup_epochs,
            "query_proto_margin": query_proto_margin,
            "query_proto_class_balanced": float(query_proto_class_balanced),
            "benefit_supervision_label_strategy_quantile": float(benefit_supervision_label_strategy == "quantile"),
            "benefit_supervision_quantile_config": benefit_supervision_quantile,
            "benefit_delta_eps_config": benefit_delta_eps,
            "benefit_supervision_quantile_warmup_epochs": benefit_supervision_quantile_warmup_epochs,
            "benefit_supervision_probe_scale": (
                0.0 if benefit_supervision_probe_scale is None else benefit_supervision_probe_scale
            ),
            "edge_scale_multiplier": current_edge_scale_multiplier,
            "alpha": float(model_out["alpha"].detach().item()),
            **prompt_aware_log,
            **prompt_log,
            **acceptance_supervision_stats,
            **benefit_supervision_stats,
            **prompt_correction_stats,
            **prompt_message_help_stats,
            **prompt_message_help_query_stats,
            **utility_receive_gate_stats,
            **utility_receive_gate_query_stats,
            **query_proto_alignment_stats,
            **edge_utility_stats,
            **correction_alignment_stats,
            **adapter_train_stats,
            **p22_grad_stats,
            **prompt_adapter_message_help_stats,
            **prompt_adapter_utility_gate_stats,
            **p21_channel_expert_stats,
            **p21_channel_utility_stats,
        }
        loss_curve.append(log_item)
        prompt_curve.append({"epoch": float(epoch), **prompt_aware_log, **prompt_log, **adapter_train_stats})

        should_eval = (
            epoch == 1
            or epoch % eval_every == 0
            or epoch == epochs
            or (p24_counterfactual_enabled and epoch == p24_route_start_epoch)
        )
        metrics: dict[str, Any] | None = None
        if should_eval:
            metrics = evaluate_prompt_graph(
                model=model,
                prompt_graph_module=prompt_graph_module,
                input_aligner=input_aligner,
                x=graph.x,
                edge_index=graph.edge_index,
                labels=graph.y,
                train_mask=label_train_mask,
                val_mask=split.val_mask,
                test_mask=split.test_mask,
                num_classes=loaded.num_classes,
                edge_scale_multiplier=current_edge_scale_multiplier,
                epoch=epoch,
                prompt_adapter_module=prompt_adapter_module,
            )
            monitor_lookup = {
                **metrics,
                "train_loss": float(loss.detach().item()),
                "total": float(loss.detach().item()),
                "cls": float(cls_loss.detach().item()),
                "edge_l1": float(edge_l1.detach().item()),
                "prompt_balance": float(prompt_balance.detach().item()),
                "p22_pattern_only_loss": float(p22_pattern_only.detach().item()),
                "p22_basis_teacher_loss": float(p22_basis_teacher.detach().item()),
                "p22_basis_usage_loss": float(p22_basis_usage.detach().item()),
                "p22_aux_loss": float(p22_aux_loss.detach().item()),
            }
            evaluated_train_loss = float(loss.detach().item())
            best_by_val_acc = _update_test_acc_diagnostic(
                best_by_val_acc,
                epoch=epoch,
                train_loss=evaluated_train_loss,
                metrics=metrics,
                selection_metric="val_acc",
                maximize=True,
            )
            best_by_train_loss = _update_test_acc_diagnostic(
                best_by_train_loss,
                epoch=epoch,
                train_loss=evaluated_train_loss,
                metrics=metrics,
                selection_metric="train_loss",
                maximize=False,
            )
            oracle_best_by_test_acc = _update_test_acc_diagnostic(
                oracle_best_by_test_acc,
                epoch=epoch,
                train_loss=evaluated_train_loss,
                metrics=metrics,
                selection_metric="test_acc",
                maximize=True,
            )
            monitor_value = float(monitor_lookup.get(monitor, metrics["val_acc"]))
            if (
                p24_counterfactual_enabled
                and epoch == p24_route_start_epoch
                and not identity_metrics
            ):
                identity_metrics = {
                    "identity_epoch": float(epoch),
                    "monitor_value": monitor_value,
                    **metrics,
                }
                _save_checkpoint(
                    identity_checkpoint_path,
                    model=model,
                    input_aligner=input_aligner,
                    prompt_graph_module=prompt_graph_module,
                    prompt_adapter_module=prompt_adapter_module,
                    epoch=epoch,
                    metrics=identity_metrics,
                )
            candidate_monitor = metrics.get("p24_val_candidate_macro_f1")
            if (
                p24_counterfactual_enabled
                and isinstance(candidate_monitor, (int, float))
                and _monitor_improved(
                    "val_macro_f1",
                    float(candidate_monitor),
                    best_candidate_val,
                    bool(best_candidate_metrics),
                )
            ):
                best_candidate_val = float(candidate_monitor)
                best_candidate_metrics = {
                    "best_candidate_epoch": float(epoch),
                    "candidate_monitor_value": float(candidate_monitor),
                    **metrics,
                }
                _save_checkpoint(
                    candidate_checkpoint_path,
                    model=model,
                    input_aligner=input_aligner,
                    prompt_graph_module=prompt_graph_module,
                    prompt_adapter_module=prompt_adapter_module,
                    epoch=epoch,
                    metrics=best_candidate_metrics,
                )
            if (
                p24_counterfactual_enabled
                and epoch > p24_route_start_epoch
                and _monitor_improved(
                    monitor,
                    monitor_value,
                    best_deployed_val,
                    bool(best_deployed_metrics),
                )
            ):
                best_deployed_val = monitor_value
                best_deployed_metrics = {
                    "best_deployed_epoch": float(epoch),
                    "monitor_value": monitor_value,
                    **metrics,
                }
                _save_checkpoint(
                    deployed_checkpoint_path,
                    model=model,
                    input_aligner=input_aligner,
                    prompt_graph_module=prompt_graph_module,
                    prompt_adapter_module=prompt_adapter_module,
                    epoch=epoch,
                    metrics=best_deployed_metrics,
                )
            if _monitor_improved(monitor, monitor_value, best_val, bool(best_metrics)):
                best_val = monitor_value
                best_metrics = {
                    "best_epoch": float(epoch),
                    "monitor_value": monitor_value,
                    "test_label_cheat_enabled": float(enable_test_label_cheat),
                    "test_label_cheat_fraction": float(test_label_cheat_fraction),
                    "actual_test_label_cheat_fraction": float(actual_test_label_cheat_fraction),
                    "test_label_cheat_count": float(cheat_test_label_count),
                    "label_train_count": float(label_train_mask.sum().item()),
                    **metrics,
                    "total": float(loss.detach().item()),
                    "full_loss": float(full_loss.detach().item()),
                    "p22_aux_loss": float(p22_aux_loss.detach().item()),
                    "gp2f_contrastive_loss": float(gp2f_aux_loss.contrastive.detach().item()),
                    "gp2f_topology_fusion_loss": float(gp2f_aux_loss.topology_fusion.detach().item()),
                    "gp2f_contrastive_mode": gp2f_aux_loss.contrastive_mode,
                    "gp2f_topology_mode": gp2f_aux_loss.topology_mode,
                    "p22_stage1_active": float(p22_stage1_active),
                    "cls": float(cls_loss.detach().item()),
                    "edge_l1": float(edge_l1.detach().item()),
                    "prompt_balance": float(prompt_balance.detach().item()),
                    "p23_feature_harm_loss": float(p23_feature_harm_loss.detach().item()),
                    "p23_feature_gate_supervision_loss": float(p23_feature_gate_supervision_loss.detach().item()),
                    "p24_keep_static_loss": float(p24_keep_static.detach().item()),
                    "p24_hub_suppression_loss": float(p24_hub_suppression.detach().item()),
                    "p24_prompt_homophily_loss": float(p24_prompt_homophily.detach().item()),
                    "p24_edge_budget_loss": float(p24_edge_budget.detach().item()),
                    "p24_receiver_budget_loss": float(p24_receiver_budget.detach().item()),
                    "prompt_role_diversity": float(prompt_role_diversity.detach().item()),
                    "prompt_acceptance": float(prompt_acceptance.detach().item()),
                    "prompt_acceptance_budget": float(prompt_acceptance_budget.detach().item()),
                    "prompt_acceptance_supervision": float(prompt_acceptance_supervision.detach().item()),
                    "prompt_benefit_supervision": float(prompt_benefit_supervision.detach().item()),
                    "prompt_usage_consistency": float(prompt_usage_consistency.detach().item()),
                    "prompt_view_entropy": float(prompt_view_entropy.detach().item()),
                    "prompt_view_prior": float(prompt_view_prior.detach().item()),
                    "class_route": float(class_route.detach().item()),
                    "key_proto": float(key_proto.detach().item()),
                    "prompt_correction_loss": float(prompt_correction_loss.detach().item()),
                    "prompt_anti_harm_loss": float(prompt_anti_harm_loss.detach().item()),
                    "prompt_message_help_loss": float(prompt_message_help.detach().item()),
                    "prompt_message_help_query_loss": float(prompt_message_help_query.detach().item()),
                    "prompt_class_anti_harm_loss": float(prompt_class_anti_harm.detach().item()),
                    "utility_receive_gate_loss": float(utility_receive_gate_loss.detach().item()),
                    "utility_receive_gate_query_loss": float(utility_receive_gate_query_loss.detach().item()),
                    "receive_gate_budget": float(receive_gate_budget.detach().item()),
                    "query_proto_alignment_loss": float(query_proto_alignment.detach().item()),
                    "edge_utility_supervision_loss": float(edge_utility_supervision.detach().item()),
                    "correction_alignment_loss": float(correction_alignment.detach().item()),
                    "correction_alignment_anti_harm_loss": float(correction_alignment_anti_harm.detach().item()),
                    "prompt_adapter_update_norm_loss": float(prompt_adapter_update_norm.detach().item()),
                    "prompt_adapter_gate_budget_loss": float(prompt_adapter_budget.detach().item()),
                    "prompt_adapter_message_help_loss": float(prompt_adapter_message_help.detach().item()),
                    "prompt_adapter_utility_gate_loss": float(prompt_adapter_utility_gate.detach().item()),
                    "prompt_adapter_gate_consistency_loss": float(prompt_adapter_gate_consistency.detach().item()),
                    "prompt_adapter_delta_consistency_loss": float(prompt_adapter_delta_consistency.detach().item()),
                    "prompt_adapter_pattern_balance_loss": float(prompt_adapter_pattern_balance.detach().item()),
                    "prompt_router_pattern_supervision_loss": float(prompt_adapter_pattern_supervision.detach().item()),
                    "prompt_router_pattern_utility_loss": float(prompt_adapter_pattern_utility.detach().item()),
                    "prompt_router_class_pattern_reliability_loss": float(
                        prompt_adapter_class_pattern_reliability.detach().item()
                    ),
                    "p22_pattern_only_loss": float(p22_pattern_only.detach().item()),
                    "p22_pattern_reg_loss": float(p22_pattern_reg.detach().item()),
                    "p22_basis_teacher_loss": float(p22_basis_teacher.detach().item()),
                    "p22_basis_usage_loss": float(p22_basis_usage.detach().item()),
                    "prompt_router_expert_utility_supervision_loss": float(
                        prompt_adapter_expert_utility_supervision.detach().item()
                    ),
                    "prompt_adapter_episode_count": float(prompt_adapter_episode_count),
                    "support_query_enabled": float(support_query_stats.get("enabled", False)),
                    "support_only_prompt_graph": float(support_only_prompt_graph),
                    "support_count": float(support_query_stats.get("support_count", 0)),
                    "query_count": float(support_query_stats.get("query_count", 0)),
                    "train_support_ratio": float(support_query_stats.get("train_support_ratio", 1.0)),
                    "support_query_fold_index": float(support_query_stats.get("fold_index", -1)),
                    "support_query_num_folds": float(support_query_stats.get("num_folds", 0)),
                    **acceptance_supervision_stats,
                    **benefit_supervision_stats,
                    **prompt_correction_stats,
                    **prompt_message_help_stats,
                    **prompt_message_help_query_stats,
                    **utility_receive_gate_stats,
                    **utility_receive_gate_query_stats,
                    **query_proto_alignment_stats,
                    **edge_utility_stats,
                    **correction_alignment_stats,
                    **p23_feature_harm_stats,
                    **adapter_train_stats,
                    **p22_grad_stats,
                    **prompt_adapter_message_help_stats,
                    **prompt_adapter_utility_gate_stats,
                    "edge_scale_multiplier": current_edge_scale_multiplier,
                    **_adapter_stats(model),
                }
                _save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    input_aligner=input_aligner,
                    prompt_graph_module=prompt_graph_module,
                    prompt_adapter_module=prompt_adapter_module,
                    epoch=epoch,
                    metrics=best_metrics,
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_every
            if p22_safe_checkpoint_enabled:
                safe_val_delta = float(metrics.get("p22_val_delta_ce", metrics.get("adapter_val_mean_delta_ce", 0.0)))
                safe_candidate = safe_val_delta >= p22_safe_checkpoint_min_val_delta_ce
                safe_delta_for_score = min(
                    p22_safe_checkpoint_delta_cap,
                    max(p22_safe_checkpoint_min_val_delta_ce, safe_val_delta),
                )
                if p22_safe_checkpoint_metric == "val_acc_plus_val_delta_ce":
                    safe_score = float(metrics["val_acc"]) + p22_safe_checkpoint_delta_weight * safe_delta_for_score
                else:
                    safe_score = float(metrics.get(p22_safe_checkpoint_metric, monitor_value))
                if safe_candidate and safe_score > best_safe_score:
                    best_safe_score = safe_score
                    safe_metrics = {
                        "safe_epoch": float(epoch),
                        "p22_safe_score": safe_score,
                        "p22_safe_checkpoint_found": 1.0,
                        "p22_safe_checkpoint_metric": p22_safe_checkpoint_metric,
                        "p22_safe_val_delta_ce": safe_val_delta,
                        "p22_safe_min_val_delta_ce": p22_safe_checkpoint_min_val_delta_ce,
                        "p22_safe_delta_weight": p22_safe_checkpoint_delta_weight,
                        "p22_safe_delta_cap": p22_safe_checkpoint_delta_cap,
                        "monitor_value": monitor_value,
                        "total": float(loss.detach().item()),
                        "full_loss": float(full_loss.detach().item()),
                        "p22_aux_loss": float(p22_aux_loss.detach().item()),
                        "p22_stage1_active": float(p22_stage1_active),
                        "p23_feature_harm_loss": float(p23_feature_harm_loss.detach().item()),
                        "p23_feature_gate_supervision_loss": float(p23_feature_gate_supervision_loss.detach().item()),
                        "p24_keep_static_loss": float(p24_keep_static.detach().item()),
                        "p24_hub_suppression_loss": float(p24_hub_suppression.detach().item()),
                        "p24_prompt_homophily_loss": float(p24_prompt_homophily.detach().item()),
                        "p24_edge_budget_loss": float(p24_edge_budget.detach().item()),
                        "p24_receiver_budget_loss": float(p24_receiver_budget.detach().item()),
                        **metrics,
                        **adapter_train_stats,
                        **p22_deployment_stats,
                        **p23_feature_harm_stats,
                        **p22_grad_stats,
                        **_adapter_stats(model),
                    }
                    _save_checkpoint(
                        safe_checkpoint_path,
                        model=model,
                        input_aligner=input_aligner,
                        prompt_graph_module=prompt_graph_module,
                        prompt_adapter_module=prompt_adapter_module,
                        epoch=epoch,
                        metrics=safe_metrics,
                    )

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if metrics is None:
                metrics = evaluate_prompt_graph(
                    model=model,
                    prompt_graph_module=prompt_graph_module,
                    input_aligner=input_aligner,
                    x=graph.x,
                    edge_index=graph.edge_index,
                    labels=graph.y,
                    train_mask=label_train_mask,
                    val_mask=split.val_mask,
                    test_mask=split.test_mask,
                    num_classes=loaded.num_classes,
                    edge_scale_multiplier=current_edge_scale_multiplier,
                    epoch=epoch,
                    prompt_adapter_module=prompt_adapter_module,
                )
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "pool": f"{metrics['pool_ratio']:.2f}",
                    "edge": f"{metrics['mean_prompt_edge_weight']:.3f}",
                    "best": f"{best_val:.4f}",
                }
            )
        if patience > 0 and epoch >= p24_early_stop_min_epoch and epochs_no_improve >= patience:
            stopped_epoch = epoch
            early_stopped = True
            if metrics is None:
                metrics = evaluate_prompt_graph(
                    model=model,
                    prompt_graph_module=prompt_graph_module,
                    input_aligner=input_aligner,
                    x=graph.x,
                    edge_index=graph.edge_index,
                    labels=graph.y,
                    train_mask=label_train_mask,
                    val_mask=split.val_mask,
                    test_mask=split.test_mask,
                    num_classes=loaded.num_classes,
                    edge_scale_multiplier=current_edge_scale_multiplier,
                    epoch=epoch,
                    prompt_adapter_module=prompt_adapter_module,
                )
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "val": f"{metrics['val_acc']:.4f}",
                    "test": f"{metrics['test_acc']:.4f}",
                    "pool": f"{metrics['pool_ratio']:.2f}",
                    "edge": f"{metrics['mean_prompt_edge_weight']:.3f}",
                    "stop": "early",
                }
            )
            break

    final_metrics = evaluate_prompt_graph(
        model=model,
        prompt_graph_module=prompt_graph_module,
        input_aligner=input_aligner,
        x=graph.x,
        edge_index=graph.edge_index,
        labels=graph.y,
        train_mask=label_train_mask,
        val_mask=split.val_mask,
        test_mask=split.test_mask,
        num_classes=loaded.num_classes,
        edge_scale_multiplier=1.0,
        epoch=stopped_epoch,
        prompt_adapter_module=prompt_adapter_module,
    )
    if p22_safe_checkpoint_enabled and not safe_metrics:
        safe_metrics = {
            "p22_safe_checkpoint_found": 0.0,
            "p22_safe_checkpoint_fallback_to_base": 1.0,
            "p22_safe_checkpoint_metric": p22_safe_checkpoint_metric,
            "p22_safe_min_val_delta_ce": p22_safe_checkpoint_min_val_delta_ce,
            "p22_safe_delta_weight": p22_safe_checkpoint_delta_weight,
            "p22_safe_delta_cap": p22_safe_checkpoint_delta_cap,
        }
    prompt_message_utility: dict[str, Any] | None = None
    if bool(training_cfg.get("diagnose_prompt_message_utility", False)):
        diagnostic_scale = float(prompt_aware_cfg.get("message_scale", 0.0))
        prompt_message_utility = diagnose_prompt_message_utility(
            model=model,
            prompt_graph_module=prompt_graph_module,
            input_aligner=input_aligner,
            x=graph.x,
            edge_index=graph.edge_index,
            labels=graph.y,
            train_mask=label_train_mask,
            val_mask=split.val_mask,
            test_mask=split.test_mask,
            message_scale=diagnostic_scale,
        )
        write_json(run_dir / "prompt_message_utility.json", prompt_message_utility)
        _write_prompt_message_utility_csv(run_dir / "prompt_message_utility_nodes.csv", prompt_message_utility)
    p24_causal_teacher_summary: dict[str, float] = {}
    if p24_causal_teacher_state is not None:
        teacher_ema = p24_causal_teacher_state["ema"].detach()
        teacher_seen = p24_causal_teacher_state["seen"].detach()
        teacher_harmful_seen = p24_causal_teacher_state["harmful_seen"].detach()
        observed = torch.isfinite(teacher_ema) & (teacher_seen > 0.0)
        stability = teacher_harmful_seen / teacher_seen.clamp_min(1.0)
        threshold = float(p24_causal_teacher_cfg.get("harmful_stability_threshold", 0.70))
        margin = float(p24_causal_teacher_cfg.get("utility_margin", 0.0005))
        stable_harmful = observed & (teacher_ema > margin) & (stability >= threshold)
        p24_causal_teacher_summary = {
            "prompt_count": float(teacher_ema.numel()),
            "observed_prompt_count": float(observed.sum().item()),
            "mean_observations": float(teacher_seen[observed].mean().item() if bool(observed.any()) else 0.0),
            "removal_gain_mean": float(teacher_ema[observed].mean().item() if bool(observed.any()) else 0.0),
            "harmful_stability_mean": float(stability[observed].mean().item() if bool(observed.any()) else 0.0),
            "stable_harmful_prompt_count": float(stable_harmful.sum().item()),
            "stable_harmful_prompt_ratio": float(
                stable_harmful.sum().item() / max(1, int(observed.sum().item()))
            ),
        }
    result = {
        "dataset": loaded.name,
        "seed": seed,
        "base_seed": int(experiment_cfg.get("base_seed", split_seed)),
        "trial_index": trial_index,
        "split_seed": split.seed,
        "model_seed": experiment_cfg.get("model_seed"),
        "trial_seed_mode": str(experiment_cfg.get("trial_seed_mode", "stream")),
        "prompt_variant": variant,
        "component_ablation": str(experiment_cfg.get("ablation", "none")),
        "test_label_cheat_enabled": enable_test_label_cheat,
        "test_label_cheat_fraction": test_label_cheat_fraction,
        "actual_test_label_cheat_fraction": actual_test_label_cheat_fraction,
        "test_label_cheat_count": cheat_test_label_count,
        "label_train_count": int(label_train_mask.sum().item()),
        "shot_setting_mode": shot_setting_mode,
        "split_protocol": split_protocol,
        "effective_shots": int(effective_shots),
        "effective_shot_ratio": None if effective_shot_ratio is None else float(effective_shot_ratio),
        "p23_freeze_input_aligner": bool(p23_freeze_input_aligner),
        "p23_freeze_base_model": bool(freeze_base_model),
        "prompt_graph_trainable_scope": prompt_graph_trainable_scope,
        "p23_train_receiver": bool(training_cfg.get("train_prompt_graph_module", True) and isinstance(prompt_graph_module, P23V01PromptModule)),
        "p23_train_classifier": bool(any(parameter.requires_grad for parameter in model.classifier.parameters())),
        "p24_train_classifier": bool(p24_train_classifier),
        "p23_train_fusion_alpha": bool(
            (getattr(model, "raw_alpha", None) is not None and getattr(model, "raw_alpha").requires_grad)
            or (getattr(model, "alpha_logit", None) is not None and getattr(model, "alpha_logit").requires_grad)
        ),
        "use_original_contrastive": bool(gp2f_aux_loss_cfg.use_original_contrastive),
        "use_original_topology_fusion": bool(gp2f_aux_loss_cfg.use_original_topology_fusion),
        "lambda_gp2f_contrastive": float(gp2f_aux_loss_cfg.lambda_ctr),
        "lambda_gp2f_topology_fusion": float(gp2f_aux_loss_cfg.lambda_fus),
        "rho": float(prompt_graph_cfg.get("rho", 0.0)),
        "topk_prompt_per_node": int(prompt_graph_cfg.get("topk_prompt_per_node", 0)),
        "num_nodes": int(graph.num_nodes),
        "num_features": loaded.num_features,
        "num_classes": loaded.num_classes,
        "checkpoint_path": str(checkpoint_path),
        "pretraining_strategy": str(pretrained_cfg.get("strategy", "GRACE")),
        "pretrained_backbone_type": type(backbone).__name__,
        "pretrained_backbone_uses_edge_weight": bool(
            getattr(backbone, "uses_edge_weight", True)
        ),
        "graphmae_edge_weight_mode": str(
            getattr(backbone, "edge_weight_mode", "native_gcn")
        ),
        "graphmae_edge_weight_strength": float(
            getattr(backbone, "edge_weight_strength", 1.0)
        ),
        "source_dim": source_dim,
        "hidden_dim": hidden_dim,
        "aligner_style": str(model_cfg.get("input_aligner", "linear")),
        "adapter_style": str(model_cfg.get("adapter_style", "stable_zero_init")),
        "use_adapters": bool(model.use_adapters),
        "fusion_alpha_style": str(model_cfg.get("fusion_alpha_style", "sigmoid")),
        "fusion_alpha_init": float(model_cfg.get("alpha_init", 0.5)),
        "fusion_mode": str(model.fusion_mode),
        "p23_max_feature_nodes": int(
            prompt_graph_cfg.get("feature_filter", {}).get("max_feature_nodes", 0)
        ),
        "source_preserving_branch_enabled": bool(
            model.fusion_mode != "adapted_only"
        ),
        "target_adaptation_branch_enabled": bool(
            model.fusion_mode != "frozen_only"
        ),
        "model_dropout": float(model_cfg.get("dropout", getattr(backbone, "dropout", 0.0))),
        "model_variant": _model_variant(model_cfg),
        "final": {**final_metrics, **init_eq},
        "best": {**best_metrics, **init_eq},
        "best_by_val_acc": best_by_val_acc,
        "best_by_train_loss": best_by_train_loss,
        "oracle_best_by_test_acc": oracle_best_by_test_acc,
        "safe": {**safe_metrics, **init_eq},
        "identity": {**identity_metrics, **init_eq},
        "best_candidate": {**best_candidate_metrics, **init_eq},
        "best_deployed": {**best_deployed_metrics, **init_eq},
        "p24_causal_prompt_teacher": p24_causal_teacher_summary,
        "prompt_graph_parameter_count": count_trainable_parameters(prompt_graph_module),
        "prompt_adapter_parameter_count": count_trainable_parameters(prompt_adapter_module),
        "class_key_initialization": class_key_init_stats,
        "pattern_key_initialization": pattern_key_init_stats,
        "p23_static_initialization": p23_static_init_stats,
        "trainable_parameters": trainable_summary,
        "optimizer": optimizer_summary,
        "regularization": {
            "lambda_edge_l1": lambda_edge_l1,
            "lambda_prompt_balance": lambda_prompt_balance,
            "lambda_prompt_role_diversity": lambda_prompt_role_diversity,
            "lambda_prompt_acceptance": lambda_prompt_acceptance,
            "lambda_prompt_acceptance_budget": lambda_prompt_acceptance_budget,
            "acceptance_budget_min": acceptance_budget_min,
            "acceptance_budget_max": acceptance_budget_max,
            "lambda_prompt_acceptance_supervision": lambda_prompt_acceptance_supervision,
            "acceptance_supervision_positive_margin": acceptance_supervision_positive_margin,
            "acceptance_supervision_negative_margin": acceptance_supervision_negative_margin,
            "acceptance_supervision_balance_targets": acceptance_supervision_balance_targets,
            "acceptance_supervision_signal": acceptance_supervision_signal,
            "lambda_prompt_usage_consistency": lambda_prompt_usage_consistency,
            "usage_consistency_margin": usage_consistency_margin,
            "usage_consistency_negative_weight": usage_consistency_negative_weight,
            "lambda_prompt_view_entropy": lambda_prompt_view_entropy,
            "lambda_class_route": lambda_class_route,
            "lambda_key_proto": lambda_key_proto,
            "lambda_prompt_benefit_supervision": lambda_prompt_benefit_supervision,
            "benefit_supervision_warmup_epochs": benefit_supervision_warmup_epochs,
            "benefit_delta_margin": benefit_delta_margin,
            "benefit_supervision_label_strategy": benefit_supervision_label_strategy,
            "benefit_supervision_quantile": benefit_supervision_quantile,
            "benefit_delta_eps": benefit_delta_eps,
            "benefit_supervision_quantile_warmup_epochs": benefit_supervision_quantile_warmup_epochs,
            "benefit_supervision_probe_scale": benefit_supervision_probe_scale,
            "benefit_supervision_balance_targets": benefit_supervision_balance_targets,
            "lambda_prompt_correction": lambda_prompt_correction,
            "lambda_prompt_anti_harm": lambda_prompt_anti_harm,
            "lambda_prompt_message_help": lambda_prompt_message_help,
            "lambda_prompt_class_anti_harm": lambda_prompt_class_anti_harm,
            "prompt_correction_warmup_epochs": prompt_correction_warmup_epochs,
            "prompt_correction_eps": prompt_correction_eps,
            "prompt_correction_target": prompt_correction_target,
            "prompt_message_help_warmup_epochs": prompt_message_help_warmup_epochs,
            "prompt_message_help_margin": prompt_message_help_margin,
            "prompt_message_help_class_balanced": prompt_message_help_class_balanced,
            "prompt_class_anti_harm_floor": prompt_class_anti_harm_floor,
            "lambda_edge_utility_supervision": lambda_edge_utility_supervision,
            "edge_utility_margin": edge_utility_margin,
            "edge_utility_warmup_epochs": edge_utility_warmup_epochs,
            "lambda_correction_alignment": lambda_correction_alignment,
            "lambda_correction_anti_harm": lambda_correction_anti_harm,
            "correction_alignment_margin": correction_alignment_margin,
            "correction_alignment_warmup_epochs": correction_alignment_warmup_epochs,
            "lambda_prompt_adapter_update_norm": lambda_prompt_adapter_update_norm,
            "lambda_prompt_adapter_gate_budget": lambda_prompt_adapter_gate_budget,
            "lambda_prompt_adapter_message_help": lambda_prompt_adapter_message_help,
            "lambda_prompt_adapter_utility_gate": lambda_prompt_adapter_utility_gate,
            "lambda_prompt_adapter_gate_consistency": lambda_prompt_adapter_gate_consistency,
            "lambda_prompt_adapter_delta_consistency": lambda_prompt_adapter_delta_consistency,
            "lambda_prompt_router_pattern_balance": lambda_prompt_router_pattern_balance,
            "lambda_prompt_router_pattern_supervision": lambda_prompt_router_pattern_supervision,
            "lambda_prompt_router_pattern_utility": lambda_prompt_router_pattern_utility,
            "lambda_prompt_router_class_pattern_reliability": lambda_prompt_router_class_pattern_reliability,
            "lambda_prompt_router_deployment_utility": lambda_prompt_router_deployment_utility,
            "lambda_prompt_router_expert_utility_supervision": lambda_prompt_router_expert_utility_supervision,
            "prompt_router_pattern_supervision_temperature": prompt_router_pattern_supervision_temperature,
            "prompt_router_pattern_supervision_probe_norm": prompt_router_pattern_supervision_probe_norm,
            "prompt_router_pattern_utility_temperature": prompt_router_pattern_utility_temperature,
            "prompt_router_pattern_utility_probe_norm": prompt_router_pattern_utility_probe_norm,
            "prompt_router_pattern_utility_margin": prompt_router_pattern_utility_margin,
            "prompt_router_pattern_utility_anti_harm_weight": prompt_router_pattern_utility_anti_harm_weight,
            "prompt_router_pattern_utility_min_teacher_delta": prompt_router_pattern_utility_min_teacher_delta,
            "prompt_router_pattern_utility_helpful_fraction": prompt_router_pattern_utility_helpful_fraction,
            "prompt_router_pattern_utility_unhelpful_node_weight": prompt_router_pattern_utility_unhelpful_node_weight,
            "prompt_router_class_pattern_reliability_temperature": (
                prompt_router_class_pattern_reliability_temperature
            ),
            "prompt_router_class_pattern_reliability_probe_norm": (
                prompt_router_class_pattern_reliability_probe_norm
            ),
            "prompt_router_class_pattern_reliability_positive_margin": (
                prompt_router_class_pattern_reliability_positive_margin
            ),
            "prompt_router_class_pattern_reliability_harmful_margin": (
                prompt_router_class_pattern_reliability_harmful_margin
            ),
            "prompt_router_class_pattern_reliability_min_class_count": (
                prompt_router_class_pattern_reliability_min_class_count
            ),
            "prompt_router_deployment_utility_margin": prompt_router_deployment_utility_margin,
            "prompt_router_deployment_utility_anti_harm_weight": (
                prompt_router_deployment_utility_anti_harm_weight
            ),
            "prompt_router_deployment_utility_anti_harm_margin": (
                prompt_router_deployment_utility_anti_harm_margin
            ),
            "prompt_router_deployment_utility_gain_reward_weight": (
                prompt_router_deployment_utility_gain_reward_weight
            ),
            "prompt_router_deployment_utility_gain_reward_cap": (
                0.0
                if prompt_router_deployment_utility_gain_reward_cap is None
                else prompt_router_deployment_utility_gain_reward_cap
            ),
            "prompt_router_deployment_utility_class_balanced": float(
                prompt_router_deployment_utility_class_balanced
            ),
            "prompt_router_expert_utility_temperature": prompt_router_expert_utility_temperature,
            "prompt_router_expert_utility_gain_temperature": (
                0.0
                if prompt_router_expert_utility_gain_temperature is None
                else prompt_router_expert_utility_gain_temperature
            ),
            "prompt_router_expert_utility_margin": prompt_router_expert_utility_margin,
            "prompt_router_expert_utility_target": prompt_router_expert_utility_target,
            "prompt_router_expert_utility_probe_norm": (
                0.0 if prompt_router_expert_utility_probe_norm is None else prompt_router_expert_utility_probe_norm
            ),
            "prompt_router_expert_utility_class_balanced": prompt_router_expert_utility_class_balanced,
            "prompt_router_expert_utility_gate_weight": prompt_router_expert_utility_gate_weight,
            "prompt_router_expert_utility_gate_target": prompt_router_expert_utility_gate_target,
            "prompt_router_expert_utility_gate_temperature": (
                0.0
                if prompt_router_expert_utility_gate_temperature is None
                else prompt_router_expert_utility_gate_temperature
            ),
            "prompt_adapter_episode_count_per_epoch": prompt_adapter_episode_count,
            "prompt_adapter_message_help_margin": prompt_adapter_message_help_margin,
            "prompt_adapter_message_help_anti_harm_weight": prompt_adapter_message_help_anti_harm_weight,
            "prompt_adapter_message_help_anti_harm_margin": prompt_adapter_message_help_anti_harm_margin,
            "prompt_adapter_message_help_class_balanced": prompt_adapter_message_help_class_balanced,
            "prompt_adapter_utility_gate_temperature": prompt_adapter_utility_gate_temperature,
            "prompt_adapter_utility_gate_margin": prompt_adapter_utility_gate_margin,
            "prompt_adapter_utility_gate_class_balanced": prompt_adapter_utility_gate_class_balanced,
            "prompt_adapter_utility_gate_source": prompt_adapter_utility_gate_source,
            "prompt_adapter_update_mask": prompt_adapter_update_mask_strategy,
            "prompt_adapter_loss_mask": prompt_adapter_loss_mask_strategy,
            "prompt_adapter_gate_budget": prompt_adapter_gate_budget,
            "edge_scale_warmup_epochs": edge_scale_warmup_epochs,
            "edge_scale_warmup_start": edge_scale_warmup_start,
        },
        "base_checkpoint_path": str(base_checkpoint_path) if base_checkpoint_path is not None else "",
        "freeze_base_model": freeze_base_model,
        "train_prompt_graph_module": bool(training_cfg.get("train_prompt_graph_module", True)),
        "train_prompt_adapter": bool(training_cfg.get("train_prompt_adapter", True)),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "early_stop_metric": monitor,
        "validation_class_coverage": float(val_class_coverage),
        "validation_present_classes": val_present_classes,
        "validation_missing_classes": val_missing_classes,
        "p24_training_schedule": {
            "counterfactual_router_enabled": p24_counterfactual_enabled,
            "route_start_epoch": p24_route_start_epoch,
            "reset_early_stop_at_route_start": p24_reset_early_stop,
            "min_route_training_epochs": p24_min_route_training_epochs,
            "effective_early_stop_min_epoch": p24_early_stop_min_epoch,
            "candidate_lr_scale_after_warmup": p24_candidate_lr_scale_after_warmup,
        },
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() and keep_checkpoint else "",
        "safe_checkpoint_path": str(safe_checkpoint_path) if safe_checkpoint_path.exists() and keep_checkpoint else "",
        "identity_checkpoint_path": str(identity_checkpoint_path) if identity_checkpoint_path.exists() and keep_checkpoint else "",
        "best_candidate_checkpoint_path": (
            str(candidate_checkpoint_path) if candidate_checkpoint_path.exists() and keep_checkpoint else ""
        ),
        "best_deployed_checkpoint_path": (
            str(deployed_checkpoint_path) if deployed_checkpoint_path.exists() and keep_checkpoint else ""
        ),
        "split_counts": _split_counts(
            graph.y,
            {
                "train": split.train_mask,
                "label_train": label_train_mask,
                "val": split.val_mask,
                "test": split.test_mask,
            },
            loaded.num_classes,
        ),
        "environment": _environment_info(),
        "run_dir": str(run_dir),
    }
    if prompt_message_utility is not None:
        result["prompt_message_utility"] = {
            key: value for key, value in prompt_message_utility.items() if key != "node_records"
        }
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "loss_curve.json", {"loss_curve": loss_curve})
    write_json(run_dir / "prompt_curve.json", {"prompt_curve": prompt_curve})
    if best_checkpoint_path.exists() and not keep_checkpoint:
        best_checkpoint_path.unlink()
    if safe_checkpoint_path.exists() and not keep_checkpoint:
        safe_checkpoint_path.unlink()
    if identity_checkpoint_path.exists() and not keep_checkpoint:
        identity_checkpoint_path.unlink()
    if candidate_checkpoint_path.exists() and not keep_checkpoint:
        candidate_checkpoint_path.unlink()
    if deployed_checkpoint_path.exists() and not keep_checkpoint:
        deployed_checkpoint_path.unlink()
    print(f"Saved results to {run_dir}")
    return result


def run(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    from pretrain_GNN.pretrain import ensure_pretrained
    ensure_pretrained(config, repo_root=repo_root)
    variant = _prompt_variant(config)
    config = _config_for_variant(config, variant)
    config = _apply_component_ablation(config)
    seeds = _resolve_run_seeds(config)
    trials_per_seed = max(1, int(config.get("experiment", {}).get("trials_per_seed", 1)))
    output_root = _resolve_path(config.get("training", {}).get("output_dir", "outputs/gp2f_prompt_p1"), base_dir=repo_root)
    target_dataset = str(config.get("experiment", {}).get("target_dataset", "Cora"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = output_root / target_dataset / timestamp
    summary_dir.mkdir(parents=True, exist_ok=True)
    scale_grid = _message_scale_grid(config)
    diagnostic_mode = bool(config.get("training", {}).get("diagnose_prompt_message_utility", False))
    if diagnostic_mode:
        if not scale_grid:
            scale_grid = [0.1, 0.25, 0.5, 1.0]
        scale_grid = [float(scale) for scale in scale_grid if float(scale) > 0.0]
        if not scale_grid:
            raise ValueError("diagnose_prompt_message_utility requires at least one positive message scale")
    use_scale_grid = (variant.startswith("p2_") or variant.startswith("p13_")) and len(scale_grid) > 1
    if diagnostic_mode:
        use_scale_grid = False
    scale_selection_metric = str(config.get("training", {}).get("message_scale_selection_metric", "val_acc"))

    results: list[dict[str, Any]] = []
    total_run_count = len(seeds) * trials_per_seed
    run_counter = 0
    for seed in seeds:
        set_seed(int(seed))
        for trial_idx in range(trials_per_seed):
            experiment_cfg = config.get("experiment", {})
            split_seed, trial_model_seed, trial_seed_mode = _resolve_trial_seed_protocol(
                experiment_cfg,
                base_seed=int(seed),
                trial_index=int(trial_idx),
                trials_per_seed=trials_per_seed,
            )
            if trial_model_seed is not None:
                set_seed(trial_model_seed)
            recorded_seed = (
                int(trial_model_seed)
                if trial_seed_mode == "increment" and trial_model_seed is not None
                else int(seed)
            )
            trial_update = {
                "experiment": {
                    "seed": recorded_seed,
                    "split_seed": split_seed,
                    "trial_index": int(trial_idx),
                    "base_seed": int(seed),
                    "reset_seed": False,
                    "trial_seed_mode": trial_seed_mode,
                    "model_seed": (
                        int(trial_model_seed) if trial_model_seed is not None else None
                    ),
                }
            }
            current_run_index = run_counter
            run_counter += 1

            if diagnostic_mode:
                for scale in scale_grid:
                    seed_config = _deep_update(
                        config,
                        {
                            **trial_update,
                            "prompt_aware": {"message_scale": float(scale), "message_scale_grid": [float(scale)]},
                            "training": {"diagnose_prompt_message_utility": True},
                        },
                    )
                    result = run_single(
                        seed_config,
                        repo_root=repo_root,
                        run_index=current_run_index,
                        total_runs=total_run_count,
                        run_group_dir=summary_dir / f"scale_{_scale_label(scale)}",
                    )
                    result["diagnostic_message_scale"] = float(scale)
                    result["selected_message_scale"] = float(scale)
                    result["message_scale_selection_metric"] = "diagnostic_fixed_scale"
                    results.append(result)
            elif use_scale_grid:
                candidates: list[dict[str, Any]] = []
                for scale in scale_grid:
                    seed_config = _deep_update(
                        config,
                        {
                            **trial_update,
                            "prompt_aware": {"message_scale": float(scale)},
                        },
                    )
                    result = run_single(
                        seed_config,
                        repo_root=repo_root,
                        run_index=current_run_index,
                        total_runs=total_run_count,
                        run_group_dir=summary_dir / f"scale_{_scale_label(scale)}",
                    )
                    result["candidate_message_scale"] = float(scale)
                    candidates.append(result)
                selected = _select_message_scale_candidate(candidates, metric=scale_selection_metric)
                selected = copy.deepcopy(selected)
                selected["selected_message_scale"] = float(selected.get("candidate_message_scale", 0.0))
                selected["message_scale_selection_metric"] = scale_selection_metric
                selected["message_scale_candidates"] = [
                    _compact_scale_candidate(candidate, scale_selection_metric) for candidate in candidates
                ]
                results.append(selected)
            else:
                seed_config = _deep_update(config, trial_update)
                result = run_single(
                    seed_config,
                    repo_root=repo_root,
                    run_index=current_run_index,
                    total_runs=total_run_count,
                    run_group_dir=summary_dir,
                )
                if scale_grid:
                    result["selected_message_scale"] = float(scale_grid[0])
                    result["message_scale_selection_metric"] = scale_selection_metric
                results.append(result)

    best_test = [float(result["best"].get("test_acc", 0.0)) for result in results]
    best_macro_f1 = [float(result["best"].get("test_macro_f1", 0.0)) for result in results]
    best_auroc = [float(result["best"].get("test_auroc", 0.0)) for result in results]
    best_auprc = [float(result["best"].get("test_auprc", 0.0)) for result in results]
    val_best_test = [
        float(result.get("best_by_val_acc", {}).get("test_acc", 0.0)) for result in results
    ]
    loss_best_test = [
        float(result.get("best_by_train_loss", {}).get("test_acc", 0.0)) for result in results
    ]
    oracle_best_test = [
        float(result.get("oracle_best_by_test_acc", {}).get("test_acc", 0.0)) for result in results
    ]
    final_test = [float(result["final"].get("test_acc", 0.0)) for result in results]
    final_macro_f1 = [float(result["final"].get("test_macro_f1", 0.0)) for result in results]
    final_auroc = [float(result["final"].get("test_auroc", 0.0)) for result in results]
    final_auprc = [float(result["final"].get("test_auprc", 0.0)) for result in results]
    safe_test = [float(result.get("safe", {}).get("test_acc", 0.0)) for result in results]
    safe_macro_f1 = [float(result.get("safe", {}).get("test_macro_f1", 0.0)) for result in results]
    safe_auroc = [float(result.get("safe", {}).get("test_auroc", 0.0)) for result in results]
    safe_auprc = [float(result.get("safe", {}).get("test_auprc", 0.0)) for result in results]
    safe_found = [float(result.get("safe", {}).get("p22_safe_checkpoint_found", 0.0)) for result in results]
    identity_test = [float(result.get("identity", {}).get("test_acc", 0.0)) for result in results]
    identity_macro_f1 = [float(result.get("identity", {}).get("test_macro_f1", 0.0)) for result in results]
    deployed_test = [float(result.get("best_deployed", {}).get("test_acc", 0.0)) for result in results]
    deployed_macro_f1 = [
        float(result.get("best_deployed", {}).get("test_macro_f1", 0.0)) for result in results
    ]
    candidate_test = [
        float(result.get("best_candidate", {}).get("p24_test_candidate_acc", 0.0)) for result in results
    ]
    candidate_macro_f1 = [
        float(result.get("best_candidate", {}).get("p24_test_candidate_macro_f1", 0.0))
        for result in results
    ]
    cheat_enabled = [float(bool(result.get("test_label_cheat_enabled", False))) for result in results]
    cheat_fractions = [float(result.get("actual_test_label_cheat_fraction", 0.0)) for result in results]
    cheat_counts = [float(result.get("test_label_cheat_count", 0.0)) for result in results]
    label_train_counts = [float(result.get("label_train_count", 0.0)) for result in results]
    summary = {
        "dataset": target_dataset,
        "prompt_variant": variant,
        "pretraining_strategy": str(results[0].get("pretraining_strategy", "GRACE")),
        "pretrained_checkpoint_path": str(results[0].get("checkpoint_path", "")),
        "pretrained_backbone_type": str(results[0].get("pretrained_backbone_type", "BaseGCN")),
        "pretrained_backbone_uses_edge_weight": bool(
            results[0].get("pretrained_backbone_uses_edge_weight", True)
        ),
        "graphmae_edge_weight_mode": str(results[0].get("graphmae_edge_weight_mode", "native_gcn")),
        "graphmae_edge_weight_strength": float(
            results[0].get("graphmae_edge_weight_strength", 1.0)
        ),
        "component_ablation": str(config.get("experiment", {}).get("ablation", "none")),
        "seeds": seeds,
        "trials_per_seed": trials_per_seed,
        "num_runs": len(results),
        "test_label_cheat_enabled": any(bool(value) for value in cheat_enabled),
        "actual_test_label_cheat_fraction_mean_std": _format_mean_std(cheat_fractions),
        "test_label_cheat_count_mean_std": _format_mean_std(cheat_counts, scale=1.0),
        "label_train_count_mean_std": _format_mean_std(label_train_counts, scale=1.0),
        "best_test_acc_mean_std": _format_mean_std(best_test),
        "best_test_macro_f1_mean_std": _format_mean_std(best_macro_f1),
        "best_test_auroc_mean_std": _format_mean_std(best_auroc),
        "best_test_auprc_mean_std": _format_mean_std(best_auprc),
        "test_acc_at_best_val_acc_mean_std": _format_mean_std(val_best_test),
        "test_acc_at_min_train_loss_mean_std": _format_mean_std(loss_best_test),
        "oracle_max_observed_test_acc_mean_std": _format_mean_std(oracle_best_test),
        "test_acc_diagnostic_scope": "evaluated_epochs",
        "final_test_acc_mean_std": _format_mean_std(final_test),
        "final_test_macro_f1_mean_std": _format_mean_std(final_macro_f1),
        "final_test_auroc_mean_std": _format_mean_std(final_auroc),
        "final_test_auprc_mean_std": _format_mean_std(final_auprc),
        "safe_test_acc_mean_std": _format_mean_std(safe_test),
        "safe_test_macro_f1_mean_std": _format_mean_std(safe_macro_f1),
        "safe_test_auroc_mean_std": _format_mean_std(safe_auroc),
        "safe_test_auprc_mean_std": _format_mean_std(safe_auprc),
        "safe_checkpoint_found_mean_std": _format_mean_std(safe_found),
        "identity_test_acc_mean_std": _format_mean_std(identity_test),
        "identity_test_macro_f1_mean_std": _format_mean_std(identity_macro_f1),
        "best_deployed_test_acc_mean_std": _format_mean_std(deployed_test),
        "best_deployed_test_macro_f1_mean_std": _format_mean_std(deployed_macro_f1),
        "best_candidate_test_acc_mean_std": _format_mean_std(candidate_test),
        "best_candidate_test_macro_f1_mean_std": _format_mean_std(candidate_macro_f1),
        "gp2f_all_trials_final_test_accuracy": _format_gp2f_metric(final_test),
        "gp2f_all_trials_final_test_f1": _format_gp2f_metric(final_macro_f1),
        "gp2f_all_trials_final_test_auroc": _format_gp2f_metric(final_auroc),
        "gp2f_all_trials_final_test_auprc": _format_gp2f_metric(final_auprc),
        "message_scale_grid": scale_grid,
        "message_scale_selection_metric": scale_selection_metric if scale_grid else "",
        "message_scale_selection_enabled": use_scale_grid,
        "diagnose_prompt_message_utility": diagnostic_mode,
        "runs": results,
        "environment": _environment_info(),
    }
    diagnostic_summary_keys = [
        "alpha",
        "pool_ratio",
        "train_pool_ratio",
        "edge_scale",
        "prompt_edge_count",
        "use_multiview_routing",
        "use_class_aware_routing",
        "use_attribute_view",
        "use_enhanced_role_view",
        "use_pattern_prompt_bank",
        "semantic_view_weight",
        "structural_view_weight",
        "role_view_weight",
        "attribute_view_weight",
        "view_gate_entropy",
        "semantic_route_margin",
        "structural_route_margin",
        "role_route_margin",
        "attribute_route_margin",
        "receive_gate_mean",
        "utility_receive_gate_mean",
        "receive_gate_budget",
        "prompt_usage_entropy",
        "prompt_usage_full_entropy",
        "pattern_prompt_usage_entropy",
        "pattern_usage_entropy",
        "dominant_prompt_slot_ratio",
        "active_prompt_slot_count@0.05",
        "residual_prompt_usage_ratio",
        "prompt_message_scale",
        "prompt_msg_norm",
        "prompt_to_original_update_norm",
        "correction_norm",
        "prompt_update_clip_ratio",
        "prompt_message_help_loss",
        "class_balanced_mean_delta_ce",
        "class_balanced_positive_delta_ratio",
        "query_class_balanced_mean_delta_ce",
        "query_class_balanced_positive_delta_ratio",
        "utility_receive_gate_loss",
        "utility_receive_gate_query_loss",
        "edge_utility_supervision_loss",
        "edge_utility_delta_corr_train",
        "correction_alignment_loss",
        "correction_alignment_anti_harm_loss",
        "correction_alignment_cosine_mean",
        "pool_score_selected_mean",
        "pool_uncertainty_mean",
        "pool_disagreement_mean",
        "p23_pool_ratio",
        "p23_pool_score_mean",
        "p23_feature_prompt_count",
        "p23_valid_feature_count",
        "p23_prompt_edge_count",
        "p23_avg_prompt_degree",
        "p23_feature_df_mean",
        "p23_feature_reliability_mean",
        "p23_projector_sync_enabled",
        "p23_projector_sync_delta_norm",
        "p23_projector_sync_delta_ratio",
        "p23_projector_sync_cached_cosine",
        "p23_projector_sync_current_norm",
        "p23_projector_sync_mix",
        "p23_projector_sync_norm_scale_mean",
        "p23_projector_sync_norm_scale_std",
        "p23_projector_sync_clip_ratio",
        "p23_projector_sync_aligned_delta_ratio",
        "p23_projector_sync_effective_delta_norm",
        "p23_projector_sync_effective_delta_ratio",
        "p23_projector_sync_effective_cosine",
        "p23_projector_sync_centered_residual_norm",
        "p23_projector_sync_centered_residual_cached_cosine",
        "p23_projector_sync_bank_stable_dim_ratio",
        "p23_projector_sync_bank_aligned_mean_error",
        "p23_projector_receiver_enabled",
        "p23_projector_receiver_signed_scale",
        "p23_projector_receiver_connected_ratio",
        "p23_projector_receiver_context_delta_norm_mean",
        "p23_projector_receiver_edge_context_norm_mean",
        "p23_projector_receiver_update_norm_mean",
        "p23_projector_receiver_update_norm_max",
        "p23_projector_receiver_update_to_base_ratio",
        "p23_no_receiver_val_acc",
        "p23_no_receiver_val_macro_f1",
        "p23_no_receiver_test_acc",
        "p23_no_receiver_test_macro_f1",
        "p23_receiver_test_acc_delta",
        "p23_receiver_test_macro_f1_delta",
        "p23_prompt_residual_enabled",
        "p23_prompt_residual_delta_norm_loss",
        "p23_prompt_residual_delta_align_loss",
        "p23_prompt_residual_gate_budget_loss",
        "p23_prompt_residual_delta_norm_mean",
        "p23_prompt_residual_delta_norm_max",
        "p23_prompt_residual_delta_to_init_ratio",
        "p23_prompt_residual_delta_cosine_to_init",
        "p23_prompt_residual_gate_mean",
        "p23_prompt_residual_gate_max",
        "p23_prompt_residual_gate_reliability_corr",
        "p23_prompt_residual_coeff_abs_mean",
        "p23_prompt_residual_coeff_entropy",
        "p23_prompt_residual_clip_fraction",
        "p23_prompt_residual_reliability_mean",
        "p23_prompt_residual_reliability_active_ratio",
        "p24_enabled",
        "p24_keep_static_loss",
        "p24_hub_suppression_loss",
        "p24_prompt_homophily_loss",
        "p24_edge_budget_loss",
        "p24_receiver_budget_loss",
        "p24_receiver_enabled",
        "p24_receiver_mix_ratio",
        "p24_receiver_scale_mean",
        "p24_receiver_scale_min",
        "p24_receiver_scale_max",
        "p24_receiver_scale_above_one_ratio",
        "p24_crossfit_enabled",
        "p24_crossfit_eligible",
        "p24_crossfit_min_train_per_class",
        "p24_crossfit_class_count",
        "p24_crossfit_target_coverage",
        "p24_edge_gate_mean",
        "p24_edge_gate_min",
        "p24_edge_gate_max",
        "p24_prompt_to_node_gate_mean",
        "p24_node_to_prompt_gate_mean",
        "p24_hub_gate_mean",
        "p24_nonhub_gate_mean",
        "p24_edge_keep_ratio",
        "p24_learned_gate_mean",
        "p24_effective_gate_mean",
        "p24_mix_ratio",
        "p24_gate_above_one_ratio",
        "p24_effective_gate_below_0p3_ratio",
        "p24_effective_gate_below_0p5_ratio",
        "p24_effective_gate_above_one_ratio",
        "p24_within_prompt_gate_std_mean",
        "p24_weight_delta_mean",
        "p24_weight_delta_abs_mean",
        "p24_candidate_confidence_mean",
        "p24_candidate_gate_mean",
        "p24_candidate_gate_min",
        "p24_candidate_gate_max",
        "p24_candidate_weight_delta_abs_mean",
        "p24_candidate_gate_clamp_ratio",
        "p24_candidate_node_l1_mean",
        "p24_candidate_node_l1_max",
        "p24_candidate_projection_enabled",
        "p24_route_p23_mean",
        "p24_route_adapt_mean",
        "p24_route_no_prompt_mean",
        "p24_route_stage_active",
        "p24_route_candidate_detached",
        "p24_route_context_detached",
        "p24_train_edge_homophily_raw",
        "p24_train_edge_homophily_weighted",
        "p24_train_same_class_gate_mean",
        "p24_train_offclass_gate_mean",
        "p24_counterfactual_router_enabled",
        "p24_counterfactual_predicted_gain_mean",
        "p24_counterfactual_predicted_gain_std",
        "p24_counterfactual_deployment_mix",
        "p24_train_static_ce",
        "p24_train_candidate_ce",
        "p24_train_deployed_ce",
        "p24_train_candidate_gain",
        "p24_train_deployed_gain",
        "p24_train_candidate_change_ratio",
        "p24_train_deployed_change_ratio",
        "p24_val_static_ce",
        "p24_val_candidate_ce",
        "p24_val_deployed_ce",
        "p24_val_candidate_gain",
        "p24_val_deployed_gain",
        "p24_val_candidate_change_ratio",
        "p24_val_deployed_change_ratio",
        "p24_test_static_ce",
        "p24_test_candidate_ce",
        "p24_test_deployed_ce",
        "p24_test_candidate_gain",
        "p24_test_deployed_gain",
        "p24_test_candidate_change_ratio",
        "p24_test_deployed_change_ratio",
        "p24_gumbel_enabled",
        "p24_gumbel_temperature",
        "p24_gumbel_pruning_strength",
        "p24_gumbel_keep_probability_mean",
        "p24_gumbel_keep_probability_min",
        "p24_gumbel_keep_probability_max",
        "p24_gumbel_probability_entropy",
        "p24_gumbel_soft_keep_ratio",
        "p24_gumbel_hard_keep_ratio",
        "p24_gumbel_prompt_keep_mean",
        "p24_gumbel_prunable_prior_mean",
        "p24_gumbel_hub_prunable_prior_mean",
        "p24_gumbel_nonhub_prunable_prior_mean",
        "p24_gumbel_hub_keep_mean",
        "p24_gumbel_nonhub_keep_mean",
        "p24_gumbel_effectively_pruned_ratio",
        "p24_gumbel_sample_variance",
        "p24_causal_prompt_teacher_loss",
        "p24_causal_prompt_teacher_count",
        "p24_causal_prompt_teacher_updated",
        "p24_causal_prompt_removal_gain_mean",
        "p24_causal_prompt_harmful_ratio",
        "p24_causal_prompt_harmful_stability_mean",
        "p24_causal_prompt_target_keep_mean",
        "p24_causal_prompt_harmful_keep_mean",
        "p24_causal_prompt_helpful_keep_mean",
        "p24_train_static_acc",
        "p24_train_static_macro_f1",
        "p24_train_candidate_acc",
        "p24_train_candidate_macro_f1",
        "p24_val_static_acc",
        "p24_val_static_macro_f1",
        "p24_val_candidate_acc",
        "p24_val_candidate_macro_f1",
        "p24_test_static_acc",
        "p24_test_static_macro_f1",
        "p24_test_candidate_acc",
        "p24_test_candidate_macro_f1",
        "candidate_pool_enabled",
        "candidate_pool_ratio",
        "candidate_pool_count",
        "candidate_pool_topk_count",
        "candidate_pool_score_mean",
        "candidate_pool_score_selected_mean",
        "candidate_pool_structural_mean",
        "candidate_pool_uncertainty_mean",
        "candidate_pool_disagreement_mean",
        "support_query_enabled",
        "support_only_prompt_graph",
        "support_count",
        "query_count",
        "pattern_key_init_coverage",
        "prompt_adapter_enabled",
        "prompt_adapter_update_norm",
        "prompt_adapter_update_max_norm",
        "prompt_adapter_raw_delta_norm",
        "prompt_adapter_delta_norm",
        "prompt_adapter_gate_mean",
        "prompt_adapter_raw_gate_mean",
        "prompt_adapter_clip_ratio",
        "p21_filter_enabled",
        "p21_beta",
        "p21_gate_mean",
        "p21_gate_min",
        "p21_gate_max",
        "p21_channel_delta_reject_norm",
        "p21_channel_delta_ego_norm",
        "p21_channel_delta_low_norm",
        "p21_channel_delta_two_norm",
        "p21_channel_delta_high_norm",
        "p21_channel_delta_compat_norm",
        "p21_channel_delta_role_norm",
        "p21_alpha_entropy",
        "p21_alpha_reject_mean",
        "p21_alpha_ego_mean",
        "p21_alpha_low_mean",
        "p21_alpha_two_mean",
        "p21_alpha_high_mean",
        "p21_alpha_compat_mean",
        "p21_alpha_role_mean",
        "p21_alpha_global_reject",
        "p21_alpha_global_ego",
        "p21_alpha_global_low",
        "p21_alpha_global_two",
        "p21_alpha_global_high",
        "p21_alpha_global_compat",
        "p21_alpha_global_role",
        "p21_channel_reject_norm",
        "p21_channel_ego_norm",
        "p21_channel_low_norm",
        "p21_channel_two_norm",
        "p21_channel_high_norm",
        "p21_channel_compat_norm",
        "p21_channel_role_norm",
        "p21_v2_filter_enabled",
        "p21_v2_compat_class_coverage",
        "p21_v2_compat_proto_coverage",
        "p21_v2_neighbor_prediction_entropy",
        "p21_ego_low_discrepancy",
        "p21_low_two_discrepancy",
        "p21_no_prompt_entropy",
        "p21_no_prompt_margin",
        "p21_oracle_best_channel_delta_ce",
        "p21_oracle_best_channel_positive_ratio",
        "p21_oracle_best_channel_acc",
        "p21_oracle_best_channel_macro_f1",
        "p21_oracle_best_channel_acc_lift_vs_no_prompt",
        "p21_oracle_best_channel_macro_f1_lift_vs_no_prompt",
        "p21_reject_best_ratio",
        "p21_low_best_ratio",
        "p21_two_best_ratio",
        "p21_high_best_ratio",
        "p21_compat_best_ratio",
        "p21_role_best_ratio",
        "p21_compat_mean_delta_ce",
        "p21_role_mean_delta_ce",
        "p21_routed_delta_ce",
        "p21_routed_positive_ratio",
        "p21_router_agreement_to_oracle",
        "p21_teacher_entropy",
        "p21_gate_supervision_loss",
        "p21_gate_target_mean",
        "p21_gate_target_std",
        "p21_gate_accuracy_to_oracle",
        "p21_v2_oracle_best_channel_delta_ce",
        "p21_v2_oracle_best_channel_positive_ratio",
        "p21_v2_oracle_best_channel_acc",
        "p21_v2_oracle_best_channel_macro_f1",
        "p21_v2_oracle_best_channel_acc_lift_vs_no_prompt",
        "p21_v2_oracle_best_channel_macro_f1_lift_vs_no_prompt",
        "p21_v2_reject_best_ratio",
        "p21_v2_low_best_ratio",
        "p21_v2_two_best_ratio",
        "p21_v2_high_best_ratio",
        "p21_v2_compat_best_ratio",
        "p21_v2_role_best_ratio",
        "p21_v2_compat_mean_delta_ce",
        "p21_v2_role_mean_delta_ce",
        "p21_v2_routed_delta_ce",
        "p21_v2_routed_positive_ratio",
        "p21_v2_router_agreement_to_oracle",
        "p21_v2_gate_target_mean",
        "p21_v2_gate_target_std",
        "p21_v2_gate_mean",
        "p21_v2_gate_accuracy_to_oracle",
        "adapter_query_mean_delta_ce",
        "adapter_query_positive_delta_ratio",
        "adapter_val_mean_delta_ce",
        "adapter_val_positive_delta_ratio",
        "adapter_test_mean_delta_ce",
        "adapter_test_positive_delta_ratio",
        "adapter_train_candidate_pool_mean_delta_ce",
        "adapter_train_candidate_pool_positive_delta_ratio",
        "adapter_train_outside_candidate_pool_mean_delta_ce",
        "adapter_train_outside_candidate_pool_positive_delta_ratio",
        "adapter_query_candidate_pool_mean_delta_ce",
        "adapter_query_candidate_pool_positive_delta_ratio",
        "adapter_query_outside_candidate_pool_mean_delta_ce",
        "adapter_query_outside_candidate_pool_positive_delta_ratio",
        "adapter_val_candidate_pool_mean_delta_ce",
        "adapter_val_candidate_pool_positive_delta_ratio",
        "adapter_val_outside_candidate_pool_mean_delta_ce",
        "adapter_val_outside_candidate_pool_positive_delta_ratio",
        "adapter_test_candidate_pool_mean_delta_ce",
        "adapter_test_candidate_pool_positive_delta_ratio",
        "adapter_test_outside_candidate_pool_mean_delta_ce",
        "adapter_test_outside_candidate_pool_positive_delta_ratio",
        "transition_C_row_entropy",
        "transition_C_diag_mean",
        "transition_C_offdiag_mean",
        "transition_C_max_mean",
        "transition_support_edge_count",
        "transition_support_nonzero_row_ratio",
        "transition_support_class_pair_coverage",
        "basis_delta_ce_ego_logprob",
        "basis_delta_ce_onehop_logprob",
        "basis_delta_ce_twohop_logprob",
        "basis_delta_ce_highpass_ego_onehop",
        "basis_delta_ce_highpass_onehop_twohop",
        "basis_delta_ce_onehop_transition_logprob",
        "basis_delta_ce_highpass_ego_transition_onehop",
        "basis_delta_ce_onehop_transition_logprob_train_best",
        "basis_delta_ce_onehop_transition_logprob_val_best",
        "basis_delta_ce_onehop_transition_logprob_test_best",
        "support_context_enabled",
        "support_context_available",
        "support_context_coverage",
        "support_context_count",
        "support_similarity_margin",
        "support_similarity_entropy",
        "support_reliability_mean",
        "support_topk_mean_score",
        "prompt_adapter_message_help_loss",
        "prompt_adapter_message_help_mean_delta_ce",
        "prompt_adapter_message_help_positive_ratio",
        "prompt_adapter_message_help_count",
        "prompt_adapter_message_help_anti_harm_loss",
        "prompt_adapter_utility_gate_loss",
        "prompt_adapter_utility_gate_target_mean",
        "prompt_adapter_utility_gate_count",
        "prompt_adapter_utility_gate_positive_ratio",
        "prompt_adapter_gate_consistency_loss",
        "prompt_adapter_delta_consistency_loss",
        "prompt_adapter_episode_count",
        "prompt_router_pattern_supervision_count",
        "prompt_router_pattern_routing_agreement",
        "prompt_router_pattern_supervision_target_entropy",
        "prompt_router_pattern_utility_loss",
        "prompt_router_pattern_utility_mean_delta_ce",
        "prompt_router_pattern_utility_positive_ratio",
        "prompt_router_pattern_utility_harmful_ratio",
        "prompt_router_pattern_utility_helpful_node_ratio",
        "prompt_router_pattern_utility_count",
        "prompt_router_pattern_utility_target_entropy",
        "prompt_router_class_pattern_reliability_loss",
        "prompt_router_class_pattern_reliability_count",
        "prompt_router_class_pattern_reliable_pair_ratio",
        "prompt_router_class_pattern_harmful_pair_ratio",
        "prompt_router_class_pattern_target_entropy",
        "prompt_router_class_pattern_nonreject_target_mass",
        "prompt_router_deployment_loss",
        "prompt_router_deployment_count",
        "prompt_router_deployment_mean_delta_ce",
        "prompt_router_deployment_positive_delta_ratio",
        "prompt_router_deployment_harmful_delta_ratio",
        "prompt_router_deployment_margin_satisfied_ratio",
        "prompt_router_deployment_prompt_ce",
        "prompt_router_deployment_no_prompt_ce",
        "prompt_router_deployment_anti_harm_loss",
        "prompt_router_deployment_gain_reward",
        "p21_channel_expert_utility_loss",
        "p21_channel_expert_loss",
        "p21_channel_expert_count",
        "p21_channel_expert_mean_delta_ce",
        "p21_channel_expert_positive_ratio",
        "p21_channel_expert_best_delta_ce",
        "p21_channel_expert_anti_harm_loss",
        "p21_channel_expert_best_channel_ratio_low",
        "p21_channel_expert_best_channel_ratio_two",
        "p21_channel_expert_best_channel_ratio_high",
        "p21_channel_expert_best_channel_ratio_compat",
        "p21_channel_expert_best_channel_ratio_role",
        "p21_channel_expert_low_mean_delta_ce",
        "p21_channel_expert_two_mean_delta_ce",
        "p21_channel_expert_high_mean_delta_ce",
        "p21_channel_expert_compat_mean_delta_ce",
        "p21_channel_expert_role_mean_delta_ce",
        "p21_channel_utility_loss",
        "p21_gate_utility_loss",
        "p21_channel_utility_gate_loss",
        "p21_channel_utility_gate_supervision_loss",
        "p21_channel_utility_count",
        "p21_channel_utility_mean_oracle_delta_ce",
        "p21_channel_utility_best_channel_delta_ce",
        "p21_channel_utility_positive_oracle_ratio",
        "p21_channel_utility_best_channel_positive_ratio",
        "p21_channel_utility_routed_delta_ce",
        "p21_channel_utility_routed_positive_ratio",
        "p21_channel_utility_routing_agreement",
        "p21_channel_utility_router_agreement_to_oracle",
        "p21_channel_utility_teacher_entropy",
        "p21_channel_utility_gate_target_mean",
        "p21_channel_utility_gate_target_std",
        "p21_channel_utility_gate_mean",
        "p21_channel_utility_gate_accuracy_to_oracle",
        "p21_channel_utility_best_channel_acc",
        "p21_channel_utility_best_channel_macro_f1",
        "p21_channel_utility_best_channel_acc_lift_vs_no_prompt",
        "p21_channel_utility_best_channel_macro_f1_lift_vs_no_prompt",
        "p21_channel_utility_reject_mean_delta_ce",
        "p21_channel_utility_low_mean_delta_ce",
        "p21_channel_utility_two_mean_delta_ce",
        "p21_channel_utility_high_mean_delta_ce",
        "p21_channel_utility_compat_mean_delta_ce",
        "p21_channel_utility_role_mean_delta_ce",
        "p21_channel_utility_reject_best_ratio",
        "p21_channel_utility_low_best_ratio",
        "p21_channel_utility_two_best_ratio",
        "p21_channel_utility_high_best_ratio",
        "p21_channel_utility_compat_best_ratio",
        "p21_channel_utility_role_best_ratio",
        "p21_channel_utility_reject_alpha_mean",
        "p21_channel_utility_low_alpha_mean",
        "p21_channel_utility_two_alpha_mean",
        "p21_channel_utility_high_alpha_mean",
        "p21_channel_utility_compat_alpha_mean",
        "p21_channel_utility_role_alpha_mean",
        "effective_lambda_p21_channel_expert_utility",
        "effective_lambda_p21_channel_utility",
        "effective_lambda_p21_gate_utility",
        "effective_lambda_prompt_router_deployment_utility",
        "p21_expert_warmup_active",
        "prompt_router_expert_loss",
        "prompt_router_expert_oracle_best_expert_gain",
        "prompt_router_expert_oracle_positive_ratio",
        "prompt_router_expert_no_prompt_acc",
        "prompt_router_expert_no_prompt_macro_f1",
        "prompt_router_expert_oracle_best_expert_acc",
        "prompt_router_expert_oracle_best_expert_macro_f1",
        "prompt_router_expert_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_router_accuracy_to_best_expert",
        "prompt_router_expert_router_soft_target_kl",
        "prompt_router_expert_no_correction_ratio",
        "prompt_router_expert_learned_weighted_delta_ce",
        "prompt_router_expert_router_loss",
        "prompt_router_expert_gate_supervision_loss",
        "prompt_router_expert_gate_target_mean",
        "prompt_router_expert_gate_mean",
        "prompt_router_expert_gate_accuracy_to_oracle",
        "prompt_router_expert_train_oracle_best_expert_gain",
        "prompt_router_expert_train_oracle_positive_ratio",
        "prompt_router_expert_train_no_prompt_acc",
        "prompt_router_expert_train_no_prompt_macro_f1",
        "prompt_router_expert_train_oracle_best_expert_acc",
        "prompt_router_expert_train_oracle_best_expert_macro_f1",
        "prompt_router_expert_train_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_train_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_train_router_accuracy_to_best_expert",
        "prompt_router_expert_train_no_correction_ratio",
        "prompt_router_expert_train_learned_weighted_delta_ce",
        "prompt_router_expert_train_gate_supervision_loss",
        "prompt_router_expert_train_gate_target_mean",
        "prompt_router_expert_train_gate_mean",
        "prompt_router_expert_train_gate_accuracy_to_oracle",
        "prompt_router_expert_val_oracle_best_expert_gain",
        "prompt_router_expert_val_oracle_positive_ratio",
        "prompt_router_expert_val_no_prompt_acc",
        "prompt_router_expert_val_no_prompt_macro_f1",
        "prompt_router_expert_val_oracle_best_expert_acc",
        "prompt_router_expert_val_oracle_best_expert_macro_f1",
        "prompt_router_expert_val_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_val_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_val_router_accuracy_to_best_expert",
        "prompt_router_expert_val_no_correction_ratio",
        "prompt_router_expert_val_learned_weighted_delta_ce",
        "prompt_router_expert_val_gate_supervision_loss",
        "prompt_router_expert_val_gate_target_mean",
        "prompt_router_expert_val_gate_mean",
        "prompt_router_expert_val_gate_accuracy_to_oracle",
        "prompt_router_expert_test_oracle_best_expert_gain",
        "prompt_router_expert_test_oracle_positive_ratio",
        "prompt_router_expert_test_no_prompt_acc",
        "prompt_router_expert_test_no_prompt_macro_f1",
        "prompt_router_expert_test_oracle_best_expert_acc",
        "prompt_router_expert_test_oracle_best_expert_macro_f1",
        "prompt_router_expert_test_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_test_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_test_router_accuracy_to_best_expert",
        "prompt_router_expert_test_no_correction_ratio",
        "prompt_router_expert_test_learned_weighted_delta_ce",
        "prompt_router_expert_test_gate_supervision_loss",
        "prompt_router_expert_test_gate_target_mean",
        "prompt_router_expert_test_gate_mean",
        "prompt_router_expert_test_gate_accuracy_to_oracle",
        "prompt_router_expert_train_pool_oracle_best_expert_gain",
        "prompt_router_expert_train_pool_oracle_positive_ratio",
        "prompt_router_expert_train_pool_no_prompt_acc",
        "prompt_router_expert_train_pool_no_prompt_macro_f1",
        "prompt_router_expert_train_pool_oracle_best_expert_acc",
        "prompt_router_expert_train_pool_oracle_best_expert_macro_f1",
        "prompt_router_expert_train_pool_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_train_pool_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_train_pool_router_accuracy_to_best_expert",
        "prompt_router_expert_train_pool_no_correction_ratio",
        "prompt_router_expert_train_pool_learned_weighted_delta_ce",
        "prompt_router_expert_train_pool_gate_supervision_loss",
        "prompt_router_expert_train_pool_gate_target_mean",
        "prompt_router_expert_train_pool_gate_mean",
        "prompt_router_expert_train_pool_gate_accuracy_to_oracle",
        "prompt_router_expert_val_pool_oracle_best_expert_gain",
        "prompt_router_expert_val_pool_oracle_positive_ratio",
        "prompt_router_expert_val_pool_no_prompt_acc",
        "prompt_router_expert_val_pool_no_prompt_macro_f1",
        "prompt_router_expert_val_pool_oracle_best_expert_acc",
        "prompt_router_expert_val_pool_oracle_best_expert_macro_f1",
        "prompt_router_expert_val_pool_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_val_pool_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_val_pool_router_accuracy_to_best_expert",
        "prompt_router_expert_val_pool_no_correction_ratio",
        "prompt_router_expert_val_pool_learned_weighted_delta_ce",
        "prompt_router_expert_val_pool_gate_supervision_loss",
        "prompt_router_expert_val_pool_gate_target_mean",
        "prompt_router_expert_val_pool_gate_mean",
        "prompt_router_expert_val_pool_gate_accuracy_to_oracle",
        "prompt_router_expert_test_pool_oracle_best_expert_gain",
        "prompt_router_expert_test_pool_oracle_positive_ratio",
        "prompt_router_expert_test_pool_no_prompt_acc",
        "prompt_router_expert_test_pool_no_prompt_macro_f1",
        "prompt_router_expert_test_pool_oracle_best_expert_acc",
        "prompt_router_expert_test_pool_oracle_best_expert_macro_f1",
        "prompt_router_expert_test_pool_oracle_best_expert_acc_lift_vs_no_prompt",
        "prompt_router_expert_test_pool_oracle_best_expert_macro_f1_lift_vs_no_prompt",
        "prompt_router_expert_test_pool_router_accuracy_to_best_expert",
        "prompt_router_expert_test_pool_no_correction_ratio",
        "prompt_router_expert_test_pool_learned_weighted_delta_ce",
        "prompt_router_expert_test_pool_gate_supervision_loss",
        "prompt_router_expert_test_pool_gate_target_mean",
        "prompt_router_expert_test_pool_gate_mean",
        "prompt_router_expert_test_pool_gate_accuracy_to_oracle",
    ]
    diagnostic_summary_keys.extend(
        [
            "p24_class_coverage_guard_loss",
            "p24_class_separation_loss",
            "p24_class_marginal_balance_loss",
            "p24_prediction_coverage_ratio",
            "p24_prediction_dominant_ratio",
            "p24_prediction_marginal_entropy_ratio",
            "p24_collapse_guard_strength",
            "p24_class_mean_margin",
        ]
    )
    for split_name in ("train", "val", "test"):
        for branch_name in ("static", "candidate", "deployed"):
            diagnostic_summary_keys.extend(
                [
                    f"p24_{split_name}_{branch_name}_prediction_coverage_ratio",
                    f"p24_{split_name}_{branch_name}_prediction_dominant_ratio",
                    f"p24_{split_name}_{branch_name}_prediction_entropy_ratio",
                ]
            )
    expert_diag_prefixes = [
        "prompt_router_expert",
        "prompt_router_expert_train",
        "prompt_router_expert_val",
        "prompt_router_expert_test",
        "prompt_router_expert_train_pool",
        "prompt_router_expert_val_pool",
        "prompt_router_expert_test_pool",
    ]
    p21_oracle_prefixes = [
        "p21_oracle_train",
        "p21_oracle_val",
        "p21_oracle_test",
        "p21_v2_oracle_train",
        "p21_v2_oracle_val",
        "p21_v2_oracle_test",
        "p21_v2_ungated_oracle_train",
        "p21_v2_ungated_oracle_val",
        "p21_v2_ungated_oracle_test",
    ]
    p21_oracle_fields = [
        "loss",
        "gate_loss",
        "gate_supervision_loss",
        "count",
        "mean_oracle_delta_ce",
        "best_channel_delta_ce",
        "positive_oracle_ratio",
        "best_channel_positive_ratio",
        "routed_delta_ce",
        "routed_positive_ratio",
        "routing_agreement",
        "router_agreement_to_oracle",
        "teacher_entropy",
        "gate_target_mean",
        "gate_target_std",
        "gate_mean",
        "gate_accuracy_to_oracle",
        "best_channel_acc",
        "best_channel_macro_f1",
        "best_channel_acc_lift_vs_no_prompt",
        "best_channel_macro_f1_lift_vs_no_prompt",
        "best_scale",
    ]
    for prefix in p21_oracle_prefixes:
        for field in p21_oracle_fields:
            diagnostic_summary_keys.append(f"{prefix}_{field}")
        oracle_channel_names = P21_V2_CHANNEL_NAMES if prefix.startswith("p21_v2_") else CHANNEL_NAMES
        for channel_name in oracle_channel_names:
            diagnostic_summary_keys.append(f"{prefix}_{channel_name}_mean_delta_ce")
            diagnostic_summary_keys.append(f"{prefix}_{channel_name}_best_ratio")
            diagnostic_summary_keys.append(f"{prefix}_{channel_name}_alpha_mean")
    oracle_scaled_fields = [
        "oracle_scaled_best_scale",
        "oracle_scaled_best_gain",
        "oracle_scaled_best_acc",
        "oracle_scaled_best_macro_f1",
        "oracle_scaled_best_acc_lift_vs_no_prompt",
        "oracle_scaled_best_macro_f1_lift_vs_no_prompt",
    ]
    expert_extra_summary_fields = [
        "gate_target_std",
    ]
    oracle_scale_grid_summary = _float_grid(
        config.get("prompt_adapter", {}).get("prompt_router_expert_oracle_scale_grid"),
        default=[0.5, 1.0, 2.0, 4.0],
    )
    for prefix in expert_diag_prefixes:
        for field in expert_extra_summary_fields:
            diagnostic_summary_keys.append(f"{prefix}_{field}")
        for field in oracle_scaled_fields:
            diagnostic_summary_keys.append(f"{prefix}_{field}")
        for scale_value in oracle_scale_grid_summary:
            label = _scale_label(scale_value)
            diagnostic_summary_keys.extend(
                [
                    f"{prefix}_oracle_scale_{label}_gain",
                    f"{prefix}_oracle_scale_{label}_acc_lift_vs_no_prompt",
                    f"{prefix}_oracle_scale_{label}_macro_f1_lift_vs_no_prompt",
                ]
            )
    for key in diagnostic_summary_keys:
        values = [
            float(result["best"][key])
            for result in results
            if isinstance(result.get("best"), dict)
            and isinstance(result["best"].get(key), (int, float))
        ]
        if values:
            precision = 6 if any(piece in key for piece in ("delta", "gain", "loss", "kl", "lift")) else 2
            summary[f"{key}_mean_std"] = _format_mean_std(values, scale=1.0, precision=precision)
    if diagnostic_mode:
        utility_by_scale: dict[str, dict[str, float]] = {}
        for scale in scale_grid:
            scale_results = [
                result.get("prompt_message_utility", {})
                for result in results
                if float(result.get("diagnostic_message_scale", -1.0)) == float(scale)
            ]
            if not scale_results:
                continue
            utility_by_scale[str(float(scale))] = {
                "runs": float(len(scale_results)),
                "mean_delta_ce": float(
                    sum(float(item.get("mean_delta_ce", 0.0)) for item in scale_results) / len(scale_results)
                ),
                "positive_delta_ratio": float(
                    sum(float(item.get("positive_delta_ratio", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "mean_ce_no_prompt": float(
                    sum(float(item.get("mean_ce_no_prompt", 0.0)) for item in scale_results) / len(scale_results)
                ),
                "mean_ce_prompt": float(
                    sum(float(item.get("mean_ce_prompt", 0.0)) for item in scale_results) / len(scale_results)
                ),
                "mean_delta_ce_train_pool": float(
                    sum(float(item.get("mean_delta_ce_train_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "positive_delta_ratio_train_pool": float(
                    sum(float(item.get("positive_delta_ratio_train_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "mean_delta_ce_val_pool": float(
                    sum(float(item.get("mean_delta_ce_val_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "positive_delta_ratio_val_pool": float(
                    sum(float(item.get("positive_delta_ratio_val_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "mean_delta_ce_test_pool": float(
                    sum(float(item.get("mean_delta_ce_test_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "positive_delta_ratio_test_pool": float(
                    sum(float(item.get("positive_delta_ratio_test_pool", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "acceptance_delta_correlation": float(
                    sum(float(item.get("acceptance_delta_correlation", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
                "structural_score_delta_correlation": float(
                    sum(float(item.get("structural_score_delta_correlation", 0.0)) for item in scale_results)
                    / len(scale_results)
                ),
            }
        summary["prompt_message_utility_by_scale"] = utility_by_scale
    write_json(summary_dir / "summary.json", summary)
    with (summary_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "base_seed",
            "split_seed",
            "trial_index",
            "pretraining_strategy",
            "checkpoint_path",
            "pretrained_backbone_type",
            "pretrained_backbone_uses_edge_weight",
            "graphmae_edge_weight_mode",
            "graphmae_edge_weight_strength",
            "test_label_cheat_enabled",
            "test_label_cheat_fraction",
            "actual_test_label_cheat_fraction",
            "test_label_cheat_count",
            "label_train_count",
            "best_epoch",
            "best_test_acc",
            "best_test_macro_f1",
            "best_test_auroc",
            "best_test_auprc",
            "best_val_acc_epoch",
            "best_val_acc",
            "test_acc_at_best_val_acc",
            "min_train_loss_epoch",
            "min_train_loss",
            "test_acc_at_min_train_loss",
            "oracle_max_test_acc_epoch",
            "oracle_max_observed_test_acc",
            "safe_checkpoint_found",
            "safe_test_acc",
            "safe_test_macro_f1",
            "safe_test_auroc",
            "safe_test_auprc",
            "safe_val_delta_ce",
            "final_test_acc",
            "final_test_macro_f1",
            "final_test_auroc",
            "final_test_auprc",
            "alpha",
            "branch_cosine",
            "rho",
            "pool_ratio",
            "train_pool_ratio",
            "candidate_pool_enabled",
            "candidate_pool_ratio",
            "candidate_pool_count",
            "candidate_pool_topk_count",
            "candidate_pool_score_mean",
            "candidate_pool_score_selected_mean",
            "candidate_pool_structural_mean",
            "candidate_pool_uncertainty_mean",
            "candidate_pool_disagreement_mean",
            "adapter_query_candidate_pool_mean_delta_ce",
            "adapter_query_candidate_pool_positive_delta_ratio",
            "adapter_query_outside_candidate_pool_mean_delta_ce",
            "adapter_query_outside_candidate_pool_positive_delta_ratio",
            "adapter_test_candidate_pool_mean_delta_ce",
            "adapter_test_candidate_pool_positive_delta_ratio",
            "adapter_test_outside_candidate_pool_mean_delta_ce",
            "adapter_test_outside_candidate_pool_positive_delta_ratio",
            "prompt_node_count",
            "prompt_edge_count",
            "edge_scale",
            "raw_edge_scale",
            "edge_scale_multiplier",
            "use_multiview_routing",
            "use_class_aware_routing",
            "use_attribute_view",
            "use_enhanced_role_view",
            "use_pattern_prompt_bank",
            "use_receiver_only_prompt",
            "use_benefit_gate",
            "num_class_prompt_slots",
            "residual_prompt_count",
            "class_key_proto_init_coverage",
            "pattern_key_init_coverage",
            "semantic_view_weight",
            "structural_view_weight",
            "role_view_weight",
            "attribute_view_weight",
            "view_gate_entropy",
            "semantic_route_margin",
            "structural_route_margin",
            "role_route_margin",
            "attribute_route_margin",
            "prompt_view_prior",
            "mean_prompt_edge_weight",
            "pool_acceptance_mean",
            "pool_acceptance_min",
            "pool_acceptance_max",
            "use_hard_acceptance",
            "hard_acceptance_ratio",
            "hard_acceptance_selected_ratio",
            "prompt_acceptance_budget",
            "prompt_acceptance_supervision",
            "acceptance_supervised_count",
            "acceptance_positive_count",
            "acceptance_negative_count",
            "acceptance_ignored_count",
            "acceptance_target_mean",
            "acceptance_score_delta_mean",
            "ce_delta_positive_ratio_train",
            "ce_delta_negative_ratio_train",
            "prompt_helpful_acceptance_precision_train",
            "prompt_harmful_rejection_precision_train",
            "prompt_benefit_supervision",
            "benefit_supervised_count",
            "benefit_positive_count",
            "benefit_negative_count",
            "benefit_ignored_count",
            "mean_delta_ce_train_pool",
            "positive_delta_ratio_train_pool",
            "benefit_weight_delta_corr_train",
            "benefit_label_strategy_quantile",
            "benefit_quantile_fraction",
            "benefit_positive_threshold",
            "benefit_negative_threshold",
            "benefit_delta_eps",
            "prompt_correction_loss",
            "prompt_anti_harm_loss",
            "prompt_correction_supervised_count",
            "prompt_correction_harmful_count",
            "prompt_correction_target_norm",
            "prompt_delta_logit_norm",
            "prompt_delta_logit_label_alignment",
            "harmful_prompt_delta_ratio",
            "prompt_message_help_loss",
            "prompt_message_help_query_loss",
            "prompt_class_anti_harm_loss",
            "class_balanced_mean_delta_ce",
            "class_balanced_positive_delta_ratio",
            "query_class_balanced_mean_delta_ce",
            "query_class_balanced_positive_delta_ratio",
            "message_help_supervised_count",
            "message_help_query_supervised_count",
            "message_help_class_count",
            "message_help_query_class_count",
            "message_help_margin",
            "message_help_class_balanced",
            "prompt_class_anti_harm_floor",
            "utility_receive_gate_loss",
            "utility_receive_gate_query_loss",
            "receive_gate_budget",
            "utility_gate_supervised_count",
            "utility_gate_query_supervised_count",
            "utility_gate_positive_count",
            "utility_gate_query_positive_count",
            "utility_gate_negative_count",
            "utility_gate_query_negative_count",
            "utility_gate_ignored_count",
            "utility_gate_query_ignored_count",
            "utility_gate_target_mean",
            "utility_gate_query_target_mean",
            "utility_gate_delta_corr_train",
            "utility_gate_query_delta_corr",
            "utility_gate_label_strategy",
            "utility_gate_query_label_strategy",
            "query_proto_alignment_loss",
            "query_proto_supervised_count",
            "query_proto_class_count",
            "query_proto_mean_delta_dist",
            "query_proto_positive_ratio",
            "correction_alignment_loss",
            "correction_alignment_anti_harm_loss",
            "correction_alignment_node_count",
            "correction_alignment_harmful_count",
            "correction_alignment_cosine_mean",
            "correction_alignment_delta_h_norm",
            "correction_alignment_delta_ce_mean",
            "support_query_enabled",
            "support_only_prompt_graph",
            "support_count",
            "query_count",
            "train_support_ratio",
            "use_utility_receive_gate",
            "utility_receive_gate_mean",
            "utility_receive_gate_min",
            "utility_receive_gate_max",
            "utility_receive_gate_floor",
            "benefit_gate_mean",
            "benefit_gate_min",
            "benefit_gate_max",
            "use_edge_utility",
            "edge_utility_mean",
            "edge_utility_min",
            "edge_utility_max",
            "edge_utility_supervision_loss",
            "edge_utility_supervised_count",
            "edge_utility_positive_count",
            "edge_utility_negative_count",
            "edge_utility_ignored_count",
            "edge_utility_target_mean",
            "edge_utility_delta_ce_mean",
            "edge_utility_delta_ce_positive_ratio",
            "edge_utility_delta_corr_train",
            "pool_strategy_id",
            "pool_selected_ratio",
            "pool_score_mean",
            "pool_score_selected_mean",
            "pool_uncertainty_mean",
            "pool_disagreement_mean",
            "use_hard_receive_gate",
            "hard_receive_ratio",
            "hard_receive_selected_ratio",
            "receive_gate_mean",
            "receive_gate_min",
            "receive_gate_max",
            "class_route",
            "key_proto",
            "class_router_hit_rate_train",
            "class_router_hit_rate_val",
            "class_router_hit_rate_test",
            "prompt_label_purity_train",
            "residual_prompt_usage_ratio",
            "p23_prompt_residual_enabled",
            "p23_prompt_residual_delta_norm_loss",
            "p23_prompt_residual_delta_align_loss",
            "p23_prompt_residual_gate_budget_loss",
            "p23_prompt_residual_delta_norm_mean",
            "p23_prompt_residual_delta_norm_max",
            "p23_prompt_residual_delta_to_init_ratio",
            "p23_prompt_residual_delta_cosine_to_init",
            "p23_prompt_residual_gate_mean",
            "p23_prompt_residual_gate_max",
            "p23_prompt_residual_gate_reliability_corr",
            "p23_prompt_residual_coeff_abs_mean",
            "p23_prompt_residual_coeff_entropy",
            "p23_prompt_residual_clip_fraction",
            "p23_prompt_residual_reliability_mean",
            "p23_prompt_residual_reliability_active_ratio",
            "p24_enabled",
            "p24_keep_static_loss",
            "p24_hub_suppression_loss",
            "p24_prompt_homophily_loss",
            "p24_edge_budget_loss",
            "p24_receiver_budget_loss",
            "p24_receiver_enabled",
            "p24_receiver_mix_ratio",
            "p24_receiver_scale_mean",
            "p24_receiver_scale_min",
            "p24_receiver_scale_max",
            "p24_receiver_scale_above_one_ratio",
            "p24_crossfit_enabled",
            "p24_crossfit_eligible",
            "p24_crossfit_min_train_per_class",
            "p24_crossfit_class_count",
            "p24_crossfit_target_coverage",
            "p24_edge_gate_mean",
            "p24_edge_gate_min",
            "p24_edge_gate_max",
            "p24_prompt_to_node_gate_mean",
            "p24_node_to_prompt_gate_mean",
            "p24_hub_gate_mean",
            "p24_nonhub_gate_mean",
            "p24_edge_keep_ratio",
            "p24_learned_gate_mean",
            "p24_effective_gate_mean",
            "p24_mix_ratio",
            "p24_gate_above_one_ratio",
            "p24_effective_gate_below_0p3_ratio",
            "p24_effective_gate_below_0p5_ratio",
            "p24_effective_gate_above_one_ratio",
            "p24_within_prompt_gate_std_mean",
            "p24_weight_delta_mean",
            "p24_weight_delta_abs_mean",
            "p24_candidate_confidence_mean",
            "p24_candidate_gate_mean",
            "p24_candidate_gate_min",
            "p24_candidate_gate_max",
            "p24_candidate_weight_delta_abs_mean",
            "p24_candidate_gate_clamp_ratio",
            "p24_candidate_node_l1_mean",
            "p24_candidate_node_l1_max",
            "p24_candidate_projection_enabled",
            "p24_route_p23_mean",
            "p24_route_adapt_mean",
            "p24_route_no_prompt_mean",
            "p24_route_stage_active",
            "p24_route_candidate_detached",
            "p24_route_context_detached",
            "p24_train_edge_homophily_raw",
            "p24_train_edge_homophily_weighted",
            "p24_train_same_class_gate_mean",
            "p24_train_offclass_gate_mean",
            "same_class_route_compactness",
            "different_class_route_separation",
            "prompt_usage_entropy",
            "prompt_usage_full_entropy",
            "pattern_prompt_usage_entropy",
            "dominant_prompt_slot_ratio",
            "active_prompt_slot_count@0.05",
            "prompt_msg_norm",
            "prompt_to_original_msg_norm",
            "node_to_prompt_msg_norm",
            "prompt_to_original_update_norm",
            "correction_norm",
            "node_to_prompt_update_norm",
            "raw_prompt_update_norm",
            "unbounded_prompt_update_norm",
            "prompt_update_clip_ratio",
            "bounded_correction_clip_ratio",
            "prompt_gate_mean",
            "prompt_gate_node_to_prompt_mean",
            "prompt_gate_prompt_to_node_mean",
            "prompt_receiver_gate_mean",
            "prompt_message_scale",
            "selected_message_scale",
            "prompt_message_norm",
            "bounded_prompt_update",
            "max_prompt_update_norm",
            "prompt_update_bound_mode",
            "receiver_version",
            "prompt_fusion",
            "prompt_slot_head_count",
            "prototype_direction_strength_mean",
            "prototype_direction_normalize",
            "pool_only_prompt_update",
            "zero_init_prompt_messages",
            "adapted_branch_delta_norm",
            "capacity_routing_enabled",
            "prompt_capacity",
            "capacity_overflow_count",
            "edge_type_counts",
            "init_original_x_delta",
            "init_logit_delta",
            "init_logit_delta_full_edge_scale",
            "init_prompt_adapter_update_norm",
            "prompt_adapter_enabled",
            "prompt_adapter_update_norm",
            "prompt_adapter_update_max_norm",
            "prompt_adapter_raw_delta_norm",
            "prompt_adapter_delta_norm",
            "prompt_adapter_gate_mean",
            "prompt_adapter_gate_min",
            "prompt_adapter_gate_max",
            "prompt_adapter_raw_gate_mean",
            "prompt_adapter_raw_gate_min",
            "prompt_adapter_raw_gate_max",
            "prompt_adapter_update_mask_ratio",
            "prompt_adapter_clip_ratio",
            "high_frequency_norm",
            "low_frequency_norm",
            "adapter_query_mean_delta_ce",
            "adapter_query_positive_delta_ratio",
            "adapter_query_mean_ce_no_prompt",
            "adapter_query_mean_ce_prompt",
            "adapter_query_count",
            "adapter_train_mean_delta_ce",
            "adapter_train_positive_delta_ratio",
            "adapter_val_mean_delta_ce",
            "adapter_val_positive_delta_ratio",
            "adapter_test_mean_delta_ce",
            "adapter_test_positive_delta_ratio",
            "prompt_adapter_update_norm_loss",
            "prompt_adapter_gate_budget_loss",
            "prompt_adapter_gate_consistency_loss",
            "prompt_adapter_delta_consistency_loss",
            "prompt_adapter_episode_count",
            "support_context_enabled",
            "support_context_available",
            "support_context_coverage",
            "support_context_count",
            "support_similarity_margin",
            "support_similarity_entropy",
            "support_reliability_mean",
            "support_reliability_min",
            "support_reliability_max",
            "support_topk_mean_score",
            "prompt_adapter_message_help_loss",
            "prompt_adapter_message_help_mean_delta_ce",
            "prompt_adapter_message_help_positive_ratio",
            "prompt_adapter_message_help_count",
            "prompt_adapter_message_help_anti_harm_loss",
            "prompt_adapter_utility_gate_loss",
            "prompt_adapter_utility_gate_target_mean",
            "prompt_adapter_utility_gate_count",
            "prompt_adapter_utility_gate_delta_mean",
            "prompt_adapter_utility_gate_positive_ratio",
            "early_stopped",
            "stopped_epoch",
            "diagnostic_message_scale",
            "utility_mean_delta_ce",
            "utility_positive_delta_ratio",
            "utility_mean_ce_no_prompt",
            "utility_mean_ce_prompt",
            "utility_mean_delta_ce_train_pool",
            "utility_positive_delta_ratio_train_pool",
            "utility_mean_delta_ce_val_pool",
            "utility_positive_delta_ratio_val_pool",
            "utility_mean_delta_ce_test_pool",
            "utility_positive_delta_ratio_test_pool",
            "utility_acceptance_delta_correlation",
            "utility_structural_score_delta_correlation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "base_seed": result.get("base_seed", result.get("split_seed", result["seed"])),
                    "split_seed": result.get("split_seed", result["seed"]),
                    "trial_index": result.get("trial_index", 0),
                    "pretraining_strategy": result.get("pretraining_strategy", "GRACE"),
                    "checkpoint_path": result.get("checkpoint_path", ""),
                    "pretrained_backbone_type": result.get("pretrained_backbone_type", "BaseGCN"),
                    "pretrained_backbone_uses_edge_weight": result.get(
                        "pretrained_backbone_uses_edge_weight", True
                    ),
                    "graphmae_edge_weight_mode": result.get("graphmae_edge_weight_mode", "native_gcn"),
                    "graphmae_edge_weight_strength": result.get("graphmae_edge_weight_strength", 1.0),
                    "test_label_cheat_enabled": result.get("test_label_cheat_enabled", False),
                    "test_label_cheat_fraction": result.get("test_label_cheat_fraction", 0.0),
                    "actual_test_label_cheat_fraction": result.get("actual_test_label_cheat_fraction", 0.0),
                    "test_label_cheat_count": result.get("test_label_cheat_count", 0),
                    "label_train_count": result.get("label_train_count", 0),
                    "best_epoch": result["best"].get("best_epoch", 0.0),
                    "best_test_acc": result["best"].get("test_acc", 0.0),
                    "best_test_macro_f1": result["best"].get("test_macro_f1", 0.0),
                    "best_test_auroc": result["best"].get("test_auroc", 0.0),
                    "best_test_auprc": result["best"].get("test_auprc", 0.0),
                    "best_val_acc_epoch": result.get("best_by_val_acc", {}).get("epoch", 0.0),
                    "best_val_acc": result.get("best_by_val_acc", {}).get("val_acc", 0.0),
                    "test_acc_at_best_val_acc": result.get("best_by_val_acc", {}).get("test_acc", 0.0),
                    "min_train_loss_epoch": result.get("best_by_train_loss", {}).get("epoch", 0.0),
                    "min_train_loss": result.get("best_by_train_loss", {}).get("train_loss", 0.0),
                    "test_acc_at_min_train_loss": result.get("best_by_train_loss", {}).get("test_acc", 0.0),
                    "oracle_max_test_acc_epoch": result.get("oracle_best_by_test_acc", {}).get("epoch", 0.0),
                    "oracle_max_observed_test_acc": result.get("oracle_best_by_test_acc", {}).get("test_acc", 0.0),
                    "safe_checkpoint_found": result.get("safe", {}).get("p22_safe_checkpoint_found", 0.0),
                    "safe_test_acc": result.get("safe", {}).get("test_acc", 0.0),
                    "safe_test_macro_f1": result.get("safe", {}).get("test_macro_f1", 0.0),
                    "safe_test_auroc": result.get("safe", {}).get("test_auroc", 0.0),
                    "safe_test_auprc": result.get("safe", {}).get("test_auprc", 0.0),
                    "safe_val_delta_ce": result.get("safe", {}).get("p22_safe_val_delta_ce", 0.0),
                    "final_test_acc": result["final"].get("test_acc", 0.0),
                    "final_test_macro_f1": result["final"].get("test_macro_f1", 0.0),
                    "final_test_auroc": result["final"].get("test_auroc", 0.0),
                    "final_test_auprc": result["final"].get("test_auprc", 0.0),
                    "alpha": result["best"].get("alpha", 0.0),
                    "branch_cosine": result["best"].get("branch_cosine", 0.0),
                    "rho": result.get("rho", 0.0),
                    "pool_ratio": result["best"].get("pool_ratio", 0.0),
                    "train_pool_ratio": result["best"].get("train_pool_ratio", 0.0),
                    "candidate_pool_enabled": result["best"].get("candidate_pool_enabled", 0.0),
                    "candidate_pool_ratio": result["best"].get("candidate_pool_ratio", 0.0),
                    "candidate_pool_count": result["best"].get("candidate_pool_count", 0.0),
                    "candidate_pool_topk_count": result["best"].get("candidate_pool_topk_count", 0.0),
                    "candidate_pool_score_mean": result["best"].get("candidate_pool_score_mean", 0.0),
                    "candidate_pool_score_selected_mean": result["best"].get("candidate_pool_score_selected_mean", 0.0),
                    "candidate_pool_structural_mean": result["best"].get("candidate_pool_structural_mean", 0.0),
                    "candidate_pool_uncertainty_mean": result["best"].get("candidate_pool_uncertainty_mean", 0.0),
                    "candidate_pool_disagreement_mean": result["best"].get("candidate_pool_disagreement_mean", 0.0),
                    "adapter_query_candidate_pool_mean_delta_ce": result["best"].get(
                        "adapter_query_candidate_pool_mean_delta_ce", 0.0
                    ),
                    "adapter_query_candidate_pool_positive_delta_ratio": result["best"].get(
                        "adapter_query_candidate_pool_positive_delta_ratio", 0.0
                    ),
                    "adapter_query_outside_candidate_pool_mean_delta_ce": result["best"].get(
                        "adapter_query_outside_candidate_pool_mean_delta_ce", 0.0
                    ),
                    "adapter_query_outside_candidate_pool_positive_delta_ratio": result["best"].get(
                        "adapter_query_outside_candidate_pool_positive_delta_ratio", 0.0
                    ),
                    "adapter_test_candidate_pool_mean_delta_ce": result["best"].get(
                        "adapter_test_candidate_pool_mean_delta_ce", 0.0
                    ),
                    "adapter_test_candidate_pool_positive_delta_ratio": result["best"].get(
                        "adapter_test_candidate_pool_positive_delta_ratio", 0.0
                    ),
                    "adapter_test_outside_candidate_pool_mean_delta_ce": result["best"].get(
                        "adapter_test_outside_candidate_pool_mean_delta_ce", 0.0
                    ),
                    "adapter_test_outside_candidate_pool_positive_delta_ratio": result["best"].get(
                        "adapter_test_outside_candidate_pool_positive_delta_ratio", 0.0
                    ),
                    "prompt_node_count": result["best"].get("prompt_node_count", 0),
                    "prompt_edge_count": result["best"].get("prompt_edge_count", 0),
                    "edge_scale": result["best"].get("edge_scale", 0.0),
                    "raw_edge_scale": result["best"].get("raw_edge_scale", 0.0),
                    "edge_scale_multiplier": result["best"].get("edge_scale_multiplier", 0.0),
                    "use_multiview_routing": result["best"].get("use_multiview_routing", 0.0),
                    "use_class_aware_routing": result["best"].get("use_class_aware_routing", 0.0),
                    "use_attribute_view": result["best"].get("use_attribute_view", 0.0),
                    "use_enhanced_role_view": result["best"].get("use_enhanced_role_view", 0.0),
                    "use_pattern_prompt_bank": result["best"].get("use_pattern_prompt_bank", 0.0),
                    "use_receiver_only_prompt": result["best"].get("use_receiver_only_prompt", 0.0),
                    "use_benefit_gate": result["best"].get("use_benefit_gate", 0.0),
                    "num_class_prompt_slots": result["best"].get("num_class_prompt_slots", 0.0),
                    "residual_prompt_count": result["best"].get("residual_prompt_count", 0.0),
                    "class_key_proto_init_coverage": result["best"].get("class_key_proto_init_coverage", 0.0),
                    "pattern_key_init_coverage": result["best"].get("pattern_key_init_coverage", 0.0),
                    "semantic_view_weight": result["best"].get("semantic_view_weight", 0.0),
                    "structural_view_weight": result["best"].get("structural_view_weight", 0.0),
                    "role_view_weight": result["best"].get("role_view_weight", 0.0),
                    "attribute_view_weight": result["best"].get("attribute_view_weight", 0.0),
                    "view_gate_entropy": result["best"].get("view_gate_entropy", 0.0),
                    "semantic_route_margin": result["best"].get("semantic_route_margin", 0.0),
                    "structural_route_margin": result["best"].get("structural_route_margin", 0.0),
                    "role_route_margin": result["best"].get("role_route_margin", 0.0),
                    "attribute_route_margin": result["best"].get("attribute_route_margin", 0.0),
                    "prompt_view_prior": result["best"].get("prompt_view_prior", 0.0),
                    "mean_prompt_edge_weight": result["best"].get("mean_prompt_edge_weight", 0.0),
                    "pool_acceptance_mean": result["best"].get("pool_acceptance_mean", 1.0),
                    "pool_acceptance_min": result["best"].get("pool_acceptance_min", 1.0),
                    "pool_acceptance_max": result["best"].get("pool_acceptance_max", 1.0),
                    "use_hard_acceptance": result["best"].get("use_hard_acceptance", 0.0),
                    "hard_acceptance_ratio": result["best"].get("hard_acceptance_ratio", 0.0),
                    "hard_acceptance_selected_ratio": result["best"].get("hard_acceptance_selected_ratio", 0.0),
                    "prompt_acceptance_budget": result["best"].get("prompt_acceptance_budget", 0.0),
                    "prompt_acceptance_supervision": result["best"].get("prompt_acceptance_supervision", 0.0),
                    "acceptance_supervised_count": result["best"].get("acceptance_supervised_count", 0.0),
                    "acceptance_positive_count": result["best"].get("acceptance_positive_count", 0.0),
                    "acceptance_negative_count": result["best"].get("acceptance_negative_count", 0.0),
                    "acceptance_ignored_count": result["best"].get("acceptance_ignored_count", 0.0),
                    "acceptance_target_mean": result["best"].get("acceptance_target_mean", 0.0),
                    "acceptance_score_delta_mean": result["best"].get("acceptance_score_delta_mean", 0.0),
                    "ce_delta_positive_ratio_train": result["best"].get("ce_delta_positive_ratio_train", 0.0),
                    "ce_delta_negative_ratio_train": result["best"].get("ce_delta_negative_ratio_train", 0.0),
                    "prompt_helpful_acceptance_precision_train": result["best"].get(
                        "prompt_helpful_acceptance_precision_train", 0.0
                    ),
                    "prompt_harmful_rejection_precision_train": result["best"].get(
                        "prompt_harmful_rejection_precision_train", 0.0
                    ),
                    "prompt_benefit_supervision": result["best"].get("prompt_benefit_supervision", 0.0),
                    "benefit_supervised_count": result["best"].get("benefit_supervised_count", 0.0),
                    "benefit_positive_count": result["best"].get("benefit_positive_count", 0.0),
                    "benefit_negative_count": result["best"].get("benefit_negative_count", 0.0),
                    "benefit_ignored_count": result["best"].get("benefit_ignored_count", 0.0),
                    "mean_delta_ce_train_pool": result["best"].get("mean_delta_ce_train_pool", 0.0),
                    "positive_delta_ratio_train_pool": result["best"].get("positive_delta_ratio_train_pool", 0.0),
                    "benefit_weight_delta_corr_train": result["best"].get("benefit_weight_delta_corr_train", 0.0),
                    "benefit_label_strategy_quantile": result["best"].get("benefit_label_strategy_quantile", 0.0),
                    "benefit_quantile_fraction": result["best"].get("benefit_quantile_fraction", 0.0),
                    "benefit_positive_threshold": result["best"].get("benefit_positive_threshold", 0.0),
                    "benefit_negative_threshold": result["best"].get("benefit_negative_threshold", 0.0),
                    "benefit_delta_eps": result["best"].get("benefit_delta_eps", 0.0),
                    "prompt_correction_loss": result["best"].get("prompt_correction_loss", 0.0),
                    "prompt_anti_harm_loss": result["best"].get("prompt_anti_harm_loss", 0.0),
                    "prompt_correction_supervised_count": result["best"].get("prompt_correction_supervised_count", 0.0),
                    "prompt_correction_harmful_count": result["best"].get("prompt_correction_harmful_count", 0.0),
                    "prompt_correction_target_norm": result["best"].get("prompt_correction_target_norm", 0.0),
                    "prompt_delta_logit_norm": result["best"].get("prompt_delta_logit_norm", 0.0),
                    "prompt_delta_logit_label_alignment": result["best"].get("prompt_delta_logit_label_alignment", 0.0),
                    "harmful_prompt_delta_ratio": result["best"].get("harmful_prompt_delta_ratio", 0.0),
                    "prompt_message_help_loss": result["best"].get("prompt_message_help_loss", 0.0),
                    "prompt_message_help_query_loss": result["best"].get("prompt_message_help_query_loss", 0.0),
                    "prompt_class_anti_harm_loss": result["best"].get("prompt_class_anti_harm_loss", 0.0),
                    "class_balanced_mean_delta_ce": result["best"].get("class_balanced_mean_delta_ce", 0.0),
                    "class_balanced_positive_delta_ratio": result["best"].get(
                        "class_balanced_positive_delta_ratio", 0.0
                    ),
                    "query_class_balanced_mean_delta_ce": result["best"].get(
                        "query_class_balanced_mean_delta_ce", 0.0
                    ),
                    "query_class_balanced_positive_delta_ratio": result["best"].get(
                        "query_class_balanced_positive_delta_ratio", 0.0
                    ),
                    "message_help_supervised_count": result["best"].get("message_help_supervised_count", 0.0),
                    "message_help_query_supervised_count": result["best"].get(
                        "message_help_query_supervised_count", 0.0
                    ),
                    "message_help_class_count": result["best"].get("message_help_class_count", 0.0),
                    "message_help_query_class_count": result["best"].get("message_help_query_class_count", 0.0),
                    "message_help_margin": result["best"].get("message_help_margin", 0.0),
                    "message_help_class_balanced": result["best"].get("message_help_class_balanced", 0.0),
                    "prompt_class_anti_harm_floor": result["best"].get("prompt_class_anti_harm_floor", 0.0),
                    "utility_receive_gate_loss": result["best"].get("utility_receive_gate_loss", 0.0),
                    "utility_receive_gate_query_loss": result["best"].get("utility_receive_gate_query_loss", 0.0),
                    "receive_gate_budget": result["best"].get("receive_gate_budget", 0.0),
                    "utility_gate_supervised_count": result["best"].get("utility_gate_supervised_count", 0.0),
                    "utility_gate_query_supervised_count": result["best"].get(
                        "utility_gate_query_supervised_count", 0.0
                    ),
                    "utility_gate_positive_count": result["best"].get("utility_gate_positive_count", 0.0),
                    "utility_gate_query_positive_count": result["best"].get(
                        "utility_gate_query_positive_count", 0.0
                    ),
                    "utility_gate_negative_count": result["best"].get("utility_gate_negative_count", 0.0),
                    "utility_gate_query_negative_count": result["best"].get(
                        "utility_gate_query_negative_count", 0.0
                    ),
                    "utility_gate_ignored_count": result["best"].get("utility_gate_ignored_count", 0.0),
                    "utility_gate_query_ignored_count": result["best"].get(
                        "utility_gate_query_ignored_count", 0.0
                    ),
                    "utility_gate_target_mean": result["best"].get("utility_gate_target_mean", 0.0),
                    "utility_gate_query_target_mean": result["best"].get("utility_gate_query_target_mean", 0.0),
                    "utility_gate_delta_corr_train": result["best"].get("utility_gate_delta_corr_train", 0.0),
                    "utility_gate_query_delta_corr": result["best"].get("utility_gate_query_delta_corr", 0.0),
                    "utility_gate_label_strategy": result["best"].get("utility_gate_label_strategy", ""),
                    "utility_gate_query_label_strategy": result["best"].get("utility_gate_query_label_strategy", ""),
                    "query_proto_alignment_loss": result["best"].get("query_proto_alignment_loss", 0.0),
                    "query_proto_supervised_count": result["best"].get("query_proto_supervised_count", 0.0),
                    "query_proto_class_count": result["best"].get("query_proto_class_count", 0.0),
                    "query_proto_mean_delta_dist": result["best"].get("query_proto_mean_delta_dist", 0.0),
                    "query_proto_positive_ratio": result["best"].get("query_proto_positive_ratio", 0.0),
                    "correction_alignment_loss": result["best"].get("correction_alignment_loss", 0.0),
                    "correction_alignment_anti_harm_loss": result["best"].get(
                        "correction_alignment_anti_harm_loss", 0.0
                    ),
                    "correction_alignment_node_count": result["best"].get("correction_alignment_node_count", 0.0),
                    "correction_alignment_harmful_count": result["best"].get(
                        "correction_alignment_harmful_count", 0.0
                    ),
                    "correction_alignment_cosine_mean": result["best"].get(
                        "correction_alignment_cosine_mean", 0.0
                    ),
                    "correction_alignment_delta_h_norm": result["best"].get(
                        "correction_alignment_delta_h_norm", 0.0
                    ),
                    "correction_alignment_delta_ce_mean": result["best"].get(
                        "correction_alignment_delta_ce_mean", 0.0
                    ),
                    "support_query_enabled": result["best"].get("support_query_enabled", 0.0),
                    "support_only_prompt_graph": result["best"].get("support_only_prompt_graph", 0.0),
                    "support_count": result["best"].get("support_count", 0.0),
                    "query_count": result["best"].get("query_count", 0.0),
                    "train_support_ratio": result["best"].get("train_support_ratio", 1.0),
                    "use_utility_receive_gate": result["best"].get("use_utility_receive_gate", 0.0),
                    "utility_receive_gate_mean": result["best"].get("utility_receive_gate_mean", 1.0),
                    "utility_receive_gate_min": result["best"].get("utility_receive_gate_min", 1.0),
                    "utility_receive_gate_max": result["best"].get("utility_receive_gate_max", 1.0),
                    "utility_receive_gate_floor": result["best"].get("utility_receive_gate_floor", 0.0),
                    "benefit_gate_mean": result["best"].get("benefit_gate_mean", 1.0),
                    "benefit_gate_min": result["best"].get("benefit_gate_min", 1.0),
                    "benefit_gate_max": result["best"].get("benefit_gate_max", 1.0),
                    "use_edge_utility": result["best"].get("use_edge_utility", 0.0),
                    "edge_utility_mean": result["best"].get("edge_utility_mean", 1.0),
                    "edge_utility_min": result["best"].get("edge_utility_min", 1.0),
                    "edge_utility_max": result["best"].get("edge_utility_max", 1.0),
                    "edge_utility_supervision_loss": result["best"].get("edge_utility_supervision_loss", 0.0),
                    "edge_utility_supervised_count": result["best"].get("edge_utility_supervised_count", 0.0),
                    "edge_utility_positive_count": result["best"].get("edge_utility_positive_count", 0.0),
                    "edge_utility_negative_count": result["best"].get("edge_utility_negative_count", 0.0),
                    "edge_utility_ignored_count": result["best"].get("edge_utility_ignored_count", 0.0),
                    "edge_utility_target_mean": result["best"].get("edge_utility_target_mean", 0.0),
                    "edge_utility_delta_ce_mean": result["best"].get("edge_utility_delta_ce_mean", 0.0),
                    "edge_utility_delta_ce_positive_ratio": result["best"].get(
                        "edge_utility_delta_ce_positive_ratio", 0.0
                    ),
                    "edge_utility_delta_corr_train": result["best"].get("edge_utility_delta_corr_train", 0.0),
                    "pool_strategy_id": result["best"].get("pool_strategy_id", 0.0),
                    "pool_selected_ratio": result["best"].get("pool_selected_ratio", 0.0),
                    "pool_score_mean": result["best"].get("pool_score_mean", 0.0),
                    "pool_score_selected_mean": result["best"].get("pool_score_selected_mean", 0.0),
                    "pool_uncertainty_mean": result["best"].get("pool_uncertainty_mean", 0.0),
                    "pool_disagreement_mean": result["best"].get("pool_disagreement_mean", 0.0),
                    "use_hard_receive_gate": result["best"].get("use_hard_receive_gate", 0.0),
                    "hard_receive_ratio": result["best"].get("hard_receive_ratio", 0.0),
                    "hard_receive_selected_ratio": result["best"].get("hard_receive_selected_ratio", 0.0),
                    "receive_gate_mean": result["best"].get("receive_gate_mean", 1.0),
                    "receive_gate_min": result["best"].get("receive_gate_min", 1.0),
                    "receive_gate_max": result["best"].get("receive_gate_max", 1.0),
                    "class_route": result["best"].get("class_route", 0.0),
                    "key_proto": result["best"].get("key_proto", 0.0),
                    "class_router_hit_rate_train": result["best"].get("class_router_hit_rate_train", 0.0),
                    "class_router_hit_rate_val": result["best"].get("class_router_hit_rate_val", 0.0),
                    "class_router_hit_rate_test": result["best"].get("class_router_hit_rate_test", 0.0),
                    "prompt_label_purity_train": result["best"].get("prompt_label_purity_train", 0.0),
                    "residual_prompt_usage_ratio": result["best"].get("residual_prompt_usage_ratio", 0.0),
                    "p23_prompt_residual_enabled": result["best"].get("p23_prompt_residual_enabled", 0.0),
                    "p23_prompt_residual_delta_norm_loss": result["best"].get(
                        "p23_prompt_residual_delta_norm_loss", 0.0
                    ),
                    "p23_prompt_residual_delta_align_loss": result["best"].get(
                        "p23_prompt_residual_delta_align_loss", 0.0
                    ),
                    "p23_prompt_residual_gate_budget_loss": result["best"].get(
                        "p23_prompt_residual_gate_budget_loss", 0.0
                    ),
                    "p23_prompt_residual_delta_norm_mean": result["best"].get(
                        "p23_prompt_residual_delta_norm_mean", 0.0
                    ),
                    "p23_prompt_residual_delta_norm_max": result["best"].get(
                        "p23_prompt_residual_delta_norm_max", 0.0
                    ),
                    "p23_prompt_residual_delta_to_init_ratio": result["best"].get(
                        "p23_prompt_residual_delta_to_init_ratio", 0.0
                    ),
                    "p23_prompt_residual_delta_cosine_to_init": result["best"].get(
                        "p23_prompt_residual_delta_cosine_to_init", 0.0
                    ),
                    "p23_prompt_residual_gate_mean": result["best"].get(
                        "p23_prompt_residual_gate_mean", 0.0
                    ),
                    "p23_prompt_residual_gate_max": result["best"].get("p23_prompt_residual_gate_max", 0.0),
                    "p23_prompt_residual_gate_reliability_corr": result["best"].get(
                        "p23_prompt_residual_gate_reliability_corr", 0.0
                    ),
                    "p23_prompt_residual_coeff_abs_mean": result["best"].get(
                        "p23_prompt_residual_coeff_abs_mean", 0.0
                    ),
                    "p23_prompt_residual_coeff_entropy": result["best"].get(
                        "p23_prompt_residual_coeff_entropy", 0.0
                    ),
                    "p23_prompt_residual_clip_fraction": result["best"].get(
                        "p23_prompt_residual_clip_fraction", 0.0
                    ),
                    "p23_prompt_residual_reliability_mean": result["best"].get(
                        "p23_prompt_residual_reliability_mean", 0.0
                    ),
                    "p23_prompt_residual_reliability_active_ratio": result["best"].get(
                        "p23_prompt_residual_reliability_active_ratio", 0.0
                    ),
                    "p24_enabled": result["best"].get("p24_enabled", 0.0),
                    "p24_keep_static_loss": result["best"].get("p24_keep_static_loss", 0.0),
                    "p24_hub_suppression_loss": result["best"].get("p24_hub_suppression_loss", 0.0),
                    "p24_prompt_homophily_loss": result["best"].get("p24_prompt_homophily_loss", 0.0),
                    "p24_edge_budget_loss": result["best"].get("p24_edge_budget_loss", 0.0),
                    "p24_edge_gate_mean": result["best"].get("p24_edge_gate_mean", 1.0),
                    "p24_edge_gate_min": result["best"].get("p24_edge_gate_min", 1.0),
                    "p24_edge_gate_max": result["best"].get("p24_edge_gate_max", 1.0),
                    "p24_prompt_to_node_gate_mean": result["best"].get("p24_prompt_to_node_gate_mean", 1.0),
                    "p24_node_to_prompt_gate_mean": result["best"].get("p24_node_to_prompt_gate_mean", 1.0),
                    "p24_hub_gate_mean": result["best"].get("p24_hub_gate_mean", 0.0),
                    "p24_nonhub_gate_mean": result["best"].get("p24_nonhub_gate_mean", 0.0),
                    "p24_edge_keep_ratio": result["best"].get("p24_edge_keep_ratio", 1.0),
                    "p24_learned_gate_mean": result["best"].get("p24_learned_gate_mean", 1.0),
                    "p24_effective_gate_mean": result["best"].get("p24_effective_gate_mean", 1.0),
                    "p24_mix_ratio": result["best"].get("p24_mix_ratio", 0.0),
                    "p24_gate_above_one_ratio": result["best"].get("p24_gate_above_one_ratio", 0.0),
                    "p24_effective_gate_below_0p3_ratio": result["best"].get(
                        "p24_effective_gate_below_0p3_ratio", 0.0
                    ),
                    "p24_effective_gate_below_0p5_ratio": result["best"].get(
                        "p24_effective_gate_below_0p5_ratio", 0.0
                    ),
                    "p24_effective_gate_above_one_ratio": result["best"].get(
                        "p24_effective_gate_above_one_ratio", 0.0
                    ),
                    "p24_within_prompt_gate_std_mean": result["best"].get(
                        "p24_within_prompt_gate_std_mean", 0.0
                    ),
                    "p24_weight_delta_mean": result["best"].get("p24_weight_delta_mean", 0.0),
                    "p24_weight_delta_abs_mean": result["best"].get("p24_weight_delta_abs_mean", 0.0),
                    "p24_candidate_confidence_mean": result["best"].get(
                        "p24_candidate_confidence_mean", 0.0
                    ),
                    "p24_candidate_gate_mean": result["best"].get("p24_candidate_gate_mean", 1.0),
                    "p24_candidate_gate_min": result["best"].get("p24_candidate_gate_min", 1.0),
                    "p24_candidate_gate_max": result["best"].get("p24_candidate_gate_max", 1.0),
                    "p24_candidate_weight_delta_abs_mean": result["best"].get(
                        "p24_candidate_weight_delta_abs_mean", 0.0
                    ),
                    "p24_candidate_gate_clamp_ratio": result["best"].get(
                        "p24_candidate_gate_clamp_ratio", 0.0
                    ),
                    "p24_candidate_node_l1_mean": result["best"].get("p24_candidate_node_l1_mean", 0.0),
                    "p24_candidate_node_l1_max": result["best"].get("p24_candidate_node_l1_max", 0.0),
                    "p24_candidate_projection_enabled": result["best"].get(
                        "p24_candidate_projection_enabled", 0.0
                    ),
                    "p24_route_p23_mean": result["best"].get("p24_route_p23_mean", 1.0),
                    "p24_route_adapt_mean": result["best"].get("p24_route_adapt_mean", 0.0),
                    "p24_route_no_prompt_mean": result["best"].get("p24_route_no_prompt_mean", 0.0),
                    "p24_route_stage_active": result["best"].get("p24_route_stage_active", 0.0),
                    "p24_route_candidate_detached": result["best"].get(
                        "p24_route_candidate_detached", 0.0
                    ),
                    "p24_route_context_detached": result["best"].get("p24_route_context_detached", 0.0),
                    "p24_train_edge_homophily_raw": result["best"].get("p24_train_edge_homophily_raw", 0.0),
                    "p24_train_edge_homophily_weighted": result["best"].get(
                        "p24_train_edge_homophily_weighted", 0.0
                    ),
                    "p24_train_same_class_gate_mean": result["best"].get("p24_train_same_class_gate_mean", 0.0),
                    "p24_train_offclass_gate_mean": result["best"].get("p24_train_offclass_gate_mean", 0.0),
                    "same_class_route_compactness": result["best"].get("same_class_route_compactness", 0.0),
                    "different_class_route_separation": result["best"].get("different_class_route_separation", 0.0),
                    "prompt_usage_entropy": result["best"].get("prompt_usage_entropy", 0.0),
                    "prompt_usage_full_entropy": result["best"].get("prompt_usage_full_entropy", 0.0),
                    "pattern_prompt_usage_entropy": result["best"].get("pattern_prompt_usage_entropy", 0.0),
                    "dominant_prompt_slot_ratio": result["best"].get("dominant_prompt_slot_ratio", 0.0),
                    "active_prompt_slot_count@0.05": result["best"].get("active_prompt_slot_count@0.05", 0.0),
                    "prompt_msg_norm": result["best"].get("prompt_msg_norm", 0.0),
                    "prompt_to_original_msg_norm": result["best"].get("prompt_to_original_msg_norm", 0.0),
                    "node_to_prompt_msg_norm": result["best"].get("node_to_prompt_msg_norm", 0.0),
                    "prompt_to_original_update_norm": result["best"].get("prompt_to_original_update_norm", 0.0),
                    "correction_norm": result["best"].get("correction_norm", 0.0),
                    "node_to_prompt_update_norm": result["best"].get("node_to_prompt_update_norm", 0.0),
                    "raw_prompt_update_norm": result["best"].get("raw_prompt_update_norm", 0.0),
                    "unbounded_prompt_update_norm": result["best"].get("unbounded_prompt_update_norm", 0.0),
                    "prompt_update_clip_ratio": result["best"].get("prompt_update_clip_ratio", 0.0),
                    "bounded_correction_clip_ratio": result["best"].get("bounded_correction_clip_ratio", 0.0),
                    "prompt_gate_mean": result["best"].get("prompt_gate_mean", 0.0),
                    "prompt_gate_node_to_prompt_mean": result["best"].get("prompt_gate_node_to_prompt_mean", 0.0),
                    "prompt_gate_prompt_to_node_mean": result["best"].get("prompt_gate_prompt_to_node_mean", 0.0),
                    "prompt_receiver_gate_mean": result["best"].get("prompt_receiver_gate_mean", 0.0),
                    "prompt_message_scale": result["best"].get("prompt_message_scale", 0.0),
                    "selected_message_scale": result.get("selected_message_scale", result["best"].get("prompt_message_scale", 0.0)),
                    "prompt_message_norm": result["best"].get("prompt_message_norm", ""),
                    "bounded_prompt_update": result["best"].get("bounded_prompt_update", 0.0),
                    "max_prompt_update_norm": result["best"].get("max_prompt_update_norm", 0.0),
                    "prompt_update_bound_mode": result["best"].get("prompt_update_bound_mode", ""),
                    "receiver_version": result["best"].get("receiver_version", ""),
                    "prompt_fusion": result["best"].get("prompt_fusion", ""),
                    "prompt_slot_head_count": result["best"].get("prompt_slot_head_count", 0.0),
                    "prototype_direction_strength_mean": result["best"].get(
                        "prototype_direction_strength_mean", 0.0
                    ),
                    "prototype_direction_normalize": result["best"].get("prototype_direction_normalize", 0.0),
                    "pool_only_prompt_update": result["best"].get("pool_only_prompt_update", 0.0),
                    "zero_init_prompt_messages": result["best"].get("zero_init_prompt_messages", 0.0),
                    "adapted_branch_delta_norm": result["best"].get("adapted_branch_delta_norm", 0.0),
                    "capacity_routing_enabled": result["best"].get("capacity_routing_enabled", 0.0),
                    "prompt_capacity": result["best"].get("prompt_capacity", 0.0),
                    "capacity_overflow_count": result["best"].get("capacity_overflow_count", 0.0),
                    "edge_type_counts": result["best"].get("edge_type_counts", [0, 0, 0]),
                    "init_original_x_delta": result["best"].get("init_original_x_delta", 0.0),
                    "init_logit_delta": result["best"].get("init_logit_delta", 0.0),
                    "init_logit_delta_full_edge_scale": result["best"].get("init_logit_delta_full_edge_scale", 0.0),
                    "init_prompt_adapter_update_norm": result["best"].get("init_prompt_adapter_update_norm", 0.0),
                    "prompt_adapter_enabled": result["best"].get("prompt_adapter_enabled", 0.0),
                    "prompt_adapter_update_norm": result["best"].get("prompt_adapter_update_norm", 0.0),
                    "prompt_adapter_update_max_norm": result["best"].get("prompt_adapter_update_max_norm", 0.0),
                    "prompt_adapter_raw_delta_norm": result["best"].get("prompt_adapter_raw_delta_norm", 0.0),
                    "prompt_adapter_delta_norm": result["best"].get("prompt_adapter_delta_norm", 0.0),
                    "prompt_adapter_gate_mean": result["best"].get("prompt_adapter_gate_mean", 0.0),
                    "prompt_adapter_gate_min": result["best"].get("prompt_adapter_gate_min", 0.0),
                    "prompt_adapter_gate_max": result["best"].get("prompt_adapter_gate_max", 0.0),
                    "prompt_adapter_raw_gate_mean": result["best"].get("prompt_adapter_raw_gate_mean", 0.0),
                    "prompt_adapter_raw_gate_min": result["best"].get("prompt_adapter_raw_gate_min", 0.0),
                    "prompt_adapter_raw_gate_max": result["best"].get("prompt_adapter_raw_gate_max", 0.0),
                    "prompt_adapter_update_mask_ratio": result["best"].get("prompt_adapter_update_mask_ratio", 0.0),
                    "prompt_adapter_clip_ratio": result["best"].get("prompt_adapter_clip_ratio", 0.0),
                    "high_frequency_norm": result["best"].get("high_frequency_norm", 0.0),
                    "low_frequency_norm": result["best"].get("low_frequency_norm", 0.0),
                    "adapter_query_mean_delta_ce": result["best"].get("adapter_query_mean_delta_ce", 0.0),
                    "adapter_query_positive_delta_ratio": result["best"].get(
                        "adapter_query_positive_delta_ratio", 0.0
                    ),
                    "adapter_query_mean_ce_no_prompt": result["best"].get("adapter_query_mean_ce_no_prompt", 0.0),
                    "adapter_query_mean_ce_prompt": result["best"].get("adapter_query_mean_ce_prompt", 0.0),
                    "adapter_query_count": result["best"].get("adapter_query_count", 0.0),
                    "adapter_train_mean_delta_ce": result["best"].get("adapter_train_mean_delta_ce", 0.0),
                    "adapter_train_positive_delta_ratio": result["best"].get(
                        "adapter_train_positive_delta_ratio", 0.0
                    ),
                    "adapter_val_mean_delta_ce": result["best"].get("adapter_val_mean_delta_ce", 0.0),
                    "adapter_val_positive_delta_ratio": result["best"].get("adapter_val_positive_delta_ratio", 0.0),
                    "adapter_test_mean_delta_ce": result["best"].get("adapter_test_mean_delta_ce", 0.0),
                    "adapter_test_positive_delta_ratio": result["best"].get(
                        "adapter_test_positive_delta_ratio", 0.0
                    ),
                    "prompt_adapter_update_norm_loss": result["best"].get("prompt_adapter_update_norm_loss", 0.0),
                    "prompt_adapter_gate_budget_loss": result["best"].get("prompt_adapter_gate_budget_loss", 0.0),
                    "prompt_adapter_gate_consistency_loss": result["best"].get(
                        "prompt_adapter_gate_consistency_loss", 0.0
                    ),
                    "prompt_adapter_delta_consistency_loss": result["best"].get(
                        "prompt_adapter_delta_consistency_loss", 0.0
                    ),
                    "prompt_adapter_episode_count": result["best"].get("prompt_adapter_episode_count", 0.0),
                    "support_context_enabled": result["best"].get("support_context_enabled", 0.0),
                    "support_context_available": result["best"].get("support_context_available", 0.0),
                    "support_context_coverage": result["best"].get("support_context_coverage", 0.0),
                    "support_context_count": result["best"].get("support_context_count", 0.0),
                    "support_similarity_margin": result["best"].get("support_similarity_margin", 0.0),
                    "support_similarity_entropy": result["best"].get("support_similarity_entropy", 0.0),
                    "support_reliability_mean": result["best"].get("support_reliability_mean", 0.0),
                    "support_reliability_min": result["best"].get("support_reliability_min", 0.0),
                    "support_reliability_max": result["best"].get("support_reliability_max", 0.0),
                    "support_topk_mean_score": result["best"].get("support_topk_mean_score", 0.0),
                    "prompt_adapter_message_help_loss": result["best"].get(
                        "prompt_adapter_message_help_loss", 0.0
                    ),
                    "prompt_adapter_message_help_mean_delta_ce": result["best"].get(
                        "prompt_adapter_message_help_mean_delta_ce", 0.0
                    ),
                    "prompt_adapter_message_help_positive_ratio": result["best"].get(
                        "prompt_adapter_message_help_positive_ratio", 0.0
                    ),
                    "prompt_adapter_message_help_count": result["best"].get(
                        "prompt_adapter_message_help_count", 0.0
                    ),
                    "prompt_adapter_message_help_anti_harm_loss": result["best"].get(
                        "prompt_adapter_message_help_anti_harm_loss", 0.0
                    ),
                    "prompt_adapter_utility_gate_loss": result["best"].get(
                        "prompt_adapter_utility_gate_loss", 0.0
                    ),
                    "prompt_adapter_utility_gate_target_mean": result["best"].get(
                        "prompt_adapter_utility_gate_target_mean", 0.0
                    ),
                    "prompt_adapter_utility_gate_count": result["best"].get(
                        "prompt_adapter_utility_gate_count", 0.0
                    ),
                    "prompt_adapter_utility_gate_delta_mean": result["best"].get(
                        "prompt_adapter_utility_gate_delta_mean", 0.0
                    ),
                    "prompt_adapter_utility_gate_positive_ratio": result["best"].get(
                        "prompt_adapter_utility_gate_positive_ratio", 0.0
                    ),
                    "early_stopped": result.get("early_stopped", False),
                    "stopped_epoch": result.get("stopped_epoch", 0),
                    "diagnostic_message_scale": result.get("diagnostic_message_scale", ""),
                    "utility_mean_delta_ce": result.get("prompt_message_utility", {}).get("mean_delta_ce", ""),
                    "utility_positive_delta_ratio": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio", ""
                    ),
                    "utility_mean_ce_no_prompt": result.get("prompt_message_utility", {}).get("mean_ce_no_prompt", ""),
                    "utility_mean_ce_prompt": result.get("prompt_message_utility", {}).get("mean_ce_prompt", ""),
                    "utility_mean_delta_ce_train_pool": result.get("prompt_message_utility", {}).get(
                        "mean_delta_ce_train_pool", ""
                    ),
                    "utility_positive_delta_ratio_train_pool": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio_train_pool", ""
                    ),
                    "utility_mean_delta_ce_val_pool": result.get("prompt_message_utility", {}).get(
                        "mean_delta_ce_val_pool", ""
                    ),
                    "utility_positive_delta_ratio_val_pool": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio_val_pool", ""
                    ),
                    "utility_mean_delta_ce_test_pool": result.get("prompt_message_utility", {}).get(
                        "mean_delta_ce_test_pool", ""
                    ),
                    "utility_positive_delta_ratio_test_pool": result.get("prompt_message_utility", {}).get(
                        "positive_delta_ratio_test_pool", ""
                    ),
                    "utility_acceptance_delta_correlation": result.get("prompt_message_utility", {}).get(
                        "acceptance_delta_correlation", ""
                    ),
                    "utility_structural_score_delta_correlation": result.get("prompt_message_utility", {}).get(
                        "structural_score_delta_correlation", ""
                    ),
                }
            )
    print("=" * 72)
    print(
        f"Summary | dataset={target_dataset} | variant={variant} | "
        f"ablation={summary['component_ablation']} | "
        f"seeds={len(seeds)} | trials_per_seed={trials_per_seed} | runs={len(results)}"
    )
    print(f"Configured-Monitor Test Acc: {summary['best_test_acc_mean_std']}")
    print(f"Best Test Macro-F1: {summary['best_test_macro_f1_mean_std']}")
    print(f"Test Acc @ Best Val Acc:  {summary['test_acc_at_best_val_acc_mean_std']}")
    print(f"Test Acc @ Min Train Loss: {summary['test_acc_at_min_train_loss_mean_std']}")
    print(f"Oracle Max Test Acc*:     {summary['oracle_max_observed_test_acc_mean_std']}")
    print("* Diagnostic only; test labels are not used for checkpoint selection.")
    print(f"Safe Test Acc:      {summary['safe_test_acc_mean_std']}")
    print(f"Safe Found:         {summary['safe_checkpoint_found_mean_std']}")
    print(f"Final Test Acc:     {summary['final_test_acc_mean_std']}")
    print(f"Final Test Macro-F1: {summary['final_test_macro_f1_mean_std']}")
    print(
        f"All {len(results)} trials | test Accuracy {_format_gp2f_metric(final_test)} | "
        f"F1 {_format_gp2f_metric(final_macro_f1)} | "
        f"AUROC {_format_gp2f_metric(final_auroc)} | "
        f"AUPRC {_format_gp2f_metric(final_auprc)}"
    )
    print("-----------------")
    print(f"Saved summary to {summary_dir / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GP2F prompt graph P1 runner")
    parser.add_argument("--config", type=str, default="configs/gp2f_prompt_p1.yaml")
    parser.add_argument("--pretrained_checkpoint_path", type=str, default=None)
    parser.add_argument("--pretraining_strategy", type=str, choices=["GRACE", "DGI", "GraphMAE"], default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--gp2f_sample_data_root", type=str, default=None)
    parser.add_argument(
        "--graphmae_edge_weight_mode",
        type=str,
        choices=["ignore", "attention_prior"],
        default=None,
    )
    parser.add_argument("--graphmae_edge_weight_strength", type=float, default=None)
    parser.add_argument("--target_dataset", type=str, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--shot_ratio", type=float, default=None)
    parser.add_argument("--shot_mode", type=str, choices=["shots", "percent"], default=None)
    parser.add_argument("--shot_value", type=float, default=None)
    parser.add_argument("--disable_auto_shot_by_dataset", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--alpha_init", type=float, default=None)
    parser.add_argument("--fusion_alpha_lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--trials_per_seed", type=int, default=None)
    parser.add_argument("--reset_seed_per_trial", action="store_true")
    parser.add_argument("--no_reset_seed_per_trial", action="store_true")
    parser.add_argument("--fixed_split_seed", type=int, default=None)
    parser.add_argument("--trial_seed_mode", type=str, choices=sorted(TRIAL_SEED_MODES), default=None)
    parser.add_argument("--prompt_variant", type=str, choices=sorted(PROMPT_GRAPH_VARIANTS), default=None)
    parser.add_argument("--ablation", type=str, choices=sorted(COMPONENT_ABLATIONS), default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--p23_pool_rho", type=float, default=None)
    parser.add_argument("--p23_pool_rho_min", type=float, default=None)
    parser.add_argument("--p23_pool_rho_max", type=float, default=None)
    parser.add_argument("--num_prompt_nodes", type=int, default=None)
    parser.add_argument("--topk_prompt_per_node", type=int, default=None)
    parser.add_argument("--capacity_factor", type=float, default=None)
    parser.add_argument("--enable_capacity_routing", action="store_true")
    parser.add_argument("--disable_capacity_routing", action="store_true")
    parser.add_argument("--edge_scale_init", type=float, default=None)
    parser.add_argument("--edge_scale_max", type=float, default=None)
    parser.add_argument("--edge_scale_warmup_epochs", type=int, default=None)
    parser.add_argument("--edge_scale_warmup_start", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--prompt_gate_init", type=float, default=None)
    parser.add_argument("--node_to_prompt_gate_init", type=float, default=None)
    parser.add_argument("--prompt_to_node_gate_init", type=float, default=None)
    parser.add_argument(
        "--prompt_message_norm",
        type=str,
        choices=["weighted_mean", "weighted_sum", "degree_mean", "layernorm"],
        default=None,
    )
    parser.add_argument(
        "--receiver_version",
        type=str,
        choices=[
            "v1_linear",
            "v2_conditioned",
            "v3_node_residual",
            "v4_multi_expert_residual",
            "v5_prototype_directional",
            "v6_classifier_directional",
        ],
        default=None,
    )
    parser.add_argument("--prompt_message_scale", type=float, default=None)
    parser.add_argument("--prompt_message_scale_grid", type=str, default=None)
    parser.add_argument("--diagnose_prompt_message_utility", action="store_true")
    parser.add_argument("--diagnostic_message_scales", type=str, default=None)
    parser.add_argument("--message_scale_selection_metric", type=str, default=None)
    parser.add_argument("--zero_init_prompt_messages", action="store_true")
    parser.add_argument("--disable_zero_init_prompt_messages", action="store_true")
    parser.add_argument("--pool_only_prompt_update", action="store_true")
    parser.add_argument("--disable_pool_only_prompt_update", action="store_true")
    parser.add_argument("--enable_bounded_prompt_update", action="store_true")
    parser.add_argument("--disable_bounded_prompt_update", action="store_true")
    parser.add_argument("--max_prompt_update_norm", type=float, default=None)
    parser.add_argument("--prompt_adapter_max_update_norm", type=float, default=None)
    parser.add_argument("--prompt_adapter_gate_budget", type=float, default=None)
    parser.add_argument("--lambda_prompt_adapter_gate_budget", type=float, default=None)
    parser.add_argument("--lambda_prompt_adapter_utility_gate", type=float, default=None)
    parser.add_argument("--lambda_prompt_router_deployment_utility", type=float, default=None)
    parser.add_argument("--prompt_router_deployment_utility_margin", type=float, default=None)
    parser.add_argument("--prompt_router_deployment_utility_anti_harm_weight", type=float, default=None)
    parser.add_argument("--prompt_router_deployment_utility_anti_harm_margin", type=float, default=None)
    parser.add_argument("--prompt_router_deployment_utility_gain_reward_weight", type=float, default=None)
    parser.add_argument("--prompt_router_deployment_utility_gain_reward_cap", type=float, default=None)
    parser.add_argument("--prompt_router_expert_utility_margin", type=float, default=None)
    parser.add_argument("--prompt_router_expert_utility_target", type=str, default=None)
    parser.add_argument("--prompt_router_expert_utility_gain_temperature", type=float, default=None)
    parser.add_argument("--prompt_router_expert_utility_gate_weight", type=float, default=None)
    parser.add_argument("--prompt_router_expert_utility_gate_target", type=str, default=None)
    parser.add_argument("--prompt_router_expert_utility_gate_temperature", type=float, default=None)
    parser.add_argument("--candidate_pool_ratio", type=float, default=None)
    parser.add_argument("--prompt_update_bound_mode", type=str, choices=["norm_clip", "tanh"], default=None)
    parser.add_argument("--lambda_edge_l1", type=float, default=None)
    parser.add_argument("--lambda_prompt_balance", type=float, default=None)
    parser.add_argument("--lambda_p23_feature_harm", type=float, default=None)
    parser.add_argument("--lambda_p23_feature_gate_supervision", type=float, default=None)
    parser.add_argument("--enable_p23_node_feature_edge_gate", action="store_true")
    parser.add_argument("--disable_p23_node_feature_edge_gate", action="store_true")
    parser.add_argument("--p23_node_feature_edge_gate_strength", type=float, default=None)
    parser.add_argument("--p23_feature_topk", type=int, default=None)
    parser.add_argument("--p23_max_feature_nodes", type=int, default=None)
    parser.add_argument("--p23_max_df_global_ratio", type=float, default=None)
    parser.add_argument("--p23_feature_edge_weight", type=float, default=None)
    parser.add_argument("--p23_anti_hub_power", type=float, default=None)
    parser.add_argument(
        "--projector_sync_fusion",
        type=str,
        choices=["direct", "anchored_bootstrap", "centered_residual", "bank_standardized"],
        default=None,
    )
    parser.add_argument("--projector_sync_mix_init", type=float, default=None)
    parser.add_argument("--projector_sync_mix_max", type=float, default=None)
    parser.add_argument("--projector_sync_warmup_epochs", type=int, default=None)
    parser.add_argument(
        "--projector_sync_norm_mode",
        type=str,
        choices=["none", "cached_per_prompt", "cached_global"],
        default=None,
    )
    parser.add_argument("--projector_sync_max_delta_ratio", type=float, default=None)
    parser.add_argument("--enable_projector_sync_prompt_grad", action="store_true")
    parser.add_argument("--disable_projector_sync_prompt_grad", action="store_true")
    parser.add_argument("--p23_feature_harm_margin", type=float, default=None)
    parser.add_argument("--p23_feature_harm_temperature", type=float, default=None)
    parser.add_argument("--p23_feature_harm_quantile", type=float, default=None)
    parser.add_argument("--p23_feature_help_quantile", type=float, default=None)
    parser.add_argument("--p23_feature_min_signal", type=float, default=None)
    parser.add_argument("--p23_feature_help_weight", type=float, default=None)
    parser.add_argument("--p23_feature_harm_warmup_epochs", type=int, default=None)
    parser.add_argument("--enable_p23_static_pool", action="store_true")
    parser.add_argument("--disable_p23_static_pool", action="store_true")
    parser.add_argument("--p23_static_pool_strategy", type=str, default=None)
    parser.add_argument("--p23_static_pool_mode", type=str, choices=["hard", "soft"], default=None)
    parser.add_argument("--p23_static_pool_ratio", type=float, default=None)
    parser.add_argument("--p23_static_pool_core_ratio", type=float, default=None)
    parser.add_argument("--p23_static_pool_expand_ratio", type=float, default=None)
    parser.add_argument("--p23_static_pool_core_scale", type=float, default=None)
    parser.add_argument("--p23_static_pool_expand_scale", type=float, default=None)
    parser.add_argument("--p23_static_pool_non_pool_scale", type=float, default=None)
    parser.add_argument("--p23_static_pool_apply_to", type=str, choices=["prompt_to_node", "both"], default=None)
    parser.add_argument("--p23_static_pool_feature_topk", type=int, default=None)
    parser.add_argument("--lambda_prompt_usage_consistency", type=float, default=None)
    parser.add_argument("--lambda_prompt_view_entropy", type=float, default=None)
    parser.add_argument("--lambda_class_route", type=float, default=None)
    parser.add_argument("--lambda_key_proto", type=float, default=None)
    parser.add_argument("--lambda_prompt_benefit_supervision", type=float, default=None)
    parser.add_argument("--benefit_supervision_warmup_epochs", type=int, default=None)
    parser.add_argument("--benefit_delta_margin", type=float, default=None)
    parser.add_argument(
        "--benefit_supervision_label_strategy",
        type=str,
        choices=["margin", "quantile", "hybrid_quantile_margin"],
        default=None,
    )
    parser.add_argument("--benefit_supervision_quantile", type=float, default=None)
    parser.add_argument("--benefit_delta_eps", type=float, default=None)
    parser.add_argument("--benefit_supervision_quantile_warmup_epochs", type=int, default=None)
    parser.add_argument("--benefit_supervision_probe_scale", type=float, default=None)
    parser.add_argument("--lambda_prompt_correction", type=float, default=None)
    parser.add_argument("--lambda_prompt_anti_harm", type=float, default=None)
    parser.add_argument("--prompt_correction_warmup_epochs", type=int, default=None)
    parser.add_argument("--prompt_correction_eps", type=float, default=None)
    parser.add_argument("--prompt_correction_target", type=str, choices=["delta_prob_to_label"], default=None)
    parser.add_argument("--lambda_prompt_message_help", type=float, default=None)
    parser.add_argument("--prompt_message_help_margin", type=float, default=None)
    parser.add_argument("--prompt_message_help_warmup_epochs", type=int, default=None)
    parser.add_argument("--lambda_prompt_class_anti_harm", type=float, default=None)
    parser.add_argument("--prompt_class_anti_harm_floor", type=float, default=None)
    parser.add_argument("--lambda_utility_receive_gate", type=float, default=None)
    parser.add_argument("--utility_receive_gate_warmup_epochs", type=int, default=None)
    parser.add_argument("--utility_receive_gate_quantile", type=float, default=None)
    parser.add_argument("--utility_receive_gate_eps", type=float, default=None)
    parser.add_argument("--enable_utility_receive_gate", action="store_true")
    parser.add_argument("--disable_utility_receive_gate", action="store_true")
    parser.add_argument("--utility_receive_gate_min", type=float, default=None)
    parser.add_argument("--utility_receive_gate_init", type=float, default=None)
    parser.add_argument("--enable_prompt_message_help_class_balanced", action="store_true")
    parser.add_argument("--disable_prompt_message_help_class_balanced", action="store_true")
    parser.add_argument("--enable_hard_receive_gate", action="store_true")
    parser.add_argument("--disable_hard_receive_gate", action="store_true")
    parser.add_argument("--hard_receive_ratio", type=float, default=None)
    parser.add_argument("--residual_prompt_count", type=int, default=None)
    parser.add_argument("--enable_class_aware_routing", action="store_true")
    parser.add_argument("--disable_class_aware_routing", action="store_true")
    parser.add_argument("--enable_class_key_proto_init", action="store_true")
    parser.add_argument("--disable_class_key_proto_init", action="store_true")
    parser.add_argument("--lambda_prompt_acceptance_budget", type=float, default=None)
    parser.add_argument("--acceptance_budget_min", type=float, default=None)
    parser.add_argument("--acceptance_budget_max", type=float, default=None)
    parser.add_argument("--enable_hard_acceptance", action="store_true")
    parser.add_argument("--disable_hard_acceptance", action="store_true")
    parser.add_argument("--hard_acceptance_ratio", type=float, default=None)
    parser.add_argument("--lambda_prompt_acceptance_supervision", type=float, default=None)
    parser.add_argument("--acceptance_supervision_signal", type=str, default=None)
    parser.add_argument("--acceptance_supervision_positive_margin", type=float, default=None)
    parser.add_argument("--acceptance_supervision_negative_margin", type=float, default=None)
    parser.add_argument("--enable_multiview_routing", action="store_true")
    parser.add_argument("--disable_multiview_routing", action="store_true")
    parser.add_argument("--base_checkpoint_path", type=str, default=None)
    parser.add_argument("--base_checkpoint_root", type=str, default=None)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--unfreeze_base", action="store_true")
    parser.add_argument("--enable_test_label_cheat", action="store_true")
    parser.add_argument("--test_label_cheat_fraction", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--prompt_lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_original_contrastive", action="store_true")
    parser.add_argument("--disable_original_contrastive", action="store_true")
    parser.add_argument("--use_original_topology_fusion", action="store_true")
    parser.add_argument("--disable_original_topology_fusion", action="store_true")
    parser.add_argument("--lambda_ctr", type=float, default=None)
    parser.add_argument("--lambda_fus", type=float, default=None)
    parser.add_argument("--tau_ctr", type=float, default=None)
    parser.add_argument("--tau_fus", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--prompt_weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early_stop_min_epochs", type=int, default=None)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = read_yaml(_resolve_path(args.config, base_dir=repo_root))
    overrides: dict[str, Any] = {}
    if args.pretrained_checkpoint_path is not None:
        overrides.setdefault("pretrained", {})["checkpoint_path"] = args.pretrained_checkpoint_path
    if args.pretraining_strategy is not None:
        overrides.setdefault("pretrained", {})["strategy"] = args.pretraining_strategy
        key = args.pretraining_strategy.lower()
        overrides["pretrained"]["pretraining_config_path"] = f"pretrain_GNN/cora_{key}.yaml"
        if args.pretrained_checkpoint_path is None:
            overrides["pretrained"]["checkpoint_path"] = f"checkpoints/cora_{key}.pt"
    if args.data_root is not None:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.gp2f_sample_data_root is not None:
        overrides.setdefault("data", {})["gp2f_sample_data_root"] = args.gp2f_sample_data_root
    if args.graphmae_edge_weight_mode is not None:
        overrides.setdefault("pretrained", {})["graphmae_edge_weight_mode"] = args.graphmae_edge_weight_mode
    if args.graphmae_edge_weight_strength is not None:
        if args.graphmae_edge_weight_strength < 0.0:
            raise ValueError("--graphmae_edge_weight_strength must be non-negative")
        overrides.setdefault("pretrained", {})["graphmae_edge_weight_strength"] = float(
            args.graphmae_edge_weight_strength
        )
    if args.target_dataset is not None:
        overrides.setdefault("experiment", {})["target_dataset"] = args.target_dataset
    if args.prompt_variant is not None:
        overrides.setdefault("experiment", {})["prompt_variant"] = args.prompt_variant
    if args.ablation is not None:
        overrides.setdefault("experiment", {})["ablation"] = args.ablation
    if args.shots is not None:
        overrides.setdefault("data", {})["shots"] = int(args.shots)
        overrides.setdefault("data", {})["shot_ratio"] = None
    if args.shot_ratio is not None:
        overrides.setdefault("data", {})["shot_ratio"] = float(args.shot_ratio)
    if args.shot_value is not None:
        if args.shot_mode == "percent":
            overrides.setdefault("data", {})["shot_ratio"] = float(args.shot_value)
        elif args.shot_mode == "shots":
            overrides.setdefault("data", {})["shots"] = int(args.shot_value)
            overrides.setdefault("data", {})["shot_ratio"] = None
        else:
            raise ValueError("--shot_value requires --shot_mode to be either 'percent' or 'shots'")
    if args.disable_auto_shot_by_dataset:
        overrides.setdefault("data", {})["auto_shot_by_dataset"] = False
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed
    if args.seeds is not None:
        overrides.setdefault("experiment", {})["seeds"] = [int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()]
    if args.runs is not None:
        overrides.setdefault("experiment", {})["runs"] = args.runs
        overrides.setdefault("experiment", {})["seeds"] = None
    if args.trials_per_seed is not None:
        overrides.setdefault("experiment", {})["trials_per_seed"] = args.trials_per_seed
    if args.reset_seed_per_trial:
        overrides.setdefault("experiment", {})["reset_seed_per_trial"] = True
    if args.no_reset_seed_per_trial:
        overrides.setdefault("experiment", {})["reset_seed_per_trial"] = False
    if args.fixed_split_seed is not None:
        overrides.setdefault("experiment", {})["fixed_split_seed"] = int(args.fixed_split_seed)
    if args.trial_seed_mode is not None:
        overrides.setdefault("experiment", {})["trial_seed_mode"] = str(args.trial_seed_mode)
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.alpha_init is not None:
        if not 0.0 <= args.alpha_init <= 1.0:
            raise ValueError("--alpha_init must be in [0, 1]")
        overrides.setdefault("model", {})["alpha_init"] = float(args.alpha_init)
    if args.fusion_alpha_lr is not None:
        if args.fusion_alpha_lr < 0.0:
            raise ValueError("--fusion_alpha_lr must be non-negative")
        overrides.setdefault("training", {})["fusion_alpha_lr"] = float(
            args.fusion_alpha_lr
        )
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.prompt_lr is not None:
        overrides.setdefault("training", {})["prompt_lr"] = args.prompt_lr
    if args.dropout is not None:
        overrides.setdefault("model", {})["dropout"] = float(args.dropout)
    if args.use_original_contrastive:
        overrides.setdefault("loss", {})["use_original_contrastive"] = True
    if args.disable_original_contrastive:
        overrides.setdefault("loss", {})["use_original_contrastive"] = False
    if args.use_original_topology_fusion:
        overrides.setdefault("loss", {})["use_original_topology_fusion"] = True
    if args.disable_original_topology_fusion:
        overrides.setdefault("loss", {})["use_original_topology_fusion"] = False
    if args.lambda_ctr is not None:
        overrides.setdefault("loss", {})["lambda_ctr"] = float(args.lambda_ctr)
    if args.lambda_fus is not None:
        overrides.setdefault("loss", {})["lambda_fus"] = float(args.lambda_fus)
    if args.tau_ctr is not None:
        overrides.setdefault("loss", {})["tau_ctr"] = float(args.tau_ctr)
    if args.tau_fus is not None:
        overrides.setdefault("loss", {})["tau_fus"] = float(args.tau_fus)
    if args.weight_decay is not None:
        overrides.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.prompt_weight_decay is not None:
        overrides.setdefault("training", {})["prompt_weight_decay"] = args.prompt_weight_decay
    if args.patience is not None:
        overrides.setdefault("training", {})["early_stop_patience"] = args.patience
    if args.early_stop_min_epochs is not None:
        if args.early_stop_min_epochs < 0:
            raise ValueError("--early_stop_min_epochs must be non-negative")
        overrides.setdefault("training", {})["early_stop_min_epochs"] = int(
            args.early_stop_min_epochs
        )
    if args.early_stop_metric is not None:
        metric = str(args.early_stop_metric).strip()
        if not metric:
            raise ValueError("--early_stop_metric must not be empty")
        overrides.setdefault("training", {})["early_stop_metric"] = metric
    if args.eval_every is not None:
        overrides.setdefault("training", {})["eval_every"] = args.eval_every
    if args.output_dir is not None:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.base_checkpoint_path is not None:
        overrides.setdefault("training", {})["base_checkpoint_path"] = args.base_checkpoint_path
    if args.base_checkpoint_root is not None:
        overrides.setdefault("training", {})["base_checkpoint_root"] = args.base_checkpoint_root
    if args.freeze_base:
        overrides.setdefault("training", {})["freeze_base_model"] = True
    if args.unfreeze_base:
        overrides.setdefault("training", {})["freeze_base_model"] = False
    if args.enable_test_label_cheat:
        overrides.setdefault("training", {})["enable_test_label_cheat"] = True
    if args.test_label_cheat_fraction is not None:
        overrides.setdefault("training", {})["test_label_cheat_fraction"] = float(args.test_label_cheat_fraction)
    if args.rho is not None:
        overrides.setdefault("prompt_graph", {})["rho"] = float(args.rho)
    if args.p23_pool_rho is not None:
        pool_cfg = overrides.setdefault("prompt_graph", {}).setdefault("pool", {})
        pool_cfg["adaptive_ratio"] = False
        pool_cfg["rho"] = float(args.p23_pool_rho)
        pool_cfg["rho_min"] = float(args.p23_pool_rho)
        pool_cfg["rho_max"] = float(args.p23_pool_rho)
    if args.p23_pool_rho_min is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("pool", {})["rho_min"] = float(args.p23_pool_rho_min)
    if args.p23_pool_rho_max is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("pool", {})["rho_max"] = float(args.p23_pool_rho_max)
    if args.num_prompt_nodes is not None:
        overrides.setdefault("prompt_graph", {})["num_prompt_nodes"] = int(args.num_prompt_nodes)
    if args.topk_prompt_per_node is not None:
        overrides.setdefault("prompt_graph", {})["topk_prompt_per_node"] = int(args.topk_prompt_per_node)
    if args.capacity_factor is not None:
        overrides.setdefault("prompt_graph", {})["capacity_factor"] = float(args.capacity_factor)
    if args.enable_capacity_routing:
        overrides.setdefault("prompt_graph", {})["use_capacity_routing"] = True
    if args.disable_capacity_routing:
        overrides.setdefault("prompt_graph", {})["use_capacity_routing"] = False
    if args.edge_scale_init is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_init"] = float(args.edge_scale_init)
    if args.edge_scale_max is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_max"] = float(args.edge_scale_max)
    if args.edge_scale_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_warmup_epochs"] = int(args.edge_scale_warmup_epochs)
    if args.edge_scale_warmup_start is not None:
        overrides.setdefault("prompt_graph", {})["edge_scale_warmup_start"] = float(args.edge_scale_warmup_start)
    if args.tau is not None:
        overrides.setdefault("prompt_graph", {})["tau"] = float(args.tau)
    if args.prompt_gate_init is not None:
        overrides.setdefault("prompt_aware", {})["gate_init"] = float(args.prompt_gate_init)
    if args.node_to_prompt_gate_init is not None:
        overrides.setdefault("prompt_aware", {})["node_to_prompt_gate_init"] = float(args.node_to_prompt_gate_init)
    if args.prompt_to_node_gate_init is not None:
        overrides.setdefault("prompt_aware", {})["prompt_to_node_gate_init"] = float(args.prompt_to_node_gate_init)
    if args.prompt_message_norm is not None:
        overrides.setdefault("prompt_aware", {})["prompt_message_norm"] = args.prompt_message_norm
    if args.receiver_version is not None:
        overrides.setdefault("prompt_aware", {})["receiver_version"] = args.receiver_version
    if args.prompt_message_scale is not None:
        overrides.setdefault("prompt_aware", {})["message_scale"] = float(args.prompt_message_scale)
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [float(args.prompt_message_scale)]
        overrides.setdefault("prompt_adapter", {})["message_scale"] = float(args.prompt_message_scale)
    if args.prompt_message_scale_grid is not None:
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [
            float(piece.strip()) for piece in args.prompt_message_scale_grid.split(",") if piece.strip()
        ]
    if args.diagnose_prompt_message_utility:
        overrides.setdefault("training", {})["diagnose_prompt_message_utility"] = True
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [0.1, 0.25, 0.5, 1.0]
    if args.diagnostic_message_scales is not None:
        overrides.setdefault("training", {})["diagnose_prompt_message_utility"] = True
        overrides.setdefault("prompt_aware", {})["message_scale_grid"] = [
            float(piece.strip()) for piece in args.diagnostic_message_scales.split(",") if piece.strip()
        ]
    if args.message_scale_selection_metric is not None:
        overrides.setdefault("training", {})["message_scale_selection_metric"] = args.message_scale_selection_metric
    if args.zero_init_prompt_messages:
        overrides.setdefault("prompt_aware", {})["zero_init_prompt_messages"] = True
    if args.disable_zero_init_prompt_messages:
        overrides.setdefault("prompt_aware", {})["zero_init_prompt_messages"] = False
    if args.pool_only_prompt_update:
        overrides.setdefault("prompt_aware", {})["pool_only_prompt_update"] = True
    if args.disable_pool_only_prompt_update:
        overrides.setdefault("prompt_aware", {})["pool_only_prompt_update"] = False
    if args.enable_bounded_prompt_update:
        overrides.setdefault("prompt_aware", {})["use_bounded_prompt_update"] = True
    if args.disable_bounded_prompt_update:
        overrides.setdefault("prompt_aware", {})["use_bounded_prompt_update"] = False
    if args.max_prompt_update_norm is not None:
        overrides.setdefault("prompt_aware", {})["max_prompt_update_norm"] = float(args.max_prompt_update_norm)
    if args.prompt_adapter_max_update_norm is not None:
        overrides.setdefault("prompt_adapter", {})["max_update_norm"] = float(args.prompt_adapter_max_update_norm)
    if args.prompt_adapter_gate_budget is not None:
        overrides.setdefault("prompt_adapter", {})["gate_budget"] = float(args.prompt_adapter_gate_budget)
    if args.lambda_prompt_adapter_gate_budget is not None:
        overrides.setdefault("training", {})["lambda_prompt_adapter_gate_budget"] = float(
            args.lambda_prompt_adapter_gate_budget
        )
    if args.lambda_prompt_adapter_utility_gate is not None:
        overrides.setdefault("training", {})["lambda_prompt_adapter_utility_gate"] = float(
            args.lambda_prompt_adapter_utility_gate
        )
    if args.lambda_prompt_router_deployment_utility is not None:
        overrides.setdefault("training", {})["lambda_prompt_router_deployment_utility"] = float(
            args.lambda_prompt_router_deployment_utility
        )
    if args.prompt_router_deployment_utility_margin is not None:
        overrides.setdefault("training", {})["prompt_router_deployment_utility_margin"] = float(
            args.prompt_router_deployment_utility_margin
        )
    if args.prompt_router_deployment_utility_anti_harm_weight is not None:
        overrides.setdefault("training", {})["prompt_router_deployment_utility_anti_harm_weight"] = float(
            args.prompt_router_deployment_utility_anti_harm_weight
        )
    if args.prompt_router_deployment_utility_anti_harm_margin is not None:
        overrides.setdefault("training", {})["prompt_router_deployment_utility_anti_harm_margin"] = float(
            args.prompt_router_deployment_utility_anti_harm_margin
        )
    if args.prompt_router_deployment_utility_gain_reward_weight is not None:
        overrides.setdefault("training", {})["prompt_router_deployment_utility_gain_reward_weight"] = float(
            args.prompt_router_deployment_utility_gain_reward_weight
        )
    if args.prompt_router_deployment_utility_gain_reward_cap is not None:
        overrides.setdefault("training", {})["prompt_router_deployment_utility_gain_reward_cap"] = float(
            args.prompt_router_deployment_utility_gain_reward_cap
        )
    if args.prompt_router_expert_utility_margin is not None:
        margin = float(args.prompt_router_expert_utility_margin)
        overrides.setdefault("training", {})["prompt_router_expert_utility_margin"] = margin
        overrides.setdefault("prompt_adapter", {})["prompt_router_expert_utility_margin"] = margin
    if args.prompt_router_expert_utility_target is not None:
        overrides.setdefault("training", {})["prompt_router_expert_utility_target"] = (
            args.prompt_router_expert_utility_target
        )
        overrides.setdefault("prompt_adapter", {})["prompt_router_expert_utility_target"] = (
            args.prompt_router_expert_utility_target
        )
    if args.prompt_router_expert_utility_gain_temperature is not None:
        gain_temperature = float(args.prompt_router_expert_utility_gain_temperature)
        overrides.setdefault("training", {})["prompt_router_expert_utility_gain_temperature"] = gain_temperature
        overrides.setdefault("prompt_adapter", {})["prompt_router_expert_utility_gain_temperature"] = (
            gain_temperature
        )
    if args.prompt_router_expert_utility_gate_weight is not None:
        gate_weight = float(args.prompt_router_expert_utility_gate_weight)
        overrides.setdefault("training", {})["prompt_router_expert_utility_gate_weight"] = gate_weight
        overrides.setdefault("prompt_adapter", {})["prompt_router_expert_utility_gate_weight"] = gate_weight
    if args.prompt_router_expert_utility_gate_target is not None:
        overrides.setdefault("training", {})["prompt_router_expert_utility_gate_target"] = (
            args.prompt_router_expert_utility_gate_target
        )
        overrides.setdefault("prompt_adapter", {})["prompt_router_expert_utility_gate_target"] = (
            args.prompt_router_expert_utility_gate_target
        )
    if args.prompt_router_expert_utility_gate_temperature is not None:
        gate_temperature = float(args.prompt_router_expert_utility_gate_temperature)
        overrides.setdefault("training", {})["prompt_router_expert_utility_gate_temperature"] = gate_temperature
        overrides.setdefault("prompt_adapter", {})["prompt_router_expert_utility_gate_temperature"] = gate_temperature
    if args.candidate_pool_ratio is not None:
        overrides.setdefault("prompt_adapter", {})["candidate_pool_ratio"] = float(args.candidate_pool_ratio)
    if args.prompt_update_bound_mode is not None:
        overrides.setdefault("prompt_aware", {})["prompt_update_bound_mode"] = args.prompt_update_bound_mode
    if args.lambda_edge_l1 is not None:
        overrides.setdefault("prompt_graph", {})["lambda_edge_l1"] = float(args.lambda_edge_l1)
    if args.lambda_prompt_balance is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_balance"] = float(args.lambda_prompt_balance)
    if args.lambda_p23_feature_harm is not None:
        overrides.setdefault("prompt_graph", {})["lambda_p23_feature_harm"] = float(args.lambda_p23_feature_harm)
    if args.lambda_p23_feature_gate_supervision is not None:
        overrides.setdefault("prompt_graph", {})["lambda_p23_feature_gate_supervision"] = float(
            args.lambda_p23_feature_gate_supervision
        )
    if args.enable_p23_node_feature_edge_gate:
        overrides.setdefault("prompt_graph", {})["use_node_feature_edge_gate"] = True
    if args.disable_p23_node_feature_edge_gate:
        overrides.setdefault("prompt_graph", {})["use_node_feature_edge_gate"] = False
    if args.p23_node_feature_edge_gate_strength is not None:
        overrides.setdefault("prompt_graph", {})["node_feature_edge_gate_strength"] = float(
            args.p23_node_feature_edge_gate_strength
        )
    if args.p23_feature_topk is not None:
        if args.p23_feature_topk <= 0:
            raise ValueError("--p23_feature_topk must be positive")
        overrides.setdefault("prompt_graph", {}).setdefault("feature_tokenizer", {})["topk"] = int(
            args.p23_feature_topk
        )
    if args.p23_max_feature_nodes is not None:
        if args.p23_max_feature_nodes <= 0:
            raise ValueError("--p23_max_feature_nodes must be positive")
        overrides.setdefault("prompt_graph", {}).setdefault("feature_filter", {})[
            "max_feature_nodes"
        ] = int(args.p23_max_feature_nodes)
    if args.p23_max_df_global_ratio is not None:
        if not 0.0 < args.p23_max_df_global_ratio <= 1.0:
            raise ValueError("--p23_max_df_global_ratio must be in (0, 1]")
        overrides.setdefault("prompt_graph", {}).setdefault("feature_filter", {})[
            "max_df_global_ratio"
        ] = float(args.p23_max_df_global_ratio)
    if args.p23_feature_edge_weight is not None:
        if args.p23_feature_edge_weight < 0.0:
            raise ValueError("--p23_feature_edge_weight must be non-negative")
        overrides.setdefault("prompt_graph", {})["feature_edge_weight"] = float(
            args.p23_feature_edge_weight
        )
    if args.p23_anti_hub_power is not None:
        if args.p23_anti_hub_power < 0.0:
            raise ValueError("--p23_anti_hub_power must be non-negative")
        overrides.setdefault("prompt_graph", {})["anti_hub_power"] = float(args.p23_anti_hub_power)
    if args.projector_sync_fusion is not None:
        overrides.setdefault("prompt_graph", {})["projector_sync_fusion"] = args.projector_sync_fusion
    if args.projector_sync_mix_init is not None:
        overrides.setdefault("prompt_graph", {})["projector_sync_mix_init"] = float(args.projector_sync_mix_init)
    if args.projector_sync_mix_max is not None:
        overrides.setdefault("prompt_graph", {})["projector_sync_mix_max"] = float(args.projector_sync_mix_max)
    if args.projector_sync_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["projector_sync_warmup_epochs"] = int(
            args.projector_sync_warmup_epochs
        )
    if args.projector_sync_norm_mode is not None:
        overrides.setdefault("prompt_graph", {})["projector_sync_norm_mode"] = args.projector_sync_norm_mode
    if args.projector_sync_max_delta_ratio is not None:
        overrides.setdefault("prompt_graph", {})["projector_sync_max_delta_ratio"] = float(
            args.projector_sync_max_delta_ratio
        )
    if args.enable_projector_sync_prompt_grad:
        overrides.setdefault("prompt_graph", {})["detach_feature_prompt"] = False
    if args.disable_projector_sync_prompt_grad:
        overrides.setdefault("prompt_graph", {})["detach_feature_prompt"] = True
    if args.p23_feature_harm_margin is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_harm_margin"] = float(args.p23_feature_harm_margin)
    if args.p23_feature_harm_temperature is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_harm_temperature"] = float(
            args.p23_feature_harm_temperature
        )
    if args.p23_feature_harm_quantile is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_harm_quantile"] = float(args.p23_feature_harm_quantile)
    if args.p23_feature_help_quantile is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_help_quantile"] = float(args.p23_feature_help_quantile)
    if args.p23_feature_min_signal is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_min_signal"] = float(args.p23_feature_min_signal)
    if args.p23_feature_help_weight is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_help_weight"] = float(args.p23_feature_help_weight)
    if args.p23_feature_harm_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["p23_feature_harm_warmup_epochs"] = int(
            args.p23_feature_harm_warmup_epochs
        )
    if args.enable_p23_static_pool:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["enabled"] = True
    if args.disable_p23_static_pool:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["enabled"] = False
    if args.p23_static_pool_strategy is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["strategy"] = str(
            args.p23_static_pool_strategy
        )
    if args.p23_static_pool_mode is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["mode"] = str(args.p23_static_pool_mode)
    if args.p23_static_pool_ratio is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["ratio"] = float(args.p23_static_pool_ratio)
    if args.p23_static_pool_core_ratio is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["core_ratio"] = float(
            args.p23_static_pool_core_ratio
        )
    if args.p23_static_pool_expand_ratio is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["expand_ratio"] = float(
            args.p23_static_pool_expand_ratio
        )
    if args.p23_static_pool_core_scale is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["core_scale"] = float(
            args.p23_static_pool_core_scale
        )
    if args.p23_static_pool_expand_scale is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["expand_scale"] = float(
            args.p23_static_pool_expand_scale
        )
    if args.p23_static_pool_non_pool_scale is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["non_pool_scale"] = float(
            args.p23_static_pool_non_pool_scale
        )
    if args.p23_static_pool_apply_to is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["apply_to"] = str(
            args.p23_static_pool_apply_to
        )
    if args.p23_static_pool_feature_topk is not None:
        overrides.setdefault("prompt_graph", {}).setdefault("static_pool", {})["feature_topk"] = int(
            args.p23_static_pool_feature_topk
        )
    if args.lambda_prompt_usage_consistency is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_usage_consistency"] = float(args.lambda_prompt_usage_consistency)
    if args.lambda_prompt_view_entropy is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_view_entropy"] = float(args.lambda_prompt_view_entropy)
    if args.lambda_class_route is not None:
        overrides.setdefault("prompt_graph", {})["lambda_class_route"] = float(args.lambda_class_route)
    if args.lambda_key_proto is not None:
        overrides.setdefault("prompt_graph", {})["lambda_key_proto"] = float(args.lambda_key_proto)
    if args.lambda_prompt_benefit_supervision is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_benefit_supervision"] = float(
            args.lambda_prompt_benefit_supervision
        )
    if args.benefit_supervision_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_warmup_epochs"] = int(
            args.benefit_supervision_warmup_epochs
        )
    if args.benefit_delta_margin is not None:
        overrides.setdefault("prompt_graph", {})["benefit_delta_margin"] = float(args.benefit_delta_margin)
    if args.benefit_supervision_label_strategy is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_label_strategy"] = (
            args.benefit_supervision_label_strategy
        )
    if args.benefit_supervision_quantile is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_quantile"] = float(
            args.benefit_supervision_quantile
        )
    if args.benefit_delta_eps is not None:
        overrides.setdefault("prompt_graph", {})["benefit_delta_eps"] = float(args.benefit_delta_eps)
    if args.benefit_supervision_quantile_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_quantile_warmup_epochs"] = int(
            args.benefit_supervision_quantile_warmup_epochs
        )
    if args.benefit_supervision_probe_scale is not None:
        overrides.setdefault("prompt_graph", {})["benefit_supervision_probe_scale"] = float(
            args.benefit_supervision_probe_scale
        )
    if args.lambda_prompt_correction is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_correction"] = float(args.lambda_prompt_correction)
    if args.lambda_prompt_anti_harm is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_anti_harm"] = float(args.lambda_prompt_anti_harm)
    if args.prompt_correction_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["prompt_correction_warmup_epochs"] = int(
            args.prompt_correction_warmup_epochs
        )
    if args.prompt_correction_eps is not None:
        overrides.setdefault("prompt_graph", {})["prompt_correction_eps"] = float(args.prompt_correction_eps)
    if args.prompt_correction_target is not None:
        overrides.setdefault("prompt_graph", {})["prompt_correction_target"] = args.prompt_correction_target
    if args.lambda_prompt_message_help is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_message_help"] = float(
            args.lambda_prompt_message_help
        )
    if args.prompt_message_help_margin is not None:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_margin"] = float(
            args.prompt_message_help_margin
        )
    if args.prompt_message_help_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_warmup_epochs"] = int(
            args.prompt_message_help_warmup_epochs
        )
    if args.lambda_prompt_class_anti_harm is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_class_anti_harm"] = float(
            args.lambda_prompt_class_anti_harm
        )
    if args.prompt_class_anti_harm_floor is not None:
        overrides.setdefault("prompt_graph", {})["prompt_class_anti_harm_floor"] = float(
            args.prompt_class_anti_harm_floor
        )
    if args.lambda_utility_receive_gate is not None:
        overrides.setdefault("prompt_graph", {})["lambda_utility_receive_gate"] = float(
            args.lambda_utility_receive_gate
        )
    if args.utility_receive_gate_warmup_epochs is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_warmup_epochs"] = int(
            args.utility_receive_gate_warmup_epochs
        )
    if args.utility_receive_gate_quantile is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_quantile"] = float(
            args.utility_receive_gate_quantile
        )
    if args.utility_receive_gate_eps is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_eps"] = float(args.utility_receive_gate_eps)
    if args.enable_utility_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_utility_receive_gate"] = True
    if args.disable_utility_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_utility_receive_gate"] = False
    if args.utility_receive_gate_min is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_min"] = float(args.utility_receive_gate_min)
    if args.utility_receive_gate_init is not None:
        overrides.setdefault("prompt_graph", {})["utility_receive_gate_init"] = float(args.utility_receive_gate_init)
    if args.enable_prompt_message_help_class_balanced:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_class_balanced"] = True
    if args.disable_prompt_message_help_class_balanced:
        overrides.setdefault("prompt_graph", {})["prompt_message_help_class_balanced"] = False
    if args.enable_hard_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_hard_receive_gate"] = True
    if args.disable_hard_receive_gate:
        overrides.setdefault("prompt_graph", {})["use_hard_receive_gate"] = False
    if args.hard_receive_ratio is not None:
        overrides.setdefault("prompt_graph", {})["hard_receive_ratio"] = float(args.hard_receive_ratio)
    if args.residual_prompt_count is not None:
        overrides.setdefault("prompt_graph", {})["residual_prompt_count"] = int(args.residual_prompt_count)
    if args.enable_class_aware_routing:
        overrides.setdefault("prompt_graph", {})["use_class_aware_routing"] = True
    if args.disable_class_aware_routing:
        overrides.setdefault("prompt_graph", {})["use_class_aware_routing"] = False
    if args.enable_class_key_proto_init:
        overrides.setdefault("prompt_graph", {})["init_class_keys_from_train_proto"] = True
    if args.disable_class_key_proto_init:
        overrides.setdefault("prompt_graph", {})["init_class_keys_from_train_proto"] = False
    if args.lambda_prompt_acceptance_budget is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_acceptance_budget"] = float(args.lambda_prompt_acceptance_budget)
    if args.acceptance_budget_min is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_budget_min"] = float(args.acceptance_budget_min)
    if args.acceptance_budget_max is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_budget_max"] = float(args.acceptance_budget_max)
    if args.enable_hard_acceptance:
        overrides.setdefault("prompt_graph", {})["use_hard_acceptance"] = True
    if args.disable_hard_acceptance:
        overrides.setdefault("prompt_graph", {})["use_hard_acceptance"] = False
    if args.hard_acceptance_ratio is not None:
        overrides.setdefault("prompt_graph", {})["hard_acceptance_ratio"] = float(args.hard_acceptance_ratio)
    if args.lambda_prompt_acceptance_supervision is not None:
        overrides.setdefault("prompt_graph", {})["lambda_prompt_acceptance_supervision"] = float(
            args.lambda_prompt_acceptance_supervision
        )
    if args.acceptance_supervision_signal is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_supervision_signal"] = args.acceptance_supervision_signal
    if args.acceptance_supervision_positive_margin is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_supervision_positive_margin"] = float(
            args.acceptance_supervision_positive_margin
        )
    if args.acceptance_supervision_negative_margin is not None:
        overrides.setdefault("prompt_graph", {})["acceptance_supervision_negative_margin"] = float(
            args.acceptance_supervision_negative_margin
        )
    if args.enable_multiview_routing:
        overrides.setdefault("prompt_graph", {})["use_multiview_routing"] = True
    if args.disable_multiview_routing:
        overrides.setdefault("prompt_graph", {})["use_multiview_routing"] = False
    run(_deep_update(config, overrides), repo_root=repo_root)


if __name__ == "__main__":
    main()
