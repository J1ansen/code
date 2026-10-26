"""Retired model implementations must not be required by the release runner."""

from pathlib import Path

import pytest
import torch

import models
from experiments.run_main_experiment import (
    PROMPT_GRAPH_VARIANTS,
    _build_prompt_adapter_module,
    _config_for_variant,
)


@pytest.mark.parametrize("variant", [
    "adaptive_filter_lite_adaptive_filter", "adaptive_filter_v2_hetero_filter",
    "pattern_class_pattern_enrichment_bank", "pattern_reliability_calibrated_basis_bank",
    "pattern_v031_conservative_reliability_basis_bank", "pattern_v04_minimal_transition_basis",
])
def test_retired_variants_are_rejected(variant):
    assert variant not in PROMPT_GRAPH_VARIANTS
    with pytest.raises(ValueError):
        _config_for_variant({}, variant)
    with pytest.raises(ValueError, match="Unsupported"):
        _build_prompt_adapter_module(
            source_dim=8, hidden_dim=8, num_classes=2,
            prompt_adapter_cfg={"enabled": True, "module_type": variant},
            device=torch.device("cpu"),
        )


def test_retired_implementations_are_removed():
    root = Path(models.__file__).parent
    for name in ("adaptive_filter_adaptive_filter", "adaptive_filter_v2_hetero_filter", "pattern_class_pattern_enrichment"):
        assert not (root / f"{name}.py").exists()
    for name in ("AdaptiveFilterLiteAdaptiveFilter", "AdaptiveFilterV2HeteroFilter", "PatternClassPatternEnrichmentBank"):
        assert not hasattr(models, name)
