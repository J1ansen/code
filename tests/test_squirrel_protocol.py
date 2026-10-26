"""Keep Squirrel-F aligned with validation-selected, learnable adaptation."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_squirrel_runs_full_160_epochs_with_validation_selection():
    config = yaml.safe_load((ROOT / "configs/main/squirrel.yaml").read_text())
    training, model = config["training"], config["model"]
    assert training["epochs"] == 160
    assert training["early_stop_patience"] == 0
    assert training["early_stop_metric"] == "val_acc"
    assert "adapter_beta_fixed" not in training
    assert model["use_adapters"] is True
    assert model["adapter_beta_init"] == 0.001
    assert config["release"]["protocol"] == "validation_selected_v1"
    entry = next(e for e in json.loads((ROOT / "configs/main/protocol.json").read_text())
                 if e["dataset"] == "squirrel")
    assert entry["selection_metric"] == entry["validation_metric"] == "val_acc"
    assert entry["full_training_epochs"] == 160
    assert entry["protocol"] == config["release"]["protocol"]
