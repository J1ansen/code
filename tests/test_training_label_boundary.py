"""Test that held-out labels cannot affect training or choose checkpoints."""
from pathlib import Path
from types import SimpleNamespace
import sys
import copy

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_main_experiment import (
    _classification_loss, _training_label_mask, _validate_selection_metric,
    _select_message_scale_candidate, _update_test_acc_diagnostic,
)
from models.prompt_graph_construction import PromptGraphConstruction


def test_training_mask_does_not_include_held_out_nodes():
    split = SimpleNamespace(train_mask=torch.tensor([True, False, False]),
                            test_mask=torch.tensor([False, False, True]))
    mask = _training_label_mask(split)
    assert torch.equal(mask, split.train_mask)
    mask[0] = False
    assert split.train_mask[0]
    split.test_mask[0] = True
    with pytest.raises(ValueError, match="disjoint"):
        _training_label_mask(split)


@pytest.mark.parametrize("metric", ["test_acc", "test_macro_f1", "test_auroc", "test_auprc"])
def test_test_metrics_cannot_select_models(metric):
    with pytest.raises(ValueError, match="training-loss or validation"):
        _validate_selection_metric(metric)
    with pytest.raises(ValueError):
        _select_message_scale_candidate([{"best": {metric: 1.0}}], metric=metric)
    with pytest.raises(ValueError):
        _update_test_acc_diagnostic({}, epoch=1, train_loss=1.0,
                                    metrics={metric: 1.0}, selection_metric=metric, maximize=True)


def test_validation_selection_ignores_better_test_scores():
    a = {"best": {"val_acc": 0.8, "test_acc": 0.1}}
    b = {"best": {"val_acc": 0.4, "test_acc": 1.0}}
    assert _select_message_scale_candidate([a, b], metric="val_acc") is a


@pytest.mark.parametrize("balanced", [False, True])
def test_held_out_labels_do_not_change_training_loss_or_gradients(balanced):
    torch.manual_seed(4)
    logits = torch.randn(6, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    changed = torch.tensor([0, 1, 0, 0, 1, 0])
    mask = torch.tensor([True, True, True, False, False, False])
    kwargs = dict(logits=logits, train_mask=mask, num_classes=2,
                  training_cfg={"class_balanced_ce": balanced})
    first = _classification_loss(labels=labels, **kwargs)
    second = _classification_loss(labels=changed, **kwargs)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.autograd.grad(first, logits)[0],
                               torch.autograd.grad(second, logits)[0])


def test_held_out_labels_do_not_change_prompt_construction():
    torch.manual_seed(9)
    config = {"num_classes": 2, "static_graph": False,
              "feature_tokenizer": {"mode": "topk_activation", "topk": 2},
              "feature_filter": {"min_df_global": 1, "max_df_global_ratio": 1.0,
                                 "min_label_count": 1, "min_purity": 0.0, "max_entropy": 1.0}}
    module = PromptGraphConstruction(3, 3, config)
    other = copy.deepcopy(module)
    features = torch.tensor([[2., 1., 0.], [0., 1., 2.], [2., 0., 1.],
                             [0., 2., 1.], [1., 2., 0.], [1., 0., 2.]])
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    kwargs = dict(z=features, x_raw=features, edge_index=edges,
                  train_mask=torch.tensor([True, True, True, False, False, False]))
    first = module._build_graph(labels=torch.tensor([0, 1, 0, 1, 0, 1]), **kwargs)
    second = other._build_graph(labels=torch.tensor([0, 1, 0, 0, 1, 0]), **kwargs)
    assert first["prompt_x"].numel() > 0
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, second[key], msg=lambda msg: key + ': ' + msg)


@pytest.mark.parametrize("field", ["early_stop_metric", "message_scale_selection_metric"])
def test_public_runner_rejects_test_selection_before_pretraining(tmp_path, monkeypatch, field):
    from experiments.run_main_experiment import run
    from pretrain_GNN import pretrain
    monkeypatch.setattr(pretrain, "ensure_pretrained", lambda *a, **k: pytest.fail("Must reject before pretraining"))
    with pytest.raises(ValueError, match="training-loss or validation"):
        run({"training": {field: "test_acc"}}, repo_root=tmp_path)
