"""Exercise the release training loop without downloading datasets or weights."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
import yaml
from torch_geometric.data import Data
from models.backbones import BaseGCN
from experiments import run_main_experiment as runner

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {'class_conditioned_pattern_prompt_router', 'utility_supervised_pattern_prompt_router',
           'hetero_prompt_adapter', 'discrete_feature_prompt', 'prompt_graph_module',
           'prompt_aware_gp2f', 'prompt_receiver', 'projected_prompt_receiver',
           'synchronized_prompt_graph', 'static_prompt_graph', 'prompt_module'}


def test_historical_model_modules_are_neither_present_nor_imported():
    for name in RETIRED:
        assert not (ROOT / f'models/{name}.py').exists()
        assert f'models.{name}' not in sys.modules


@pytest.mark.parametrize('name', ['cora', 'citeseer', 'pubmed', 'wikics', 'actor',
                                  'chameleon', 'squirrel', 'minesweeper'])
def test_main_training_loop_on_small_graph(tmp_path, monkeypatch, name):
    config = yaml.safe_load((ROOT / f'configs/main/{name}.yaml').read_text())
    config = runner._config_for_variant(config, config['experiment']['prompt_variant'])
    config['experiment'].update(seed=12345, split_seed=12345, trial_index=0)
    config['data'].update(shots=2, shot_ratio=None, auto_shot_by_dataset=False,
                          val_per_class=2, split_protocol='generated', gp2f_test_ratio=None)
    config['training'].update(epochs=3, early_stop_patience=0, early_stop_min_epochs=0,
                              output_dir=str(tmp_path / name), log_every=1,
                              keep_best_checkpoint=False, diagnose_prompt_message_utility=False)
    config['prompt_graph']['feature_filter'].update(min_df_global=1, max_df_global_ratio=1.0,
                                                   min_label_count=1, min_purity=0.0, max_entropy=1.0)
    config['prompt_graph']['feature_tokenizer'] = {'mode': 'topk_activation', 'topk': 2}
    torch.manual_seed(8)
    graph = Data(x=torch.rand(36, 8), y=torch.arange(36) % 3,
                 edge_index=torch.stack([torch.arange(36), torch.arange(36).roll(1)]))
    monkeypatch.setattr(runner, 'load_node_dataset',
                        lambda *a, **k: SimpleNamespace(name=name, data=graph.clone(), num_features=8, num_classes=3))
    def backbone(*args, **kwargs):
        model = BaseGCN(8, 8)
        model.requires_grad_(False)
        return model
    monkeypatch.setattr(runner, 'load_pretrained_backbone', backbone)
    result = runner.run_single(config, repo_root=tmp_path)
    assert result['stopped_epoch'] == 3
