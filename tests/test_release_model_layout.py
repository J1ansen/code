"""Check the paper-aligned release layout and retired selector boundary."""
from pathlib import Path
import pytest
import yaml
import torch
from models.prompting_adaptation import PromptingAdaptation

ROOT = Path(__file__).resolve().parents[1]


def test_model_filenames_do_not_use_experiment_number_prefixes():
    import re
    assert not [p.name for p in (ROOT / 'models').glob('*.py')
                if re.match(r'p\d+_', p.name)]


@pytest.mark.parametrize('architecture', ['hierarchical', 'hierarchical_gumbel',
                                         'relation_aware_hub', 'independent_edge'])
def test_removed_architectures_are_rejected(architecture):
    with pytest.raises(ValueError, match='shared edge-calibrator'):
        PromptingAdaptation(8, 8, {'prompt_edge_denoising': {'architecture': architecture}})


@pytest.mark.parametrize('name', ['cora', 'citeseer', 'pubmed', 'wikics', 'actor',
                                  'chameleon', 'squirrel', 'minesweeper'])
def test_main_config_constructs_paper_model_and_roundtrips(name):
    config = yaml.safe_load((ROOT / f'configs/main/{name}.yaml').read_text())
    assert config['experiment']['prompt_variant'] == 'prompting_adaptation'
    cfg = dict(config['prompt_graph'], num_classes=3)
    model = PromptingAdaptation(8, 8, cfg)
    clone = PromptingAdaptation(8, 8, cfg)
    clone.load_state_dict(model.state_dict(), strict=True)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, clone.state_dict()[key], rtol=0, atol=0)
