from pathlib import Path

import pytest

from veloroute.artifacts import load_config
from veloroute.kang_cellot import run_cellot


@pytest.mark.parametrize('policy', [
    {},
    {'cellot': {'training_enabled': False}},
    {'cellot': {'execution': 'reuse_existing_only', 'training_enabled': True}},
    {'cellot': {'execution': 'train', 'training_enabled': False}},
])
@pytest.mark.parametrize('smoke_steps', [None, 2])
def test_cellot_disabled_before_io_or_gpu_use(tmp_path, monkeypatch, policy, smoke_steps):
    def forbidden(*args, **kwargs):
        pytest.fail('Disabled CellOT training accessed data or the GPU')

    monkeypatch.setattr('veloroute.kang_cellot.enforce_gpu_policy', forbidden)
    monkeypatch.setattr('veloroute.kang_cellot.official_modules', forbidden)
    monkeypatch.setattr('veloroute.kang_cellot.load_pack', forbidden)
    output = tmp_path/'must_not_be_created'
    with pytest.raises(RuntimeError, match='CellOT training disabled'):
        run_cellot({}, policy, '101', 0, output, smoke_steps=smoke_steps)
    assert not output.exists()


def test_registered_cellot_policy_is_reuse_only():
    config = load_config(Path(__file__).resolve().parents[1]/'configs/veloroute_kang_extended_20260914.yaml')
    policy = config['cellot']
    assert policy['execution'] == 'reuse_existing_only'
    assert policy['training_enabled'] is False
    assert policy['source_acquisition_enabled'] is False
    assert 'iterations' not in policy
