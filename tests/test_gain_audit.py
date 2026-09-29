"""Regression checks for the 2026-09-21 training/provenance audit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from veloroute.velocity_background import BACKGROUND_FORMAT, load_background
from veloroute.velocity_diagnostics import centered_velocity_r2


def script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/'phase0'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_background_rejects_legacy_and_reordered_cells(tmp_path):
    path = tmp_path/'old.npz'
    np.savez(path, train_r_bg=np.zeros((2, 4)), train_v_bg=np.zeros((2, 3)))
    with pytest.raises(ValueError, match='regenerate'):
        load_background(path, 'train')
    path = tmp_path/'new.npz'
    np.savez(path, format=BACKGROUND_FORMAT, train_cell_ids=np.array(['a', 'b']),
             train_r_bg=np.zeros((2, 4)), train_v_bg=np.zeros((2, 3)))
    assert load_background(path, 'train', cell_ids=['a', 'b'])['v'].shape == (2, 3)
    with pytest.raises(ValueError, match='order'):
        load_background(path, 'train', cell_ids=['b', 'a'])


def test_background_normalization_fits_only_training_cells(monkeypatch):
    module = script('precompute_velocity_background')
    prepared = []
    def prepare(values, components, *, cell_ids, scope):
        prepared.append(list(cell_ids))
    model = SimpleNamespace(dynamics=SimpleNamespace(prepare=prepare),
        make_context=lambda s, u, t, m, **kw: SimpleNamespace(representation=s, velocity=s))
    monkeypatch.setattr(module, 'build_model', lambda *a, **kw:
                        (model, SimpleNamespace(components=np.eye(3)), torch.device('cpu')))
    def read(gene, source):
        role = 'train' if source.name.startswith('train') else 'validation'
        pack = dict(z=np.ones((2, 3)), cell_ids=np.array([f'{role}:0', f'{role}:1']),
                    conditions=np.array(['A', 'A']))
        return np.ones((2, 6)), pack, {}
    monkeypatch.setattr(module, 'read_gene_input', read)
    monkeypatch.setattr(module, 'condition_token_sets', lambda *a:
                        (np.ones((2, 1, 5)), np.ones((2, 1), dtype=bool), None))
    result = module.contexts(Path('fold'), Path('genes'), Path('conditions.npz'))
    assert prepared == [['train:0', 'train:1']]
    assert result['validation']['cell_ids'].tolist() == ['validation:0', 'validation:1']


def test_mainline_evaluation_records_actual_seed(monkeypatch, tmp_path):
    module = script('run_gain_mainline')
    pack = dict(z=np.array([[0., 0.], [1., 1.]]), conditions=np.array(['A', 'A']))
    monkeypatch.setattr(module, 'load_pack', lambda *a, **kw: (pack, {}))
    path = tmp_path/'predictions.npz'
    np.savez(path, **pack, seed=17)
    rows = module.evaluate(tmp_path, path, 'gfg_joint', seed=17)
    assert rows[0]['seed'] == 17
    with pytest.raises(ValueError, match='seed mismatch'):
        module.evaluate(tmp_path, path, 'gfg_joint', seed=0)


def test_constant_direction_cannot_masquerade_as_cellwise_variance_explained():
    observed = np.array([[99.], [101.], [99.], [101.]])
    predicted = np.full_like(observed, 100.)
    uncentered = 1.-np.square(observed-predicted).sum()/np.square(observed).sum()
    assert uncentered > .999
    assert centered_velocity_r2(observed, predicted) == pytest.approx(0.)
    assert centered_velocity_r2(observed, observed) == pytest.approx(1.)
