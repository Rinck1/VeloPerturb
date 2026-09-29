"""Frozen-candidate orchestration with explicitly mocked, synthetic I/O boundaries."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from veloroute import frozen_candidates as fc
from veloroute.artifacts import save_json, sha256
from veloroute.model import load_checkpoint


def design(tmp_path):
    return dict(research_status='exploratory_not_preregistered', confirmation='sealed',
        seeds=[0, 1, 2], arms=['static', 'real', 'shuffled', 'uniform'],
        fold=str(tmp_path/'fold'), conditions=str(tmp_path/'conditions.fixture'),
        model=dict(hidden_dim=8, router_hidden_dim=8, time_hidden_dim=8,
                   residual_blocks=1, expert_rank=2, n_experts=2, top_k=2, use_velocity=True),
        candidate_steps=2, router_steps=2, training_OT_samples_per_condition=6,
        candidate_labels='synthetic_test_fixture', batch_size=4, learning_rate=.0001,
        lambda_dynamic=.1, temperature=.5, epsilon=.1, sinkhorn_iterations=500,
        n_bootstrap=100, bootstrap_seed=17, familywise_alpha=.05,
        validation_conditions=['TEST_V1', 'TEST_V2'], device='cpu', cpu_threads=1)


@pytest.mark.parametrize('field,value,match', [
    ('seeds', [0], 'three seeds'),
    ('confirmation', 'open', 'sealed'),
    ('arms', ['real'], 'four-arm'),
    ('candidate_steps', 0, 'positive'),
    ('research_status', 'formal', 'exploratory'),
])
def test_invalid_design_rejected_before_reading_data(tmp_path, field, value, match):
    config = design(tmp_path)
    config[field] = value
    path = tmp_path/'design.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        fc.run_frozen_candidates(path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_freeze_all_predictions_before_future_evaluation_and_share_experts(tmp_path, monkeypatch):
    config = design(tmp_path)
    fold = tmp_path/'fold'
    fold.mkdir()
    for name in ('train_source.npz', 'train_target.npz'):
        (fold/name).write_text('MOCKED IO: synthetic test fixture, not real data')
    Path(config['conditions']).write_text('MOCKED IO: not actual protein embeddings')
    config_path = tmp_path/'design.yaml'
    config_path.write_text(yaml.safe_dump(config))
    rng = np.random.default_rng(17)
    z = rng.normal(0, .03, (12, 2)).astype('float32')
    source = dict(z=z, velocity=rng.normal(0, .03, z.shape).astype('float32'),
                  conditions=np.repeat(['TEST_A', 'TEST_B'], 6), depth=np.full(12, 100),
                  cell_ids=np.array([f'TEST_SOURCE_{i}' for i in range(12)]))
    target = dict(z=z+.01, conditions=source['conditions'],
                  cell_ids=np.array([f'TEST_TARGET_{i}' for i in range(12)]))
    metadata = dict(role='train', transform_hash='TEST', fit_ids_hash='TEST',
                    kind='engineering', task='ER-short')
    # Mock the dataset/provider boundary; never relabel actual synthetic packs for production.
    def mock_load(path, *, expected_side):
        assert Path(path).name == f'train_{expected_side}.npz'
        return (source, {**metadata, 'day': 4}) if expected_side == 'source' else (target, {**metadata, 'day': 5})
    monkeypatch.setattr(fc, 'load_pack', mock_load)
    monkeypatch.setattr(fc, 'condition_vectors', lambda *_: (
        np.zeros((12, 3), dtype='float32'), {'kind': 'engineering', 'source': 'MOCKED_TEST_PRIOR'}))
    root = tmp_path/'output'
    events = []

    def mock_predict(checkpoint, source_path, condition_path, output, *, seed, transform_path):
        assert Path(source_path).name == 'validation_source.npz'
        Path(output).mkdir()
        Path(output, 'predictions.npz').write_bytes(b'SYNTHETIC TEST PLACEHOLDER')
        events.append('predict')

    def mock_evaluate(prediction, target_path, output, *, target_genes_path):
        assert events.count('predict') == 12
        frozen = json.loads((root/'frozen_predictions.json').read_text())
        assert len(frozen) == 12
        for record in frozen:
            assert sha256(Path(record['directory'])/'predictions.npz') == record['sha256']
        events.append('evaluate')
        seed = int(Path(prediction).parent.name[-1])
        return [dict(condition=label, arm='replaced_by_runner', seed=seed, energy_distance=1.)
                for label in config['validation_conditions']]

    monkeypatch.setattr(fc, 'predict_model', mock_predict)
    monkeypatch.setattr(fc, 'evaluate_prediction', mock_evaluate)
    summary = fc.run_frozen_candidates(config_path, root)
    assert summary['experts_verified_frozen'] and not summary['confirmation_evaluated']
    assert events == ['predict']*12+['evaluate']*12
    assert all(comparison['gain'] == 0 for comparison in summary['comparisons'])
    for seed in config['seeds']:
        candidates, _ = load_checkpoint(root/f'candidates_seed{seed}/model.pt')
        for arm in config['arms']:
            model, meta = load_checkpoint(root/f'{arm}_seed{seed}/train/model.pt')
            assert meta['experts_frozen']
            for name, tensor in candidates.field.state_dict().items():
                assert torch.equal(tensor, model.field.state_dict()[name])
            if arm == 'uniform':
                assert all(torch.count_nonzero(parameter) == 0 for parameter in model.router.parameters())
