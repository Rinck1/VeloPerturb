import json

import numpy as np
import yaml

from veloroute.artifacts import save_json, sha256
from veloroute.latent import save_pack
from veloroute.protocol import REQUIRED_CHECKS, freeze_protocol
from veloroute.probes import run_g1


def test_g1_reads_future_only_after_all_predictions_frozen(tmp_path, monkeypatch):
    # These approval flags are an isolated unit-test fixture, NOT a real G1 run.
    rng = np.random.default_rng(1)
    paths = {}
    for role, labels in [('train', ['TOY_A', 'TOY_B']), ('validation', ['TOY_C', 'TOY_D'])]:
        for side, day in [('source', 4), ('target', 5)]:
            labels_array = np.repeat(labels, 20)
            z = rng.normal(size=(40, 3)).astype(np.float32)
            arrays = dict(z=z, cell_ids=np.array([f'TOY:{role}:{side}:{i}' for i in range(40)]), conditions=labels_array)
            if side == 'source':
                arrays.update(velocity=rng.normal(size=z.shape).astype(np.float32), u_features=rng.normal(size=z.shape).astype(np.float32),
                              u_predicted=np.zeros_like(z), depth=np.full(40, 100.))
            metadata = {'side': side, 'role': role, 'day': day, 'task': 'ER-short', 'kind': 'development',
                        'transform_hash': 'TOY_transform', 'fit_ids_hash': 'TOY_train_only', 'formal_ready': True,
                        'unit_test_fixture': True}
            path = tmp_path/f'{role}_{side}.npz'
            save_pack(path, arrays, metadata)
            paths[f'{role}_{side}'] = path
    conditions = tmp_path/'conditions.npz'
    with conditions.open('xb') as stream:
        np.savez_compressed(stream, conditions=np.array(['TOY_A', 'TOY_B', 'TOY_C', 'TOY_D']), embeddings=np.eye(4),
            metadata_json=np.array(json.dumps({'source': 'unit_test_fixture_NOT_real_ESM', 'frozen': True})))
    evidence = {f'{key}_sha256': sha256(path) for key, path in paths.items()}
    evidence['condition_embeddings_sha256'] = sha256(conditions)
    save_json(tmp_path/'data.json', evidence)
    (tmp_path/'splits.csv').write_text('unit_test_only\n')
    (tmp_path/'velocity.json').write_text('{"unit_test_only":true}')
    probe_config = dict(validation_conditions=['TOY_C', 'TOY_D'], arms=['A0', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'],
                        training_cells_per_condition=16, ridge_alpha=1., ot_epsilon=.1, permutation_block_size=5,
                        n_bootstrap=100, bootstrap_seed=1, familywise_alpha=.05, condition_projection_dim=2, condition_projection_seed=1)
    (tmp_path/'probe.yaml').write_text(yaml.safe_dump(probe_config))
    config = dict(task='ER-short', metric='energy_distance_v_statistic', seeds=[0, 1, 2],
                  minimum_useful_gain=.01, noninferiority_margin=.01, effect_margin_rationale='UNIT TEST ONLY',
                  readiness={key: True for key in REQUIRED_CHECKS}, split_manifest=str(tmp_path/'splits.csv'),
                  data_provenance=str(tmp_path/'data.json'), velocity_provenance=str(tmp_path/'velocity.json'),
                  probe_config=str(tmp_path/'probe.yaml'))
    frozen = freeze_protocol(config)
    save_json(tmp_path/'frozen.json', frozen)
    from veloroute import probes
    original = probes.load_pack
    def guarded(path, **kwargs):
        if path == paths['validation_target']:
            records = json.loads((tmp_path/'g1/frozen_predictions.json').read_text())
            assert len(records) == 21
            assert all(sha256(row['path']) == row['sha256'] for row in records)
        return original(path, **kwargs)
    monkeypatch.setattr(probes, 'load_pack', guarded)
    result = run_g1(tmp_path/'frozen.json', paths['train_source'], paths['train_target'], paths['validation_source'],
                    paths['validation_target'], conditions, tmp_path/'g1')
    assert result['status'] in {'REVIEW_REQUIRED', 'NO_GO_OR_INSUFFICIENT'}
    assert not result['automatic_training_authorized']
