import json

import numpy as np
import pytest

from veloroute.experiments import condition_vectors, validate_pair, train_model, evaluate_prediction
from veloroute.latent import save_pack
from veloroute.pipeline_smoke import run_pipeline_smoke
from veloroute.probes import permute_local


def test_permutation_is_bijective_and_stratum_local():
    source = {'z': np.arange(60).reshape(20, 3), 'depth': np.repeat([100, 300], 10),
              'conditions': np.tile(np.repeat(['A', 'B'], 5), 2)}
    values = np.arange(40).reshape(20, 2)
    permuted, indices = permute_local(values, source, seed=1, neighbors=3)
    assert sorted(indices) == list(range(20))
    np.testing.assert_equal(source['conditions'][indices], source['conditions'])
    np.testing.assert_equal(source['depth'][indices], source['depth'])
    np.testing.assert_equal(permuted, values[indices])


def test_embeddings_never_silently_fall_back(tmp_path):
    path = tmp_path/'conditions.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, conditions=np.array(['A']), embeddings=np.ones((1, 3)),
            metadata_json=np.array(json.dumps({'source': 'synthetic', 'frozen': True})))
    with pytest.raises(ValueError, match='Missing condition'):
        condition_vectors(path, ['B'])


def test_whole_file_pipeline(tmp_path):
    summary = run_pipeline_smoke(tmp_path/'pipeline', seed=2, steps=6)
    assert summary['status'] == 'ENGINEERING_PASS'
    assert summary['evaluation_rows'] == 6
    assert not summary['confirmation_evaluated']
    path = tmp_path/'pipeline'
    # Real raw data cannot use the synthetic engineering waiver.
    with np.load(path/'fold/train_source.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != 'metadata_json'}
        sm = json.loads(str(data['metadata_json']))
    with np.load(path/'fold/train_target.npz', allow_pickle=False) as data:
        target = {key: data[key] for key in data.files if key != 'metadata_json'}
        tm = json.loads(str(data['metadata_json']))
    sm['kind'] = tm['kind'] = 'engineering'
    save_pack(path/'real_source.npz', arrays, sm)
    save_pack(path/'real_target.npz', target, tm)
    with pytest.raises(ValueError, match='real data needs G1'):
        train_model(path/'real_source.npz', path/'real_target.npz', path/'synthetic_conditions.npz', {},
                    path/'forbidden_train', engineering=True)
    # Tampering with frozen predictions blocks evaluation.
    prediction = path/'predict_real/predictions.npz'
    with prediction.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='modified'):
        evaluate_prediction(path/'predict_real', path/'fold/validation_target.npz', path/'forbidden_eval')
