import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w4_learned_head as learned
from w4_arms import CORE_ARMS
from w4_shorttime import DEVELOPMENT_TFS, predict_development


def test_development_head_end_to_end_reads_only_training_target_rows(tmp_path, monkeypatch):
    rng = np.random.default_rng(13)
    names = ['TRAIN0', 'TRAIN1', 'TRAIN2', 'TRAIN3', 'TRAIN4', 'CTRL', 'AAVS1', *DEVELOPMENT_TFS]
    labels = np.repeat(names, 50)
    paths = []
    for day in (4, 5):
        s, u = [sparse.csr_matrix(rng.poisson(rate, (len(labels), 12))) for rate in (12, 5)]
        obs = pd.DataFrame(dict(condition=labels, day=day,
            technical_qc_pass=True, guide_call_pass=True,
            role=['development' if x in DEVELOPMENT_TFS else 'control' if x in ('CTRL','AAVS1') else 'train'
                  for x in labels]), index=[f'd{day}_cell{i}' for i in range(len(labels))])
        data = ad.AnnData(s.copy(), obs=obs, var=pd.DataFrame(index=[f'g{i}' for i in range(12)]))
        data.layers['spliced'], data.layers['unspliced'] = s, u
        path = tmp_path/f'day{day}.h5ad'
        data.write_h5ad(path); paths.append(path)
    embeddings = tmp_path/'conditions.npz'
    np.savez(embeddings, conditions=np.array(names), embeddings=rng.normal(size=(len(names), 4)),
        metadata_json=json.dumps(dict(frozen=True)))
    baseline = tmp_path/'baseline'
    predict_development(*paths, embeddings, baseline, seeds=(0,), n_hvg=10, dimensions=5)
    config = json.loads((learned.ROOT/'configs/w4_nojvp_head_development_v1.json').read_text())
    config.update(baseline_predictions=str(baseline), embeddings=str(embeddings), seeds=[0],
        steps=3, hidden=4, training_partitions_per_condition=2, diagnostic_partitions_per_condition=1)
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    original, accesses = learned.read_count_rows, []
    def read(path, layer, rows):
        if Path(path) == paths[1]:
            accesses.extend(np.asarray(rows).tolist())
        return original(path, layer, rows)
    monkeypatch.setattr(learned, 'read_count_rows', read)
    output = tmp_path/'predictions'
    learned.predict(config_path, output)
    assert accesses and not set(labels[accesses]) & set(DEVELOPMENT_TFS)
    with np.load(output/'predictions.npz') as actual, np.load(baseline/'predictions.npz') as expected:
        for arm in CORE_ARMS:
            np.testing.assert_array_equal(actual[arm], expected[arm])
        for arm in learned.ARMS:
            assert actual[arm].shape == (1, 4, 5) and np.isfinite(actual[arm]).all()
    summary = learned.evaluate(output, tmp_path/'evaluation')
    assert summary['status'] == 'NOJVP_HEAD_DEVELOPMENT_EVALUATED'
    assert set(summary['mse']) == set(CORE_ARMS) | set(learned.ARMS)
    assert not summary['router_enabled'] and not summary['head_future_supervision']
