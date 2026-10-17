import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w5_scntseq_minutes as minutes
from w4_artifacts import sha256
from w4_streams import SourceTransform


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path/'repo'; repo.mkdir()
    rng = np.random.default_rng(9)
    times = np.repeat([0,15,30,60,120], 30)
    gene = np.linspace(60, 140, 12)
    s = rng.poisson(np.broadcast_to(gene, (len(times),12)) + times[:,None]*.1)
    u = rng.poisson(np.broadcast_to(gene*.5, (len(times),12)) + times[:,None]*.08)
    new = rng.poisson(20, s.shape)
    total = new+rng.poisson(90, s.shape)
    obs = pd.DataFrame(dict(time_minutes=times, spliced_counts_common_genes=s.sum(1),
        unspliced_counts_common_genes=u.sum(1)), index=[f'cell{i}' for i in range(len(times))])
    a = ad.AnnData(sparse.csr_matrix(total), obs=obs, var=pd.DataFrame(index=[f'g{i}' for i in range(12)]))
    for key,value in dict(spliced=s, unspliced=u, new=new, total=total).items():
        a.layers[key] = sparse.csr_matrix(value)
    data = repo/'fixture.h5ad'; a.write_h5ad(data)
    (repo/'PLANS').mkdir(); (repo/'configs').mkdir()
    (repo/'PLANS/w5_prereg_addendum.md').write_text('# Synthetic test\n\n状态：**已冻结**\n')
    config = json.loads((minutes.ROOT/'configs/w5_scntseq_minutes_v1.json').read_text())
    config.update(data_path=str(data), data_sha256=sha256(data), n_hvg=8, dimensions=3, clusters=2)
    path = repo/'configs/config.json'; path.write_text(json.dumps(config))
    subprocess.run(['git','init','-q',str(repo)], check=True)
    subprocess.run(['git','-C',str(repo),'add','PLANS','configs'], check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
        'commit','-qm','synthetic frozen external protocol'], check=True)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    return repo, path, commit, obs


def test_nonnegative_scale_and_physical_duration():
    feature = np.array([1.,2.,3.])
    scale, degenerate = minutes.nonnegative_scale(feature, feature*2)
    assert scale == 2 and not degenerate
    assert minutes.nonnegative_scale(feature, -feature)[0] == 0
    assert minutes.nonnegative_scale(np.zeros(3), feature) == (0., True)
    vectors = {arm:feature for arm in minutes.ARMS}
    scales = {arm:2. for arm in minutes.ARMS}
    np.testing.assert_allclose(minutes.forecast(np.ones(3), vectors, scales, 30),
        np.broadcast_to(1+60*feature, (len(minutes.ARMS),3)))


def test_first_prediction_freezes_before_next_source_and_last_target_stays_sealed(fixture, tmp_path, monkeypatch):
    repo, config_path, commit, obs = fixture
    output = tmp_path/'predictions'
    original, accesses = minutes.read_count_rows, []
    prediction_done = False
    def watch(path, layer, rows):
        time = set(obs.iloc[rows]['time_minutes'])
        assert len(time) == 1
        time = next(iter(time)); accesses.append((time,layer))
        if time == 60:
            assert (output/'prediction_30_to_60_manifest.json').is_file()
        if time == 120:
            assert prediction_done, '120-minute target cannot enter prediction'
            assert layer == 'spliced'
        return original(path,layer,rows)
    monkeypatch.setattr(minutes, 'read_count_rows', watch)
    minutes.predict(config_path, output, commit, repo=repo)
    assert not any(t == 120 for t,_ in accesses)
    manifest = json.loads((output/'prediction_manifest.json').read_text())
    assert manifest['source_transform_fit_minutes'] == [0]
    assert manifest['scale_fit_minutes'] == [15,30]
    transform = SourceTransform.load(output/'source_transform.npz')
    assert set(transform.source_ids) == set(obs.index[obs.time_minutes == 0])
    before = sha256(output/'calibration.json')
    prediction_done = True
    result = minutes.evaluate(output, tmp_path/'evaluation', repo=repo)
    assert result['status'] == 'SCNT_MINUTES_EVALUATED'
    assert len(result['mean_mse']) == 9 and not result['significance_test_performed']
    assert not result['new_RNA_is_net_velocity'] and not result['router_enabled']
    assert sha256(output/'calibration.json') == before


def test_post_freeze_config_change_is_not_silently_used(fixture):
    repo, config_path, commit, _ = fixture
    config = json.loads(config_path.read_text())
    config['dimensions'] = 4
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='changed after freeze'):
        minutes.verify_config(config_path, commit, repo=repo)
