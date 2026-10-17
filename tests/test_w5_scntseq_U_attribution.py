import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w5_scntseq_U_attribution as attribution
import w5_scntseq_minutes as minutes
import w5_scntseq_repair as repair
from w4_streams import SourceTransform
from test_w5_scntseq_minutes import fixture


def test_U_factorial_separates_composition_and_total():
    base = dict(U=np.array([2.,8.]),S=np.array([10.,10.]),ratio=.2)
    source = dict(U=np.array([12.,8.]),S=np.array([5.,15.]),ratio=.4)
    np.testing.assert_array_equal(attribution.counterfactual_u(source,base,'current','current'),source['U'])
    np.testing.assert_array_equal(attribution.counterfactual_u(source,base,'baseline','current'),2*base['U'])
    np.testing.assert_array_equal(attribution.counterfactual_u(source,base,'current','baseline'),.5*source['U'])
    np.testing.assert_array_equal(attribution.counterfactual_u(source,base,'baseline','baseline'),base['U'])
    for composition in ('current','baseline'):
        for total in ('current','baseline'):
            fields = attribution.field_components(base,base,np.ones(2),np.eye(2),composition,total)
            for value in fields.values():np.testing.assert_array_equal(value,np.zeros(2))
    with pytest.raises(ValueError):attribution.counterfactual_u(source,base,'shuffle','current')


def test_equal_cell_pool_removes_cell_depth_weights_but_preserves_ratios():
    transform = SimpleNamespace(target_sum=100.,genes=np.arange(2))
    transform.normalized = lambda s,u: (np.asarray(s.toarray() if sparse.issparse(s) else s)*100/np.asarray(s.sum(1)),
        np.asarray(u.toarray() if sparse.issparse(u) else u)*100/np.asarray(s.sum(1)))
    transform.state_normalized = lambda s: np.log1p(s)
    s = sparse.csr_matrix([[1.,3.],[3.,1.]])
    u = sparse.csr_matrix([[1.,0.],[0.,2.]])
    first,_ = attribution.pooled_statistics(transform,s,u)
    scale = sparse.diags([100.,1.])
    second,_ = attribution.pooled_statistics(transform,scale@s,scale@u)
    for key in ('S','U','ratio'):
        np.testing.assert_allclose(first['equal_cell'][key],second['equal_cell'][key])
    assert not np.allclose(first['depth_weighted']['U'],second['depth_weighted']['U'])
    assert first['equal_cell']['ratio'] == pytest.approx(.375)


def test_full_attribution_preserves_original_scores_and_target_order(fixture,tmp_path,monkeypatch):
    repo,original_config,commit,obs = fixture
    first = tmp_path/'original';minutes.predict(original_config,first,commit,repo=repo)
    old = json.loads(original_config.read_text())
    rc = json.loads((repair.ROOT/'configs/w5_scntseq_repair_v2.json').read_text())
    rc.update(original_predictions=str(first),data_path=old['data_path'],data_sha256=old['data_sha256'])
    repair_config = repo/'configs/repair.json';repair_config.write_text(json.dumps(rc))
    (repo/'PLANS/w5_scntseq_repair_v2.md').write_text('Fixture repair\n')
    repaired = tmp_path/'repair';repair.predict(repair_config,repaired,repo=repo)
    original_hash = minutes.sha256(repaired/'predictions.npz')
    config = json.loads((repair.ROOT/'configs/w5_scntseq_U_attribution_v1.json').read_text())
    config.update(repair_predictions=str(repaired))
    config_path = repo/'configs/attribution.json';config_path.write_text(json.dumps(config))
    (repo/config['protocol_path']).write_text('Fixture attribution\n')
    output = tmp_path/'attribution'
    original_read = minutes.read_count_rows
    done = False
    def watch(path,layer,rows):
        minute = set(obs.iloc[rows]['time_minutes']).pop()
        if minute == 60:assert (output/'prediction_30_to_60_manifest.json').is_file()
        if minute == 120:assert done and layer == 'spliced'
        return original_read(path,layer,rows)
    monkeypatch.setattr(minutes,'read_count_rows',watch)
    attribution.predict(config_path,output,repo=repo)
    with np.load(output/'predictions.npz',allow_pickle=False) as f:
        assert f['predictions'].shape == (2,20,3)
        assert np.isfinite(f['predictions']).all()
    done = True
    result = attribution.evaluate(output,tmp_path/'evaluation',repo=repo)
    assert result['models'] == 20 and not result['original_results_changed']
    assert not result['technical_vs_biological_total_identified']
    assert minutes.sha256(repaired/'predictions.npz') == original_hash
