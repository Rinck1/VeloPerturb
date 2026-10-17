import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w5_scntseq_minutes as minutes
import w5_scntseq_repair as repair
from w4_streams import SourceTransform
from test_w5_scntseq_minutes import fixture


def test_regularized_gamma_borrows_source_prior_without_zero_or_future_input():
    s,u = np.array([10.,2000.]),np.array([0.,4000.])
    gamma,info = repair.regularized_gamma(s,u)
    assert np.isfinite(gamma).all() and (gamma > 0).all()
    assert info['prior_only_gene_indices'] == [0]
    assert not info['rates_are_measured_physical_decay']
    tiny,_ = repair.regularized_gamma(s,u,prior=1e-8)
    np.testing.assert_allclose(tiny,u/s,atol=1e-8)
    with pytest.raises(ValueError):
        repair.regularized_gamma(s,np.zeros_like(u))


def test_finite_horizon_preserves_calibration_and_does_not_grow_unbounded():
    for kernel in ('linear','cap','decay'):
        assert repair.effective_minutes(15,15,kernel) == 15
    assert repair.effective_minutes(60,15,'linear') == 60
    assert repair.effective_minutes(60,15,'cap') == 15
    assert 15 < repair.effective_minutes(60,15,'decay') < 15/(-np.expm1(-1))
    np.testing.assert_allclose(repair.effective_minutes(1e6,15,'decay'),15/(-np.expm1(-1)))


def test_ridge_shrinks_scale_but_velocity_cannot_be_flipped():
    f = np.array([1.,0.,0.])
    fitted = repair.fit_readout([f],2*f,[True],ridge=1)
    np.testing.assert_allclose(fitted['coefficients'],[1.])
    np.testing.assert_allclose(repair.fit_readout([f],-2*f,[True])['coefficients'],[0.],atol=1e-8)
    np.testing.assert_allclose(repair.fit_readout([f],-2*f,[False])['coefficients'],[-1.])


def test_baseline_field_centering_is_zero_at_baseline():
    population = dict(mean_u=np.array([1.,2.]),mean_s=np.array([.2,.8]),
        pool_u=np.array([1.5,2.5]),pool_s=np.array([.4,.6]))
    for aggregation in ('mean','pool'):
        v = repair.velocity_feature(population,population,np.ones(2),np.eye(2),aggregation,'baseline0')
        np.testing.assert_array_equal(v,np.zeros(2))


def test_mechanism_decomposition_adds_to_centered_velocity():
    from w5_scntseq_repair_mechanism import components
    base = dict(pool_s=np.array([.3,.6]),pool_u=np.array([.2,.4]),
        normalized_pool_s=np.array([1.,2.]),normalized_pool_u=np.array([.4,1.2]),full_U_to_S=.5)
    source = dict(pool_s=np.array([.4,.7]),pool_u=np.array([.3,.6]),
        normalized_pool_s=np.array([2.,3.]),normalized_pool_u=np.array([.9,2.4]),full_U_to_S=.6)
    gamma = np.array([.2,.4])
    values = components(source,base,gamma,np.eye(2))
    expected = (source['pool_u']-gamma*source['pool_s'])-(base['pool_u']-gamma*base['pool_s'])
    np.testing.assert_allclose(values['S_kinetic']+values['U_derivative'],expected)


def test_repair_sufficient_statistics_match_original_velocity(fixture):
    repo,config_path,commit,_ = fixture
    config,_,data = minutes.verify_config(config_path,commit,repo=repo)
    obs,genes = minutes.read_metadata(data)
    layers,ids = minutes.load_population(data,obs,0,config,source=True)
    t = SourceTransform.fit(layers['spliced'],layers['unspliced'],cell_ids=ids,allowed_ids=set(ids),
        gene_names=genes,n_hvg=8,dimensions=3,clusters=2)
    state,old = minutes.population_features(t,layers)
    new = repair.sufficient_features(t,layers)
    np.testing.assert_allclose(state,new['state'])
    for aggregation in ('mean','pool'):
        actual = repair.velocity_feature(new,new,t.gamma,t.components,aggregation,'raw')
        np.testing.assert_allclose(actual,old[f'vel_{aggregation}'],atol=1e-12)


def test_full_repair_grid_does_not_fit_test_targets_or_change_v1(fixture,tmp_path,monkeypatch):
    repo,original_config,commit,obs = fixture
    first = tmp_path/'original'
    minutes.predict(original_config,first,commit,repo=repo)
    original_digest = minutes.sha256(first/'predictions.npz')
    old = json.loads(original_config.read_text())
    config = json.loads((repair.ROOT/'configs/w5_scntseq_repair_v2.json').read_text())
    config.update(original_predictions=str(first),data_path=old['data_path'],data_sha256=old['data_sha256'])
    config_path = repo/'configs/repair.json';config_path.write_text(json.dumps(config))
    (repo/'PLANS/w5_scntseq_repair_v2.md').write_text('Synthetic exploratory repair fixture\n')
    output = tmp_path/'repair'
    original_read = minutes.read_count_rows
    prediction_done = False
    def watch(path,layer,rows):
        time = set(obs.iloc[rows]['time_minutes']);assert len(time) == 1
        minute = next(iter(time))
        if minute == 60:
            assert (output/'prediction_30_to_60_manifest.json').is_file()
        if minute == 120:
            assert prediction_done and layer == 'spliced'
        return original_read(path,layer,rows)
    monkeypatch.setattr(minutes,'read_count_rows',watch)
    repair.predict(config_path,output,repo=repo)
    assert len(repair.model_specs(config)) == 67
    with np.load(output/'predictions.npz') as saved:
        assert saved['predictions'].shape == (2,67,3) and np.isfinite(saved['predictions']).all()
    prediction_done = True
    result = repair.evaluate(output,tmp_path/'evaluation',repo=repo)
    assert result['exploratory'] and not result['independent_confirmation']
    assert not result['original_results_changed']
    assert result['compared_models'] == 67
    assert minutes.sha256(first/'predictions.npz') == original_digest
