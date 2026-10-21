"""No-retraining population-token masks, common IDs, and all-forecast freeze."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15a_cellflow_common_evaluation as common
import w15a_l2_token_intervention as intervention
from w4_artifacts import sha256
from test_w15a_common_evaluation import reference
from test_w15a_l2_reporter import complete_grid


@pytest.fixture
def intervention_grid(complete_grid,reference,tmp_path,monkeypatch):
    study=json.loads((complete_grid/'study_manifest.json').read_text());pack=Path(study['pack'])
    old=json.loads((reference/'study_manifest.json').read_text());old_pm=json.loads((Path(old['pack'])/'manifest.json').read_text())
    records=[]
    for row in old_pm['prediction_groups']:
        with np.load(row['path'],allow_pickle=False) as f:z=f['z'][::-1].copy();ids=f['cell_ids'][::-1].copy()
        start=row['source_hour'];path=pack/f'source_{start}.npz'
        np.savez_compressed(path,z=z,cell_ids=ids,cell_kin=z+7.,
            population_kin=np.full(3,(start-5)/5.),population_Ssum=np.zeros(3),population_directU=np.zeros(3))
        records.append(dict(time_hours=start,path=str(path),sha256=sha256(path),source_key=f'full_source_{start}'))
    (pack/'manifest.json').write_text(json.dumps(dict(development_sources=records)))
    np.savez_compressed(pack/'token_normalizer.npz',training_source_keys=np.array(['train']),
        **{'mean_'+k:np.ones(3) for k in ('kin','Ssum','directU','cell_kin')},
        **{'scale_'+k:np.full(3,2.) for k in ('kin','Ssum','directU','cell_kin')})
    for seed in (0,1,2):
        path=complete_grid/'cf_kin'/f'seed{seed}'/'train/summary.json';m=json.loads(path.read_text())
        m.update(pack_directory=str(pack),config={});path.write_text(json.dumps(m))
    monkeypatch.setattr(intervention,'restore_solver',lambda training,summary,*a:SimpleNamespace(seed=summary['seed']))
    calls=[]
    def integrate(solver,x,conditions,**kwargs):
        assert len(x)==8 and conditions['population_token'].shape==(8,1,3)
        assert np.all(conditions['cell_token']==0)
        # Original toy source05 endpoint: x+.25+.001*seed. Neither clock nor
        # source/time IDs are mutated by replacing the group token.
        calls.append(dict(seed=solver.seed,token=conditions['population_token'].copy(),tau=kwargs['tau_end']))
        return x+.25+.001*solver.seed+(conditions['population_token'][:,0,:]+.5)*.01
    monkeypatch.setattr(intervention,'predict_cellflow_tau',integrate)
    return complete_grid,reference,pack,calls


def test_two_masks_are_distinct_normalized_vectors_and_preserve_every_other_input():
    conditions=dict(population_token=np.ones((4,1,3)),cell_token=np.zeros((4,1,3)),
        bio_time=np.arange(8).reshape(4,2),bio_delta=np.full((4,2),24.),perturbation=np.ones((4,1,1)))
    ref=np.array([-.5,-1.,-2.])
    zero=intervention.mask_conditions(conditions,'training_mean',ref)
    baseline=intervention.mask_conditions(conditions,'source05_reference',ref)
    assert np.all(zero['population_token']==0)
    np.testing.assert_array_equal(baseline['population_token'][0,0],ref)
    assert np.all(conditions['population_token']==1)
    for key in conditions:
        if key!='population_token':
            np.testing.assert_array_equal(zero[key],conditions[key]);np.testing.assert_array_equal(baseline[key],conditions[key])
    with pytest.raises(ValueError,match='two approved'):intervention.mask_conditions(conditions,'new_mask',ref)


def test_no_approval_means_no_data_access():
    with pytest.raises(ValueError,match='approval'):
        intervention.run_intervention('missing_grid','missing_output')


def test_all108_forecasts_frozen_before_target_score_and_original_grid_untouched(intervention_grid,tmp_path,monkeypatch):
    grid,reference,pack,calls=intervention_grid;output=tmp_path/'intervention'
    reference_info=common.reference_sources(reference);target_paths={str(r['path']) for r in reference_info['target_caches'].values()}
    original=np.load;reads=[];weights={s:sha256(grid/'cf_kin'/f'seed{s}'/'train/model.msgpack') for s in (0,1,2)}
    grid_status=(grid/'status.json').read_bytes();original_metrics=(grid/'evaluation/summary.json').read_bytes()
    def checked(path,*a,**kw):
        if str(path) in target_paths:
            frozen=json.loads((output/'whole_intervention_frozen.json').read_text())
            assert frozen['models']==9 and frozen['prediction_windows']==108
            assert len(frozen['predictions'])==9 and len(calls)==72
            reads.append(str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(np,'load',checked)
    result=intervention.run_intervention(grid,output,approved=True)
    assert len(calls)==72 and len(reads)==3 and result['models_trained']==0
    assert result['all108_predictions_frozen_before_intervention_scoring']
    assert result['original_energy_reproduction_max_error']==0.
    assert result['origin05_reference_endpoint_max_error']<1e-5
    assert not result['definition']['masks_identical']
    assert len(result['horizon_comparisons'])==12
    assert (grid/'status.json').read_bytes()==grid_status
    assert (grid/'evaluation/summary.json').read_bytes()==original_metrics
    assert all(sha256(grid/'cf_kin'/f'seed{s}'/'train/model.msgpack')==weights[s] for s in weights)
    assert json.loads((output/'status.json').read_text())['status']=='complete'


def test_reference05_must_be_exact_raw_zero(intervention_grid,tmp_path):
    grid,reference,pack,calls=intervention_grid
    manifest=json.loads((pack/'manifest.json').read_text());record=next(r for r in manifest['development_sources'] if r['time_hours']==5)
    path=Path(record['path']);data=intervention.load_npz(path);data['population_kin']=np.ones(3)
    np.savez_compressed(path,**data);record['sha256']=sha256(path)
    (pack/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='zero exactly'):
        intervention.predict_masked_pair(grid/'cf_kin/seed0/train',tmp_path/'bad',reference)
    assert not calls


def test_origin05_identity_mismatch_stops_before_any_target_score(intervention_grid,tmp_path,monkeypatch):
    grid,reference,pack,calls=intervention_grid
    original=intervention.predict_cellflow_tau
    monkeypatch.setattr(intervention,'predict_cellflow_tau',lambda *a,**kw:original(*a,**kw)+10.)
    targets={str(r['path']) for r in common.reference_sources(reference)['target_caches'].values()};native=np.load
    def reject(path,*a,**kw):
        assert str(path) not in targets,'No target may open when source05 replay fails'
        return native(path,*a,**kw)
    monkeypatch.setattr(np,'load',reject)
    with pytest.raises(ValueError,match='reproduce original'):
        intervention.run_intervention(grid,tmp_path/'bad_identity',approved=True)


def test_changed_frozen_checkpoint_refused_before_predictions(intervention_grid,tmp_path):
    grid,reference,pack,calls=intervention_grid
    (grid/'cf_kin/seed0/train/model.msgpack').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checkpoint changed'):
        intervention.run_intervention(grid,tmp_path/'bad_model',approved=True)
    assert not calls
