"""Common-source identity and three-seed whole-grid freeze regression."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
import w15a_cellflow_common_evaluation as common
from w15a_cellflow_adapter import array_hash


@pytest.fixture
def reference(tmp_path):
    root=tmp_path/'reference';root.mkdir();pack=tmp_path/'old_pack';pack.mkdir()
    groups=[];samples={};caches=[];rng=np.random.default_rng(3)
    for start in (5,10,15):
        z=rng.normal(size=(12,3)).astype('float32');ids=np.array([f'source_{start}_{i}' for i in range(12)])
        path=pack/f'source_{start}.npz';np.savez_compressed(path,z=z,cell_ids=ids)
        groups.append(dict(source_hour=start,target_hour=start+5,path=str(path),sha256=sha256(path)))
        sample=rng.permutation(12)[:8];samples[f'{start}_to_{start+5}']=sample.tolist()
        cache=root/f'target_{start+5}.npz'
        np.savez_compressed(cache,target=z[sample]+.1,target_rows=np.arange(8))
        caches.append(dict(source_hour=start,target_hour=start+5,path=str(cache),sha256=sha256(cache)))
    (pack/'manifest.json').write_text(json.dumps(dict(prediction_groups=groups)))
    (root/'source_samples.json').write_text(json.dumps(samples))
    (root/'study_manifest.json').write_text(json.dumps(dict(pack=str(pack),pack_manifest_sha256=sha256(pack/'manifest.json'),
        evaluation_caches=caches)))
    return root


def forecasts(reference,*,weighted=False):
    r=common.reference_sources(reference);result={}
    for key,(start,stop,clock) in common.KEYS.items():
        x=r['sources'][start]['z']+.1
        if weighted:
            result[key]=dict(x=np.repeat(x[:,None,:],2,axis=1),q=np.tile([.25,.75],(len(x),1)),
                source_ids=r['sources'][start]['cell_ids'])
        else:result[key]=dict(x=x,source_ids=r['sources'][start]['cell_ids'])
    return result


def make_grid(reference,tmp_path,*,arm='cf_static',weighted=False):
    paths=[]
    for seed in (0,1,2):
        path=tmp_path/f'{arm}_seed{seed}'
        common.save_common_predictions(path,forecasts(reference,weighted=weighted),reference_root=reference,
            arm=arm,seed=seed,model_family='CellFlow_L3' if weighted else 'CellFlow_L2')
        paths.append(path)
    return paths


def test_source_alignment_is_by_identity_not_position():
    expected=np.array(['c','a']);actual=np.array(['a','b','c'])
    np.testing.assert_array_equal(common.align_source_ids(expected,actual),[2,0])
    with pytest.raises(ValueError,match='missing'):common.align_source_ids(['z'],actual)
    with pytest.raises(ValueError,match='Duplicate'):common.align_source_ids(['a'],['a','a'])


def test_three_same_named_manifests_use_unique_frozen_index(reference,tmp_path,monkeypatch):
    paths=make_grid(reference,tmp_path);target_paths={str(r['path']) for r in common.reference_sources(reference)['target_caches'].values()}
    original=np.load;target_reads=[];output=tmp_path/'evaluation'
    def checked(path,*args,**kwargs):
        if str(path) in target_paths:
            assert (output/'frozen_grid.json').is_file()
            target_reads.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',checked)
    result=common.evaluate_common(paths,output,reference_root=reference,expected_arms=['cf_static'])
    assert result['expected_seeds']==[0,1,2] and result['windows_per_arm']==12
    assert len(result['metrics'])==12 and len(target_reads)==3
    assert len(json.loads((output/'frozen_grid.json').read_text())['predictions'])==3


def test_missing_whole_seed_refused_before_target_reads(reference,tmp_path,monkeypatch):
    paths=make_grid(reference,tmp_path);targets={str(r['path']) for r in common.reference_sources(reference)['target_caches'].values()}
    original=np.load
    def reject(path,*a,**kw):
        assert str(path) not in targets,'Target read before complete grid'
        return original(path,*a,**kw)
    monkeypatch.setattr(np,'load',reject)
    with pytest.raises(ValueError,match='Incomplete'):
        common.evaluate_common(paths[:2],tmp_path/'bad',reference_root=reference)


def test_each_seed_requires_all_twelve_keys(reference,tmp_path):
    paths=make_grid(reference,tmp_path);p=paths[1]/'prediction_manifest.json';m=json.loads(p.read_text())
    m['windows']=m['windows'][:-1];p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='twelve'):
        common.freeze_common_grid(paths,reference_root=reference)


def test_every_seed_must_match_source_ids_elementwise(reference,tmp_path):
    paths=make_grid(reference,tmp_path);path=paths[1]/'predictions.npz'
    with np.load(path,allow_pickle=False) as f:values={k:f[k] for k in f.files}
    key=next(iter(common.KEYS));values[key+'_source_ids']=values[key+'_source_ids'][::-1]
    np.savez_compressed(path,**values)
    mp=paths[1]/'prediction_manifest.json';m=json.loads(mp.read_text());m['predictions_sha256']=sha256(path)
    for r in m['windows']:
        if r['key']==key:r['source_ids_sha256']=array_hash(values[key+'_source_ids'])
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='identical common source IDs'):
        common.freeze_common_grid(paths,reference_root=reference)


def test_tampered_prediction_refused(reference,tmp_path):
    paths=make_grid(reference,tmp_path)
    with (paths[0]/'predictions.npz').open('ab') as stream:stream.write(b'mutated')
    with pytest.raises(ValueError,match='changed'):common.freeze_common_grid(paths,reference_root=reference)


def test_weighted_candidates_and_point_measure_share_scorer(reference,tmp_path):
    points=make_grid(reference,tmp_path);weighted=make_grid(reference,tmp_path,arm='moe_static',weighted=True)
    scored=common.evaluate_common(points+weighted,tmp_path/'both',reference_root=reference,
        expected_arms=['cf_static','moe_static'])
    point={r['key']:r for r in scored['metrics'] if r['arm']=='cf_static'}
    mix={r['key']:r for r in scored['metrics'] if r['arm']=='moe_static'}
    for key in point:
        assert point[key]['energy']==pytest.approx(mix[key]['energy'],abs=1e-12)


def test_predict_reorders_full_source_context_then_subsamples(reference,tmp_path,monkeypatch):
    # No GPU or real model is trained: official integration is separately tested.
    r=common.reference_sources(reference);sourcepack=tmp_path/'new_pack';sourcepack.mkdir();records=[]
    for start in (5,10,15):
        old=json.loads((Path(json.loads((reference/'study_manifest.json').read_text())['pack'])/'manifest.json').read_text())
        original=next(p for p in old['prediction_groups'] if p['source_hour']==start)
        with np.load(original['path'],allow_pickle=False) as f:z=f['z'][::-1];ids=f['cell_ids'][::-1]
        path=sourcepack/f'source_{start}.npz'
        np.savez_compressed(path,z=z,cell_ids=ids,cell_kin=z+7.,population_kin=np.full(3,4.),
            population_Ssum=np.zeros(3),population_directU=np.zeros(3))
        records.append(dict(time_hours=start,path=str(path),sha256=sha256(path),source_key=f'cohort_{start}'))
    np.savez_compressed(sourcepack/'token_normalizer.npz',training_source_keys=np.array(['train']),
        **{'mean_'+k:np.zeros(3) for k in ('kin','Ssum','directU','cell_kin')},
        **{'scale_'+k:np.ones(3) for k in ('kin','Ssum','directU','cell_kin')})
    (sourcepack/'manifest.json').write_text(json.dumps(dict(development_sources=records,source_transform_sha256='toy')))
    training=tmp_path/'training';training.mkdir();(training/'model.msgpack').write_bytes(b'toy_model')
    (training/'summary.json').write_text(json.dumps(dict(model_sha256=sha256(training/'model.msgpack'),
        pack_directory=str(sourcepack),config={},arm='cf_kin_cell',seed=0)))
    monkeypatch.setattr(common,'create_solver',lambda *a,**kw:SimpleNamespace(vf_state_inference='template'))
    common.add_cellflow_paths()
    from flax import serialization
    monkeypatch.setattr(serialization,'from_bytes',lambda *a,**kw:'toy_loaded')
    calls=[];native_conditions=common.condition_inputs
    def cond(pack,*a,**kw):
        assert len(pack['z'])==12  # full cohort, not eight sampled cells
        return native_conditions(pack,*a,**kw)
    monkeypatch.setattr(common,'condition_inputs',cond)
    def integrate(solver,x,conditions,**kwargs):
        np.testing.assert_allclose(conditions['cell_token'][:,0,:],x+7.)
        calls.append(x.copy());return x+.2
    monkeypatch.setattr(common,'predict_cellflow_tau',integrate)
    # Target caches need not even exist to create source-only common predictions.
    for cache in r['target_caches'].values():Path(cache['path']).unlink()
    result=common.predict_common(training,tmp_path/'predicted',reference)
    assert not result['target_read'] and len(calls)==12
    common.freeze_common_grid([tmp_path/'predicted'],reference_root=reference,expected_seeds=(0,))
