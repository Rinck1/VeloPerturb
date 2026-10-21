import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15g_response_program_probe as module


def fixture(tmp_path,monkeypatch):
    rng=np.random.default_rng(18)
    train=[f'KO{i}' for i in range(8)];val=['V1','V2','V3'];test=['T1']
    split=tmp_path/'split.json';split.write_text(json.dumps(dict(train=train,val=val,test=test)))
    path=tmp_path/'conditions.npz'
    np.savez_compressed(path,conditions=np.array(train+val+test),embeddings=rng.normal(size=(12,6)),
        metadata_json=np.array(json.dumps(dict(frozen=True,no_cell_expression_used=True))))
    def source(labels):
        conditions=np.repeat(labels,3);n=len(conditions)
        return dict(z=rng.normal(size=(n,4)),u_features=rng.normal(size=(n,4)),
            velocity=rng.normal(size=(n,4)),conditions=conditions)
    st=source(train);sv=source(val)
    target=dict(z=st['z']+.3*st['velocity']+.1,conditions=st['conditions'])
    inputs={'train_source.npz':st,'validation_source.npz':sv,'train_target.npz':target}
    calls=[]
    def reader(path,expected_side):
        calls.append(path.name)
        assert path.name in inputs,'Held-out destination was opened during fitting'
        return inputs[path.name],dict(day=24 if expected_side=='target' else 6)
    monkeypatch.setattr(module,'load_pack',reader)
    c=dict(fold=str(tmp_path),split=str(split),conditions=str(path),programs=2,
        protein_dimensions=2,ridge_alphas=[1,10],shuffle_seeds=[0,1,2],source_hours=6,target_hours=24)
    return c,calls,inputs


def test_programs_and_predictions_are_frozen_without_heldout_targets(tmp_path,monkeypatch):
    c,calls,inputs=fixture(tmp_path,monkeypatch)
    prepared=module.prepare(c)
    prediction,coefficients=module.fit_predictions(prepared,c)
    assert calls==['train_source.npz','train_target.npz','validation_source.npz']
    assert prepared['axes'].shape==(2,4)
    assert all(row['positive_direction_coefficient']>=0 for row in coefficients)
    for arms in prediction.values():
        assert len(arms)==11
        assert all(x.shape==(3,4) and np.isfinite(x).all() for x in arms.values())
    actual_means=inputs['train_target.npz']['z'].reshape(8,3,4).mean(1)-inputs['train_source.npz']['z'].reshape(8,3,4).mean(1)
    np.testing.assert_allclose(prepared['mean'],actual_means.mean(0))


def test_direction_permutations_and_scale_invariance():
    x=np.random.default_rng(7).normal(size=(9,5))
    order=module.derangement(9,0)
    assert np.all(order!=np.arange(9))
    assert set(order)==set(range(9))
    np.testing.assert_allclose(module.unit(x),module.unit(100*x))
    np.testing.assert_array_equal(module.unit(np.zeros((2,4))),np.zeros((2,4)))


def test_correct_direction_predicts_known_synthetic_program_signs(tmp_path,monkeypatch):
    c,_,_=fixture(tmp_path,monkeypatch)
    p=module.prepare(c)
    direction=p['validation_features']['direction'];axes=p['axes']
    truth=p['mean']+direction@axes
    heldout=dict(z=p['source']['state']+truth,conditions=p['source']['labels'])
    arms={'S_U':np.broadcast_to(p['mean'],truth.shape),
        'velocity_direction':truth,'U_program_direction':p['mean']+.5*direction@axes,
        'positive_direction_correction':truth,'shared_response':np.zeros_like(truth)}
    arms.update({f'shuffled_direction_perm{i}':p['mean']-direction@axes for i in range(3)})
    rows,gains,alignment=module.evaluate(p,{'1':arms},heldout)
    assert gains[0]['gain_vs_S_U']>.999
    assert gains[0]['gain_vs_mean_shuffle']>.999
    assert alignment['mean_cosine']>.999
    assert alignment['program_sign_accuracy']==1.
