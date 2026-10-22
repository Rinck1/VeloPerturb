import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15m_velocity_discrimination as module


def fixture():
    rng=np.random.default_rng(41)
    norm=dict(axes=np.eye(3)[:2],velocity_reference=np.zeros(3),U_reference=np.zeros(3),response_mean=np.zeros(3))
    groups=[]
    for label in ('A','B'):
        v=rng.normal(size=(7,3));z=rng.normal(size=(7,3));u=rng.normal(size=(7,3))
        groups.append(dict(label=label,z=z,U=u,U_normalized=u,velocity=v,time_hours=6,
            velocity_program=module.unit(v[:,:2]),U_program=module.unit(u[:,:2]),target=z+.1*v))
    return groups,norm,module.fit_source_calibration(groups,norm)


def test_direction_is_unit_and_magnitude_changes_do_not_change_direction():
    groups,norm,cal=fixture();g=groups[0]
    a=module.representation(g,norm,cal,'direction')
    changed=dict(g,velocity=g['velocity']*np.arange(1,8)[:,None])
    np.testing.assert_allclose(a,module.representation(changed,norm,cal,'direction'),atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),1.,atol=1e-7)
    assert not np.allclose(module.representation(g,norm,cal,'magnitude_only'),module.representation(changed,norm,cal,'magnitude_only'))


def test_absolute_and_magnitude_features_discard_sign_but_raw_preserves_it():
    groups,norm,cal=fixture();g=groups[0];reversed_g=dict(g,velocity=-g['velocity'])
    for name in ('magnitude_only','unsigned_direction'):
        np.testing.assert_array_equal(module.representation(g,norm,cal,name),module.representation(reversed_g,norm,cal,name))
    np.testing.assert_allclose(module.representation(g,norm,cal,'direction_and_magnitude'),-module.representation(reversed_g,norm,cal,'direction_and_magnitude'))


def test_all_representations_preserve_identical_U_slots_and_shapes():
    groups,norm,cal=fixture();g=groups[0]
    for name in ('direction',*module.NEW_VARIANTS):
        result=module.with_features(g,module.representation(g,norm,cal,name))
        assert result['kinetic_input'].shape==(7,5)
        np.testing.assert_array_equal(result['kinetic_input'][:,:3],g['U_normalized'].astype('float32'))


def test_calibration_does_not_depend_on_training_targets():
    groups,norm,cal=fixture()
    changed=[dict(g,target=np.full_like(g['target'],1e9)) for g in groups]
    other=module.fit_source_calibration(changed,norm)
    for key in cal:np.testing.assert_array_equal(cal[key],other[key])


@pytest.mark.parametrize('n',[2,3,4,5,7,8,13])
def test_local_permutation_preserves_cohort_and_has_no_fixed_points(n):
    states=np.random.default_rng(4).normal(size=(n,3))
    order=module.local_derangement(states,9)
    np.testing.assert_array_equal(np.sort(order),np.arange(n))
    assert np.all(order!=np.arange(n))
    np.testing.assert_array_equal(order,module.local_derangement(states,9))


def test_frozen_changes_are_source_only_and_renormalization_is_identity():
    groups,norm,cal=fixture();g=dict(groups[0]);g.pop('target')
    for name in module.INTERVENTIONS:
        if name in ('cohort_q','uniform_q'):continue
        feature=module.intervention_features(g,norm,cal,name)
        assert feature.shape==(7,2) and np.isfinite(feature).all()
    np.testing.assert_allclose(module.intervention_features(g,norm,cal,'renormalize'),g['velocity_program'],atol=1e-7)
    np.testing.assert_array_equal(module.intervention_features(g,norm,cal,'reverse'),-g['velocity_program'])


def test_one_scalar_probe_detects_positive_and_negative_geometry():
    norm=dict(axes=np.eye(2),velocity_reference=np.zeros(2),U_reference=np.zeros(2),response_mean=np.zeros(2))
    groups=[dict(z=np.zeros((3,2)),velocity=np.tile(v,(3,1)),U=np.tile(v,(3,1)),target=np.tile(2*np.asarray(v),(3,1))) for v in ((1.,0.),(0.,1.))]
    held=[{k:v for k,v in g.items() if k!='target'} for g in groups]
    coef,p=module.fit_direction_probe(groups,held,norm)
    assert coef['positive_velocity']==2. and coef['positive_reverse']==0.
    np.testing.assert_allclose(p['positive_velocity'],np.array([[2.,0.],[0.,2.]]))
    negative=[dict(g,target=-g['target']) for g in groups]
    coef,p=module.fit_direction_probe(negative,held,norm)
    assert coef['positive_velocity']==0. and coef['positive_reverse']==2. and coef['signed_velocity']==-2.
    with pytest.raises(ValueError,match='future'):module.fit_direction_probe(groups,groups,norm)


def test_energy_matches_direct_weighted_formula_and_reweighting():
    from scipy.spatial.distance import cdist
    x=np.array([[[0.,0.],[1.,0.]],[[0.,1.],[1.,1.]]]);q=np.array([[.8,.2],[.3,.7]])
    y=np.array([[.1,.4],[.3,.8]])
    w=q.ravel()/2;a=x.reshape(-1,2)
    expected=(2*w@cdist(a,y).mean(1)-w@cdist(a,a)@w-cdist(y,y).mean())/np.sqrt(2)
    result=module.empirical_metrics(x,q,y,np.eye(2))
    assert result['energy']==pytest.approx(expected)
    assert result['variance_ratio']>0
