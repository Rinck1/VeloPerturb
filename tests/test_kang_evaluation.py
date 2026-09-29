import numpy as np
import pytest

from veloroute.artifacts import load_config
from veloroute.kang_evaluation import donor_bootstrap,mode_posterior,weighted_energy
from veloroute.metrics import energy_distance


def test_weighted_energy_matches_empirical_and_weight_expansion():
    rng=np.random.default_rng(8);x=rng.normal(size=(5,3));y=rng.normal(size=(7,3))
    assert weighted_energy(x,np.ones(5),y)==pytest.approx(energy_distance(x,y))
    assert weighted_energy(x,np.arange(1,6),y)==pytest.approx(energy_distance(np.repeat(x,np.arange(1,6),axis=0),y))
    assert weighted_energy(x,np.ones(5),x)==pytest.approx(0,abs=1e-9)


def test_training_mode_definition_is_applied_without_fitting_target():
    d=dict(axis=[1.,0.],means=[-2.,2.],variances=[.2,.2],weights=[.5,.5])
    x=np.array([[-2,10],[2,10],[0,20.]])
    p=mode_posterior(x,d)
    assert np.allclose(p.sum(1),1) and p[0,0]>.99 and p[1,1]>.99
    assert p[2,0]==pytest.approx(.5)


def test_inference_unit_is_donor_not_seed_or_celltype():
    c=load_config('configs/veloroute_kang_20260914.yaml');rows=[]
    for donor in c['donors']:
        for seed in c['experiments']['seeds']:
            for arm,error in [('gfg_joint_router',1.),('static_router',2.)]:
                rows.append(dict(donor=donor,seed=seed,arm=arm,cell_type=c['cell_types']['primary'],
                    mode_approved=True,adequate_test_count=True,energy_distance=error))
    result=donor_bootstrap(rows,c,'static_router','energy_distance',multimodal_only=True)
    assert result['n_donors']==8 and result['gain']==pytest.approx(1.) and result['ci_low']==pytest.approx(1.)
    for row in rows:row['mode_approved']=False
    assert donor_bootstrap(rows,c,'static_router','energy_distance',multimodal_only=True)['gain'] is None
