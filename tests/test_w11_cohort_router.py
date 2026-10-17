from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w10_route_interventions import intervention_probabilities
from w11_cohort_router import CohortResponseRouter,fit


def config():
    return dict(hidden=8,expert_rank=4,router_hidden=8,experts=2,calibration_hours=24,maximum_state_correction=2.,
        learning_rate=.0003,weight_decay=.0001,steps=3,source_batch=4,target_batch=4,checkpoint_steps=[],
        residual_regularization=.001,centering='full_source',estimator='V',distance_kernel='direct')


def norm():return dict(z_mean=np.zeros(3),z_scale=np.ones(3),v_mean=np.zeros(3),v_scale=np.ones(3))


def test_cohort_routing_is_constant_permutation_invariant_and_preserves_mean_and_gradient():
    torch.manual_seed(1);z,v=torch.randn(11,3),torch.randn(11,3);g=torch.tensor([.1,.2,.3])
    model=CohortResponseRouter(**norm(),config=config(),arm='velocity_router',seed=1).double()
    z,v,g=z.double(),v.double(),g.double();x,q,_=model(z,v,g,5.)
    torch.testing.assert_close(q,q[:1].expand_as(q),atol=0,rtol=0)
    permutation=torch.randperm(len(z));xx,qq,_=model(z[permutation],v[permutation],g,5.)
    torch.testing.assert_close(xx,x[permutation],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(qq,q[permutation],atol=1e-12,rtol=1e-12)
    mean=(x*q[:,:,None]).sum(1).mean(0)
    torch.testing.assert_close(mean,z.mean(0)+5*g,atol=1e-12,rtol=1e-12)
    mean.sum().backward();assert max(float(p.grad.abs().max()) for p in model.parameters())<1e-10


def test_static_cohort_is_velocity_blind_but_real_cohort_can_use_joint_fingerprint():
    z,v=torch.randn(13,3),torch.randn(13,3)
    for arm in ('static_router','velocity_router'):
        model=CohortResponseRouter(**norm(),config=config(),arm=arm,seed=2)
        x,q,_=model(z,v,torch.zeros(3),5.);other,oq,_=model(z,10*v,torch.zeros(3),5.)
        if arm=='static_router':torch.testing.assert_close(x,other,atol=0,rtol=0)
        else:assert not torch.allclose(q,oq)


def test_intervention_weights_preserve_usage_when_permuted_and_train_all_cohort_arms():
    q=np.array([[.1,.9],[.3,.7],[.8,.2]]);p=intervention_probabilities(q,0,0)
    np.testing.assert_allclose(p['cohort_average'].mean(0),q.mean(0))
    np.testing.assert_allclose(p['permuted_assignment'].mean(0),q.mean(0))
    rng=np.random.default_rng(1)
    groups=[dict(z=rng.normal(size=(12,3)),velocity=rng.normal(size=(12,3)),
        target=rng.normal(size=(13,3)),global_rate=np.zeros(3))]
    for arm in ('static_router','velocity_router','shuffled_velocity_router'):
        model,history=fit(groups,norm(),config(),arm=arm,seed=0,device='cpu')
        assert len(history)>=2 and all(torch.isfinite(p).all() for p in model.parameters())
