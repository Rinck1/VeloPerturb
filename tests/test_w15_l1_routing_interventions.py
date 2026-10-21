import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w13_time_model import TimeResponseRouter, elapsed_clock
from w15_l1_routing_interventions import INTERVENTIONS, intervene


def model():
    norm = dict(z_mean=np.zeros(3), z_scale=np.ones(3), v_mean=np.zeros(3), v_scale=np.ones(3))
    config = dict(experts=3, hidden=8, expert_rank=4, router_hidden=6,
        maximum_state_correction=2., calibration_hours=24, time_kernel='decay', rk4_step_hours=3.)
    return TimeResponseRouter(norm,config,arm='velocity_router',seed=1)


@pytest.mark.parametrize('variant',INTERVENTIONS)
def test_all_interventions_preserve_shared_mean(variant):
    m=model();z=torch.randn(12,3);v=torch.randn(12,3);rate=torch.ones(3)*.1
    x,q,anchor=intervene(m,z,v,rate,10.,variant)
    torch.testing.assert_close((x*q[:,:,None]).sum(1).mean(0),anchor,atol=2e-6,rtol=0)


def test_soft_matches_original_integrator_and_reweight_keeps_relative_geometry():
    m=model();z=torch.randn(12,3);v=torch.randn(12,3);rate=torch.zeros(3)
    old,old_q,_=m(z,v,rate,10.)
    soft,q,_=intervene(m,z,v,rate,10.,'soft')
    torch.testing.assert_close(soft,old,atol=2e-6,rtol=0)
    torch.testing.assert_close(q,old_q,atol=0,rtol=0)
    x,_,_=intervene(m,z,v,rate,10.,'cohort_reweight')
    torch.testing.assert_close(x-x[0,0],soft-soft[0,0],atol=2e-6,rtol=0)


def test_reintegration_changes_coupled_geometry_not_just_final_weights():
    m=model();z=torch.randn(12,3);v=torch.randn(12,3);rate=torch.zeros(3)
    a,q,_=intervene(m,z,v,rate,15.,'uniform_reweight')
    b,qq,_=intervene(m,z,v,rate,15.,'uniform_reintegrate')
    torch.testing.assert_close(q,qq,atol=0,rtol=0)
    assert float((a-b).abs().max())>1e-7
