from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w6_local_router import LocalResponseRouter
from w12_privileged_router import fit_student,routing_q


def test_privileged_student_is_velocity_blind_and_keeps_teacher_experts_unchanged():
    rng=np.random.default_rng(2);norm=dict(z_mean=np.zeros(3),z_scale=np.ones(3),v_mean=np.zeros(3),v_scale=np.ones(3))
    cfg=dict(hidden=8,expert_rank=4,router_hidden=8,experts=2,calibration_hours=24,maximum_state_correction=2.)
    teacher=LocalResponseRouter(**norm,config=cfg,arm='velocity_router',seed=1)
    original={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    # Deliberately no target-expression keys are provided to distillation.
    groups=[dict(z=rng.normal(size=(20,3)),velocity=rng.normal(size=(20,3)))]
    student,_,_,history,_=fit_student(teacher,groups,teacher_arm='velocity_router',seed=1,steps=30,device='cpu')
    for key,value in original.items():
        torch.testing.assert_close(value,teacher.state_dict()[key],atol=0,rtol=0)
        if not key.startswith('router.'):torch.testing.assert_close(value,student.state_dict()[key],atol=0,rtol=0)
    z,v=torch.randn(7,3),torch.randn(7,3)
    torch.testing.assert_close(routing_q(student,z,v),routing_q(student,z,100*v),atol=0,rtol=0)
    assert all(p.requires_grad == name.startswith('router.') for name,p in student.named_parameters())
    assert history[-1]['train_source_KL'] < history[0]['train_source_KL']
