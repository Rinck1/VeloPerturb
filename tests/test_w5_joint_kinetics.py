import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_arms import fit_ridge_training_cv
from w5_joint_kinetics import JointPopulationForecaster,fit_task_model,predict_delta


def sample():
    rng = np.random.default_rng(1)
    base,velocity = rng.normal(size=(8,6)),rng.normal(size=(8,3))
    target = base[:,:3]+.4*velocity
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/w5_joint_kinetics_renge_v1.json').read_text())
    config.update(steps=8,batch_size=8,hidden=4)
    return base,velocity,target,config


def test_joint_initialization_is_exact_analytic_readout():
    base,v,y,c = sample();controls = np.zeros(len(base),dtype=bool)
    scale,ridge,_ = fit_ridge_training_cv(np.c_[base,v],y,controls=controls)
    model = JointPopulationForecaster(base,v,y,controls=controls,arm='velocity_joint',config=c,seed=0)
    np.testing.assert_allclose(predict_delta(model,base,v),ridge.predict(scale.transform(np.c_[base,v])),atol=1e-6)


def test_task_gradient_trains_velocity_head_and_static_does_not_read_it():
    base,v,y,c = sample();controls = np.zeros(len(base),dtype=bool)
    bank = dict(base=base+.05,velocity=v+.2,delta=y)
    joint,info = fit_task_model(base,v,y,bank,c,controls=controls,arm='velocity_joint',seed=0,device='cpu')
    assert info['head_task_gradient_max'] > 0 and np.linalg.norm(joint.head.up.weight.detach().numpy()) > 0
    static,_ = fit_task_model(base,v,y,bank,c,controls=controls,arm='static_task',seed=0,device='cpu')
    np.testing.assert_array_equal(predict_delta(static,base,v),predict_delta(static,base,v*999))


def test_scifate2_adapter_runs_end_to_end_on_tiny_counts(tmp_path,monkeypatch):
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    import w5_joint_scifate2 as adapter
    rng = np.random.default_rng(7)
    groups = [(rep,f'D{day}') for rep in ('r1','r2','r3') for day in range(3,9)]
    groups += [('r4',f'{h:02d}h') for h in (5,10,15,20)]
    reps,times = zip(*[(rep,tp) for rep,tp in groups for _ in range(60)])
    s = sparse.csr_matrix(rng.poisson(20,size=(len(reps),8)).astype(float))
    u = sparse.csr_matrix(rng.poisson(3,size=s.shape).astype(float))
    data = ad.AnnData(X=s,obs=pd.DataFrame(dict(rep=reps,timepoint=times),
        index=[f'cell{i}' for i in range(len(reps))]))
    data.layers['spliced'] = s;data.layers['unspliced'] = u
    path = tmp_path/'paired.h5ad';data.write_h5ad(path)
    config = json.loads((adapter.ROOT/'configs/w5_joint_kinetics_scifate2_v1.json').read_text())
    config.update(data_path=str(path),data_sha256=adapter.sha256(path),n_hvg=8,dimensions=3,clusters=2,
        minimum_spliced=50,minimum_unspliced=2,steps=2,seeds=[0],training_partitions_per_condition=1)
    config_path = tmp_path/'config.json';config_path.write_text(json.dumps(config))
    pack = tmp_path/'pack';adapter.prepare(config_path,pack)
    training = tmp_path/'training';adapter.train(pack,training/'seed0',0,'cpu')
    evaluation = tmp_path/'evaluation';adapter.evaluate(pack,training,evaluation,False)
    result = json.loads((evaluation/'summary.json').read_text())
    assert not result['r4_used_for_fitting'] and len(result['mse_by_interval']['velocity_joint']) == 3
