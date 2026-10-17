from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w6_local_router import LocalResponseRouter,weighted_energy,local_pooled_velocity


def configuration():
    return dict(experts=4,hidden=12,expert_rank=4,router_hidden=8,calibration_hours=24,maximum_state_correction=2.)


def test_actual_mixture_mean_is_preserved_and_velocity_only_enters_routing():
    rng = np.random.default_rng(2)
    z,v = [torch.tensor(rng.normal(size=(13,3)),dtype=torch.float32) for _ in range(2)]
    g = torch.tensor([.1,-.2,.3])
    for arm in ('k1','static_router','velocity_router'):
        model = LocalResponseRouter(np.zeros(3),np.ones(3),np.zeros(3),np.ones(3),configuration(),arm=arm,seed=0)
        x,q,_ = model(z,v,g,5.)
        torch.testing.assert_close((x*q[:,:,None]).sum(1).mean(0),z.mean(0)+5*g,atol=1e-6,rtol=1e-6)
        other,oq,_ = model(z,v*100,g,5.)
        if arm != 'velocity_router':
            torch.testing.assert_close(x,other,atol=0,rtol=0);torch.testing.assert_close(q,oq,atol=0,rtol=0)
        else:
            assert not torch.allclose(q,oq)
            loss = weighted_energy(x,q,z+.2);loss.backward()
            assert sum(float(p.grad.square().sum()) for p in model.router.parameters()) > 0


def test_energy_scores_weighted_modes_not_the_average_endpoint():
    x = torch.tensor([[[-1.],[1.]]],dtype=torch.float64)
    q = torch.tensor([[.5,.5]],dtype=torch.float64)
    target = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    assert abs(float(weighted_energy(x,q,target))) < 1e-10
    collapsed = x.mean(1,keepdim=True)
    assert float(weighted_energy(collapsed,torch.ones((1,1),dtype=torch.float64),target)) > .5


def test_local_pool_with_all_neighbors_matches_whole_population():
    from w4_streams import SourceTransform
    rng = np.random.default_rng(3)
    s,u = [sparse.csr_matrix(rng.poisson(n,size=(16,6)).astype(float)) for n in (10,2)]
    ids = np.array([str(i) for i in range(16)])
    transform = SourceTransform.fit(s,u,cell_ids=ids,allowed_ids=set(ids),gene_names=np.array(list('abcdef')),
        n_hvg=6,dimensions=3,clusters=2)
    z = transform.state(s)
    sn,un = transform.normalized(s.sum(0),u.sum(0))
    expected = transform.projected_rate(sn,un-transform.gamma*sn)[0]
    v = local_pooled_velocity(transform,s,u,z,np.zeros(3),16)
    np.testing.assert_allclose(v,np.tile(expected,(16,1)),atol=1e-7)


def test_local_pipeline_runs_on_tiny_paired_counts(tmp_path):
    import json
    import anndata as ad
    import pandas as pd
    import w5_joint_scifate2 as global_run
    import w6_router_scifate2 as local_run
    rng = np.random.default_rng(12)
    groups = [(rep,f'D{day}') for rep in ('r1','r2','r3') for day in range(3,9)]
    groups += [('r4',f'{h:02d}h') for h in (5,10,15,20)]
    reps,times = zip(*[(rep,tp) for rep,tp in groups for _ in range(60)])
    s,u = [sparse.csr_matrix(rng.poisson(n,size=(len(reps),8)).astype(float)) for n in (20,3)]
    data = ad.AnnData(X=s,obs=pd.DataFrame(dict(rep=reps,timepoint=times),
        index=[f'cell{i}' for i in range(len(reps))]))
    data.layers['spliced'] = s;data.layers['unspliced'] = u
    path = tmp_path/'paired.h5ad';data.write_h5ad(path)
    cfg = json.loads((global_run.ROOT/'configs/w5_joint_kinetics_scifate2_v1.json').read_text())
    cfg.update(data_path=str(path),data_sha256=global_run.sha256(path),n_hvg=8,dimensions=3,clusters=2,
        minimum_spliced=50,minimum_unspliced=2,steps=2,seeds=[0],training_partitions_per_condition=1)
    cfgpath = tmp_path/'global_config.json';cfgpath.write_text(json.dumps(cfg))
    pack = tmp_path/'global_pack';training = tmp_path/'global_training'
    global_run.prepare(cfgpath,pack);global_run.train(pack,training/'seed0',0,'cpu')
    cfg = json.loads((global_run.ROOT/'configs/w6_local_router_scifate2_v1.json').read_text())
    cfg.update(global_pack=str(pack),global_training=str(training),seeds=[0],experts=2,neighbors=8,
        hidden=8,expert_rank=4,router_hidden=8,steps=2,source_batch=8,target_batch=8,
        evaluation_source_cells=8,evaluation_target_cells=8)
    cfgpath = tmp_path/'local_config.json';cfgpath.write_text(json.dumps(cfg))
    pack = tmp_path/'local_pack';training = tmp_path/'local_training'
    local_run.prepare(cfgpath,pack)
    local_run.train(pack,training/'seed0',0,'cpu')
    evaluation = tmp_path/'evaluation'
    local_run.evaluate(pack,training,evaluation,False,'cpu')
    result = json.loads((evaluation/'summary.json').read_text())
    assert result['global_mean_frozen'] and not result['r4_used_for_fitting']
    assert result['mean_anchor_max_error'] < 2e-6
    assert set(result['metrics']) == {'identity','global_translation','k1','static_router','velocity_router'}
