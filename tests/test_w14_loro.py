"""Structural tests before any held-out biological score is calculated."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/phase0'))
from w14_data import fit_geometry,pooled_log_features
from w14_loro import energy,freeze_grid,jobs
from w14_model import arm_feature,fit,make_model,permute_source,source_predictions
from w6_local_router import LocalResponseRouter
from w10_recipe_model import fit_bridge


def config():return json.loads((ROOT/'configs/w14_loro_v1.json').read_text())


def norm(d=50):
    result=dict(z_mean=np.zeros(d),z_scale=np.ones(d))
    for k,n in [('velocity',d),('direct_U',d),('pooled_SU',2*d)]:
        result[k+'_mean']=np.zeros(n);result[k+'_scale']=np.ones(n)
    return result


def test_grid_is_81_jobs_with_separate_shuffle_mixtures():
    j=jobs(config());assert len(j)==81
    assert len({(x['heldout'],x['seed'],x['name']) for x in j})==81
    assert sum(x['arm']=='shuffled_v' for x in j)==45


def test_parameter_counts_and_shared_initial_experts():
    models={a:make_model(norm(),config(),a,0) for a in ('static','real','direct_U','nonlinear_SU')}
    assert sum(p.numel() for p in models['real'].parameters())==76248
    assert sum(p.numel() for p in models['nonlinear_SU'].parameters())==76268
    for m in models.values():
        for p,q in zip(m.experts.parameters(),models['real'].experts.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_static_input_and_construction_need_no_kinetics():
    n=dict(z_mean=np.zeros(3),z_scale=np.ones(3));m=make_model(n,config(),'static',0)
    z=torch.randn(12,3);g=torch.zeros(3)
    a=m(z,torch.zeros_like(z),g,24)[0];b=m(z,torch.full_like(z,float('nan')),g,24)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert arm_feature(dict(z=z.numpy()),'static').shape==(12,3)


@pytest.mark.parametrize('arm',['static','real','direct_U','nonlinear_SU'])
def test_population_mean_and_router_derivative(arm):
    m=make_model(norm(3),config(),arm,0);z=torch.randn(20,3);v=torch.randn(20,6 if arm=='nonlinear_SU' else 3)
    g=torch.randn(3);x,q,_=m(z,v,g,24)
    mean=(x*q[:,:,None]).sum(1).mean(0)
    torch.testing.assert_close(mean,z.mean(0)+24*g,atol=1e-5,rtol=0)
    gradients=torch.autograd.grad(mean.sum(),tuple(m.router.parameters()))
    assert max(float(t.abs().max()) for t in gradients)<1e-5


def test_shuffle_preserves_source_marginal_and_rng_identity():
    x=np.arange(300).reshape(100,3)
    a=permute_source(x,2,3,4);b=permute_source(x,2,3,4)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(np.sort(a,axis=0),np.sort(x,axis=0))
    assert not np.array_equal(a,permute_source(x,2,4,4))


def test_raw_features_have_no_gamma_dependency():
    c=np.eye(3);s=np.ones((6,3));u=np.arange(18).reshape(6,3)
    raw,su=pooled_log_features(c,s,u)
    np.testing.assert_array_equal(raw,np.log1p(u));np.testing.assert_array_equal(su[:,3:],raw)


def test_geometry_uses_only_supplied_training_baselines():
    rng=np.random.default_rng(1);groups=[]
    for rep in ('r2','r3'):
        groups.append(dict(s=sparse.csr_matrix(rng.poisson(3,(30,12))),u=sparse.csr_matrix(rng.poisson(1,(30,12))),
            ids=np.array([f'{rep}_{i}' for i in range(30)])))
    c=dict(config(),n_hvg=8,dimensions=3,clusters=2)
    a=fit_geometry(groups,np.array([f'g{i}' for i in range(12)]),c)
    # No held-out matrix is accepted by the fit boundary; arbitrary changes to
    # a separate held-out object cannot change the fit or enter the allowlist.
    heldout=rng.normal(size=(100,12));heldout[:]=1e9
    b=fit_geometry(groups,np.array([f'g{i}' for i in range(12)]),c)
    assert a.fingerprint()==b.fingerprint()
    assert all(not p.startswith('r1') for p in a.source_ids)


def test_unchanged_real_b1_training_is_numerically_identical():
    rng=np.random.default_rng(3);c=dict(config(),steps=3,checkpoint_steps=[],source_batch=8,target_batch=9,log_interval=1)
    g=dict(z=rng.normal(size=(20,3)).astype('float32'),velocity=rng.normal(size=(20,3)).astype('float32'),
        target=rng.normal(size=(20,3)).astype('float32'),global_rate=np.zeros(3,dtype='float32'))
    n=norm(3)
    old_norm=dict(z_mean=n['z_mean'],z_scale=n['z_scale'],v_mean=n['velocity_mean'],v_scale=n['velocity_scale'])
    old,_=fit_bridge([g],old_norm,c,arm='velocity_router',seed=1,device='cpu')
    new,_=fit([g],n,c,arm='real',seed=1,device='cpu')
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)


def test_missing_predictions_rejected_before_target_reads(tmp_path):
    with pytest.raises(FileNotFoundError):freeze_grid(tmp_path,dict(jobs=jobs(config()),config=config(),primary_checkpoint=36000))


def test_block_energy_matches_direct_definition():
    rng=np.random.default_rng(2);x=rng.normal(size=(13,3));y=rng.normal(size=(7,3));w=rng.uniform(size=13);w/=w.sum()
    from scipy.spatial.distance import cdist
    expected=(2*w@cdist(x,y).mean(1)-w@cdist(x,x)@w-cdist(y,y).mean())/np.sqrt(3)
    assert energy(x,w,y,block=4)==pytest.approx(expected,abs=1e-12)


def test_recentered_interventions_preserve_same_mean():
    rng=np.random.default_rng(2);g=dict(z=rng.normal(size=(25,3)).astype('float32'),
        velocity=rng.normal(size=(25,3)).astype('float32'),global_rate=np.ones(3)*.03)
    m=make_model(norm(3),config(),'real',0);p=source_predictions(m,g,'real',0,0,0,24)
    for x,q in [('candidates','q'),('cohort_candidates','cohort_q')]:
        np.testing.assert_allclose((p[x]*p[q][:,:,None]).sum(1).mean(0),p['anchor'],atol=1e-10)


def test_tiny_complete_grid_freezes_then_scores(tmp_path):
    """Exercise packs, per-fold means, every arm and tamper refusal without real data."""
    import anndata as ad
    import pandas as pd
    from w14_data import prepare_fold
    from w14_loro import evaluate,train_job
    rng=np.random.default_rng(3);labels=[(r,d) for r in ('r1','r2','r3') for d in (3,4,5) for _ in range(24)]
    obs=pd.DataFrame(dict(rep=[r for r,d in labels],timepoint=[f'D{d}' for r,d in labels]),
        index=[f'cell{i}' for i in range(len(labels))])
    s=sparse.csr_matrix(rng.poisson(4,(len(labels),12)));u=sparse.csr_matrix(rng.poisson(2,(len(labels),12)))
    path=tmp_path/'toy.h5ad';ad.AnnData(X=s,obs=obs,layers=dict(spliced=s,unspliced=u)).write_h5ad(path)
    mc=json.loads((ROOT/config()['mean_config']).read_text());mc['steps']=2
    mp=tmp_path/'mean_config.json';mp.write_text(json.dumps(mc))
    c=dict(config(),data_path=str(path),mean_config=str(mp),training_days=[3,4,5],seeds=[0],shuffle_seeds=[0],
        minimum_spliced=1,minimum_unspliced=0,neighbors=4,n_hvg=8,dimensions=3,clusters=2,
        steps=2,checkpoint_steps=[2],experts=2,hidden=8,expert_rank=4,router_hidden=4,nonlinear_router_hidden=3,
        source_batch=8,target_batch=8,log_interval=1,evaluation_source_cells=12,evaluation_target_cells=12)
    root=tmp_path/'study';root.mkdir();m=dict(config=c,jobs=jobs(c),primary_checkpoint=2,frozen_sha256={})
    (root/'study_manifest.json').write_text(json.dumps(m))
    for rep in c['replicates']:
        fm=prepare_fold(c,rep,root/'folds'/rep)
        for job in [j for j in m['jobs'] if j['heldout']==rep]:train_job(root,m,fm,job,'cpu')
    frozen=freeze_grid(root,m);assert len(frozen)==30
    result=evaluate(root,tmp_path/'evaluation','cpu');assert result['total_training_jobs']==15
    p=Path(frozen[0]['path']);p.write_bytes(p.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='changed'):freeze_grid(root,m)
