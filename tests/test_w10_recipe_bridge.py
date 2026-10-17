import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
from w6_local_router import fit_local
from w9_population_router import fit_population, make_training_state
from w10_recipe_model import bridge_energy, fit_bridge, shuffle_velocity


def norm(d=3):
    return dict(z_mean=np.zeros(d),z_scale=np.ones(d),v_mean=np.zeros(d),v_scale=np.ones(d))


def config(**updates):
    return dict(dict(hidden=8,expert_rank=4,router_hidden=8,experts=2,architecture='small',residual_blocks=0,
        calibration_hours=24,maximum_state_correction=2.,learning_rate=.0003,weight_decay=.0001,
        residual_regularization=.001,routing_regularization=.002,steps=5,checkpoint_steps=[],
        source_batch=4,target_batch=4,log_interval=300,centering='batch',estimator='V',
        distance_kernel='heuristic',regularization_scope='batch',shuffle_seed_offset=100000),**updates)


def groups():
    rng = np.random.default_rng(7)
    return [dict(z=rng.normal(size=(9,3)),velocity=rng.normal(size=(9,3)),
        target=rng.normal(size=(11,3)),global_rate=np.zeros(3)) for _ in range(2)]


@pytest.mark.parametrize('arm',['static_router','velocity_router'])
def test_old_recipe_reproduces_original_optimizer_trajectory(arm):
    data,cfg = groups(),config()
    old,_ = fit_local(data,norm(),cfg,arm=arm,seed=1,device='cpu')
    new,_ = fit_bridge(data,norm(),cfg,arm=arm,seed=1,device='cpu')
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,new.state_dict()[key],atol=0,rtol=0)


@pytest.mark.parametrize('arm',['static_router','velocity_router'])
def test_new_recipe_reproduces_w9_optimizer_trajectory(arm):
    data,cfg = groups(),config(centering='full_source',estimator='U',distance_kernel='direct',
        regularization_scope='full_source',learning_rate=.0001)
    old,_ = fit_population(data,norm(),cfg,arm=arm,seed=1,device='cpu')
    new,_ = fit_bridge(data,norm(),cfg,arm=arm,seed=1,device='cpu')
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,new.state_dict()[key],atol=0,rtol=0)


def test_adjacent_registered_recipes_change_exactly_one_factor():
    path = Path(__file__).resolve().parents[1]/'configs/w10_scifate2_recipe_bridge_v1.json'
    cfg = json.loads(path.read_text());factors = ('centering','estimator','distance_kernel','regularization_scope','learning_rate')
    changes = [tuple(k for k in factors if a[k]!=b[k]) for a,b in zip(cfg['recipes'],cfg['recipes'][1:])]
    assert changes == [('distance_kernel',),('centering',),('estimator',),('regularization_scope',),('learning_rate',)]
    assert len(cfg['recipes'])*len(cfg['arms'])*len(cfg['seeds'])==cfg['maximum_initial_jobs']==54


def test_shuffle_preserves_source_velocity_marginal_without_consuming_sampling_rng():
    x = np.arange(90).reshape(30,3);before = np.random.get_state()
    y = shuffle_velocity(x,2,0);yy = shuffle_velocity(x,2,0);other = shuffle_velocity(x,2,10000)
    np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(np.sort(y,axis=0),x)
    assert not np.array_equal(x,y) and not np.array_equal(y,other)
    for a,b in zip(before,np.random.get_state()):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('scope',['batch','full_source'])
def test_shuffled_and_real_arms_keep_same_initial_parameters_and_resume_exactly(scope):
    cfg = config(centering=scope,distance_kernel='direct',steps=5,checkpoint_steps=[2])
    states = []
    def checkpoint(step,model,optimizer,generator,history):
        states.append(copy.deepcopy(make_training_state(model,optimizer,generator,step,history)))
    full,_ = fit_bridge(groups(),norm(),cfg,arm='shuffled_velocity_router',seed=1,device='cpu',checkpoint=checkpoint)
    continued,_ = fit_bridge(groups(),norm(),cfg,arm='shuffled_velocity_router',seed=1,device='cpu',resume=states[0])
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,continued.state_dict()[key],atol=0,rtol=0)
    real,_ = fit_bridge(groups(),norm(),config(steps=0),arm='velocity_router',seed=1,device='cpu')
    null,_ = fit_bridge(groups(),norm(),config(steps=0),arm='shuffled_velocity_router',seed=1,device='cpu')
    for key,value in real.state_dict().items():torch.testing.assert_close(value,null.state_dict()[key],atol=0,rtol=0)


def test_batch_centered_u_estimator_is_not_silently_called_unbiased():
    x = torch.randn(4,2,3);q = torch.ones(4,2)/2;y = torch.randn(4,3)
    with pytest.raises(ValueError,match='full-source'):
        bridge_energy(x,q,y,config(estimator='U',distance_kernel='direct'))


def test_three_arm_artifact_prediction_evaluation_report_pipeline(tmp_path):
    from w10_recipe_bridge import train_job,aggregate
    from w7_router_scaling import score_case
    root,pack = tmp_path/'study',tmp_path/'pack';root.mkdir();pack.mkdir()
    data = groups()[0];z,v,y = [data[k].astype('float32') for k in ('z','velocity','target')]
    np.savez(pack/'normalization.npz',**norm())
    train = pack/'train.npz';np.savez(train,z=z,velocity=v,target=y,global_rates=np.zeros((1,3)))
    source = pack/'source.npz';anchor = z.astype(float).mean(0)+.1
    np.savez(source,z=z,velocity=v,duration=5.,anchors=anchor[None],source_mean=z.astype(float).mean(0))
    pm = dict(training_groups=[dict(path=str(train),sha256=sha256(train))],
        prediction_groups=[dict(path=str(source),sha256=sha256(source),source_hour=5,target_hour=10)])
    (pack/'manifest.json').write_text(json.dumps(pm))
    cache = root/'evaluation.npz'
    np.savez(cache,target=y[:7].astype(float),target_mean=y.astype(float).mean(0),
        target_cov=np.cov(y.astype(float),rowvar=False,bias=True),anchors=anchor[None])
    cfg = config(steps=2,checkpoint_steps=[1,2]);jobs = [];cases = []
    arms = ('static_router','velocity_router','shuffled_velocity_router')
    for arm in arms:
        model,_ = fit_bridge(groups(),norm(),config(steps=0),arm=arm,seed=0,device='cpu')
        count = sum(p.numel() for p in model.parameters())
        jobs.append(dict(setting='b0',experts=2,hidden=8,arm=arm,seed=0,config=cfg,expected_parameters=count,worker=0))
        for step in (1,2):cases.append(dict(setting='b0',experts=2,hidden=8,arm=arm,step=step,worker=0))
    manifest = dict(pack=str(pack),pack_manifest_sha256=sha256(pack/'manifest.json'),jobs=jobs,cases=cases,
        config=dict(seeds=[0],energy_block_size=4,workers=1,steps=2,checkpoint_steps=[1,2],
                    recipes=[dict(name='b0')],arms=arms),
        evaluation_caches=[dict(path=str(cache),sha256=sha256(cache),source_hour=5,target_hour=10)])
    (root/'study_manifest.json').write_text(json.dumps(manifest))
    (root/'source_samples.json').write_text(json.dumps({'5_to_10':list(range(6))}))
    for job in jobs:train_job(root,manifest,job,'cpu')
    for case in cases:score_case(root,manifest,case,'cpu')
    (root/'worker0_complete.json').write_text('{}');result = aggregate(root)
    assert result['training_jobs']==3 and result['cases']==6
    assert result['mean_anchor_max_error']<1e-6
    assert len(result['comparisons'])==2 and (root/'report/comparisons.csv').is_file()
