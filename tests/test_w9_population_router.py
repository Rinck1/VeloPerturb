import copy
import itertools
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w6_local_router import energy_from_weights
from w9_population_router import PopulationResponseRouter, independent_source_energy, fit_population, make_training_state


def config(**updates):
    return dict(dict(architecture='small',hidden=8,expert_rank=4,router_hidden=8,experts=2,residual_blocks=1,
        calibration_hours=24,maximum_state_correction=2.,learning_rate=.0001,weight_decay=.0001,
        residual_regularization=.001,routing_regularization=.002,steps=4,checkpoint_steps=[],
        source_batch=4,target_batch=4,log_interval=1),**updates)


def norm(d=3):
    return dict(z_mean=np.zeros(d),z_scale=np.ones(d),v_mean=np.zeros(d),v_scale=np.ones(d))


def test_unbiased_energy_expectation_integrates_experts_and_excludes_source_draw_blocks():
    x = torch.tensor([[[-1.],[2.]],[[.5],[3.]]],dtype=torch.float64)
    q = torch.tensor([[.2,.8],[.7,.3]],dtype=torch.float64)
    y = torch.tensor([[-.5],[1.5]],dtype=torch.float64)
    scores = []
    for source in itertools.product(range(2),repeat=2):
        for target in itertools.product(range(2),repeat=2):
            scores.append(independent_source_energy(x[list(source)],q[list(source)],y[list(target)]))
    exact = energy_from_weights(x.reshape(-1,1),q.reshape(-1)/2,y)
    torch.testing.assert_close(torch.stack(scores).mean(),exact,atol=1e-12,rtol=1e-12)


def test_energy_is_not_loss_of_collapsed_expert_mean_and_can_be_negative():
    x = torch.tensor([[[-1.],[1.]],[[-1.],[1.]]],dtype=torch.float64)
    q = torch.full((2,2),.5,dtype=torch.float64); y = torch.tensor([[-1.],[1.]],dtype=torch.float64)
    score = independent_source_energy(x,q,y)
    assert score < 0  # a U-statistic is not a nonnegative empirical metric
    collapsed = independent_source_energy(x.mean(1,keepdim=True),torch.ones((2,1),dtype=torch.float64),y)
    assert collapsed-score > .5


def test_population_mean_is_conserved_in_value_and_parameter_derivative():
    torch.manual_seed(1); z,v = [torch.randn((11,3),dtype=torch.float64) for _ in range(2)]
    g = torch.tensor([.2,-.1,.05],dtype=torch.float64)
    for architecture in ('small','residual_large'):
        model = PopulationResponseRouter(norm(),config(architecture=architecture),arm='velocity_router',seed=3).double()
        x,q,_ = model(z,v,g,5.)
        mean = (x*q[:,:,None]).sum(1).mean(0)
        torch.testing.assert_close(mean,z.mean(0)+5*g,atol=1e-12,rtol=1e-12)
        mean.sum().backward()
        assert max(float(p.grad.abs().max()) for p in model.parameters()) < 1e-10


def test_source_only_velocity_routing_and_every_parameter_has_shape_gradient():
    torch.manual_seed(4); z = torch.randn((13,3)); v = torch.randn((13,3),requires_grad=True)
    for arm in ('static_router','velocity_router'):
        model = PopulationResponseRouter(norm(),config(architecture='residual_large'),arm=arm,seed=2)
        x,q,raw = model(z,v,torch.zeros(3),5.)
        other,oq,_ = model(z,100*v,torch.zeros(3),5.)
        if arm=='static_router':
            torch.testing.assert_close(x,other,atol=0,rtol=0);torch.testing.assert_close(q,oq,atol=0,rtol=0)
        else:
            assert not torch.allclose(q,oq)
        assert torch.autograd.grad(raw.sum(),v,allow_unused=True,retain_graph=True)[0] is None
        loss = independent_source_energy(x[:5],q[:5],z[5:]+.2)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters())


def test_replication_of_population_does_not_change_individual_candidates():
    z,v = torch.randn((9,3)),torch.randn((9,3))
    model = PopulationResponseRouter(norm(),config(),arm='velocity_router',seed=0)
    x,q,_ = model(z,v,torch.zeros(3),5.)
    xx,qq,_ = model(z.repeat(2,1),v.repeat(2,1),torch.zeros(3),5.)
    torch.testing.assert_close(x,xx[:9],atol=1e-6,rtol=1e-6);torch.testing.assert_close(q,qq[:9],atol=1e-6,rtol=1e-6)


def test_large_model_really_has_18m_trainable_parameters_and_no_frozen_fallback():
    import json
    root = Path(__file__).resolve().parents[1]
    specs = json.loads((root/'configs/w9_scifate2_large_audited_v1.json').read_text())['models']
    for spec in specs:
        for k in (4,8):
            cfg = config(**{key:value for key,value in spec.items() if key!='name'},experts=k)
            model = PopulationResponseRouter(norm(50),cfg,arm='velocity_router',seed=0)
            count = sum(p.numel() for p in model.parameters())
            expected = {'small':{4:52876,8:76248},'large':{4:18191372,8:18516952}}
            assert count == expected[spec['name']][k]
            assert all(p.requires_grad for p in model.parameters())
            assert not hasattr(model,'static_field')


def test_population_objective_gradients_match_finite_differences():
    z,v = torch.randn((7,3),dtype=torch.float64),torch.randn((7,3),dtype=torch.float64)
    model = PopulationResponseRouter(norm(),config(),arm='velocity_router',seed=1).double()
    def objective():
        x,q,_ = model(z,v,torch.zeros(3,dtype=torch.float64),5.)
        return independent_source_energy(x[:4],q[:4],z[4:]+.3)
    parameter = model.router[-1].weight
    exact = torch.autograd.grad(objective(),parameter)[0][0,0]
    old = float(parameter[0,0].detach()); eps = 1e-5
    with torch.no_grad():parameter[0,0]=old+eps
    plus = objective().detach()
    with torch.no_grad():parameter[0,0]=old-eps
    minus = objective().detach()
    with torch.no_grad():parameter[0,0]=old
    torch.testing.assert_close(exact,(plus-minus)/(2*eps),atol=1e-8,rtol=1e-4)


def test_optimizer_and_rng_resume_exactly_preserves_training():
    rng = np.random.default_rng(5)
    groups = [dict(z=rng.normal(size=(12,3)),velocity=rng.normal(size=(12,3)),
        target=rng.normal(size=(15,3)),global_rate=np.zeros(3)) for _ in range(2)]
    cfg = config(steps=5,checkpoint_steps=[2]); states = []
    def checkpoint(step,model,optimizer,generator,history):
        states.append(copy.deepcopy(make_training_state(model,optimizer,generator,step,history)))
    full,_ = fit_population(groups,norm(),cfg,arm='velocity_router',seed=3,device='cpu',checkpoint=checkpoint)
    continued,_ = fit_population(groups,norm(),cfg,arm='velocity_router',seed=3,device='cpu',resume=states[0])
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,continued.state_dict()[key],atol=0,rtol=0)


def test_audited_training_checkpoints_and_frozen_evaluation_end_to_end(tmp_path):
    import json
    from w9_large_training import train_job
    from w7_router_scaling import score_case, aggregate
    from w4_artifacts import sha256
    root,pack = tmp_path/'study',tmp_path/'pack';root.mkdir();pack.mkdir()
    rng = np.random.default_rng(1)
    z,v,y = [rng.normal(size=s).astype('float32') for s in ((12,3),(12,3),(14,3))]
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
    for arm in ('static_router','velocity_router'):
        count = sum(p.numel() for p in PopulationResponseRouter(norm(),cfg,arm=arm,seed=0).parameters())
        jobs.append(dict(setting='small_k2_w8',experts=2,hidden=8,arm=arm,seed=0,
            config=cfg,expected_parameters=count,worker=0,cost=1))
        for step in (1,2): cases.append(dict(setting='small_k2_w8',experts=2,hidden=8,arm=arm,step=step,worker=0))
    manifest = dict(pack=str(pack),pack_manifest_sha256=sha256(pack/'manifest.json'),jobs=jobs,cases=cases,
        config=dict(seeds=[0],energy_block_size=4,workers=1,baseline_width=8,steps=2,
            saturation_consecutive_doublings=2,saturation_relative_improvement=.002),
        evaluation_caches=[dict(path=str(cache),sha256=sha256(cache),source_hour=5,target_hour=10)])
    (root/'study_manifest.json').write_text(json.dumps(manifest))
    (root/'source_samples.json').write_text(json.dumps({'5_to_10':list(range(6))}))
    for job in jobs:
        train_job(root,manifest,job,'cpu')
        state = torch.load(root/'training'/job['setting']/job['arm']/'seed0/resume_state.pt',
            map_location='cpu',weights_only=True)
        assert state['training_state']['step']==2 and state['training_state']['optimizer']['state']
    for case in cases: score_case(root,manifest,case,'cpu')
    (root/'worker0_complete.json').write_text('{}');aggregate(root,False)
    result = json.loads((root/'report/summary.json').read_text())
    assert result['cases']==4 and result['mean_anchor_max_error']<1e-6
