import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w6_local_router import LocalResponseRouter
from w9_population_router import make_training_state
from w13_time_model import TimeResponseRouter, elapsed_clock, fit_time, rk4_step


def config(**updates):
    return dict(dict(hidden=8,expert_rank=4,router_hidden=8,experts=2,calibration_hours=24,
        maximum_state_correction=2.,time_kernel='decay',rk4_step_hours=6.,learning_rate=.0001,
        weight_decay=.0001,residual_regularization=.001,routing_regularization=.002,
        steps=3,checkpoint_steps=[1,3],source_batch=4,target_batch=4,log_interval=1,
        centering='batch',estimator='V',distance_kernel='direct',regularization_scope='batch',
        shuffle_seed_offset=100000),**updates)


def norm():
    return dict(z_mean=np.zeros(3),z_scale=np.ones(3),v_mean=np.zeros(3),v_scale=np.ones(3))


def data():
    rng = np.random.default_rng(9)
    return [dict(z=rng.normal(size=(9,3)),velocity=rng.normal(size=(9,3)),
                 target=rng.normal(size=(11,3)),global_rate=rng.normal(size=3)*.01)]


def model(arm='velocity_router'):
    return TimeResponseRouter(norm(),config(),arm=arm,seed=0).double()


def tensors():
    g = data()[0]
    return [torch.as_tensor(g[k],dtype=torch.float64) for k in ('z','velocity','global_rate')]


def test_clock_matches_daily_anchor_and_zero_origin():
    assert elapsed_clock(0,24,'decay')==0
    assert elapsed_clock(24,24,'decay')==24
    with pytest.raises(ValueError):elapsed_clock(-1,24,'decay')


def test_rk4_is_fourth_order_on_known_linear_system():
    errors = []
    for count in (4,8):
        x = torch.ones(1,dtype=torch.float64)
        for i in range(count):x = rk4_step(lambda t,y:y,x,i/count,1/count)
        errors.append(abs(float(x)-np.e))
    assert errors[0]/errors[1]>14


@pytest.mark.parametrize('dynamics',['endpoint_time','ode'])
def test_zero_residual_field_gives_exact_frozen_mean_translation(dynamics):
    m = model();z,v,g = tensors()
    for head in m.experts:
        torch.nn.init.zeros_(head[-1].weight);torch.nn.init.zeros_(head[-1].bias)
    for hours in (0,5,10,15,24):
        x,q,_ = m(z,v,g,hours,dynamics=dynamics)
        expected = (z+elapsed_clock(hours,24,'decay')*g)[:,None,:].expand_as(x)
        torch.testing.assert_close(x,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('dynamics',['endpoint_time','ode'])
def test_weighted_mean_and_its_parameter_derivative_are_preserved(dynamics):
    m = model();z,v,g = tensors()
    x,q,_ = m(z,v,g,15,dynamics=dynamics)
    error = (x*q[:,:,None]).sum(1).mean(0)-z.mean(0)-elapsed_clock(15,24,'decay')*g
    torch.testing.assert_close(error,torch.zeros_like(error),atol=1e-12,rtol=0)
    gradients = torch.autograd.grad(error.sum(),list(m.parameters()),allow_unused=True)
    assert all(grad is None or float(grad.abs().max())<1e-11 for grad in gradients)


def test_static_router_does_not_read_velocity_even_nan_values():
    m = model('static_router');z,v,g = tensors()
    a,qa,_ = m(z,v,g,5)
    b,qb,_ = m(z,torch.full_like(v,float('nan')),g,5)
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(qa,qb,atol=0,rtol=0)


def test_endpoint_time_zero_projection_starts_as_existing_endpoint():
    cfg = config();old = LocalResponseRouter(**norm(),config=cfg,arm='velocity_router',seed=0).double()
    new = model();new.load_state_dict(old.state_dict(),strict=False)
    z,v,g = tensors();a,qa,_ = old(z,v,g,elapsed_clock(15,24,'decay'))
    b,qb,_ = new(z,v,g,15,dynamics='endpoint_time')
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);torch.testing.assert_close(qa,qb)


def test_fixed_route_restart_retains_original_elapsed_time_and_context():
    m = model();z,v,g = tensors();ctx = m.context(z,v,g)
    full,_,_ = m.integrate(ctx,18)
    _,residual,_ = m.integrate(ctx,6)
    restarted,_,_ = m.integrate(ctx,18,start=6,residual=residual)
    torch.testing.assert_close(full,restarted,atol=1e-12,rtol=1e-12)
    with pytest.raises(ValueError,match='Restart'):m.integrate(ctx,18,start=6)


def test_modes_remain_distinct_and_all_model_parts_get_finite_task_gradients():
    m = model();z,v,g = tensors()
    with torch.no_grad():
        for index,expert in enumerate(m.experts):expert[-1].bias.fill_(index*.5)
    x,q,_ = m(z,v,g,24)
    assert float((x[:,0]-x[:,1]).abs().max())>.1
    target = torch.tensor([1.,-2.,3.],dtype=z.dtype)
    ((x-target).square()*q[:,:,None]).mean().backward()
    for block in (m.backbone,m.experts,m.router,m.time_projection):
        gradients = [p.grad for p in block.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0


@pytest.mark.parametrize('dynamics',['endpoint_time','ode'])
def test_time_fit_resume_is_bit_exact_and_adds_only_small_time_projection(dynamics):
    cfg = config();old = LocalResponseRouter(**norm(),config=cfg,arm='velocity_router',seed=0)
    snapshots = []
    def checkpoint(step,m,opt,rng,history):
        snapshots.append(copy.deepcopy(make_training_state(m,opt,rng,step,history)))
    kwargs = dict(arm='shuffled_velocity_router',seed=0,dynamics=dynamics,device='cpu',initial_state=old.state_dict())
    full,result = fit_time(data(),norm(),cfg,**kwargs,checkpoint=checkpoint)
    resumed,_ = fit_time(data(),norm(),cfg,**kwargs,resume=snapshots[0])
    for key,value in full.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)
    assert result['parameters']==sum(p.numel() for p in old.parameters())+4*cfg['hidden']


def test_registered_time_budget_is_fixed_and_excludes_step_three():
    path = Path(__file__).resolve().parents[1]/'configs/w13_scifate2_time_v1.json'
    cfg = json.loads(path.read_text())
    assert cfg['steps']==4000 and cfg['initial_endpoint_step']==36000
    assert cfg['dynamics']==['endpoint_time','ode'] and not cfg['r4_used_for_optimizer']
    assert 'direct_U' in cfg['excluded_step_3'] and set(cfg['physical_gpus'])=={1,2,3}


def test_time_training_checkpoint_and_six_window_evaluation_pipeline(tmp_path,monkeypatch):
    from w4_artifacts import sha256
    from w13_time_training import train_one
    import w13_frozen_horizons as horizons
    cfg = config(steps=2,checkpoint_steps=[2]);group = data()[0]
    pack = tmp_path/'pack';pack.mkdir();reference = tmp_path/'reference';reference.mkdir()
    np.savez(pack/'normalization.npz',**norm())
    training = pack/'training.npz'
    np.savez(training,**{k:group[k] for k in ('z','velocity','target')},global_rates=group['global_rate'][None])
    sources = [];caches = []
    for start,stop in ((5,10),(10,15),(15,20)):
        path = pack/f'source{start}.npz'
        np.savez(path,z=group['z'],velocity=group['velocity'],source_mean=group['z'].mean(0))
        sources.append(dict(source_hour=start,target_hour=stop,path=str(path),sha256=sha256(path)))
        cache = reference/f'target{stop}.npz';y = group['target']
        np.savez(cache,target=y,target_mean=y.mean(0),target_cov=np.cov(y,rowvar=False,bias=True))
        caches.append(dict(target_hour=stop,path=str(cache),sha256=sha256(cache)))
    pm = dict(config=dict(global_pack='global_features',global_training='global_models'),
        base_config=dict(calibration_hours=24,time_kernel='decay'),
        training_groups=[dict(path=str(training),sha256=sha256(training))],prediction_groups=sources)
    (pack/'manifest.json').write_text(json.dumps(pm))
    manifest = dict(pack=str(pack),pack_manifest_sha256=sha256(pack/'manifest.json'),
        config=dict(seeds=[0],initial_endpoint_step=36000),evaluation_caches=caches)
    (reference/'study_manifest.json').write_text(json.dumps(manifest))
    (reference/'source_samples.json').write_text(json.dumps({f'{s}_to_{s+5}':list(range(6)) for s in (5,10,15)}))
    global_pack = tmp_path/'global_features';global_pack.mkdir()
    np.savez(global_pack/'source_pack.npz',base=np.zeros((3,3)),velocity=np.zeros((3,3)),
             intervals=np.array([[5,10],[10,15],[15,20]]))
    global_model = tmp_path/'global_models/seed0';global_model.mkdir(parents=True)
    torch.save(dict(state_dict={'input_mean':torch.zeros(6),'input_scale':torch.ones(6),
        'readout.weight':torch.zeros(3,6),'readout.bias':torch.tensor(group['global_rate'],dtype=torch.float32)}),
        global_model/'analytic_task.pt')
    root = tmp_path/'time_training';root.mkdir();(root/'study_manifest.json').write_text(json.dumps(manifest))
    records = []
    for arm in ('static_router','velocity_router','shuffled_velocity_router'):
        model_arm = 'velocity_router' if arm=='shuffled_velocity_router' else arm
        initial = tmp_path/f'initial_{arm}.pt';old = LocalResponseRouter(**norm(),config=cfg,arm=model_arm,seed=0)
        torch.save(dict(state_dict=old.state_dict(),arm=arm,seed=0,step=36000),initial)
        job = dict(setting='ode',dynamics='ode',arm=arm,seed=0,config=cfg,initial_path=str(initial),
            initial_sha256=sha256(initial),expected_parameters=sum(p.numel() for p in old.parameters())+4*cfg['hidden'])
        train_one(root,manifest,job,'cpu');train_one(root,manifest,job,'cpu')
        path = root/'training'/'ode'/arm/'seed0/model_step2.pt'
        records.append(dict(arm=arm,seed=0,path=str(path),sha256=sha256(path)))
    spec = tmp_path/'forecast_spec.json'
    spec.write_text(json.dumps(dict(reference_root=str(reference),models=[dict(name='tiny_ode',kind='ode',checkpoints=records)])))
    refined,refined_config = horizons.load_model(records[0],'ode','cpu',step_override=3.)
    assert refined.max_step==refined_config['rk4_step_hours']==3.
    unchanged = torch.load(records[0]['path'],map_location='cpu',weights_only=True)
    assert unchanged['config']['rk4_step_hours']==6.
    for key,value in refined.state_dict().items():
        torch.testing.assert_close(value,unchanged['state_dict'][key],atol=0,rtol=0)
    with pytest.raises(ValueError,match='only valid for ODE'):
        horizons.load_model(records[0],'endpoint','cpu',step_override=3.)
    monkeypatch.setattr(horizons,'ROOT',tmp_path)
    result = horizons.evaluate(spec,tmp_path/'evaluated','cpu')
    assert result['status']=='FROZEN_TIME_HORIZONS_COMPLETE'
    assert [r['windows'] for r in result['horizon_summary']]==[3,2,1]
    frozen = json.loads((tmp_path/'evaluated/frozen_predictions.json').read_text())
    assert frozen['all_predictions_ready_before_evaluation'] and len(frozen['predictions'])==18
