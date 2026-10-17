"""Train a fixed small ODE grid after the direct-distance continuation finishes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w7_router_scaling import atomic_json, training_path
from w9_large_training import read_training
from w9_population_router import make_training_state
from w13_time_model import TimeResponseRouter, fit_time


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__),*[Path(__file__).with_name(n) for n in
    ('w13_time_model.py','w10_recipe_model.py','w9_large_training.py','w9_population_router.py',
     'w6_local_router.py','w4_artifacts.py')]]


def prepare(parent, output, config_path):
    parent,config_path = Path(parent),Path(config_path)
    reference = json.loads((parent/'study_manifest.json').read_text())
    config = json.loads(config_path.read_text())
    if config['r4_used_for_optimizer'] or not config['reset_adam_equally_for_time_models']:
        raise ValueError('Time experiment requires source-only matched optimization')
    pack = Path(reference['pack']);pm = json.loads((pack/'manifest.json').read_text())
    with np.load(pack/'normalization.npz',allow_pickle=False) as saved:
        norm = {k:saved[k].copy() for k in saved.files}
    jobs = []
    for worker,seed in enumerate(config['seeds']):
        for arm in config['arms']:
            old_job = next(j for j in reference['jobs'] if (j['arm'],j['seed'])==(arm,seed))
            initial = training_path(parent,old_job)/f'model_step{config["initial_endpoint_step"]}.pt'
            for dynamics in config['dynamics']:
                mc = dict(old_job['config'],**{k:config[k] for k in
                    ('steps','checkpoint_steps','learning_rate','rk4_step_hours','log_interval')},
                    time_kernel=pm['base_config']['time_kernel'])
                model = TimeResponseRouter(norm,mc,arm=arm,seed=seed)
                jobs.append(dict(setting=dynamics,dynamics=dynamics,arm=arm,seed=seed,config=mc,worker=worker,
                    initial_path=str(initial.resolve()),initial_sha256=sha256(initial),
                    expected_parameters=sum(p.numel() for p in model.parameters())))
    run = Artifacts(output,seed=config['seeds'],inputs=[config_path,parent/'study_manifest.json',
        pack/'manifest.json',pack/'normalization.npz',ROOT/'PLANS/w13_time_dynamics.md']+CODE)
    manifest = dict(config=config,jobs=jobs,pack=str(pack),pack_manifest_sha256=reference['pack_manifest_sha256'],
        parent=str(parent),code_hashes={str(p):sha256(p) for p in CODE},
        r4_used_for_optimizer=False,mean_frozen=True,scope='same_series_development',
        optimization_budget='36000 shared endpoint updates plus 4000 matched new updates; Adam reset equally',
        compute_budget_matched=False,endpoint_and_ode_parameter_counts_matched=True)
    run.json('study_manifest.json',manifest);run.json('status.json',dict(status='ready',jobs=len(jobs)))
    return manifest


def train_one(root,manifest,job,device):
    directory = training_path(root,job)
    if (directory/'complete.json').exists():
        complete = json.loads((directory/'complete.json').read_text())
        if complete['job'] != job or sha256(complete['model_path']) != complete['model_sha256']:
            raise ValueError('Completed model identity changed')
        return
    if directory.exists():
        raise ValueError('Incomplete experiment retained; explicit resume required')
    if shutil.disk_usage(root).free < 2*1024**3:
        raise OSError('Less than 2 GiB free')
    pm,groups,norm,_ = read_training(root,manifest,job)
    initial = Path(job['initial_path'])
    if sha256(initial) != job['initial_sha256']:
        raise ValueError('Warm-start model changed')
    saved = torch.load(initial,map_location='cpu',weights_only=True)
    if (saved['arm'],saved['seed'],saved['step']) != (job['arm'],job['seed'],manifest['config']['initial_endpoint_step']):
        raise ValueError('Warm-start identity mismatch')
    run = Artifacts(directory,seed=job['seed'],inputs=[root/'study_manifest.json',initial,
        Path(manifest['pack'])/'manifest.json',Path(manifest['pack'])/'normalization.npz']+CODE)
    began = time.monotonic()
    def checkpoint(step,model,optimizer,generator,history):
        path = run.directory/f'model_step{step}.pt'
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            arm=job['arm'],seed=job['seed'],step=step,config=job['config'],dynamics=job['dynamics']),path)
        temp = run.directory/'resume_state.pt.tmp'
        torch.save(dict(job=job,training_state=make_training_state(model,optimizer,generator,step,history)),temp)
        temp.replace(run.directory/'resume_state.pt')
        print(json.dumps(dict(status='TIME_MODEL_CHECKPOINT',dynamics=job['dynamics'],arm=job['arm'],seed=job['seed'],
            step=step,elapsed_seconds=time.monotonic()-began)),flush=True)
    model,result = fit_time(groups,norm,job['config'],arm=job['arm'],seed=job['seed'],
        dynamics=job['dynamics'],device=device,initial_state=saved['state_dict'],checkpoint=checkpoint,
        progress=lambda row:run.json('status.json',dict(status='training',job=job,**row)))
    if result['parameters'] != job['expected_parameters']:
        raise ValueError('Parameter count changed')
    path = run.directory/f'model_step{job["config"]["steps"]}.pt'
    run.json('training.json',result)
    run.json('complete.json',dict(job=job,model_path=str(path.resolve()),model_sha256=sha256(path),
        parameters=result['parameters'],wall_seconds=time.monotonic()-began))
    run.json('status.json',dict(status='complete'));del model
    if device != 'cpu':torch.cuda.empty_cache()


def benchmark(root,output,device):
    """Training-only timing and step-halving check; never reads r4 targets."""
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    job = manifest['jobs'][0];_,groups,norm,_ = read_training(root,manifest,job)
    initial = torch.load(job['initial_path'],map_location='cpu',weights_only=True)['state_dict']
    run = Artifacts(output,seed=job['seed'],inputs=[root/'study_manifest.json']+CODE)
    times = []
    for dynamics in ('endpoint_time','ode'):
        cfg = dict(job['config'],steps=25,checkpoint_steps=[])
        if device != 'cpu':torch.cuda.synchronize()
        began = time.monotonic()
        model,_ = fit_time(groups,norm,cfg,arm=job['arm'],seed=job['seed'],dynamics=dynamics,
                         device=device,initial_state=initial)
        if device != 'cpu':torch.cuda.synchronize()
        times.append(dict(dynamics=dynamics,steps=25,seconds=time.monotonic()-began))
    group = groups[0]
    args = [torch.as_tensor(group[k][:64] if k!='global_rate' else group[k],dtype=torch.float32,device=device)
            for k in ('z','velocity','global_rate')]
    with torch.no_grad():
        coarse,q,_ = model(*args,24,dynamics='ode')
        fine,_,_ = model(*args,24,dynamics='ode',max_step=job['config']['rk4_step_hours']/2)
        rms = float(((coarse-fine).square()*q[:,:,None]).sum(1).mean().sqrt())
        motion = float(((coarse-args[0][:,None,:]).square()*q[:,:,None]).sum(1).mean().sqrt())
    if not np.isfinite(rms) or rms/max(motion,1e-12)>.01:
        raise ArithmeticError('Training-only RK4 step-halving error exceeds 1% of movement')
    result = dict(status='TRAINING_ONLY_TIME_SMOKE_PASSED',timing=times,
        solver_relative_difference=rms/max(motion,1e-12),r4_targets_read=False,
        weights_discarded_after_smoke=True,parameters=sum(p.numel() for p in model.parameters()))
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    print(json.dumps(result),flush=True)
    return result


def worker(root,index,device):
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    try:
        for path,digest in manifest['code_hashes'].items():
            if sha256(path) != digest:
                raise ValueError('Time-model training code changed after freezing')
        for job in [j for j in manifest['jobs'] if j['worker']==index]:
            atomic_json(root/f'worker{index}.json',dict(status='training',job=job))
            train_one(root,manifest,job,device)
        atomic_json(root/f'worker{index}.json',dict(status='complete'))
        atomic_json(root/f'worker{index}_complete.json',dict(status='complete'))
    except Exception as error:
        atomic_json(root/f'worker{index}.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--worker',type=int);parser.add_argument('--benchmark-output')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda']);args = parser.parse_args()
    if args.benchmark_output:benchmark(args.root,args.benchmark_output,args.device)
    elif args.worker is not None:worker(args.root,args.worker,args.device)
    else:parser.error('--worker or --benchmark-output required')
