"""Four-card audited capacity study, with shared frozen W7 evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w6_local_router import predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import atomic_json, score_case, aggregate, training_path
from w9_population_router import PopulationResponseRouter, fit_population, make_training_state


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__), *[Path(__file__).with_name(n) for n in ('w9_population_router.py',
    'w7_router_scaling.py', 'w6_local_router.py', 'w6_router_scifate2.py', 'w4_artifacts.py')]]


def prepare(config_path, output, audit_xml):
    config_path = Path(config_path); cfg = json.loads(config_path.read_text())
    if cfg['gfg_jvp_enabled'] or cfg['r4_used_for_optimizer'] or cfg['trainable_mean_predictor']:
        raise ValueError('No GFG/JVP or changed mean/test fitting in this capacity study')
    import xml.etree.ElementTree as ET
    suites = ET.parse(audit_xml).getroot()
    selected = [suites] if suites.tag == 'testsuite' else list(suites)
    if not selected or sum(int(s.get('tests','0')) for s in selected) == 0 or any(
        int(s.get('failures','0'))+int(s.get('errors','0'))+int(s.get('skipped','0')) for s in selected):
        raise ValueError('Passing, nonempty, unskipped audit tests required before launch')
    old_root = ROOT/cfg['reuse_study']; old = json.loads((old_root/'study_manifest.json').read_text())
    pack = Path(old['pack']); pm = json.loads((pack/'manifest.json').read_text())
    if sha256(pack/'manifest.json') != old['pack_manifest_sha256']: raise ValueError('Source feature manifest changed')
    if sha256(pack/'normalization.npz') != pm['normalization_sha256']: raise ValueError('Training normalization changed')
    base = dict(pm['config'])
    for key in ('steps','checkpoint_steps','source_batch','target_batch','learning_rate','weight_decay',
                'residual_regularization','routing_regularization'):
        base[key] = cfg[key]
    base['experts'] = cfg['baseline_experts']; base['log_interval'] = 200
    jobs = []; cases = []; parameters = {}
    with np.load(pack/'normalization.npz',allow_pickle=False) as saved: norm = {k:saved[k].copy() for k in saved.files}
    for spec in cfg['models']:
        mc = dict(base, **spec); setting = f'{spec["name"]}_k{mc["experts"]}_w{mc["hidden"]}'
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                model = PopulationResponseRouter(norm,mc,arm=arm,seed=seed)
                count = sum(p.numel() for p in model.parameters()); parameters[setting] = count
                jobs.append(dict(setting=setting, experts=mc['experts'], hidden=mc['hidden'], arm=arm,
                    seed=seed, config=mc, cost=count*mc['steps'], expected_parameters=count))
            for step in cfg['checkpoint_steps']:
                cases.append(dict(setting=setting,experts=mc['experts'],hidden=mc['hidden'],arm=arm,step=step))
    loads = [0]*cfg['workers']
    for job in sorted(jobs,key=lambda j:-j['cost']):
        index = int(np.argmin(loads)); job['worker'] = index; loads[index] += job['cost']
    for i,case in enumerate(cases): case['worker'] = i%cfg['workers']
    run = Artifacts(output,seed=cfg['seeds'],inputs=[config_path,Path(audit_xml),old_root/'study_manifest.json',
        pack/'manifest.json',pack/'normalization.npz',ROOT/'PLANS/w9_large_model_mathematical_audit.md']+CODE)
    samples = json.loads((old_root/'source_samples.json').read_text())
    run.json('source_samples.json',{f'{p["source_hour"]}_to_{p["target_hour"]}':
        samples[f'{p["source_hour"]}_to_{p["target_hour"]}'] for p in pm['prediction_groups']})
    run.json('study_manifest.json',dict(status='AUDITED_CAPACITY_DESIGN_READY',config=cfg,pack=str(pack),
        pack_manifest_sha256=sha256(pack/'manifest.json'),jobs=jobs,cases=cases,
        evaluation_caches=old['evaluation_caches'],parameter_counts=parameters,
        audit_xml_sha256=sha256(audit_xml),r4_used_for_optimizer=False,
        population_centering='full_source_differentiable',training_energy='source_draw_U_statistic',
        scope='same_series_development'))
    run.json('status.json',dict(status='complete',jobs=len(jobs),cases=len(cases),parameters=parameters))
    print(json.dumps(dict(status='AUDITED_CAPACITY_READY',jobs=len(jobs),cases=len(cases),parameters=parameters)),flush=True)


def read_training(root,manifest,job):
    pack = Path(manifest['pack']); pm = json.loads((pack/'manifest.json').read_text())
    if sha256(pack/'manifest.json') != manifest['pack_manifest_sha256']: raise ValueError('Changed feature manifest')
    if 'normalization_sha256' in pm and sha256(pack/'normalization.npz') != pm['normalization_sha256']:
        raise ValueError('Changed source-only normalization')
    index = manifest['config']['seeds'].index(job['seed']); groups = []
    for item in pm['training_groups']:
        if sha256(item['path']) != item['sha256']: raise ValueError('Changed training counts/features')
        with np.load(item['path'],allow_pickle=False) as saved:
            groups.append({**{k:saved[k].copy() for k in ('z','velocity','target')},
                'global_rate':saved['global_rates'][index].copy()})
    with np.load(pack/'normalization.npz',allow_pickle=False) as saved: norm = {k:saved[k].copy() for k in saved.files}
    return pm,groups,norm,index


def train_job(root,manifest,job,device):
    directory = training_path(root,job)
    if (directory/'prediction_manifest.json').is_file():
        done = json.loads((directory/'prediction_manifest.json').read_text())
        if done['job'] != job: raise ValueError('Completed job identity changed')
        return
    if directory.exists(): raise ValueError('Incomplete run retained; resume_state.pt contains optimizer/RNG for explicit recovery')
    pm,groups,norm,seed_index = read_training(root,manifest,job)
    pack = Path(manifest['pack']); files = [Path(i['path']) for i in pm['training_groups']+pm['prediction_groups']]
    run = Artifacts(directory,seed=job['seed'],inputs=[root/'study_manifest.json',pack/'manifest.json',
        pack/'normalization.npz',root/'source_samples.json']+files+CODE)
    samples = json.loads((root/'source_samples.json').read_text()); artifacts = []; timing = []
    if device != 'cpu': torch.cuda.reset_peak_memory_stats()
    began = time.monotonic(); paused = 0.
    def checkpoint(step,model,optimizer,generator,history):
        nonlocal paused
        if device != 'cpu': torch.cuda.synchronize()
        started = time.monotonic(); elapsed = started-began-paused
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            config=job['config'],arm=job['arm'],seed=job['seed'],step=step),run.directory/f'model_step{step}.pt')
        # Latest optimizer snapshot is atomically replaced; all deployment checkpoints remain.
        state = make_training_state(model,optimizer,generator,step,history)
        temporary = run.directory/'resume_state.pt.tmp'
        torch.save(dict(training_state=state,job=job,pack_manifest_sha256=manifest['pack_manifest_sha256']),temporary)
        temporary.replace(run.directory/'resume_state.pt')
        for item in pm['prediction_groups']:
            if sha256(item['path']) != item['sha256']: raise ValueError('Prediction sources changed')
            with np.load(item['path'],allow_pickle=False) as saved:
                z,v,duration = saved['z'],saved['velocity'],float(saved['duration'])
                anchor,source_mean = saved['anchors'][seed_index],saved['source_mean']
            # Always the full source population, not independent inference mini-batches.
            x,q = predict_local(model,z,v,(anchor-source_mean)/duration,duration)
            x,q = x.astype(float),q.astype(float); q /= q.sum(1,keepdims=True)
            correction = anchor-(x*q[:,:,None]).sum(1).mean(0); x += correction
            if np.linalg.norm(correction)>1e-3: raise ArithmeticError('Mean correction exceeds floating point roundoff')
            x = x.astype('float32'); mean,cov = moments(x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(q))
            usage = q.mean(0); entropy = float(-(q*np.log(np.maximum(q,1e-12))).sum(1).mean())
            label = f'{item["source_hour"]}_to_{item["target_hour"]}'; si = samples[label]
            path = run.directory/f'step{step}_{label}.npz'
            np.savez_compressed(path,candidates=x[si],q=q[si],full_mean=mean,full_cov=cov,anchor=anchor,
                usage=usage,route_entropy=entropy,source_cells=len(q),active_modes=int((usage>=.01).sum()),
                effective_modes=float(np.exp(-(usage*np.log(np.maximum(usage,1e-12))).sum())),
                routing_information=float(-(usage*np.log(np.maximum(usage,1e-12))).sum()-entropy))
            artifacts.append(dict(step=step,source_hour=item['source_hour'],target_hour=item['target_hour'],
                path=str(path.resolve()),sha256=sha256(path)))
        timing.append(dict(step=step,training_seconds=elapsed,
            peak_allocated_bytes=0 if device=='cpu' else torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(status='AUDITED_CAPACITY_CHECKPOINT',setting=job['setting'],arm=job['arm'],
            seed=job['seed'],**timing[-1])),flush=True)
        paused += time.monotonic()-started
    def progress(row): run.json('status.json',dict(status='training',job=job,**row))
    model,history = fit_population(groups,norm,job['config'],arm=job['arm'],seed=job['seed'],device=device,
        progress=progress,checkpoint=checkpoint)
    if history['parameters'] != job['expected_parameters']: raise ValueError('Parameter count mismatch')
    run.json('training.json',dict(**history,checkpoint_timing=timing))
    run.json('prediction_manifest.json',dict(status='AUDITED_CAPACITY_PREDICTIONS_READY',job=job,
        artifacts=artifacts,parameters=history['parameters'],checkpoint_timing=timing,
        global_mean_frozen=True,r4_used_for_optimizer=False,optimizer_state_saved=True))
    run.json('status.json',dict(status='complete')); del model
    if device!='cpu': torch.cuda.empty_cache()


def benchmark(root,output,steps,device):
    if steps < 31: raise ValueError('Need warm-up plus at least one complete 15-group cycle')
    root = Path(root); manifest = json.loads((root/'study_manifest.json').read_text())
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy(device)
    job = next(j for j in manifest['jobs'] if j['config']['architecture']=='residual_large' and j['arm']=='velocity_router')
    _,groups,norm,_ = read_training(root,manifest,job)
    config = dict(job['config'],steps=steps,checkpoint_steps=[],log_interval=15)
    run = Artifacts(output,seed=job['seed'],inputs=[root/'study_manifest.json']+CODE)
    if device!='cpu': torch.cuda.reset_peak_memory_stats()
    rows = []
    def progress(row): rows.append(row);run.json('status.json',dict(status='benchmarking',**row))
    model,history = fit_population(groups,norm,config,arm=job['arm'],seed=job['seed'],device=device,progress=progress)
    seconds = (rows[-1]['elapsed_seconds']-rows[1]['elapsed_seconds'])/(rows[-1]['step']-rows[1]['step'])
    result = dict(status='AUDITED_GPU_BENCHMARK_COMPLETE',parameters=history['parameters'],seconds_per_step=seconds,
        estimated_12000_step_training_seconds=seconds*12000,maximum_training_source_cells=max(len(g['z']) for g in groups),
        peak_allocated_bytes=0 if device=='cpu' else torch.cuda.max_memory_allocated(),
        gradient_finite=True,mean_anchor_max_error=max(r['mean_anchor_error'] for r in rows),fitted_model_discarded=True)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'));print(json.dumps(result),flush=True)


def worker(root,index,device):
    root = Path(root); manifest = json.loads((root/'study_manifest.json').read_text())
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy(device)
    # Formal training additionally requires the actual CUDA math/throughput pilot.
    pilot = json.loads((root/'gpu_benchmark'/'summary.json').read_text())
    if pilot['status']!='AUDITED_GPU_BENCHMARK_COMPLETE' or not pilot['gradient_finite']:
        raise ValueError('CUDA audit pilot has not passed')
    jobs = sorted([j for j in manifest['jobs'] if j['worker']==index],key=lambda j:-j['cost'])
    for i,job in enumerate(jobs):
        atomic_json(root/f'worker{index}.json',dict(status='training',completed_jobs=i,total_jobs=len(jobs),job=job))
        train_job(root,manifest,job,device)
    atomic_json(root/f'training_worker{index}_complete.json',dict(status='complete',jobs=len(jobs)))
    deadline = time.monotonic()+24*3600
    while not all((root/f'training_worker{i}_complete.json').exists() for i in range(manifest['config']['workers'])):
        if time.monotonic()>deadline: raise TimeoutError('Capacity training grid incomplete')
        atomic_json(root/f'worker{index}.json',dict(status='waiting_for_prediction_grid',completed_jobs=len(jobs)))
        time.sleep(15)
    for case in [c for c in manifest['cases'] if c['worker']==index]:
        for cache in manifest['evaluation_caches']:
            if sha256(cache['path']) != cache['sha256']: raise ValueError('Frozen evaluation cache changed')
        atomic_json(root/f'worker{index}.json',dict(status='evaluating',case=case));score_case(root,manifest,case,device)
    atomic_json(root/f'worker{index}_complete.json',dict(status='complete'))
    atomic_json(root/f'worker{index}.json',dict(status='complete',completed_jobs=len(jobs)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('prepare');p.add_argument('--config',default='configs/w9_scifate2_large_audited_v1.json')
    p.add_argument('--output',required=True);p.add_argument('--audit-xml',required=True)
    p = sub.add_parser('benchmark');p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=45);p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p = sub.add_parser('worker');p.add_argument('--root',required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p = sub.add_parser('aggregate');p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true')
    args = parser.parse_args()
    if args.stage=='prepare':prepare(args.config,args.output,args.audit_xml)
    elif args.stage=='benchmark':benchmark(args.root,args.output,args.steps,args.device)
    elif args.stage=='worker':worker(args.root,args.worker,args.device)
    else:aggregate(args.root,args.wait)
