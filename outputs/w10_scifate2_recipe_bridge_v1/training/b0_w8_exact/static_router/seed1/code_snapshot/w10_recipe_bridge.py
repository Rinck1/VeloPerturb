"""Run a bounded recipe bridge, report all arms, then matched Adam continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w6_local_router import LocalResponseRouter, predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import atomic_json, score_case, training_path, write_csv
from w9_large_training import read_training
from w9_population_router import make_training_state
from w10_recipe_model import fit_bridge, model_arm, shuffle_velocity


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__), *[Path(__file__).with_name(n) for n in (
    'w10_recipe_model.py','w9_large_training.py','w9_population_router.py','w7_router_scaling.py',
    'w6_local_router.py','w6_router_scifate2.py','w4_artifacts.py')]]


def prepare(config_path, output, *, parent=None, selected=None):
    config_path = Path(config_path); cfg = json.loads(config_path.read_text())
    if cfg['r4_used_for_optimizer'] or cfg['gfg_jvp_enabled'] or cfg['trainable_mean_predictor']:
        raise ValueError('Keep frozen source dynamics and the mean task')
    if cfg['workers'] != len(cfg['physical_gpus']) or not set(cfg['physical_gpus']) <= {0,1,2,3}:
        raise ValueError('Invalid GPU allocation')
    if len(cfg['seeds']) != cfg['workers']: raise ValueError('One seed per worker for matched recipes')
    factors = ('centering','estimator','distance_kernel','regularization_scope','learning_rate')
    for previous,current in zip(cfg['recipes'],cfg['recipes'][1:]):
        if sum(previous[k] != current[k] for k in factors) != 1:
            raise ValueError('Adjacent bridge recipes must change exactly one factor')
    old_root = ROOT/cfg['reuse_study']; old = json.loads((old_root/'study_manifest.json').read_text())
    pack = Path(old['pack']); pm = json.loads((pack/'manifest.json').read_text())
    if sha256(pack/'manifest.json') != old['pack_manifest_sha256']: raise ValueError('Feature manifest changed')
    parent_root = Path(parent) if parent else None
    recipes = cfg['recipes'] if parent is None else [r for r in cfg['recipes'] if r['name']==selected]
    if not recipes: raise ValueError('Unknown continuation recipe')
    if parent is not None:
        parent_manifest = json.loads((parent_root/'study_manifest.json').read_text())
        if parent_manifest['pack_manifest_sha256'] != old['pack_manifest_sha256']:
            raise ValueError('Continuation feature pack changed')
        cfg['steps'] = max(cfg['continuation_steps']); cfg['checkpoint_steps'] = cfg['continuation_steps']
    base = dict(pm['config'])
    for key in ('steps','checkpoint_steps','source_batch','target_batch','weight_decay',
                'residual_regularization','routing_regularization','experts','hidden','expert_rank',
                'router_hidden','log_interval','shuffle_seed_offset'):
        base[key] = cfg[key]
    jobs = []; cases = []
    with np.load(pack/'normalization.npz',allow_pickle=False) as saved: norm = {k:saved[k].copy() for k in saved.files}
    for recipe in recipes:
        setting = recipe['name']; mc = dict(base,**{k:v for k,v in recipe.items() if k!='name'})
        for arm in cfg['arms']:
            for worker,seed in enumerate(cfg['seeds']):
                model = LocalResponseRouter(**norm,config=mc,arm=model_arm(arm),seed=seed)
                job = dict(setting=setting,experts=mc['experts'],hidden=mc['hidden'],arm=arm,seed=seed,
                    config=mc,worker=worker,expected_parameters=sum(p.numel() for p in model.parameters()))
                if parent is not None:
                    path = training_path(parent_root,job)/'resume_state.pt'
                    job['resume_from'] = dict(path=str(path.resolve()),sha256=sha256(path))
                jobs.append(job)
            for step in cfg['checkpoint_steps']:
                cases.append(dict(setting=setting,experts=mc['experts'],hidden=mc['hidden'],arm=arm,
                    step=step,worker=len(cases)%cfg['workers']))
    if parent is None and len(jobs) > cfg['maximum_initial_jobs']: raise ValueError('Initial job budget exceeded')
    inputs = [config_path,old_root/'study_manifest.json',pack/'manifest.json',pack/'normalization.npz',
              ROOT/'PLANS/w10_recipe_bridge.md']+CODE
    if parent: inputs += [parent_root/'report/summary.json']
    run = Artifacts(output,seed=cfg['seeds'],inputs=inputs)
    source_samples = json.loads((old_root/'source_samples.json').read_text())
    run.json('source_samples.json',{f'{p["source_hour"]}_to_{p["target_hour"]}':
        source_samples[f'{p["source_hour"]}_to_{p["target_hour"]}'] for p in pm['prediction_groups']})
    manifest = dict(status='RECIPE_BRIDGE_READY',config=cfg,pack=str(pack),
        pack_manifest_sha256=sha256(pack/'manifest.json'),jobs=jobs,cases=cases,
        evaluation_caches=old['evaluation_caches'],parent=str(parent_root) if parent else None,
        scope='same_series_development',r4_used_for_optimizer=False,
        code_hashes={str(p):sha256(p) for p in CODE},selected_continuation_recipe=selected)
    run.json('study_manifest.json',manifest);run.json('status.json',dict(status='complete',jobs=len(jobs),cases=len(cases)))
    print(json.dumps(dict(status='RECIPE_BRIDGE_READY',root=str(output),jobs=len(jobs),cases=len(cases))),flush=True)


def train_job(root, manifest, job, device):
    directory = training_path(root,job)
    if (directory/'prediction_manifest.json').is_file():
        if json.loads((directory/'prediction_manifest.json').read_text())['job'] != job:
            raise ValueError('Completed job identity changed')
        return
    if directory.exists(): raise ValueError(f'Incomplete run retained for explicit recovery: {directory}')
    if shutil.disk_usage(root).free < 2*1024**3: raise OSError('Less than 2 GiB free; do not risk existing outputs')
    pm,groups,norm,seed_index = read_training(root,manifest,job)
    pack = Path(manifest['pack']); resume = None; initial_seconds = 0.
    inputs = [root/'study_manifest.json',pack/'manifest.json',pack/'normalization.npz',root/'source_samples.json']
    inputs += [Path(i['path']) for i in pm['training_groups']+pm['prediction_groups']]+CODE
    if 'resume_from' in job:
        ref = job['resume_from'];path = Path(ref['path'])
        if sha256(path) != ref['sha256']: raise ValueError('Continuation optimizer snapshot changed')
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved['pack_manifest_sha256'] != manifest['pack_manifest_sha256']: raise ValueError('Resume pack mismatch')
        old_job = saved['job']
        for key in ('setting','arm','seed'): 
            if old_job[key] != job[key]: raise ValueError('Resume arm/seed/recipe mismatch')
        allowed = {'steps','checkpoint_steps'}
        if {k:v for k,v in old_job['config'].items() if k not in allowed} != {
                k:v for k,v in job['config'].items() if k not in allowed}:
            raise ValueError('Continuation may change only duration/checkpoints')
        resume = saved['training_state'];inputs.append(path)
        parent_pm = json.loads((path.parent/'prediction_manifest.json').read_text())
        initial_seconds = parent_pm['checkpoint_timing'][-1]['training_seconds']
    run = Artifacts(directory,seed=job['seed'],inputs=inputs)
    samples = json.loads((root/'source_samples.json').read_text()); artifacts = []; timing = []
    if device != 'cpu': torch.cuda.reset_peak_memory_stats()
    began = time.monotonic(); paused = 0.
    def checkpoint(step,model,optimizer,generator,history):
        nonlocal paused
        if device != 'cpu': torch.cuda.synchronize()
        started = time.monotonic();elapsed = initial_seconds+started-began-paused
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            config=job['config'],arm=job['arm'],seed=job['seed'],step=step),run.directory/f'model_step{step}.pt')
        temporary = run.directory/'resume_state.pt.tmp'
        torch.save(dict(training_state=make_training_state(model,optimizer,generator,step,history),
            job=job,pack_manifest_sha256=manifest['pack_manifest_sha256']),temporary)
        temporary.replace(run.directory/'resume_state.pt')
        for source_index,item in enumerate(pm['prediction_groups']):
            if sha256(item['path']) != item['sha256']: raise ValueError('Prediction source changed')
            with np.load(item['path'],allow_pickle=False) as saved:
                z,v,duration = saved['z'],saved['velocity'],float(saved['duration'])
                anchor,source_mean = saved['anchors'][seed_index],saved['source_mean']
            if job['arm']=='shuffled_velocity_router':
                v = shuffle_velocity(v,job['seed'],10000+source_index,job['config']['shuffle_seed_offset'])
            x,q = predict_local(model,z,v,(anchor-source_mean)/duration,duration)
            x,q = x.astype(float),q.astype(float);q /= q.sum(1,keepdims=True)
            correction = anchor-(x*q[:,:,None]).sum(1).mean(0)
            if np.linalg.norm(correction)>1e-3: raise ArithmeticError('Mean correction exceeds roundoff')
            x += correction;x = x.astype('float32'); mean,cov = moments(x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(q))
            usage = q.mean(0);entropy = float(-(q*np.log(np.maximum(q,1e-12))).sum(1).mean())
            label = f'{item["source_hour"]}_to_{item["target_hour"]}';si = samples[label]
            path = run.directory/f'step{step}_{label}.npz'
            np.savez_compressed(path,candidates=x[si],q=q[si],full_mean=mean,full_cov=cov,anchor=anchor,
                usage=usage,route_entropy=entropy,source_cells=len(q),active_modes=int((usage>=.01).sum()),
                effective_modes=float(np.exp(-(usage*np.log(np.maximum(usage,1e-12))).sum())),
                routing_information=float(-(usage*np.log(np.maximum(usage,1e-12))).sum()-entropy))
            artifacts.append(dict(step=step,source_hour=item['source_hour'],target_hour=item['target_hour'],
                path=str(path.resolve()),sha256=sha256(path)))
        timing.append(dict(step=step,training_seconds=elapsed,
            peak_allocated_bytes=0 if device=='cpu' else torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(status='RECIPE_CHECKPOINT',setting=job['setting'],arm=job['arm'],seed=job['seed'],
            **timing[-1])),flush=True)
        paused += time.monotonic()-started
    def progress(row): run.json('status.json',dict(status='training',job=job,**row))
    model,history = fit_bridge(groups,norm,job['config'],arm=job['arm'],seed=job['seed'],device=device,
        progress=progress,checkpoint=checkpoint,resume=resume)
    if history['parameters'] != job['expected_parameters']: raise ValueError('Parameter count mismatch')
    run.json('training.json',dict(**history,checkpoint_timing=timing))
    run.json('prediction_manifest.json',dict(status='RECIPE_PREDICTIONS_READY',job=job,artifacts=artifacts,
        parameters=history['parameters'],checkpoint_timing=timing,global_mean_frozen=True,
        r4_used_for_optimizer=False,optimizer_state_saved=True))
    run.json('status.json',dict(status='complete'));del model
    if device != 'cpu': torch.cuda.empty_cache()


def worker(root,index,device):
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy(device)
    try:
        for path,digest in manifest['code_hashes'].items():
            if sha256(path) != digest: raise ValueError('Training code changed after design freezing')
        jobs = [j for j in manifest['jobs'] if j['worker']==index]
        for i,job in enumerate(jobs):
            check_peer_failures(root,manifest)
            atomic_json(root/f'worker{index}.json',dict(status='training',completed_jobs=i,total_jobs=len(jobs),job=job))
            train_job(root,manifest,job,device)
        atomic_json(root/f'training_worker{index}_complete.json',dict(status='complete',jobs=len(jobs)))
        deadline = time.monotonic()+6*3600
        while not all((root/f'training_worker{i}_complete.json').exists() for i in range(manifest['config']['workers'])):
            check_peer_failures(root,manifest)
            if time.monotonic()>deadline: raise TimeoutError('Recipe training grid incomplete')
            atomic_json(root/f'worker{index}.json',dict(status='waiting_for_prediction_grid',completed_jobs=len(jobs)))
            time.sleep(10)
        for cache in manifest['evaluation_caches']:
            if sha256(cache['path']) != cache['sha256']: raise ValueError('Evaluation cache changed')
        for case in [c for c in manifest['cases'] if c['worker']==index]:
            atomic_json(root/f'worker{index}.json',dict(status='evaluating',case=case));score_case(root,manifest,case,device)
        atomic_json(root/f'worker{index}_complete.json',dict(status='complete'))
        atomic_json(root/f'worker{index}.json',dict(status='complete',completed_jobs=len(jobs)))
    except Exception as error:
        atomic_json(root/f'worker{index}.json',dict(status='failed',error=repr(error)));raise


def aggregate(root):
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    if (root/'report/summary.json').exists(): return json.loads((root/'report/summary.json').read_text())
    if not all((root/f'worker{i}_complete.json').is_file() for i in range(manifest['config']['workers'])):
        raise ValueError('Entire recipe grid must finish before aggregation')
    paths = [root/'evaluations'/c['setting']/c['arm']/f'step{c["step"]}'/'summary.json' for c in manifest['cases']]
    results = [json.loads(p.read_text()) for p in paths]
    # Summary files share a basename; record their hashes, without ambiguous snapshots.
    run = Artifacts(root/'report',seed=manifest['config']['seeds'],inputs=[root/'study_manifest.json']+CODE)
    rows = [];intervals = [];seed_rows = [];routing = []
    for result in results:
        identity = {k:result['case'][k] for k in ('setting','arm','step')}
        rows.append(dict(**identity,**result['metrics'],parameters=result['parameters'],
            effective_modes=float(np.mean([r['effective_modes'] for r in result['routing']]))))
        intervals += [dict(**identity,**r) for r in result['intervals']]
        seed_rows += [dict(**identity,**r) for r in result['seed_intervals']]
        routing += [dict(**identity,**{k:v for k,v in r.items() if k!='usage'},usage=json.dumps(r['usage']))
                    for r in result['routing']]
    lookup = {(r['setting'],r['arm'],r['step']):r for r in rows};comparisons = []
    for row in rows:
        if row['arm']!='velocity_router': continue
        setting,step = row['setting'],row['step'];controls = {}
        for arm in ('static_router','shuffled_velocity_router'):
            controls[arm] = lookup[(setting,arm,step)]['energy_distance']
        comparisons.append(dict(setting=setting,step=step,real_energy=row['energy_distance'],
            static_energy=controls['static_router'],shuffled_energy=controls['shuffled_velocity_router'],
            gain_vs_static=1-row['energy_distance']/controls['static_router'],
            gain_vs_shuffled=1-row['energy_distance']/controls['shuffled_velocity_router']))
    recipes = [r['name'] for r in manifest['config']['recipes']];bridge = []
    for step in manifest['config']['checkpoint_steps']:
        for previous,current in zip(recipes,recipes[1:]):
            for arm in manifest['config']['arms']:
                if (previous,arm,step) not in lookup or (current,arm,step) not in lookup: continue
                a,b = lookup[(previous,arm,step)],lookup[(current,arm,step)]
                bridge.append(dict(previous=previous,current=current,arm=arm,step=step,
                    previous_energy=a['energy_distance'],current_energy=b['energy_distance'],
                    relative_improvement=1-b['energy_distance']/a['energy_distance']))
    final = [r for r in comparisons if r['step']==manifest['config']['steps']]
    eligible = [r for r in final if r['gain_vs_static']>0 and r['gain_vs_shuffled']>0]
    chosen = min(eligible,key=lambda r:r['real_energy']) if eligible else None
    result = dict(status='RECIPE_BRIDGE_COMPLETE',training_jobs=len(manifest['jobs']),cases=len(results),
        comparisons=comparisons,selected_for_continuation=chosen,
        velocity_pairing_advantage_at_final_step=bool(eligible),scope='same_series_development',
        global_mean_frozen=True,mean_anchor_max_error=max(r['mean_anchor_max_error'] for r in results),
        frozen_evaluation_summaries=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        interpretation='Descriptive development comparisons, not independent confirmation or fate inference')
    for filename,values in (('grid.csv',rows),('comparisons.csv',comparisons),('bridge_effects.csv',bridge),
                            ('interval_metrics.csv',intervals),('seed_metrics.csv',seed_rows),('routing.csv',routing)):
        if values: write_csv(run.directory/filename,values)
    run.json('summary.json',result)
    # Runtime report is generated by the tracked experiment, with full results retained.
    report = '# W10 recipe bridge\n\nAll fixed recipes, arms, seeds and checkpoints are reported.\n\n'
    report += '| Recipe | Steps | Real energy | vs static | vs shuffled |\n|---|---:|---:|---:|---:|\n'
    report += '\n'.join(f'| {r["setting"]} | {r["step"]} | {r["real_energy"]:.6f} | '
        f'{100*r["gain_vs_static"]:+.3f}% | {100*r["gain_vs_shuffled"]:+.3f}% |' for r in comparisons)
    report += '\n\nThe mean branch is frozen in all arms; this isolates local routing, not whole-model S-only.\n'
    (run.directory/'RESULTS.md').write_text(report)
    run.json('status.json',dict(status='complete'));print(json.dumps(result),flush=True)
    return result


def check_peer_failures(root,manifest):
    for index in range(manifest['config']['workers']):
        path = root/f'worker{index}.json'
        if path.exists() and json.loads(path.read_text())['status']=='failed':
            raise RuntimeError(f'Worker {index} failed: {path}')


def wait_for_workers(root):
    manifest = json.loads((root/'study_manifest.json').read_text());deadline = time.monotonic()+6*3600
    while not all((root/f'worker{i}_complete.json').is_file() for i in range(manifest['config']['workers'])):
        check_peer_failures(root,manifest)
        if time.monotonic()>deadline: raise TimeoutError('Bounded recipe study exceeded six hours')
        time.sleep(20)


def advance(root,config_path,job_root):
    root,job_root = Path(root),Path(job_root)
    status = root/'continuous_status.json'
    try:
        atomic_json(status,dict(status='waiting_for_bridge'))
        wait_for_workers(root);result = aggregate(root)
        chosen = result['selected_for_continuation']
        if chosen is None:
            atomic_json(status,dict(status='STOPPED_CORE_ROUTER_PAIRING_PROBLEM',
                reason='No fixed recipe beats both its matched static and shuffled local-router controls at 12000 steps',
                report=str(root/'report/RESULTS.md')))
            return
        next_root = root.parent/(root.name+'_continuation');cfg = json.loads(Path(config_path).read_text())
        prepare(config_path,next_root,parent=root,selected=chosen['setting'])
        for index,gpu in enumerate(cfg['physical_gpus']):
            subprocess.run([sys.executable,str(ROOT/'scripts/phase0/launch_gpu_job.py'),'--gpu',str(gpu),
                '--script','scripts/phase0/w10_recipe_bridge.py','--output',str(job_root/f'continuation_gpu{gpu}'),
                '--','worker','--root',str(next_root),'--worker',str(index)],check=True,cwd=ROOT)
        atomic_json(status,dict(status='continuing_matched_three_arms',selected=chosen,root=str(next_root)))
        wait_for_workers(next_root);continued = aggregate(next_root)
        improvements = [r for r in continued['comparisons'] if r['gain_vs_static']>0 and r['gain_vs_shuffled']>0
                        and r['real_energy']<chosen['real_energy']]
        atomic_json(status,dict(status='ADVANTAGE_EXPANDED' if improvements else 'CONTINUATION_REACHED_BOUNDARY',
            initial_selection=chosen,improved_checkpoints=improvements,continuation=str(next_root),
            report=str(next_root/'report/RESULTS.md'),all_checkpoints_reported=True))
    except Exception as error:
        atomic_json(status,dict(status='STOPPED_EXECUTION_PROBLEM',error=repr(error)));raise


if __name__=='__main__':
    parser = argparse.ArgumentParser();sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('prepare');p.add_argument('--config',default='configs/w10_scifate2_recipe_bridge_v1.json')
    p.add_argument('--output',required=True)
    p = sub.add_parser('worker');p.add_argument('--root',required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p = sub.add_parser('aggregate');p.add_argument('--root',required=True)
    p = sub.add_parser('advance');p.add_argument('--root',required=True);p.add_argument('--job-root',required=True)
    p.add_argument('--config',default='configs/w10_scifate2_recipe_bridge_v1.json')
    args = parser.parse_args()
    if args.stage=='prepare': prepare(args.config,args.output)
    elif args.stage=='worker': worker(args.root,args.worker,args.device)
    elif args.stage=='aggregate': aggregate(args.root)
    else: advance(args.root,args.config,args.job_root)
