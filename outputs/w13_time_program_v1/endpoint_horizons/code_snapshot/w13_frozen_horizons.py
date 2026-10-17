"""One frozen six-window evaluator for endpoint and genuinely integrated models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w6_local_router import LocalResponseRouter, predict_local
from w6_router_scifate2 import moments
from w7_router_horizons import WINDOWS, read_rate
from w7_router_scaling import blocked_energy, write_csv
from w10_recipe_model import model_arm, shuffle_velocity
from w13_time_model import TimeResponseRouter, elapsed_clock


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__),*[Path(__file__).with_name(n) for n in
    ('w13_time_model.py','w7_router_horizons.py','w10_recipe_model.py','w6_local_router.py',
     'w6_router_scifate2.py','w7_router_scaling.py','w4_artifacts.py')]]
ARMS = ('static_router','velocity_router','shuffled_velocity_router')


def load_model(record, kind, device):
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Frozen model changed')
    saved = torch.load(record['path'],map_location='cpu',weights_only=True)
    sd = saved['state_dict'];norm = {k:sd[k].numpy() for k in ('z_mean','z_scale','v_mean','v_scale')}
    if (saved['arm'],saved['seed']) != (record['arm'],record['seed']):
        raise ValueError('Model identity mismatch')
    if kind == 'endpoint':
        model = LocalResponseRouter(**norm,config=saved['config'],arm=model_arm(record['arm']),seed=record['seed'])
    else:
        if saved['dynamics'] != kind:
            raise ValueError('Time-model integration identity mismatch')
        model = TimeResponseRouter(norm,saved['config'],arm=record['arm'],seed=record['seed'])
    model.load_state_dict(sd);model.to(device).eval()
    return model,saved['config']


def evaluate(spec_path, output, device):
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    spec_path = Path(spec_path);spec = json.loads(spec_path.read_text())
    study_root = Path(spec['reference_root']);study = json.loads((study_root/'study_manifest.json').read_text())
    pack = Path(study['pack']);pm = json.loads((pack/'manifest.json').read_text())
    if sha256(pack/'manifest.json') != study['pack_manifest_sha256']:
        raise ValueError('Feature pack changed')
    config,bc = pm['config'],pm['base_config'];seeds = study['config']['seeds']
    samples = json.loads((study_root/'source_samples.json').read_text())
    global_pack = ROOT/config['global_pack'];global_root = ROOT/config['global_training']
    run = Artifacts(output,seed=seeds,inputs=[spec_path,pack/'manifest.json',
        study_root/'source_samples.json',global_pack/'source_pack.npz']+CODE)
    with np.load(global_pack/'source_pack.npz',allow_pickle=False) as saved:
        base,velocity,origins = [saved[k].copy() for k in ('base','velocity','intervals')]
    origins = origins[:,0];rates = {};mean_models = []
    for seed in seeds:
        path = global_root/f'seed{seed}'/'analytic_task.pt'
        mean_models.append(dict(seed=seed,path=str(path),sha256=sha256(path)))
        rates[seed] = read_rate(path,base,velocity,'analytic_task')
    run.json('mean_models.json',dict(models=mean_models))
    frozen = [];solver = []
    for recipe in spec['models']:
        records = recipe['checkpoints']
        if sorted((r['arm'],r['seed']) for r in records) != sorted((a,s) for a in ARMS for s in seeds):
            raise ValueError('Whole registered arm/seed grid required')
        for arm in ARMS:
            models = {seed:load_model(next(r for r in records if (r['arm'],r['seed'])==(arm,seed)),
                                     recipe['kind'],device) for seed in seeds}
            for start,stop in WINDOWS:
                source_index,item = next((i,p) for i,p in enumerate(pm['prediction_groups']) if p['source_hour']==start)
                if sha256(item['path']) != item['sha256']:
                    raise ValueError('Prediction source changed')
                with np.load(item['path'],allow_pickle=False) as saved:
                    z,v,source_mean = [saved[k].copy() for k in ('z','velocity','source_mean')]
                index = int(np.flatnonzero(origins==start)[0]);si = samples[f'{start}_to_{start+5}']
                duration = elapsed_clock(stop-start,bc['calibration_hours'],bc['time_kernel'])
                xs = [];qs = [];means = [];covs = []
                for seed in seeds:
                    model,cfg = models[seed];rate = rates[seed][index]
                    vv = shuffle_velocity(v,seed,10000+source_index,cfg['shuffle_seed_offset']) if arm=='shuffled_velocity_router' else v
                    if recipe['kind']=='endpoint':
                        x,q = predict_local(model,z,vv,rate,duration)
                    else:
                        with torch.no_grad():
                            args = [torch.as_tensor(a,dtype=torch.float32,device=device) for a in (z,vv,rate)]
                            xx,qq,_ = model(*args,stop-start,dynamics=recipe['kind'])
                            if recipe['kind']=='ode' and (start,stop)==(5,20):
                                fine,_,_ = model(*args,stop-start,dynamics='ode',max_step=cfg['rk4_step_hours']/2)
                                rms = float(((fine-xx).square()*qq[:,:,None]).sum(1).mean().sqrt())
                                movement = float(((xx-args[0][:,None,:]).square()*qq[:,:,None]).sum(1).mean().sqrt())
                                solver.append(dict(model=recipe['name'],arm=arm,seed=seed,
                                    coarse_step_hours=cfg['rk4_step_hours'],fine_step_hours=cfg['rk4_step_hours']/2,
                                    endpoint_weighted_rms_difference=rms,relative_to_movement=rms/max(movement,1e-12)))
                            x,q = xx.cpu().numpy(),qq.cpu().numpy()
                    x,q = x.astype(float),q.astype(float);q /= q.sum(1,keepdims=True)
                    anchor = source_mean+duration*rate
                    correction = anchor-(x*q[:,:,None]).sum(1).mean(0)
                    if np.linalg.norm(correction)>1e-3:
                        raise ArithmeticError('Mean correction exceeds roundoff')
                    x += correction;x = x.astype('float32')
                    m,c = moments(x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(q))
                    xs.append(x[si]);qs.append(q[si]);means.append(m);covs.append(c)
                mean = np.mean(means,axis=0)
                cov = np.mean([c+np.outer(m-mean,m-mean) for m,c in zip(means,covs)],axis=0)
                path = run.directory/f'{recipe["name"]}_{arm}_{start}_to_{stop}.npz'
                np.savez_compressed(path,candidates=np.concatenate(xs),q=np.concatenate(qs),full_mean=mean,full_cov=cov)
                frozen.append(dict(model=recipe['name'],arm=arm,source_hour=start,target_hour=stop,
                    path=str(path.resolve()),sha256=sha256(path)))
                print(json.dumps(dict(status='TIME_PREDICTIONS_FROZEN',model=recipe['name'],arm=arm,
                                      start=start,stop=stop)),flush=True)
            del models
    # All recipes, seeds, arms and horizons are immutable before opening any evaluation cache.
    run.json('frozen_predictions.json',dict(predictions=frozen,all_predictions_ready_before_evaluation=True,
        source_only_predictions=True,optimizer_steps=0,mean_models=mean_models))
    if solver:write_csv(run.directory/'solver_convergence.csv',solver)
    rows = []
    for start,stop in WINDOWS:
        cache = next(c for c in study['evaluation_caches'] if c['target_hour']==stop)
        if sha256(cache['path']) != cache['sha256']:
            raise ValueError('Target cache changed')
        with np.load(cache['path'],allow_pickle=False) as saved:
            y,target_mean,target_cov = [saved[k].copy() for k in ('target','target_mean','target_cov')]
        yt = torch.as_tensor(y,dtype=torch.float64,device=device)
        for record in [r for r in frozen if (r['source_hour'],r['target_hour'])==(start,stop)]:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Frozen forecast changed')
            with np.load(record['path'],allow_pickle=False) as saved:
                x,q,mean,cov = [saved[k].copy() for k in ('candidates','q','full_mean','full_cov')]
            with torch.no_grad():
                energy = float(blocked_energy(torch.as_tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float64,device=device),
                    torch.as_tensor(q.reshape(-1)/len(q),dtype=torch.float64,device=device),yt,1024))
            rows.append(dict(model=record['model'],arm=record['arm'],source_hour=start,target_hour=stop,
                horizon_hours=stop-start,energy_distance=energy,mean_mse=float(np.square(mean-target_mean).mean()),
                variance_ratio=float(np.trace(cov)/np.trace(target_cov))))
        print(json.dumps(dict(status='TIME_WINDOW_EVALUATED',start=start,stop=stop)),flush=True)
    comparisons = []
    for name in [r['name'] for r in spec['models']]:
        for start,stop in WINDOWS:
            group = {r['arm']:r for r in rows if (r['model'],r['source_hour'],r['target_hour'])==(name,start,stop)}
            real = group['velocity_router']['energy_distance']
            comparisons.append(dict(model=name,source_hour=start,target_hour=stop,horizon_hours=stop-start,
                real_energy=real,static_energy=group['static_router']['energy_distance'],
                shuffled_energy=group['shuffled_velocity_router']['energy_distance'],
                gain_vs_static=1-real/group['static_router']['energy_distance'],
                gain_vs_shuffled=1-real/group['shuffled_velocity_router']['energy_distance']))
    averages = []
    for name in [r['name'] for r in spec['models']]:
        for hours in (5,10,15):
            group = [r for r in comparisons if r['model']==name and r['horizon_hours']==hours]
            real,static,shuffled = [float(np.mean([r[k] for r in group])) for k in ('real_energy','static_energy','shuffled_energy')]
            averages.append(dict(model=name,horizon_hours=hours,windows=len(group),real_energy=real,
                static_energy=static,shuffled_energy=shuffled,gain_vs_static=1-real/static,gain_vs_shuffled=1-real/shuffled))
    write_csv(run.directory/'metrics.csv',rows);write_csv(run.directory/'comparisons.csv',comparisons)
    write_csv(run.directory/'horizon_summary.csv',averages)
    result = dict(status='FROZEN_TIME_HORIZONS_COMPLETE',horizon_summary=averages,
        windows_by_horizon={'5':3,'10':2,'15':1},scope='same_series_development',individual_fates=False,
        independent_series=False,solver_convergence=solver,models=spec['models'])
    run.json('summary.json',result)
    report = '# W13 frozen forecast horizons\n\nAll fixed arms and windows; weighted three-seed predictive mixtures.\n\n'
    report += '| Model | Horizon (h) | Windows | Energy | vs static | vs shuffled |\n|---|---:|---:|---:|---:|---:|\n'
    report += '\n'.join(f'| {r["model"]} | {r["horizon_hours"]} | {r["windows"]} | {r["real_energy"]:.6f} | '
        f'{100*r["gain_vs_static"]:+.3f}% | {100*r["gain_vs_shuffled"]:+.3f}% |' for r in averages)
    report += '\n\nShared overlapping windows are not independent biological replicates. '
    report += 'All models retain the same analytical mean branch; static denotes the local router control, not whole-model U-free prediction.\n'
    (run.directory/'RESULTS.md').write_text(report)
    run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda']);args = parser.parse_args()
    evaluate(args.spec,args.output,args.device)
