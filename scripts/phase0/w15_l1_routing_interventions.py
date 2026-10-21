"""Frozen final-L1 ODE route interventions, without any parameter updates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w6_router_scifate2 import moments
from w7_router_horizons import WINDOWS, read_rate
from w7_router_scaling import blocked_energy, write_csv
from w10_recipe_model import shuffle_velocity
from w13_frozen_horizons import load_model
from w13_time_model import elapsed_clock

ROOT = Path(__file__).resolve().parents[2]
INTERVENTIONS = ('soft', 'cohort_reweight', 'uniform_reweight', 'cohort_reintegrate', 'uniform_reintegrate')


def intervene(model, z, velocity, rate, hours, intervention):
    if intervention not in INTERVENTIONS:
        raise ValueError('Unknown frozen L1 intervention')
    context = model.context(z, velocity, rate)
    q = context['q']
    if intervention.startswith('cohort_'):
        q = q.mean(0, keepdim=True).expand_as(q)
    elif intervention.startswith('uniform_'):
        q = torch.full_like(q, 1/q.shape[1])
    if intervention.endswith('_reintegrate'):
        context = dict(context, q=q)
    positions, _, _ = model.integrate(context, hours)
    anchor = context['source_mean'] + elapsed_clock(hours, model.calibration_hours, model.kernel)*rate
    # Preserve the original soft path bitwise. Only a changed terminal weighting
    # requires translation; production scoring also applies W13's float64 anchor.
    if intervention.endswith('_reweight'):
        positions = positions + anchor - (positions*q[:, :, None]).sum(1).mean(0)
    return positions, q, anchor


def run(output, prediction_root, device, *, publish=False):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy(device)
    prediction_root = Path(prediction_root).resolve()
    if not prediction_root.is_relative_to(Path('/data/yuchang/veloroute_w15_l1')):
        raise ValueError('Large predictions must stay in the task-specific data directory')
    spec_path = ROOT/'outputs/w13_time_program_v1/ode_refinement_spec.json'
    spec = json.loads(spec_path.read_text()); recipe = spec['models'][0]
    study_root = Path(spec['reference_root'])
    study = json.loads((study_root/'study_manifest.json').read_text())
    pack = Path(study['pack'])
    if sha256(pack/'manifest.json') != study['pack_manifest_sha256']:
        raise ValueError('Original feature pack changed')
    pm = json.loads((pack/'manifest.json').read_text())
    config, bc = pm['config'], pm['base_config']; seeds = study['config']['seeds']
    source_samples = json.loads((study_root/'source_samples.json').read_text())
    mean_record_path = ROOT/'outputs/w13_time_program_v1/ode_refinement/mean_models.json'
    mean_record = json.loads(mean_record_path.read_text())
    for path, expected in mean_record['data_hashes'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'Original W13 source/code input changed: {path}')
    mean_root, global_pack = [ROOT/config[k] for k in ('global_training', 'global_pack')]
    with np.load(global_pack/'source_pack.npz', allow_pickle=False) as f:
        base, velocity, origins = [f[k].copy() for k in ('base', 'velocity', 'intervals')]
    rates = {}
    for seed in seeds:
        path = mean_root/f'seed{seed}'/'analytic_task.pt'
        expected = next(r for r in mean_record['models'] if r['seed']==seed)
        if sha256(path) != expected['sha256']:
            raise ValueError('Frozen analytical mean changed')
        rates[seed] = read_rate(path, base, velocity, 'analytic_task')
    prediction_root.mkdir(parents=True, exist_ok=False)
    run = Artifacts(output, seed=seeds, inputs=[Path(__file__), spec_path, mean_record_path])
    variants = list(INTERVENTIONS)+['static', 'shuffled']; frozen = []
    for start, stop in WINDOWS:
        source_index, item = next((i,p) for i,p in enumerate(pm['prediction_groups']) if p['source_hour']==start)
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Frozen prediction source changed')
        with np.load(item['path'], allow_pickle=False) as f:
            z, v, source_mean = [f[k].copy() for k in ('z', 'velocity', 'source_mean')]
        selected = source_samples[f'{start}_to_{start+5}']
        rate_index = int(np.flatnonzero(origins[:,0]==start)[0])
        for variant in variants:
            xs, qs, means, covs = [], [], [], []
            for seed in seeds:
                arm = 'static_router' if variant=='static' else 'shuffled_velocity_router' if variant=='shuffled' else 'velocity_router'
                record = next(r for r in recipe['checkpoints'] if (r['arm'],r['seed'])==(arm,seed))
                model, mc = load_model(record, 'ode', device, step_override=recipe['rk4_step_hours'])
                vv = shuffle_velocity(v,seed,10000+source_index,mc['shuffle_seed_offset']) if variant=='shuffled' else v
                with torch.no_grad():
                    args = [torch.as_tensor(a,dtype=torch.float32,device=device) for a in (z,vv,rates[seed][rate_index])]
                    x,q,_ = intervene(model,*args,stop-start,variant if variant in INTERVENTIONS else 'soft')
                    x,q = x.cpu().numpy().astype(float),q.cpu().numpy().astype(float)
                q /= q.sum(1,keepdims=True)
                anchor = source_mean+elapsed_clock(stop-start,bc['calibration_hours'],bc['time_kernel'])*rates[seed][rate_index]
                x += anchor-(x*q[:,:,None]).sum(1).mean(0);x=x.astype('float32')
                mean,cov = moments(x.reshape(-1,x.shape[-1]),q.ravel()/len(q))
                xs.append(x[selected]);qs.append(q[selected]);means.append(mean);covs.append(cov)
                del model
            mean=np.mean(means,axis=0)
            cov=np.mean([c+np.outer(m-mean,m-mean) for m,c in zip(means,covs)],axis=0)
            path=prediction_root/f'{variant}_{start}_to_{stop}.npz'
            np.savez_compressed(path,candidates=np.concatenate(xs),q=np.concatenate(qs),full_mean=mean,full_cov=cov)
            frozen.append(dict(variant=variant,source_hour=start,target_hour=stop,path=str(path),sha256=sha256(path)))
            run.json('status.json',dict(status='freezing_predictions',completed=len(frozen),total=42))
    run.json('frozen_predictions.json',dict(predictions=frozen,optimizer_steps=0,
        all_predictions_ready_before_evaluation=True,original_checkpoints=recipe['checkpoints'],
        entire_source_population_integrated=True,step_hours=recipe['rk4_step_hours']))
    # No target cache is opened before all 42 method/window forecasts are frozen.
    rows=[]
    for start,stop in WINDOWS:
        target=next(r for r in study['evaluation_caches'] if r['target_hour']==stop)
        if sha256(target['path']) != target['sha256']:
            raise ValueError('Original evaluation cache changed')
        with np.load(target['path'],allow_pickle=False) as f:
            y,tm,tc=[f[k].copy() for k in ('target','target_mean','target_cov')]
        for record in [r for r in frozen if (r['source_hour'],r['target_hour'])==(start,stop)]:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Frozen intervention prediction changed')
            with np.load(record['path'],allow_pickle=False) as f:
                x,q,mean,cov=[f[k].copy() for k in ('candidates','q','full_mean','full_cov')]
            with torch.no_grad():
                energy=float(blocked_energy(torch.as_tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float64,device=device),
                    torch.as_tensor(q.ravel()/len(q),dtype=torch.float64,device=device),
                    torch.as_tensor(y,dtype=torch.float64,device=device),1024))
            rows.append(dict(source_hour=start,target_hour=stop,horizon_hours=stop-start,
                variant=record['variant'],energy=energy,mean_mse=float(((mean-tm)**2).mean()),
                variance_ratio=float(np.trace(cov)/np.trace(tc))))
        run.json('status.json',dict(status='scoring',windows_completed=sum(r['variant']=='soft' for r in rows)))
    reference_path=ROOT/'outputs/w13_time_program_v1/ode_refinement/metrics.csv'
    reference=list(csv.DictReader(reference_path.open()));differences=[]
    for row in rows:
        if row['variant'] not in ('soft','static','shuffled'):
            continue
        arm={'soft':'velocity_router','static':'static_router','shuffled':'shuffled_velocity_router'}[row['variant']]
        old=next(r for r in reference if r['arm']==arm and int(r['source_hour'])==row['source_hour']
            and int(r['target_hour'])==row['target_hour'])
        differences.append(abs(row['energy']-float(old['energy_distance'])))
    run.json('reference_reproduction.json',dict(checked=len(differences),reference_sha256=sha256(reference_path),
        maximum_absolute_energy_difference=max(differences),tolerance=1e-5))
    if max(differences)>1e-5:
        raise ArithmeticError('Original soft/static/shuffled reference not reproduced')
    summaries=[]
    for horizon in (5,10,15):
        energies={v:float(np.mean([r['energy'] for r in rows if r['horizon_hours']==horizon and r['variant']==v]))
            for v in variants}
        original_gain=1-energies['soft']/energies['static']
        for variant in variants:
            gain=1-energies[variant]/energies['static']
            summaries.append(dict(horizon_hours=horizon,variant=variant,energy=energies[variant],
                gain_vs_static=gain,retained_gain_fraction=gain/original_gain if original_gain>1e-12 else None))
    write_csv(run.directory/'metrics.csv',rows)
    write_csv(run.directory/'horizon_summary.csv',summaries)
    result=dict(status='FROZEN_FINAL_L1_ROUTING_INTERVENTIONS_COMPLETE',optimizer_steps=0,
        scope='same_series_development',interventions=variants,horizon_summary=summaries,
        reweighting_and_reintegration_separate=True,new_biological_replication=False)
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# Frozen final L1 routing interventions\n\n'+json.dumps(result,indent=2)+'\n')
    if publish:
        from w14_publication_recovery import publish_records
        doc=ROOT/'docs/current/w15_l1_routing_results.md'
        if doc.exists():
            raise ValueError('Existing final L1 intervention report retained')
        doc.write_text((run.directory/'RESULTS.md').read_text())
        try:
            publication=publish_records(ROOT,[doc]+[run.directory/n for n in
                ('RESULTS.md','summary.json','metrics.csv','horizon_summary.csv','frozen_predictions.json','reference_reproduction.json')],
                'L1: distinguish final ODE endpoint reweighting from route reintegration')
        except Exception as error:
            publication=dict(status='publication_pending',error=repr(error),computation_complete=True)
        run.json('publication.json',publication)
    run.json('status.json',dict(status='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--prediction-root',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    run(args.output,args.prediction_root,args.device,publish=args.publish)
