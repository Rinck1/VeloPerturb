"""Frozen W6 K4 models: five-, ten-, fifteen-hour origin-only forecasts, no retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w5_scntseq_repair import effective_minutes
from w6_local_router import LocalResponseRouter, predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import blocked_energy, write_csv


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('identity', 'static_translation', 'global_translation', 'k1', 'static_router', 'velocity_router')
WINDOWS = ((5,10), (10,15), (15,20), (5,15), (10,20), (5,20))


def read_rate(path, base, velocity, arm):
    saved = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    features = base if arm == 'static_task' else np.concatenate((base,velocity), axis=1)
    features = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        result = torch.nn.functional.linear((features-saved['input_mean'])/saved['input_scale'],
            saved['readout.weight'], saved['readout.bias'])
    return result.numpy().astype(float)


def run_horizons(output, evaluation_root, device):
    pack = ROOT/'outputs/w6_local_router_scifate2_pack_v1'
    pm = json.loads((pack/'manifest.json').read_text()); config = pm['config']; bc = pm['base_config']
    model_root = ROOT/'outputs/w6_local_router_scifate2_v1'
    global_root = ROOT/config['global_training']; global_pack = ROOT/config['global_pack']
    eroot = Path(evaluation_root); study = json.loads((eroot/'study_manifest.json').read_text())
    # Feature packs and frozen models suffice for all predictions. Evaluation caches are opened later.
    source_samples = json.loads((eroot/'source_samples.json').read_text())
    code = [Path(__file__), *[Path(__file__).with_name(n) for n in ('w7_router_scaling.py', 'w6_local_router.py',
        'w6_router_scifate2.py', 'w4_artifacts.py', 'w5_scntseq_repair.py')]]
    model_paths = [model_root/f'seed{s}'/f'{a}.pt' for s in config['seeds'] for a in ARMS[3:]]
    model_paths += [global_root/f'seed{s}'/f'{a}.pt' for s in config['seeds'] for a in ('static_task', 'analytic_task')]
    run = Artifacts(output, seed=config['seeds'], inputs=[pack/'manifest.json', eroot/'study_manifest.json',
        eroot/'source_samples.json', global_pack/'source_pack.npz']+model_paths+code)
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    with np.load(global_pack/'source_pack.npz', allow_pickle=False) as saved:
        base, velocity = saved['base'].copy(), saved['velocity'].copy()
        origins = saved['intervals'][:,0].copy()
    models = {}; rates = {}
    for seed in config['seeds']:
        for arm in ('static_task', 'analytic_task'):
            rates[seed,arm] = read_rate(global_root/f'seed{seed}'/f'{arm}.pt', base, velocity, arm)
        for arm in ARMS[3:]:
            saved = torch.load(model_root/f'seed{seed}'/f'{arm}.pt', map_location='cpu', weights_only=True)
            sd = saved['state_dict']; norm = {k:sd[k].numpy() for k in ('z_mean','z_scale','v_mean','v_scale')}
            model = LocalResponseRouter(**norm, config=saved['config'], arm=arm, seed=seed).to(device)
            model.load_state_dict(sd); model.eval(); models[seed,arm] = model
    frozen = []
    for start, stop in WINDOWS:
        item = next(p for p in pm['prediction_groups'] if p['source_hour'] == start)
        if sha256(item['path']) != item['sha256']: raise ValueError('Source features changed')
        with np.load(item['path'], allow_pickle=False) as saved:
            z, v, source_mean = saved['z'], saved['velocity'], saved['source_mean']
        index = int(np.flatnonzero(origins == start)[0])
        si = source_samples[f'{start}_to_{start+5}']
        duration = effective_minutes(stop-start, bc['calibration_hours'], bc['time_kernel'])
        for arm in ARMS:
            means, covs, xs, qs = [], [], [], []
            for seed in config['seeds']:
                rate = np.zeros_like(source_mean) if arm == 'identity' else rates[seed,
                    'static_task' if arm == 'static_translation' else 'analytic_task'][index]
                anchor = source_mean+duration*rate
                if arm in ARMS[:3]:
                    x = (z.astype(float)+anchor-source_mean)[:,None,:]; q = np.ones((len(z),1))
                else:
                    x, q = predict_local(models[seed,arm], z, v, rate, duration)
                    x, q = x.astype(float), q.astype(float)
                q /= q.sum(1, keepdims=True)
                x += anchor-(x*q[:,:,None]).sum(1).mean(0)
                x = x.astype('float32')
                mean, cov = moments(x.reshape(-1,x.shape[-1]), q.reshape(-1)/len(q))
                means.append(mean); covs.append(cov); xs.append(x[si]); qs.append(q[si])
            mean = np.mean(means, axis=0)
            cov = np.mean([c+np.outer(m-mean,m-mean) for m,c in zip(means,covs)], axis=0)
            path = run.directory/f'{arm}_{start}_to_{stop}.npz'
            np.savez_compressed(path, candidates=np.concatenate(xs), q=np.concatenate(qs), full_mean=mean, full_cov=cov)
            frozen.append(dict(arm=arm, source_hour=start, target_hour=stop, path=str(path.resolve()), sha256=sha256(path)))
        print(json.dumps(dict(status='HORIZON_PREDICTIONS_READY', start=start, stop=stop)), flush=True)
    run.json('frozen_predictions.json', dict(predictions=frozen, optimizer_steps=0,
        duration_kernel=bc['time_kernel'], model='frozen_W6_K4_4000_steps', all_predictions_ready_before_evaluation=True))
    rows = []
    for start, stop in WINDOWS:
        cache = next(c for c in study['evaluation_caches'] if c['target_hour'] == stop)
        if sha256(cache['path']) != cache['sha256']: raise ValueError('Evaluation cache changed')
        with np.load(cache['path'], allow_pickle=False) as saved:
            y, target_mean, target_cov = saved['target'], saved['target_mean'], saved['target_cov']
        for arm in ARMS:
            item = next(p for p in frozen if (p['arm'],p['source_hour'],p['target_hour']) == (arm,start,stop))
            if sha256(item['path']) != item['sha256']: raise ValueError('Prediction changed')
            with np.load(item['path'], allow_pickle=False) as saved:
                x, q, mean, cov = [saved[k].copy() for k in ('candidates','q','full_mean','full_cov')]
            with torch.no_grad():
                energy = float(blocked_energy(torch.as_tensor(x.reshape(-1,x.shape[-1]), dtype=torch.float64, device=device),
                    torch.as_tensor(q.reshape(-1)/len(q), dtype=torch.float64, device=device),
                    torch.as_tensor(y, dtype=torch.float64, device=device), 1024))
            rows.append(dict(source_hour=start,target_hour=stop,horizon_hours=stop-start,arm=arm,
                energy_distance=energy,mean_mse=float(np.square(mean-target_mean).mean()),
                variance_ratio=float(np.trace(cov)/np.trace(target_cov))))
        print(json.dumps(dict(status='HORIZON_EVALUATED', start=start, stop=stop)), flush=True)
    comparisons = []
    for start, stop in WINDOWS:
        by_arm = {r['arm']:r for r in rows if (r['source_hour'],r['target_hour']) == (start,stop)}
        d = by_arm['velocity_router']['energy_distance']; a = by_arm['global_translation']['mean_mse']
        comparisons.append(dict(source_hour=start,target_hour=stop,horizon_hours=stop-start,
            router_gain_vs_static=1-d/by_arm['static_router']['energy_distance'],
            router_gain_vs_global_translation=1-d/by_arm['global_translation']['energy_distance'],
            analytic_mean_gain_vs_static=1-a/by_arm['static_translation']['mean_mse']))
    write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'comparisons.csv', comparisons)
    run.json('summary.json', dict(status='FROZEN_ROUTER_HORIZON_STUDY_COMPLETE', comparisons=comparisons,
        maximum_testable_horizon_hours=15, windows_by_horizon={'5':3,'10':2,'15':1},
        fixed_origin_comparisons=[c for c in comparisons if c['source_hour'] == 5], optimizer_steps=0,
        additional_training=False, scope='same_series_development', individual_fates=False))
    (run.directory/'RESULTS.md').write_text('# Frozen router forecast horizons\n\n'
        'W6 K4 models, no retraining; fixed daily-calibrated decay kernel. All origin-to-future windows '
        'within the observed r4 series, at most 15 hours. Predictions are built before evaluation.\n\n'
        +json.dumps(comparisons,indent=2)+'\n')
    run.json('status.json', dict(status='complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--evaluation-root',default='outputs/w7_scifate2_router_scaling_v1')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu'); args = parser.parse_args()
    run_horizons(args.output,args.evaluation_root,args.device)
