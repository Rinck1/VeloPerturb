"""Bounded sci-FATE2 K/width/training-duration study on the frozen W6 features."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_count_rows, read_metadata
from w4_streams import SourceTransform
from w6_local_router import fit_local, predict_local
from w6_router_scifate2 import moments


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__), *[Path(__file__).with_name(n) for n in
    ('w6_local_router.py', 'w6_router_scifate2.py', 'w4_streams.py', 'w4_shorttime.py', 'w4_artifacts.py')]]


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def blocked_energy(x, weights, target, block=1024):
    """Exactly the W6 energy V-statistic, bounded memory, including diagonal terms."""
    if block < 1 or len(x) == 0 or len(target) == 0:
        raise ValueError('Nonempty empirical distributions and positive block size required')
    w = weights/weights.sum()
    cross = x.new_zeros(()); within = x.new_zeros(()); within_y = x.new_zeros(())
    for i in range(0, len(x), block):
        a, wa = x[i:i+block], w[i:i+block]
        for j in range(0, len(target), block):
            cross += wa@torch.cdist(a, target[j:j+block]).sum(1)/len(target)
        for j in range(i, len(x), block):
            term = wa@(torch.cdist(a, x[j:j+block])@w[j:j+block])
            within += term*(1 if i == j else 2)
    for i in range(0, len(target), block):
        for j in range(i, len(target), block):
            within_y += torch.cdist(target[i:i+block], target[j:j+block]).sum()*(1 if i == j else 2)
    return 2*cross-within-within_y/len(target)**2


def build_design(config, base):
    """One-variable-at-a-time settings; duration checkpoints share a single run."""
    settings = [(k, config['baseline_width']) for k in config['expert_counts']]
    settings += [(config['baseline_experts'], w) for w in config['backbone_widths']
                 if w != config['baseline_width']]
    jobs = []; cases = []
    for k, width in settings:
        setting = f'k{k}_w{width}'
        long = (k, width) == (config['baseline_experts'], config['baseline_width'])
        model_config = dict(base, experts=k, hidden=width,
            steps=config['long_steps'] if long else config['steps'],
            checkpoint_steps=config['long_checkpoint_steps'] if long else config['checkpoint_steps'])
        for arm in (['k1'] if k == 1 else config['arms']):
            for step in model_config['checkpoint_steps']:
                cases.append(dict(setting=setting, experts=k, hidden=width, arm=arm, step=step))
            for seed in config['seeds']:
                jobs.append(dict(setting=setting, experts=k, hidden=width, arm=arm, seed=seed,
                    config=model_config, cost=model_config['steps']*(k*k+width*width/4096)))
    # Greedy equal-cost assignment keeps the expensive K32 jobs from landing on one card.
    loads = [0.]*config['workers']
    for job in sorted(jobs, key=lambda j: -j['cost']):
        worker = int(np.argmin(loads)); job['worker'] = worker; loads[worker] += job['cost']
    eval_loads = [0.]*config['workers']
    for case in sorted(cases, key=lambda c: -c['experts']**2):
        worker = int(np.argmin(eval_loads)); case['worker'] = worker
        eval_loads[worker] += case['experts']**2
    return jobs, cases


def prepare(config_path, output):
    config_path = Path(config_path); config = json.loads(config_path.read_text())
    pack = ROOT/config['pack']; pm = json.loads((pack/'manifest.json').read_text())
    reference = ROOT/config['reference_evaluation']
    samples_path = reference/'evaluation_samples.json'
    samples = json.loads(samples_path.read_text())
    base = pm['config']; bc = pm['base_config']
    if config['seeds'] != base['seeds'] or config['r4_used_for_optimizer']:
        raise ValueError('Keep the W6 seeds and training split')
    jobs, cases = build_design(config, base)
    run = Artifacts(output, seed=config['seeds'], inputs=[config_path, pack/'manifest.json', samples_path,
        Path(pm['data_path']), Path(pm['transform_path'])]+CODE)
    transform = SourceTransform.load(pm['transform_path']); obs, _ = read_metadata(pm['data_path'])
    rep, tp = [obs[k].astype(str).to_numpy() for k in ('rep', 'timepoint')]
    sources = {}; caches = []
    for item in pm['prediction_groups']:
        start, stop = item['source_hour'], item['target_hour']; label = f'{start}_to_{stop}'
        si = np.asarray(samples[label]['source_rows']); ti = np.asarray(samples[label]['target_rows'])
        sources[label] = si.tolist()
        rows = np.flatnonzero((rep == bc['prediction_replicate']) & (tp == f'{stop:02d}h'))
        s, u = [read_count_rows(pm['data_path'], key, rows) for key in ('spliced', 'unspliced')]
        keep = (np.asarray(s.sum(1)).ravel() >= bc['minimum_spliced']) & (np.asarray(u.sum(1)).ravel() >= bc['minimum_unspliced'])
        target = transform.state(s[keep])
        with np.load(item['path'], allow_pickle=False) as saved:
            z = saved['z'].astype(float); mean = saved['source_mean']; anchors = saved['anchors']
        z += mean-z.mean(0)
        path = run.directory/f'evaluation_{label}.npz'
        np.savez_compressed(path, target=target[ti], target_mean=target.mean(0),
            target_cov=np.cov(target, rowvar=False, bias=True), target_cells=len(target),
            source=z[si], source_mean=mean, source_cov=np.cov(z, rowvar=False, bias=True),
            source_cells=len(z), anchors=anchors, source_rows=si, target_rows=ti)
        caches.append(dict(source_hour=start, target_hour=stop, path=str(path.resolve()), sha256=sha256(path)))
    run.json('source_samples.json', sources)
    run.json('study_manifest.json', dict(status='SCALING_DESIGN_READY', config=config, pack=str(pack.resolve()),
        pack_manifest_sha256=sha256(pack/'manifest.json'), jobs=jobs, cases=cases,
        evaluation_caches=caches, r4_used_for_optimizer=False, scope='same_series_development'))
    run.json('status.json', dict(status='complete', jobs=len(jobs), evaluation_cases=len(cases)))
    print(json.dumps(dict(status='SCALING_DESIGN_READY', jobs=len(jobs), cases=len(cases))), flush=True)


def training_path(root, job):
    return root/'training'/job['setting']/job['arm']/f'seed{job["seed"]}'


def train_job(root, manifest, job, device):
    directory = training_path(root, job)
    if (directory/'prediction_manifest.json').exists():
        saved = json.loads((directory/'prediction_manifest.json').read_text())
        if saved['job'] != job: raise ValueError('Existing job configuration changed')
        return
    if directory.exists(): raise ValueError(f'Incomplete job, preserve and restart under a new study root: {directory}')
    pack = Path(manifest['pack']); pm = json.loads((pack/'manifest.json').read_text())
    seed_index = manifest['config']['seeds'].index(job['seed'])
    files = [Path(i['path']) for i in pm['training_groups']+pm['prediction_groups']]
    run = Artifacts(directory, seed=job['seed'], inputs=[root/'study_manifest.json', pack/'manifest.json',
        pack/'normalization.npz', root/'source_samples.json']+files+CODE)
    source_samples = json.loads((root/'source_samples.json').read_text())
    groups = []
    for item in pm['training_groups']:
        if sha256(item['path']) != item['sha256']: raise ValueError('Training features changed')
        with np.load(item['path'], allow_pickle=False) as saved:
            groups.append({**{k:saved[k].copy() for k in ('z', 'velocity', 'target')},
                           'global_rate':saved['global_rates'][seed_index].copy()})
    with np.load(pack/'normalization.npz', allow_pickle=False) as saved:
        normalization = {k:saved[k].copy() for k in saved.files}
    if device != 'cpu': torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic(); paused = 0.; artifacts = []; timing = []
    def checkpoint(step, model):
        nonlocal paused
        if device != 'cpu': torch.cuda.synchronize()
        began = time.monotonic(); elapsed = began-start_time-paused
        path = run.directory/f'model_step{step}.pt'
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            config=job['config'], arm=job['arm'], seed=job['seed'], step=step), path)
        for item in pm['prediction_groups']:
            if sha256(item['path']) != item['sha256']: raise ValueError('Prediction source changed')
            with np.load(item['path'], allow_pickle=False) as saved:
                z, v = saved['z'], saved['velocity']; duration = float(saved['duration'])
                anchor = saved['anchors'][seed_index]; source_mean = saved['source_mean']
            positions, q = predict_local(model, z, v, (anchor-source_mean)/duration, duration)
            positions, q = positions.astype(float), q.astype(float); q /= q.sum(1, keepdims=True)
            shift = anchor-(positions*q[:,:,None]).sum(1).mean(0); positions += shift
            # Match W6 storage rounding BEFORE both moments and subsampling.
            positions = positions.astype('float32')
            mean, cov = moments(positions.reshape(-1, positions.shape[-1]), q.reshape(-1)/len(q))
            usage = q.mean(0); entropy = float(-(q*np.log(np.maximum(q, 1e-12))).sum(1).mean())
            label = f'{item["source_hour"]}_to_{item["target_hour"]}'
            si = source_samples[label]
            path = run.directory/f'step{step}_{label}.npz'
            np.savez_compressed(path, candidates=positions[si], q=q[si], full_mean=mean, full_cov=cov,
                anchor=anchor, usage=usage, route_entropy=entropy, source_cells=len(q),
                active_modes=int((usage >= .01).sum()), effective_modes=float(np.exp(-(usage*np.log(np.maximum(usage,1e-12))).sum())),
                routing_information=float(-(usage*np.log(np.maximum(usage,1e-12))).sum()-entropy))
            artifacts.append(dict(step=step, source_hour=item['source_hour'], target_hour=item['target_hour'],
                path=str(path.resolve()), sha256=sha256(path)))
        timing.append(dict(step=step, training_seconds=elapsed,
            peak_allocated_bytes=0 if device == 'cpu' else torch.cuda.max_memory_allocated()))
        paused += time.monotonic()-began
        print(json.dumps(dict(status='SCALING_CHECKPOINT_READY', setting=job['setting'], arm=job['arm'],
            seed=job['seed'], **timing[-1])), flush=True)
    def progress(row): run.json('status.json', dict(status='training', job=job, **row))
    model, history = fit_local(groups, normalization, job['config'], arm=job['arm'], seed=job['seed'],
        device=device, progress=progress, checkpoint=checkpoint)
    run.json('training.json', dict(**history, checkpoint_timing=timing))
    run.json('prediction_manifest.json', dict(status='SCALING_PREDICTIONS_READY', job=job, artifacts=artifacts,
        parameters=history['parameters'], checkpoint_timing=timing, global_mean_frozen=True, r4_used_for_optimizer=False))
    run.json('status.json', dict(status='complete'))
    del model
    if device != 'cpu': torch.cuda.empty_cache()


def score_case(root, manifest, case, device):
    output = root/'evaluations'/case['setting']/case['arm']/f'step{case["step"]}'
    if (output/'summary.json').exists(): return
    jobs = [j for j in manifest['jobs'] if j['setting'] == case['setting'] and j['arm'] == case['arm']]
    frozen = []; manifests = []
    for job in jobs:
        pm = json.loads((training_path(root, job)/'prediction_manifest.json').read_text())
        if pm['job'] != job or pm['r4_used_for_optimizer']: raise ValueError('Model split or identity changed')
        manifests.append(pm)
        for a in pm['artifacts']:
            if a['step'] == case['step']:
                if sha256(a['path']) != a['sha256']: raise ValueError('Prediction artifact changed')
                frozen.append(dict(seed=job['seed'], **a))
    run = Artifacts(output, seed=manifest['config']['seeds'], inputs=[root/'study_manifest.json']+
        [Path(a['path']) for a in frozen]+[Path(c['path']) for c in manifest['evaluation_caches']]+CODE)
    rows = []; routing = []; seed_rows = []
    block = manifest['config']['energy_block_size']
    for cache in manifest['evaluation_caches']:
        with np.load(cache['path'], allow_pickle=False) as saved:
            truth = {k:saved[k].copy() for k in saved.files}
        means, covs, xs, ws = [], [], [], []
        y = torch.as_tensor(truth['target'], dtype=torch.float64, device=device)
        for seed in manifest['config']['seeds']:
            item = next(a for a in frozen if a['seed'] == seed and a['source_hour'] == cache['source_hour'])
            with np.load(item['path'], allow_pickle=False) as saved:
                data = {k:saved[k].copy() for k in saved.files}
            x = data['candidates'].astype(float).reshape(-1, y.shape[1]); w = data['q'].reshape(-1)/len(data['q'])
            xs.append(x); ws.append(w/len(jobs)); means.append(data['full_mean']); covs.append(data['full_cov'])
            with torch.no_grad():
                seed_energy = float(blocked_energy(torch.as_tensor(x, device=device),
                    torch.as_tensor(w, device=device), y, block))
            seed_rows.append(dict(seed=seed, source_hour=cache['source_hour'], target_hour=cache['target_hour'],
                energy_distance=seed_energy))
            routing.append(dict(seed=seed, source_hour=cache['source_hour'], target_hour=cache['target_hour'],
                active_modes=int(data['active_modes']), effective_modes=float(data['effective_modes']),
                route_entropy=float(data['route_entropy']), routing_information=float(data['routing_information']),
                usage=data['usage'].tolist()))
        mean = np.mean(means, 0)
        cov = np.mean([c+np.outer(m-mean,m-mean) for m,c in zip(means,covs)], 0)
        with torch.no_grad():
            energy = float(blocked_energy(torch.as_tensor(np.concatenate(xs), device=device),
                torch.as_tensor(np.concatenate(ws), device=device), y, block))
        ratio = float(np.trace(cov)/np.trace(truth['target_cov']))
        rows.append(dict(source_hour=cache['source_hour'], target_hour=cache['target_hour'], energy_distance=energy,
            mean_mse=float(np.square(mean-truth['target_mean']).mean()), variance_ratio=ratio,
            absolute_log_variance_error=abs(math.log(max(ratio, 1e-12))),
            covariance_relative_error=float(np.linalg.norm(cov-truth['target_cov'])/np.linalg.norm(truth['target_cov'])),
            mean_anchor_error=float(np.linalg.norm(mean-truth['anchors'].mean(0)))))
    metric_keys = ('energy_distance', 'mean_mse', 'absolute_log_variance_error', 'covariance_relative_error')
    result = dict(status='SCALING_CASE_EVALUATED', case=case,
        metrics={key:float(np.mean([r[key] for r in rows])) for key in metric_keys}, intervals=rows,
        parameters=manifests[0]['parameters'], routing=routing, seed_intervals=seed_rows,
        average_training_seconds=float(np.mean([next(t['training_seconds'] for t in pm['checkpoint_timing']
            if t['step'] == case['step']) for pm in manifests])),
        maximum_peak_allocated_bytes=max(next(t['peak_allocated_bytes'] for t in pm['checkpoint_timing']
            if t['step'] == case['step']) for pm in manifests),
        mean_anchor_max_error=max(r['mean_anchor_error'] for r in rows))
    run.json('summary.json', result); write_csv(output/'metrics.csv', rows)
    write_csv(output/'seed_metrics.csv', seed_rows); run.json('status.json', dict(status='complete'))
    print(json.dumps(dict(status='SCALING_CASE_EVALUATED', **case, **result['metrics'])), flush=True)


def worker(root, index, device):
    root = Path(root); manifest = json.loads((root/'study_manifest.json').read_text())
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    pack_manifest = Path(manifest['pack'])/'manifest.json'
    if sha256(pack_manifest) != manifest['pack_manifest_sha256']: raise ValueError('Feature manifest changed')
    jobs = sorted([j for j in manifest['jobs'] if j['worker'] == index], key=lambda j:-j['cost'])
    for i, job in enumerate(jobs):
        atomic_json(root/f'worker{index}.json', dict(status='training', completed_jobs=i, total_jobs=len(jobs), job=job))
        train_job(root, manifest, job, device)
    atomic_json(root/f'training_worker{index}_complete.json', dict(status='complete', jobs=len(jobs)))
    deadline = time.monotonic()+12*3600
    while not all((root/f'training_worker{i}_complete.json').exists() for i in range(manifest['config']['workers'])):
        if time.monotonic() > deadline: raise TimeoutError('Scaling training workers incomplete')
        atomic_json(root/f'worker{index}.json', dict(status='waiting_for_prediction_grid', completed_jobs=len(jobs)))
        time.sleep(10)
    for case in [c for c in manifest['cases'] if c['worker'] == index]:
        atomic_json(root/f'worker{index}.json', dict(status='evaluating', case=case))
        score_case(root, manifest, case, device)
    atomic_json(root/f'worker{index}_complete.json', dict(status='complete'))
    atomic_json(root/f'worker{index}.json', dict(status='complete', completed_jobs=len(jobs)))


def aggregate(root, wait):
    root = Path(root); manifest = json.loads((root/'study_manifest.json').read_text())
    deadline = time.monotonic()+12*3600
    while not all((root/f'worker{i}_complete.json').exists() for i in range(manifest['config']['workers'])):
        if not wait or time.monotonic() > deadline: raise TimeoutError('Scaling evaluation workers incomplete')
        print(json.dumps(dict(status='WAITING_FOR_SCALING_WORKERS')), flush=True); time.sleep(15)
    paths = [root/'evaluations'/c['setting']/c['arm']/f'step{c["step"]}'/'summary.json' for c in manifest['cases']]
    run = Artifacts(root/'report', seed=manifest['config']['seeds'], inputs=[root/'study_manifest.json']+CODE)
    results = [json.loads(p.read_text()) for p in paths]
    rows = []; intervals = []; seed_rows = []; routing_rows = []
    for result in results:
        c = result['case']; identity = {k:c[k] for k in ('setting', 'experts', 'hidden', 'arm', 'step')}
        row = dict(**identity, **result['metrics'], parameters=result['parameters'],
            training_seconds=result['average_training_seconds'], peak_allocated_bytes=result['maximum_peak_allocated_bytes'],
            mean_active_modes=float(np.mean([r['active_modes'] for r in result['routing']])),
            mean_effective_modes=float(np.mean([r['effective_modes'] for r in result['routing']])),
            mean_routing_information=float(np.mean([r['routing_information'] for r in result['routing']])))
        rows.append(row)
        intervals += [dict(**identity, **r) for r in result['intervals']]
        seed_rows += [dict(**identity, **r) for r in result['seed_intervals']]
        routing_rows += [dict(**identity, **{k:v for k,v in r.items() if k != 'usage'}, usage=json.dumps(r['usage']))
                         for r in result['routing']]
    comparisons = []
    for r in rows:
        if r['arm'] != 'velocity_router': continue
        control = next(c for c in rows if (c['setting'], c['step'], c['arm']) == (r['setting'], r['step'], 'static_router'))
        comparisons.append(dict(setting=r['setting'], experts=r['experts'], hidden=r['hidden'], step=r['step'],
            velocity_energy=r['energy_distance'], static_energy=control['energy_distance'],
            relative_velocity_gain=1-r['energy_distance']/control['energy_distance']))
    for name, values in (('grid.csv', rows), ('interval_metrics.csv', intervals), ('seed_metrics.csv', seed_rows),
                         ('routing.csv', routing_rows), ('velocity_comparisons.csv', comparisons)):
        write_csv(run.directory/name, values)
    dynamic = [r for r in rows if r['arm'] == 'velocity_router']
    best = min(dynamic, key=lambda r:r['energy_distance'])
    # Only K-doublings with fixed width, fixed training steps count toward K saturation.
    cfg = manifest['config']
    k_curve = sorted([r for r in rows if r['hidden'] == cfg['baseline_width'] and r['step'] == cfg['steps']
        and r['arm'] in ('k1', 'velocity_router')], key=lambda r:r['experts'])
    transitions = [dict(from_K=a['experts'], to_K=b['experts'],
        relative_improvement=1-b['energy_distance']/a['energy_distance']) for a,b in zip(k_curve,k_curve[1:])]
    n = cfg['saturation_consecutive_doublings']; threshold = cfg['saturation_relative_improvement']
    saturation = next((transitions[i+n-1]['to_K'] for i in range(len(transitions)-n+1)
        if all(t['relative_improvement'] < threshold for t in transitions[i:i+n])), None)
    result = dict(status='SCIFATE2_ROUTER_SCALING_COMPLETE', training_jobs=len(manifest['jobs']), cases=len(results),
        best_observed_development_configuration=best, velocity_comparisons=comparisons,
        K_doubling_curve=transitions, practical_K_saturation_candidate=saturation,
        saturation_rule=dict(relative_improvement_threshold=threshold, consecutive_doublings=n),
        no_saturation_means_tested_range_only=True, global_mean_frozen=True, scope='same_series_development',
        mean_anchor_max_error=max(r['mean_anchor_max_error'] for r in results),
        frozen_evaluation_summaries=[dict(path=str(p), sha256=sha256(p)) for p in paths])
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# sci-FATE2 router scaling\n\n'
        'One router, K response experts; all K/width/checkpoint results are reported. '
        'The global mean and W6 evaluation samples are fixed. This is same-series development, '
        'not a new independent holdout. Seed ensembles mix weighted candidates, not endpoint averages.\n\n'
        +json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', default='configs/w7_scifate2_router_scaling_v1.json')
    p.add_argument('--output', required=True)
    p = sub.add_parser('worker'); p.add_argument('--root', required=True); p.add_argument('--worker', type=int, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p = sub.add_parser('aggregate'); p.add_argument('--root', required=True); p.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.config, args.output)
    elif args.stage == 'worker': worker(args.root, args.worker, args.device)
    else: aggregate(args.root, args.wait)
