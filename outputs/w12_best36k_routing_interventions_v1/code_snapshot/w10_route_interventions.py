"""Frozen-model interventions separate cohort mode masses from cell assignments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w6_local_router import LocalResponseRouter, predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import blocked_energy, training_path, write_csv


VARIANTS = ('learned','cohort_average','permuted_assignment','uniform','hard_argmax')


def intervention_probabilities(q,seed,source_index):
    q = np.asarray(q,dtype=float);q = q/q.sum(1,keepdims=True)
    rng = np.random.default_rng(np.random.SeedSequence([700000,seed,source_index]))
    result = dict(learned=q,cohort_average=np.broadcast_to(q.mean(0),q.shape).copy(),
        permuted_assignment=q[rng.permutation(len(q))].copy(),uniform=np.full_like(q,1/q.shape[1]),
        hard_argmax=np.eye(q.shape[1])[q.argmax(1)])
    if any(not np.allclose(p.sum(1),1) or (p<0).any() for p in result.values()):
        raise ValueError('Invalid intervention weights')
    return result


def run(root,output,recipe,step):
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    pack = Path(manifest['pack']);pm = json.loads((pack/'manifest.json').read_text())
    if sha256(pack/'manifest.json') != manifest['pack_manifest_sha256']:raise ValueError('Source pack changed')
    jobs = [j for j in manifest['jobs'] if j['setting']==recipe and j['arm']=='velocity_router']
    checkpoints = [training_path(root,j)/f'model_step{step}.pt' for j in jobs]
    sources = [Path(i['path']) for i in pm['prediction_groups']]
    inputs = [root/'study_manifest.json',pack/'manifest.json',pack/'normalization.npz',Path(__file__)]+checkpoints+sources
    run = Artifacts(output,seed=manifest['config']['seeds'],inputs=inputs)
    with np.load(pack/'normalization.npz',allow_pickle=False) as saved:norm = {k:saved[k].copy() for k in saved.files}
    samples = json.loads((root/'source_samples.json').read_text());frozen = []
    for seed_index,job in enumerate(jobs):
        checkpoint = torch.load(checkpoints[seed_index],map_location='cpu',weights_only=True)
        model = LocalResponseRouter(**norm,config=job['config'],arm='velocity_router',seed=job['seed'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        for source_index,item in enumerate(pm['prediction_groups']):
            if sha256(item['path']) != item['sha256']:raise ValueError('Prediction source changed')
            with np.load(item['path'],allow_pickle=False) as saved:
                z,v,duration = saved['z'],saved['velocity'],float(saved['duration'])
                si = manifest['config']['seeds'].index(job['seed'])
                anchor,source_mean = saved['anchors'][si],saved['source_mean']
            x,q = predict_local(model,z,v,(anchor-source_mean)/duration,duration)
            x = x.astype(float);probabilities = intervention_probabilities(q,job['seed'],source_index)
            shifts,means,covs = [],[],[]
            for variant in VARIANTS:
                p = probabilities[variant];shift = anchor-(x*p[:,:,None]).sum(1).mean(0)
                mean,cov = moments((x+shift).reshape(-1,x.shape[-1]),p.reshape(-1)/len(p))
                shifts.append(shift);means.append(mean);covs.append(cov)
            label = f'{item["source_hour"]}_to_{item["target_hour"]}';indices = samples[label]
            path = run.directory/f'seed{job["seed"]}_{label}.npz'
            np.savez_compressed(path,candidates=x[indices].astype('float32'),
                q=np.stack([probabilities[k][indices] for k in VARIANTS]),shifts=np.stack(shifts),
                full_mean=np.stack(means),full_cov=np.stack(covs))
            frozen.append(dict(seed=job['seed'],source_hour=item['source_hour'],target_hour=item['target_hour'],
                path=str(path.resolve()),sha256=sha256(path)))
    # All source-only interventions and mean corrections are frozen before reading evaluation targets.
    run.json('prediction_manifest.json',dict(status='FROZEN_ROUTING_INTERVENTIONS',artifacts=frozen,
        variants=VARIANTS,recipe=recipe,step=step,future_targets_read=False,model_retrained=False))
    rows = []
    for cache in manifest['evaluation_caches']:
        if sha256(cache['path']) != cache['sha256']:raise ValueError('Evaluation cache changed')
        with np.load(cache['path'],allow_pickle=False) as saved:truth = {k:saved[k].copy() for k in saved.files}
        y = torch.as_tensor(truth['target'],dtype=torch.float64)
        for variant_index,variant in enumerate(VARIANTS):
            xs,ws,means,covs = [],[],[],[]
            for record in [r for r in frozen if r['source_hour']==cache['source_hour']]:
                if sha256(record['path']) != record['sha256']:raise ValueError('Frozen intervention changed')
                with np.load(record['path'],allow_pickle=False) as saved:
                    x = saved['candidates'].astype(float)+saved['shifts'][variant_index]
                    q = saved['q'][variant_index]
                    xs.append(x.reshape(-1,x.shape[-1]));ws.append(q.reshape(-1)/(len(q)*len(jobs)))
                    means.append(saved['full_mean'][variant_index]);covs.append(saved['full_cov'][variant_index])
            mean = np.mean(means,0)
            cov = np.mean([c+np.outer(m-mean,m-mean) for c,m in zip(covs,means)],0)
            with torch.no_grad():
                energy = float(blocked_energy(torch.as_tensor(np.concatenate(xs)),
                    torch.as_tensor(np.concatenate(ws)),y,manifest['config']['energy_block_size']))
            ratio = float(np.trace(cov)/np.trace(truth['target_cov']))
            rows.append(dict(variant=variant,source_hour=cache['source_hour'],target_hour=cache['target_hour'],
                energy_distance=energy,variance_ratio=ratio,absolute_log_variance_error=abs(math.log(ratio)),
                covariance_relative_error=float(np.linalg.norm(cov-truth['target_cov'])/np.linalg.norm(truth['target_cov'])),
                mean_mse=float(np.mean((mean-truth['target_mean'])**2)),
                mean_anchor_error=float(np.linalg.norm(mean-truth['anchors'].mean(0)))))
    metrics = [{**dict(variant=v),**{key:float(np.mean([r[key] for r in rows if r['variant']==v])) for key in
               ('energy_distance','mean_mse','absolute_log_variance_error','covariance_relative_error')}} for v in VARIANTS]
    write_csv(run.directory/'metrics.csv',metrics);write_csv(run.directory/'interval_metrics.csv',rows)
    run.json('summary.json',dict(status='FROZEN_ROUTING_INTERVENTIONS_COMPLETE',recipe=recipe,step=step,
        metrics=metrics,scope='same_series_development',model_retrained=False,
        mean_anchor_max_error=max(r['mean_anchor_error'] for r in rows),
        qualification='Changing routing only in a trained model; not equivalent to training a cohort-only model'))
    (run.directory/'RESULTS.md').write_text('# Frozen routing interventions\n\n'+json.dumps(metrics,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--recipe',default='b0_w8_exact');parser.add_argument('--step',type=int,default=12000)
    args = parser.parse_args();run(args.root,args.output,args.recipe,args.step)
