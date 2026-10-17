"""Source-only marginal calibration: can kinetic cohort masses rescue an S student?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch

from w4_artifacts import Artifacts,sha256
from w6_local_router import LocalResponseRouter,predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import blocked_energy,training_path,write_csv
from w10_recipe_model import model_arm,shuffle_velocity
from w12_privileged_router import routing_q


VARIANTS=('S_student','S_student_kinetic_mass_calibrated','teacher_cohort_average')


def calibrate_rows(probabilities,target_mass,*,iterations=1000,tolerance=1e-8):
    """KL projection to row sums 1 and source-cohort column means target_mass."""
    q=np.asarray(probabilities,dtype=float);mass=np.asarray(target_mass,dtype=float)
    if q.ndim!=2 or mass.shape!=(q.shape[1],) or (q<0).any() or (mass<=0).any():raise ValueError('Positive aligned source routing required')
    mass=mass/mass.sum();logq=np.log(np.maximum(q,1e-12));v=np.zeros(q.shape[1])
    for step in range(iterations):
        u=-logsumexp(logq+v[None],axis=1)
        v=np.log(len(q)*mass)-logsumexp(logq+u[:,None],axis=0)
        if step%10==0 or step+1==iterations:
            u=-logsumexp(logq+v[None],axis=1);p=np.exp(logq+u[:,None]+v[None])
            if np.max(np.abs(p.mean(0)-mass))<tolerance:return p,step+1
    raise ArithmeticError('Source-marginal projection did not converge')


def run(root,output):
    root=Path(root);manifest=json.loads((root/'study_manifest.json').read_text())
    pack=Path(manifest['pack']);pm=json.loads((pack/'manifest.json').read_text())
    if sha256(pack/'manifest.json')!=manifest['pack_manifest_sha256']:raise ValueError('Source pack changed')
    paths=[training_path(root,j)/'model_step4000.pt' for j in manifest['jobs']]
    inputs=[root/'study_manifest.json',pack/'manifest.json',pack/'normalization.npz',Path(__file__)]+paths
    run=Artifacts(output,seed=manifest['config']['seeds'],inputs=inputs)
    with np.load(pack/'normalization.npz',allow_pickle=False) as s:norm={k:s[k].copy() for k in s.files}
    samples=json.loads((root/'source_samples.json').read_text());frozen=[];calibration=[]
    for job,path in zip(manifest['jobs'],paths):
        if sha256(job['teacher']['path'])!=job['teacher']['sha256']:raise ValueError('Teacher changed')
        student=LocalResponseRouter(**norm,config=job['config'],arm='static_router',seed=job['seed'])
        student.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict']);student.eval()
        teacher=LocalResponseRouter(**norm,config=job['config'],arm=model_arm(job['arm']),seed=job['seed'])
        teacher.load_state_dict(torch.load(job['teacher']['path'],map_location='cpu',weights_only=True)['state_dict']);teacher.eval()
        for source_index,item in enumerate(pm['prediction_groups']):
            if sha256(item['path'])!=item['sha256']:raise ValueError('Source input changed')
            with np.load(item['path'],allow_pickle=False) as s:
                z,v,duration=s['z'],s['velocity'],float(s['duration']);si=manifest['config']['seeds'].index(job['seed'])
                anchor,source_mean=s['anchors'][si],s['source_mean']
            tv=shuffle_velocity(v,job['seed'],10000+source_index) if job['arm']=='shuffled_velocity_router' else v
            with torch.no_grad():tq=routing_q(teacher,torch.tensor(z),torch.tensor(tv)).numpy().astype(float)
            mass=tq.mean(0);mass/=mass.sum()
            x,sq=predict_local(student,z,np.zeros_like(v),(anchor-source_mean)/duration,duration)
            x=x.astype(float);sq=sq.astype(float);sq/=sq.sum(1,keepdims=True)
            calibrated,iterations=calibrate_rows(sq,mass)
            probabilities=(sq,calibrated,np.broadcast_to(mass,sq.shape).copy())
            means=[];covs=[];shifts=[]
            for p in probabilities:
                shift=anchor-(x*p[:,:,None]).sum(1).mean(0)
                mean,cov=moments((x+shift).reshape(-1,x.shape[-1]),p.reshape(-1)/len(p))
                shifts.append(shift);means.append(mean);covs.append(cov)
            label=f'{item["source_hour"]}_to_{item["target_hour"]}';indices=samples[label]
            path=run.directory/f'{job["arm"]}_seed{job["seed"]}_{label}.npz'
            np.savez_compressed(path,candidates=x[indices].astype('float32'),q=np.stack([p[indices] for p in probabilities]),
                shifts=np.stack(shifts),full_mean=np.stack(means),full_cov=np.stack(covs))
            frozen.append(dict(arm=job['arm'],seed=job['seed'],source_hour=item['source_hour'],target_hour=item['target_hour'],path=str(path.resolve()),sha256=sha256(path)))
            calibration.append(dict(arm=job['arm'],seed=job['seed'],source_hour=item['source_hour'],iterations=iterations,
                source_mass_KL=float(np.sum(mass*np.log(np.maximum(mass,1e-12)/np.maximum(sq.mean(0),1e-12)))),
                source_mass_L1=float(abs(mass-sq.mean(0)).sum()),
                calibrated_mass_max_error=float(abs(calibrated.mean(0)-mass).max())))
    run.json('prediction_manifest.json',dict(status='SOURCE_CALIBRATED_PREDICTIONS_FROZEN',artifacts=frozen,
        variants=VARIANTS,target_read_for_calibration=False,model_retrained=False))
    rows=[]
    for cache in manifest['evaluation_caches']:
        if sha256(cache['path'])!=cache['sha256']:raise ValueError('Evaluation cache changed')
        with np.load(cache['path'],allow_pickle=False) as s:truth={k:s[k].copy() for k in s.files}
        y=torch.tensor(truth['target'],dtype=torch.float64)
        for arm in manifest['config']['arms']:
            for index,variant in enumerate(VARIANTS):
                xs=[];ws=[]
                for record in [r for r in frozen if r['arm']==arm and r['source_hour']==cache['source_hour']]:
                    if sha256(record['path'])!=record['sha256']:raise ValueError('Frozen calibration changed')
                    with np.load(record['path'],allow_pickle=False) as s:
                        x=s['candidates'].astype(float)+s['shifts'][index];q=s['q'][index]
                        xs.append(x.reshape(-1,x.shape[-1]));ws.append(q.reshape(-1)/(len(q)*len(manifest['config']['seeds'])))
                with torch.no_grad():energy=float(blocked_energy(torch.tensor(np.concatenate(xs)),torch.tensor(np.concatenate(ws)),y,manifest['config']['energy_block_size']))
                rows.append(dict(teacher_arm=arm,variant=variant,source_hour=cache['source_hour'],target_hour=cache['target_hour'],energy_distance=energy))
    metrics=[dict(teacher_arm=arm,variant=variant,energy_distance=float(np.mean([r['energy_distance'] for r in rows if r['teacher_arm']==arm and r['variant']==variant])))
             for arm in manifest['config']['arms'] for variant in VARIANTS]
    write_csv(run.directory/'metrics.csv',metrics);write_csv(run.directory/'interval_metrics.csv',rows);write_csv(run.directory/'source_calibration.csv',calibration)
    run.json('summary.json',dict(status='SOURCE_MASS_CALIBRATION_COMPLETE',metrics=metrics,
        scope='same_series_development',target_read_for_calibration=False,learned_parameters_added=0,
        qualification='Teacher kinetic router is still needed to infer cohort masses; not U-free deployment'))
    run.json('status.json',dict(status='complete'));print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)
