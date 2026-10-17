"""Source-cohort response mixtures: pool joint state/dynamics votes, not fates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from w4_artifacts import Artifacts,sha256
from w6_local_router import LocalResponseRouter,predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import atomic_json,score_case,training_path,write_csv
from w9_large_training import read_training
from w9_population_router import make_training_state
from w10_recipe_model import bridge_energy,model_arm,shuffle_velocity


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__),*[Path(__file__).with_name(n) for n in (
    'w10_recipe_model.py','w10_recipe_bridge.py','w9_population_router.py','w9_large_training.py','w6_local_router.py',
    'w7_router_scaling.py','w6_router_scifate2.py','w4_artifacts.py')]]


class CohortResponseRouter(LocalResponseRouter):
    """Average source-cell response votes BEFORE assigning mixture weights.

    q_cohort = mean_i softmax(router(z_i,v_i)); every source shares q_cohort.
    Joint cell permutations are invariant; independently permuting v can change
    the joint source-population fingerprint. Experts still consume state only.
    """
    def forward(self,z,velocity,global_rate,duration):
        state = (z-self.z_mean)/self.z_scale
        kinetic = ((velocity-self.v_mean)/self.v_scale).clamp(-5,5) if self.arm=='velocity_router' else torch.zeros_like(velocity)
        votes = self.router(torch.cat((state,kinetic),dim=1)).softmax(-1)
        q = votes.mean(0,keepdim=True).expand(len(z),-1)
        h = self.backbone(state)
        raw = torch.stack([expert(h) for expert in self.experts],dim=1).tanh()*self.limit
        rates = raw*self.z_scale/self.calibration_hours
        mean_rate = (q[:,:,None]*rates).sum(1).mean(0)
        return z[:,None,:]+duration*(global_rate[None,None,:]+rates-mean_rate[None,None,:]),q,raw


def fit(groups,normalization,config,*,arm,seed,device,checkpoint=None,progress=None):
    model = CohortResponseRouter(**normalization,config=config,arm=model_arm(arm),seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    tensors = []
    for index,group in enumerate(groups):
        values = dict(group)
        if arm=='shuffled_velocity_router':values['velocity']=shuffle_velocity(group['velocity'],seed,index)
        tensors.append({k:torch.as_tensor(values[k],dtype=torch.float32,device=device) for k in ('z','velocity','target','global_rate')})
    generator = np.random.default_rng(seed);history = [];began = time.monotonic()
    for step in range(config['steps']):
        index = step%len(tensors);group = tensors[index]
        si = torch.as_tensor(generator.integers(len(group['z']),size=config['source_batch']),device=device)
        ti = torch.as_tensor(generator.integers(len(group['target']),size=config['target_batch']),device=device)
        full,q,raw = model(group['z'],group['velocity'],group['global_rate'],config['calibration_hours'])
        energy = bridge_energy(full[si],q[si],group['target'][ti],config)/math.sqrt(full.shape[-1])
        # Cell-wise routing mutual information is identically zero for cohort weights.
        loss = energy+config['residual_regularization']*raw[si].square().mean()
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite cohort-router loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==0 or (step+1)%300==0 or step+1==config['steps']:
            row = dict(step=step+1,group_index=index,energy=float(energy.detach()),loss=float(loss.detach()),
                gradient_norm=float(gradient),mode_usage=q[0].detach().cpu().tolist(),elapsed_seconds=time.monotonic()-began)
            history.append(row)
            if progress:progress(row)
        if checkpoint and step+1 in config['checkpoint_steps']:checkpoint(step+1,model,optimizer,generator,history)
    model.eval();return model,history


def prepare(source,output):
    source = Path(source);old = json.loads((source/'study_manifest.json').read_text())
    reference = next(j for j in old['jobs'] if j['setting']=='b2_full_center')
    cfg = dict(old['config']);cfg['recipes']=[dict(name='cohort_full_center_V')]
    jobs=[];cases=[]
    for arm in cfg['arms']:
        for worker,seed in enumerate(cfg['seeds']):
            config = dict(reference['config'],routing_scope='cohort',routing_regularization=0.)
            jobs.append(dict(setting='cohort_full_center_V',experts=config['experts'],hidden=config['hidden'],
                arm=arm,seed=seed,worker=worker,config=config,expected_parameters=reference['expected_parameters']))
        for step in cfg['checkpoint_steps']:
            cases.append(dict(setting='cohort_full_center_V',experts=reference['experts'],hidden=reference['hidden'],
                arm=arm,step=step,worker=len(cases)%cfg['workers']))
    run = Artifacts(output,seed=cfg['seeds'],inputs=[source/'study_manifest.json',ROOT/'PLANS/w11_cohort_router.md']+CODE)
    samples = json.loads((source/'source_samples.json').read_text())
    run.json('source_samples.json',{k:v for k,v in samples.items() if '_to_' in k})
    run.json('study_manifest.json',dict(status='COHORT_ROUTER_READY',config=cfg,jobs=jobs,cases=cases,
        pack=old['pack'],pack_manifest_sha256=old['pack_manifest_sha256'],evaluation_caches=old['evaluation_caches'],
        code_hashes={str(p):sha256(p) for p in CODE},source_bridge=str(source),scope='same_series_development',
        r4_used_for_optimizer=False))
    run.json('status.json',dict(status='complete',jobs=len(jobs),cases=len(cases)))


def train_job(root,manifest,job,device):
    directory = training_path(root,job)
    if (directory/'prediction_manifest.json').exists():return
    if directory.exists():raise ValueError('Preserve interrupted cohort run; use explicit recovery')
    pm,groups,norm,index = read_training(root,manifest,job);pack = Path(manifest['pack'])
    run = Artifacts(directory,seed=job['seed'],inputs=[root/'study_manifest.json',pack/'manifest.json',
        pack/'normalization.npz']+[Path(p['path']) for p in pm['training_groups']+pm['prediction_groups']]+CODE)
    samples = json.loads((root/'source_samples.json').read_text());artifacts=[];timing=[]
    began = time.monotonic();paused = 0.
    torch.cuda.reset_peak_memory_stats() if device!='cpu' else None
    def checkpoint(step,model,optimizer,generator,history):
        nonlocal paused
        if device!='cpu':torch.cuda.synchronize()
        started = time.monotonic()
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            config=job['config'],arm=job['arm'],seed=job['seed'],step=step),run.directory/f'model_step{step}.pt')
        temporary=run.directory/'resume_state.pt.tmp'
        torch.save(dict(training_state=make_training_state(model,optimizer,generator,step,history),job=job),temporary)
        temporary.replace(run.directory/'resume_state.pt')
        for source_index,item in enumerate(pm['prediction_groups']):
            if sha256(item['path'])!=item['sha256']:raise ValueError('Source features changed')
            with np.load(item['path'],allow_pickle=False) as saved:
                z,v,duration=saved['z'],saved['velocity'],float(saved['duration'])
                anchor,source_mean=saved['anchors'][index],saved['source_mean']
            if job['arm']=='shuffled_velocity_router':v=shuffle_velocity(v,job['seed'],10000+source_index)
            x,q=predict_local(model,z,v,(anchor-source_mean)/duration,duration)
            x,q=x.astype(float),q.astype(float);q/=q.sum(1,keepdims=True)
            shift=anchor-(x*q[:,:,None]).sum(1).mean(0)
            if np.linalg.norm(shift)>1e-3:raise ArithmeticError('Mean correction exceeds roundoff')
            x=(x+shift).astype('float32');mean,cov=moments(x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(q))
            usage=q.mean(0);entropy=float(-(usage*np.log(np.maximum(usage,1e-12))).sum())
            label=f'{item["source_hour"]}_to_{item["target_hour"]}';path=run.directory/f'step{step}_{label}.npz';si=samples[label]
            np.savez_compressed(path,candidates=x[si],q=q[si],full_mean=mean,full_cov=cov,anchor=anchor,usage=usage,
                route_entropy=entropy,source_cells=len(q),active_modes=int((usage>=.01).sum()),
                effective_modes=float(np.exp(entropy)),routing_information=0.)
            artifacts.append(dict(step=step,source_hour=item['source_hour'],target_hour=item['target_hour'],
                path=str(path.resolve()),sha256=sha256(path)))
        timing.append(dict(step=step,training_seconds=started-began-paused,
            peak_allocated_bytes=0 if device=='cpu' else torch.cuda.max_memory_allocated()))
        paused+=time.monotonic()-started
        print(json.dumps(dict(status='COHORT_CHECKPOINT',arm=job['arm'],seed=job['seed'],**timing[-1])),flush=True)
    def progress(row):run.json('status.json',dict(status='training',**row))
    model,history=fit(groups,norm,job['config'],arm=job['arm'],seed=job['seed'],device=device,checkpoint=checkpoint,progress=progress)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=job['expected_parameters']:raise ValueError('Cohort parameter count changed')
    run.json('training.json',dict(history=history,parameters=parameters,cohort_routing=True))
    run.json('prediction_manifest.json',dict(status='COHORT_PREDICTIONS_READY',job=job,artifacts=artifacts,
        parameters=parameters,checkpoint_timing=timing,global_mean_frozen=True,r4_used_for_optimizer=False))
    run.json('status.json',dict(status='complete'))


def worker(root,index,device):
    root=Path(root);manifest=json.loads((root/'study_manifest.json').read_text())
    from veloroute.gpu_policy import enforce_gpu_policy
    from w10_recipe_bridge import check_peer_failures
    enforce_gpu_policy(device)
    try:
        for path,digest in manifest['code_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Cohort code changed after design freeze')
        jobs=[j for j in manifest['jobs'] if j['worker']==index]
        for i,job in enumerate(jobs):
            check_peer_failures(root,manifest)
            atomic_json(root/f'worker{index}.json',dict(status='training',completed_jobs=i,job=job))
            train_job(root,manifest,job,device)
        atomic_json(root/f'training_worker{index}_complete.json',dict(status='complete'))
        deadline=time.monotonic()+3600
        while not all((root/f'training_worker{i}_complete.json').exists() for i in range(manifest['config']['workers'])):
            check_peer_failures(root,manifest)
            if time.monotonic()>deadline:raise TimeoutError('Cohort study exceeded one hour')
            time.sleep(10)
        for case in [c for c in manifest['cases'] if c['worker']==index]:score_case(root,manifest,case,device)
        atomic_json(root/f'worker{index}_complete.json',dict(status='complete'))
        atomic_json(root/f'worker{index}.json',dict(status='complete',completed_jobs=len(jobs)))
    except Exception as error:
        atomic_json(root/f'worker{index}.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--root',required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p=sub.add_parser('aggregate');p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.source,args.output)
    elif args.stage=='worker':worker(args.root,args.worker,args.device)
    else:
        from w10_recipe_bridge import aggregate,wait_for_workers
        root=Path(args.root)
        if args.wait:wait_for_workers(root)
        aggregate(root)
