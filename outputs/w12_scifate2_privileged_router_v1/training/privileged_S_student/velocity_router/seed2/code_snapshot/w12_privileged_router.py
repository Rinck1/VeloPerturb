"""Distill frozen dynamics-trained responsibilities into an S-only local router."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from w4_artifacts import Artifacts,sha256
from w6_local_router import LocalResponseRouter,predict_local
from w6_router_scifate2 import moments
from w7_router_scaling import atomic_json,score_case,training_path
from w9_large_training import read_training
from w9_population_router import make_training_state
from w10_recipe_model import model_arm,shuffle_velocity


ROOT=Path(__file__).resolve().parents[2]
CODE=[Path(__file__),*[Path(__file__).with_name(n) for n in ('w6_local_router.py','w6_router_scifate2.py',
    'w7_router_scaling.py','w9_large_training.py','w9_population_router.py','w10_recipe_model.py','w4_artifacts.py')]]


def routing_q(model,z,v):
    state=(z-model.z_mean)/model.z_scale
    kinetic=((v-model.v_mean)/model.v_scale).clamp(-5,5) if model.arm=='velocity_router' else torch.zeros_like(v)
    return model.router(torch.cat((state,kinetic),1)).softmax(-1)


def fit_student(teacher,groups,*,teacher_arm,seed,steps,device):
    """No target expression enters distillation; only teacher source-side soft labels."""
    student=copy.deepcopy(teacher);student.arm='static_router'
    for parameter in student.parameters():parameter.requires_grad_(False)
    for parameter in student.router.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(student.router.parameters(),lr=.001,weight_decay=.0001)
    tensors=[]
    with torch.no_grad():
        for index,group in enumerate(groups):
            velocity=group['velocity']
            if teacher_arm=='shuffled_velocity_router':velocity=shuffle_velocity(velocity,seed,index)
            z=torch.as_tensor(group['z'],dtype=torch.float32,device=device)
            v=torch.as_tensor(velocity,dtype=torch.float32,device=device)
            tensors.append((z,routing_q(teacher,z,v).detach()))
    generator=np.random.default_rng(seed);history=[];began=time.monotonic()
    for step in range(steps):
        z,target=tensors[step%len(tensors)]
        si=torch.as_tensor(generator.integers(len(z),size=512),device=device)
        q=routing_q(student,z[si],torch.zeros_like(z[si]))
        kl=(target[si]*(target[si].clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(1).mean()
        if not torch.isfinite(kl):raise ArithmeticError('Nonfinite privileged-router distillation')
        optimizer.zero_grad(set_to_none=True);kl.backward()
        torch.nn.utils.clip_grad_norm_(student.router.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==0 or (step+1)%1000==0 or step+1==steps:
            history.append(dict(step=step+1,train_source_KL=float(kl.detach())))
    student.eval()
    return student,optimizer,generator,history,time.monotonic()-began


def prepare(source,output):
    source=Path(source);old=json.loads((source/'study_manifest.json').read_text())
    teachers=[j for j in old['jobs'] if j['setting']=='b1_direct_distance']
    if len(teachers)!=9:raise ValueError('Need all three matched direct-distance teachers and seeds')
    cfg=dict(old['config'],steps=4000,checkpoint_steps=[4000],recipes=[dict(name='privileged_S_student')])
    jobs=[];cases=[]
    for teacher in teachers:
        path=training_path(source,teacher)/'model_step12000.pt'
        jobs.append(dict(setting='privileged_S_student',experts=teacher['experts'],hidden=teacher['hidden'],
            arm=teacher['arm'],seed=teacher['seed'],worker=teacher['worker'],
            config=dict(teacher['config'],steps=4000,checkpoint_steps=[4000],learning_rate=.001,source_batch=512,
                        training_objective='teacher_source_KL',student_routing_inputs='S_only'),
            expected_parameters=teacher['expected_parameters'],teacher=dict(path=str(path.resolve()),sha256=sha256(path))))
    for index,arm in enumerate(cfg['arms']):
        cases.append(dict(setting='privileged_S_student',experts=8,hidden=128,arm=arm,step=4000,worker=index))
    run=Artifacts(output,seed=cfg['seeds'],inputs=[source/'study_manifest.json',ROOT/'PLANS/w12_privileged_router.md']+CODE)
    samples=json.loads((source/'source_samples.json').read_text())
    run.json('source_samples.json',{k:v for k,v in samples.items() if '_to_' in k})
    run.json('study_manifest.json',dict(status='PRIVILEGED_DISTILLATION_READY',config=cfg,jobs=jobs,cases=cases,
        pack=old['pack'],pack_manifest_sha256=old['pack_manifest_sha256'],evaluation_caches=old['evaluation_caches'],
        code_hashes={str(p):sha256(p) for p in CODE},source_study=str(source),r4_used_for_optimizer=False,
        student_routing_inputs='S_only',shared_frozen_mean_inputs='S_and_analytic_dynamics',
        arm_names_refer_to_teacher_not_student=True,scope='same_series_development'))
    run.json('status.json',dict(status='complete'))


def train_job(root,manifest,job,device):
    directory=training_path(root,job)
    if (directory/'prediction_manifest.json').exists():return
    if directory.exists():raise ValueError('Keep interrupted student outputs for explicit recovery')
    pm,groups,norm,index=read_training(root,manifest,job);pack=Path(manifest['pack']);cp=job['teacher']
    if sha256(cp['path'])!=cp['sha256']:raise ValueError('Frozen teacher changed')
    run=Artifacts(directory,seed=job['seed'],inputs=[root/'study_manifest.json',pack/'manifest.json',pack/'normalization.npz',
        Path(cp['path'])]+[Path(p['path']) for p in pm['training_groups']+pm['prediction_groups']]+CODE)
    teacher=LocalResponseRouter(**norm,config=job['config'],arm=model_arm(job['arm']),seed=job['seed']).to(device)
    teacher.load_state_dict(torch.load(cp['path'],map_location=device,weights_only=True)['state_dict']);teacher.eval()
    student,optimizer,generator,history,seconds=fit_student(teacher,groups,teacher_arm=job['arm'],seed=job['seed'],
        steps=manifest['config']['steps'],device=device)
    step=manifest['config']['steps']
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in student.state_dict().items()},config=job['config'],
        arm='static_router',teacher_arm=job['arm'],seed=job['seed'],step=step),run.directory/f'model_step{step}.pt')
    torch.save(dict(training_state=make_training_state(student,optimizer,generator,step,history),job=job),run.directory/'resume_state.pt')
    samples=json.loads((root/'source_samples.json').read_text());artifacts=[];source_KL=[]
    for source_index,item in enumerate(pm['prediction_groups']):
        if sha256(item['path'])!=item['sha256']:raise ValueError('Source input changed')
        with np.load(item['path'],allow_pickle=False) as saved:
            z,v,duration=saved['z'],saved['velocity'],float(saved['duration'])
            anchor,source_mean=saved['anchors'][index],saved['source_mean']
        x,q=predict_local(student,z,np.zeros_like(v),(anchor-source_mean)/duration,duration)
        tv=shuffle_velocity(v,job['seed'],10000+source_index) if job['arm']=='shuffled_velocity_router' else v
        with torch.no_grad():
            zz,vv=[torch.as_tensor(a,dtype=torch.float32,device=device) for a in (z,tv)]
            tq=routing_q(teacher,zz,vv);sq=torch.as_tensor(q,device=device)
            source_KL.append(float((tq*(tq.clamp_min(1e-12).log()-sq.clamp_min(1e-12).log())).sum(1).mean()))
        x,q=x.astype(float),q.astype(float);q/=q.sum(1,keepdims=True)
        correction=anchor-(x*q[:,:,None]).sum(1).mean(0)
        if np.linalg.norm(correction)>1e-3:raise ArithmeticError('Mean correction exceeds roundoff')
        x=(x+correction).astype('float32');mean,cov=moments(x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(q))
        usage=q.mean(0);entropy=float(-(q*np.log(np.maximum(q,1e-12))).sum(1).mean())
        label=f'{item["source_hour"]}_to_{item["target_hour"]}';path=run.directory/f'step{step}_{label}.npz';si=samples[label]
        np.savez_compressed(path,candidates=x[si],q=q[si],full_mean=mean,full_cov=cov,anchor=anchor,
            usage=usage,route_entropy=entropy,source_cells=len(q),active_modes=int((usage>=.01).sum()),
            effective_modes=float(np.exp(-(usage*np.log(np.maximum(usage,1e-12))).sum())),
            routing_information=float(-(usage*np.log(np.maximum(usage,1e-12))).sum()-entropy))
        artifacts.append(dict(step=step,source_hour=item['source_hour'],target_hour=item['target_hour'],path=str(path.resolve()),sha256=sha256(path)))
    parameters=sum(p.numel() for p in student.parameters())
    run.json('training.json',dict(history=history,student_route_trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
        prediction_source_teacher_student_KL=source_KL,student_routing_inputs='S_only'))
    run.json('prediction_manifest.json',dict(status='STUDENT_PREDICTIONS_READY',job=job,artifacts=artifacts,
        parameters=parameters,checkpoint_timing=[dict(step=step,training_seconds=seconds,
        peak_allocated_bytes=0 if device=='cpu' else torch.cuda.max_memory_allocated())],
        r4_used_for_optimizer=False,student_routing_inputs='S_only',global_mean_frozen=True))
    run.json('status.json',dict(status='complete'))
    print(json.dumps(dict(status='STUDENT_READY',teacher_arm=job['arm'],seed=job['seed'],seconds=seconds)),flush=True)


def worker(root,index,device):
    from veloroute.gpu_policy import enforce_gpu_policy
    from w10_recipe_bridge import check_peer_failures
    root=Path(root);m=json.loads((root/'study_manifest.json').read_text());enforce_gpu_policy(device)
    try:
        for path,digest in m['code_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Student code changed after design freeze')
        for i,job in enumerate(j for j in m['jobs'] if j['worker']==index):
            atomic_json(root/f'worker{index}.json',dict(status='training',completed_jobs=i,job=job));train_job(root,m,job,device)
        atomic_json(root/f'training_worker{index}_complete.json',dict(status='complete'))
        deadline=time.monotonic()+3600
        while not all((root/f'training_worker{i}_complete.json').exists() for i in range(m['config']['workers'])):
            check_peer_failures(root,m)
            if time.monotonic()>deadline:raise TimeoutError('Student grid timeout')
            time.sleep(5)
        for case in (c for c in m['cases'] if c['worker']==index):score_case(root,m,case,device)
        atomic_json(root/f'worker{index}_complete.json',dict(status='complete'));atomic_json(root/f'worker{index}.json',dict(status='complete'))
    except Exception as error:
        atomic_json(root/f'worker{index}.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--root',required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p=sub.add_parser('aggregate');p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.source,args.output)
    elif args.stage=='worker':worker(args.root,args.worker,args.device)
    else:
        from w10_recipe_bridge import wait_for_workers,aggregate
        root=Path(args.root)
        if args.wait:wait_for_workers(root)
        aggregate(root)
