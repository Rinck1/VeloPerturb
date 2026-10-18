"""Frozen W14 81-job cross-replicate grid, source-only predictions, then scoring."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from w4_artifacts import Artifacts,sha256
from w5_joint_kinetics import fit_task_model,predict_delta
from w7_router_scaling import atomic_json,write_csv
from w14_data import CountReader,SourceTransform,load_npz,prepare_fold
from w14_model import ARMS,fit,source_predictions


ROOT=Path(__file__).resolve().parents[2]
CODE=[Path(__file__).with_name(n) for n in ('w14_loro.py','w14_model.py','w14_data.py',
    'w4_artifacts.py','w4_shorttime.py','w4_streams.py','w4_arms.py','w4_kinetic_head.py',
    'w5_joint_kinetics.py','w5_joint_scifate2.py','w5_scntseq_repair.py','w6_local_router.py',
    'w7_router_scaling.py','w9_population_router.py','w10_recipe_model.py','w14_program.py','w14_synthetic.py')]
PROTOCOLS=[ROOT/'PLANS/w14_confirmation_loro.md',ROOT/'PLANS/w14_execution_amendment_20261018.md']


def jobs(config):
    result=[]
    for fi,rep in enumerate(config['replicates']):
        for seed in config['seeds']:
            for arm in ('static','real','shuffled_v','direct_U','nonlinear_SU'):
                for ss in config['shuffle_seeds'] if arm=='shuffled_v' else [0]:
                    result.append(dict(heldout=rep,worker=fi,seed=seed,arm=arm,shuffle_seed=ss,
                        name=f'{arm}_perm{ss}' if arm=='shuffled_v' else arm))
    return result


def freeze(config_path,output,audit_xml,synthetic):
    config_path=Path(config_path);c=json.loads(config_path.read_text())
    if c['centering']!='batch' or c['estimator']!='V' or c['distance_kernel']!='direct':
        raise ValueError('W14 must retain b1 objective')
    suites=ET.parse(audit_xml).getroot();suites=[suites] if suites.tag=='testsuite' else list(suites)
    if not suites or not sum(int(s.get('tests',0)) for s in suites) or any(
        int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites):
        raise ValueError('Passing implementation tests required')
    syn=json.loads(Path(synthetic).read_text())
    if syn['status']!='W14_SYNTHETIC_IMPLEMENTATION_CHECK_PASSED':raise ValueError('Synthetic implementation failed')
    data=ROOT/c['data_path']
    if sha256(data)!=c['data_sha256']:raise ValueError('Raw S/U matrix changed')
    inputs=[config_path,ROOT/c['mean_config'],Path(audit_xml),Path(synthetic)]+PROTOCOLS+CODE
    run=Artifacts(output,seed=c['seeds'],inputs=inputs)
    manifest=dict(config=c,jobs=jobs(c),freeze_commit=run.context['git_commit'],scope='internal_fixed_recipe_LORO_24h',
        frozen_sha256={str(p.resolve()):sha256(p) for p in [config_path,ROOT/c['mean_config']]+PROTOCOLS+CODE},
        data_sha256=c['data_sha256'],tests_xml_sha256=sha256(audit_xml),synthetic_sha256=sha256(synthetic),
        heldout_scores_read_before_freeze=False,primary_checkpoint=36000)
    run.json('study_manifest.json',manifest);run.json('status.json',dict(status='ready',jobs=len(manifest['jobs'])))
    print(json.dumps(dict(status='W14_FROZEN',jobs=len(manifest['jobs']),freeze_commit=manifest['freeze_commit'])),flush=True)


def verify_lock(root):
    m=json.loads((Path(root)/'study_manifest.json').read_text())
    for p,h in m['frozen_sha256'].items():
        if sha256(p)!=h:raise ValueError(f'Frozen code/config changed: {p}')
    return m


def fit_means(root,fold,m,seed,device):
    output=fold/'means'/f'seed{seed}'
    if (output/'complete.json').exists():
        done=json.loads((output/'complete.json').read_text())
        if sha256(output/'rates.npz')!=done['rates_sha256']:raise ValueError('Mean rates changed')
        return load_npz(output/'rates.npz')
    run=Artifacts(output,seed=seed,inputs=[root/'study_manifest.json',fold/'mean_training.npz']+CODE)
    a=load_npz(fold/'mean_training.npz');mc=json.loads((ROOT/m['config']['mean_config']).read_text())
    fm=json.loads((fold/'manifest.json').read_text());source=[load_npz(g['path']) for g in fm['source_groups']]
    models={};histories={}
    for arm in ('analytic_task','static_task'):
        model,history=fit_task_model(a['base'],a['velocity'],a['delta'],
            {k:a['bank_'+k] for k in ('base','velocity','delta')},mc,
            controls=np.zeros(len(a['base']),dtype=bool),arm=arm,seed=seed,device=device)
        models[arm]=model;histories[arm]=history
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm),output/(arm+'.pt'))
    rates=dict(training=predict_delta(models['analytic_task'],a['base'],a['velocity']))
    for arm in models:
        rates[arm]=predict_delta(models[arm],np.stack([s['base'] for s in source]),np.stack([s['population_velocity'] for s in source]))
    np.savez_compressed(output/'rates.npz',**rates)
    run.json('training.json',histories);run.json('complete.json',dict(rates_sha256=sha256(output/'rates.npz')))
    return rates


def job_directory(root,job):
    return Path(root)/'training'/job['heldout']/f'seed{job["seed"]}'/job['name']


def train_job(root,m,fm,job,device):
    output=job_directory(root,job)
    if (output/'prediction_manifest.json').exists():return
    c=m['config'];fold=root/'folds'/job['heldout'];rates=fit_means(root,fold,m,job['seed'],device)
    run=Artifacts(output,seed=job['seed'],inputs=[root/'study_manifest.json',fold/'manifest.json',
        fold/'normalization.npz',fold/'means'/f'seed{job["seed"]}'/'rates.npz']+CODE)
    groups=[]
    for i,g in enumerate(fm['training_groups']):
        if sha256(g['path'])!=g['sha256']:raise ValueError('Training group changed')
        groups.append(dict(load_npz(g['path']),global_rate=rates['training'][i]))
    def checkpoint(step,model,optimizer,rng,history):
        path=output/f'model_step{step}.pt'
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,job=job,config=c),path)
        tmp=output/'resume_state.pt.tmp'
        torch.save(dict(step=step,job=job,model=model.state_dict(),optimizer=optimizer.state_dict(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history),tmp)
        tmp.replace(output/'resume_state.pt')
    model,history=fit(groups,load_npz(fold/'normalization.npz'),c,arm=job['arm'],seed=job['seed'],
        shuffle_seed=job['shuffle_seed'],device=device,checkpoint=checkpoint,
        progress=lambda r:run.json('status.json',dict(status='training',job=job,**r)))
    artifacts=[]
    for i,g in enumerate(fm['source_groups']):
        if sha256(g['path'])!=g['sha256']:raise ValueError('Source group changed')
        source=load_npz(g['path']);source['global_rate']=rates['analytic_task'][i]
        result=source_predictions(model,source,job['arm'],job['seed'],job['shuffle_seed'],i,c['calibration_hours'])
        si=source['score_indices']
        for k in ('candidates','q','cohort_candidates','cohort_q'):result[k]=result[k][si]
        result.update(source_ids=source['cell_ids'][si],source_indices=si,
            static_mean_anchor=source['z'].astype(float).mean(0)+c['calibration_hours']*rates['static_task'][i])
        path=output/f'prediction_D{g["source_day"]}.npz';np.savez_compressed(path,**result)
        artifacts.append(dict(source_day=g['source_day'],target_day=g['target_day'],path=str(path.resolve()),sha256=sha256(path)))
    run.json('training.json',history)
    run.json('prediction_manifest.json',dict(job=job,artifacts=artifacts,checkpoint_step=c['steps'],
        model_sha256=sha256(output/f'model_step{c["steps"]}.pt'),heldout_future_used=False,
        all_arm_shared_mean=True,primary_checkpoint_only=True))
    run.json('status.json',dict(status='complete',job=job))
    print(json.dumps(dict(status='W14_JOB_COMPLETE',job=job,parameters=history['parameters'])),flush=True)


def worker(root,index,device):
    root=Path(root);m=verify_lock(root)
    if device!='cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    rep=m['config']['replicates'][index]
    try:
        atomic_json(root/f'worker{index}.json',dict(status='preparing_fold',heldout=rep))
        fm=prepare_fold(m['config'],rep,root/'folds'/rep)
        for job in [j for j in m['jobs'] if j['worker']==index]:
            atomic_json(root/f'worker{index}.json',dict(status='training',job=job));train_job(root,m,fm,job,device)
        atomic_json(root/f'worker{index}.json',dict(status='complete',heldout=rep))
    except Exception as error:
        atomic_json(root/f'worker{index}.json',dict(status='failed',heldout=rep,error=repr(error)));raise


def freeze_grid(root,m):
    """Validate EVERY prediction before a held-out scoring target is requested."""
    frozen=[]
    for job in m['jobs']:
        directory=job_directory(root,job);pm=json.loads((directory/'prediction_manifest.json').read_text())
        if pm['job']!=job or pm['heldout_future_used'] or pm['checkpoint_step']!=m['primary_checkpoint']:
            raise ValueError('Prediction grid contract differs')
        if sha256(directory/f'model_step{m["primary_checkpoint"]}.pt')!=pm['model_sha256']:
            raise ValueError('Deployment checkpoint changed')
        if [r['source_day'] for r in pm['artifacts']]!=m['config']['training_days'][:-1]:
            raise ValueError('Prediction intervals incomplete')
        for row in pm['artifacts']:
            if sha256(row['path'])!=row['sha256']:raise ValueError('Prediction artifact changed')
            frozen.append(dict(job=job,**row))
    return frozen


def energy(x,w,y,device='cpu',block=512):
    x,y,w=[torch.as_tensor(a,dtype=torch.float64,device=device) for a in (x,y,w)]
    w=w/w.sum()
    def distances(a,b):return torch.cdist(a,b,compute_mode='donot_use_mm_for_euclid_dist')
    cross=x.new_zeros(());within=x.new_zeros(());target=x.new_zeros(())
    for i in range(0,len(x),block):
        cross+=w[i:i+block]@distances(x[i:i+block],y).mean(1)
        for j in range(0,len(x),block):
            within+=w[i:i+block]@distances(x[i:i+block],x[j:j+block])@w[j:j+block]
    for i in range(0,len(y),block):target+=distances(y[i:i+block],y).sum()/(len(y)**2)
    return float((2*cross-within-target).cpu())/math.sqrt(x.shape[1])


def evaluate(root,output,device):
    root=Path(root);m=verify_lock(root);c=m['config'];frozen=freeze_grid(root,m)
    if device!='cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    run=Artifacts(output,seed=c['seeds'],inputs=[root/'study_manifest.json']+CODE)
    run.json('frozen_grid.json',dict(predictions=frozen,total=len(frozen),targets_scored_before_freeze=False))
    reader=CountReader(ROOT/c['data_path'],c);rows=[]
    names=['static','real','direct_U','nonlinear_SU']+[f'shuffled_v_perm{i}' for i in c['shuffle_seeds']]
    for rep in c['replicates']:
        fold=root/'folds'/rep;transform=SourceTransform.load(fold/'source_transform.npz')
        fm=json.loads((fold/'manifest.json').read_text())
        if sha256(fold/'source_transform.npz')!=fm['transform_sha256']:raise ValueError('Fold transform changed')
        for day in c['training_days'][:-1]:
            target=reader.load(rep,day+1);y=transform.state(target['s'])
            ti=np.random.default_rng(c['evaluation_seed']).choice(len(y),min(len(y),c['evaluation_target_cells']),replace=False)
            y=y[ti]
            for name in names:
                selected=[r for r in frozen if r['job']['heldout']==rep and r['source_day']==day and r['job']['name']==name]
                if len(selected)!=len(c['seeds']):raise ValueError('Seed mixture incomplete')
                packs=[load_npz(r['path']) for r in selected]
                for p in packs[1:]:np.testing.assert_array_equal(packs[0]['source_ids'],p['source_ids'])
                x=np.concatenate([p['candidates'].reshape(-1,c['dimensions']) for p in packs])
                w=np.concatenate([p['q'].ravel()/len(p['q'])/len(packs) for p in packs])
                e=energy(x,w,y,device)
                row=dict(heldout=rep,source_day=day,target_day=day+1,arm=name,energy=e,
                    source_cells=len(packs[0]['q']),target_cells=len(y),cohort_average_energy=None,
                    mean_mse=float(((np.mean([p['anchor'] for p in packs],0)-transform.state(target['s']).mean(0))**2).mean()),
                    static_mean_mse=float(((np.mean([p['static_mean_anchor'] for p in packs],0)-transform.state(target['s']).mean(0))**2).mean()))
                if name=='real':
                    ax=np.concatenate([p['cohort_candidates'].reshape(-1,c['dimensions']) for p in packs])
                    aw=np.concatenate([p['cohort_q'].ravel()/len(p['cohort_q'])/len(packs) for p in packs])
                    row['cohort_average_energy']=energy(ax,aw,y,device)
                rows.append(row)
                run.json('status.json',dict(status='evaluating',heldout=rep,source_day=day,arm=name,completed=len(rows)))
            run.json('metrics_so_far.json',dict(rows=rows))
    comparisons=[]
    for rep in c['replicates']:
        for control in [n for n in names if n!='real']:
            gains=[]
            for day in c['training_days'][:-1]:
                r={q['arm']:q for q in rows if q['heldout']==rep and q['source_day']==day}
                gains.append(1-r['real']['energy']/r[control]['energy'])
            comparisons.append(dict(heldout=rep,control=control,mean_gain=float(np.mean(gains))))
    static=[r['mean_gain'] for r in comparisons if r['control']=='static']
    p1=bool(np.mean(static)>0 and sum(v>0 for v in static)>=2)
    global_energy={name:float(np.mean([r['energy'] for r in rows if r['arm']==name])) for name in names}
    shuffled=[n for n in names if n.startswith('shuffled')]
    per_fold_shuffle=[]
    for rep in c['replicates']:
        per_fold_shuffle.append(np.mean([r['mean_gain'] for r in comparisons if r['heldout']==rep and r['control'] in shuffled]))
    p2=bool(all(global_energy['real']<global_energy[n] for n in shuffled) and sum(v>0 for v in per_fold_shuffle)>=2)
    result=dict(status='W14_GRID_EVALUATED',P1=p1,P2=p2,
        classification='DIRECTIONALLY_CONFIRMED' if p1 and p2 else 'PARTIALLY_CONFIRMED' if p1 or p2 else 'NOT_CONFIRMED',
        overall_gain_vs_static=float(np.mean(static)),global_energy=global_energy,comparisons=comparisons,
        scope='internal_fixed_recipe_cross_replicate_24h_not_new_short_horizon_replication',
        biological_significance_claim=False,independent_dataset=False,total_training_jobs=len(m['jobs']))
    write_csv(run.directory/'metrics.csv',rows);write_csv(run.directory/'comparisons.csv',comparisons)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    report=f'# W14 cross-replicate results\n\nP1: {p1}\n\nP2: {p2}\n\nClassification: {result["classification"]}\n\n'
    report+='24-hour internal validation; not a verdict on the 5–15h development claim.\n\n'+json.dumps(result,indent=2)+'\n'
    (run.directory/'RESULTS.md').write_text(report)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','worker','evaluate'])
    p.add_argument('--root',default='outputs/w14_loro_v1');p.add_argument('--config',default='configs/w14_loro_v1.json')
    p.add_argument('--worker',type=int,choices=[0,1,2]);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output');p.add_argument('--audit-xml');p.add_argument('--synthetic');a=p.parse_args()
    if a.action=='freeze':freeze(a.config,a.root,a.audit_xml,a.synthetic)
    elif a.action=='worker':worker(a.root,a.worker,a.device)
    else:evaluate(a.root,a.output,a.device)
