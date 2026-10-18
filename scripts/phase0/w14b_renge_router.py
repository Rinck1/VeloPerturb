"""One prespecified exploratory RENGE transfer; queue after W14, never retune."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from w4_artifacts import Artifacts,sha256
from w4_arms import design,fit_ridge_training_cv
from w4_shorttime import condition_embeddings,read_count_rows,read_metadata
from w4_streams import SourceTransform,group_mean
from w7_router_scaling import atomic_json,write_csv
from w14_data import load_npz,source_features
from w14_loro import energy
from w14_model import fit,source_predictions


ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'PLANS/w14b_renge_router_protocol.md'
CODE=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('w14_model.py','w14_data.py','w14_loro.py',
    'w6_local_router.py','w10_recipe_model.py','w4_streams.py','w4_shorttime.py','w4_arms.py','w4_artifacts.py')]


def mean_features(transform,s,u,groups,labels):
    pooled_s=np.vstack([np.asarray(s[groups==label].sum(0)) for label in labels])
    pooled_u=np.vstack([np.asarray(u[groups==label].sum(0)) for label in labels])
    sn,un=transform.normalized(pooled_s,pooled_u)
    return dict(state=group_mean(transform.state(s),groups,labels),
        vel_pool=transform.projected_rate(sn,un-transform.gamma*sn))


def mean_anchors(train,test,target,et,ev,controls):
    x,xe=design(train,et,'vel_pool'),design(test,ev,'vel_pool')
    scale,ridge,selection=fit_ridge_training_cv(x,target-train['state'],controls=controls)
    return (train['state']+ridge.predict(scale.transform(x)),test['state']+ridge.predict(scale.transform(xe)),selection)


def prepare(output):
    output=Path(output);oldroot=ROOT/'outputs/w4_h1_predictions_v1'
    lockpath=ROOT/'outputs/w4_protocol_lock_v1/protocol_lock.json';lock=json.loads(lockpath.read_text())
    grid=json.loads((oldroot/'grid_manifest.json').read_text())
    for name,item in lock['data_hashes'].items():
        if sha256(item['path'])!=item['sha256']:raise ValueError(f'W4 input changed: {name}')
    c=json.loads((ROOT/'configs/w14_loro_v1.json').read_text())
    c=dict(c,steps=12000,checkpoint_steps=[12000],shuffle_seeds=[0,1,2],primary_checkpoint=12000)
    run=Artifacts(output,seed=c['seeds'],inputs=[PROTOCOL,ROOT/'configs/w14_loro_v1.json',lockpath,oldroot/'grid_manifest.json']+CODE)
    metadata={day:read_metadata(lock['data_hashes'][f'day{day}']['path'])[0] for day in (2,3,4)}
    records=[];jobs=[]
    for index,record in enumerate(grid['records']):
        old=oldroot/record['directory'];fm=json.loads((old/'fold_manifest.json').read_text())
        if sha256(old/'fold_manifest.json')!=record['fold_manifest_sha256']:raise ValueError('W4 fold manifest changed')
        transform_path=old/'transform_seed0.npz'
        if sha256(transform_path)!=fm['transform_sha256']['0']:raise ValueError('W4 fold geometry changed')
        transform=SourceTransform.load(transform_path);day,stop=fm['source_day'],fm['target_day'];tf=fm['heldout_TF']
        source_path=lock['data_hashes'][f'day{day}']['path'];target_path=lock['data_hashes'][f'day{stop}']['path']
        si,vi,ti=[np.array(fm[k],dtype=int) for k in ('training_source_rows','heldout_source_rows','training_target_rows')]
        s,u=[read_count_rows(source_path,k,si) for k in ('spliced','unspliced')]
        sv,uv=[read_count_rows(source_path,k,vi) for k in ('spliced','unspliced')]
        future=read_count_rows(target_path,'spliced',ti)
        labels_s=metadata[day]['condition'].astype(str).to_numpy()[si]
        labels_t=metadata[stop]['condition'].astype(str).to_numpy()[ti]
        labels=sorted(fm['training_TFs']+fm['controls'])
        if tf in labels or tf in labels_t:raise ValueError('Held-out TF entered training')
        train=mean_features(transform,s,u,labels_s,labels)
        test=mean_features(transform,sv,uv,np.array([tf]*len(vi)),[tf])
        target_means=group_mean(transform.state(future),labels_t,labels)
        et=condition_embeddings(lock['data_hashes']['embeddings']['path'],labels)
        ev=condition_embeddings(lock['data_hashes']['embeddings']['path'],[tf])
        anchors,anchor,selection=mean_anchors(train,test,target_means,et,ev,np.isin(labels,fm['controls']))
        oldpred=load_npz(old/'predictions.npz')['vel_pool'][0]
        np.testing.assert_allclose(anchor,oldpred,atol=1e-5,rtol=0)
        name=f'day{day}_to_day{stop}__{tf}';directory=output/'folds'/name;directory.mkdir(parents=True)
        baseline=dict(state=np.zeros(c['dimensions']),velocity=np.zeros(c['dimensions']))
        groups=[];feature_rows=[]
        for i,label in enumerate(labels):
            mask=labels_s==label;target=transform.state(future[labels_t==label]).astype('float32')
            f=source_features(transform,dict(s=s[mask],u=u[mask],ids=metadata[day].index[si[mask]].to_numpy(dtype=str)),baseline,c['neighbors'])
            rate=(anchors[i]-f['z'].astype(float).mean(0))/24
            path=directory/f'train_{label}.npz';np.savez_compressed(path,**f,target=target,global_rate=rate)
            groups.append(dict(label=label,path=str(path.resolve()),sha256=sha256(path)))
            feature_rows.append(f)
        norm={}
        for key in ('z','velocity','direct_U','pooled_SU'):
            a=np.vstack([f[key] for f in feature_rows]);norm[key+'_mean']=a.mean(0)
            norm[key+'_scale']=np.maximum(a.std(0),1e-3 if key=='z' else 1e-5)
        np.savez_compressed(directory/'normalization.npz',**norm)
        f=source_features(transform,dict(s=sv,u=uv,ids=metadata[day].index[vi].to_numpy(dtype=str)),baseline,c['neighbors'])
        rate=(anchor[0]-f['z'].astype(float).mean(0))/24
        score=np.random.default_rng(0).choice(len(vi),min(len(vi),512),replace=False)
        path=directory/'source.npz';np.savez_compressed(path,**f,global_rate=rate,score_indices=score)
        row=dict(name=name,index=index,TF=tf,source_day=day,target_day=stop,training_groups=groups,
            source_path=str(path.resolve()),source_sha256=sha256(path),transform_path=str(transform_path.resolve()),
            transform_sha256=sha256(transform_path),normalization_sha256=sha256(directory/'normalization.npz'),
            target_path=target_path,mean_selection=selection,mean_reproduces_W4=True)
        atomic_json(directory/'manifest.json',row);records.append(row)
        for seed in c['seeds']:
            for arm in ('static','real','shuffled_v','direct_U'):
                for ss in c['shuffle_seeds'] if arm=='shuffled_v' else [0]:
                    jobs.append(dict(fold=name,worker=index%3,seed=seed,arm=arm,shuffle_seed=ss,
                        name=f'shuffled_v_perm{ss}' if arm=='shuffled_v' else arm))
        run.json('status.json',dict(status='preparing',completed_folds=len(records)))
    run.json('study_manifest.json',dict(config=c,folds=records,jobs=jobs,scope='EXPLORATORY_EARLY_RESPONSE_ONLY',
        frozen_sha256={str(p.resolve()):sha256(p) for p in [PROTOCOL]+CODE},old_grid_sha256=sha256(oldroot/'grid_manifest.json')))
    run.json('status.json',dict(status='ready',folds=len(records),jobs=len(jobs)))


def verify(root):
    m=json.loads((root/'study_manifest.json').read_text())
    for p,h in m['frozen_sha256'].items():
        if sha256(p)!=h:raise ValueError(f'Frozen exploration code changed: {p}')
    return m


def train_worker(root,index):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda');root=Path(root);m=verify(root);c=m['config']
    for job in [j for j in m['jobs'] if j['worker']==index]:
        fold=root/'folds'/job['fold'];fm=json.loads((fold/'manifest.json').read_text())
        output=root/'training'/job['fold']/f'seed{job["seed"]}'/job['name']
        run=Artifacts(output,seed=job['seed'],inputs=[root/'study_manifest.json',fold/'manifest.json']+CODE)
        groups=[]
        for g in fm['training_groups']:
            if sha256(g['path'])!=g['sha256']:raise ValueError('Training group changed')
            groups.append(load_npz(g['path']))
        def checkpoint(step,model,optimizer,rng,history):
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,job=job),output/'model.pt')
        model,history=fit(groups,load_npz(fold/'normalization.npz'),c,arm=job['arm'],seed=job['seed'],
            shuffle_seed=job['shuffle_seed'],device='cuda',checkpoint=checkpoint,
            progress=lambda r:run.json('status.json',dict(status='training',job=job,**r)))
        if sha256(fm['source_path'])!=fm['source_sha256']:raise ValueError('Prediction source changed')
        source=load_npz(fm['source_path']);p=source_predictions(model,source,job['arm'],job['seed'],job['shuffle_seed'],0,24)
        si=source['score_indices']
        for key in ('candidates','q','cohort_candidates','cohort_q'):p[key]=p[key][si]
        p.update(source_ids=source['cell_ids'][si]);np.savez_compressed(output/'predictions.npz',**p)
        run.json('training.json',history)
        run.json('prediction_manifest.json',dict(job=job,path=str((output/'predictions.npz').resolve()),
            sha256=sha256(output/'predictions.npz'),model_sha256=sha256(output/'model.pt'),heldout_future_used=False))
        run.json('status.json',dict(status='complete',job=job));del model;torch.cuda.empty_cache()
    atomic_json(root/f'worker{index}.json',dict(status='complete'))


def evaluate(root,output):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda');root=Path(root);m=verify(root);frozen=[]
    for job in m['jobs']:
        d=root/'training'/job['fold']/f'seed{job["seed"]}'/job['name']
        pm=json.loads((d/'prediction_manifest.json').read_text())
        if pm['job']!=job or pm['heldout_future_used'] or sha256(pm['path'])!=pm['sha256'] or sha256(d/'model.pt')!=pm['model_sha256']:
            raise ValueError('Complete frozen prediction grid required')
        frozen.append(pm)
    run=Artifacts(output,seed=[0,1,2],inputs=[root/'study_manifest.json']+CODE)
    run.json('frozen_grid.json',dict(predictions=frozen));rows=[]
    names=['static','real','direct_U']+[f'shuffled_v_perm{i}' for i in range(3)]
    for fm in m['folds']:
        transform=SourceTransform.load(fm['transform_path'])
        obs,_=read_metadata(fm['target_path'])
        from w4_shorttime import qc_mask
        idx=np.flatnonzero(qc_mask(obs)&(obs['condition'].astype(str).to_numpy()==fm['TF']))
        y=transform.state(read_count_rows(fm['target_path'],'spliced',idx))
        ti=np.random.default_rng(0).choice(len(y),min(len(y),512),replace=False);y=y[ti]
        for name in names:
            selected=[p for p in frozen if p['job']['fold']==fm['name'] and p['job']['name']==name]
            if len(selected)!=3:raise ValueError('Incomplete seed mixture')
            packs=[load_npz(p['path']) for p in selected]
            x=np.concatenate([p['candidates'].reshape(-1,50) for p in packs])
            w=np.concatenate([p['q'].ravel()/len(p['q'])/3 for p in packs])
            e=energy(x,w,y,'cuda')
            rows.append(dict(TF=fm['TF'],source_day=fm['source_day'],target_day=fm['target_day'],arm=name,energy=e,
                source_cells=len(packs[0]['q']),target_cells=len(y)))
        run.json('status.json',dict(status='evaluating',completed_rows=len(rows)))
    gains=[]
    for tf in sorted(set(r['TF'] for r in rows)):
        for control in names:
            if control=='real':continue
            values=[]
            for day in (2,3):
                by={r['arm']:r['energy'] for r in rows if r['TF']==tf and r['source_day']==day}
                values.append(1-by['real']/by[control])
            gains.append(dict(TF=tf,control=control,mean_gain=float(np.mean(values))))
    result=dict(status='RENGE_EXPLORATORY_ROUTER_TRANSFER_COMPLETE',jobs=len(m['jobs']),
        gain_by_control={n:float(np.mean([r['mean_gain'] for r in gains if r['control']==n])) for n in names if n!='real'},
        confirmatory=False,perturbation_fate_claim=False,fixed_recipe_no_tuning=True)
    write_csv(run.directory/'metrics.csv',rows);write_csv(run.directory/'gains.csv',gains);run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# RENGE router transfer: EXPLORATORY\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'))


def launch(gpu,script,output,arguments):
    subprocess.run([sys.executable,str(ROOT/'scripts/phase0/launch_gpu_job.py'),'--gpu',str(gpu),'--script',script,
        '--output',str(output),'--',*arguments],cwd=ROOT,check=True)


def program(root,job_root):
    root,job_root=Path(root).resolve(),Path(job_root).resolve()
    prepare(root);deadline=time.monotonic()+7*86400
    while time.monotonic()<deadline:
        p=ROOT/'outputs/w14_program_v1/status.json'
        state=json.loads(p.read_text()) if p.exists() else dict(status='not_started')
        if state['status']=='failed':raise RuntimeError('W14 implementation failed; stop downstream migration')
        if state['status']=='complete':break
        atomic_json(root/'queue_status.json',dict(status='waiting_for_W14_completion'));time.sleep(20)
    else:raise TimeoutError('W14 dependency timeout')
    for i,gpu in enumerate((1,2,3)):
        launch(gpu,'scripts/phase0/w14b_renge_router.py',job_root/f'renge_worker{i}',
            ['worker','--root',str(root),'--worker',str(i)])
    while time.monotonic()<deadline:
        paths=[job_root/f'renge_worker{i}'/'status.json' for i in range(3)]
        states=[json.loads(p.read_text()) if p.exists() else dict(status='launching') for p in paths]
        if any(s['status']=='failed' for s in states):raise RuntimeError('RENGE exploratory worker failed')
        if all(s['status']=='complete' for s in states):break
        time.sleep(20)
    else:raise TimeoutError('RENGE grid timeout')
    launch(1,'scripts/phase0/w14b_renge_router.py',job_root/'renge_evaluation',
        ['evaluate','--root',str(root),'--output',str(root/'evaluation')])
    while time.monotonic()<deadline:
        p=job_root/'renge_evaluation/status.json'
        state=json.loads(p.read_text()) if p.exists() else dict(status='launching')
        if state['status']=='failed':raise RuntimeError('RENGE exploratory evaluation failed')
        if state['status']=='complete':break
        time.sleep(20)
    else:raise TimeoutError('RENGE evaluation timeout')
    destination=ROOT/'docs/current/w14b_renge_results.md'
    if destination.exists():raise ValueError('Existing RENGE transfer report retained')
    destination.write_text((root/'evaluation/RESULTS.md').read_text())
    files=[str(destination.relative_to(ROOT))]+[str((root/'evaluation'/n).relative_to(ROOT)) for n in
        ('RESULTS.md','summary.json','metrics.csv','gains.csv','frozen_grid.json')]
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','W14-B3: report single fixed-recipe RENGE router transfer'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','master'],cwd=ROOT,check=True)
    atomic_json(root/'queue_status.json',dict(status='complete',report=str(destination)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','evaluate','program'])
    p.add_argument('--root',default='outputs/w14b_renge_router_v1');p.add_argument('--output')
    p.add_argument('--worker',type=int,choices=[0,1,2]);p.add_argument('--job-root',default='outputs/w14_jobs_20261018')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='worker':train_worker(a.root,a.worker)
    elif a.action=='evaluate':evaluate(a.root,a.output)
    else:program(a.root,a.job_root)
