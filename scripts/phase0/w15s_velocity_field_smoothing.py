"""Secondary fixed field-moment teacher; no changes to the running count grid."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts,sha256
import w15r_velocity_denoise_expand as expansion

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CASE='field_w01'


def field_teacher(training,source,norm,T):
    v=T@source['velocity'];u=T@source['u_features']
    labels=source['conditions'];keys=sorted(set(labels))
    reference=dict(norm,velocity_reference=np.mean([v[labels==key].mean(0) for key in keys],axis=0),
        U_reference=np.mean([u[labels==key].mean(0) for key in keys],axis=0))
    lookup={str(key):i for i,key in enumerate(source['cell_ids'])};groups=[]
    for g in training:
        ix=np.array([lookup[str(key)] for key in g['cell_ids']])
        groups.append(dict(g,velocity=np.asarray(v[ix],dtype='float32'),U=np.asarray(u[ix],dtype='float32')))
    return groups,reference


def prepare(configuration,output):
    import w15h_traxler_program_router as base
    from veloroute.latent import FrozenSplicingTransform
    from w7_router_scaling import write_csv
    plan=json.loads(Path(configuration).read_text());c=json.loads(Path(plan['base_configuration']).read_text())
    fold=Path(c['fold']);source,_=base.load_pack(fold/'train_source.npz',expected_side='source')
    val,_=base.load_pack(fold/'validation_source.npz',expected_side='source')
    training,_,norm,split,_=base.prepare(c);sm=plan['smoothing']
    T=expansion.source_operator(source['z'],source['conditions'],sm['neighbors'],sm['self_weight'])
    groups,reference=field_teacher(training,source,norm,T)
    libraries=sorted({str(key).split(':',1)[0] for key in val['cell_ids']})
    run=Artifacts(output,seed=c['seeds'],inputs=[Path(configuration),Path(__file__),ROOT/'PLANS/w15s_velocity_field_smoothing_20261022.md',
        *expansion.source_inputs(plan),*(Path(plan['raw_directory'])/(lib+'.h5ad') for lib in libraries)])
    try:
        gp=json.loads(Path(plan['graph_configuration']).read_text());graphs,programs,_=expansion.build_graphs(groups,reference,plan,gp)
        cache={}
        for g in programs:
            cache[f'field::{g["label"]}::source_ids']=g['cell_ids']
            for arm,edge in graphs[g['label']].items():cache[f'field::{g["label"]}::{arm}']=edge
        np.savez_compressed(run.directory/'source_graphs.npz',**cache)
        np.savez_compressed(run.directory/'training_programs.npz',**norm,
            field_velocity_reference=reference['velocity_reference'],field_U_reference=reference['U_reference'])
        transform=FrozenSplicingTransform.load(fold/'transform.npz')
        s,u,depth=expansion.counts.read_selected(val,transform,plan['raw_directory'])
        Tv=expansion.source_operator(val['z'],val['conditions'],sm['neighbors'],sm['self_weight']);rows=[]
        for seed in plan['source_noise_seeds']:
            a,b=expansion.counts.split_counts(s,u,depth,seed)
            la=expansion.counts.latent(*expansion.counts.normalized(*a,transform.target_sum),transform,norm)
            lb=expansion.counts.latent(*expansion.counts.normalized(*b,transform.target_sum),transform,norm)
            for name,ref in [('velocity_arrow50',reference['velocity_reference']),('U50',reference['U_reference'])]:
                ca=(Tv@la[name]-ref)@norm['axes'].T;cb=(Tv@lb[name]-ref)@norm['axes'].T
                value=expansion.counts.cosine(ca,cb)
                for ko in sorted(set(val['conditions'])):
                    selected=value[val['conditions']==ko]
                    rows.append(dict(seed=seed,KO=ko,modality=name,mean_cosine=float(selected.mean()),
                        negative_fraction=float(np.mean(selected<0))))
        write_csv(run.directory/'source_reliability.csv',rows)
        result=dict(status='FIELD_SMOOTHING_SOURCE_READY',graph_sha256=sha256(run.directory/'source_graphs.npz'),
            programs_sha256=sha256(run.directory/'training_programs.npz'),train_cells=len(source['z']),
            source_reliability={name:{key:float(np.mean([r[key] for r in rows if r['modality']==name]))
                for key in ('mean_cosine','negative_fraction')} for name in ('velocity_arrow50','U50')},
            test_targets_read=False,validation_targets_read=False,GFG=False,JVP=False)
        run.json('ready.json',result);run.json('status.json',dict(status='complete'))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


def verify(records,c,split):
    expected={(CASE,arm,seed,ko) for arm in expansion.ARMS for seed in c['seeds'] for ko in split['val']}
    expected|={('raw_w01',arm,seed,ko) for arm in expansion.graph.ARMS+expansion.nulls.ARMS for seed in c['seeds'] for ko in split['val']}
    expected|={('input_reference',arm,seed,ko) for arm in expansion.REFERENCE_ARMS for seed in c['seeds'] for ko in split['val']}
    expected|={('count_reference','velocity_graph',seed,ko) for seed in c['seeds'] for ko in split['val']}
    if len(records)!=len(expected) or {(r['case'],r['arm'],r['seed'],r['KO']) for r in records}!=expected:
        raise ValueError('Freeze complete field and reference grid before future target access')
    for r in records:
        if sha256(r['path'])!=r['sha256'] or sha256(r['model'])!=r['model_sha256']:raise ValueError('Frozen field/reference artifact changed')


def evaluate(output,plan,c,split,records):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics
    from w7_router_scaling import write_csv
    from threadpoolctl import threadpool_limits
    verify(records,c,split)
    target,meta=base.load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
    if meta['day']!=24 or set(target['conditions'])!=set(split['val']):raise ValueError('Wrong development targets')
    with np.load(Path(output)/'preparation/training_programs.npz',allow_pickle=False) as f:axes=f['axes']
    rows=[]
    with threadpool_limits(limits=1):
        for ko in sorted(split['val']):
            y=target['z'][target['conditions']==ko]
            y=y[np.random.default_rng(c['evaluation_seed']).choice(len(y),min(len(y),c['evaluation_cells']),replace=False)]
            ids=None
            for r in [r for r in records if r['KO']==ko]:
                p=base.read_npz(r['path'])
                if ids is None:ids=p['source_ids']
                else:np.testing.assert_array_equal(ids,p['source_ids'])
                q=np.asarray(p['full_source_q'],dtype=float);average=q.mean(0)
                rows.append(dict(case=r['case'],arm=r['arm'],seed=r['seed'],KO=ko,
                    effective_experts=float(np.exp(-np.sum(average*np.log(np.maximum(average,1e-12))))),
                    mean_cell_entropy=float(-np.sum(q*np.log(np.maximum(q,1e-12)),axis=1).mean()),
                    **empirical_metrics(p['candidates'],p['q'],y,axes)))
    by={(r['case'],r['arm'],r['seed'],r['KO']):r for r in rows};comparisons=[];summaries=[]
    controls={'no_graph':(('raw_w01','no_graph'),),'original_velocity_graph':(('raw_w01','velocity_graph'),),
        'direct_U_input':(('input_reference','U_program_input'),),'count_moment_smoothing':(('count_reference','velocity_graph'),),
        'same_case_U_graph':((CASE,'U_graph'),),'same_case_S_local_shuffle':((CASE,'S_local_shuffle'),),
        'same_case_mean_three_shuffles':tuple((CASE,a) for a in ('shuffled_velocity_graph','global_shuffle_1','global_shuffle_2'))}
    for control,pairs in controls.items():
        selected=[]
        for seed in c['seeds']:
            for ko in split['val']:
                real=by[(CASE,'velocity_graph',seed,ko)]['energy'];other=float(np.mean([by[(a,b,seed,ko)]['energy'] for a,b in pairs]))
                row=dict(control=control,seed=seed,KO=ko,energy_gain=1-real/other);comparisons.append(row);selected.append(row)
        seeds={str(seed):float(np.mean([r['energy_gain'] for r in selected if r['seed']==seed])) for seed in c['seeds']}
        kos={ko:float(np.mean([r['energy_gain'] for r in selected if r['KO']==ko])) for ko in split['val']}
        summaries.append(dict(control=control,energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
            by_seed=seeds,by_KO=kos,winning_KOs=sum(v>0 for v in kos.values()),all_seeds_positive=all(v>0 for v in seeds.values())))
    write_csv(Path(output)/'metrics.csv',rows);write_csv(Path(output)/'comparisons.csv',comparisons)
    return dict(comparisons=summaries,means={arm:{key:float(np.mean([r[key] for r in rows if r['case']==CASE and r['arm']==arm]))
        for key in ('energy','mean_mse','variance_ratio','log_variance_error','effective_experts','mean_cell_entropy')} for arm in expansion.ARMS})


def controller(configuration,output):
    plan=json.loads(Path(configuration).read_text());output=Path(output)
    run=Artifacts(output,seed=plan['model_seeds'],inputs=[Path(configuration),Path(__file__),
        ROOT/'PLANS/w15s_velocity_field_smoothing_20261022.md',*expansion.source_inputs(plan)])
    deadline=time.monotonic()+plan['maximum_hours']*3600
    try:
        run.json('status.json',dict(status='preparing',total_fits=18))
        prepared=prepare(configuration,output/'preparation');run.json('source_diagnostics.json',prepared)
        launch=expansion.launch_job(configuration,output,Path(plan['job_directory'])/CASE,'worker',plan['cases'][0])
        run.json('launch_receipts.json',dict(jobs=[launch]))
        while True:
            path=output/'cases'/CASE/'status.json';state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('status')=='failed':raise RuntimeError('Field worker failed; logs preserved')
            count_path=Path(plan['count_study'])/'status.json';count_state=json.loads(count_path.read_text())
            if count_state.get('status')=='failed':raise RuntimeError('Count study reference failed; field outputs retained')
            run.json('status.json',dict(status='training' if state.get('status')!='complete' else 'waiting_count_reference',
                completed_fits=state.get('completed_fits',0),total_fits=18,count_study_status=count_state.get('status')))
            if state.get('status')=='complete' and count_state.get('status')=='complete':break
            if time.monotonic()>deadline:raise TimeoutError('Field study deadline')
            time.sleep(10)
        c=json.loads(Path(plan['base_configuration']).read_text());split=json.loads(Path(c['split']).read_text())
        records=expansion.parent_records(plan)
        records.extend(json.loads((output/'cases'/CASE/'frozen_grid.json').read_text())['predictions'])
        for original in json.loads((Path(plan['count_study'])/'cases/smooth_w01/frozen_grid.json').read_text())['predictions']:
            if original['arm']=='velocity_graph':
                r=copy.deepcopy(original);r['case']='count_reference';records.append(r)
        verify(records,c,split)
        run.json('whole_grid_frozen.json',dict(predictions=records,validation_targets_read_before_freeze=False,test_targets_read=False))
        run.json('status.json',dict(status='evaluating',completed_fits=18,total_fits=18))
        result=evaluate(output,plan,c,split,records)
        result.update(status='VELOCITY_FIELD_SMOOTHING_COMPLETE',new_fits=18,predictions=len(records),source_diagnostics=prepared,
            parameters=389704,GFG=False,JVP=False,test_targets_read=False,secondary_candidate_not_W15R_primary=True)
        run.json('summary.json',result)
        (output/'RESULTS.md').write_text('# Fixed field-smoothing comparison\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_fits=18,total_fits=18))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


def launch(configuration,output):
    from veloroute.artifacts import save_json
    from w15i_followup_queue import worker_environment
    plan=json.loads(Path(configuration).read_text());directory=Path(plan['job_directory'])/'controller'
    if directory.exists() or Path(output).exists():raise ValueError('New study and job directory required')
    directory.mkdir(parents=True)
    env=worker_environment(Path(output).parent/'runtime_cpu');env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    command=[sys.executable,str(Path(__file__).resolve()),'--mode','controller','--configuration',str(Path(configuration).resolve()),'--output',str(Path(output).resolve())]
    save_json(directory/'job.json',dict(physical_gpu=None,gpu_uuid=None,cwd=str(ROOT),command=command))
    with (directory/'job.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,str(HERE/'launch_gpu_job.py'),'--worker',str(directory)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(directory/'launcher.json',dict(pid=child.pid,detached=True,physical_gpu=None))
    return dict(status='LAUNCHED',pid=child.pid,physical_gpu=None,job_directory=str(directory))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['launch','controller'],required=True)
    parser.add_argument('--configuration',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=launch(args.configuration,args.output) if args.mode=='launch' else controller(args.configuration,args.output)
    print(json.dumps(result,indent=2),flush=True)
