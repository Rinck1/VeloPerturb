"""Fixed source-moment denoising x graph-strength expansion, three GPU lanes."""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts,sha256
import w15o_velocity_neighborhood as graph
import w15p_velocity_graph_controls as nulls
import w15q_velocity_countsplit as counts

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARMS=('velocity_graph','U_graph','shuffled_velocity_graph','global_shuffle_1','global_shuffle_2','S_local_shuffle')
CASE_NAMES=('smooth_w01','smooth_w05','raw_w05')
REFERENCE_ARMS=('U_program_input','velocity_program_input')


def source_operator(z,labels,neighbors,self_weight):
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import cdist
    z=np.asarray(z);labels=np.asarray(labels);n=len(z)
    if len(labels)!=n or not 0<=self_weight<=1 or neighbors<1:raise ValueError('Invalid smoothing specification')
    row=[np.arange(n)];col=[np.arange(n)];value=[np.full(n,self_weight)]
    for label in sorted(set(labels)):
        ix=np.flatnonzero(labels==label)
        if len(ix)<2:raise ValueError('Cannot smooth a singleton source cohort')
        k=min(neighbors,len(ix)-1);distance=cdist(z[ix],z[ix],metric='sqeuclidean');np.fill_diagonal(distance,np.inf)
        other=np.argsort(distance,axis=1,kind='stable')[:,:k]
        row.append(np.repeat(ix,k));col.append(ix[other].ravel());value.append(np.full(len(ix)*k,(1-self_weight)/k))
    result=csr_matrix((np.concatenate(value),(np.concatenate(row),np.concatenate(col))),shape=(n,n))
    np.testing.assert_allclose(np.asarray(result.sum(1)).ravel(),1.,atol=1e-12)
    return result


def teacher_groups(training,source,s,u,operator,transform,norm):
    """Denoise normalized count moments BEFORE the nonlinear velocity formula."""
    values=counts.latent(operator@s,operator@u,transform,norm)
    lookup={str(x):i for i,x in enumerate(source['cell_ids'])}
    indices={g['label']:np.array([lookup[str(x)] for x in g['cell_ids']]) for g in training}
    updated=dict(norm,velocity_reference=np.mean([values['velocity_arrow50'][indices[g['label']]].mean(0)
        for g in training],axis=0),U_reference=np.mean([values['U50'][indices[g['label']]].mean(0) for g in training],axis=0))
    groups=[]
    for g in training:
        ix=indices[g['label']]
        groups.append(dict(g,velocity=values['velocity_arrow50'][ix].astype('float32'),U=values['U50'][ix].astype('float32')))
    return groups,updated


def build_graphs(training,norm,plan,graph_plan):
    from w15i_program_router_followups import cell_programs
    groups,_=cell_programs(training,norm,0)
    extra,mappings=nulls.make_controls(groups,norm,plan,graph_plan)
    result={}
    for g in groups:
        original=graph.source_graphs(g,graph_plan['state_candidates'],graph_plan['graph_neighbors'])
        result[g['label']]={arm:(original[arm] if arm in original else extra[g['label']][arm]) for arm in ARMS}
    return result,groups,mappings


def reliability(source,s,u,depth,transform,norm,smooth_norm,plan):
    operator=source_operator(source['z'],source['conditions'],plan['smoothing']['neighbors'],plan['smoothing']['self_weight'])
    rows=[]
    for seed in plan['source_noise_seeds']:
        a,b=counts.split_counts(s,u,depth,seed)
        sa,ua=counts.normalized(*a,transform.target_sum);sb,ub=counts.normalized(*b,transform.target_sum)
        for teacher,T,reference in [('raw',None,norm),('smooth',operator,smooth_norm)]:
            first=counts.latent(sa if T is None else T@sa,ua if T is None else T@ua,transform,reference)
            second=counts.latent(sb if T is None else T@sb,ub if T is None else T@ub,transform,reference)
            for name in ('velocity_arrow50','velocity_program5','U_program5'):
                value=counts.cosine(first[name],second[name])
                for ko in sorted(set(source['conditions'])):
                    selected=value[source['conditions']==ko]
                    rows.append(dict(teacher=teacher,seed=seed,KO=ko,modality=name,
                        mean_cosine=float(selected.mean()),negative_fraction=float(np.mean(selected<0))))
    return rows


def source_inputs(plan):
    c=json.loads(Path(plan['base_configuration']).read_text());fold=Path(c['fold'])
    return [Path(plan['base_configuration']),Path(plan['graph_configuration']),
        ROOT/'PLANS/w15r_velocity_denoise_expand_20261022.md',Path(__file__),
        HERE/'w15o_velocity_neighborhood.py',HERE/'w15p_velocity_graph_controls.py',HERE/'w15q_velocity_countsplit.py',
        HERE/'w15n_velocity_auxiliary.py',HERE/'w15i_program_router_followups.py',HERE/'w15h_traxler_program_router.py',
        HERE/'w15m_velocity_discrimination.py',HERE/'w15g_response_program_probe.py',HERE/'w15a_cellflow_moe.py',
        HERE/'w15a_cellflow_moe_official.py',HERE/'w15a_cellflow_moe_real_predict.py',HERE/'w15c_renge_l3.py',
        HERE/'w15a_cellflow_adapter.py',HERE/'w15a_cellflow_train.py',HERE/'w4_artifacts.py',HERE/'w4_shorttime.py',
        ROOT/'src/veloroute/latent.py',fold/'train_source.npz',fold/'train_target.npz',fold/'validation_source.npz',
        fold/'transform.npz',Path(c['split']),Path(c['conditions']),Path(plan['parent'])/'whole_grid_frozen.json',
        Path(plan['original_input_parent'])/'frozen_grid.json']


def prepare(configuration,output):
    import w15h_traxler_program_router as base
    from veloroute.latent import FrozenSplicingTransform
    from w15i_program_router_followups import cell_programs
    from w7_router_scaling import write_csv
    plan=json.loads(Path(configuration).read_text());c=json.loads(Path(plan['base_configuration']).read_text())
    graph_plan=json.loads(Path(plan['graph_configuration']).read_text());fold=Path(c['fold'])
    source,_=base.load_pack(fold/'train_source.npz',expected_side='source')
    validation,_=base.load_pack(fold/'validation_source.npz',expected_side='source')
    libraries=sorted({str(x).split(':',1)[0] for x in np.r_[source['cell_ids'],validation['cell_ids']]})
    inputs=[Path(configuration),*source_inputs(plan),*(Path(plan['raw_directory'])/(lib+'.h5ad') for lib in libraries)]
    run=Artifacts(output,seed=plan['model_seeds'],inputs=inputs)
    try:
        training,_,norm,split,_=base.prepare(c);transform=FrozenSplicingTransform.load(fold/'transform.npz')
        s,u,depth=counts.read_selected(source,transform,plan['raw_directory'])
        sn,un=counts.normalized(s,u,depth,transform.target_sum);original=counts.latent(sn,un,transform,norm)
        for name,key in [('velocity_arrow50','velocity'),('U50','u_features'),('S_state50','z')]:
            np.testing.assert_allclose(original[name],source[key],rtol=1e-5,atol=2e-5)
        T=source_operator(source['z'],source['conditions'],plan['smoothing']['neighbors'],plan['smoothing']['self_weight'])
        smoothed,smooth_norm=teacher_groups(training,source,sn,un,T,transform,norm)
        cache={};variance=[];diagnostics=[]
        parent_records=json.loads((Path(plan['parent'])/'whole_grid_frozen.json').read_text())['predictions']
        original_graph_root=Path(next(r for r in parent_records if r['arm']=='velocity_graph')['model']).parents[3]
        with np.load(original_graph_root/'source_graphs.npz',allow_pickle=False) as f:original_graphs={key:f[key] for key in f.files}
        with np.load(Path(plan['parent'])/'source_graphs.npz',allow_pickle=False) as f:extra_graphs={key:f[key] for key in f.files}
        for teacher,groups,reference in [('raw',training,norm),('smooth',smoothed,smooth_norm)]:
            graphs,programs,_=build_graphs(groups,reference,plan,graph_plan)
            for g in programs:
                cache[f'{teacher}::{g["label"]}::source_ids']=g['cell_ids']
                for arm,edge in graphs[g['label']].items():
                    if teacher=='raw':
                        key=f'{g["label"]}__{arm}'
                        np.testing.assert_array_equal(edge,(original_graphs if key in original_graphs else extra_graphs)[key])
                    cache[f'{teacher}::{g["label"]}::{arm}']=edge
                cache[f'{teacher}::{g["label"]}::velocity_program']=g['velocity_program']
                cache[f'{teacher}::{g["label"]}::U_program']=g['U_program']
                for key,field,ref in [('velocity','velocity','velocity_reference'),('U','U','U_reference')]:
                    projected=(np.asarray(g[field],dtype=float)-reference[ref])@reference['axes'].T
                    variance.append(dict(teacher=teacher,KO=g['label'],modality=key,within_KO_variance=float(projected.var(0).sum())))
        np.savez_compressed(run.directory/'source_graphs.npz',**cache)
        np.savez_compressed(run.directory/'training_programs.npz',**norm,
            smooth_velocity_reference=smooth_norm['velocity_reference'],smooth_U_reference=smooth_norm['U_reference'])
        write_csv(run.directory/'teacher_variance.csv',variance)
        sv,uv,dv=counts.read_selected(validation,transform,plan['raw_directory'])
        noise=reliability(validation,sv,uv,dv,transform,norm,smooth_norm,plan)
        write_csv(run.directory/'source_reliability.csv',noise)
        noise_summary={teacher:{name:{key:float(np.mean([r[key] for r in noise if r['teacher']==teacher and r['modality']==name]))
            for key in ('mean_cosine','negative_fraction')} for name in ('velocity_arrow50','velocity_program5','U_program5')}
            for teacher in ('raw','smooth')}
        by={(r['teacher'],r['KO'],r['modality']):r['within_KO_variance'] for r in variance}
        retained={name:float(np.mean([by[('smooth',ko,name)]/max(by[('raw',ko,name)],1e-12) for ko in split['train']]))
            for name in ('velocity','U')}
        result=dict(status='DENOISE_SOURCE_PREPARATION_COMPLETE',train_cells=len(source['z']),validation_source_cells=len(validation['z']),
            train_KOs=split['train'],validation_KOs=split['val'],test_KOs=split['test'],future_validation_targets_read=False,
            graph_sha256=sha256(run.directory/'source_graphs.npz'),programs_sha256=sha256(run.directory/'training_programs.npz'),
            source_reliability=noise_summary,mean_within_KO_variance_retained=retained,GFG=False,JVP=False)
        run.json('ready.json',result);run.json('status.json',dict(status='complete'))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


def worker(configuration,output,case_name):
    from w15k_traxler_l2_cell import wait_for_free_gpu
    from veloroute.gpu_policy import enforce_gpu_policy
    plan=json.loads(Path(configuration).read_text());case=next(x for x in plan['cases'] if x['name']==case_name)
    wait_for_free_gpu(case['gpu'],time.monotonic()+1200,lambda s:print(json.dumps(s),flush=True));enforce_gpu_policy('cuda')
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from cellflow.utils import match_linear
    c=json.loads(Path(plan['base_configuration']).read_text());preparation=Path(output)/'preparation'
    ready=json.loads((preparation/'ready.json').read_text());graph_file=preparation/'source_graphs.npz'
    if sha256(graph_file)!=ready['graph_sha256']:raise ValueError('Prepared graphs changed')
    with np.load(graph_file,allow_pickle=False) as f:cache={key:f[key] for key in f.files}
    training,heldout,norm,split,_=base.prepare(c);teacher=case['teacher']
    graphs={g['label']:{arm:cache[f'{teacher}::{g["label"]}::{arm}'] for arm in ARMS} for g in training}
    for g in training:np.testing.assert_array_equal(g['cell_ids'],cache[f'{teacher}::{g["label"]}::source_ids'])
    run=Artifacts(Path(output)/'cases'/case_name,seed=c['seeds'],inputs=[Path(configuration),Path(__file__),
        preparation/'ready.json',graph_file,preparation/'training_programs.npz'])
    try:
        if c['seeds']!=plan['model_seeds'] or c['steps']!=plan['steps']:raise ValueError('Original fixed budget changed')
        model=base.create_model(c,training[0]['z'].shape[1],0,training[0]['embedding'])
        matcher=functools.partial(match_linear,epsilon=c['ot_epsilon'],scale_cost='mean',max_iterations=c['ot_max_iterations'])
        step=graph.make_step(model,matcher);integrate=base.make_candidate_integrator(model);records=[];summaries=[]
        deadline=time.monotonic()+plan['maximum_hours']*3600
        for seed in c['seeds']:
            parent_model=next(r for r in json.loads((Path(plan['parent'])/'whole_grid_frozen.json').read_text())['predictions']
                if r['seed']==seed and r['arm']=='velocity_graph')
            expected=json.loads((Path(parent_model['model']).parent/'summary.json').read_text())['initial_parameters_sha256']
            for arm in ARMS:
                if time.monotonic()>deadline:raise TimeoutError('Fixed expansion worker deadline')
                directory=run.directory/'fits'/f'seed{seed}'/arm
                fit_run=Artifacts(directory,seed=seed,inputs=[Path(configuration),Path(__file__)])
                run.json('status.json',dict(status='training',case=case_name,seed=seed,arm=arm,completed_fits=len(summaries),total_fits=18))
                params,initial,history=graph.fit(model,step,training,graphs,c,case,arm,seed,
                    lambda row:fit_run.json('status.json',dict(status='training',**row)))
                initial_hash=hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if initial_hash!=expected:raise ValueError('Matched parent initialization changed')
                checkpoint=directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(params))
                params=serialization.from_bytes(initial,checkpoint.read_bytes())
                for source in heldout:
                    prediction=base.predict(model,integrate,params,graph.deployment_source(source,c['programs']),c)
                    path=directory/(source['label']+'.npz');np.savez_compressed(path,**prediction)
                    records.append(dict(case=case_name,KO=source['label'],seed=seed,arm=arm,path=str(path),sha256=sha256(path),
                        model=str(checkpoint),model_sha256=sha256(checkpoint),explicit_velocity_input=False,heldout_future_used=False))
                fit_run.json('training_history.json',dict(history=history))
                summary=dict(status='complete',case=case_name,seed=seed,arm=arm,parameters=sum(v.size for v in jax.tree_util.tree_leaves(params)),
                    initial_parameters_sha256=initial_hash,graph_js_last=history[-1]['graph_js'])
                fit_run.json('summary.json',summary);fit_run.json('status.json',dict(status='complete'));summaries.append(summary)
                print(json.dumps(dict(status='denoise_model_complete',case=case_name,seed=seed,arm=arm,completed_fits=len(summaries),total_fits=18)),flush=True)
        run.json('frozen_grid.json',dict(predictions=records,validation_targets_read=False,test_targets_read=False))
        run.json('summary.json',dict(status='DENOISE_EXPANSION_LANE_COMPLETE',case=case_name,new_fits=18,predictions=len(records)))
        run.json('status.json',dict(status='complete',completed_fits=18,total_fits=18))
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


def parent_records(plan):
    records=copy.deepcopy(json.loads((Path(plan['parent'])/'whole_grid_frozen.json').read_text())['predictions'])
    for r in records:r['case']='raw_w01'
    reference=json.loads((Path(plan['original_input_parent'])/'frozen_grid.json').read_text())['predictions']
    names={'S_U_U_program':'U_program_input','S_U_velocity_program':'velocity_program_input'}
    for original in reference:
        if original['arm'] in names:
            r=copy.deepcopy(original);r.update(case='input_reference',arm=names[original['arm']]);records.append(r)
    return records


def verify_grid(records,plan,c,split):
    expected={(case,arm,seed,ko) for case in CASE_NAMES for arm in ARMS for seed in c['seeds'] for ko in split['val']}
    expected|={('raw_w01',arm,seed,ko) for arm in graph.ARMS+nulls.ARMS for seed in c['seeds'] for ko in split['val']}
    expected|={('input_reference',arm,seed,ko) for arm in REFERENCE_ARMS for seed in c['seeds'] for ko in split['val']}
    actual={(r['case'],r['arm'],r['seed'],r['KO']) for r in records}
    if len(records)!=len(expected) or actual!=expected:raise ValueError('Entire expansion grid must be frozen before target access')
    checked=set()
    for r in records:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Prediction changed after freezing')
        if r['model'] not in checked:
            if sha256(r['model'])!=r['model_sha256']:raise ValueError('Model changed after freezing')
            checked.add(r['model'])


def comparison_summary(rows,c,split):
    by={(r['case'],r['arm'],r['seed'],r['KO']):r for r in rows};summaries=[];comparisons=[]
    for case in ('raw_w01',)+CASE_NAMES:
        controls={'no_graph':(('raw_w01','no_graph'),),'original_velocity_graph':(('raw_w01','velocity_graph'),),
            'direct_U_input':(('input_reference','U_program_input'),),'direct_velocity_input':(('input_reference','velocity_program_input'),),
            'same_case_U_graph':((case,'U_graph'),),'same_case_S_local_shuffle':((case,'S_local_shuffle'),),
            'same_case_mean_three_shuffles':tuple((case,a) for a in ('shuffled_velocity_graph','global_shuffle_1','global_shuffle_2'))}
        for control,keys in controls.items():
            selected=[]
            for seed in c['seeds']:
                for ko in split['val']:
                    real=by[(case,'velocity_graph',seed,ko)]['energy'];other=float(np.mean([by[(a,b,seed,ko)]['energy'] for a,b in keys]))
                    row=dict(case=case,control=control,seed=seed,KO=ko,real_energy=real,control_energy=other,energy_gain=1-real/other)
                    comparisons.append(row);selected.append(row)
            seeds={str(seed):float(np.mean([r['energy_gain'] for r in selected if r['seed']==seed])) for seed in c['seeds']}
            kos={ko:float(np.mean([r['energy_gain'] for r in selected if r['KO']==ko])) for ko in split['val']}
            summaries.append(dict(case=case,control=control,energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
                by_seed=seeds,by_KO=kos,winning_KOs=sum(v>0 for v in kos.values()),all_seeds_positive=all(v>0 for v in seeds.values())))
    return summaries,comparisons


def evaluate(output,plan,c,split,records):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics
    from w7_router_scaling import write_csv
    from threadpoolctl import threadpool_limits
    verify_grid(records,plan,c,split)
    target,metadata=base.load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
    if metadata.get('day')!=c['target_hours'] or set(target['conditions'])!=set(split['val']):raise ValueError('Wrong evaluation targets')
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
    write_csv(Path(output)/'metrics.csv',rows);summary,comparisons=comparison_summary(rows,c,split)
    write_csv(Path(output)/'comparisons.csv',comparisons)
    main=[r for r in summary if r['case']==plan['primary_candidate']]
    needed=('no_graph','direct_U_input','same_case_U_graph','same_case_S_local_shuffle','same_case_mean_three_shuffles')
    consistent=all(next(r for r in main if r['control']==key)['all_seeds_positive'] for key in needed)
    target_gain=next(r for r in main if r['control']=='no_graph')['energy_gain']>=plan['target_gain_vs_no_graph']
    return dict(primary_candidate=plan['primary_candidate'],primary_comparisons=main,all_comparisons=summary,
        primary_consistent_across_controls=consistent,target_gain_vs_no_graph_reached=target_gain,
        primary_expansion_goal_met=consistent and target_gain,
        means={f'{case}/{arm}':{key:float(np.mean([r[key] for r in rows if r['case']==case and r['arm']==arm]))
            for key in ('energy','mean_mse','variance_ratio','log_variance_error','effective_experts','mean_cell_entropy')}
            for case,arm in sorted({(r['case'],r['arm']) for r in rows})})


def launch_job(configuration,output,job_directory,mode,case=None):
    from veloroute.artifacts import save_json
    from veloroute.gpu_policy import gpu_uuids,ALLOWED_PHYSICAL_GPUS,RESERVED_PHYSICAL_GPUS,MAX_VISIBLE_GPUS
    from w15i_followup_queue import worker_environment
    plan=json.loads(Path(configuration).read_text());gpu=None if mode=='controller' else case['gpu']
    directory=Path(job_directory).resolve()
    if directory.exists() or not directory.is_relative_to(ROOT/'outputs'):raise ValueError('New project-local job directory required')
    directory.mkdir(parents=True)
    env=worker_environment(Path(output).parent/('runtime_cpu' if gpu is None else f'runtime_gpu{gpu}'))
    if gpu is None:env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    command=[sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--configuration',str(Path(configuration).resolve()),'--output',str(Path(output).resolve())]
    if case is not None:command.extend(['--case',case['name']])
    job=dict(physical_gpu=gpu,gpu_uuid=None if gpu is None else gpu_uuids()[gpu],command=command,cwd=str(ROOT),
        allowed_gpus=list(ALLOWED_PHYSICAL_GPUS),reserved_gpus=list(RESERVED_PHYSICAL_GPUS),max_visible_gpus=MAX_VISIBLE_GPUS)
    save_json(directory/'job.json',job)
    with (directory/'job.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,str(HERE/'launch_gpu_job.py'),'--worker',str(directory)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(directory/'launcher.json',dict(pid=child.pid,physical_gpu=gpu,detached=True))
    return dict(status='LAUNCHED',pid=child.pid,physical_gpu=gpu,mode=mode,case=None if case is None else case['name'],job_directory=str(directory))


def controller(configuration,output):
    plan=json.loads(Path(configuration).read_text());output=Path(output)
    if tuple(plan['arms'])!=ARMS or tuple(c['name'] for c in plan['cases'])!=CASE_NAMES:
        raise ValueError('Unregistered fixed expansion grid')
    run=Artifacts(output,seed=plan['model_seeds'],inputs=[Path(configuration),*source_inputs(plan)])
    deadline=time.monotonic()+plan['maximum_hours']*3600
    try:
        run.json('status.json',dict(status='preparing',new_fits=54))
        prepared=prepare(configuration,output/'preparation')
        run.json('source_diagnostics.json',prepared)
        receipts=[]
        for case in plan['cases']:
            receipts.append(launch_job(configuration,output,Path(plan['job_directory'])/case['name'],'worker',case))
        run.json('launch_receipts.json',dict(jobs=receipts))
        while True:
            states={}
            for case in CASE_NAMES:
                path=output/'cases'/case/'status.json';states[case]=json.loads(path.read_text()) if path.exists() else {}
            if any(s.get('status')=='failed' for s in states.values()):raise RuntimeError('An expansion lane failed; inspect preserved logs')
            complete=sum(int(s.get('completed_fits',0)) for s in states.values())
            run.json('status.json',dict(status='training',completed_fits=complete,total_fits=54,
                cases={name:{k:s.get(k) for k in ('status','seed','arm','completed_fits','total_fits')} for name,s in states.items()}))
            if all(s.get('status')=='complete' for s in states.values()):break
            if time.monotonic()>deadline:raise TimeoutError('Fixed expansion controller deadline')
            time.sleep(10)
        c=json.loads(Path(plan['base_configuration']).read_text());split=json.loads(Path(c['split']).read_text())
        records=parent_records(plan)
        for case in CASE_NAMES:records.extend(json.loads((output/'cases'/case/'frozen_grid.json').read_text())['predictions'])
        verify_grid(records,plan,c,split)
        run.json('whole_grid_frozen.json',dict(predictions=records,validation_targets_read_before_freeze=False,test_targets_read=False))
        run.json('status.json',dict(status='evaluating',completed_fits=54,total_fits=54))
        result=evaluate(output,plan,c,split,records)
        result.update(status='VELOCITY_DENOISE_EXPANSION_COMPLETE',new_fits=54,predictions=len(records),source_diagnostics=prepared,
            parameters=389704,GFG=False,JVP=False,test_targets_read=False)
        run.json('summary.json',result)
        (output/'RESULTS.md').write_text('# Fixed velocity denoising expansion\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_fits=54,total_fits=54))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['launch','controller','worker'],required=True)
    parser.add_argument('--configuration',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case')
    args=parser.parse_args()
    if args.mode=='launch':
        plan=json.loads(Path(args.configuration).read_text())
        if Path(args.output).exists():raise ValueError('Do not overwrite prior study')
        result=launch_job(args.configuration,args.output,Path(plan['job_directory'])/'controller','controller')
    elif args.mode=='controller':result=controller(args.configuration,args.output)
    else:result=worker(args.configuration,args.output,args.case)
    if result is not None:print(json.dumps(result,indent=2),flush=True)
