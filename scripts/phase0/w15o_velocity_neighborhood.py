"""Training-only source-neighborhood supervision of the unchanged S/U router.

Velocity defines relations, not a regression target, inference feature or ODE
arrow. Five matched arms distinguish a kinetic prior from generic smoothing.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
from w15n_velocity_auxiliary import shared_input, deployment_source

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARMS=('no_graph','state_graph','velocity_graph','shuffled_velocity_graph','U_graph')


def unit(values):
    values=np.asarray(values,dtype='float64')
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


def source_graphs(group, candidates, neighbors):
    """Only source state and source programs are inspected; never targets."""
    from scipy.spatial.distance import cdist
    x=np.asarray(group['z'],dtype='float64');n=len(x)
    if n<2 or not 1<=neighbors<=candidates:raise ValueError('Invalid source graph size')
    distances=cdist(x,x,metric='sqeuclidean');np.fill_diagonal(distances,np.inf)
    pool=np.argsort(distances,axis=1,kind='stable')[:,:min(candidates,n-1)]
    k=min(neighbors,pool.shape[1]);graphs={'state_graph':pool[:,:k].copy()}
    row=np.arange(n)[:,None]
    for arm,key in [('velocity_graph','velocity_program'),('shuffled_velocity_graph','shuffled_velocity_program'),
                    ('U_graph','U_program')]:
        feature=unit(group[key])
        if feature.ndim!=2 or len(feature)!=n or not np.isfinite(feature).all():
            raise ValueError('Invalid source program')
        dot=np.einsum('id,ijd->ij',feature,feature[pool])
        rank=np.argsort(1-np.clip(dot,-1,1)**2,axis=1,kind='stable')[:,:k]
        graphs[arm]=pool[row,rank]
    graphs['no_graph']=graphs['state_graph'].copy()
    for edge in graphs.values():
        if (edge==row).any() or (edge<0).any() or (edge>=n).any():raise ValueError('Invalid self or out-of-range edge')
    return graphs


def graph_diagnostics(group, graphs):
    """Descriptive train-source relationships, not paired future responses."""
    x=np.asarray(group['z'],dtype='float64');v=unit(group['velocity_program']);u=unit(group['U_program'])
    rows=[]
    for arm,edge in graphs.items():
        if arm=='no_graph':continue
        overlap=lambda other:float(np.mean([len(set(a)&set(b))/len(a) for a,b in zip(edge,graphs[other])]))
        vcos=np.einsum('id,ijd->ij',v,v[edge]);ucos=np.einsum('id,ijd->ij',u,u[edge])
        rows.append(dict(KO=group['label'],graph=arm,cells=len(x),neighbors=edge.shape[1],
            source_distance=float(np.linalg.norm(x[:,None]-x[edge],axis=-1).mean()),
            velocity_squared_cosine=float(np.square(vcos).mean()),U_squared_cosine=float(np.square(ucos).mean()),
            velocity_neighbor_mse=float(np.mean((v-v[edge].mean(1))**2)),
            overlap_state=overlap('state_graph'),overlap_U=overlap('U_graph'),overlap_velocity=overlap('velocity_graph')))
    return rows


def normalized_js(p,q):
    import jax.numpy as jnp
    if p.shape!=q.shape or p.ndim!=2:raise ValueError('Paired router probabilities required')
    m=(p+q)/2
    log=lambda a:jnp.log(jnp.maximum(a,1e-12))
    each=.5*jnp.sum(p*(log(p)-log(m))+q*(log(q)-log(m)),axis=-1)
    return jnp.mean(each)/jnp.log(2.)


def graph_loss(model,params,source,kinetic,neighbor_source,neighbor_kinetic):
    p=model.apply(dict(params=params),source,kinetic,method=model.route)
    q=model.apply(dict(params=params),neighbor_source,neighbor_kinetic,method=model.route)
    return normalized_js(p,q)


def make_step(model,match_fn):
    import jax
    import jax.numpy as jnp
    from ott.solvers import utils as solver_utils
    from w15a_cellflow_moe import align_ot_indices_jit,paired_flow_matching_loss

    @jax.jit
    def step(state,rng,source,target,kinetic,neighbor_source,neighbor_kinetic,condition,weight):
        pair_key,tau_key=jax.random.split(rng)
        si,ti=solver_utils.sample_joint(pair_key,match_fn(source,target))
        paired=align_ot_indices_jit(source,kinetic,target,si,ti,condition=condition)
        tau=jax.random.uniform(tau_key,(len(si),1),minval=0.,maxval=1.)
        def objective(params):
            fm,metrics=paired_flow_matching_loss(model,dict(params=params),paired,tau)
            relation=graph_loss(model,params,paired.source_x,paired.kinetic_input,
                neighbor_source[si],neighbor_kinetic[si])
            contribution=source.shape[1]*weight*relation
            return fm+contribution,dict(metrics,fm_loss=fm,graph_js=relation,graph_contribution=contribution)
        (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(state.params)
        return state.apply_gradients(grads=gradient),dict(metrics,loss=loss,source_indices=si,target_indices=ti)
    return step


def fit(model,step,training,graphs,c,plan,arm,seed,callback):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    import w15h_traxler_program_router as base
    first=training[0];x=first['z'][:1]
    initial=model.init(jax.random.PRNGKey(seed),jnp.asarray(x),jnp.asarray(shared_input(first,c['programs'])[:1]),
        jnp.asarray(x),.5,base.condition(first,1,c))['params']
    state=TrainState.create(apply_fn=model.apply,params=initial,tx=optax.adam(c['learning_rate']))
    inputs={g['label']:shared_input(g,c['programs']) for g in training}
    rng=np.random.default_rng(seed);neighbor_rng=np.random.default_rng(seed+1000000)
    history=[];started=time.monotonic();weight=0. if arm=='no_graph' else plan['graph_weight_normalized']
    for update in range(1,c['steps']+1):
        g=training[int(rng.integers(len(training)))];n=c['batch_size'];kinetic=inputs[g['label']]
        si=rng.choice(len(g['z']),n,replace=len(g['z'])<n);ti=rng.choice(len(g['target']),n,replace=len(g['target'])<n)
        edge=graphs[g['label']][arm];ni=edge[si,neighbor_rng.integers(edge.shape[1],size=n)]
        state,metrics=step(state,jax.random.fold_in(jax.random.PRNGKey(seed),update),
            jnp.asarray(g['z'][si]),jnp.asarray(g['target'][ti]),jnp.asarray(kinetic[si]),
            jnp.asarray(g['z'][ni]),jnp.asarray(kinetic[ni]),base.condition(g,n,c),jnp.asarray(weight,dtype=jnp.float32))
        if update==1 or update%c['log_interval']==0 or update==c['steps']:
            values={k:float(v) for k,v in metrics.items() if k not in ('source_indices','target_indices')}
            if not np.isfinite(list(values.values())).all():raise FloatingPointError('Nonfinite graph update')
            row=dict(step=update,elapsed_seconds=time.monotonic()-started,**values);history.append(row);callback(row)
    return state.params,initial,history


def verify_grid(records,c,split):
    expected={(seed,arm,ko) for seed in c['seeds'] for arm in ARMS for ko in split['val']}
    if len(records)!=len(expected) or {(r['seed'],r['arm'],r['KO']) for r in records}!=expected:
        raise ValueError('Freeze entire neighborhood grid before evaluation')
    for record in records:
        if sha256(record['path'])!=record['sha256'] or sha256(record['model'])!=record['model_sha256']:
            raise ValueError('Frozen neighborhood artifact changed')


def evaluate(output,c,split,norm,records):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics,summarize_pairs
    from w7_router_scaling import write_csv
    verify_grid(records,c,split)
    targets,metadata=base.load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
    if metadata.get('day')!=c['target_hours'] or set(targets['conditions'])!=set(split['val']):
        raise ValueError('Wrong development target identity')
    rows=[];baseline_checks=[]
    for ko in sorted(split['val']):
        y=targets['z'][targets['conditions']==ko]
        y=y[np.random.default_rng(c['evaluation_seed']).choice(len(y),min(len(y),c['evaluation_cells']),replace=False)]
        ids=None
        for r in [r for r in records if r['KO']==ko]:
            prediction=base.read_npz(r['path'])
            if ids is None:ids=prediction['source_ids']
            else:np.testing.assert_array_equal(ids,prediction['source_ids'])
            q=np.asarray(prediction['full_source_q'],dtype='float64');average=q.mean(0)
            rows.append(dict(KO=ko,seed=r['seed'],arm=r['arm'],
                effective_experts=float(np.exp(-np.sum(average*np.log(np.maximum(average,1e-12))))),
                mean_cell_entropy=float(-np.sum(q*np.log(np.maximum(q,1e-12)),axis=1).mean()),
                **empirical_metrics(prediction['candidates'],prediction['q'],y,norm['axes'])))
            if r['arm']=='no_graph':
                parent=Path(r['parent_prediction']);previous=base.read_npz(parent)
                np.testing.assert_array_equal(prediction['source_ids'],previous['source_ids'])
                baseline_checks.append(dict(KO=ko,seed=r['seed'],
                    q_max_abs=float(np.max(np.abs(prediction['q']-previous['q']))),
                    candidates_max_abs=float(np.max(np.abs(prediction['candidates']-previous['candidates'])))))
    write_csv(output/'metrics.csv',rows);write_csv(output/'baseline_reproduction.csv',baseline_checks)
    by={(r['KO'],r['seed'],r['arm']):r for r in rows};comparisons=[];primary=[]
    for control in ARMS:
        if control=='velocity_graph':continue
        selected=[]
        for ko in split['val']:
            for seed in c['seeds']:
                real=by[(ko,seed,'velocity_graph')];other=by[(ko,seed,control)]
                row=dict(KO=ko,seed=seed,control=control,energy_gain=1-real['energy']/other['energy'])
                comparisons.append(row);selected.append(row)
        per_seed={str(s):float(np.mean([r['energy_gain'] for r in selected if r['seed']==s])) for s in c['seeds']}
        per_ko={ko:float(np.mean([r['energy_gain'] for r in selected if r['KO']==ko])) for ko in split['val']}
        primary.append(dict(control=control,energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
            by_seed=per_seed,by_KO=per_ko,winning_KOs=sum(v>0 for v in per_ko.values())))
    write_csv(output/'comparisons.csv',comparisons)
    keys=('energy','mean_mse','variance_ratio','log_variance_error','effective_experts','mean_cell_entropy')
    return dict(arms_vs_no_graph=summarize_pairs(rows,'arm','no_graph'),velocity_vs_controls=primary,
        means={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm])) for key in keys} for arm in ARMS},
        baseline_reproduction_max_abs={key:max(r[key] for r in baseline_checks) for key in ('q_max_abs','candidates_max_abs')})


def launch(configuration,output,job_directory):
    from veloroute.artifacts import save_json
    from veloroute.gpu_policy import gpu_uuids,ALLOWED_PHYSICAL_GPUS,RESERVED_PHYSICAL_GPUS,MAX_VISIBLE_GPUS
    from w15i_followup_queue import worker_environment
    directory=Path(job_directory).resolve();output=Path(output).resolve()
    if directory.exists() or output.exists() or not directory.is_relative_to(ROOT/'outputs'):
        raise ValueError('New output and project-local job directory required')
    directory.mkdir(parents=True)
    env=worker_environment(output.parent/'runtime_gpu0');env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    job=dict(physical_gpu=0,gpu_uuid=gpu_uuids()[0],cwd=str(ROOT),
        command=[sys.executable,str(Path(__file__).resolve()),'--configuration',str(Path(configuration).resolve()),'--output',str(output)],
        allowed_gpus=list(ALLOWED_PHYSICAL_GPUS),reserved_gpus=list(RESERVED_PHYSICAL_GPUS),max_visible_gpus=MAX_VISIBLE_GPUS)
    save_json(directory/'job.json',job)
    with (directory/'job.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,str(HERE/'launch_gpu_job.py'),'--worker',str(directory)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(directory/'launcher.json',dict(pid=child.pid,physical_gpu=0,detached=True))
    return dict(status='LAUNCHED',pid=child.pid,physical_gpu=0,job_directory=str(directory))


def program(configuration,output):
    from w15k_traxler_l2_cell import wait_for_free_gpu
    from veloroute.gpu_policy import enforce_gpu_policy
    plan=json.loads(Path(configuration).read_text())
    if plan['gpu']!=0 or tuple(plan['arms'])!=ARMS:raise ValueError('Unregistered neighborhood grid')
    wait_for_free_gpu(0,time.monotonic()+600,lambda s:print(json.dumps(s),flush=True));enforce_gpu_policy('cuda')
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from cellflow.utils import match_linear
    from w15i_program_router_followups import cell_programs
    from w7_router_scaling import write_csv
    c=json.loads(Path(plan['base_configuration']).read_text());parent=Path(plan['parent'])
    if c['seeds']!=plan['model_seeds'] or c['steps']!=plan['steps']:raise ValueError('Parent budget changed')
    protocol=ROOT/'PLANS/w15o_velocity_neighborhood_20261022.md'
    inputs=[Path(configuration),Path(plan['base_configuration']),protocol,Path(__file__),HERE/'w15n_velocity_auxiliary.py',
        HERE/'w15h_traxler_program_router.py',HERE/'w15i_program_router_followups.py',HERE/'w15m_velocity_discrimination.py',
        HERE/'w15a_cellflow_moe.py',HERE/'w15a_cellflow_moe_official.py',HERE/'w15a_cellflow_moe_real_predict.py',
        HERE/'w15c_renge_l3.py',HERE/'w15a_cellflow_adapter.py',HERE/'w15a_cellflow_train.py',HERE/'w4_artifacts.py',
        Path(c['split']),Path(c['conditions']),parent/'training_programs.npz',parent/'frozen_grid.json',
        Path(c['fold'])/'train_source.npz',Path(c['fold'])/'train_target.npz',Path(c['fold'])/'validation_source.npz',
        Path(c['fold'])/'transform.npz']
    run=Artifacts(output,seed=c['seeds'],inputs=inputs);deadline=time.monotonic()+plan['maximum_hours']*3600
    planned=len(ARMS)*len(c['seeds'])
    try:
        training,heldout,norm,split,_=base.prepare(c);saved=base.read_npz(parent/'training_programs.npz')
        for key in norm:np.testing.assert_array_equal(norm[key],saved[key])
        training,_=cell_programs(training,norm,plan['shuffle_seed'])
        graphs={g['label']:source_graphs(g,plan['state_candidates'],plan['graph_neighbors']) for g in training}
        diagnostics=[row for g in training for row in graph_diagnostics(g,graphs[g['label']])]
        write_csv(run.directory/'source_graph_diagnostics.csv',diagnostics)
        np.savez_compressed(run.directory/'source_graphs.npz',**{f'{label}__{arm}':edge for label,group in graphs.items() for arm,edge in group.items()})
        np.savez_compressed(run.directory/'training_programs.npz',**norm)
        parent_records=json.loads((parent/'frozen_grid.json').read_text())['predictions']
        parent_paths={(r['seed'],r['KO']):r['path'] for r in parent_records if r['arm']=='S_U_router'}
        run.json('study_manifest.json',dict(plan=plan,config=c,planned_fits=planned,train_KOs=split['train'],
            validation_KOs=split['val'],reserved_test_KOs=split['test'],test_targets_read=False,
            graph_source_only=True,velocity_never_a_forward_input=True,graph_discarded_at_deployment=True))
        model=base.create_model(c,training[0]['z'].shape[1],0,training[0]['embedding'])
        matcher=functools.partial(match_linear,epsilon=c['ot_epsilon'],scale_cost='mean',max_iterations=c['ot_max_iterations'])
        step=make_step(model,matcher);integrate=base.make_candidate_integrator(model)
        records=[];summaries=[];initializations={}
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic()>deadline:raise TimeoutError('Neighborhood experiment deadline exceeded')
                directory=run.directory/'fits'/f'seed{seed}'/arm
                fit_run=Artifacts(directory,seed=seed,inputs=[Path(configuration),Path(__file__)])
                run.json('status.json',dict(status='training',seed=seed,arm=arm,completed_fits=len(summaries),total_fits=planned))
                params,initial,history=fit(model,step,training,graphs,c,plan,arm,seed,
                    lambda row:fit_run.json('status.json',dict(status='training',**row)))
                initial_hash=hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if seed in initializations and initializations[seed]!=initial_hash:raise ValueError('Matched initialization changed')
                initializations[seed]=initial_hash
                parent_hash=json.loads((parent/'fits'/f'seed{seed}'/'S_U_velocity_program'/'summary.json').read_text())['initial_parameters_sha256']
                if initial_hash!=parent_hash:raise ValueError('Parent initialization changed')
                checkpoint=directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(params))
                params=serialization.from_bytes(initial,checkpoint.read_bytes())
                for source in heldout:
                    prediction=base.predict(model,integrate,params,deployment_source(source,c['programs']),c)
                    path=directory/(source['label']+'.npz');np.savez_compressed(path,**prediction)
                    records.append(dict(KO=source['label'],seed=seed,arm=arm,path=str(path),sha256=sha256(path),
                        model=str(checkpoint),model_sha256=sha256(checkpoint),explicit_velocity_input=False,
                        parent_prediction=parent_paths[(seed,source['label'])],heldout_future_used=False))
                fit_run.json('training_history.json',dict(history=history))
                summary=dict(status='complete',seed=seed,arm=arm,steps=c['steps'],
                    parameters=sum(v.size for v in jax.tree_util.tree_leaves(params)),initial_parameters_sha256=initial_hash,
                    graph_js_first=history[0]['graph_js'],graph_js_last=history[-1]['graph_js'])
                fit_run.json('summary.json',summary);fit_run.json('status.json',dict(status='complete'));summaries.append(summary)
                print(json.dumps(dict(status='neighborhood_fit_complete',seed=seed,arm=arm,completed=len(summaries),total=planned)),flush=True)
        verify_grid(records,c,split)
        run.json('whole_grid_frozen.json',dict(predictions=records,validation_targets_read_before_freeze=False,test_targets_read=False))
        run.json('status.json',dict(status='evaluating',completed_fits=planned,total_fits=planned))
        results=evaluate(run.directory,c,split,norm,records)
        source_summary={arm:{key:float(np.mean([r[key] for r in diagnostics if r['graph']==arm]))
            for key in ('source_distance','velocity_squared_cosine','U_squared_cosine','velocity_neighbor_mse','overlap_state','overlap_U')}
            for arm in ARMS if arm!='no_graph'}
        result=dict(status='VELOCITY_NEIGHBORHOOD_SUPERVISION_COMPLETE',new_fits=planned,predictions=len(records),
            parameters=summaries[0]['parameters'],fit_diagnostics=summaries,source_graph_summary=source_summary,
            GFG=False,JVP=False,test_targets_read=False,explicit_velocity_input=False,**results)
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Velocity source-neighborhood role\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_fits=planned,total_fits=planned))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--job-directory');args=parser.parse_args()
    if args.launch:
        if not args.job_directory:parser.error('--launch requires --job-directory')
        result=launch(args.configuration,args.output,args.job_directory)
    else:result=program(args.configuration,args.output)
    print(json.dumps(result,indent=2),flush=True)
