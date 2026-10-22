"""Additional fixed global and S-local nulls for the completed W15O grid."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
import w15o_velocity_neighborhood as graph

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARMS=('global_shuffle_1','global_shuffle_2','S_local_shuffle')


def make_controls(training,norm,plan,parent_plan):
    from w15i_program_router_followups import cell_programs
    from w15m_velocity_discrimination import local_derangement
    result={g['label']:{} for g in training};mappings={}
    for shuffle_seed,arm in zip(plan['global_shuffle_seeds'],ARMS[:2]):
        changed,_=cell_programs(training,norm,shuffle_seed)
        for g in changed:
            edges=graph.source_graphs(g,parent_plan['state_candidates'],parent_plan['graph_neighbors'])
            result[g['label']][arm]=edges['shuffled_velocity_graph']
    for g in training:
        identity=f"{g['label']}|{g['time_hours']}|graph_local_null"
        seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')+plan['local_shuffle_seed']
        order=local_derangement(g['z'],seed)
        changed=dict(g,velocity_program=np.asarray(g['velocity_program'])[order])
        result[g['label']]['S_local_shuffle']=graph.source_graphs(changed,parent_plan['state_candidates'],
            parent_plan['graph_neighbors'])['velocity_graph']
        mappings[g['label']]=dict(source_ids=g['cell_ids'].tolist(),permutation=order.tolist(),
            mean_S_distance=float(np.linalg.norm(g['z']-g['z'][order],axis=1).mean()))
    return result,mappings


def validate_grid(records,c,split):
    expected={(seed,arm,ko) for seed in c['seeds'] for arm in graph.ARMS+ARMS for ko in split['val']}
    if len(records)!=len(expected) or {(r['seed'],r['arm'],r['KO']) for r in records}!=expected:
        raise ValueError('Freeze complete parent and new null grid before target access')
    for r in records:
        if sha256(r['path'])!=r['sha256'] or sha256(r['model'])!=r['model_sha256']:
            raise ValueError('Frozen parent or new null artifact changed')


def paired_summary(rows,c,split):
    """Average null ERRORS before forming each paired relative gain."""
    by={(r['KO'],r['seed'],r['arm']):r for r in rows};comparisons=[];summaries=[]
    controls={a:(a,) for a in graph.ARMS+ARMS if a!='velocity_graph'}
    controls['mean_three_global_shuffles']=('shuffled_velocity_graph','global_shuffle_1','global_shuffle_2')
    for control,arms in controls.items():
        selected=[]
        for ko in split['val']:
            for seed in c['seeds']:
                real=by[(ko,seed,'velocity_graph')]['energy']
                null=float(np.mean([by[(ko,seed,arm)]['energy'] for arm in arms]))
                row=dict(KO=ko,seed=seed,control=control,real_energy=real,control_energy=null,energy_gain=1-real/null)
                selected.append(row);comparisons.append(row)
        by_seed={str(seed):float(np.mean([r['energy_gain'] for r in selected if r['seed']==seed])) for seed in c['seeds']}
        by_ko={ko:float(np.mean([r['energy_gain'] for r in selected if r['KO']==ko])) for ko in split['val']}
        summaries.append(dict(control=control,energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
            by_seed=by_seed,by_KO=by_ko,winning_KOs=sum(v>0 for v in by_ko.values()),
            all_seeds_positive=all(v>0 for v in by_seed.values())))
    return summaries,comparisons


def evaluate(output,c,split,norm,records):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics,summarize_pairs
    from w7_router_scaling import write_csv
    validate_grid(records,c,split)
    target,metadata=base.load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
    if metadata.get('day')!=c['target_hours'] or set(target['conditions'])!=set(split['val']):
        raise ValueError('Wrong development target identity')
    rows=[]
    for ko in sorted(split['val']):
        y=target['z'][target['conditions']==ko]
        y=y[np.random.default_rng(c['evaluation_seed']).choice(len(y),min(len(y),c['evaluation_cells']),replace=False)]
        ids=None
        for r in [r for r in records if r['KO']==ko]:
            p=base.read_npz(r['path'])
            if ids is None:ids=p['source_ids']
            else:np.testing.assert_array_equal(ids,p['source_ids'])
            q=np.asarray(p['full_source_q'],dtype='float64');average=q.mean(0)
            rows.append(dict(KO=ko,seed=r['seed'],arm=r['arm'],
                effective_experts=float(np.exp(-np.sum(average*np.log(np.maximum(average,1e-12))))),
                mean_cell_entropy=float(-np.sum(q*np.log(np.maximum(q,1e-12)),axis=1).mean()),
                **empirical_metrics(p['candidates'],p['q'],y,norm['axes'])))
    summaries,comparisons=paired_summary(rows,c,split)
    write_csv(output/'metrics.csv',rows);write_csv(output/'comparisons.csv',comparisons)
    return dict(velocity_vs_controls=summaries,arms_vs_no_graph=summarize_pairs(rows,'arm','no_graph'),
        means={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm]))
            for key in ('energy','mean_mse','variance_ratio','log_variance_error','effective_experts','mean_cell_entropy')}
            for arm in graph.ARMS+ARMS})


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
    plan=json.loads(Path(configuration).read_text());parent_plan=json.loads(Path(plan['base_configuration']).read_text())
    if tuple(plan['arms'])!=ARMS or plan['gpu']!=0:raise ValueError('Unregistered null grid')
    if plan['graph_weight_normalized']!=parent_plan['graph_weight_normalized']:raise ValueError('Fixed graph weight changed')
    wait_for_free_gpu(0,time.monotonic()+600,lambda s:print(json.dumps(s),flush=True));enforce_gpu_policy('cuda')
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from cellflow.utils import match_linear
    import functools
    from w15i_program_router_followups import cell_programs
    from w7_router_scaling import write_csv
    parent=Path(plan['parent']);c=json.loads(Path(parent_plan['base_configuration']).read_text())
    if c['seeds']!=plan['model_seeds'] or c['steps']!=plan['steps']:raise ValueError('Original model budget changed')
    inputs=[Path(configuration),Path(plan['base_configuration']),Path(parent_plan['base_configuration']),Path(__file__),
        ROOT/'PLANS/w15p_velocity_graph_controls_20261022.md',HERE/'w15o_velocity_neighborhood.py',
        HERE/'w15n_velocity_auxiliary.py',HERE/'w15h_traxler_program_router.py',HERE/'w15i_program_router_followups.py',
        HERE/'w15m_velocity_discrimination.py',HERE/'w15a_cellflow_moe.py',HERE/'w15a_cellflow_moe_official.py',
        HERE/'w15a_cellflow_moe_real_predict.py',HERE/'w15c_renge_l3.py',HERE/'w15a_cellflow_adapter.py',
        HERE/'w15a_cellflow_train.py',HERE/'w4_artifacts.py',parent/'whole_grid_frozen.json',parent/'training_programs.npz',
        Path(c['split']),Path(c['conditions']),Path(c['fold'])/'train_source.npz',Path(c['fold'])/'train_target.npz',
        Path(c['fold'])/'validation_source.npz',Path(c['fold'])/'transform.npz']
    run=Artifacts(output,seed=c['seeds'],inputs=inputs);deadline=time.monotonic()+plan['maximum_hours']*3600
    try:
        training,heldout,norm,split,_=base.prepare(c)
        saved=base.read_npz(parent/'training_programs.npz')
        for key in norm:np.testing.assert_array_equal(norm[key],saved[key])
        training,_=cell_programs(training,norm,parent_plan['shuffle_seed'])
        graphs,mappings=make_controls(training,norm,plan,parent_plan)
        np.savez_compressed(run.directory/'source_graphs.npz',**{f'{label}__{arm}':edge for label,group in graphs.items() for arm,edge in group.items()})
        records=copy.deepcopy(json.loads((parent/'whole_grid_frozen.json').read_text())['predictions'])
        graph.verify_grid(records,c,split)
        run.json('study_manifest.json',dict(plan=plan,original_graph_plan=parent_plan,train_KOs=split['train'],
            validation_KOs=split['val'],reserved_test_KOs=split['test'],new_fits=9,reused_fits=15,
            test_targets_read=False,local_source_permutations=mappings))
        model=base.create_model(c,training[0]['z'].shape[1],0,training[0]['embedding'])
        matcher=functools.partial(match_linear,epsilon=c['ot_epsilon'],scale_cost='mean',max_iterations=c['ot_max_iterations'])
        step=graph.make_step(model,matcher);integrate=base.make_candidate_integrator(model);summaries=[]
        for seed in c['seeds']:
            expected=json.loads((parent/'fits'/f'seed{seed}'/'velocity_graph'/'summary.json').read_text())['initial_parameters_sha256']
            for arm in ARMS:
                if time.monotonic()>deadline:raise TimeoutError('Null followup deadline exceeded')
                directory=run.directory/'fits'/f'seed{seed}'/arm
                fit_run=Artifacts(directory,seed=seed,inputs=[Path(configuration),Path(__file__)])
                run.json('status.json',dict(status='training',seed=seed,arm=arm,completed_fits=len(summaries),total_fits=9))
                params,initial,history=graph.fit(model,step,training,graphs,c,parent_plan,arm,seed,
                    lambda row:fit_run.json('status.json',dict(status='training',**row)))
                initial_hash=hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if initial_hash!=expected:raise ValueError('Matched parent initialization changed')
                checkpoint=directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(params))
                params=serialization.from_bytes(initial,checkpoint.read_bytes())
                for source in heldout:
                    prediction=base.predict(model,integrate,params,graph.deployment_source(source,c['programs']),c)
                    path=directory/(source['label']+'.npz');np.savez_compressed(path,**prediction)
                    records.append(dict(KO=source['label'],seed=seed,arm=arm,path=str(path),sha256=sha256(path),
                        model=str(checkpoint),model_sha256=sha256(checkpoint),explicit_velocity_input=False,heldout_future_used=False))
                fit_run.json('training_history.json',dict(history=history))
                summary=dict(status='complete',seed=seed,arm=arm,steps=c['steps'],
                    parameters=sum(v.size for v in jax.tree_util.tree_leaves(params)),initial_parameters_sha256=initial_hash,
                    graph_js_first=history[0]['graph_js'],graph_js_last=history[-1]['graph_js'])
                fit_run.json('summary.json',summary);fit_run.json('status.json',dict(status='complete'));summaries.append(summary)
                print(json.dumps(dict(status='graph_null_fit_complete',seed=seed,arm=arm,completed=len(summaries),total=9)),flush=True)
        validate_grid(records,c,split)
        run.json('whole_grid_frozen.json',dict(predictions=records,validation_targets_read_before_freeze=False,test_targets_read=False))
        run.json('status.json',dict(status='evaluating',completed_fits=9,total_fits=9))
        results=evaluate(run.directory,c,split,norm,records)
        consistent=lambda controls:all(next(r for r in results['velocity_vs_controls']
            if r['control']==control)['all_seeds_positive'] for control in controls)
        null_consistency=consistent(('mean_three_global_shuffles','S_local_shuffle'))
        original_rule=consistent(('no_graph','state_graph','shuffled_velocity_graph','U_graph'))
        result=dict(status='VELOCITY_GRAPH_NULL_STABILITY_COMPLETE',new_fits=9,reused_fits=15,predictions=len(records),
            parameters=summaries[0]['parameters'],fit_diagnostics=summaries,
            new_null_contrasts_all_seeds_positive=null_consistency,original_W15O_stability_rule_satisfied=original_rule,
            test_targets_read=False,GFG=False,JVP=False,velocity_arrow_claim=False,**results)
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Velocity graph null stability\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_fits=9,total_fits=9))
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
