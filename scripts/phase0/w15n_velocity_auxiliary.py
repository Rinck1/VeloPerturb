"""Source-only supervised router representation; deployment has no velocity input.

Four matched arms, one fixed loss weight. The training-only linear head is
removed before original S/U-router and independent-expert ODE inference.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARMS=('no_auxiliary','velocity_auxiliary','shuffled_velocity_auxiliary','U_program_auxiliary')


def auxiliary_model(base_model, programs):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from w15a_cellflow_moe import KineticMoE, KineticEncoder, ExpertHead

    class SupervisedRouter(nn.Module):
        experts: int
        width: int
        programs: int

        @nn.compact
        def __call__(self, source, encoded, *, include_auxiliary=False):
            hidden=nn.silu(nn.Dense(self.width,name='hidden')(jnp.concatenate((source,encoded),axis=-1)))
            q=jax.nn.softmax(nn.Dense(self.experts,name='logits')(hidden),axis=-1)
            if include_auxiliary:
                return q,nn.Dense(self.programs,name='auxiliary')(hidden)
            return q

    class AuxiliaryMoE(KineticMoE):
        programs: int = 5

        def setup(self):
            c=self.config
            self.kinetic_encoder=KineticEncoder(c.kinetic_width,c.kinetic_embedding)
            self.router=SupervisedRouter(c.experts,c.router_width,self.programs)
            self.expert_heads=tuple(ExpertHead(c.dimensions,c.expert_width,name=f'expert_{k}') for k in range(c.experts))

        def auxiliary_prediction(self, source, kinetic_input):
            _,prediction=self.router(source,self.kinetic_encoder(kinetic_input),include_auxiliary=True)
            return prediction

        def __call__(self, source, kinetic_input, moving_state, tau, condition):
            # Initialize shared original parameters in their original call order;
            # initialize the new head last, preserving the deployed initialization.
            result=super().__call__(source,kinetic_input,moving_state,tau,condition)
            result['auxiliary']=self.auxiliary_prediction(source,kinetic_input)
            return result

    return AuxiliaryMoE(backbone=base_model.backbone,config=base_model.config,programs=programs)


def strip_auxiliary(params):
    result=copy.deepcopy(params)
    del result['router']['auxiliary']
    return result


def shared_input(group, programs):
    """No velocity or program field is inspected, in training or deployment."""
    u=np.asarray(group['U_normalized'],dtype='float32')
    return np.concatenate((u,np.zeros((len(u),programs),dtype='float32')),axis=1)


def deployment_source(group, programs):
    if 'target' in group:raise ValueError('Deployment cannot contain a future target')
    keys=('label','z','embedding','cell_ids','time_hours','score_indices')
    result={key:group[key] for key in keys}
    result['kinetic_input']=shared_input(group,programs)
    return result


def supervision(group, arm, programs):
    if arm=='no_auxiliary':return np.zeros((len(group['z']),programs),dtype='float32')
    key={'velocity_auxiliary':'velocity_program','shuffled_velocity_auxiliary':'shuffled_velocity_program',
         'U_program_auxiliary':'U_program'}[arm]
    value=np.asarray(group[key],dtype='float32')
    if value.shape!=(len(group['z']),programs) or not np.isfinite(value).all():raise ValueError('Invalid source supervision')
    return value


def loss_on_paired(model, params, source, kinetic_input, target, auxiliary_target, tau, condition, auxiliary_weight):
    import jax.numpy as jnp
    from w15a_cellflow_moe import flow_matching_loss
    u=target-source;moving=source+tau*u
    result=model.apply(dict(params=params),source,kinetic_input,moving,tau,condition)
    fm,metrics=flow_matching_loss(result['q'],result['velocities'],u,model.config.balance_coefficient)
    mse=jnp.mean(jnp.square(result['auxiliary']-auxiliary_target))
    contribution=source.shape[1]*auxiliary_weight*mse
    return fm+contribution,dict(metrics,fm_loss=fm,auxiliary_mse=mse,auxiliary_contribution=contribution)


def make_step(model, match_fn):
    import jax
    import jax.numpy as jnp
    from ott.solvers import utils as solver_utils
    from w15a_cellflow_moe import align_ot_indices_jit

    @jax.jit
    def step(state,rng,source,target,kinetic_input,auxiliary_target,condition,auxiliary_weight):
        pair_key,tau_key=jax.random.split(rng)
        si,ti=solver_utils.sample_joint(pair_key,match_fn(source,target))
        paired=align_ot_indices_jit(source,kinetic_input,target,si,ti,condition=condition)
        tau=jax.random.uniform(tau_key,(len(si),1),minval=0.,maxval=1.)
        def objective(params):
            return loss_on_paired(model,params,paired.source_x,paired.kinetic_input,paired.target_x,
                auxiliary_target[si],tau,paired.condition,auxiliary_weight)
        (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(state.params)
        norm=lambda tree:jnp.sqrt(sum(jnp.square(v).sum() for v in jax.tree_util.tree_leaves(tree)))
        new=state.apply_gradients(grads=gradient)
        return new,dict(metrics,loss=loss,router_gradient_norm=norm(gradient['router']),
            auxiliary_head_gradient_norm=norm(gradient['router']['auxiliary']),
            kinetic_gradient_norm=norm(gradient['kinetic_encoder']),source_indices=si,target_indices=ti)
    return step


def fit(model, step, training, c, arm, seed, weight, callback):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    import w15h_traxler_program_router as base
    first=training[0];x=first['z'][:1]
    initial=model.init(jax.random.PRNGKey(seed),jnp.asarray(x),jnp.asarray(shared_input(first,c['programs'])[:1]),
        jnp.asarray(x),.5,base.condition(first,1,c))['params']
    state=TrainState.create(apply_fn=model.apply,params=initial,tx=optax.adam(c['learning_rate']))
    groups=[dict(g,kinetic_input=shared_input(g,c['programs']),auxiliary_target=supervision(g,arm,c['programs'])) for g in training]
    rng=np.random.default_rng(seed);history=[];started=time.monotonic()
    effective_weight=0. if arm=='no_auxiliary' else weight
    for update in range(1,c['steps']+1):
        g=groups[int(rng.integers(len(groups)))];n=c['batch_size']
        si=rng.choice(len(g['z']),n,replace=len(g['z'])<n);ti=rng.choice(len(g['target']),n,replace=len(g['target'])<n)
        state,metrics=step(state,jax.random.fold_in(jax.random.PRNGKey(seed),update),
            jnp.asarray(g['z'][si]),jnp.asarray(g['target'][ti]),jnp.asarray(g['kinetic_input'][si]),
            jnp.asarray(g['auxiliary_target'][si]),base.condition(g,n,c),jnp.asarray(effective_weight,dtype=jnp.float32))
        if update==1 or update%c['log_interval']==0 or update==c['steps']:
            values={k:float(v) for k,v in metrics.items() if k not in ('source_indices','target_indices')}
            if not np.isfinite(list(values.values())).all():raise FloatingPointError('Nonfinite auxiliary training')
            row=dict(step=update,elapsed_seconds=time.monotonic()-started,**values);history.append(row);callback(row)
    return state.params,initial,history


def verify_grid(records,c,split):
    expected={(s,a,ko) for s in c['seeds'] for a in ARMS for ko in split['val']}
    if len(records)!=len(expected) or {(r['seed'],r['arm'],r['KO']) for r in records}!=expected:
        raise ValueError('Freeze the entire auxiliary grid before scoring')
    for r in records:
        if sha256(r['path'])!=r['sha256'] or sha256(r['model'])!=r['model_sha256']:
            raise ValueError('Frozen auxiliary artifact changed')


def evaluate(output,c,split,norm,records):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics,summarize_pairs
    from w7_router_scaling import write_csv
    verify_grid(records,c,split)
    targets,metadata=base.load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
    if metadata.get('day')!=c['target_hours'] or set(targets['conditions'])!=set(split['val']):
        raise ValueError('Wrong validation target identity')
    rows=[]
    for ko in sorted(split['val']):
        y=targets['z'][targets['conditions']==ko]
        y=y[np.random.default_rng(c['evaluation_seed']).choice(len(y),min(len(y),c['evaluation_cells']),replace=False)]
        source_ids=None
        for r in [r for r in records if r['KO']==ko]:
            prediction=base.read_npz(r['path'])
            if source_ids is None:source_ids=prediction['source_ids']
            else:np.testing.assert_array_equal(source_ids,prediction['source_ids'])
            rows.append(dict(KO=ko,seed=r['seed'],arm=r['arm'],source_cells=len(prediction['q']),target_cells=len(y),
                **empirical_metrics(prediction['candidates'],prediction['q'],y,norm['axes'])))
    write_csv(output/'metrics.csv',rows)
    summaries=summarize_pairs(rows,'arm','no_auxiliary')
    by={(r['KO'],r['seed'],r['arm']):r for r in rows}
    comparisons=[]
    for real in [r for r in rows if r['arm']=='velocity_auxiliary']:
        for control in ARMS:
            if control=='velocity_auxiliary':continue
            other=by[(real['KO'],real['seed'],control)]
            comparisons.append(dict(KO=real['KO'],seed=real['seed'],control=control,
                energy_gain=1-real['energy']/other['energy'],mean_mse_gain=1-real['mean_mse']/max(other['mean_mse'],1e-12)))
    write_csv(output/'comparisons.csv',comparisons)
    primary=[]
    for control in ARMS:
        if control=='velocity_auxiliary':continue
        selected=[r for r in comparisons if r['control']==control]
        per_seed={str(s):float(np.mean([r['energy_gain'] for r in selected if r['seed']==s])) for s in c['seeds']}
        per_ko={ko:float(np.mean([r['energy_gain'] for r in selected if r['KO']==ko])) for ko in split['val']}
        primary.append(dict(control=control,energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
            winning_KOs=sum(v>0 for v in per_ko.values()),by_seed=per_seed,by_KO=per_ko))
    return dict(arms_vs_no_auxiliary=summaries,velocity_vs_controls=primary,
        means={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm]))
            for key in ('energy','mean_mse','variance_ratio','log_variance_error')} for arm in ARMS})


def launch(configuration,output,job_directory):
    from veloroute.artifacts import save_json
    from veloroute.gpu_policy import gpu_uuids,ALLOWED_PHYSICAL_GPUS,RESERVED_PHYSICAL_GPUS,MAX_VISIBLE_GPUS
    from w15i_followup_queue import worker_environment
    directory=Path(job_directory).resolve();output=Path(output).resolve()
    if directory.exists() or output.exists() or not directory.is_relative_to(ROOT/'outputs'):
        raise ValueError('New output and project-local job directory required')
    directory.mkdir(parents=True)
    env=worker_environment(output.parent/'runtime_gpu0')
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
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
    if plan['gpu']!=0 or tuple(plan['arms'])!=ARMS:raise ValueError('Unregistered four-arm GPU0 grid')
    wait_for_free_gpu(0,time.monotonic()+600,lambda s:print(json.dumps(s),flush=True));enforce_gpu_policy('cuda')
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from cellflow.utils import match_linear
    from w15i_program_router_followups import cell_programs
    c=json.loads(Path(plan['base_configuration']).read_text());parent=Path(plan['parent'])
    if c['seeds']!=plan['model_seeds'] or c['steps']!=plan['steps']:raise ValueError('Parent budget changed')
    inputs=[Path(configuration),Path(plan['base_configuration']),Path(__file__),
        HERE/'w15h_traxler_program_router.py',HERE/'w15i_program_router_followups.py',HERE/'w15m_velocity_discrimination.py',
        HERE/'w15a_cellflow_moe.py',HERE/'w15a_cellflow_moe_official.py',HERE/'w15a_cellflow_moe_real_predict.py',
        HERE/'w15c_renge_l3.py',HERE/'w15a_cellflow_adapter.py',HERE/'w15a_cellflow_train.py',HERE/'w4_artifacts.py',
        Path(c['split']),Path(c['conditions']),parent/'training_programs.npz',parent/'frozen_grid.json',
        Path(c['fold'])/'train_source.npz',Path(c['fold'])/'train_target.npz',Path(c['fold'])/'validation_source.npz',
        Path(c['fold'])/'transform.npz']
    run=Artifacts(output,seed=c['seeds'],inputs=inputs);deadline=time.monotonic()+plan['maximum_hours']*3600
    try:
        training,heldout,norm,split,_=base.prepare(c)
        saved=base.read_npz(parent/'training_programs.npz')
        for key in norm:np.testing.assert_array_equal(norm[key],saved[key])
        training,mapping=cell_programs(training,norm,plan['shuffle_seed'])
        np.savez_compressed(run.directory/'training_programs.npz',**norm)
        run.json('study_manifest.json',dict(plan=plan,config=c,train_KOs=split['train'],validation_KOs=split['val'],
            reserved_test_KOs=split['test'],planned_fits=12,training_shuffle_mapping=mapping,
            velocity_never_a_forward_input=True,auxiliary_head_discarded_before_prediction=True,test_targets_read=False))
        deployed=base.create_model(c,training[0]['z'].shape[1],0,training[0]['embedding'])
        model=auxiliary_model(deployed,c['programs'])
        match_fn=functools.partial(match_linear,epsilon=c['ot_epsilon'],scale_cost='mean',max_iterations=c['ot_max_iterations'])
        step=make_step(model,match_fn);integrate=base.make_candidate_integrator(deployed)
        records=[];initializations={};fit_summaries=[]
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic()>deadline:raise TimeoutError('Auxiliary experiment deadline exceeded')
                directory=run.directory/'fits'/f'seed{seed}'/arm
                fit_run=Artifacts(directory,seed=seed,inputs=[Path(configuration),Path(__file__)])
                run.json('status.json',dict(status='training',arm=arm,seed=seed,completed_fits=len(fit_summaries),total_fits=12))
                params,initial,history=fit(model,step,training,c,arm,seed,plan['auxiliary_weight_normalized'],
                    lambda r:fit_run.json('status.json',dict(status='training',**r)))
                initial_hash=hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if seed in initializations and initializations[seed]!=initial_hash:raise ValueError('Four-arm initialization mismatch')
                initializations[seed]=initial_hash
                parent_initial=json.loads((parent/'fits'/f'seed{seed}'/'S_U_velocity_program'/'summary.json').read_text())['initial_parameters_sha256']
                if hashlib.sha256(serialization.to_bytes(strip_auxiliary(initial))).hexdigest()!=parent_initial:
                    raise ValueError('Original deployed initialization changed')
                (directory/'training_model.msgpack').write_bytes(serialization.to_bytes(params))
                deploy_params=strip_auxiliary(params);deploy_initial=strip_auxiliary(initial)
                checkpoint=directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(deploy_params))
                deploy_params=serialization.from_bytes(deploy_initial,checkpoint.read_bytes())
                for source in heldout:
                    clean_source=deployment_source(source,c['programs'])
                    prediction=base.predict(deployed,integrate,deploy_params,clean_source,c)
                    path=directory/(source['label']+'.npz');np.savez_compressed(path,**prediction)
                    records.append(dict(KO=source['label'],seed=seed,arm=arm,path=str(path),sha256=sha256(path),
                        model=str(checkpoint),model_sha256=sha256(checkpoint),heldout_future_used=False,
                        explicit_velocity_input=False,auxiliary_head_present=False))
                train_count=sum(v.size for v in jax.tree_util.tree_leaves(params));deploy_count=sum(v.size for v in jax.tree_util.tree_leaves(deploy_params))
                fit_run.json('training_history.json',dict(history=history))
                summary=dict(status='complete',seed=seed,arm=arm,steps=c['steps'],training_parameters=train_count,
                    deployment_parameters=deploy_count,initial_parameters_sha256=initial_hash,
                    auxiliary_head_gradient_first=history[0]['auxiliary_head_gradient_norm'],
                    auxiliary_mse_first=history[0]['auxiliary_mse'],auxiliary_mse_last=history[-1]['auxiliary_mse'])
                fit_run.json('summary.json',summary);fit_run.json('status.json',dict(status='complete'));fit_summaries.append(summary)
                print(json.dumps(dict(status='auxiliary_fit_complete',seed=seed,arm=arm,completed=len(fit_summaries),total=12)),flush=True)
        verify_grid(records,c,split)
        run.json('whole_grid_frozen.json',dict(predictions=records,validation_targets_read_before_freeze=False,
            test_targets_read=False,explicit_velocity_at_inference=False,auxiliary_head_at_inference=False))
        run.json('status.json',dict(status='evaluating',completed_fits=12,total_fits=12))
        results=evaluate(run.directory,c,split,norm,records)
        result=dict(status='VELOCITY_TRAIN_ONLY_AUXILIARY_COMPLETE',new_fits=12,seeds=c['seeds'],
            training_parameters=fit_summaries[0]['training_parameters'],deployment_parameters=fit_summaries[0]['deployment_parameters'],
            train_KOs=len(training),validation_KOs=len(heldout),auxiliary_weight_normalized=plan['auxiliary_weight_normalized'],
            velocity_is_auxiliary_target_only=True,explicit_velocity_at_inference=False,auxiliary_head_at_inference=False,
            GFG=False,JVP=False,test_targets_read=False,fit_diagnostics=fit_summaries,**results)
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Training-only velocity auxiliary\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_fits=12,total_fits=12))
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
