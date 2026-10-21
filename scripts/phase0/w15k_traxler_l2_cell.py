"""Matched native CellFlow L2 and two extra cell-level L3 shuffle controls.

Use the identical Traxler 6->24h source transform, KO split, cell programs,
public ESM conditions and scoring IDs. L2 receives frozen source S and U/program
features as conditions; L3 receives the same information at its router. L2's
width is chosen solely by parameter count, never by held-out outcomes.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts, sha256
from veloroute.artifacts import save_json
from veloroute.gpu_policy import (ALLOWED_PHYSICAL_GPUS, MAX_VISIBLE_GPUS,
    RESERVED_PHYSICAL_GPUS, enforce_gpu_policy, gpu_uuids)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
L2_ARMS = ('CF_S_U', 'CF_S_U_velocity_program', 'CF_S_U_shuffled_velocity_program', 'CF_S_U_U_program')
L3_ARMS = ('S_U_router', 'S_U_velocity_program', 'S_U_shuffled_velocity_program', 'S_U_U_program')
ARM_MAP = dict(zip(L2_ARMS, L3_ARMS))


def wait_for_free_gpu(gpu, deadline, callback):
    uuid = gpu_uuids()[gpu]
    while True:
        memory = int(subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15).strip())
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()
        pids = [int(row.split(',')[1]) for row in rows if row.split(',')[0].strip() == uuid]
        if not pids and memory <= 256: return
        callback(dict(status='waiting_for_gpu', physical_gpu=gpu, memory_mib=memory, active_pids=pids))
        if time.monotonic() > deadline: raise TimeoutError('Approved GPU wait exceeded deadline')
        time.sleep(30)


def launch(configuration, output, lane, gpu, job_directory):
    from w15i_followup_queue import worker_environment
    if (lane, gpu) not in (('l2', 0), ('cell_shuffles', 7)):
        raise ValueError('This queue assigns L2 to GPU0 and cell shuffle controls to GPU7')
    directory = (ROOT/job_directory).resolve()
    if not directory.is_relative_to(ROOT/'outputs') or Path(output).exists():
        raise ValueError('Use a new output and project-local job directory')
    directory.mkdir(parents=True, exist_ok=False)
    env = worker_environment(Path(output).parent/f'runtime_gpu{gpu}')
    job = dict(physical_gpu=gpu, gpu_uuid=gpu_uuids()[gpu],
        command=[sys.executable, str(Path(__file__).resolve()), '--configuration',
                 str(Path(configuration).resolve()), '--output', str(Path(output).resolve()),
                 '--lane', lane, '--gpu', str(gpu), '--job-record', str(directory)],
        cwd=str(ROOT), allowed_gpus=list(ALLOWED_PHYSICAL_GPUS),
        reserved_gpus=list(RESERVED_PHYSICAL_GPUS), max_visible_gpus=MAX_VISIBLE_GPUS,
        resource_authorization='GPU0 and explicitly authorized GPU7; do not use GPU6',
        runtime_environment={k:env[k] for k in ('JAX_PLATFORMS', 'LD_LIBRARY_PATH', 'TMPDIR',
            'JAX_COMPILATION_CACHE_DIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH')})
    save_json(directory/'job.json', job)
    with (directory/'job.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, str(HERE/'launch_gpu_job.py'), '--worker', str(directory)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    save_json(directory/'launcher.json', dict(pid=process.pid, physical_gpu=gpu, detached=True))
    return dict(status='LAUNCHED', pid=process.pid, physical_gpu=gpu, lane=lane, job_directory=str(directory))


def prepare_cells(c, permutation=0):
    import numpy as np
    import w15h_traxler_program_router as base
    from w15i_program_router_followups import cell_programs
    training, heldout, norm, split, _ = base.prepare(dict(c, shuffle_seed=permutation))
    training, tp = cell_programs(training, norm, permutation)
    heldout, vp = cell_programs(heldout, norm, permutation+100000)
    return training, heldout, norm, split, dict(train=tp, validation=vp)


def l2_condition(group, arm, c):
    import numpy as np
    import w15h_traxler_program_router as base
    if arm not in ARM_MAP: raise ValueError('Unknown L2 arm')
    result = {k:np.asarray(v, dtype='float32') for k,v in base.condition(group, len(group['z']), c).items()}
    # L3 sees source S in its router. Give L2 that same frozen source state,
    # rather than making it infer S0 from its moving interpolation state.
    result['source_state'] = np.asarray(group['z'], dtype='float32')[:,None,:]
    result['source_rna'] = base.apply_arm(group, ARM_MAP[arm], c)['kinetic_input'][:,None,:]
    return result


def native_field(c, dimensions):
    from w15a_cellflow_adapter import add_cellflow_paths
    add_cellflow_paths()
    from cellflow.networks import ConditionalVelocityField
    layers = {k:[] for k in ('perturbation', 'bio_time', 'bio_delta', 'source_state', 'source_rna')}
    for key in ('source_state', 'source_rna'):
        layers[key] = [dict(layer_type='mlp', dims=(64, c['condition_dim']), act_last_layer=False)]
    return ConditionalVelocityField(output_dim=dimensions, max_combination_length=1,
        condition_embedding_dim=c['condition_dim'], condition_mode='deterministic', regularization=0.,
        pooling=c['pooling'], time_freqs=c['time_freqs'],
        time_encoder_dims=(c['time_width'],)*c['time_depth'],
        hidden_dims=(c['width'],)*c['depth'], decoder_dims=(c['width'],)*c['depth'],
        covariates_not_pooled=('bio_time', 'bio_delta', 'source_state', 'source_rna'), layers_before_pool=layers)


def parameter_count(field, condition, dimensions, embedding_dim):
    import jax
    import jax.numpy as jnp
    shapes = jax.eval_shape(lambda key:field.init(key, t=jnp.zeros((1,1)),
        x_t=jnp.zeros((1,dimensions)), cond={k:jnp.asarray(v[:1]) for k,v in condition.items()},
        encoder_noise=jnp.zeros((1,embedding_dim)), train=False), jax.random.PRNGKey(0))
    return int(sum(value.size for value in jax.tree_util.tree_leaves(shapes['params'])))


def match_width(c, condition, target_parameters, candidates, tolerance):
    rows = []
    dimensions = condition['source_state'].shape[-1]
    for width in candidates:
        count = parameter_count(native_field(dict(c, width=width), dimensions),
                                condition, dimensions, c['condition_dim'])
        rows.append(dict(width=width, parameters=count, relative_gap=abs(count/target_parameters-1)))
    selected = min(rows, key=lambda r:(r['relative_gap'], r['width']))
    if selected['relative_gap'] > tolerance: raise ValueError('No L2 width matches the frozen L3 parameter budget')
    return dict(c, width=selected['width']), dict(target_parameters=target_parameters,
        selected=selected, candidates=rows, selection_uses_outcomes=False)


def create_solver(c, condition, dimensions, seed):
    import jax
    import optax
    from cellflow.solvers import OTFlowMatching
    from cellflow._compat import ConstantNoiseFlow
    from cellflow.utils import match_linear
    field = native_field(c, dimensions)
    match = functools.partial(match_linear, epsilon=c['ot_epsilon'], scale_cost='mean',
                              max_iterations=c['ot_max_iterations'])
    return OTFlowMatching(field, ConstantNoiseFlow(0.), match_fn=match,
        rng=jax.random.PRNGKey(seed), optimizer=optax.adam(c['learning_rate']), conditions=condition, ema=1.)


def predict_l2(solver, source, condition, c):
    import numpy as np
    from w15a_cellflow_adapter import predict_cellflow_tau
    if 'target' in source: raise ValueError('Future not allowed in the L2 prediction input')
    indices = source['score_indices']; pieces = []; batch = c['prediction_batch']
    for start in range(0,len(indices),batch):
        ix = indices[start:start+batch]; n = len(ix)
        z = np.pad(source['z'][ix], ((0,batch-n),(0,0)), mode='edge')
        cond = {k:np.pad(v[ix], ((0,batch-n),(0,0),(0,0)), mode='edge') for k,v in condition.items()}
        pieces.append(predict_cellflow_tau(solver, z, cond, tau_end=1., per_particle=True)[:n])
    x = np.concatenate(pieces)
    if not np.isfinite(x).all(): raise FloatingPointError('Nonfinite L2 ODE prediction')
    return dict(candidates=x[:,None,:].astype('float32'), q=np.ones((len(x),1),dtype='float32'),
        source_ids=source['cell_ids'][indices], source_indices=indices)


def evaluate_l2(root, c, frozen, split, norm, parent):
    import numpy as np
    import w15h_traxler_program_router as base
    from w7_router_scaling import write_csv
    expected = {(s,a,ko) for s in c['seeds'] for a in L2_ARMS for ko in split['val']}
    if len(frozen) != len(expected) or {(r['seed'],r['arm'],r['KO']) for r in frozen} != expected:
        raise ValueError('Freeze every L2 model forecast before scoring')
    for r in frozen:
        if sha256(r['path']) != r['sha256'] or sha256(r['model']) != r['model_sha256']:
            raise ValueError('Frozen L2 artifact changed')
    targets, tm = base.load_pack(Path(c['fold'])/'validation_target.npz', expected_side='target')
    if tm.get('day') != c['target_hours']: raise ValueError('Wrong validation target time')
    parent_grid = json.loads((Path(parent)/'frozen_grid.json').read_text())['predictions']
    rows, comparisons, cross_line = [], [], []
    with (Path(parent)/'metrics.csv').open() as stream:
        reference = {(r['KO'],int(r['seed']),r['arm']):r for r in csv.DictReader(stream)}
    for ko in sorted(split['val']):
        y = targets['z'][targets['conditions']==ko]
        y = y[np.random.default_rng(c['evaluation_seed']).choice(len(y), min(len(y),c['evaluation_cells']), replace=False)]
        for seed in c['seeds']:
            group = {}
            for arm in L2_ARMS:
                r = next(r for r in frozen if (r['KO'],r['seed'],r['arm'])==(ko,seed,arm))
                p = base.read_npz(r['path'])
                old = next(r for r in parent_grid if (r['KO'],r['seed'],r['arm'])==(ko,seed,ARM_MAP[arm]))
                if sha256(old['path']) != old['sha256']: raise ValueError('L3 reference prediction changed')
                np.testing.assert_array_equal(p['source_ids'],base.read_npz(old['path'])['source_ids'])
                metrics = base.mixture_metrics(p['candidates'],p['q'],y,norm['axes'],device='cuda')
                row = dict(KO=ko,seed=seed,arm=arm,source_cells=len(p['q']),target_cells=len(y),**metrics)
                rows.append(row); group[arm]=row
                l3 = reference[(ko,seed,ARM_MAP[arm])]
                cross_line.append(dict(KO=ko,seed=seed,arm=arm,L2_energy=metrics['energy'],
                    L3_energy=float(l3['energy']),L3_gain_over_L2=1-float(l3['energy'])/metrics['energy']))
            for control in L2_ARMS:
                if control == L2_ARMS[1]: continue
                comparisons.append(dict(KO=ko,seed=seed,control=control,
                    energy_gain=1-group[L2_ARMS[1]]['energy']/group[control]['energy'],
                    mean_mse_gain=1-group[L2_ARMS[1]]['mean_mse']/max(group[control]['mean_mse'],1e-12)))
    write_csv(root/'metrics.csv',rows); write_csv(root/'comparisons.csv',comparisons)
    write_csv(root/'L2_L3_comparisons.csv',cross_line)
    fields=('energy','mean_mse','program_mean_mse','variance_ratio','log_variance_error')
    means={a:{m:float(np.mean([r[m] for r in rows if r['arm']==a])) for m in fields} for a in L2_ARMS}
    gains={a:float(np.mean([r['energy_gain'] for r in comparisons if r['control']==a])) for a in L2_ARMS if a!=L2_ARMS[1]}
    return dict(status='TRAXLER_L2_CELL_PLUGIN_COMPLETE',train_KOs=len(split['train']),
        validation_KOs=len(split['val']),reserved_test_KOs=len(split['test']),test_targets_read=False,
        fits=12,means=means,mean_paired_velocity_gains=gains,
        L3_gain_over_L2={a:float(np.mean([r['L3_gain_over_L2'] for r in cross_line if r['arm']==a])) for a in L2_ARMS},
        task='6h_to_24h_18hour_early_response_continuation',GFG=False,JVP=False,
        source_information='identical_S50_U50_and_5_cell_program_slots_as_L3',
        source_condition_location='frozen_source_S_and_U_program_tokens_in_native_CellFlow',
        router=False,experts=1,mean_anchor=False,aggregation='equal_KO_equal_seed',
        independent_confirmation=False,validation_previously_used_for_development=True)


def run_l2(plan, output, inputs):
    import numpy as np
    import jax
    from flax import serialization
    from w15a_cellflow_train import fit_groups
    config=json.loads(Path(plan['base_configuration']).read_text())
    parent=Path(plan['cell_parent'])
    training,heldout,norm,split,mapping=prepare_cells(config)
    saved=np.load(parent/'training_programs.npz',allow_pickle=False)
    for key in norm: np.testing.assert_array_equal(norm[key],saved[key])
    target=json.loads((parent/'fits/seed0/S_U_velocity_program/summary.json').read_text())['parameters']
    run=Artifacts(output,seed=config['seeds'],inputs=inputs)
    try:
        config,match=match_width(config,l2_condition(training[0],L2_ARMS[0],config),target,
                               plan['L2_width_candidates'],plan['parameter_tolerance'])
        run.json('parameter_match.json',match)
        np.savez_compressed(run.directory/'training_programs.npz',**norm)
        run.json('study_manifest.json',dict(config=config,arms=list(L2_ARMS),parameter_match=match,
            train_KOs=sorted(split['train']),validation_KOs=sorted(split['val']),test_KOs=sorted(split['test']),
            shuffle_mapping=mapping,planned_fits=12,test_targets_read=False))
        frozen=[];initializations={};deadline=time.monotonic()+plan['maximum_hours']*3600
        for seed in config['seeds']:
            for arm in L2_ARMS:
                if time.monotonic()>deadline: raise TimeoutError('L2 fixed-grid deadline exceeded')
                fit=Artifacts(run.directory/'fits'/f'seed{seed}'/arm,seed=seed,inputs=[Path(__file__),Path(plan['base_configuration'])])
                groups=[dict(z=g['z'],target=g['target'],condition=l2_condition(g,arm,config)) for g in training]
                solver=create_solver(config,{k:v[:1] for k,v in groups[0]['condition'].items()},training[0]['z'].shape[1],seed)
                initial_hash=hashlib.sha256(serialization.to_bytes(solver.vf_state.params)).hexdigest()
                if seed in initializations and initial_hash!=initializations[seed]: raise ValueError('L2 arm initializations differ')
                initializations[seed]=initial_hash
                run.json('status.json',dict(status='training',completed_models=len(frozen)//len(heldout),total_models=12,seed=seed,arm=arm))
                history=fit_groups(solver,groups,config,seed,
                    callback=lambda row:fit.json('status.json',dict(status='training',**row)))
                checkpoint=fit.directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(solver.vf_state_inference))
                solver.vf_state_inference=serialization.from_bytes(solver.vf_state_inference,checkpoint.read_bytes())
                model_hash=sha256(checkpoint)
                fit.json('training_history.json',dict(history=history))
                for source in heldout:
                    p=predict_l2(solver,source,l2_condition(source,arm,config),config)
                    path=fit.directory/(source['label']+'.npz');np.savez_compressed(path,**p)
                    frozen.append(dict(KO=source['label'],seed=seed,arm=arm,path=str(path.resolve()),sha256=sha256(path),
                        model=str(checkpoint.resolve()),model_sha256=model_hash,heldout_future_used=False))
                fit.json('summary.json',dict(status='complete',parameters=sum(p.size for p in jax.tree_util.tree_leaves(solver.vf_state.params)),
                    seed=seed,arm=arm,width=config['width'],initial_parameters_sha256=initial_hash,steps=config['steps'],
                    native_CellFlow=True,router=False,experts=1,heldout_future_used=False,backend=[str(d) for d in jax.devices()]))
                fit.json('status.json',dict(status='complete'))
                run.json('status.json',dict(status='training',completed_models=len(frozen)//len(heldout),total_models=12,last_seed=seed,last_arm=arm))
                print(json.dumps(dict(status='L2_model_complete',seed=seed,arm=arm,completed_models=len(frozen)//len(heldout))),flush=True)
        run.json('frozen_grid.json',dict(predictions=frozen,validation_targets_read=False,test_targets_read=False))
        result=evaluate_l2(run.directory,config,frozen,split,norm,parent)
        result.update(parameter_match=match,steps_per_fit=config['steps'])
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Matched Traxler L2 vs L3\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_models=12,total_models=12))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


def run_cell_shuffles(plan, output, inputs):
    import numpy as np
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from w7_router_scaling import write_csv
    config=json.loads(Path(plan['base_configuration']).read_text());parent=Path(plan['cell_parent'])
    parent_grid=json.loads((parent/'frozen_grid.json').read_text())['predictions']
    arm=L3_ARMS[2];run=Artifacts(output,seed=config['seeds'],inputs=inputs)
    completed=0;deadline=time.monotonic()+plan['maximum_hours']*3600
    try:
        run.json('study_manifest.json',dict(plan=plan,new_fits=6,parent=str(parent),test_targets_read=False,
            shuffle_scope='within_KO_cells_same_as_cell_parent',permutation_seeds=[0,1,2]))
        for permutation in plan['additional_shuffle_seeds']:
            training,heldout,norm,split,mapping=prepare_cells(config,permutation)
            saved=base.read_npz(parent/'training_programs.npz')
            for key in norm:np.testing.assert_array_equal(norm[key],saved[key])
            part=Artifacts(run.directory/f'permutation{permutation}',seed=config['seeds'],inputs=[Path(__file__)])
            part.json('study_manifest.json',dict(config=config,shuffle_seed=permutation,mapping=mapping))
            frozen=[dict(r) for r in parent_grid if r['arm']!=arm]
            model=base.create_model(config,training[0]['z'].shape[1],0,training[0]['embedding'])
            step=base.make_step(model,config);integrate=base.make_candidate_integrator(model)
            for seed in config['seeds']:
                if time.monotonic()>deadline:raise TimeoutError('Cell shuffle deadline exceeded')
                fit=Artifacts(part.directory/f'seed{seed}',seed=seed,inputs=[Path(__file__)])
                run.json('status.json',dict(status='training',completed_models=completed,total_models=6,permutation=permutation,seed=seed))
                params,initial,history=base.train(model,step,[base.apply_arm(g,arm,config) for g in training],config,seed,
                    lambda row:fit.json('status.json',dict(status='training',**row)))
                initial_hash=hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                ref=json.loads((parent/'fits'/f'seed{seed}'/'S_U_velocity_program'/'summary.json').read_text())
                if initial_hash!=ref['initial_parameters_sha256']:raise ValueError('Cell control initialization differs from parent')
                checkpoint=fit.directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(params))
                params=serialization.from_bytes(initial,checkpoint.read_bytes());model_hash=sha256(checkpoint)
                fit.json('training_history.json',dict(history=history))
                for source in heldout:
                    prediction=base.predict(model,integrate,params,base.apply_arm(source,arm,config),config)
                    path=fit.directory/(source['label']+'.npz');np.savez_compressed(path,**prediction)
                    frozen.append(dict(KO=source['label'],seed=seed,arm=arm,path=str(path.resolve()),sha256=sha256(path),
                        model=str(checkpoint.resolve()),model_sha256=model_hash,heldout_future_used=False))
                fit.json('summary.json',dict(status='complete',steps=config['steps'],initial_parameters_sha256=initial_hash,
                    parameters=sum(p.size for p in jax.tree_util.tree_leaves(params)),permutation=permutation,seed=seed))
                fit.json('status.json',dict(status='complete'));completed+=1
                print(json.dumps(dict(status='cell_shuffle_complete',permutation=permutation,seed=seed,completed_models=completed)),flush=True)
            part.json('frozen_grid.json',dict(predictions=frozen,test_targets_read=False))
            result=base.evaluate(part.directory,config,frozen,split,norm)
            result.update(new_fits=3,reused_parent_models=9,shuffle_seed=permutation,response_feature_scope='single_cell')
            part.json('summary.json',result);part.json('status.json',dict(status='complete'))
        with (parent/'metrics.csv').open() as stream:original=list(csv.DictReader(stream))
        shuffle=[dict(permutation=0,**r) for r in original if r['arm']==arm]
        for permutation in plan['additional_shuffle_seeds']:
            with (run.directory/f'permutation{permutation}'/'metrics.csv').open() as stream:
                shuffle.extend(dict(permutation=permutation,**r) for r in csv.DictReader(stream) if r['arm']==arm)
        write_csv(run.directory/'all_shuffle_metrics.csv',shuffle)
        paired=[]
        for r in original:
            if r['arm']!=L3_ARMS[1]:continue
            controls=[float(s['energy']) for s in shuffle if (s['KO'],s['seed'])==(r['KO'],r['seed'])]
            if len(controls)!=3:raise ValueError('Exactly three fixed cell permutations required')
            null=float(np.mean(controls));paired.append(dict(KO=r['KO'],seed=int(r['seed']),real_energy=float(r['energy']),
                three_shuffle_mean_energy=null,real_gain_vs_three_shuffle=1-float(r['energy'])/null))
        write_csv(run.directory/'three_shuffle_comparisons.csv',paired)
        result=dict(status='TRAXLER_THREE_CELL_SHUFFLES_COMPLETE',new_fits=6,test_targets_read=False,
            permutation_seeds=[0,*plan['additional_shuffle_seeds']],
            real_gain_vs_three_shuffle=float(np.mean([r['real_gain_vs_three_shuffle'] for r in paired])),
            mean_gain_by_seed={str(s):float(np.mean([r['real_gain_vs_three_shuffle'] for r in paired if r['seed']==s])) for s in config['seeds']})
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Three single-cell shuffle controls\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_models=6,total_models=6))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


def program(configuration, output, lane, gpu, job_record=None):
    if enforce_gpu_policy('cuda') != (gpu,):raise ValueError('Visible physical GPU must match the assigned lane')
    plan=json.loads(Path(configuration).read_text())
    wait_for_free_gpu(gpu,time.monotonic()+plan['gpu_wait_hours']*3600,
        lambda value:save_json(Path(job_record)/'queue_status.json',value) if job_record else print(json.dumps(value),flush=True))
    if job_record:save_json(Path(job_record)/'queue_status.json',dict(status='started',physical_gpu=gpu,lane=lane))
    config=json.loads(Path(plan['base_configuration']).read_text());fold=Path(config['fold'])
    inputs=[Path(configuration),Path(plan['base_configuration']),Path(__file__),
        HERE/'w15h_traxler_program_router.py',HERE/'w15i_program_router_followups.py',
        HERE/'w15a_cellflow_adapter.py',HERE/'w15a_cellflow_train.py',HERE/'w15a_cellflow_moe.py',
        HERE/'w15a_cellflow_moe_official.py',HERE/'w15a_cellflow_moe_real_predict.py',
        HERE/'w15c_renge_l3.py',HERE/'w15g_response_program_probe.py',
        Path(plan['cell_parent'])/'frozen_grid.json',Path(plan['cell_parent'])/'training_programs.npz',
        Path(config['split']),Path(config['conditions']),fold/'train_source.npz',fold/'train_target.npz',
        fold/'validation_source.npz',fold/'transform.npz']
    if lane=='l2':return run_l2(plan,output,inputs)
    if lane=='cell_shuffles':return run_cell_shuffles(plan,output,inputs)
    raise ValueError('Unknown queue lane')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--configuration',required=True);p.add_argument('--output',required=True)
    p.add_argument('--lane',choices=('l2','cell_shuffles'),required=True);p.add_argument('--gpu',type=int,choices=(0,7),required=True)
    p.add_argument('--launch-job');p.add_argument('--job-record')
    args=p.parse_args()
    result=(launch(args.configuration,args.output,args.lane,args.gpu,args.launch_job) if args.launch_job else
            program(args.configuration,args.output,args.lane,args.gpu,args.job_record))
    print(json.dumps(result,indent=2))
