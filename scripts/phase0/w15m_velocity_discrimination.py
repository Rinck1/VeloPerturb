"""Small matched representation tests and frozen direction diagnostics.

No GFG, JVP, new split, model selection or reserved test targets. Heavy imports
are deferred until the GPU worker; feature and geometry tests are CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NEW_VARIANTS = ('unsigned_direction', 'magnitude_only', 'direction_and_magnitude')
PARENT_ARMS = ('S_U_router', 'S_U_velocity_program', 'S_U_shuffled_velocity_program', 'S_U_U_program')
INTERVENTIONS = ('original', 'renormalize', 'zero', 'reverse', 'unsigned', 'U_direction',
                 'cohort_direction', 'shuffle0', 'shuffle1', 'shuffle2', 'S_matched_shuffle',
                 'cohort_q', 'uniform_q')


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def project_velocity(group, norm):
    return (np.asarray(group['velocity'], dtype=float)-norm['velocity_reference']) @ norm['axes'].T


def fit_source_calibration(training, norm):
    if not training: raise ValueError('Training sources required')
    projected = np.concatenate([project_velocity(g, norm) for g in training])
    magnitude = np.log1p(np.linalg.norm(projected, axis=1))
    states = np.concatenate([g['z'] for g in training]).astype(float)
    return dict(magnitude_mean=np.asarray(magnitude.mean()),
                magnitude_scale=np.asarray(max(magnitude.std(), 1e-5)),
                projected_RMS=np.asarray(max(np.sqrt(np.mean(np.sum(projected**2, axis=1))), 1e-8)),
                S_mean=states.mean(0), S_scale=np.maximum(states.std(0), 1e-5))


def representation(group, norm, calibration, variant):
    projected = project_velocity(group, norm)
    if variant == 'direction': return unit(projected).astype('float32')
    if variant == 'unsigned_direction': return np.abs(unit(projected)).astype('float32')
    if variant == 'direction_and_magnitude': return (projected/calibration['projected_RMS']).astype('float32')
    if variant == 'magnitude_only':
        magnitude = (np.log1p(np.linalg.norm(projected, axis=1))-calibration['magnitude_mean'])/calibration['magnitude_scale']
        return np.repeat(magnitude[:, None], projected.shape[1], axis=1).astype('float32')/np.sqrt(projected.shape[1])
    raise ValueError('Unknown representation')


def with_features(group, features):
    if features.ndim != 2 or len(features) != len(group['z']) or not np.isfinite(features).all():
        raise ValueError('Invalid source features')
    return dict(group, kinetic_input=np.concatenate((group['U_normalized'], features), axis=1).astype('float32'))


def local_derangement(states, seed):
    """Greedy nearest-S pairs; final odd triplet uses a cycle, never a fixed point."""
    states = np.asarray(states, dtype=float)
    n = len(states)
    if n < 2: raise ValueError('At least two cells required')
    remaining = list(np.random.default_rng(seed).permutation(n)); order = np.empty(n, dtype=int)
    while len(remaining) > 3:
        first = remaining.pop()
        nearest = int(np.argmin(np.sum((states[remaining]-states[first])**2, axis=1)))
        second = remaining.pop(nearest); order[first] = second; order[second] = first
    for i, index in enumerate(remaining): order[index] = remaining[(i+1) % len(remaining)]
    if not np.array_equal(np.sort(order), np.arange(n)) or np.any(order == np.arange(n)):
        raise ArithmeticError('Invalid local permutation')
    return order


def intervention_features(group, norm, calibration, name):
    from w15g_response_program_probe import derangement
    direction = np.asarray(group['velocity_program'])
    if name == 'original': return direction
    if name == 'renormalize': return unit(direction).astype('float32')
    if name == 'zero': return np.zeros_like(direction)
    if name == 'reverse': return -direction
    if name == 'unsigned': return np.abs(direction)
    if name == 'U_direction': return group['U_program']
    if name == 'cohort_direction':
        return np.broadcast_to(unit(project_velocity(group, norm).mean(0, keepdims=True)), direction.shape).astype('float32')
    identity = f"{group['label']}|{group['time_hours']}|direction_diagnostic"
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')
    if name.startswith('shuffle'):
        order = derangement(len(direction), seed+int(name[-1]))
    elif name == 'S_matched_shuffle':
        order = local_derangement((group['z']-calibration['S_mean'])/calibration['S_scale'], seed)
    else: raise ValueError('Unknown frozen intervention')
    return direction[order]


def fit_direction_probe(training, heldout, norm):
    """One global scalar per representation; all target dependence is TRAINING only."""
    if any('target' in g for g in heldout): raise ValueError('Probe source contains a future')
    residual = np.stack([np.asarray(g['target']).mean(0)-np.asarray(g['z']).mean(0)-norm['response_mean']
                         for g in training]) @ norm['axes'].T
    def features(groups, kind):
        reference = norm['velocity_reference'] if kind == 'velocity' else norm['U_reference']
        key = 'velocity' if kind == 'velocity' else 'U'
        return unit(np.stack([np.asarray(g[key], dtype=float).mean(0)-reference for g in groups]) @ norm['axes'].T)
    train_v, train_u = features(training, 'velocity'), features(training, 'U')
    value = lambda x: float(np.sum(x*residual)/max(np.sum(x*x), 1e-12))
    beta_v, beta_u = value(train_v), value(train_u)
    coefficients = dict(common_mean_shift=0., positive_velocity=max(beta_v, 0.),
                        positive_reverse=max(-beta_v, 0.), signed_velocity=beta_v, signed_U=beta_u)
    v, u = features(heldout, 'velocity'), features(heldout, 'U')
    base = np.stack([np.asarray(g['z'], dtype=float).mean(0) for g in heldout])+norm['response_mean']
    predictions = {key:base + coefficients[key] * (u if key == 'signed_U' else -v if key == 'positive_reverse' else v) @ norm['axes']
                   for key in coefficients}
    return coefficients, predictions


def empirical_metrics(candidates, q, y, axes, *, distances=None):
    from scipy.spatial.distance import cdist
    x = np.asarray(candidates, dtype=float).reshape(-1, candidates.shape[-1])
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    if q.shape != candidates.shape[:2] or np.any(q < 0) or not np.allclose(q.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid mixture weights')
    w = q.ravel()/len(q); w = w/w.sum()
    if distances is None: distances = cdist(x,x), cdist(x,y).mean(1), cdist(y,y).mean()
    xx, xy, yy = distances
    mean, truth = w @ x, y.mean(0)
    variance = max(float(w @ (x*x).sum(1)-mean@mean), 0.)
    ratio = variance/max(float(y.var(0).sum()), 1e-12)
    return dict(energy=float(2*w@xy-w@xx@w-yy)/np.sqrt(x.shape[1]),
                mean_mse=float(((mean-truth)**2).mean()),
                program_mean_mse=float((((mean-truth)@axes.T)**2).mean()),
                variance_ratio=ratio, log_variance_error=abs(float(np.log(max(ratio, 1e-12)))))


def summarize_pairs(rows, name_key, baseline):
    groups = {(r['KO'], r['seed']):r for r in rows if r[name_key] == baseline}
    output = []
    for name in sorted({r[name_key] for r in rows}):
        selected = [r for r in rows if r[name_key] == name]
        values = [1-r['energy']/groups[(r['KO'],r['seed'])]['energy'] for r in selected]
        by_seed = {str(s):float(np.mean([v for r,v in zip(selected,values) if r['seed']==s])) for s in sorted({r['seed'] for r in selected})}
        by_ko = {ko:float(np.mean([v for r,v in zip(selected,values) if r['KO']==ko])) for ko in sorted({r['KO'] for r in selected})}
        output.append(dict(name=name, baseline=baseline, energy=float(np.mean([r['energy'] for r in selected])),
                           relative_gain=float(np.mean(values)), winning_KOs=sum(v>0 for v in by_ko.values()),
                           by_seed=by_seed, by_KO=by_ko))
    return output


def launch(configuration, output, job_directory, gpu):
    from veloroute.artifacts import save_json
    from veloroute.gpu_policy import ALLOWED_PHYSICAL_GPUS, RESERVED_PHYSICAL_GPUS, MAX_VISIBLE_GPUS, gpu_uuids
    from w15i_followup_queue import worker_environment
    if gpu != 0: raise ValueError('This small queue uses approved GPU0')
    directory = Path(job_directory).resolve(); output = Path(output).resolve()
    if not directory.is_relative_to(ROOT/'outputs') or directory.exists() or output.exists():
        raise ValueError('Use new project-local job and new output directories')
    directory.mkdir(parents=True)
    env = worker_environment(output.parent/'runtime_gpu0')
    job = dict(physical_gpu=gpu, gpu_uuid=gpu_uuids()[gpu], cwd=str(ROOT),
        command=[sys.executable,str(Path(__file__).resolve()),'--configuration',str(Path(configuration).resolve()),'--output',str(output)],
        allowed_gpus=list(ALLOWED_PHYSICAL_GPUS),reserved_gpus=list(RESERVED_PHYSICAL_GPUS),max_visible_gpus=MAX_VISIBLE_GPUS)
    save_json(directory/'job.json',job)
    with (directory/'job.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,str(HERE/'launch_gpu_job.py'),'--worker',str(directory)],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save_json(directory/'launcher.json',dict(pid=child.pid,physical_gpu=gpu,detached=True))
    return dict(status='LAUNCHED',pid=child.pid,physical_gpu=gpu,job_directory=str(directory))


def program(configuration, output):
    from veloroute.gpu_policy import enforce_gpu_policy
    from w15k_traxler_l2_cell import wait_for_free_gpu
    plan = json.loads(Path(configuration).read_text())
    # No GPU array is allocated before the free-device check.
    wait_for_free_gpu(plan['gpu'],time.monotonic()+600,lambda state:print(json.dumps(state),flush=True))
    enforce_gpu_policy('cuda')
    import jax
    import jax.numpy as jnp
    from flax import serialization
    import w15h_traxler_program_router as base
    from w15i_program_router_followups import cell_programs, route_all
    from w7_router_scaling import write_csv
    c = json.loads(Path(plan['base_configuration']).read_text()); parent=Path(plan['parent'])
    if tuple(plan['new_variants']) != NEW_VARIANTS or tuple(plan['frozen_interventions']) != INTERVENTIONS:
        raise ValueError('Unregistered discrimination grid')
    if c['seeds'] != plan['model_seeds'] or c['steps'] != plan['steps']:
        raise ValueError('Matched parent seeds and fixed budget required')
    inputs=[Path(configuration),Path(plan['base_configuration']),Path(__file__),
        HERE/'w15h_traxler_program_router.py',HERE/'w15i_program_router_followups.py',
        HERE/'w15a_cellflow_moe.py',HERE/'w15a_cellflow_moe_official.py',
        HERE/'w15a_cellflow_moe_real_predict.py',HERE/'w15c_renge_l3.py',
        HERE/'w15a_cellflow_adapter.py',HERE/'w15a_cellflow_train.py',HERE/'w15g_response_program_probe.py',
        HERE/'w4_artifacts.py',Path(c['split']),Path(c['conditions']),parent/'frozen_grid.json',
        parent/'training_programs.npz',Path(c['fold'])/'train_source.npz',Path(c['fold'])/'train_target.npz',
        Path(c['fold'])/'validation_source.npz',Path(c['fold'])/'transform.npz']
    run=Artifacts(output,seed=c['seeds'],inputs=inputs); deadline=time.monotonic()+plan['maximum_hours']*3600
    try:
        training,heldout,norm,split,_=base.prepare(c)
        saved=base.read_npz(parent/'training_programs.npz')
        for key in norm: np.testing.assert_array_equal(norm[key],saved[key])
        training,_=cell_programs(training,norm,0); heldout,_=cell_programs(heldout,norm,100000)
        calibration=fit_source_calibration(training,norm)
        np.savez_compressed(run.directory/'source_calibration.npz',**calibration)
        parent_grid=json.loads((parent/'frozen_grid.json').read_text())['predictions']
        expected={(s,a,ko) for s in c['seeds'] for a in PARENT_ARMS for ko in split['val']}
        if {(r['seed'],r['arm'],r['KO']) for r in parent_grid} != expected or len(parent_grid)!=len(expected):
            raise ValueError('Incomplete parent grid')
        for r in parent_grid:
            if sha256(r['path']) != r['sha256'] or sha256(r['model']) != r['model_sha256']:
                raise ValueError('Parent artifact changed')
        run.json('study_manifest.json',dict(plan=plan,config=c,train_KOs=split['train'],validation_KOs=split['val'],
            reserved_test_KOs=split['test'],new_fits=9,reused_fits=12,source_calibration_train_only=True,
            validation_targets_read_before_freeze=False,test_targets_read=False))
        model=base.create_model(c,training[0]['z'].shape[1],0,training[0]['embedding'])
        train_step=base.make_step(model,c); integrate=base.make_candidate_integrator(model)
        # Freeze source-only geometric predictions before opening any validation outcome.
        coefficients,probe=fit_direction_probe(training,heldout,norm)
        probe_path=run.directory/'direction_probe_predictions.npz'
        np.savez_compressed(probe_path,labels=np.array([g['label'] for g in heldout]),**probe)
        run.json('direction_probe_coefficients.json',dict(coefficients=coefficients,fit_scope='training_KOs_only',
            coefficient_units='PCA_displacement_per_training_interval',hours=c['target_hours']-c['source_hours']))
        frozen=list(parent_grid); interventions=[]; parameters=None
        for seed in c['seeds']:
            real=[r for r in parent_grid if r['seed']==seed and r['arm']=='S_U_velocity_program']
            first=with_features(training[0],training[0]['velocity_program'])
            initial=model.init(jax.random.PRNGKey(seed),jnp.asarray(first['z'][:1]),jnp.asarray(first['kinetic_input'][:1]),
                jnp.asarray(first['z'][:1]),.5,base.condition(first,1,c))['params']
            initial_hash=hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
            parent_summary=json.loads((Path(real[0]['model']).parent/'summary.json').read_text())
            if initial_hash!=parent_summary['initial_parameters_sha256']:
                raise ValueError('Matched initialization changed')
            params=serialization.from_bytes(initial,Path(real[0]['model']).read_bytes())
            parameters=sum(p.size for p in jax.tree_util.tree_leaves(params))
            for g in heldout:
                record=next(r for r in real if r['KO']==g['label']); prediction=base.read_npz(record['path'])
                np.testing.assert_array_equal(prediction['source_ids'],g['cell_ids'][g['score_indices']])
                original=route_all(model,params,with_features(g,g['velocity_program']),c)
                np.testing.assert_allclose(original,prediction['full_source_q'],atol=1e-6,rtol=1e-5)
                for name in INTERVENTIONS:
                    if name=='original': q=prediction['q']
                    elif name=='cohort_q':q=np.broadcast_to(prediction['full_source_mean_q'],prediction['q'].shape)
                    elif name=='uniform_q':q=np.full_like(prediction['q'],1./prediction['q'].shape[1])
                    else:q=route_all(model,params,with_features(g,intervention_features(g,norm,calibration,name)),c)[g['score_indices']]
                    path=run.directory/f"frozen_seed{seed}_{g['label']}_{name}.npz"
                    np.savez_compressed(path,q=q,source_ids=prediction['source_ids'])
                    interventions.append(dict(KO=g['label'],seed=seed,intervention=name,path=str(path),sha256=sha256(path),
                        parent_prediction=record['path'],parent_sha256=record['sha256']))
            for variant in NEW_VARIANTS:
                if time.monotonic()>deadline:raise TimeoutError('Small experiment deadline exceeded')
                fit=Artifacts(run.directory/'fits'/f'seed{seed}'/variant,seed=seed,inputs=[Path(configuration),Path(__file__)])
                run.json('status.json',dict(status='training',seed=seed,variant=variant,completed_new_fits=(len(frozen)-len(parent_grid))//len(heldout),total_new_fits=9))
                groups=[with_features(g,representation(g,norm,calibration,variant)) for g in training]
                params,template,history=base.train(model,train_step,groups,c,seed,
                    lambda row:fit.json('status.json',dict(status='training',**row)))
                if hashlib.sha256(serialization.to_bytes(template)).hexdigest()!=initial_hash:
                    raise ValueError('Variant initialization differs')
                checkpoint=fit.directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(params))
                params=serialization.from_bytes(template,checkpoint.read_bytes())
                if sum(p.size for p in jax.tree_util.tree_leaves(params))!=parameters:
                    raise ValueError('Parameter budgets changed')
                fit.json('training_history.json',dict(history=history))
                for g in heldout:
                    prediction=base.predict(model,integrate,params,with_features(g,representation(g,norm,calibration,variant)),c)
                    path=fit.directory/(g['label']+'.npz');np.savez_compressed(path,**prediction)
                    frozen.append(dict(KO=g['label'],seed=seed,arm=variant,path=str(path),sha256=sha256(path),
                        model=str(checkpoint),model_sha256=sha256(checkpoint),heldout_future_used=False))
                fit.json('summary.json',dict(status='complete',variant=variant,seed=seed,steps=c['steps'],parameters=parameters,
                    initial_parameters_sha256=initial_hash,model_sha256=sha256(checkpoint)))
                fit.json('status.json',dict(status='complete'))
                print(json.dumps(dict(status='representation_fit_complete',seed=seed,variant=variant)),flush=True)
        arms=(*PARENT_ARMS,*NEW_VARIANTS)
        expected={(s,a,ko) for s in c['seeds'] for a in arms for ko in split['val']}
        if len(frozen)!=len(expected) or {(r['seed'],r['arm'],r['KO']) for r in frozen}!=expected:
            raise ValueError('Incomplete registered prediction grid')
        if len(interventions)!=len(c['seeds'])*len(heldout)*len(INTERVENTIONS):
            raise ValueError('Incomplete frozen interventions')
        for r in frozen+interventions:
            if sha256(r['path'])!=r['sha256']:raise ValueError('Frozen artifact changed')
        run.json('whole_grid_frozen.json',dict(predictions=frozen,interventions=interventions,
            probe_path=str(probe_path),probe_sha256=sha256(probe_path),validation_targets_read_before_freeze=False,
            test_targets_read=False,new_fits=9,new_ODE_integrations_for_frozen_interventions=0))
        run.json('status.json',dict(status='evaluating',completed_new_fits=9,total_new_fits=9))
        targets,tm=base.load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
        if tm['day']!=c['target_hours'] or set(targets['conditions'])!=set(split['val']):
            raise ValueError('Wrong development target cohort')
        from scipy.spatial.distance import cdist
        y_by={}
        for ko in split['val']:
            y=targets['z'][targets['conditions']==ko]
            y_by[ko]=y[np.random.default_rng(c['evaluation_seed']).choice(len(y),min(len(y),c['evaluation_cells']),replace=False)]
        rows=[];mechanism=[];reference={}
        with (parent/'metrics.csv').open() as stream:
            parent_metrics={(r['KO'],int(r['seed']),r['arm']):float(r['energy']) for r in csv.DictReader(stream)}
        for r in frozen:
            prediction=base.read_npz(r['path']);y=y_by[r['KO']]
            row=dict(KO=r['KO'],seed=r['seed'],arm=r['arm'],**empirical_metrics(prediction['candidates'],prediction['q'],y,norm['axes']))
            rows.append(row)
            if r['arm'] in PARENT_ARMS:
                reference[(r['KO'],r['seed'],r['arm'])]=abs(row['energy']-parent_metrics[(r['KO'],r['seed'],r['arm'])])
            if r['arm']=='S_U_velocity_program':
                x=prediction['candidates'].astype(float).reshape(-1,prediction['candidates'].shape[-1]);y=y.astype(float)
                distances=cdist(x,x),cdist(x,y).mean(1),cdist(y,y).mean()
                for item in interventions:
                    if (item['KO'],item['seed'])!=(r['KO'],r['seed']):continue
                    q=base.read_npz(item['path'])['q']
                    mechanism.append(dict(KO=r['KO'],seed=r['seed'],intervention=item['intervention'],
                        **empirical_metrics(prediction['candidates'],q,y,norm['axes'],distances=distances)))
        if max(reference.values())>1e-7:raise ArithmeticError('Parent Energy reproduction failed')
        write_csv(run.directory/'representation_metrics.csv',rows);write_csv(run.directory/'frozen_metrics.csv',mechanism)
        probe_rows=[]
        # Full observed KO means, unlike the capped distribution metric above.
        for index,g in enumerate(heldout):
            truth=targets['z'][targets['conditions']==g['label']].astype(float).mean(0)
            for name,prediction in probe.items():
                residual=prediction[index]-truth
                probe_rows.append(dict(KO=g['label'],probe=name,mean_mse=float(np.mean(residual**2)),
                    program_mean_mse=float(np.mean((residual@norm['axes'].T)**2))))
        write_csv(run.directory/'direction_probe_metrics.csv',probe_rows)
        representation_summary=summarize_pairs(rows,'arm','S_U_velocity_program')
        mechanism_summary=summarize_pairs(mechanism,'intervention','original')
        base_probe={r['KO']:r for r in probe_rows if r['probe']=='common_mean_shift'}
        probe_summary={name:dict(coefficient=coefficients[name],mean_mse=float(np.mean([r['mean_mse'] for r in probe_rows if r['probe']==name])),
            relative_gain=float(np.mean([1-r['mean_mse']/max(base_probe[r['KO']]['mean_mse'],1e-12) for r in probe_rows if r['probe']==name])))
            for name in coefficients}
        write_csv(run.directory/'representation_summary.csv',[{k:v for k,v in r.items() if k not in ('by_seed','by_KO')} for r in representation_summary])
        write_csv(run.directory/'frozen_summary.csv',[{k:v for k,v in r.items() if k not in ('by_seed','by_KO')} for r in mechanism_summary])
        result=dict(status='VELOCITY_DISCRIMINATION_COMPLETE',new_fits=9,parameters=parameters,seeds=c['seeds'],
            train_KOs=len(training),validation_KOs=len(heldout),test_targets_read=False,GFG=False,JVP=False,
            parent_energy_reproduction_max_abs_error=max(reference.values()),representations=representation_summary,
            frozen_interventions=mechanism_summary,direction_probe=probe_summary,
            interpretation='exploratory_development_representation_and_inference_sensitivity_not_causal_fate_or_GFG_validation')
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Velocity discrimination\n\nPositive relative gain is better than the stated baseline.\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',completed_new_fits=9,total_new_fits=9))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--job-directory');parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.launch:
        if not args.job_directory:parser.error('--launch requires --job-directory')
        result=launch(args.configuration,args.output,args.job_directory,args.gpu)
    else:result=program(args.configuration,args.output)
    print(json.dumps(result,indent=2),flush=True)
