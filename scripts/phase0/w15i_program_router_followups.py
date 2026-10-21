"""Sequential, bounded follow-ups after the frozen Traxler K8 grid.

Queue one matched 0->6h cohort-program grid (12 new fits), a 6->24h
single-cell-program grid (9 new fits, reuse 3 identical S+U baselines), and
zero-training interventions on the completed 6->24h cohort-program models.
Do not edit or restart the parent grid or the independent RENGE workers.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import w15h_traxler_program_router as base
from w4_artifacts import Artifacts, sha256
from w7_router_scaling import write_csv
from flax import serialization
import jax
import jax.numpy as jnp


HERE = Path(__file__).resolve().parent
INTERVENTIONS = ('original', 'velocity_zero', 'velocity_reverse',
                 'velocity_cross_KO_shuffle', 'cohort_q', 'uniform_q', 'U_zero')


def wait_for_predecessor(path, deadline, callback):
    while True:
        status = json.loads(Path(path).read_text()) if Path(path).exists() else {}
        if status.get('status') == 'complete': return
        if status.get('status') == 'failed':
            raise RuntimeError('Parent job failed; dependent training not started')
        if time.monotonic() > deadline: raise TimeoutError('Parent dependency deadline exceeded')
        callback(dict(status='waiting_for_predecessor', predecessor=str(path)))
        time.sleep(15)


def cell_programs(collection, normalizer, shuffle_seed):
    """Keep U inputs; change the five program slots from cohort to cell values."""
    result, permutations = [], {}
    for original in collection:
        g = dict(original)
        for key, ref in (('velocity', 'velocity_reference'), ('U', 'U_reference')):
            g[key+'_program'] = base.unit((g[key].astype(float)-normalizer[ref]) @
                                          normalizer['axes'].T).astype('float32')
        identity = '|'.join((g['label'], str(float(g['time_hours'])), 'within_KO_cell_shuffle'))
        identity_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')
        order = base.derangement(len(g['z']), identity_seed+shuffle_seed)
        g['shuffled_velocity_program'] = g['velocity_program'][order].copy()
        permutations[g['label']] = {str(g['cell_ids'][i]): str(g['cell_ids'][order[i]])
                                    for i in range(len(order))}
        result.append(g)
    return result, permutations


def read_baselines(parent, seeds, labels):
    records = json.loads((Path(parent)/'frozen_grid.json').read_text())['predictions']
    selected = [r for r in records if r['arm'] == 'S_U_router']
    expected = {(s, k) for s in seeds for k in labels}
    if len(selected) != len(expected) or {(r['seed'], r['KO']) for r in selected} != expected:
        raise ValueError('Incomplete reusable S+U baseline grid')
    initial = {}
    for seed in seeds:
        record = json.loads((Path(parent)/'fits'/f'seed{seed}'/'S_U_router'/'summary.json').read_text())
        initial[seed] = record['initial_parameters_sha256']
    return copy.deepcopy(selected), initial


def run_case(c, output, *, scope, parent=None, inputs=(), deadline=None):
    training, heldout, norm, split, permutations = base.prepare(c)
    if scope == 'cell':
        training, tp = cell_programs(training, norm, c['shuffle_seed'])
        heldout, vp = cell_programs(heldout, norm, c['shuffle_seed']+100000)
        permutations = dict(train=tp, validation=vp)
    elif scope != 'cohort': raise ValueError('Unknown response-feature scope')
    if parent:
        if scope != 'cell': raise ValueError('Baseline reuse is only planned for the cell follow-up')
        saved = base.read_npz(Path(parent)/'training_programs.npz')
        for key in norm: np.testing.assert_array_equal(norm[key], saved[key])
        frozen, initialization = read_baselines(parent, c['seeds'], split['val'])
    else: frozen, initialization = [], {}
    arms = tuple(a for a in base.ARMS if not parent or a != 'S_U_router')
    planned = len(arms)*len(c['seeds'])
    run = Artifacts(output, seed=c['seeds'], inputs=list(inputs))
    try:
        np.savez_compressed(run.directory/'training_programs.npz', **norm)
        run.json('study_manifest.json', dict(config=c, response_feature_scope=scope,
            train_KOs=sorted(split['train']), validation_KOs=sorted(split['val']),
            reserved_test_KOs=sorted(split['test']), test_targets_read=False,
            new_fits=planned, reused_baseline_fits=len(c['seeds']) if parent else 0,
            reused_baseline_parent=str(parent) if parent else None,
            shuffle_mapping=permutations, validation_previously_used_for_development=True))
        model = base.create_model(c, training[0]['z'].shape[1], 0, training[0]['embedding'])
        step = base.make_step(model, c); integrate = base.make_candidate_integrator(model)
        complete = 0; started = time.monotonic()
        for seed in c['seeds']:
            for arm in arms:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError('Follow-up training deadline exceeded')
                directory = run.directory/'fits'/f'seed{seed}'/arm
                fit = Artifacts(directory, seed=seed, inputs=list(inputs))
                run.json('status.json', dict(status='training', completed_new_models=complete,
                    total_new_models=planned, seed=seed, arm=arm))
                params, initial, history = base.train(model, step,
                    [base.apply_arm(g, arm, c) for g in training], c, seed,
                    lambda row: fit.json('status.json', dict(status='training', **row)))
                initial_hash = hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if seed in initialization and initial_hash != initialization[seed]:
                    raise ValueError('Matched or reused baseline initial parameters differ')
                initialization[seed] = initial_hash
                checkpoint = directory/'model.msgpack'
                checkpoint.write_bytes(serialization.to_bytes(params))
                params = serialization.from_bytes(initial, checkpoint.read_bytes())
                model_hash = sha256(checkpoint)
                fit.json('training_history.json', dict(history=history))
                for source in heldout:
                    prediction = base.predict(model, integrate, params, base.apply_arm(source, arm, c), c)
                    path = directory/(source['label']+'.npz')
                    np.savez_compressed(path, **prediction)
                    frozen.append(dict(KO=source['label'], seed=seed, arm=arm,
                        path=str(path.resolve()), sha256=sha256(path), model=str(checkpoint.resolve()),
                        model_sha256=model_hash, heldout_future_used=False))
                fit.json('summary.json', dict(status='complete', seed=seed, arm=arm,
                    response_feature_scope=scope, steps=c['steps'],
                    parameters=sum(p.size for p in jax.tree_util.tree_leaves(params)),
                    initial_parameters_sha256=initial_hash, model_sha256=model_hash,
                    independent_expert_ODEs=True, heldout_future_used=False))
                fit.json('status.json', dict(status='complete'))
                complete += 1
                run.json('status.json', dict(status='training', completed_new_models=complete,
                    total_new_models=planned, elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(status='followup_model_complete', task=str(output),
                    arm=arm, seed=seed, completed=complete, total=planned)), flush=True)
        run.json('frozen_grid.json', dict(predictions=frozen, validation_targets_read=False,
                                         test_targets_read=False))
        run.json('status.json', dict(status='evaluating', completed_new_models=planned,
                                     total_new_models=planned))
        result = base.evaluate(run.directory, c, frozen, split, norm)
        result.update(status='TRAXLER_PROGRAM_ROUTER_FOLLOWUP_COMPLETE',
            task=f"KO_background_conditioned_{c['source_hours']}to{c['target_hours']}h_response",
            response_feature_scope=scope, new_fits=planned, reused_baseline_fits=3 if parent else 0,
            shuffle_scope='within_KO_cells' if scope == 'cell' else 'across_KOs_within_partition')
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Traxler router follow-up\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete', completed_new_models=planned, total_new_models=planned))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


def route_all(model, params, source, c):
    parts = []; batch = c['prediction_batch']
    for begin in range(0, len(source['z']), batch):
        n = min(batch, len(source['z'])-begin)
        pad = lambda value: np.pad(value[begin:begin+n], ((0, batch-n), (0, 0)), mode='edge')
        q = model.apply(dict(params=params), jnp.asarray(pad(source['z'])),
                        jnp.asarray(pad(source['kinetic_input'])), method=model.route)
        parts.append(np.asarray(q)[:n])
    return np.concatenate(parts)


def intervention_input(group, name, c):
    if name == 'velocity_zero': return base.apply_arm(group, 'S_U_router', c)
    if name == 'velocity_cross_KO_shuffle':
        return base.apply_arm(group, 'S_U_shuffled_velocity_program', c)
    if name == 'velocity_reverse': group = dict(group, velocity_program=-group['velocity_program'])
    if name == 'U_zero': group = dict(group, U_normalized=np.zeros_like(group['U_normalized']))
    return base.apply_arm(group, 'S_U_velocity_program', c)


def frozen_interventions(parent, output, c, inputs):
    """Change only frozen mixture weights. No ODE or train update is repeated."""
    parent = Path(parent)
    training, heldout, norm, split, _ = base.prepare(c)
    saved_norm = base.read_npz(parent/'training_programs.npz')
    for key in norm: np.testing.assert_array_equal(norm[key], saved_norm[key])
    records = json.loads((parent/'frozen_grid.json').read_text())['predictions']
    records = [r for r in records if r['arm'] == 'S_U_velocity_program']
    if len(records) != len(c['seeds'])*len(heldout):
        raise ValueError('Complete parent velocity-model predictions required')
    run = Artifacts(output, seed=c['seeds'], inputs=[*inputs, parent/'frozen_grid.json'])
    try:
        model = base.create_model(c, training[0]['z'].shape[1], 0, training[0]['embedding'])
        first = base.apply_arm(training[0], 'S_U_velocity_program', c)
        frozen = []
        for seed in c['seeds']:
            seed_records = [r for r in records if r['seed'] == seed]
            template = model.init(jax.random.PRNGKey(seed), jnp.asarray(first['z'][:1]),
                jnp.asarray(first['kinetic_input'][:1]), jnp.asarray(first['z'][:1]), .5,
                base.condition(first, 1, c))['params']
            model_path = Path(seed_records[0]['model'])
            if sha256(model_path) != seed_records[0]['model_sha256']:
                raise ValueError('Parent checkpoint changed')
            params = serialization.from_bytes(template, model_path.read_bytes())
            for g in heldout:
                record = next(r for r in seed_records if r['KO'] == g['label'])
                if sha256(record['path']) != record['sha256']:
                    raise ValueError('Parent prediction changed')
                p = base.read_npz(record['path'])
                np.testing.assert_array_equal(p['source_ids'], g['cell_ids'][g['score_indices']])
                reproduced = route_all(model, params, base.apply_arm(g, 'S_U_velocity_program', c), c)
                np.testing.assert_allclose(reproduced, p['full_source_q'], atol=1e-6, rtol=1e-5)
                for name in INTERVENTIONS:
                    if name == 'original': q = p['q']
                    elif name == 'cohort_q': q = np.broadcast_to(p['full_source_mean_q'], p['q'].shape)
                    elif name == 'uniform_q': q = np.full_like(p['q'], 1./p['q'].shape[1])
                    else: q = route_all(model, params, intervention_input(g, name, c), c)[g['score_indices']]
                    path = run.directory/f"seed{seed}_{g['label']}_{name}.npz"
                    np.savez_compressed(path, q=q, source_ids=p['source_ids'])
                    frozen.append(dict(KO=g['label'], seed=seed, intervention=name,
                        path=str(path.resolve()), sha256=sha256(path),
                        parent_prediction=record['path'], parent_sha256=record['sha256'],
                        model_sha256=record['model_sha256']))
        run.json('frozen_interventions.json', dict(predictions=frozen, train_updates=0,
            new_ODE_integrations=0, test_targets_read=False, new_validation_target_reads_before_freeze=False))
        target, _ = base.load_pack(Path(c['fold'])/'validation_target.npz', expected_side='target')
        rows = []
        for record in frozen:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Frozen intervention changed')
            p = base.read_npz(record['parent_prediction']); q = base.read_npz(record['path'])['q']
            y = target['z'][target['conditions'] == record['KO']]
            ix = np.random.default_rng(c['evaluation_seed']).choice(len(y), min(len(y), c['evaluation_cells']), replace=False)
            rows.append(dict(KO=record['KO'], seed=record['seed'], intervention=record['intervention'],
                **base.mixture_metrics(p['candidates'], q, y[ix], norm['axes'], device='cuda')))
        write_csv(run.directory/'metrics.csv', rows)
        with (parent/'metrics.csv').open() as stream:
            reference = {(r['KO'], int(r['seed'])): float(r['energy']) for r in csv.DictReader(stream)
                         if r['arm'] == 'S_U_velocity_program'}
        error = max(abs(r['energy']-reference[(r['KO'], r['seed'])]) for r in rows if r['intervention'] == 'original')
        if error > 1e-8: raise ArithmeticError('Original parent Energy failed reproduction')
        summary = []
        for name in INTERVENTIONS:
            selected = [r for r in rows if r['intervention'] == name]
            changes = [(r['energy']-reference[(r['KO'], r['seed'])])/
                        max(reference[(r['KO'], r['seed'])], 1e-12) for r in selected]
            summary.append(dict(intervention=name, energy=float(np.mean([r['energy'] for r in selected])),
                mean_paired_relative_error_excess=float(np.mean(changes))))
        write_csv(run.directory/'intervention_summary.csv', summary)
        result = dict(status='TRAXLER_FROZEN_ROUTER_INTERVENTIONS_COMPLETE', train_updates=0,
            new_ODE_integrations=0, models=len(c['seeds']), frozen_predictions=len(frozen),
            original_energy_reproduction_max_abs_error=error, test_targets_read=False,
            interpretation='inference_sensitivity_not_causal_biophysical_direction_proof', interventions=summary)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Frozen router interventions\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete'))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


def program(configuration, output):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda')
    plan = json.loads(Path(configuration).read_text())
    original_config = Path(plan['base_configuration'])
    parent_config = json.loads(original_config.read_text())
    code = [HERE/name for name in ('w15h_traxler_program_router.py', 'w15g_response_program_probe.py',
        'w15c_renge_l3.py', 'w15a_cellflow_moe.py', 'w15a_cellflow_moe_official.py',
        'w15a_cellflow_moe_real_predict.py', 'w15a_cellflow_train.py', 'w15a_cellflow_adapter.py',
        'w14_loro.py', 'w4_artifacts.py')]
    inputs = [Path(configuration), original_config, Path(__file__), *code]
    run = Artifacts(output, seed=parent_config['seeds'], inputs=inputs)
    deadline = time.monotonic()+plan['maximum_hours']*3600
    try:
        wait_for_predecessor(plan['predecessor_job_status'], deadline, lambda s: run.json('status.json', s))
        run.json('queue_manifest.json', dict(plan=plan, new_fits=21, reused_baseline_fits=3,
            frozen_parent_interventions=7, physical_gpu=0, reserved_GPUs=[6, 7], test_targets_read=False))
        run.json('status.json', dict(status='running_frozen_interventions', new_training_completed=0))
        mechanism = frozen_interventions(plan['parent'], run.directory/'frozen_6to24', parent_config, inputs)
        results = []
        for task in plan['tasks']:
            if time.monotonic() > deadline: raise TimeoutError('Follow-up queue exceeded deadline')
            c = dict(parent_config, **task['overrides'])
            fold = Path(c['fold'])
            task_inputs = [*inputs, Path(c['split']), Path(c['conditions']), fold/'train_source.npz',
                fold/'train_target.npz', fold/'validation_source.npz', fold/'transform.npz']
            run.json('status.json', dict(status='training', task=task['name'],
                completed_tasks=len(results), total_tasks=len(plan['tasks'])))
            result = run_case(c, run.directory/task['name'], scope=task['scope'],
                parent=plan['parent'] if task.get('reuse_baseline') else None,
                inputs=task_inputs, deadline=deadline)
            results.append(dict(name=task['name'], result=result))
        result = dict(status='TRAXLER_ROUTER_FOLLOWUP_QUEUE_COMPLETE', tasks=results,
            frozen_mechanism=mechanism, new_fits=21, reused_baseline_fits=3, test_targets_read=False)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Traxler queued follow-ups\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete', completed_new_models=21))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--configuration', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); print(json.dumps(program(args.configuration, args.output), indent=2))
