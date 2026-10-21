"""Matched S+U router experiment with source-cohort response-program features.

This is a full distribution predictor, not the preceding ridge mean probe.
Training-only KO displacements define five axes. Each arm routes on source S,
cell U and five extra slots (zero / velocity direction / shuffled / U direction).
The official CellFlow state stream and K independent ODE experts are unchanged.
No physical cosine constraint, mean anchor, GFG, JVP or held-out future is used
to construct inputs. The 6h -> 24h task is an 18h early-response continuation.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
from w15g_response_program_probe import derangement, read_npz, response_basis, unit
from w15c_renge_l3 import create_model, source_state_conditions
from w15a_cellflow_moe_official import make_official_ot_train_step
from w15a_cellflow_moe_real_predict import make_candidate_integrator
from veloroute.latent import load_pack

import jax
import jax.numpy as jnp
import optax
from cellflow.utils import match_linear
from flax import serialization
from flax.training.train_state import TrainState


ARMS = ('S_U_router', 'S_U_velocity_program', 'S_U_shuffled_velocity_program',
        'S_U_U_program')
ROOT = Path(__file__).resolve().parents[2]


def condition(group, count, c):
    return source_state_conditions(count, group['time_hours'], group['embedding'],
                                   delta_hours=c['target_hours']-c['source_hours'])


def prepare(c):
    """The only future opened here is the training-KO target distribution."""
    fold = Path(c['fold'])
    split = json.loads(Path(c['split']).read_text())
    if any(set(split[a]) & set(split[b]) for a, b in
           (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('KO partitions overlap')
    source, sm = load_pack(fold/'train_source.npz', expected_side='source')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    validation, vm = load_pack(fold/'validation_source.npz', expected_side='source')
    if (sm.get('day'), tm.get('day'), vm.get('day')) != (
            c['source_hours'], c['target_hours'], c['source_hours']):
        raise ValueError('Wrong source/target biological times')
    for pack, role in ((source, 'train'), (target, 'train'), (validation, 'val')):
        if set(map(str, pack['conditions'])) != set(split[role]):
            raise ValueError('Frozen KO assignment changed')
    esm = read_npz(c['conditions'])
    metadata = json.loads(str(esm['metadata_json']))
    if not metadata.get('frozen') or not metadata.get('no_cell_expression_used'):
        raise ValueError('Frozen public protein conditions required')
    vectors = {str(k): np.asarray(v, dtype='float32')
               for k, v in zip(esm['conditions'], esm['embeddings'])}

    def groups(pack, training):
        result = []
        for label in sorted(set(map(str, pack['conditions']))):
            mask = pack['conditions'] == label
            g = dict(label=label, z=np.asarray(pack['z'][mask], dtype='float32'),
                U=np.asarray(pack['u_features'][mask], dtype='float32'),
                velocity=np.asarray(pack['velocity'][mask], dtype='float32'),
                cell_ids=np.asarray(pack['cell_ids'][mask], dtype=str),
                time_hours=c['source_hours'], embedding=vectors[label])
            if training:
                g['target'] = np.asarray(target['z'][target['conditions'] == label], dtype='float32')
            else:
                g['score_indices'] = np.random.default_rng(c['evaluation_seed']).choice(
                    len(g['z']), min(len(g['z']), c['evaluation_cells']), replace=False)
            result.append(g)
        return result

    training, heldout = groups(source, True), groups(validation, False)
    displacement = np.stack([g['target'].astype(float).mean(0)-g['z'].astype(float).mean(0)
                             for g in training])
    response_mean, axes, singular = response_basis(displacement, c['programs'])
    if len(axes) != c['programs']:
        raise ValueError('Insufficient training KOs for the fixed response axes')
    references = dict(
        velocity=np.mean([g['velocity'].astype(float).mean(0) for g in training], axis=0),
        U=np.mean([g['U'].astype(float).mean(0) for g in training], axis=0))
    normalizer = dict(U_mean=source['u_features'].astype(float).mean(0),
        U_scale=np.maximum(source['u_features'].astype(float).std(0), 1e-5),
        axes=axes, singular_values=singular, response_mean=response_mean,
        velocity_reference=references['velocity'], U_reference=references['U'])
    permutations = {}
    for name, collection, offset in (('train', training, 0), ('validation', heldout, 100000)):
        order = derangement(len(collection), c['shuffle_seed']+offset)
        for g in collection:
            g['U_normalized'] = ((g['U']-normalizer['U_mean'])/normalizer['U_scale']).astype('float32')
            for key in ('velocity', 'U'):
                projected = (g[key].astype(float).mean(0)-references[key]) @ axes.T
                g[key+'_program'] = unit(projected[None])[0].astype('float32')
        for i, g in enumerate(collection):
            g['shuffled_velocity_program'] = collection[order[i]]['velocity_program'].copy()
        permutations[name] = {g['label']: collection[order[i]]['label']
                              for i, g in enumerate(collection)}
    return training, heldout, normalizer, split, permutations


def apply_arm(group, arm, c):
    if arm not in ARMS:
        raise ValueError('Unregistered program-router arm')
    feature = (np.zeros(c['programs'], dtype='float32') if arm == 'S_U_router' else
               group[{'S_U_velocity_program': 'velocity_program',
                      'S_U_shuffled_velocity_program': 'shuffled_velocity_program',
                      'S_U_U_program': 'U_program'}[arm]])
    extra = np.broadcast_to(feature, (len(group['z']), c['programs']))
    inputs = np.concatenate((group['U_normalized'], extra), axis=1).astype('float32')
    if not np.isfinite(inputs).all():
        raise FloatingPointError('Nonfinite source-only router input')
    return dict(group, kinetic_input=inputs)


def make_step(model, c):
    match = functools.partial(match_linear, epsilon=c['ot_epsilon'], scale_cost='mean',
                              max_iterations=c['ot_max_iterations'])
    return make_official_ot_train_step(model, match)


def train(model, step, groups, c, seed, callback):
    g = groups[0]
    initial = model.init(jax.random.PRNGKey(seed), jnp.asarray(g['z'][:1]),
        jnp.asarray(g['kinetic_input'][:1]), jnp.asarray(g['z'][:1]), .5, condition(g, 1, c))['params']
    state = TrainState.create(apply_fn=model.apply, params=initial, tx=optax.adam(c['learning_rate']))
    rng = np.random.default_rng(seed)
    history, started = [], time.monotonic()
    for update in range(1, c['steps']+1):
        g = groups[int(rng.integers(len(groups)))]; n = c['batch_size']
        si = rng.choice(len(g['z']), n, replace=len(g['z']) < n)
        ti = rng.choice(len(g['target']), n, replace=len(g['target']) < n)
        state, metrics = step(state, jax.random.fold_in(jax.random.PRNGKey(seed), update),
            jnp.asarray(g['z'][si]), jnp.asarray(g['target'][ti]),
            jnp.asarray(g['kinetic_input'][si]), condition(g, n, c))
        if update == 1 or update % c['log_interval'] == 0 or update == c['steps']:
            values = {key: float(value) for key, value in metrics.items()
                      if key not in ('source_indices', 'target_indices')}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError('Nonfinite router update')
            row = dict(step=update, elapsed_seconds=time.monotonic()-started, **values)
            history.append(row); callback(row)
    return state.params, initial, history


def predict(model, integrate, params, source, c):
    if 'target' in source:
        raise ValueError('Prediction must not receive future targets')
    batch = c['prediction_batch']; all_q = []
    for begin in range(0, len(source['z']), batch):
        end = min(begin+batch, len(source['z'])); n = end-begin
        pad = lambda value: np.pad(value[begin:end], ((0, batch-n), (0, 0)), mode='edge')
        q = model.apply(dict(params=params), jnp.asarray(pad(source['z'])),
                        jnp.asarray(pad(source['kinetic_input'])), method=model.route)
        all_q.append(np.asarray(q)[:n])
    full_q = np.concatenate(all_q)
    np.testing.assert_allclose(full_q.sum(1), 1., atol=1e-6)
    indices = source['score_indices']; candidates = []
    for begin in range(0, len(indices), batch):
        ix = indices[begin:begin+batch]; n = len(ix)
        z = np.pad(source['z'][ix], ((0, batch-n), (0, 0)), mode='edge')
        candidates.append(np.asarray(integrate(params, jnp.asarray(z), condition(source, batch, c), 1.))[:n])
    return dict(candidates=np.concatenate(candidates), q=full_q[indices],
        source_ids=source['cell_ids'][indices], source_indices=indices,
        full_source_ids=source['cell_ids'], full_source_q=full_q,
        full_source_mean_q=full_q.mean(0))


def mixture_metrics(candidates, q, target, axes, device='cpu'):
    from w14_loro import energy
    candidates, q, target = [np.asarray(x, dtype='float64') for x in (candidates, q, target)]
    if candidates.shape[:2] != q.shape or candidates.shape[-1] != target.shape[-1]:
        raise ValueError('Prediction mixture dimensions differ')
    if not all(np.isfinite(a).all() for a in (candidates, q, target)) or (q < 0).any():
        raise ValueError('Invalid finite prediction mixture')
    np.testing.assert_allclose(q.sum(1), 1., atol=1e-6)
    x = candidates.reshape(-1, candidates.shape[-1]); w = q.ravel()/len(q)
    mean = w @ x; truth = target.mean(0)
    variance = max(0., float(w @ (x*x).sum(1)-mean@mean))
    ratio = variance/max(float(target.var(0).sum()), 1e-12)
    population_q = q.mean(0)
    entropy = -float(np.sum(population_q*np.log(np.maximum(population_q, 1e-30))))
    return dict(energy=energy(x, w, target, device), mean_mse=float(((mean-truth)**2).mean()),
        program_mean_mse=float((((mean-truth) @ np.asarray(axes).T)**2).mean()),
        variance_ratio=ratio, log_variance_error=abs(float(np.log(max(ratio, 1e-12)))),
        population_effective_experts=float(np.exp(entropy)),
        mean_cell_entropy=-float(np.mean(np.sum(q*np.log(np.maximum(q, 1e-30)), axis=1))))


def evaluate(root, c, frozen, split, normalizer):
    from w7_router_scaling import write_csv
    expected = {(seed, arm, label) for seed in c['seeds'] for arm in ARMS for label in split['val']}
    if {(r['seed'], r['arm'], r['KO']) for r in frozen} != expected or len(frozen) != len(expected):
        raise ValueError('All registered predictions must exist before evaluation')
    for record in frozen:
        if sha256(record['path']) != record['sha256'] or sha256(record['model']) != record['model_sha256']:
            raise ValueError('Frozen prediction/checkpoint changed')
    target, tm = load_pack(Path(c['fold'])/'validation_target.npz', expected_side='target')
    if tm.get('day') != c['target_hours'] or set(map(str, target['conditions'])) != set(split['val']):
        raise ValueError('Wrong validation target cohort')
    rows, comparisons = [], []
    for label in sorted(split['val']):
        y = target['z'][target['conditions'] == label]
        ix = np.random.default_rng(c['evaluation_seed']).choice(len(y), min(len(y), c['evaluation_cells']), replace=False)
        y = y[ix]; common_ids = None
        for seed in c['seeds']:
            for arm in ARMS:
                record = next(r for r in frozen if (r['KO'], r['seed'], r['arm']) == (label, seed, arm))
                p = read_npz(record['path'])
                if common_ids is None: common_ids = p['source_ids']
                else: np.testing.assert_array_equal(common_ids, p['source_ids'])
                rows.append(dict(KO=label, seed=seed, arm=arm, source_cells=len(p['q']), target_cells=len(y),
                    **mixture_metrics(p['candidates'], p['q'], y, normalizer['axes'], device='cuda')))
            metrics = {r['arm']: r for r in rows if r['KO'] == label and r['seed'] == seed}
            real = metrics['S_U_velocity_program']
            for control in ARMS:
                if control == 'S_U_velocity_program': continue
                comparisons.append(dict(KO=label, seed=seed, control=control,
                    **{f'{name}_gain': 1-real[name]/max(metrics[control][name], 1e-12)
                       for name in ('energy', 'mean_mse', 'program_mean_mse', 'log_variance_error')}))
    write_csv(root/'metrics.csv', rows); write_csv(root/'comparisons.csv', comparisons)
    means = {arm: {name: float(np.mean([r[name] for r in rows if r['arm'] == arm]))
        for name in ('energy', 'mean_mse', 'program_mean_mse', 'variance_ratio',
                     'log_variance_error', 'population_effective_experts', 'mean_cell_entropy')} for arm in ARMS}
    gains = {arm: {key: float(np.mean([r[key] for r in comparisons if r['control'] == arm]))
        for key in ('energy_gain', 'mean_mse_gain', 'program_mean_mse_gain', 'log_variance_error_gain')}
        for arm in ARMS if arm != 'S_U_velocity_program'}
    return dict(status='TRAXLER_PROGRAM_ROUTER_DEVELOPMENT_COMPLETE', train_KOs=len(split['train']),
        validation_KOs=len(split['val']), reserved_test_KOs=len(split['test']), test_targets_read=False,
        task='KO_background_conditioned_6to24h_response', source_hours=c['source_hours'],
        target_hours=c['target_hours'], biological_delta_hours=c['target_hours']-c['source_hours'],
        fits=len(ARMS)*len(c['seeds']), steps_per_fit=c['steps'], frozen_predictions=len(frozen),
        experts=8, ODE='independent_Tsit5_experts_fixed_source_q', mean_anchor=False,
        velocity_estimator='source_only_beta1_tail_OLS_not_GFG', JVP=False,
        physical_direction_constraint=False, response_features='cohort_mean_training_centered_unit_projection',
        means=means, mean_paired_relative_gains=gains, aggregation='equal_KO_equal_seed_no_seed_ensemble',
        mean_energy_definition='weighted_V_statistic_divided_by_sqrt_PCA_dimensions',
        validation_previously_used_for_development=True, independent_confirmation=False,
        response_components_are_not_observed_fates=True, automatic_retuning=False)


def program(configuration, output):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda')
    c = json.loads(Path(configuration).read_text())
    if c['experts'] != 8 or c['source_hours'] != 6 or c['target_hours'] != 24:
        raise ValueError('The queued recipe is fixed to K8 and 6h -> 24h')
    fold = Path(c['fold'])
    code = [Path(__file__).with_name(name) for name in ('w15g_response_program_probe.py',
        'w15c_renge_l3.py', 'w15a_cellflow_moe.py', 'w15a_cellflow_moe_official.py',
        'w15a_cellflow_moe_real_predict.py', 'w15a_cellflow_train.py', 'w15a_cellflow_adapter.py',
        'w14_loro.py', 'w4_artifacts.py')]
    run = Artifacts(output, seed=c['seeds'], inputs=[Path(configuration), Path(__file__), *code,
        Path(c['split']), Path(c['conditions']), fold/'train_source.npz', fold/'train_target.npz',
        fold/'validation_source.npz', fold/'transform.npz'])
    started = time.monotonic(); deadline = started+c['maximum_hours']*3600
    try:
        training, heldout, normalizer, split, permutations = prepare(c)
        np.savez_compressed(run.directory/'training_programs.npz', **normalizer)
        run.json('study_manifest.json', dict(config=c, arms=list(ARMS), planned_fits=12,
            train_KOs=sorted(split['train']), validation_KOs=sorted(split['val']), test_KOs=sorted(split['test']),
            test_targets_read=False, validation_targets_read=False, shuffle_mapping=permutations,
            cell_U_dimensions=training[0]['U'].shape[1], program_dimensions=len(normalizer['axes']),
            program_variance_explained=float(np.sum(normalizer['singular_values'][:c['programs']]**2)/
                                             np.sum(normalizer['singular_values']**2)),
            validation_previously_used_for_development=True))
        model = create_model(c, training[0]['z'].shape[1], 0, training[0]['embedding'])
        step = make_step(model, c); integrate = make_candidate_integrator(model)
        frozen, initialization = [], {}
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic() > deadline:
                    raise TimeoutError('Bounded router experiment exceeded its deadline')
                directory = run.directory/'fits'/f'seed{seed}'/arm
                fitrun = Artifacts(directory, seed=seed, inputs=[Path(configuration), Path(__file__)])
                groups = [apply_arm(g, arm, c) for g in training]
                run.json('status.json', dict(status='training', completed_models=len(frozen)//len(heldout),
                    total_models=12, seed=seed, arm=arm, stage='training'))
                params, initial, history = train(model, step, groups, c, seed,
                    lambda row: fitrun.json('status.json', dict(status='training', **row)))
                initial_hash = hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if seed in initialization and initialization[seed] != initial_hash:
                    raise ValueError('Matched arms did not share identical initial parameters')
                initialization[seed] = initial_hash
                checkpoint = directory/'model.msgpack'; checkpoint.write_bytes(serialization.to_bytes(params))
                params = serialization.from_bytes(initial, checkpoint.read_bytes())
                model_hash = sha256(checkpoint)
                fitrun.json('training_history.json', dict(history=history))
                for source in heldout:
                    p = predict(model, integrate, params, apply_arm(source, arm, c), c)
                    path = directory/(source['label']+'.npz'); np.savez_compressed(path, **p)
                    frozen.append(dict(KO=source['label'], seed=seed, arm=arm, path=str(path.resolve()),
                        sha256=sha256(path), model=str(checkpoint.resolve()), model_sha256=model_hash,
                        heldout_future_used=False))
                fitrun.json('summary.json', dict(status='complete', arm=arm, seed=seed,
                    parameters=sum(x.size for x in jax.tree_util.tree_leaves(params)),
                    steps=c['steps'], initial_parameters_sha256=initial_hash, model_sha256=model_hash,
                    fixed_source_q=True, independent_expert_ODEs=True, heldout_future_used=False,
                    backend=[str(device) for device in jax.devices()]))
                fitrun.json('status.json', dict(status='complete'))
                run.json('status.json', dict(status='training', completed_models=len(frozen)//len(heldout),
                    total_models=12, last_arm=arm, last_seed=seed, elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(status='model_complete', seed=seed, arm=arm,
                    completed_models=len(frozen)//len(heldout), elapsed_seconds=time.monotonic()-started)), flush=True)
        run.json('frozen_grid.json', dict(predictions=frozen, validation_targets_read=False, test_targets_read=False))
        run.json('status.json', dict(status='evaluating', completed_models=12, total_models=12))
        result = evaluate(run.directory, c, frozen, split, normalizer)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Traxler S+U response-program router\n\n'+
            json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete', completed_models=12, total_models=12))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(program(args.configuration, args.output), indent=2))
