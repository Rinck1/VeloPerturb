"""Source-only relational contrast for the fixed perturbation-conditioned MoE.

Positive-only agreement is replaced by a local ranking constraint. Signed and
unsigned program geometry are matched controls, not observed fate labels.
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
import w15o_velocity_neighborhood as graph
import w15t_conditioned_velocity_router as conditional

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ('signed_velocity', 'unsigned_velocity', 'shuffled_velocity', 'signed_U')
REFERENCE_ARMS = ('no_graph', 'positive_velocity_graph')


def contrast_graphs(group, candidates, neighbors, margin_scale):
    from scipy.spatial.distance import cdist
    x = np.asarray(group['z'], dtype='float64')
    if len(x) < 3 or not 1 <= neighbors <= candidates or not 0 <= margin_scale <= 1:
        raise ValueError('Invalid contrast graph settings')
    distance = cdist(x, x, metric='sqeuclidean')
    np.fill_diagonal(distance, np.inf)
    pool = np.argsort(distance, axis=1, kind='stable')[:, :min(candidates, len(x)-1)]
    k = min(neighbors, pool.shape[1]//2)
    rows = np.arange(len(x))[:, None]
    result = {}
    for arm, key in [('signed_velocity', 'velocity_program'), ('unsigned_velocity', 'velocity_program'),
                     ('shuffled_velocity', 'shuffled_velocity_program'), ('signed_U', 'U_program')]:
        value = graph.unit(group[key])
        dot = np.clip(np.einsum('id,ijd->ij', value, value[pool]), -1, 1)
        similarity = dot**2 if arm == 'unsigned_velocity' else (1+dot)/2
        order = np.argsort(similarity, axis=1, kind='stable')
        pi, ni = order[:, -k:][:, ::-1], order[:, :k]
        positive, negative = pool[rows, pi], pool[rows, ni]
        margin = margin_scale*np.maximum(similarity[rows, pi]-similarity[rows, ni], 0.)
        result[arm] = dict(positive=positive, negative=negative, margin=margin.astype('float32'))
    return result


def js_each(p, q):
    import jax.numpy as jnp
    m = (p+q)/2
    log = lambda a: jnp.log(jnp.maximum(a, 1e-12))
    return .5*jnp.sum(p*(log(p)-log(m))+q*(log(q)-log(m)), axis=-1)/jnp.log(2.)


def contrast_loss(model, params, x, u, px, pu, nx, nu, condition, margin):
    import jax.numpy as jnp
    route = lambda state, kinetic: model.apply(dict(params=params), state, kinetic, condition, method=model.route)
    q, qp, qn = route(x, u), route(px, pu), route(nx, nu)
    positive, negative = js_each(q, qp), js_each(q, qn)
    return jnp.mean(jnp.maximum(positive-negative+margin, 0.)), (jnp.mean(positive), jnp.mean(negative))


def make_step(model, match_fn):
    import jax
    from ott.solvers import utils as solver_utils
    from w15a_cellflow_moe import align_ot_indices_jit, paired_flow_matching_loss

    @jax.jit
    def step(state, rng, source, target, kinetic, px, pu, nx, nu, margin, condition, weight):
        pair_key, tau_key = jax.random.split(rng)
        si, ti = solver_utils.sample_joint(pair_key, match_fn(source, target))
        paired = align_ot_indices_jit(source, kinetic, target, si, ti, condition=condition)
        tau = jax.random.uniform(tau_key, (len(si), 1))

        def objective(params):
            fm, metrics = paired_flow_matching_loss(model, dict(params=params), paired, tau)
            relation, (positive, negative) = contrast_loss(model, params, paired.source_x, paired.kinetic_input,
                px[si], pu[si], nx[si], nu[si], paired.condition, margin[si])
            contribution = source.shape[1]*weight*relation
            return fm+contribution, dict(metrics, fm_loss=fm, relation_loss=relation,
                positive_js=positive, negative_js=negative, relation_contribution=contribution)
        (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(state.params)
        return state.apply_gradients(grads=gradient), dict(metrics, loss=loss, source_indices=si, target_indices=ti)
    return step


def fit(model, step, training, graphs, c, plan, arm, seed, callback):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    import w15h_traxler_program_router as base
    g = training[0]
    x = jnp.asarray(g['z'][:1])
    initial = model.init(jax.random.PRNGKey(seed), x, jnp.asarray(shared_input(g, c['programs'])[:1]),
                         x, .5, base.condition(g, 1, c))['params']
    state = TrainState.create(apply_fn=model.apply, params=initial, tx=optax.adam(c['learning_rate']))
    inputs = {g['label']: shared_input(g, c['programs']) for g in training}
    rng, neighbor_rng = np.random.default_rng(seed), np.random.default_rng(seed+1000000)
    history, started = [], time.monotonic()
    for update in range(1, c['steps']+1):
        g = training[int(rng.integers(len(training)))]; n = c['batch_size']; u = inputs[g['label']]
        si = rng.choice(len(g['z']), n, replace=len(g['z']) < n)
        ti = rng.choice(len(g['target']), n, replace=len(g['target']) < n)
        edges = graphs[g['label']][arm]
        choice = neighbor_rng.integers(edges['positive'].shape[1], size=n)
        pi, ni, margin = [edges[k][si, choice] for k in ('positive', 'negative', 'margin')]
        state, metrics = step(state, jax.random.fold_in(jax.random.PRNGKey(seed), update),
            jnp.asarray(g['z'][si]), jnp.asarray(g['target'][ti]), jnp.asarray(u[si]),
            jnp.asarray(g['z'][pi]), jnp.asarray(u[pi]), jnp.asarray(g['z'][ni]), jnp.asarray(u[ni]),
            jnp.asarray(margin), base.condition(g, n, c), jnp.asarray(plan['graph_weight_normalized']))
        if update == 1 or update % c['log_interval'] == 0 or update == c['steps']:
            values = {k: float(v) for k, v in metrics.items() if k not in ('source_indices', 'target_indices')}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError('Nonfinite relational update')
            row = dict(step=update, elapsed_seconds=time.monotonic()-started, **values)
            history.append(row); callback(row)
    return state.params, initial, history


def references(parent):
    records = json.loads((Path(parent)/'whole_grid_frozen.json').read_text())['predictions']
    rename = {'no_graph': 'no_graph', 'velocity_graph': 'positive_velocity_graph'}
    return [dict(r, arm=rename[r['arm']], reference=True) for r in records
            if r['case'] == 'conditioned' and r['arm'] in rename]


def verify(records, c, split, arms, projection_path):
    expected = {(s, a, k) for s in c['seeds'] for a in tuple(arms)+REFERENCE_ARMS for k in split['val']}
    if len(records) != len(expected) or {(r['seed'], r['arm'], r['KO']) for r in records} != expected:
        raise ValueError('Whole new and reference grid must be frozen before scoring')
    cached = {}
    projection_hash = sha256(projection_path)
    for r in records:
        for path, digest in ((r['path'], r['sha256']), (r['model'], r['model_sha256'])):
            if path not in cached:
                cached[path] = sha256(path)
            if cached[path] != digest:
                raise ValueError('Frozen model/prediction changed')
        if r.get('condition_projection_sha256') != projection_hash:
            raise ValueError('Shared condition projection changed')


def evaluate(output, c, split, norm, records, plan):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics
    from w7_router_scaling import write_csv
    verify(records, c, split, plan['arms'], output/'condition_projection.npz')
    targets, metadata = base.load_pack(Path(c['fold'])/'validation_target.npz', expected_side='target')
    if metadata.get('day') != c['target_hours'] or set(targets['conditions']) != set(split['val']):
        raise ValueError('Wrong target cohort')
    rows = []
    for ko in split['val']:
        y = targets['z'][targets['conditions'] == ko]
        y = y[np.random.default_rng(c['evaluation_seed']).choice(len(y), min(len(y), c['evaluation_cells']), replace=False)]
        ids = None
        for r in [r for r in records if r['KO'] == ko]:
            p = base.read_npz(r['path'])
            if ids is None:
                ids = p['source_ids']
            else:
                np.testing.assert_array_equal(ids, p['source_ids'])
            q = np.asarray(p['full_source_q'], dtype='float64'); avg = q.mean(0)
            rows.append(dict(KO=ko, seed=r['seed'], arm=r['arm'],
                effective_experts=float(np.exp(-np.sum(avg*np.log(np.maximum(avg, 1e-12))))),
                mean_cell_entropy=float(-np.sum(q*np.log(np.maximum(q, 1e-12)), axis=1).mean()),
                **empirical_metrics(p['candidates'], p['q'], y, norm['axes'])))
    by = {(r['seed'], r['KO'], r['arm']): r for r in rows}
    comparisons, pairs = [], []
    all_arms = tuple(plan['arms'])+REFERENCE_ARMS
    for arm in all_arms:
        if arm == plan['real_arm']:
            continue
        selected = []
        for seed in c['seeds']:
            for ko in split['val']:
                gain = 1-by[(seed, ko, plan['real_arm'])]['energy']/by[(seed, ko, arm)]['energy']
                row = dict(seed=seed, KO=ko, control=arm, energy_gain=gain)
                selected.append(row); pairs.append(row)
        per_seed = {str(s): float(np.mean([r['energy_gain'] for r in selected if r['seed'] == s])) for s in c['seeds']}
        per_ko = {ko: float(np.mean([r['energy_gain'] for r in selected if r['KO'] == ko])) for ko in split['val']}
        comparisons.append(dict(control=arm, energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
                                by_seed=per_seed, by_KO=per_ko, winning_KOs=sum(v > 0 for v in per_ko.values())))
    primary = [r for r in comparisons if r['control'] in plan['primary_controls']]
    if len(primary) != len(plan['primary_controls']):
        raise ValueError('Missing primary control')
    stable = all(r['energy_gain'] > 0 and all(v > 0 for v in r['by_seed'].values()) for r in primary)
    lookup = {r['control']: r for r in comparisons}
    keys = ('energy', 'mean_mse', 'variance_ratio', 'log_variance_error', 'effective_experts', 'mean_cell_entropy')
    write_csv(output/'metrics.csv', rows); write_csv(output/'comparisons.csv', pairs)
    return dict(comparisons=comparisons, primary_screen_passed=stable,
        expansion_target_met=stable and lookup['no_graph']['energy_gain'] >= .02 and lookup['positive_velocity_graph']['energy_gain'] > 0,
        means={a: {k: float(np.mean([r[k] for r in rows if r['arm'] == a])) for k in keys} for a in all_arms})


def launch(configuration, output, job_directory, script=None):
    from veloroute.artifacts import save_json
    from veloroute.gpu_policy import gpu_uuids, ALLOWED_PHYSICAL_GPUS, RESERVED_PHYSICAL_GPUS, MAX_VISIBLE_GPUS
    from w15i_followup_queue import worker_environment
    plan = json.loads(Path(configuration).read_text()); gpu = plan['gpu']
    directory, output = Path(job_directory).resolve(), Path(output).resolve()
    if directory.exists() or output.exists() or not directory.is_relative_to(ROOT/'outputs'):
        raise ValueError('New output and project-local job directory required')
    directory.mkdir(parents=True)
    env = worker_environment(output.parent/f'runtime_gpu{gpu}')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    job = dict(physical_gpu=gpu, gpu_uuid=gpu_uuids()[gpu], cwd=str(ROOT),
        command=[sys.executable, str(Path(script or __file__).resolve()), '--configuration', str(Path(configuration).resolve()), '--output', str(output)],
        allowed_gpus=list(ALLOWED_PHYSICAL_GPUS), reserved_gpus=list(RESERVED_PHYSICAL_GPUS), max_visible_gpus=MAX_VISIBLE_GPUS)
    save_json(directory/'job.json', job)
    with (directory/'job.log').open('xb') as stream:
        child = subprocess.Popen([sys.executable, str(HERE/'launch_gpu_job.py'), '--worker', str(directory)], cwd=ROOT,
            env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(directory/'launcher.json', dict(pid=child.pid, physical_gpu=gpu, detached=True))
    return dict(status='LAUNCHED', pid=child.pid, physical_gpu=gpu, job_directory=str(directory))


def prepare_run(configuration, output, script):
    from w15k_traxler_l2_cell import wait_for_free_gpu
    from veloroute.gpu_policy import enforce_gpu_policy
    plan = json.loads(Path(configuration).read_text())
    wait_for_free_gpu(plan['gpu'], time.monotonic()+600, lambda s: print(json.dumps(s), flush=True))
    enforce_gpu_policy('cuda')
    import w15h_traxler_program_router as base
    c = json.loads(Path(plan['base_configuration']).read_text()); parent = Path(plan['parent'])
    if c['seeds'] != plan['model_seeds'] or c['steps'] != plan['steps']:
        raise ValueError('Parent budget changed')
    inputs = [Path(configuration), Path(plan['base_configuration']), ROOT/'PLANS/w15u_v_velocity_expansion_20261022.md',
        Path(script), HERE/'w15t_conditioned_velocity_router.py', HERE/'w15o_velocity_neighborhood.py',
        HERE/'w15n_velocity_auxiliary.py', HERE/'w15h_traxler_program_router.py', HERE/'w15i_program_router_followups.py',
        HERE/'w15a_cellflow_moe.py', HERE/'w15a_cellflow_moe_official.py', HERE/'w15a_cellflow_moe_real_predict.py',
        HERE/'w15c_renge_l3.py', HERE/'w15a_cellflow_adapter.py', HERE/'w4_artifacts.py',
        Path(c['conditions']), Path(c['split']), Path(c['fold'])/'train_source.npz', Path(c['fold'])/'train_target.npz',
        Path(c['fold'])/'validation_source.npz', Path(c['fold'])/'transform.npz', parent/'whole_grid_frozen.json',
        parent/'training_programs.npz', parent/'condition_projection.npz']
    if Path(script).name != Path(__file__).name:
        inputs.append(Path(__file__))
    run = Artifacts(output, seed=c['seeds'], inputs=inputs)
    training, heldout, norm, split, _ = base.prepare(c)
    saved = base.read_npz(parent/'training_programs.npz')
    for key in norm:
        np.testing.assert_array_equal(norm[key], saved[key])
    projection = base.read_npz(parent/'condition_projection.npz')
    (run.directory/'condition_projection.npz').write_bytes((parent/'condition_projection.npz').read_bytes())
    np.savez_compressed(run.directory/'training_programs.npz', **norm)
    model = conditional.conditioned_model(base.create_model(c, 50, 0, training[0]['embedding']), projection)
    records = references(parent)
    run.json('study_manifest.json', dict(plan=plan, config=c, train_KOs=split['train'], validation_KOs=split['val'],
        reserved_test_KOs=split['test'], test_targets_read=False, validation_previously_used=True, JVP=False,
        model_parameters=391752, source_input='original_S_U_plus_zero5', condition_projection_shared_with_parent=True))
    return plan, c, run, training, heldout, norm, split, model, records


def save_fit(run, fit_run, model, params, initial, history, heldout, c, plan, seed, arm, records):
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    parent_summary = json.loads((Path(plan['parent'])/'fits'/f'seed{seed}'/'no_graph'/'summary.json').read_text())
    initial_hash = hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
    if initial_hash != parent_summary['initial_parameters_sha256']:
        raise ValueError('Conditional parent initialization changed')
    checkpoint = fit_run.directory/'model.msgpack'
    checkpoint.write_bytes(serialization.to_bytes(params))
    # Reuse ONE compiled independent-expert solver across fits.
    if not hasattr(save_fit, '_integrator') or save_fit._model is not model:
        save_fit._integrator = base.make_candidate_integrator(model); save_fit._model = model
    for source in heldout:
        p = conditional.predict(model, save_fit._integrator, params, deployment_source(source, c['programs']), c)
        path = fit_run.directory/(source['label']+'.npz'); np.savez_compressed(path, **p)
        records.append(dict(KO=source['label'], seed=seed, arm=arm, path=str(path), sha256=sha256(path),
            model=str(checkpoint), model_sha256=sha256(checkpoint),
            condition_projection_sha256=sha256(run.directory/'condition_projection.npz'),
            explicit_velocity_input=False, heldout_future_used=False))
    summary = dict(status='complete', seed=seed, arm=arm, steps=c['steps'],
                   parameters=sum(v.size for v in jax.tree_util.tree_leaves(params)), initial_parameters_sha256=initial_hash)
    fit_run.json('training_history.json', dict(history=history)); fit_run.json('summary.json', summary)
    fit_run.json('status.json', dict(status='complete'))
    return summary


def finish(run, plan, c, split, norm, records, summaries):
    from threadpoolctl import threadpool_limits
    verify(records, c, split, plan['arms'], run.directory/'condition_projection.npz')
    run.json('whole_grid_frozen.json', dict(predictions=records, validation_targets_read_before_freeze=False, test_targets_read=False))
    run.json('status.json', dict(status='evaluating', completed_fits=len(summaries)))
    with threadpool_limits(limits=1):
        results = evaluate(run.directory, c, split, norm, records, plan)
    result = dict(status=plan['kind']+'_COMPLETE', new_fits=len(summaries), new_predictions=len(summaries)*len(split['val']),
                  parameters=summaries[0]['parameters'], fit_diagnostics=summaries, test_targets_read=False, JVP=False, **results)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# '+plan['kind']+'\n\n'+json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete', completed_fits=len(summaries), total_fits=len(summaries)))
    return result


def program(configuration, output):
    plan, c, run, training, heldout, norm, split, model, records = prepare_run(configuration, output, __file__)
    from cellflow.utils import match_linear
    from w15i_program_router_followups import cell_programs
    training, _ = cell_programs(training, norm, plan['shuffle_seed'])
    graphs = {g['label']: contrast_graphs(g, plan['state_candidates'], plan['graph_neighbors'], plan['margin_scale']) for g in training}
    np.savez_compressed(run.directory/'contrast_graphs.npz', **{f'{ko}__{arm}__{key}': value
        for ko, arms in graphs.items() for arm, fields in arms.items() for key, value in fields.items()})
    step = make_step(model, functools.partial(match_linear, epsilon=c['ot_epsilon'], scale_cost='mean', max_iterations=c['ot_max_iterations']))
    summaries = []; deadline = time.monotonic()+plan['maximum_hours']*3600
    try:
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic() > deadline:
                    raise TimeoutError('Relational contrast deadline')
                fit_run = Artifacts(run.directory/'fits'/f'seed{seed}'/arm, seed=seed, inputs=[Path(configuration), Path(__file__)])
                run.json('status.json', dict(status='training', seed=seed, arm=arm, completed_fits=len(summaries), total_fits=12))
                params, initial, history = fit(model, step, training, graphs, c, plan, arm, seed,
                    lambda row: fit_run.json('status.json', dict(status='training', **row)))
                summaries.append(save_fit(run, fit_run, model, params, initial, history, heldout, c, plan, seed, arm, records))
                print(json.dumps(dict(status='contrast_fit_complete', seed=seed, arm=arm, completed=len(summaries))), flush=True)
        return finish(run, plan, c, split, norm, records, summaries)
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--configuration', required=True); p.add_argument('--output', required=True)
    p.add_argument('--launch', action='store_true'); p.add_argument('--job-directory')
    args = p.parse_args()
    if args.launch:
        if not args.job_directory: p.error('--launch requires --job-directory')
        result = launch(args.configuration, args.output, args.job_directory)
    else: result = program(args.configuration, args.output)
    print(json.dumps(result, indent=2), flush=True)
