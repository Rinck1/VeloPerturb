"""Matched perturbation-conditioned routing with training-only kinetic graphs.

Only q gains explicit c. Expert ODEs, source S/U, graph construction and the
training budget stay fixed. GFG and JVP are not involved in this experiment.
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
from w15n_velocity_auxiliary import shared_input, deployment_source
import w15o_velocity_neighborhood as graph

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ('no_graph', 'velocity_graph', 'shuffled_velocity_graph', 'U_graph')


def fit_condition_projection(training, components):
    """One public embedding per TRAIN KO; no expression/future/validation input."""
    labels = [g['label'] for g in training]
    if len(set(labels)) != len(labels):
        raise ValueError('One condition embedding per training KO required')
    x = np.stack([np.asarray(g['embedding'], dtype='float64') for g in training])
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= components < min(x.shape):
        raise ValueError('Invalid training condition projection')
    mean = x.mean(0)
    _, singular, axes = np.linalg.svd(x-mean, full_matrices=False)
    axes = axes[:components].copy()
    if singular[components-1] <= max(singular[0], 1.)*1e-10:
        raise ValueError('Insufficient training condition rank')
    for row in axes:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    scales = np.maximum(((x-mean)@axes.T).std(0), 1e-6)
    return dict(mean=mean.astype('float32'), matrix=(axes.T/scales).astype('float32'),
                training_labels=np.asarray(labels, dtype=str), singular_values=singular[:components])


def conditioned_model(base_model, projection):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from w15a_cellflow_moe import KineticMoE, KineticEncoder, ExpertHead, state_condition_only

    class ConditionalRouter(nn.Module):
        experts: int
        width: int

        @nn.compact
        def __call__(self, source, encoded, context):
            hidden = nn.Dense(self.width, name='hidden')(jnp.concatenate((source, encoded), axis=-1))
            shift = nn.Dense(self.width, use_bias=False, kernel_init=nn.initializers.zeros,
                             name='condition_hidden')(context)
            return jax.nn.softmax(nn.Dense(self.experts, name='logits')(nn.silu(hidden+shift)), axis=-1)

    class ConditionalMoE(KineticMoE):
        condition_mean: tuple = ()
        condition_matrix: tuple = ()

        def setup(self):
            c = self.config
            self.kinetic_encoder = KineticEncoder(c.kinetic_width, c.kinetic_embedding)
            self.router = ConditionalRouter(c.experts, c.router_width)
            self.expert_heads = tuple(ExpertHead(c.dimensions, c.expert_width, name=f'expert_{k}')
                                      for k in range(c.experts))

        def route(self, source, kinetic_input, condition):
            state_condition_only(condition)
            embedding = condition['perturbation']
            if embedding.ndim != 3 or embedding.shape[1] != 1 or len(embedding) != len(source):
                raise ValueError('Router requires aligned single-KO condition tokens')
            context = (embedding[:, 0]-jnp.asarray(self.condition_mean))@jnp.asarray(self.condition_matrix)
            # Conditions have no path into the kinetic encoder.
            return self.router(source, self.kinetic_encoder(kinetic_input), context)

        def __call__(self, source, kinetic_input, moving_state, tau, condition):
            return dict(q=self.route(source, kinetic_input, condition),
                        velocities=self.velocities(moving_state, tau, condition))

    return ConditionalMoE(backbone=base_model.backbone, config=base_model.config,
        condition_mean=tuple(map(float, projection['mean'])),
        condition_matrix=tuple(tuple(map(float, row)) for row in projection['matrix']))


def strip_condition_branch(params):
    result = copy.deepcopy(params)
    del result['router']['condition_hidden']
    return result


def graph_loss(model, params, source, kinetic, neighbor_source, neighbor_kinetic, condition):
    p = model.apply(dict(params=params), source, kinetic, condition, method=model.route)
    # Graphs are within KO and within source time: neighbors carry the SAME c.
    q = model.apply(dict(params=params), neighbor_source, neighbor_kinetic, condition, method=model.route)
    return graph.normalized_js(p, q)


def make_step(model, match_fn):
    import jax
    import jax.numpy as jnp
    from ott.solvers import utils as solver_utils
    from w15a_cellflow_moe import align_ot_indices_jit, paired_flow_matching_loss

    @jax.jit
    def step(state, rng, source, target, kinetic, neighbor_source, neighbor_kinetic, condition, weight):
        pair_key, tau_key = jax.random.split(rng)
        si, ti = solver_utils.sample_joint(pair_key, match_fn(source, target))
        paired = align_ot_indices_jit(source, kinetic, target, si, ti, condition=condition)
        tau = jax.random.uniform(tau_key, (len(si), 1), minval=0., maxval=1.)

        def objective(params):
            fm, metrics = paired_flow_matching_loss(model, dict(params=params), paired, tau)
            relation = graph_loss(model, params, paired.source_x, paired.kinetic_input,
                                  neighbor_source[si], neighbor_kinetic[si], paired.condition)
            contribution = source.shape[1]*weight*relation
            return fm+contribution, dict(metrics, fm_loss=fm, graph_js=relation, graph_contribution=contribution)

        (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(state.params)
        context_norm = jnp.sqrt(jnp.sum(gradient['router']['condition_hidden']['kernel']**2))
        return state.apply_gradients(grads=gradient), dict(metrics, loss=loss, condition_gradient_norm=context_norm,
                                                          source_indices=si, target_indices=ti)
    return step


def predict(model, integrate, params, source, c):
    """Source S/U+c only. q is fixed across the independent expert trajectories."""
    import jax.numpy as jnp
    import w15h_traxler_program_router as base
    if 'target' in source:
        raise ValueError('Prediction must not receive future targets')
    batch = c['prediction_batch']
    all_q = []
    for begin in range(0, len(source['z']), batch):
        end = min(begin+batch, len(source['z']))
        n = end-begin
        pad = lambda value: np.pad(value[begin:end], ((0, batch-n), (0, 0)), mode='edge')
        q = model.apply(dict(params=params), jnp.asarray(pad(source['z'])),
                        jnp.asarray(pad(source['kinetic_input'])), base.condition(source, batch, c), method=model.route)
        all_q.append(np.asarray(q)[:n])
    full_q = np.concatenate(all_q)
    np.testing.assert_allclose(full_q.sum(1), 1., atol=1e-6)
    indices = source['score_indices']
    candidates = []
    for begin in range(0, len(indices), batch):
        ix = indices[begin:begin+batch]
        z = np.pad(source['z'][ix], ((0, batch-len(ix)), (0, 0)), mode='edge')
        candidates.append(np.asarray(integrate(params, jnp.asarray(z), base.condition(source, batch, c), 1.))[:len(ix)])
    return dict(candidates=np.concatenate(candidates), q=full_q[indices], source_ids=source['cell_ids'][indices],
                source_indices=indices, full_source_ids=source['cell_ids'], full_source_q=full_q,
                full_source_mean_q=full_q.mean(0))


def verify_grid(records, c, split, projection_path=None):
    expected = {(case, seed, arm, ko) for case in ('conditioned', 'original')
                for seed in c['seeds'] for arm in ARMS for ko in split['val']}
    if len(records) != len(expected) or {(r['case'], r['seed'], r['arm'], r['KO']) for r in records} != expected:
        raise ValueError('Freeze entire conditioned and reference grid before evaluation')
    checked = {}
    for r in records:
        for path, digest in ((r['path'], r['sha256']), (r['model'], r['model_sha256'])):
            if path not in checked:
                checked[path] = sha256(path)
            if checked[path] != digest:
                raise ValueError('Frozen conditioned/reference artifact changed')
        if r['case'] == 'conditioned' and (projection_path is None or sha256(projection_path) != r['condition_projection_sha256']):
            raise ValueError('Frozen condition projection changed')


def summarize(rows, seeds, labels):
    """Paired equal-KO, equal-seed effects; c effects are NOT velocity effects."""
    by = {(r['case'], r['seed'], r['KO'], r['arm']): r for r in rows}
    pairs, comparisons = [], []

    def compare(name, real_case, real_arm, control_case, control_arm):
        selected = []
        for seed in seeds:
            for ko in labels:
                real = by[(real_case, seed, ko, real_arm)]
                control = by[(control_case, seed, ko, control_arm)]
                row = dict(comparison=name, seed=seed, KO=ko, energy_gain=1-real['energy']/control['energy'])
                pairs.append(row)
                selected.append(row)
        per_seed = {str(s): float(np.mean([r['energy_gain'] for r in selected if r['seed'] == s])) for s in seeds}
        per_ko = {ko: float(np.mean([r['energy_gain'] for r in selected if r['KO'] == ko])) for ko in labels}
        item = dict(comparison=name, energy_gain=float(np.mean([r['energy_gain'] for r in selected])),
                    by_seed=per_seed, by_KO=per_ko, winning_KOs=sum(v > 0 for v in per_ko.values()))
        comparisons.append(item)
        return item

    primary = [compare('velocity_vs_conditioned_'+arm, 'conditioned', 'velocity_graph', 'conditioned', arm)
               for arm in ARMS if arm != 'velocity_graph']
    c_only = compare('condition_only_no_graph', 'conditioned', 'no_graph', 'original', 'no_graph')
    joint = compare('conditioned_vs_original_velocity_graph', 'conditioned', 'velocity_graph', 'original', 'velocity_graph')
    interaction = []
    for seed in seeds:
        for ko in labels:
            get = lambda case, arm: by[(case, seed, ko, arm)]['energy']
            old = get('original', 'no_graph')-get('original', 'velocity_graph')
            new = get('conditioned', 'no_graph')-get('conditioned', 'velocity_graph')
            interaction.append(dict(seed=seed, KO=ko, original_velocity_absolute_gain=old,
                                    conditioned_velocity_absolute_gain=new, difference=new-old))
    stable = all(r['energy_gain'] > 0 and all(v > 0 for v in r['by_seed'].values()) for r in primary)
    keys = ('energy', 'mean_mse', 'variance_ratio', 'log_variance_error', 'effective_experts', 'mean_cell_entropy')
    means = {case: {arm: {key: float(np.mean([r[key] for r in rows if r['case'] == case and r['arm'] == arm]))
                         for key in keys} for arm in ARMS} for case in ('conditioned', 'original')}
    return dict(primary_comparisons=primary, other_comparisons=[c_only, joint],
                conditioned_velocity_screen_passed=stable,
                expansion_target_met=stable and primary[0]['energy_gain'] >= .02 and joint['energy_gain'] > 0,
                mean_absolute_velocity_increment_difference=float(np.mean([r['difference'] for r in interaction])),
                means=means), pairs, interaction


def evaluate(output, c, split, norm, records):
    import w15h_traxler_program_router as base
    from w15m_velocity_discrimination import empirical_metrics
    from w7_router_scaling import write_csv
    verify_grid(records, c, split, output/'condition_projection.npz')
    targets, metadata = base.load_pack(Path(c['fold'])/'validation_target.npz', expected_side='target')
    if metadata.get('day') != c['target_hours'] or set(targets['conditions']) != set(split['val']):
        raise ValueError('Wrong development target identity')
    rows = []
    for ko in sorted(split['val']):
        y = targets['z'][targets['conditions'] == ko]
        y = y[np.random.default_rng(c['evaluation_seed']).choice(len(y), min(len(y), c['evaluation_cells']), replace=False)]
        ids = None
        for r in [r for r in records if r['KO'] == ko]:
            p = base.read_npz(r['path'])
            if ids is None:
                ids = p['source_ids']
            else:
                np.testing.assert_array_equal(ids, p['source_ids'])
            q = np.asarray(p['full_source_q'], dtype='float64')
            avg = q.mean(0)
            rows.append(dict(case=r['case'], KO=ko, seed=r['seed'], arm=r['arm'],
                effective_experts=float(np.exp(-np.sum(avg*np.log(np.maximum(avg, 1e-12))))),
                mean_cell_entropy=float(-np.sum(q*np.log(np.maximum(q, 1e-12)), axis=1).mean()),
                **empirical_metrics(p['candidates'], p['q'], y, norm['axes'])))
    result, pairs, interaction = summarize(rows, c['seeds'], split['val'])
    write_csv(output/'metrics.csv', rows)
    write_csv(output/'comparisons.csv', pairs)
    write_csv(output/'velocity_condition_interaction.csv', interaction)
    return result


def launch(configuration, output, job_directory):
    from veloroute.artifacts import save_json
    from veloroute.gpu_policy import gpu_uuids, ALLOWED_PHYSICAL_GPUS, RESERVED_PHYSICAL_GPUS, MAX_VISIBLE_GPUS
    from w15i_followup_queue import worker_environment
    directory, output = Path(job_directory).resolve(), Path(output).resolve()
    if directory.exists() or output.exists() or not directory.is_relative_to(ROOT/'outputs'):
        raise ValueError('New output and project-local job directory required')
    directory.mkdir(parents=True)
    env = worker_environment(output.parent/'runtime_gpu0')
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    job = dict(physical_gpu=0, gpu_uuid=gpu_uuids()[0], cwd=str(ROOT),
        command=[sys.executable, str(Path(__file__).resolve()), '--configuration', str(Path(configuration).resolve()), '--output', str(output)],
        allowed_gpus=list(ALLOWED_PHYSICAL_GPUS), reserved_gpus=list(RESERVED_PHYSICAL_GPUS), max_visible_gpus=MAX_VISIBLE_GPUS)
    save_json(directory/'job.json', job)
    with (directory/'job.log').open('xb') as stream:
        child = subprocess.Popen([sys.executable, str(HERE/'launch_gpu_job.py'), '--worker', str(directory)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(directory/'launcher.json', dict(pid=child.pid, physical_gpu=0, detached=True))
    return dict(status='LAUNCHED', pid=child.pid, physical_gpu=0, job_directory=str(directory))


def program(configuration, output):
    from w15k_traxler_l2_cell import wait_for_free_gpu
    from veloroute.gpu_policy import enforce_gpu_policy
    plan = json.loads(Path(configuration).read_text())
    if tuple(plan['arms']) != ARMS or plan['gpu'] != 0:
        raise ValueError('Unregistered conditional-router grid')
    wait_for_free_gpu(0, time.monotonic()+600, lambda s: print(json.dumps(s), flush=True))
    enforce_gpu_policy('cuda')
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from cellflow.utils import match_linear
    from w15i_program_router_followups import cell_programs

    c = json.loads(Path(plan['base_configuration']).read_text())
    if (c['seeds'], c['steps'], c['experts']) != (plan['model_seeds'], plan['steps'], plan['experts']):
        raise ValueError('Parent training budget changed')
    parent = Path(plan['normalizer_parent'])
    graph_parent = Path(plan['graph_parent'])
    inputs = [Path(configuration), Path(plan['base_configuration']), ROOT/'PLANS/w15t_conditioned_velocity_router_20261022.md',
        Path(__file__), HERE/'w15o_velocity_neighborhood.py', HERE/'w15n_velocity_auxiliary.py',
        HERE/'w15h_traxler_program_router.py', HERE/'w15i_program_router_followups.py', HERE/'w15m_velocity_discrimination.py',
        HERE/'w15a_cellflow_moe.py', HERE/'w15a_cellflow_moe_official.py', HERE/'w15a_cellflow_moe_real_predict.py',
        HERE/'w15c_renge_l3.py', HERE/'w15a_cellflow_adapter.py', HERE/'w15a_cellflow_train.py', HERE/'w4_artifacts.py',
        Path(c['split']), Path(c['conditions']), parent/'training_programs.npz', graph_parent/'whole_grid_frozen.json',
        graph_parent/'source_graphs.npz', Path(c['fold'])/'train_source.npz', Path(c['fold'])/'train_target.npz',
        Path(c['fold'])/'validation_source.npz', Path(c['fold'])/'transform.npz']
    run = Artifacts(output, seed=c['seeds'], inputs=inputs)
    deadline = time.monotonic()+plan['maximum_hours']*3600
    planned = len(ARMS)*len(c['seeds'])
    try:
        training, heldout, norm, split, _ = base.prepare(c)
        saved = base.read_npz(parent/'training_programs.npz')
        for key in norm:
            np.testing.assert_array_equal(norm[key], saved[key])
        projection = fit_condition_projection(training, plan['condition_components'])
        projection_path = run.directory/'condition_projection.npz'
        np.savez_compressed(projection_path, **projection)
        projection_hash = sha256(projection_path)
        training, _ = cell_programs(training, norm, plan['shuffle_seed'])
        graphs = {g['label']: graph.source_graphs(g, plan['state_candidates'], plan['graph_neighbors']) for g in training}
        old_graphs = base.read_npz(graph_parent/'source_graphs.npz')
        for ko, arms in graphs.items():
            for arm in ARMS:
                np.testing.assert_array_equal(arms[arm], old_graphs[f'{ko}__{arm}'])
        np.savez_compressed(run.directory/'source_graphs.npz', **{f'{ko}__{a}': x for ko, arms in graphs.items() for a, x in arms.items() if a in ARMS})
        np.savez_compressed(run.directory/'training_programs.npz', **norm)
        records = [dict(r, case='original') for r in json.loads((graph_parent/'whole_grid_frozen.json').read_text())['predictions'] if r['arm'] in ARMS]
        run.json('study_manifest.json', dict(plan=plan, config=c, planned_fits=planned,
            train_KOs=split['train'], validation_KOs=split['val'], reserved_test_KOs=split['test'],
            validation_previously_used_for_development=True, independent_confirmation=False,
            condition_projection_training_KOs=projection['training_labels'].tolist(),
            velocity_never_a_forward_input=True, graph_source_only=True, GFG=False, JVP=False, test_targets_read=False))
        plain = base.create_model(c, training[0]['z'].shape[1], 0, training[0]['embedding'])
        model = conditioned_model(plain, projection)
        matcher = functools.partial(match_linear, epsilon=c['ot_epsilon'], scale_cost='mean', max_iterations=c['ot_max_iterations'])
        step, integrate = make_step(model, matcher), base.make_candidate_integrator(model)
        summaries, initializations = [], {}
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic() > deadline:
                    raise TimeoutError('Conditional-router deadline exceeded')
                directory = run.directory/'fits'/f'seed{seed}'/arm
                fit_run = Artifacts(directory, seed=seed, inputs=[Path(configuration), Path(__file__)])
                run.json('status.json', dict(status='training', seed=seed, arm=arm, completed_fits=len(summaries), total_fits=planned))
                params, initial, history = graph.fit(model, step, training, graphs, c, plan, arm, seed,
                    lambda row: fit_run.json('status.json', dict(status='training', **row)))
                initial_hash = hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if seed in initializations and initializations[seed] != initial_hash:
                    raise ValueError('Matched conditional initialization changed')
                initializations[seed] = initial_hash
                parent_hash = json.loads((graph_parent/'fits'/f'seed{seed}'/'no_graph'/'summary.json').read_text())['initial_parameters_sha256']
                if hashlib.sha256(serialization.to_bytes(strip_condition_branch(initial))).hexdigest() != parent_hash:
                    raise ValueError('Original initialization not preserved under zero c branch')
                checkpoint = directory/'model.msgpack'
                checkpoint.write_bytes(serialization.to_bytes(params))
                params = serialization.from_bytes(initial, checkpoint.read_bytes())
                kernel = params['router']['condition_hidden']['kernel']
                if not float(np.linalg.norm(kernel)) > 0:
                    raise ValueError('Condition branch did not learn')
                for source in heldout:
                    prediction = predict(model, integrate, params, deployment_source(source, c['programs']), c)
                    path = directory/(source['label']+'.npz')
                    np.savez_compressed(path, **prediction)
                    records.append(dict(KO=source['label'], seed=seed, arm=arm, case='conditioned', path=str(path),
                        sha256=sha256(path), model=str(checkpoint), model_sha256=sha256(checkpoint),
                        condition_projection_sha256=projection_hash, explicit_velocity_input=False, heldout_future_used=False))
                summary = dict(status='complete', seed=seed, arm=arm, steps=c['steps'],
                    parameters=sum(v.size for v in jax.tree_util.tree_leaves(params)), initial_parameters_sha256=initial_hash,
                    condition_kernel_norm=float(np.linalg.norm(kernel)), condition_gradient_first=history[0]['condition_gradient_norm'],
                    graph_js_first=history[0]['graph_js'], graph_js_last=history[-1]['graph_js'])
                fit_run.json('training_history.json', dict(history=history))
                fit_run.json('summary.json', summary)
                fit_run.json('status.json', dict(status='complete'))
                summaries.append(summary)
                print(json.dumps(dict(status='conditioned_router_fit_complete', seed=seed, arm=arm,
                                      completed=len(summaries), total=planned)), flush=True)
        verify_grid(records, c, split, projection_path)
        run.json('whole_grid_frozen.json', dict(predictions=records, validation_targets_read_before_freeze=False, test_targets_read=False))
        run.json('status.json', dict(status='evaluating', completed_fits=planned, total_fits=planned))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            results = evaluate(run.directory, c, split, norm, records)
        result = dict(status='CONDITIONED_VELOCITY_ROUTER_COMPLETE', new_fits=planned, new_predictions=planned*len(split['val']),
            frozen_predictions=len(records), parameters=summaries[0]['parameters'], fit_diagnostics=summaries,
            GFG=False, JVP=False, test_targets_read=False, explicit_velocity_input=False, **results)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Perturbation-conditioned velocity router\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete', completed_fits=planned, total_fits=planned))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--job-directory')
    args = parser.parse_args()
    if args.launch:
        if not args.job_directory:
            parser.error('--launch requires --job-directory')
        result = launch(args.configuration, args.output, args.job_directory)
    else:
        result = program(args.configuration, args.output)
    print(json.dumps(result, indent=2), flush=True)
