"""Train a target-domain S/U count estimator, then freeze its kinetic graphs.

This is a plain count-denoising splicing estimator, NOT original GFG or JVP.
The original state/U inputs and pretrained-free downstream MoE stay unchanged.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts
import w15q_velocity_countsplit as counts
import w15u_velocity_contrast as study

ARMS = ('estimated_velocity', 'shuffled_estimated_velocity', 'estimated_U')


def count_features(s, u, depth, target_sum):
    sn, un = counts.normalized(s, u, depth, target_sum)
    return np.concatenate((np.log1p(sn), np.log1p(un)), axis=1).astype('float32')


def training_views(s, u, depth, target_sum, seeds):
    """Complementary molecule splits, never cell-level biological replicates."""
    x, y, exposure = [], [], []
    for seed in seeds:
        pair = counts.split_counts(s, u, depth, seed)
        for a, b in ((pair[0], pair[1]), (pair[1], pair[0])):
            x.append(count_features(*a, target_sum))
            y.append(np.concatenate(b[:2], axis=1).astype('float32'))
            exposure.append((b[2]/target_sum).astype('float32')[:, None])
    full = count_features(s, u, depth, target_sum)
    mean, scale = full.mean(0), np.maximum(full.std(0), .1)
    sn, un = counts.normalized(s, u, depth, target_sum)
    # Bias starts at train-gene expected rates, not copied per-cell noisy counts.
    log_rates = np.log(np.maximum(np.concatenate((sn.mean(0), un.mean(0))), 1e-4)).astype('float32')
    modality_scale = np.maximum(np.array([s.mean()/2, u.mean()/2], dtype='float32'), .01)
    normalizer = dict(mean=mean, scale=scale, log_rates=log_rates, modality_scale=modality_scale)
    return dict(x=(np.stack(x)-mean)/scale, y=np.stack(y), exposure=np.stack(exposure)), normalizer


def estimator_model(normalizer, width, bottleneck):
    import flax.linen as nn
    import jax.numpy as jnp

    class CountEstimator(nn.Module):
        log_rates: tuple
        width: int
        bottleneck: int

        @nn.compact
        def __call__(self, source_log_counts):
            # There are no c, time, future-state or velocity arguments.
            h = nn.silu(nn.Dense(self.width, name='encoder')(source_log_counts))
            h = nn.silu(nn.Dense(self.bottleneck, name='bottleneck')(h))
            h = nn.silu(nn.Dense(self.width, name='decoder')(h))
            output = nn.Dense(len(self.log_rates), kernel_init=nn.initializers.normal(.001),
                bias_init=lambda key, shape, dtype: jnp.asarray(self.log_rates, dtype=dtype), name='log_rates')(h)
            return jnp.clip(output, -12., 12.)
    return CountEstimator(tuple(map(float, normalizer['log_rates'])), width, bottleneck)


def poisson_loss(log_rates, target_counts, exposure, modality_scale):
    import jax.numpy as jnp
    log_mu = log_rates+jnp.log(exposure)
    work = jnp.exp(log_mu)-target_counts*log_mu
    s, u = jnp.split(work, 2, axis=-1)
    return jnp.mean(s)/modality_scale[0]+jnp.mean(u)/modality_scale[1]


def fit_estimator(views, normalizer, plan, seed, callback):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    model = estimator_model(normalizer, plan['estimator_width'], plan['estimator_bottleneck'])
    initial = model.init(jax.random.PRNGKey(seed), jnp.asarray(views['x'][0, :1]))['params']
    state = TrainState.create(apply_fn=model.apply, params=initial,
        tx=optax.adamw(plan['estimator_learning_rate'], weight_decay=1e-5))

    @jax.jit
    def step(state, x, y, exposure, scales):
        loss, grad = jax.value_and_grad(lambda p: poisson_loss(
            model.apply(dict(params=p), x), y, exposure, scales))(state.params)
        return state.apply_gradients(grads=grad), loss

    data = {key: jnp.asarray(value) for key, value in views.items()}
    scales = jnp.asarray(normalizer['modality_scale'])
    rng, history, started = np.random.default_rng(seed), [], time.monotonic()
    for update in range(1, plan['estimator_steps']+1):
        view = int(rng.integers(len(views['x'])))
        ix = rng.integers(views['x'].shape[1], size=plan['estimator_batch_size'])
        state, loss = step(state, data['x'][view, ix], data['y'][view, ix], data['exposure'][view, ix], scales)
        if update == 1 or update % 250 == 0 or update == plan['estimator_steps']:
            value = float(loss)
            if not np.isfinite(value): raise FloatingPointError('Nonfinite count estimator objective')
            row = dict(step=update, poisson_work_loss=value, elapsed_seconds=time.monotonic()-started)
            history.append(row); callback(row)
    return model, state.params, initial, history


def infer_counts(model, params, s, u, depth, normalizer, target_sum, batch=256):
    import jax
    import jax.numpy as jnp
    x = (count_features(s, u, depth, target_sum)-normalizer['mean'])/normalizer['scale']
    apply = jax.jit(lambda inputs: jnp.exp(model.apply(dict(params=params), inputs)))
    result = []
    for begin in range(0, len(x), batch):
        part = x[begin:begin+batch]; n = len(part)
        part = np.pad(part, ((0, batch-n), (0, 0)), mode='edge')
        result.append(np.asarray(apply(jnp.asarray(part)))[:n])
    value = np.concatenate(result)
    if not np.isfinite(value).all() or (value < 0).any(): raise FloatingPointError('Invalid learned count rates')
    return tuple(np.split(value, 2, axis=1))


def kinetic_projection(s, u, transform):
    """Same frozen RNA/log/PCA geometry as raw velocity; no physical re-scaling."""
    vg = np.where(transform.supported, (u-s*transform.gamma)/(1+s), 0.)
    v = (vg@transform.components.T)/transform.velocity_scale
    U = np.log1p(u)@transform.components.T-transform.u_mean
    return v.astype('float32'), U.astype('float32')


def estimated_groups(training, source_ids, velocity, U, norm, shuffle_seed):
    from w15i_program_router_followups import cell_programs
    lookup = {str(k): i for i, k in enumerate(source_ids)}
    result = []
    for original in training:
        g = dict(original); ix = np.array([lookup[str(k)] for k in g['cell_ids']])
        g['velocity'], g['U'] = velocity[ix], U[ix]
        # Crucial: U_normalized is NOT changed. Main model sees original S/U.
        result.append(g)
    learned_norm = dict(norm, velocity_reference=np.mean([g['velocity'].astype(float).mean(0) for g in result], axis=0),
                        U_reference=np.mean([g['U'].astype(float).mean(0) for g in result], axis=0))
    return cell_programs(result, learned_norm, shuffle_seed)[0], learned_norm


def program(configuration, output):
    plan, c, run, training, heldout, norm, split, model, records = study.prepare_run(configuration, output, __file__)
    import jax
    from flax import serialization
    from cellflow.utils import match_linear
    from veloroute.latent import load_pack, FrozenSplicingTransform
    from w7_router_scaling import write_csv
    from w15i_program_router_followups import cell_programs
    source, _ = load_pack(Path(c['fold'])/'train_source.npz', expected_side='source')
    transform = FrozenSplicingTransform.load(Path(c['fold'])/'transform.npz')
    run.json('status.json', dict(status='reading_training_source_counts', total_fits=9, total_estimators=3))
    s, u, depth = counts.read_selected(source, transform, plan['raw_directory'])
    sn, un = counts.normalized(s, u, depth, transform.target_sum)
    raw_v, raw_U = kinetic_projection(sn, un, transform)
    np.testing.assert_allclose(raw_v, source['velocity'], rtol=0, atol=2e-6)
    np.testing.assert_allclose(raw_U, source['u_features'], rtol=0, atol=2e-6)
    np.savez_compressed(run.directory/'training_counts.npz', S=s, U=u, depth=depth, cell_ids=source['cell_ids'])
    views, normalizer = training_views(s, u, depth, transform.target_sum, plan['count_split_seeds'])
    np.savez_compressed(run.directory/'count_normalizer.npz', **normalizer)
    raw_groups = cell_programs(training, norm, plan['shuffle_seed'])[0]
    step = study.conditional.make_step(model, functools.partial(match_linear, epsilon=c['ot_epsilon'], scale_cost='mean', max_iterations=c['ot_max_iterations']))
    summaries, estimator_summaries, diagnostics = [], [], []
    deadline = time.monotonic()+plan['maximum_hours']*3600
    try:
        for seed in c['seeds']:
            erun = Artifacts(run.directory/'estimators'/f'seed{seed}', seed=seed,
                inputs=[Path(configuration), Path(__file__), Path(c['fold'])/'train_source.npz'])
            run.json('status.json', dict(status='training_count_estimator', seed=seed, completed_fits=len(summaries), total_fits=9))
            estimator, ep, initial, ehistory = fit_estimator(views, normalizer, plan, seed,
                lambda row: erun.json('status.json', dict(status='training', **row)))
            (erun.directory/'model.msgpack').write_bytes(serialization.to_bytes(ep))
            shat, uhat = infer_counts(estimator, ep, s, u, depth, normalizer, transform.target_sum)
            velocity, U = kinetic_projection(shat, uhat, transform)
            groups, learned_norm = estimated_groups(training, source['cell_ids'], velocity, U, norm, plan['shuffle_seed'])
            np.savez_compressed(erun.directory/'source_estimates.npz', cell_ids=source['cell_ids'], velocity=velocity, U=U,
                                velocity_reference=learned_norm['velocity_reference'], U_reference=learned_norm['U_reference'])
            graphs = {}
            for g, raw in zip(groups, raw_groups):
                built = study.graph.source_graphs(g, plan['state_candidates'], plan['graph_neighbors'])
                graphs[g['label']] = dict(estimated_velocity=built['velocity_graph'],
                    shuffled_estimated_velocity=built['shuffled_velocity_graph'], estimated_U=built['U_graph'])
                for name, key in [('velocity', 'velocity'), ('U', 'U')]:
                    projected = g[key]@norm['axes'].T; original = raw[key]@norm['axes'].T
                    diagnostics.append(dict(seed=seed, KO=g['label'], modality=name,
                        within_KO_program_variance_retained=float(np.var(projected, axis=0).sum()/max(np.var(original, axis=0).sum(), 1e-12))))
            np.savez_compressed(erun.directory/'source_graphs.npz', **{f'{ko}__{a}': v for ko, arms in graphs.items() for a, v in arms.items()})
            esummary = dict(seed=seed, steps=plan['estimator_steps'], parameters=sum(v.size for v in jax.tree_util.tree_leaves(ep)),
                first_work_loss=ehistory[0]['poisson_work_loss'], last_work_loss=ehistory[-1]['poisson_work_loss'],
                original_GFG=False, JVP=False, c_input=False, future_target_input=False,
                learned_count_denoiser=True, kinetic_formula='frozen_beta1_gamma_residual_after_count_estimation')
            erun.json('summary.json', esummary); erun.json('training_history.json', dict(history=ehistory))
            erun.json('status.json', dict(status='complete')); estimator_summaries.append(esummary)
            for arm in ARMS:
                if time.monotonic() > deadline: raise TimeoutError('Target estimator experiment deadline')
                fit_run = Artifacts(run.directory/'fits'/f'seed{seed}'/arm, seed=seed, inputs=[Path(configuration), Path(__file__)])
                run.json('status.json', dict(status='training_router', seed=seed, arm=arm, completed_fits=len(summaries), total_fits=9))
                params, initial, history = study.graph.fit(model, step, groups, graphs, c, plan, arm, seed,
                    lambda row: fit_run.json('status.json', dict(status='training', **row)))
                summaries.append(study.save_fit(run, fit_run, model, params, initial, history, heldout, c, plan, seed, arm, records))
                print(json.dumps(dict(status='estimated_graph_fit_complete', seed=seed, arm=arm, completed=len(summaries))), flush=True)
        run.json('estimator_summary.json', dict(estimators=estimator_summaries))
        write_csv(run.directory/'source_variance_diagnostics.csv', diagnostics)
        return study.finish(run, plan, c, split, norm, records, summaries)
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--configuration', required=True); p.add_argument('--output', required=True)
    p.add_argument('--launch', action='store_true'); p.add_argument('--job-directory'); args = p.parse_args()
    if args.launch:
        if not args.job_directory: p.error('--launch requires --job-directory')
        result = study.launch(args.configuration, args.output, args.job_directory, script=__file__)
    else: result = program(args.configuration, args.output)
    print(json.dumps(result, indent=2), flush=True)
