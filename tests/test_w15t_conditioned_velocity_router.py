import copy
import functools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15t_conditioned_velocity_router as module
from test_w15n_velocity_auxiliary import fixture


def models():
    g, c, base = fixture()
    plain = base.create_model(c, 3, 0, g['embedding'])
    projection = dict(mean=np.zeros(5, dtype='float32'), matrix=np.eye(5, 2, dtype='float32'))
    model = module.conditioned_model(plain, projection)
    return g, c, base, plain, model


def initialized():
    import jax
    import jax.numpy as jnp
    g, c, base, plain, model = models()
    x = jnp.asarray(g['z'][:4])
    u = jnp.asarray(module.shared_input(g, c['programs'])[:4])
    cond = base.condition(g, 4, c)
    args = (x, u, x, .5, cond)
    params = model.init(jax.random.PRNGKey(0), *args)['params']
    return g, c, base, plain, model, x, u, cond, args, params


def test_condition_projection_only_uses_training_public_embeddings(tmp_path):
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Expression/future used for condition projection')
    rng = np.random.default_rng(22)
    training = [dict(label=f'T{i}', embedding=rng.normal(size=5), target=Forbidden(), z=Forbidden()) for i in range(6)]
    result = module.fit_condition_projection(training, 2)
    x = np.stack([g['embedding'] for g in training])
    projected = (x-result['mean'])@result['matrix']
    np.testing.assert_allclose(projected.mean(0), 0., atol=1e-6)
    np.testing.assert_allclose(projected.std(0), 1., atol=1e-6)
    assert result['training_labels'].tolist() == [f'T{i}' for i in range(6)]
    np.savez_compressed(tmp_path/'projection.npz', **result)
    with np.load(tmp_path/'projection.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['matrix'], result['matrix'])
    with pytest.raises(ValueError):
        module.fit_condition_projection(training+[training[0]], 2)
    with pytest.raises(ValueError, match='rank'):
        module.fit_condition_projection([dict(label=str(i), embedding=np.ones(5)) for i in range(4)], 2)


def test_zero_condition_branch_exactly_preserves_original_initialization():
    import jax
    from flax import serialization
    _, _, _, plain, model, _, _, _, args, params = initialized()
    original = plain.init(jax.random.PRNGKey(0), *args)['params']
    assert serialization.to_bytes(original) == serialization.to_bytes(module.strip_condition_branch(params))
    assert sum(x.size for x in jax.tree_util.tree_leaves(params))-sum(x.size for x in jax.tree_util.tree_leaves(original)) == 2*64
    actual = model.apply(dict(params=params), *args)
    reference = plain.apply(dict(params=original), *args)
    np.testing.assert_array_equal(actual['q'], reference['q'])
    np.testing.assert_array_equal(actual['velocities'], reference['velocities'])


def test_c_can_change_q_without_changing_kinetic_encoder_or_expert_fields():
    import jax.numpy as jnp
    _, _, _, _, model, x, u, cond, _, params = initialized()
    params = copy.deepcopy(params)
    params['router']['condition_hidden']['kernel'] = jnp.arange(128, dtype=jnp.float32).reshape(2, 64)/100
    alternate = dict(cond, perturbation=cond['perturbation']*-.5)
    q1 = model.apply(dict(params=params), x, u, cond, method=model.route)
    q2 = model.apply(dict(params=params), x, u, alternate, method=model.route)
    assert float(jnp.max(jnp.abs(q1-q2))) > 1e-4
    encoded1 = model.apply(dict(params=params), u, method=model.encode_kinetics)
    erased = copy.deepcopy(params)
    erased['router']['condition_hidden']['kernel'] *= 0
    encoded2 = model.apply(dict(params=erased), u, method=model.encode_kinetics)
    np.testing.assert_array_equal(encoded1, encoded2)
    fields1 = model.apply(dict(params=params), x, .5, cond, method=model.velocities)
    fields2 = model.apply(dict(params=erased), x, .5, cond, method=model.velocities)
    np.testing.assert_array_equal(fields1, fields2)
    with pytest.raises(TypeError):
        model.apply(dict(params=params), u, alternate, method=model.encode_kinetics)


def test_graph_gradient_is_router_and_U_encoder_only_including_new_c_branch():
    import jax
    import jax.numpy as jnp
    _, _, _, _, model, x, u, cond, _, params = initialized()
    gradient = jax.grad(lambda p: module.graph_loss(model, p, x, u, x[::-1], u[::-1], cond))(params)
    norm = lambda t: sum(float(jnp.square(v).sum()) for v in jax.tree_util.tree_leaves(t))
    assert norm(gradient['router']['condition_hidden']) > 0
    assert norm(gradient['router']['hidden']) > 0
    assert norm(gradient['kinetic_encoder']) > 0
    assert norm(gradient['backbone']) == 0
    assert all(norm(value) == 0 for key, value in gradient.items() if key.startswith('expert_'))


def test_ot_duplicates_reindex_condition_and_neighbors_consistently():
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    _, _, _, _, model, x, u, cond, _, params = initialized()
    cond = dict(cond, perturbation=cond['perturbation']*jnp.arange(1, 5)[:, None, None])
    params = copy.deepcopy(params)
    params['router']['condition_hidden']['kernel'] = jnp.arange(128, dtype=jnp.float32).reshape(2, 64)/500
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(.001))
    coupling = jnp.zeros((4, 4)).at[2, :].set(.25)
    _, metrics = module.make_step(model, lambda s, t: coupling)(state, jax.random.PRNGKey(9),
        x, x+.2, u, x[::-1], u[::-1], cond, jnp.asarray(.1))
    ix = metrics['source_indices']
    np.testing.assert_array_equal(ix, np.full(4, 2))
    aligned = jax.tree_util.tree_map(lambda value: value[ix], cond)
    expected = module.graph_loss(model, params, x[ix], u[ix], x[::-1][ix], u[::-1][ix], aligned)
    np.testing.assert_allclose(metrics['graph_js'], expected, rtol=1e-5)
    assert metrics['condition_gradient_norm'] > 0


def test_actual_ot_training_learns_condition_and_independent_ODE_prediction():
    import jax
    from flax import serialization
    from cellflow.utils import match_linear
    g, c, base, _, model = models()
    matcher = functools.partial(match_linear, epsilon=.1, scale_cost='mean', max_iterations=20)
    graphs = {g['label']: module.graph.source_graphs(g, 5, 2)}
    params, initial, history = module.graph.fit(model, module.make_step(model, matcher), [g], graphs,
        c, dict(graph_weight_normalized=.1), 'velocity_graph', 0, lambda row: None)
    assert history[-1]['step'] == 2 and history[0]['condition_gradient_norm'] > 0
    assert np.isfinite(history[-1]['loss'])
    assert np.linalg.norm(params['router']['condition_hidden']['kernel']) > 0
    restored = serialization.from_bytes(initial, serialization.to_bytes(params))
    source = dict(g)
    source.pop('target')
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Deployment read velocity')
    for key in ('velocity', 'velocity_program', 'shuffled_velocity_program', 'U_program'):
        source[key] = Forbidden()
    prediction = module.predict(model, base.make_candidate_integrator(model), restored, module.deployment_source(source, 2), c)
    assert prediction['candidates'].shape == (3, 8, 3)
    np.testing.assert_allclose(prediction['q'].sum(1), 1., atol=1e-6)
    assert len(jax.tree_util.tree_leaves(restored)) > 0


def test_all_arms_identical_initialization_and_fixed_zero_extra_slots():
    g, c, _, _, model = models()
    from flax import serialization
    from cellflow.utils import match_linear
    graphs = {g['label']: module.graph.source_graphs(g, 5, 2)}
    step = module.make_step(model, functools.partial(match_linear, epsilon=.1, scale_cost='mean', max_iterations=20))
    hashes = []
    for arm in module.ARMS:
        _, initial, _ = module.graph.fit(model, step, [g], graphs, c, dict(graph_weight_normalized=.1), arm, 0, lambda row: None)
        hashes.append(serialization.to_bytes(initial))
    assert len(set(hashes)) == 1
    assert not module.shared_input(g, 2)[:, 3:].any()


def test_condition_only_gains_cannot_pass_velocity_screen():
    rows = []
    for case in ('original', 'conditioned'):
        for seed in (0, 1, 2):
            for arm in module.ARMS:
                error = 1. if case == 'original' else .5
                if case == 'conditioned' and arm == 'velocity_graph':
                    error = .55
                rows.append(dict(case=case, seed=seed, KO='A', arm=arm, energy=error, mean_mse=1.,
                    variance_ratio=1., log_variance_error=0., effective_experts=8., mean_cell_entropy=.5))
    summary, pairs, interaction = module.summarize(rows, [0, 1, 2], ['A'])
    assert not summary['conditioned_velocity_screen_passed']
    assert not summary['expansion_target_met']
    assert summary['other_comparisons'][0]['energy_gain'] > 0
    assert summary['primary_comparisons'][0]['energy_gain'] < 0
    assert len(pairs) == 15 and len(interaction) == 3


def test_partial_grid_rejected_before_heldout_target_read(tmp_path, monkeypatch):
    _, c, base = fixture()
    monkeypatch.setattr(base, 'load_pack', lambda *a, **k: pytest.fail('Premature target read'))
    with pytest.raises(ValueError, match='entire conditioned'):
        module.evaluate(tmp_path, c, dict(val=['A']), {}, [])
