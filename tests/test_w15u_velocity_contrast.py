import copy
import functools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15u_velocity_contrast as module
from test_w15t_conditioned_velocity_router import models, initialized


def test_contrast_edges_source_only_disjoint_local_and_nonnegative_margins():
    g, _, _, _, _ = models()
    class Forbidden:
        def __array__(self, *args, **kwargs): raise AssertionError('Future target read')
    g['target'] = Forbidden()
    graphs = module.contrast_graphs(g, 6, 2, .1)
    assert set(graphs) == set(module.ARMS)
    for fields in graphs.values():
        for i, (p, n) in enumerate(zip(fields['positive'], fields['negative'])):
            assert i not in p and i not in n and not set(p) & set(n)
        assert fields['margin'].shape == (7, 2)
        assert (fields['margin'] >= 0).all() and (fields['margin'] <= .1).all()


def test_signed_geometry_distinguishes_cell_reversal_but_not_global_reversal():
    g, _, _, _, _ = models()
    before = module.contrast_graphs(g, 6, 2, .1)
    changed = copy.deepcopy(g); changed['velocity_program'][::2] *= -1
    after = module.contrast_graphs(changed, 6, 2, .1)
    np.testing.assert_array_equal(before['unsigned_velocity']['positive'], after['unsigned_velocity']['positive'])
    assert not np.array_equal(before['signed_velocity']['positive'], after['signed_velocity']['positive'])
    changed = copy.deepcopy(g); changed['velocity_program'] *= -1
    global_reverse = module.contrast_graphs(changed, 6, 2, .1)
    np.testing.assert_array_equal(before['signed_velocity']['positive'], global_reverse['signed_velocity']['positive'])


def test_identical_programs_have_zero_margin_without_arbitrary_dynamic_separation():
    g, _, _, _, _ = models(); g['velocity_program'][:] = [1., 0.]
    fields = module.contrast_graphs(g, 6, 2, .1)['signed_velocity']
    np.testing.assert_array_equal(fields['margin'], 0.)


def test_js_each_agrees_with_previous_mean_definition():
    import jax.numpy as jnp
    p = jnp.asarray([[1., 0.], [.2, .8]]); q = p[::-1]
    np.testing.assert_allclose(module.js_each(p, q).mean(), module.graph.normalized_js(p, q), atol=1e-7)
    np.testing.assert_allclose(module.js_each(p, p), 0., atol=1e-7)


def test_contrast_gradient_only_router_and_kinetic_representation():
    import jax
    import jax.numpy as jnp
    _, _, _, _, model, x, u, cond, _, params = initialized()
    grad = jax.grad(lambda p: module.contrast_loss(model, p, x, u, x[::-1], u[::-1], x, u, cond, jnp.full(4, .1))[0])(params)
    norm = lambda t: sum(float(jnp.square(v).sum()) for v in jax.tree_util.tree_leaves(t))
    assert norm(grad['router']) > 0 and norm(grad['kinetic_encoder']) > 0
    assert norm(grad['backbone']) == 0
    assert all(norm(v) == 0 for k, v in grad.items() if k.startswith('expert_'))


def test_actual_contrast_update_and_independent_expert_prediction():
    from cellflow.utils import match_linear
    g, c, base, _, model = models()
    graphs = {g['label']: module.contrast_graphs(g, 6, 2, .1)}
    matcher = functools.partial(match_linear, epsilon=.1, scale_cost='mean', max_iterations=20)
    params, initial, history = module.fit(model, module.make_step(model, matcher), [g], graphs, c,
        dict(graph_weight_normalized=.1), 'signed_velocity', 0, lambda r: None)
    assert np.isfinite(history[-1]['loss']) and history[-1]['step'] == 2
    source = dict(g); source.pop('target')
    p = module.conditional.predict(model, base.make_candidate_integrator(model), params, module.deployment_source(source, 2), c)
    assert p['candidates'].shape == (3, 8, 3)
    np.testing.assert_allclose(p['q'].sum(1), 1., atol=1e-6)


def test_zero_contrast_weight_matches_parent_sampling_and_updates():
    import jax
    from flax import serialization
    from cellflow.utils import match_linear
    g, c, _, _, model = models()
    matcher = functools.partial(match_linear, epsilon=.1, scale_cost='mean', max_iterations=20)
    graphs = {g['label']: module.contrast_graphs(g, 6, 2, .1)}
    fitted, initial, _ = module.fit(model, module.make_step(model, matcher), [g], graphs, c,
        dict(graph_weight_normalized=0.), 'signed_velocity', 0, lambda r: None)
    old_graphs = {g['label']: module.graph.source_graphs(g, 6, 2)}
    original, original_initial, _ = module.graph.fit(model, module.conditional.make_step(model, matcher), [g], old_graphs,
        c, dict(graph_weight_normalized=.1), 'no_graph', 0, lambda r: None)
    assert serialization.to_bytes(initial) == serialization.to_bytes(original_initial)
    for a, b in zip(jax.tree_util.tree_leaves(fitted), jax.tree_util.tree_leaves(original)):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-6)


def test_partial_grid_fails_before_target_read(tmp_path, monkeypatch):
    _, c, base, _, _ = models()
    monkeypatch.setattr(base, 'load_pack', lambda *a, **k: pytest.fail('Premature target read'))
    with pytest.raises(ValueError, match='Whole new'):
        module.evaluate(tmp_path, c, dict(val=['A']), {}, [], dict(arms=list(module.ARMS)))
