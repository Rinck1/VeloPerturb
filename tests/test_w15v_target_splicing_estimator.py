from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15v_target_splicing_estimator as module


def fixture():
    rng = np.random.default_rng(3)
    s = rng.poisson(8., size=(12, 3))
    u = rng.poisson(3., size=(12, 3))
    depth = s.sum(1)+25
    return s, u, depth


def test_complementary_views_preserve_raw_counts_and_exposures():
    s, u, d = fixture()
    views, norm = module.training_views(s, u, d, 100., [0, 1])
    assert views['x'].shape == (4, 12, 6)
    np.testing.assert_array_equal(views['y'][0]+views['y'][1], np.concatenate((s, u), axis=1))
    np.testing.assert_allclose(views['exposure'][0]+views['exposure'][1], (d/100.)[:, None], atol=1e-7)
    assert np.isfinite(views['x']).all() and (views['y'] >= 0).all()
    np.testing.assert_array_equal(norm['mean'], module.count_features(s, u, d, 100.).mean(0))


def test_U_input_uses_S_depth_not_its_own_library_size():
    s, u, d = fixture()
    a = module.count_features(s, u, d, 100.)
    b = module.count_features(s, u*2, d, 100.)
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    assert not np.allclose(a[:, 3:], b[:, 3:])


def test_poisson_work_loss_exposure_and_stationary_derivative():
    import jax
    import jax.numpy as jnp
    log = jnp.log(jnp.full((2, 6), 2.))
    y = jnp.full((2, 6), 4.)
    exposure = jnp.full((2, 1), 2.)
    scales = jnp.ones(2)
    grad = jax.grad(lambda z: module.poisson_loss(z, y, exposure, scales))(log)
    np.testing.assert_allclose(grad, 0., atol=1e-7)
    value = module.poisson_loss(log, y, exposure, scales)
    np.testing.assert_allclose(value, 2*(4-4*np.log(4)), rtol=1e-6)


def test_actual_estimator_training_inference_and_no_condition_argument():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    s, u, d = fixture()
    views, norm = module.training_views(s, u, d, 100., [0])
    plan = dict(estimator_width=8, estimator_bottleneck=3, estimator_steps=3,
                estimator_batch_size=4, estimator_learning_rate=.001)
    model, params, initial, history = module.fit_estimator(views, norm, plan, 0, lambda r: None)
    assert history[-1]['step'] == 3 and np.isfinite(history[-1]['poisson_work_loss'])
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(initial)))
    restored = serialization.from_bytes(initial, serialization.to_bytes(params))
    shat, uhat = module.infer_counts(model, restored, s, u, d, norm, 100., batch=5)
    assert shat.shape == s.shape and uhat.shape == u.shape
    assert (shat > 0).all() and (uhat > 0).all()
    np.testing.assert_allclose(np.concatenate((shat, uhat), axis=1),
        np.exp(model.apply(dict(params=params), (module.count_features(s, u, d, 100.)-norm['mean'])/norm['scale'])), rtol=1e-6)
    with pytest.raises(TypeError):
        model.apply(dict(params=params), jnp.ones((2, 6)), dict(perturbation=jnp.ones((2, 1, 3))))


def test_kinetic_projection_same_chain_rule_and_support_mask():
    t = SimpleNamespace(gamma=np.array([1., 2., 1.]), supported=np.array([True, True, False]),
        components=np.eye(3), velocity_scale=2., u_mean=np.array([.1, .2, .3]))
    s = np.array([[2., 3., 5.]])
    u = np.array([[4., 2., 100.]])
    v, U = module.kinetic_projection(s, u, t)
    np.testing.assert_allclose(v, [[1/3, -.5, 0.]], atol=1e-7)
    np.testing.assert_allclose(U, np.log1p(u)-t.u_mean, rtol=1e-6)


def test_learned_teacher_does_not_replace_main_model_U_or_state_inputs():
    from test_w15t_conditioned_velocity_router import models
    g, _, _, _, _ = models()
    norm = dict(axes=np.eye(3)[:2], velocity_reference=np.zeros(3), U_reference=np.zeros(3))
    rng = np.random.default_rng(1)
    v, U = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    before = module.study.shared_input(g, 2)
    groups, trained = module.estimated_groups([g], g['cell_ids'], v, U, norm, 0)
    np.testing.assert_array_equal(module.study.shared_input(groups[0], 2), before)
    np.testing.assert_array_equal(groups[0]['z'], g['z'])
    np.testing.assert_array_equal(groups[0]['velocity'], v)
    np.testing.assert_array_equal(groups[0]['U'], U)
    np.testing.assert_allclose(trained['velocity_reference'], v.mean(0))
    assert groups[0]['velocity_program'].shape == (7, 2)


def test_target_library_size_cannot_change_input_features():
    s, u, d = fixture()
    views, norm = module.training_views(s, u, d, 100., [0])
    x = views['x'].copy()
    views['exposure'] *= 100
    np.testing.assert_array_equal(views['x'], x)
    assert not np.any(norm['scale'] <= 0)
