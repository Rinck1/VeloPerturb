"""Actual official-CellFlow hidden integration tests, run on CPU only."""
from pathlib import Path
import functools
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase0"))
from w15a_cellflow_adapter import add_cellflow_paths
add_cellflow_paths()

from cellflow.networks import ConditionalVelocityField
from cellflow.utils import match_linear
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import serialization

from w15a_cellflow_moe import (MoEConfig, arm_input, integrate_mixture, weighted_energy)
from w15a_cellflow_moe_official import (HiddenFeatureField, hidden_field_from_native,
    official_moe_from_native, make_official_ot_train_step, official_source_record)


@pytest.fixture
def official():
    from w15a_cellflow_train import create_solver
    config = dict(width=8, depth=1, condition_dim=4, time_freqs=8, time_width=8,
                  time_depth=1, pooling="mean", learning_rate=.001,
                  ot_epsilon=.1, ot_max_iterations=30)
    state_condition = dict(perturbation=jnp.ones((4, 1, 1)),
        bio_time=jnp.full((4, 1, 8), .2), bio_delta=jnp.full((4, 1, 8), .4))
    full = dict(state_condition, population_token=jnp.zeros((4, 1, 2)),
                cell_token=jnp.zeros((4, 1, 2)))
    solver = create_solver(config, {k:v[:1] for k,v in full.items()}, 2, 0)
    x = jnp.array([[.1, -.1], [.2, .5], [-.2, .8], [1., -.5]], dtype=jnp.float32)
    tau = jnp.full((4, 1), .3)
    noise = jnp.zeros((4, solver.vf.condition_embedding_dim))
    return solver.vf, solver.vf_state.params, x, tau, noise, state_condition, full


def test_official_decoder_hidden_reconstructs_native_projection(official):
    native, params, x, tau, noise, _, full = official
    hidden_field = hidden_field_from_native(native)
    hidden, mean, logvar = hidden_field.apply(dict(params=params), tau, x, full, noise, train=False)
    actual, actual_mean, actual_logvar = native.apply(dict(params=params), tau, x, full, noise, train=False)
    projected = hidden @ params["output_layer"]["kernel"] + params["output_layer"]["bias"]
    np.testing.assert_allclose(projected, actual, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(mean, actual_mean)
    np.testing.assert_array_equal(logvar, actual_logvar)
    assert hidden.shape == (len(x), native.decoder_dims[-1])


def test_hidden_subclass_inherits_native_encoders_and_has_no_projection_params(official):
    native, _, x, tau, noise, _, full = official
    hidden = hidden_field_from_native(native)
    key = jax.random.PRNGKey(20)
    hparams = hidden.init(key, tau, x, full, noise, train=False)["params"]
    native_params = native.init(key, tau, x, full, noise, train=False)["params"]
    assert "output_layer" not in hparams
    assert set(native_params) - set(hparams) == {"output_layer"}
    for key in hparams:
        for a, b in zip(jax.tree_util.tree_leaves(hparams[key]), jax.tree_util.tree_leaves(native_params[key])):
            np.testing.assert_array_equal(a, b)


def test_actual_official_backbone_composes_with_router_and_heads(official):
    native, _, x, tau, _, state_condition, _ = official
    model = official_moe_from_native(native, config=MoEConfig(dimensions=2))
    kappa = jnp.sin(x)
    variables = model.init(jax.random.PRNGKey(0), x, kappa, x, tau, state_condition)
    output = jax.jit(model.apply)(variables, x, kappa, x, tau, state_condition)
    assert output["q"].shape == (4, 8)
    assert output["velocities"].shape == (4, 8, 2)
    assert "field" in variables["params"]["backbone"]
    assert "output_layer" not in variables["params"]["backbone"]["field"]
    forbidden = dict(state_condition, population_token=jnp.zeros((4, 1, 2)))
    with pytest.raises(ValueError, match="forbidden"):
        model.apply(variables, x, kappa, x, tau, forbidden)


def test_all_arms_initialize_same_official_backbone_and_experts(official):
    native, _, x, tau, _, condition, _ = official
    model = official_moe_from_native(native, config=MoEConfig(dimensions=2))
    kappa = jnp.sin(x)
    parameters = []
    for arm in ("moe_static", "moe_kin", "moe_kin_shuf", "moe_directU"):
        feature = arm_input(arm, kappa, direct_u=x * 2, permutation=[1, 2, 3, 0])
        parameters.append(model.init(jax.random.PRNGKey(0), x, feature, x, tau, condition)["params"])
    for other in parameters[1:]:
        for a, b in zip(jax.tree_util.tree_leaves(parameters[0]), jax.tree_util.tree_leaves(other)):
            np.testing.assert_array_equal(a, b)


def test_official_ot_end_to_end_update_serialize_independent_paths(official):
    native, _, source, tau, _, condition, _ = official
    model = official_moe_from_native(native, config=MoEConfig(dimensions=2))
    kinetic = jnp.sin(source)
    initial = model.init(jax.random.PRNGKey(2), source, kinetic, source, tau, condition)
    state = TrainState.create(apply_fn=model.apply, params=initial["params"], tx=optax.adam(.001))
    matching = functools.partial(match_linear, epsilon=.1, scale_cost="mean", max_iterations=30)
    step = make_official_ot_train_step(model, matching)
    target = source + jnp.array([.4, -.2])
    values = []
    for update in range(3):
        state, metrics = step(state, jax.random.PRNGKey(update + 30), source, target, kinetic, condition)
        values.append(float(metrics["loss"]))
    assert np.isfinite(values).all()
    assert float(metrics["kinetic_gradient_norm"]) > 0
    assert float(metrics["router_gradient_norm"]) > 0
    restored = serialization.from_bytes(state.params, serialization.to_bytes(state.params))
    outputs = integrate_mixture(model, dict(params=restored), source, kinetic, condition, steps=3, tau_end=.3)
    assert outputs["candidates"].shape == (4, 8, 2)
    assert np.isfinite(weighted_energy(outputs["candidates"], outputs["q"], target))
    assert not np.allclose(outputs["candidates"][:, 0], outputs["candidates"][:, 1])


def test_unsupported_dropout_and_source_modification_are_not_silently_accepted(official):
    native, *_ = official
    with pytest.raises(ValueError, match="Nonzero dropout"):
        hidden_field_from_native(native.clone(hidden_dropout=.1))
    record = official_source_record()
    assert len(record["sha256"]) == 64
    assert record["upstream_source_modified"] is False


def test_actual_official_state_stream_does_not_read_kappa(official):
    native, _, x, tau, _, condition, _ = official
    model = official_moe_from_native(native, config=MoEConfig(dimensions=2))
    kappa = jnp.sin(x)
    variables = model.init(jax.random.PRNGKey(0), x, kappa, x, tau, condition)
    first = model.apply(variables, x, kappa, x, tau, condition)
    second = model.apply(variables, x, kappa + 7, x, tau, condition)
    np.testing.assert_array_equal(first["velocities"], second["velocities"])
    assert not np.allclose(first["q"], second["q"])


def test_official_ot_indices_do_not_depend_on_router_weights(official):
    native, _, source, tau, _, condition, _ = official
    model = official_moe_from_native(native, config=MoEConfig(dimensions=2))
    kappa = jnp.sin(source)
    variables = model.init(jax.random.PRNGKey(0), source, kappa, source, tau, condition)
    matching = functools.partial(match_linear, epsilon=.1, scale_cost="mean", max_iterations=30)
    step = make_official_ot_train_step(model, matching)
    left = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optax.adam(.001))
    changed_params = dict(variables["params"])
    changed_params["router"] = jax.tree_util.tree_map(lambda value: value + 2, changed_params["router"])
    right = TrainState.create(apply_fn=model.apply, params=changed_params, tx=optax.adam(.001))
    key = jax.random.PRNGKey(13)
    _, a = step(left, key, source, source + .2, kappa, condition)
    _, b = step(right, key, source, source + .2, kappa, condition)
    np.testing.assert_array_equal(a["source_indices"], b["source_indices"])
    np.testing.assert_array_equal(a["target_indices"], b["target_indices"])
