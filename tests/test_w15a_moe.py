"""CPU mathematical/structural tests; these do not establish biological gains."""
import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase0"))
from w15a_cellflow_moe import (ARMS, KineticMoE, MoEConfig, align_ot_indices,
    arm_input, flow_matching_loss, integrate_mixture, require_l2_freeze,
    sampled_energy, weighted_energy, kinetic_sensitivity, state_condition_only,
    paired_flow_matching_loss, align_ot_indices_jit, l2_scientific_sections)


class StubBackbone(nn.Module):
    """An explicit stub, never represented as an official CellFlow backbone."""
    width: int = 12

    @nn.compact
    def __call__(self, x, tau, condition):
        return nn.silu(nn.Dense(self.width)(jnp.concatenate((x, tau, condition), axis=-1)))


@pytest.fixture
def setup():
    c = MoEConfig(dimensions=3, experts=2, expert_width=7, kinetic_width=5,
                  kinetic_embedding=2, router_width=6)
    model = KineticMoE(StubBackbone(), c)
    x = jnp.array([[.2, -.7, 1.], [1., 2., -.3], [-.4, .8, .5]])
    kappa = x * .7 + .1
    condition = jnp.array([[.1, .2], [.2, .3], [.3, .4]])
    variables = model.init(jax.random.PRNGKey(0), x, kappa, x, .4, condition)
    return model, variables, x, kappa, condition


def test_exact_per_expert_expectation_not_mixed_mse():
    q = jnp.array([[.5, .5]])
    velocity = jnp.array([[[1.], [-1.]]])
    loss, metrics = flow_matching_loss(q, velocity, jnp.array([[0.]]))
    assert float(loss) == 1.
    assert float(metrics["matching"]) == 1.
    assert float(jnp.square((q[..., None] * velocity).sum(axis=1)).sum()) == 0.


def test_flow_matching_loss_gradient_matches_explicit_expectation():
    q = jnp.array([[.2, .8], [.7, .3]])
    velocity = jnp.arange(12, dtype=float).reshape(2, 2, 3) / 10
    target = jnp.zeros((2, 3))
    actual = jax.grad(lambda value: flow_matching_loss(q, value, target)[0])(velocity)
    np.testing.assert_allclose(actual, 2 * q[..., None] * velocity / len(q), atol=1e-6)


def test_four_arms_have_identical_parameters_and_static_zero(setup):
    model, variables, x, kappa, condition = setup
    count = lambda v: sum(a.size for a in jax.tree_util.tree_leaves(v))
    counts = []
    for arm in ARMS:
        value = arm_input(arm, kappa, direct_u=kappa * 3, permutation=[2, 0, 1])
        initialized = model.init(jax.random.PRNGKey(0), x, value, x, .4, condition)
        counts.append(count(initialized))
        for left, right in zip(jax.tree_util.tree_leaves(variables), jax.tree_util.tree_leaves(initialized)):
            np.testing.assert_array_equal(left, right)
    assert len(set(counts)) == 1
    np.testing.assert_array_equal(arm_input("moe_static", kappa), np.zeros_like(kappa))


def test_shuffle_preserves_marginals_and_validates_permutation(setup):
    _, _, _, kappa, _ = setup
    shuffled = arm_input("moe_kin_shuf", kappa, permutation=[2, 0, 1])
    np.testing.assert_array_equal(shuffled, kappa[np.array([2, 0, 1])])
    with pytest.raises(ValueError, match="full source-cohort permutation"):
        arm_input("moe_kin_shuf", kappa, permutation=[0, 0, 1])


def test_state_stream_independent_of_kinetics_and_router(setup):
    model, variables, x, kappa, condition = setup
    def output(k):
        return model.apply(variables, x, k, x, .4, condition)["velocities"]
    np.testing.assert_array_equal(output(kappa), output(-kappa * 17))
    np.testing.assert_array_equal(jax.jacrev(output)(kappa), 0)
    changed = jax.tree_util.tree_map(lambda value: value + 9, variables["params"]["router"])
    changed_variables = dict(params=dict(variables["params"], router=changed))
    np.testing.assert_array_equal(output(kappa),
        model.apply(changed_variables, x, .4, condition, method=model.velocities))


def test_velocity_encoder_does_not_read_raw_source_x(setup):
    model, variables, x, kappa, _ = setup
    def encoded(raw_x):
        del raw_x
        return model.apply(variables, kappa, method=model.encode_kinetics)
    np.testing.assert_array_equal(jax.jacrev(encoded)(x), 0)
    assert model.apply(variables, kappa, method=model.encode_kinetics).shape == (3, 2)
    assert "source_x" not in __import__("inspect").signature(model.encode_kinetics).parameters


def test_training_reaches_both_kinetic_encoder_and_experts(setup):
    model, variables, x, kappa, condition = setup
    def loss(params):
        result = model.apply(dict(params=params), x, kappa, x + .2, .4, condition)
        return flow_matching_loss(result["q"], result["velocities"], jnp.ones_like(x))[0]
    gradients = jax.grad(loss)(variables["params"])
    norm = lambda tree: sum(float(jnp.square(v).sum()) for v in jax.tree_util.tree_leaves(tree))
    assert norm(gradients["kinetic_encoder"]) > 0
    assert norm(gradients["router"]) > 0
    assert norm(gradients["expert_0"]) > 0
    assert norm(gradients["backbone"]) > 0


def test_ot_source_attributes_follow_resampled_indices(setup):
    _, _, source, kappa, _ = setup
    si, ti = [2, 0, 2], [1, 2, 0]
    paired = align_ot_indices(source, kappa, source + 5, si, ti)
    np.testing.assert_array_equal(paired.source_x, source[np.array(si)])
    np.testing.assert_array_equal(paired.kinetic_input, kappa[np.array(si)])
    np.testing.assert_array_equal(paired.target_x, (source + 5)[np.array(ti)])


def test_external_ot_pairing_independent_of_router_parameters(setup):
    from scipy.optimize import linear_sum_assignment
    model, variables, source, kappa, condition = setup
    target = source[::-1] + .1
    state_only_pairing = lambda: linear_sum_assignment(np.linalg.norm(
        np.asarray(source)[:, None] - np.asarray(target)[None], axis=-1))
    before = state_only_pairing()
    changed = dict(params=jax.tree_util.tree_map(lambda value: value * -100, variables["params"]))
    model.apply(changed, source, kappa, source, .5, condition)
    after = state_only_pairing()
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


def test_q_is_source_only_and_frozen_for_whole_trajectory(setup):
    model, variables, source, kappa, condition = setup
    result = integrate_mixture(model, variables, source, kappa, condition, steps=5)
    assert result["routing_evaluations"] == 1
    expected = model.apply(variables, source, kappa, method=model.route)
    np.testing.assert_array_equal(result["q"], expected)
    np.testing.assert_array_equal(result["q_history"], np.repeat(np.asarray(expected)[None], 5, axis=0))
    q2 = model.apply(variables, source, kappa, method=model.route)
    model.apply(variables, source, .9, condition * 3, method=model.velocities)
    np.testing.assert_array_equal(q2, expected)


def test_each_expert_integrates_own_path_and_interventions_change_only_q(setup):
    model, variables, source, kappa, condition = setup
    outputs = [integrate_mixture(model, variables, source, kappa, condition,
                                steps=4, q_intervention=intervention)
               for intervention in ("none", "cohort_q", "uniform_q")]
    assert not np.allclose(outputs[0]["candidates"][:, 0], outputs[0]["candidates"][:, 1])
    for result in outputs[1:]:
        np.testing.assert_array_equal(result["candidates"], outputs[0]["candidates"])
    np.testing.assert_allclose(outputs[1]["q"], np.repeat(
        np.asarray(outputs[0]["q"]).mean(axis=0)[None], len(source), axis=0), atol=1e-7)
    np.testing.assert_array_equal(outputs[2]["q"], np.full((3, 2), .5))


def test_zero_tau_is_exact_identity(setup):
    model, variables, source, kappa, condition = setup
    result = integrate_mixture(model, variables, source, kappa, condition, tau_end=0., steps=3)
    np.testing.assert_array_equal(result["candidates"], np.repeat(np.asarray(source)[:, None], 2, axis=1))


def test_exact_weighted_energy_matches_independent_mc_not_naive_vstat():
    candidates = np.array([[[-1.], [1.]]])
    q = np.array([[.5, .5]])
    target = np.array([[0.]])
    exact = weighted_energy(candidates, q, target)
    assert exact == 1.
    rng = np.random.default_rng(0)
    independent = np.array([sampled_energy(candidates, q, target, rng) for _ in range(1000)])
    assert abs(independent.mean() - exact) < 4 * independent.std(ddof=1) / np.sqrt(len(independent))
    naive = [sampled_energy(candidates, q, target, rng, independent_within=False) for _ in range(100)]
    assert np.mean(naive) == 2.  # Same-source off-diagonal expert mass is missing.


def test_real_training_blocked_without_frozen_l2_configuration():
    with pytest.raises(ValueError, match="awaits the frozen L2"):
        require_l2_freeze({})
    with pytest.raises(ValueError, match="awaits the frozen L2"):
        require_l2_freeze(dict(cf_static_frozen=True, freeze_commit="not-enough"))


def test_default_registered_shape_and_asymmetric_experts():
    model = KineticMoE(StubBackbone())
    x = jnp.zeros((2, 50))
    kappa = jnp.ones((2, 50))
    condition = jnp.zeros((2, 3))
    variables = model.init(jax.random.PRNGKey(0), x, kappa, x, .2, condition)
    params = variables["params"]
    assert params["kinetic_encoder"]["hidden"]["kernel"].shape == (50, 64)
    assert params["kinetic_encoder"]["embedding"]["kernel"].shape == (64, 16)
    assert params["router"]["hidden"]["kernel"].shape == (66, 64)
    assert params["router"]["logits"]["kernel"].shape == (64, 8)
    assert params["expert_0"]["hidden"]["kernel"].shape == (12, 128)
    assert params["expert_0"]["velocity"]["kernel"].shape == (128, 50)
    assert not np.array_equal(params["expert_0"]["velocity"]["kernel"],
                              params["expert_1"]["velocity"]["kernel"])
    result = model.apply(variables, x, kappa, x, .2, condition)
    assert result["q"].shape == (2, 8)
    assert result["velocities"].shape == (2, 8, 50)


def test_named_kinetic_conditions_cannot_enter_state_stream():
    from flax.core import freeze
    with pytest.raises(ValueError, match="forbidden"):
        state_condition_only(dict(bio_time=jnp.ones((2, 8)), population_token=jnp.zeros((2, 50))))
    with pytest.raises(ValueError, match="forbidden"):
        state_condition_only(freeze(dict(cell_token=jnp.ones((2, 50)))))
    assert "bio_time" in state_condition_only(dict(bio_time=jnp.ones((2, 8))))


def test_sensitivity_diagnostic_is_inference_only(setup):
    model, variables, source, kappa, _ = setup
    summary = kinetic_sensitivity(model, variables, source, kappa, kappa)
    assert float(summary["mean_route_tv"]) == 0.
    assert bool(summary["u_insensitive"])
    assert float(summary["embedding_median_cosine"]) == pytest.approx(1., abs=1e-6)


def test_rk4_constant_independent_expert_fields_match_analytic_solution(setup):
    model, variables, source, kappa, condition = setup
    params = jax.tree_util.tree_map(jnp.zeros_like, variables["params"])
    for k in range(model.config.experts):
        params[f"expert_{k}"]["velocity"]["bias"] = jnp.full((3,), k + 1.)
    constant_variables = dict(params=params)
    result = integrate_mixture(model, constant_variables, source, kappa, condition, tau_end=.7, steps=4)
    expected = np.asarray(source)[:, None] + .7 * np.arange(1, 3)[None, :, None]
    np.testing.assert_allclose(result["candidates"], expected, atol=2e-6)


def test_ot_alignment_reindexes_source_conditions_too(setup):
    _, _, source, kappa, condition = setup
    paired = align_ot_indices(source, kappa, source + 5, [2, 0, 2], [1, 2, 0], condition=condition)
    np.testing.assert_array_equal(paired.condition, condition[np.array([2, 0, 2])])


def test_complete_paired_loss_uses_source_route_and_fm_clock(setup):
    model, variables, source, kappa, condition = setup
    paired = align_ot_indices(source, kappa, source + .3, [2, 0, 1], [1, 2, 0], condition=condition)
    tau = jnp.array([.1, .5, .9])[:, None]
    loss, _ = paired_flow_matching_loss(model, variables, paired, tau)
    expected = model.apply(variables, paired.source_x, paired.kinetic_input,
        (1 - tau) * paired.source_x + tau * paired.target_x, tau, paired.condition)
    manual, _ = flow_matching_loss(expected["q"], expected["velocities"], paired.target_x - paired.source_x)
    assert float(loss) == pytest.approx(float(manual), abs=1e-6)
    gradient = jax.grad(lambda params: paired_flow_matching_loss(model, dict(params=params), paired, tau)[0])(
        variables["params"])
    assert sum(float(jnp.square(v).sum()) for v in jax.tree_util.tree_leaves(gradient["kinetic_encoder"])) > 0


def test_jit_dynamic_ot_indices_carry_duplicate_source_attributes_and_conditions(setup):
    model, variables, source, kappa, condition = setup
    compiled = jax.jit(lambda si, ti: align_ot_indices_jit(source, kappa, source + .3, si, ti,
                                                          condition=dict(bio_time=condition)))
    for si, ti in (([2, 0, 2], [1, 2, 0]), ([1, 1, 0], [0, 2, 1])):
        result = compiled(jnp.array(si), jnp.array(ti))
        np.testing.assert_array_equal(result.source_x, source[np.array(si)])
        np.testing.assert_array_equal(result.kinetic_input, kappa[np.array(si)])
        np.testing.assert_array_equal(result.condition["bio_time"], condition[np.array(si)])
    compiled_loss = jax.jit(lambda params, si, ti: paired_flow_matching_loss(
        model, dict(params=params), align_ot_indices_jit(source, kappa, source + .3, si, ti,
            condition=condition), jnp.array([.1, .5, .9]))[0])
    loss = compiled_loss(variables["params"], jnp.array([2, 0, 2]), jnp.array([1, 2, 0]))
    assert np.isfinite(float(loss))


def test_frozen_l2_gate_checks_config_and_adapter_file_hashes(tmp_path):
    import hashlib
    import json
    config_path = tmp_path / "config.json"
    adapter_path = tmp_path / "adapter.py"
    config = dict(backbone=dict(width=32), optimizer=dict(lr=.001), solver=dict(method="RK4", steps=32))
    config_path.write_text(json.dumps(config))
    adapter_path.write_text("# Test fixture, not an official CellFlow adapter.\n")
    record = dict(config, cf_static_frozen=True, freeze_commit="f" * 40,
        configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        official_backbone_adapter="fixture.hidden", official_backbone_adapter_sha256=
            hashlib.sha256(adapter_path.read_bytes()).hexdigest())
    assert require_l2_freeze(record, configuration_path=config_path, adapter_path=adapter_path) == record
    with pytest.raises(ValueError, match="files are required"):
        require_l2_freeze(record)
    with pytest.raises(ValueError, match="metadata disagrees"):
        require_l2_freeze(dict(record, backbone=dict(width=99)), configuration_path=config_path, adapter_path=adapter_path)
    adapter_path.write_text("# Changed adapter bytes.\n")
    with pytest.raises(ValueError, match="SHA256 changed"):
        require_l2_freeze(record, configuration_path=config_path, adapter_path=adapter_path)
    adapter_path.write_text("# Test fixture, not an official CellFlow adapter.\n")
    config_path.write_text("{}")
    with pytest.raises(ValueError, match="SHA256 changed"):
        require_l2_freeze(record, configuration_path=config_path, adapter_path=adapter_path)


def test_freeze_gate_accepts_actual_flat_l2_scientific_schema_without_rewriting(tmp_path):
    import hashlib
    import json
    flat = dict(width=128, depth=2, condition_dim=32, time_width=128, time_depth=2,
        time_freqs=64, pooling="mean", learning_rate=.0003, steps=5000, batch_size=128,
        ot_epsilon=.1, ot_max_iterations=200)
    config_path = tmp_path / "actual_frozen_l2_configuration.json"
    adapter_path = tmp_path / "actual_hidden_adapter_fixture.py"
    config_path.write_text(json.dumps(flat))
    adapter_path.write_text("# Unit test fixture only\n")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = dict(l2_scientific_sections(flat), cf_static_frozen=True, freeze_commit="e" * 40,
        configuration_sha256=digest, official_backbone_adapter="test.hidden",
        official_backbone_adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest())
    require_l2_freeze(record, configuration_path=config_path, adapter_path=adapter_path)
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == digest
    assert json.loads(config_path.read_text()) == flat
