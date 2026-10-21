import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15h_traxler_program_router as module


def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)
    train = [f'KO{i}' for i in range(8)]; val = ['V1', 'V2', 'V3']; test = ['T1']
    split = tmp_path/'split.json'
    split.write_text(json.dumps(dict(train=train, val=val, test=test)))
    esm = tmp_path/'conditions.npz'
    np.savez_compressed(esm, conditions=np.array(train+val+test), embeddings=rng.normal(size=(12, 6)),
        metadata_json=np.array(json.dumps(dict(frozen=True, no_cell_expression_used=True))))

    def source(labels):
        conditions = np.repeat(labels, 4); n = len(conditions)
        return dict(z=rng.normal(0, .1, size=(n, 3)).astype('float32'),
            u_features=rng.normal(size=(n, 3)).astype('float32'),
            velocity=rng.normal(size=(n, 3)).astype('float32'), conditions=conditions,
            cell_ids=np.array([f'{label}_{i}' for i, label in enumerate(conditions)]))

    st = source(train); sv = source(val)
    target = dict(z=st['z']+.1*st['velocity']+.03, conditions=st['conditions'])
    inputs = {'train_source.npz': st, 'train_target.npz': target, 'validation_source.npz': sv}
    calls = []

    def reader(path, expected_side):
        calls.append(path.name)
        assert path.name in inputs, 'Held-out future was opened while building model inputs'
        return inputs[path.name], dict(day=24 if expected_side == 'target' else 6)

    monkeypatch.setattr(module, 'load_pack', reader)
    c = dict(fold=str(tmp_path), split=str(split), conditions=str(esm), source_hours=6, target_hours=24,
        programs=2, shuffle_seed=0, evaluation_cells=4, evaluation_seed=0, prediction_batch=4,
        width=8, depth=1, condition_dim=4, time_freqs=8, time_width=8, time_depth=1,
        pooling='mean', learning_rate=.001, ot_epsilon=.1, ot_max_iterations=20,
        steps=2, batch_size=4, log_interval=1)
    return c, calls, inputs


def test_source_only_preparation_and_matched_U_inputs(tmp_path, monkeypatch):
    c, calls, inputs = fixture(tmp_path, monkeypatch)
    train, validation, norm, split, permutations = module.prepare(c)
    assert calls == ['train_source.npz', 'train_target.npz', 'validation_source.npz']
    assert len(train) == 8 and len(validation) == 3
    assert all('target' not in g for g in validation)
    np.testing.assert_allclose(norm['axes'] @ norm['axes'].T, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(norm['U_mean'], inputs['train_source.npz']['u_features'].astype(float).mean(0))
    for side in ('train', 'validation'):
        mapping = permutations[side]
        assert all(k != v for k, v in mapping.items())
        assert set(mapping) == set(mapping.values())
    for g in train+validation:
        arms = [module.apply_arm(g, arm, c) for arm in module.ARMS]
        assert all(a['kinetic_input'].shape == (4, 5) for a in arms)
        assert not arms[0]['kinetic_input'][:, 3:].any()
        for a in arms:
            np.testing.assert_array_equal(a['kinetic_input'][:, :3], g['U_normalized'])
        np.testing.assert_allclose(arms[1]['kinetic_input'][:, 3:],
                                   np.broadcast_to(g['velocity_program'], (4, 2)))
    expected = module.source_state_conditions(4, 6, train[0]['embedding'], delta_hours=18)
    for key, value in module.condition(train[0], 4, c).items():
        np.testing.assert_array_equal(value, expected[key])


def test_validation_sources_do_not_change_training_response_axes(tmp_path, monkeypatch):
    c, _, inputs = fixture(tmp_path, monkeypatch)
    a = module.prepare(c)
    for key in ('z', 'u_features', 'velocity'):
        inputs['validation_source.npz'][key] *= 10000
    b = module.prepare(c)
    for key in a[2]:
        np.testing.assert_array_equal(a[2][key], b[2][key])
    for first, second in zip(a[0], b[0]):
        np.testing.assert_array_equal(first['velocity_program'], second['velocity_program'])


def test_actual_ot_update_saved_model_and_independent_ODE(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from flax import serialization

    c, _, _ = fixture(tmp_path, monkeypatch)
    groups, heldout, _, _, _ = module.prepare(c)
    model = module.create_model(c, 3, 0, groups[0]['embedding'])
    initial_bytes = []
    for arm in module.ARMS:
        g = module.apply_arm(groups[0], arm, c)
        params = model.init(jax.random.PRNGKey(0), jnp.asarray(g['z'][:1]),
            jnp.asarray(g['kinetic_input'][:1]), jnp.asarray(g['z'][:1]), .5,
            module.condition(g, 1, c))['params']
        initial_bytes.append(serialization.to_bytes(params))
    assert len(set(initial_bytes)) == 1
    prepared = [module.apply_arm(g, 'S_U_velocity_program', c) for g in groups]
    params, template, history = module.train(model, module.make_step(model, c), prepared, c, 0, lambda r: None)
    assert len(history) == 2
    assert np.isfinite(history[-1]['loss'])
    assert history[-1]['kinetic_gradient_norm'] > 0
    assert history[-1]['router_gradient_norm'] > 0
    restored = serialization.from_bytes(template, serialization.to_bytes(params))
    source = module.apply_arm(heldout[0], 'S_U_velocity_program', c)
    pred = module.predict(model, module.make_candidate_integrator(model), restored, source, c)
    assert pred['candidates'].shape == (4, 8, 3)
    assert np.isfinite(pred['candidates']).all()
    np.testing.assert_allclose(pred['q'].sum(1), 1., atol=1e-6)
    # No velocity/U argument is present in the expert ODE condition stream.
    assert set(module.condition(source, 4, c)) == {'perturbation', 'bio_time', 'bio_delta'}


def test_weighted_distribution_metrics_known_identical_mixture():
    target = np.array([[-1., 0.], [1., 0.]])
    candidates = np.repeat(target[:, None], 2, axis=1)
    q = np.full((2, 2), .5)
    result = module.mixture_metrics(candidates, q, target, np.eye(2))
    assert abs(result['energy']) < 1e-10
    assert result['mean_mse'] == 0.
    assert result['program_mean_mse'] == 0.
    assert result['variance_ratio'] == 1.
    assert result['log_variance_error'] == 0.
    assert result['population_effective_experts'] == 2.
    with pytest.raises(ValueError, match='Invalid finite'):
        module.mixture_metrics(candidates, -q, target, np.eye(2))


def test_incomplete_grid_rejected_before_validation_future(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Incomplete grid must not open held-out targets')
    monkeypatch.setattr(module, 'load_pack', forbidden)
    with pytest.raises(ValueError, match='All registered'):
        module.evaluate(tmp_path, dict(seeds=[0, 1, 2]), [], dict(val=['A']), {})
