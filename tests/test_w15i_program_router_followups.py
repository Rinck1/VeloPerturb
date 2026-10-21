import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15i_program_router_followups as module


def groups():
    rng = np.random.default_rng(3)
    collection = []
    for label in ('A', 'B'):
        collection.append(dict(label=label, time_hours=6, z=rng.normal(size=(7, 3)),
            U=rng.normal(size=(7, 3)), U_normalized=rng.normal(size=(7, 3)),
            velocity=rng.normal(size=(7, 3)), cell_ids=np.array([label+str(i) for i in range(7)]),
            velocity_program=np.array([.6, .8]), U_program=np.array([1., 0.]),
            shuffled_velocity_program=np.array([0., 1.]), embedding=np.ones(2)))
    norm = dict(axes=np.eye(3)[:2], velocity_reference=np.array([.1, -.2, .3]),
                U_reference=np.array([-.2, .4, -.1]))
    return collection, norm


def test_cell_features_keep_U_and_shuffle_within_KO():
    original, norm = groups()
    changed, mapping = module.cell_programs(original, norm, 0)
    for raw, g in zip(original, changed):
        np.testing.assert_array_equal(raw['U_normalized'], g['U_normalized'])
        assert raw['velocity_program'].shape == (2,)
        assert g['velocity_program'].shape == (7, 2)
        expected = module.base.unit((raw['velocity']-norm['velocity_reference']) @ norm['axes'].T)
        np.testing.assert_allclose(g['velocity_program'], expected, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(g['velocity_program'], axis=1), 1., atol=1e-6)
        np.testing.assert_allclose(np.sort(g['velocity_program'], axis=0),
                                   np.sort(g['shuffled_velocity_program'], axis=0))
        assert set(mapping[g['label']]) == set(g['cell_ids'])
        assert all(a != b and b.startswith(g['label']) for a, b in mapping[g['label']].items())
        inputs = module.base.apply_arm(g, 'S_U_velocity_program', dict(programs=2))['kinetic_input']
        np.testing.assert_allclose(inputs[:, :3], raw['U_normalized'], atol=1e-7)
        np.testing.assert_allclose(inputs[:, 3:], g['velocity_program'])


def test_intervention_changes_only_named_input_slots():
    original, _ = groups(); g = original[0]; c = dict(programs=2)
    normal = module.intervention_input(g, 'original', c)['kinetic_input']
    zero = module.intervention_input(g, 'velocity_zero', c)['kinetic_input']
    reverse = module.intervention_input(g, 'velocity_reverse', c)['kinetic_input']
    uzero = module.intervention_input(g, 'U_zero', c)['kinetic_input']
    shuffle = module.intervention_input(g, 'velocity_cross_KO_shuffle', c)['kinetic_input']
    np.testing.assert_array_equal(normal[:, :3], zero[:, :3])
    np.testing.assert_array_equal(normal[:, :3], reverse[:, :3])
    np.testing.assert_array_equal(normal[:, 3:], -reverse[:, 3:])
    assert not zero[:, 3:].any()
    assert not uzero[:, :3].any()
    np.testing.assert_array_equal(normal[:, 3:], uzero[:, 3:])
    np.testing.assert_allclose(shuffle[:, 3:], np.broadcast_to(g['shuffled_velocity_program'], (7, 2)))


def test_dependency_complete_and_failed_do_not_sleep(tmp_path):
    path = tmp_path/'status.json'
    path.write_text(json.dumps(dict(status='complete')))
    module.wait_for_predecessor(path, 0., lambda s: pytest.fail('Completed dependency should not wait'))
    path.write_text(json.dumps(dict(status='failed')))
    with pytest.raises(RuntimeError, match='Parent job failed'):
        module.wait_for_predecessor(path, 0., lambda s: None)


def test_reuse_all_and_only_registered_baselines(tmp_path):
    rows = [dict(seed=seed, KO=ko, arm=arm, path=f'{seed}/{ko}/{arm}')
            for seed in (0, 1, 2) for ko in ('A', 'B') for arm in module.base.ARMS]
    (tmp_path/'frozen_grid.json').write_text(json.dumps(dict(predictions=rows)))
    for seed in (0, 1, 2):
        path = tmp_path/'fits'/f'seed{seed}'/'S_U_router'
        path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps(dict(initial_parameters_sha256=f'initial_{seed}')))
    selected, initial = module.read_baselines(tmp_path, [0, 1, 2], ['A', 'B'])
    assert len(selected) == 6 and all(r['arm'] == 'S_U_router' for r in selected)
    assert initial == {seed: f'initial_{seed}' for seed in (0, 1, 2)}
    with pytest.raises(ValueError, match='Incomplete reusable'):
        module.read_baselines(tmp_path, [0, 1, 2], ['A', 'B', 'Missing'])


def test_short_time_configuration_passes_true_zero_and_six_hours():
    original, _ = groups()
    g = dict(original[0], time_hours=0)
    c = dict(source_hours=0, target_hours=6)
    actual = module.base.condition(g, 4, c)
    expected = module.base.source_state_conditions(4, 0, g['embedding'], delta_hours=6)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_no_intervention_uses_a_target_as_model_input():
    original, _ = groups(); g = original[0]
    for name in module.INTERVENTIONS:
        prepared = module.intervention_input(g, name, dict(programs=2))
        assert 'target' not in prepared
        assert prepared['kinetic_input'].shape == (7, 5)
