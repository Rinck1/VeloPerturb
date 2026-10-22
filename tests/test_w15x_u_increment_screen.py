import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15x_u_increment_screen as module


def fixture():
    rng = np.random.default_rng(6); n, g = 48, 8
    s = rng.normal(size=(n, g)); innovation = rng.normal(size=(n, g))
    local = np.stack((s, -1/(1+np.exp(-s)), s*s, rng.uniform(.2, .5, size=(n, g))), axis=-1)
    data = dict(labels=np.array([f'KO{i}' for i in range(n)]), local=local,
        A=2*s+innovation, log_U=s+innovation, y=.2*s+3*innovation+rng.normal(0, .03, size=(n, g)),
        z0=rng.normal(size=(n, 3)), esm=rng.normal(size=(n, 5)), technical=rng.normal(size=(n, 1)),
        supported=np.ones(g, dtype=bool), components=np.eye(g)[:3], gene_ids=np.array([f'g{i}' for i in range(g)]))
    plan=dict(inner_folds=4, outer_folds=5, seed=22, ridge=1., source_PCs=1, condition_PCs=1,
        minimum_innovation_fraction=1e-6, minimum_inner_gain=.05, minimum_winning_inner_folds=3,
        minimum_selected_genes=2, minimum_valid_outer_folds=3, minimum_valid_conditions=20,
        shuffle_seeds=[0, 1, 2], bootstrap_repetitions=100, confidence=.95)
    return data, plan


def test_selector_recovers_strong_positive_U_signal():
    data, plan = fixture()
    result = module.selection_statistics(data, np.arange(48), plan)
    assert result['positive_kinetic_A']['mask'].sum() >= 6
    assert result['free_log_U']['mask'].sum() >= 6
    assert np.median(result['positive_kinetic_A']['gain']) > .7


def test_S_determined_U_cannot_create_fake_ridge_residual_increment():
    data, plan = fixture(); data['A'] = data['local'][:, :, 0]*2+data['local'][:, :, 1]
    data['log_U'] = 3*data['local'][:, :, 0]
    nuisance = module.nuisance_fit(data, np.arange(36), np.arange(36, 48), plan)
    assert np.max(np.abs(nuisance['A_train'])) < 1e-8
    assert np.max(np.abs(nuisance['A_query'])) < 1e-8
    predictions, beta = module.incremental_fit(nuisance, 'positive_kinetic_A', plan, 'synthetic')
    np.testing.assert_array_equal(beta, 0.)
    np.testing.assert_array_equal(predictions['real'], nuisance['base_query'])
    result = module.selection_statistics(data, np.arange(48), plan)
    assert not any(result[f]['mask'].any() for f in module.FAMILIES)


def test_outer_response_cannot_influence_inner_selection_or_outer_prediction():
    data, plan = fixture(); train, query = np.arange(36), np.arange(36, 48)
    first = module.selection_statistics(data, train, plan)
    nuisance = module.nuisance_fit(data, train, query, plan)
    altered = copy.deepcopy(data); altered['y'][query] = 1000000.
    second = module.selection_statistics(altered, train, plan)
    other = module.nuisance_fit(altered, train, query, plan)
    for family in module.FAMILIES:
        np.testing.assert_array_equal(first[family]['mask'], second[family]['mask'])
        np.testing.assert_array_equal(first[family]['gain'], second[family]['gain'])
    np.testing.assert_array_equal(nuisance['base_query'], other['base_query'])
    np.testing.assert_array_equal(nuisance['response_residual'], other['response_residual'])


def test_context_PCA_and_scaling_never_fit_query_sources():
    data, plan = fixture(); train, query = np.arange(36), np.arange(36, 48)
    xt, _ = module.designs(data, train, query, plan)
    changed = copy.deepcopy(data)
    for key in ('z0', 'esm', 'local', 'technical'): changed[key][query] *= 100000.
    other_xt, _ = module.designs(changed, train, query, plan)
    np.testing.assert_array_equal(xt, other_xt)


def test_positive_and_free_families_do_not_confuse_negative_U_markers_with_direction():
    data, plan = fixture(); data['y'] = -3*(data['A']-2*data['local'][:, :, 0])
    result = module.selection_statistics(data, np.arange(48), plan)
    assert not result['positive_kinetic_A']['mask'].any()
    assert result['free_kinetic_A']['mask'].sum() >= 6


def test_permutations_are_reproducible_nonself_bijections():
    order = module.derangement(20, 9)
    np.testing.assert_array_equal(np.sort(order), np.arange(20))
    assert np.all(order != np.arange(20))
    np.testing.assert_array_equal(order, module.derangement(20, 9))


def test_program_axes_cannot_use_unselected_genes():
    rng = np.random.default_rng(2); r = rng.normal(size=(20, 8)); mask = np.arange(8) < 5
    axes = module.program_axes(r, mask, 3)
    assert axes.shape == (3, 8)
    np.testing.assert_array_equal(axes[:, ~mask], 0.)
    np.testing.assert_allclose(axes@axes.T, np.eye(3), atol=1e-12)
    assert module.program_axes(r, np.zeros(8, bool), 3).shape == (0, 8)


def test_empty_selection_is_not_promoted_as_predictive_signal():
    data, plan = fixture(); n = module.nuisance_fit(data, np.arange(36), np.arange(36, 48), plan)
    n['target_variance'] = np.maximum(data['y'][:36].var(0), 1e-8)
    predictions, _ = module.incremental_fit(n, 'positive_kinetic_A', plan, 'test')
    rows = []
    for family in module.FAMILIES:
        rows.extend(module.score_fold(data, np.arange(36, 48), n, predictions, np.zeros(8, bool),
            np.empty((0, 8)), plan, family, 0))
    results = module.summarize(rows, plan)
    assert all(r['valid_KOs'] == 0 and not r['predictive_screen_passed'] for r in results)


def test_condition_bootstrap_keeps_a_known_paired_effect():
    _, plan = fixture()
    result = module.bootstrap_comparison(np.full(30, .5), np.ones(30), plan)
    assert result['gain'] == result['ci_low'] == result['ci_high'] == .5
