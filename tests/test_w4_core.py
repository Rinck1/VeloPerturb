import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_arms import (RIDGE_GRID, average_predictions_then_errors, fit_ridge_training_cv,
                     paired_gain_test)
from w4_streams import (SourceTransform, fit_gamma_tail, normalized_pair,
                        population_features, swap_pooled_u)


@pytest.fixture
def example():
    rng = np.random.default_rng(20)
    s = rng.poisson(10, (90, 12)).astype(float)
    u = rng.poisson(5, s.shape).astype(float)
    ids = np.array([f's{i}' for i in range(len(s))])
    transform = SourceTransform.fit(s, u, cell_ids=ids, allowed_ids=set(ids),
        gene_names=[f'g{i}' for i in range(s.shape[1])], n_hvg=10, dimensions=5, clusters=4)
    return s, u, ids, transform


def test_analytic_jacobian_matches_fixed_library_finite_difference(example):
    s, u, _, t = example
    sn, un = t.normalized(s, u); rate = un-t.gamma*sn
    epsilon = 1e-5
    numerical = (t.state_normalized(sn+epsilon*rate)-t.state_normalized(sn-epsilon*rate))/(2*epsilon)
    analytic = t.projected_rate(sn, rate)
    assert np.linalg.norm(numerical-analytic)/np.linalg.norm(analytic) < 1e-4


def test_linear_pool_equals_depth_weighted_mean_not_arbitrary_cell_mean(example):
    s, u, _, t = example
    sn, un = t.normalized(s, u)
    ps, pu = t.normalized(s.sum(0, keepdims=True), u.sum(0, keepdims=True))
    weights = s.sum(1)/s.sum()
    linear_mean = (weights[:, None]*(un-t.gamma*sn)).sum(0) @ t.components.T
    linear_pool = ((pu-t.gamma*ps) @ t.components.T)[0]
    np.testing.assert_allclose(linear_pool, linear_mean, atol=1e-12)
    equal_s = s/s.sum(1, keepdims=True)*10000
    equal_u = u/s.sum(1, keepdims=True)*10000
    es, eu = t.normalized(equal_s, equal_u)
    eps, epu = t.normalized(equal_s.sum(0, keepdims=True), equal_u.sum(0, keepdims=True))
    np.testing.assert_allclose((eu-t.gamma*es).mean(0), (epu-t.gamma*eps)[0], atol=1e-12)


def test_future_count_changes_cannot_change_source_fit(example, tmp_path):
    s, u, ids, t = example
    before = t.fingerprint()
    future = s*100
    t.state(future)
    assert t.fingerprint() == before
    other = SourceTransform.fit(s, u, cell_ids=ids, allowed_ids=set(ids), gene_names=t.gene_names,
        n_hvg=10, dimensions=5, clusters=4)
    assert before == other.fingerprint()
    t.save(tmp_path/'transform.npz')
    assert SourceTransform.load(tmp_path/'transform.npz').fingerprint() == before
    with pytest.raises(ValueError, match='forbidden'):
        SourceTransform.fit(s, u, cell_ids=ids, allowed_ids=set(ids[1:]), gene_names=t.gene_names,
            n_hvg=10, dimensions=5, clusters=4)


def test_library_normalization_uses_full_panel_before_gene_subset(example):
    s, u, _, t = example
    sn, un = t.normalized(s, u)
    np.testing.assert_allclose(sn, (s*10000/s.sum(1, keepdims=True))[:, t.genes])
    np.testing.assert_allclose(un, (u*10000/s.sum(1, keepdims=True))[:, t.genes])
    with pytest.raises(ValueError, match='Zero-depth'):
        normalized_pair(np.zeros_like(s), u)


def test_two_tail_gamma_matches_hand_calculation():
    s = np.arange(1, 101, dtype=float)[:, None]
    u = 2*s+3
    mask = (s[:, 0] <= np.quantile(s, .05)) | (s[:, 0] >= np.quantile(s, .95))
    np.testing.assert_allclose(fit_gamma_tail(s, u), (s[mask].T@u[mask]/np.square(s[mask]).sum()).ravel())


def test_swap_uses_training_donors_preserves_depth_and_does_not_touch_s(example):
    s, u, ids, t = example
    groups = np.repeat(['A', 'B', 'C'], 30)
    f = population_features(t, s, u, groups, ['A','B','C'], ids)
    before = f['pooled_s'].copy()
    swapped, audit = swap_pooled_u(f['pooled_u'], f['state'], f['labels'],
        f['pooled_u'][:2], f['state'][:2], f['labels'][:2])
    np.testing.assert_array_equal(before, f['pooled_s'])
    np.testing.assert_allclose(swapped.sum(1), f['pooled_u'].sum(1))
    assert all(r['donor'] in ('A','B') and r['recipient'] != r['donor'] for r in audit)
    with pytest.raises(ValueError, match='eligible'):
        swap_pooled_u(f['pooled_u'][:1], f['state'][:1], ['A'], f['pooled_u'][:1], f['state'][:1], ['A'])


def test_ridge_inner_scores_match_independent_sklearn_refits():
    rng = np.random.default_rng(21)
    x, y = rng.normal(size=(9, 25)), rng.normal(size=(9, 4))
    _, model, selection = fit_ridge_training_cv(x, y)
    expected = []
    for alpha in RIDGE_GRID:
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            scaler = StandardScaler().fit(x[keep])
            reference = Ridge(alpha=alpha).fit(scaler.transform(x[keep]), y[keep])
            errors.append(np.square(reference.predict(scaler.transform(x[i:i+1]))[0]-y[i]).sum())
        expected.append(np.mean(errors))
    np.testing.assert_allclose(selection['inner_TF_sse'], expected, atol=1e-9)
    assert model.alpha == RIDGE_GRID[np.argmin(expected)]


def test_seeds_are_averaged_before_scoring_and_zero_gains_do_not_support():
    truth = np.zeros((4, 2)); seed_predictions = np.stack([np.ones_like(truth), -np.ones_like(truth)])
    np.testing.assert_array_equal(average_predictions_then_errors(seed_predictions, truth), np.zeros(4))
    assert not paired_gain_test(np.ones((8,2)), np.ones((8,2)), np.ones((8,2)))['supported']


def test_source_transform_rejects_negative_counts_and_duplicate_genes(example):
    s, u, ids, t = example
    u[0,0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        t.velocity(s, u)
    u[0,0] = 1
    with pytest.raises(ValueError, match='identity'):
        SourceTransform.fit(s, u, cell_ids=ids, allowed_ids=set(ids),
            gene_names=['same']*12, n_hvg=10, dimensions=5, clusters=4)
