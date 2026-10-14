import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_validation import (LogStateProjection, conditional_score_test, count_parts,
                           local_u_control, split_us, two_stage_iv)


@pytest.mark.parametrize('sparse_input', [False, True])
def test_count_split_conserves_both_modalities_and_is_reproducible(sparse_input):
    raw = np.arange(60).reshape(12, 5)
    raw = sparse.csr_matrix(raw) if sparse_input else raw
    ua, sa = split_us(raw, raw, parts=3, seed=17)
    ub, sb = split_us(raw, raw, parts=3, seed=17)
    dense = lambda x: x.toarray() if sparse.issparse(x) else x
    np.testing.assert_array_equal(dense(sum(ua)), dense(raw))
    np.testing.assert_array_equal(dense(sum(sa)), dense(raw))
    for a, b in zip(ua + sa, ub + sb):
        np.testing.assert_array_equal(dense(a), dense(b))
    assert not np.array_equal(dense(ua[0]), dense(sa[0]))
    assert not np.array_equal(dense(sa[0]), dense(raw))


@pytest.mark.parametrize('bad', [np.array([[1.2]]), np.array([[-1]]), np.array([[np.nan]])])
def test_split_refuses_nonraw_counts(bad):
    with pytest.raises(ValueError):
        count_parts(bad, 2, np.random.default_rng(0))


def test_u_control_preserves_s_strata_depth_and_true_locality():
    rng = np.random.default_rng(0)
    us = np.column_stack((rng.uniform(1, 3, (40, 5)), np.ones((40, 5)) * 3))
    z = rng.normal(size=(40, 4)); strata = np.repeat(['a', 'b'], 20)
    result, order, audit = local_u_control(us, z, strata, seed=0, neighbors=5)
    np.testing.assert_array_equal(result[:, 5:], us[:, 5:])
    np.testing.assert_array_equal(strata[order], strata)
    np.testing.assert_allclose(result[:, :5].sum(1), us[:, :5].sum(1))
    assert len(np.unique(order)) == len(us)
    assert audit['moved_fraction'] > .5
    assert audit['all_moved_within_mutual_knn']
    np.testing.assert_array_equal(result, local_u_control(us, z, strata, seed=0, neighbors=5)[0])


def test_unmatched_and_zero_depth_controls_are_explicit():
    us = np.array([[0., 0., 3., 3.], [1., 1., 3., 3.], [1., 2., 3., 3.]])
    result, order, audit = local_u_control(us, np.arange(3)[:, None], ['a', 'a', 'b'])
    np.testing.assert_array_equal(result, us)
    assert audit['moved_fraction'] == 0 and audit['unchanged_cells'] == 3


def test_velocity_projection_is_derivative_of_same_state_map():
    rng = np.random.default_rng(1)
    s = rng.uniform(1, 10, (40, 8)); v = rng.normal(size=s.shape)
    p = LogStateProjection.fit(s[:20], dimensions=4)
    eps = 1e-5
    finite = (p.state(s + eps * v) - p.state(s - eps * v)) / (2 * eps)
    np.testing.assert_allclose(p.velocity(s, v), finite, atol=1e-8, rtol=1e-7)


def test_conditional_score_detects_mixed_sign_increment():
    rng = np.random.default_rng(11)
    c = rng.normal(size=(300, 3)); x = c[:, :2] + rng.normal(size=(300, 2))
    y = c.sum(1) + x[:, 0] - x[:, 1] + rng.normal(size=300)
    result = conditional_score_test(y, x, c, simulations=199, seed=5)
    assert result['p_value'] <= .01 and result['added_rank'] == 2


def test_conditional_score_unidentifiable_when_x_is_only_state():
    rng = np.random.default_rng(12); c = rng.normal(size=(100, 2))
    result = conditional_score_test(c[:, 0] + rng.normal(size=100), c, c, simulations=99)
    assert not result['identifiable'] and result['p_value'] == 1


def test_conditional_null_preserves_large_state_association():
    rng = np.random.default_rng(20); rejected = 0
    for rep in range(80):
        c = rng.normal(size=(180, 3)); x = 2 * c[:, :1] + rng.normal(size=(180, 1))
        y = 5 * c[:, 0] + rng.normal(size=180) * (.3 + np.abs(c[:, 1]))
        rejected += conditional_score_test(y, x, c, simulations=199, seed=rep)['p_value'] < .05
    assert rejected <= 12  # smoke only; the registered larger calibration is separate


def test_iv_includes_extra_in_first_stage_and_recovers_coefficients():
    rng = np.random.default_rng(4); n = 30000
    latent = rng.normal(size=n); extra = latent + rng.normal(size=n)
    sb = latent + rng.normal(size=n); sc = latent + rng.normal(size=n)
    y = 2 * latent + .7 * extra + rng.normal(size=n)
    fit = two_stage_iv(y, sb, extra, sc)
    assert abs(fit['extra_coefficient'] - .7) < .06
    assert abs(fit['state_coefficient'] - 2) < .06
    assert not fit['weak_instrument']


def test_iv_flags_weak_instrument():
    rng = np.random.default_rng(4)
    y, sb, extra, independent = rng.normal(size=(4, 2000))
    assert two_stage_iv(y, sb, extra, independent)['weak_instrument']
