from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w16b_scntseq_label_reference as module


def plan():
    return dict(label_duration_minutes=120., minimum_baseline_total_gene_counts=20,
        minimum_baseline_new_gene_counts=5, minimum_baseline_old_gene_counts=5)


def test_baseline_label_fraction_recovers_known_decay_and_mapping():
    gamma = np.array([.002, .01]); r0 = np.array([1000., 2000.]); n0 = r0*(-np.expm1(-gamma*120))
    fitted = module.fit_label_baseline(r0, n0, np.array([2., 4.]), .01, plan())
    np.testing.assert_allclose(fitted['gamma'], gamma)
    np.testing.assert_allclose(fitted['mapping_to_S'], .2)
    assert fitted['support'].all()


def test_missing_new_or_old_support_does_not_invent_decay_rates():
    result = module.fit_label_baseline(np.array([100., 100., 10.]), np.array([0., 100., 3.]),
        np.ones(3), .01, plan())
    np.testing.assert_array_equal(result['support'], False)
    np.testing.assert_array_equal(result['gamma'], 0.)


@pytest.mark.parametrize('minute', [15, 30, 60])
def test_onset_reference_recovers_induced_and_repressed_rates_with_long_label_window(minute):
    gamma = np.array([.002, .01]); r0 = np.array([10., 20.]); alpha = np.array([.1, .05])
    total = r0*np.exp(-gamma*minute)+(alpha/gamma)*(-np.expm1(-gamma*minute))
    new = r0*(np.exp(-gamma*minute)-np.exp(-gamma*120))+(alpha/gamma)*(-np.expm1(-gamma*minute))
    baseline = dict(R0=r0, gamma=gamma, support=np.ones(2, dtype=bool))
    actual, diagnostics = module.onset_reference(total, new, baseline, minute, 120.)
    np.testing.assert_allclose(actual, alpha-gamma*total, atol=1e-12)
    assert diagnostics['negative_synthesis_fraction'] == 0.
    assert actual[0] > 0 and actual[1] < 0


def test_long_label_prestimulus_contribution_is_not_called_current_synthesis():
    gamma = np.array([.005]); r0 = np.array([10.]); minute = 15
    total = r0*np.exp(-gamma*minute)
    new = r0*(np.exp(-gamma*minute)-np.exp(-gamma*120))
    actual, _ = module.onset_reference(total, new, dict(R0=r0, gamma=gamma, support=np.ones(1, bool)), minute, 120.)
    np.testing.assert_allclose(actual, -gamma*total, atol=1e-12)
    assert new[0] > 0 and actual[0] < 0


def test_negative_synthesis_clipping_is_reported():
    baseline = dict(R0=np.array([10.]), gamma=np.array([.005]), support=np.ones(1, bool))
    actual, diag = module.onset_reference(np.array([5.]), np.array([0.]), baseline, 15, 120.)
    assert diag['negative_synthesis_fraction'] == 1.
    np.testing.assert_allclose(actual, -.005*5)


def test_invalid_label_interval_and_observations_are_rejected():
    baseline = dict(R0=np.ones(1), gamma=np.ones(1)*.01, support=np.ones(1, bool))
    with pytest.raises(ValueError): module.onset_reference(np.ones(1), np.zeros(1), baseline, 121, 120.)
    with pytest.raises(ValueError): module.onset_reference(np.ones(1), np.ones(1)*2, baseline, 30, 120.)


def test_reverse_velocity_is_a_distinct_nonnegative_column_not_a_free_sign_relabeling():
    populations = {0:dict(state=np.zeros(3)), 15:dict(state=np.ones(3))}
    vectors = {15:dict(velocity=np.array([1., 2., 3.]))}
    x, positive = module.design('S_velocity', populations, vectors, 15, 0)
    reverse, reverse_positive = module.design('S_reverse_velocity', populations, vectors, 15, 0)
    np.testing.assert_array_equal(x[1], -reverse[1])
    assert positive == reverse_positive == [False, True]
