from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15y_velocity_variance_flux as module
from test_w15x_u_increment_screen import fixture as increment_fixture


def fixture():
    data, plan = increment_fixture(); rng = np.random.default_rng(15)
    n, g = data['y'].shape
    data['variance'] = np.ones((n, g))
    data['flux'] = {name:rng.normal(size=(n, g)) for name in
        ('velocity_positive', 'direct_U', 'shuffle_0', 'shuffle_1', 'shuffle_2')}
    data['y'] = 3*data['flux']['velocity_positive']/1.1+.1*data['local'][:, :, 0]
    plan['variance_floor_fraction'] = .1
    return data, plan


def test_covariance_matches_deterministic_variance_derivative():
    rng = np.random.default_rng(3); z, v = rng.normal(size=(100, 4)), rng.normal(size=(100, 4))
    dt = .0001
    derivative = ((z+dt*v).var(0)-(z-dt*v).var(0))/(2*dt)
    np.testing.assert_allclose(derivative, 2*module.covariance_diagonal(z, v), atol=1e-10)
    np.testing.assert_allclose(module.covariance_diagonal(z, -v), -module.covariance_diagonal(z, v), atol=1e-12)


def test_shuffle_preserves_source_library_and_singletons():
    ids = np.array(['B1:a', 'B1:b', 'B2:c', 'B2:d', 'B3:e'])
    order = module.stratified_permutation(ids, 0, 'KO')
    np.testing.assert_array_equal(np.sort(order), np.arange(5))
    libraries = np.array([x.split(':')[0] for x in ids])
    np.testing.assert_array_equal(libraries[order], libraries)
    assert order[-1] == 4 and np.all(order[:4] != np.arange(4))


def test_known_positive_expansion_signal_improves_variance_prediction():
    data, plan = fixture(); train, query = np.arange(36), np.arange(36, 48)
    pred, alpha = module.predict_fold(data, train, query, plan)
    assert 2.8 < alpha['velocity_positive'] < 3.2
    assert alpha['velocity_reverse'] == 0.
    real_mse = np.mean((pred['velocity_positive']-data['y'][query])**2)
    base_mse = np.mean((pred['S']-data['y'][query])**2)
    assert real_mse < base_mse*.1


def test_wrong_direction_is_identified_instead_of_called_forward_velocity():
    data, plan = fixture(); data['y'] *= -1
    _, alpha = module.predict_fold(data, np.arange(36), np.arange(36, 48), plan)
    assert alpha['velocity_positive'] == 0.
    assert alpha['velocity_reverse'] > 0.
    assert alpha['velocity_free'] < 0.


def test_outer_target_variance_never_enters_training_or_source_prediction():
    data, plan = fixture(); train, query = np.arange(36), np.arange(36, 48)
    pred, alpha = module.predict_fold(data, train, query, plan)
    data['y'][query] = 1e8
    other, other_alpha = module.predict_fold(data, train, query, plan)
    for name in pred: np.testing.assert_array_equal(pred[name], other[name])
    assert alpha == other_alpha
