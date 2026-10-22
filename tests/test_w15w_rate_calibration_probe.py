from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15w_rate_calibration_probe as module


def test_positive_two_coefficient_fit_matches_independent_nnls_solution():
    from scipy.optimize import nnls
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 5, 2))
    y = np.einsum('ngi,gi->ng', x, rng.normal(size=(5, 2)))+rng.normal(size=(30, 5))*.1
    fitted = module.fit_coefficients(x, y, ridge=1., positive=True)
    centered = x-x.mean(0)
    scale = np.maximum(np.sqrt((centered**2).mean(0)), 1e-8)
    xn, yn = centered/scale, y-y.mean(0)
    for g in range(5):
        expected, _ = nnls(np.vstack((xn[:, g], np.eye(2))), np.concatenate((yn[:, g], np.zeros(2))))
        np.testing.assert_allclose(fitted['coefficient'][g]*scale[g], expected, atol=1e-9)
    assert (fitted['coefficient'] >= 0).all()


def test_unconstrained_fit_keeps_negative_coefficients_when_needed():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 3, 2))
    y = -x[:, :, 0]+2*x[:, :, 1]+3
    fitted = module.fit_coefficients(x, y, ridge=.001)
    assert (fitted['coefficient'][:, 0] < 0).all()
    np.testing.assert_allclose(module.predict_coefficients(fitted, x), y, atol=.001)


def test_constant_training_features_preserve_training_mean_prediction():
    x = np.ones((10, 3, 2)); y = np.arange(30).reshape(10, 3)
    model = module.fit_coefficients(x, y, positive=True)
    np.testing.assert_array_equal(model['coefficient'], 0.)
    np.testing.assert_allclose(module.predict_coefficients(model, x), np.broadcast_to(y.mean(0), y.shape))


def test_frozen_coefficients_do_not_use_heldout_response_or_recenter_on_queries():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 2, 1)); y = 2*x[:, :, 0]+1
    model = module.fit_coefficients(x, y, positive=True)
    queries = np.array([[[100.], [-20.]], [[.1], [.2]]])
    together = module.predict_coefficients(model, queries)
    separately = np.concatenate([module.predict_coefficients(model, q[None]) for q in queries])
    np.testing.assert_array_equal(together, separately)
