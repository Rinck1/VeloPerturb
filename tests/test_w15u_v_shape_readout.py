from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15u_v_shape_readout as module


def test_centered_energy_removes_only_global_translation():
    x = np.array([[0., 1.], [1., 0.], [2., 2.]])
    w = np.ones(3)/3
    raw, shape = module.energy_pair(x+10, w, x)
    assert raw > 1 and shape < 1e-7


def test_within_between_decomposition_sums_to_mixture_variance():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 3, 2)); q = rng.dirichlet(np.ones(3), size=4)
    result = module.mixture_variance(x, q)
    mean = (q[:, :, None]*x).sum(1).mean(0)
    total = (q*np.sum((x-mean)**2, axis=-1)).sum(1).mean()
    np.testing.assert_allclose(result['within_source_variance']+result['between_source_variance'], total)


def test_hard_route_has_no_within_source_mixture_variance():
    x = np.array([[[1.], [-1.]], [[2.], [-2.]]]); q = np.array([[1., 0.], [0., 1.]])
    assert module.mixture_variance(x, q)['within_source_fraction'] == 0
