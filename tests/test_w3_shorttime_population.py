import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_shorttime_population import assert_shared_geometry, fit_ridge_training_cv, population_means
from w3_validation import LogStateProjection
from w3_gfg_v2_control import seeded_control


def test_coordinate_mismatch_is_rejected_before_fitting():
    rng = np.random.default_rng(0); s = rng.poisson(5, (40, 6)).astype(float)
    u = rng.poisson(3, (40, 6)).astype(float); p = LogStateProjection.fit(s, dimensions=3)
    z = p.state(s)
    assert_shared_geometry(np.concatenate((u, s), 1), dict(z=z), p)
    with pytest.raises(AssertionError):
        assert_shared_geometry(np.concatenate((u, s), 1), dict(z=z+1), p)


def test_means_use_declared_labels_and_refuse_missing_population():
    x = np.array([[1., 2.], [3., 4.], [9., 8.]])
    np.testing.assert_allclose(population_means(x, ['A', 'A', 'B'], ['B', 'A']), [[9, 8], [2, 3]])
    with pytest.raises(ValueError):
        population_means(x, ['A', 'A', 'B'], ['C'])


def test_no_jvp_estimator_backpropagates_without_derivative_decoder(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('JVP forbidden for short-time estimator')
    monkeypatch.setattr(torch.autograd.functional, 'jvp', forbidden)
    m = seeded_control(8, seed=0, state_dim=3, hidden=(16, 16), codes=4)
    values = torch.poisson(torch.full((24, 16), 3.))
    ids = [str(i) for i in range(24)]; m.fit_source(values, cell_ids=ids, allowed_ids=set(ids))
    u, s = values.chunk(2, -1)
    out = m.core(u, s, m.u_mean, m.u_std, m.s_mean, m.s_std, m.innovation(u, s), m.gamma_ss)
    out['native'].backward()
    assert torch.isfinite(m.core.vel_head.weight.grad).all()
    assert m.core.vel_head.weight.grad.abs().sum()>0


def test_group_ridge_selects_only_declared_training_grid():
    rng = np.random.default_rng(0); x = rng.normal(size=(14, 5)); y = x[:, :2]+.1*rng.normal(size=(14, 2))
    scaler, model, selection = fit_ridge_training_cv(x, y, [.1, 1., 10.])
    assert selection['alpha'] in [.1, 1., 10.]
    assert model.predict(scaler.transform(x[:2])).shape == (2, 2)
