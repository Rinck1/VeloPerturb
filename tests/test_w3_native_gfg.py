import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_7_native_train import raw_variants, row_normalization, velocity_comparison


def test_u_controls_preserve_selected_depth_and_local_condition():
    rng = np.random.default_rng(1)
    u = rng.poisson(3, size=(100, 7)).astype(float)
    u[0] = 0
    s = rng.poisson(4, size=u.shape).astype(float)
    state = rng.normal(size=(100, 4)); labels = np.repeat(['A', 'B'], 50)
    variants, audit = raw_variants(u, s, state, labels, seed=0, neighbors=8)
    for k in ('V3_local_U', 'V4_global_U'):
        np.testing.assert_allclose(variants[k].sum(1), u.sum(1))
    np.testing.assert_array_equal(labels[audit['local_order']], labels)
    np.testing.assert_array_equal(variants['V0_real'], u)
    assert not variants['V7_zero_U'].any()
    assert audit['global_moved_fraction'] == .99


def test_frozen_normalization_recovers_factor_and_rejects_gene_scaling():
    raw = np.array([[1., 2., 3.], [4., 5., 6.]])
    normalized = raw * np.array([.3, 2.])[:, None]
    np.testing.assert_allclose(row_normalization(raw, normalized), [.3, 2.])
    with pytest.raises(ValueError, match='normalization'):
        row_normalization(raw, normalized * [1, 1, 2])


def test_zero_velocity_is_not_a_valid_direction():
    result = velocity_comparison(np.ones((5, 3)), np.zeros((5, 3)))
    assert result['cos_median'] is None
    assert result['valid_direction_fraction'] == 0
    assert result['norm_ratio_median'] == 0
