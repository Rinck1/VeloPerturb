import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_6a_source_screen import (affine_cca, clone_map, clone_mean, corr, fit_cca,
                                nuisance_slopes, project_cca, source_residuals)
from w3_validation import conditional_score_test


def test_clone_metadata_excludes_ambiguous_assignments(tmp_path):
    path = tmp_path / 'clones.csv'
    path.write_text('cell_barcode,clone_barcode,n_barcodes\na,c1,1\nb,c2,2\nc,c3,1\n')
    assert clone_map(path, {'a', 'b'}) == {'a': 'c1'}
    assert clone_map(path, {'a', 'b'}, False) == {'a': 'c1', 'b': 'c2'}


def test_clone_means_preserve_registered_order():
    np.testing.assert_allclose(clone_mean(np.array([[1.], [3.], [9.]]), ['a', 'a', 'b'], ['b', 'a']), [[9.], [2.]])
    with pytest.raises(ValueError, match='Empty'):
        clone_mean(np.ones((2, 1)), ['a', 'a'], ['a', 'b'])


def test_source_preprocessing_is_not_fit_on_heldout_cells():
    rng = np.random.default_rng(1)
    s, u = rng.normal(size=(2, 90, 12)); train = np.arange(90) < 60
    a = source_residuals(s, u, train, dimensions=4, neighbors=5)
    altered_s, altered_u = s.copy(), u.copy()
    altered_s[70:] += 100; altered_u[70:] -= 40
    b = source_residuals(altered_s, altered_u, train, dimensions=4, neighbors=5)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x[:70], y[:70], atol=1e-10)


def test_nuisance_slopes_frozen_to_training_cells():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(100, 10)); train = np.arange(100) < 60
    s1, s2 = latent + rng.normal(size=latent.shape) * .2, latent + rng.normal(size=latent.shape) * .2
    u1, u2 = 2 * latent + rng.normal(size=latent.shape) * .2, 2 * latent + rng.normal(size=latent.shape) * .2
    fit = nuisance_slopes(s1, s2, u1, u2, train, min_correlation=.05, max_abs=100)
    u1[~train] += 100
    other = nuisance_slopes(s1, s2, u1, u2, train, min_correlation=.05, max_abs=100)
    for a, b in zip(fit, other):
        np.testing.assert_array_equal(a, b)
    assert fit[2].all()


def test_cca_export_affine_map_matches_fitted_transform():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 100, 20))
    ea, eb = fit_cca(a[:70], b[:70], .1, 10)
    for x, encoder in [(a, ea), (b, eb)]:
        w, offset = affine_cca(encoder, 3)
        np.testing.assert_allclose(x @ w + offset, project_cca(x, encoder)[:, :3], atol=1e-10)


def test_heldout_cca_recovers_a_repeatable_direction():
    rng = np.random.default_rng(4)
    common = rng.normal(size=(240, 1))
    a = common @ rng.normal(size=(1, 20)) + rng.normal(size=(240, 20)) * .2
    b = common @ rng.normal(size=(1, 20)) + rng.normal(size=(240, 20)) * .2
    ea, eb = fit_cca(a[:160], b[:160], .1, 10)
    assert corr(project_cca(a[160:], ea)[:, 0], project_cca(b[160:], eb)[:, 0]) > .9


def test_cca_state_only_common_direction_does_not_imply_increment():
    rng = np.random.default_rng(5); rejections = 0
    for rep in range(40):
        state = rng.normal(size=(240, 3))
        a = state @ rng.normal(size=(3, 15)) + rng.normal(size=(240, 15))
        b = state @ rng.normal(size=(3, 15)) + rng.normal(size=(240, 15))
        ea, eb = fit_cca(a[:160], b[:160], .1, 8)
        x, y = project_cca(a[160:], ea)[:, 0], project_cca(b[160:], eb)[:, 0]
        test = conditional_score_test(y, x, state[160:], simulations=199, seed=rep)
        rejections += test['p_value'] < .05
    assert rejections <= 6
