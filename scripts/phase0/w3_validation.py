"""W3 source-only transforms and conditional diagnostics, independent of W2.

The local U control is a matched perturbation, NOT an exact conditional randomizer.
The wild score test assumes independent rows, a correctly specified linear null
mean and adequate residual degrees of freedom. It is asymptotic, not exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def matrix(value):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix')
    return x


def count_parts(counts, parts, rng):
    """Equal-depth multinomial thinning; never round normalized counts to UMIs."""
    if parts < 2:
        raise ValueError('At least two count parts are required')
    is_sparse = sparse.issparse(counts)
    a = sparse.csr_matrix(counts).copy() if is_sparse else np.asarray(counts).copy()
    values = a.data if is_sparse else a
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Count-split requires nonnegative integer raw counts')
    remaining = a.astype(np.int64)
    result = []
    for left in range(parts, 1, -1):
        piece = remaining.copy()
        if is_sparse:
            piece.data = rng.binomial(remaining.data, 1 / left)
            piece.eliminate_zeros()
        else:
            piece = rng.binomial(remaining, 1 / left)
        result.append(piece)
        remaining = remaining - piece
    result.append(remaining)
    return tuple(result)


def split_us(u, s, *, parts=2, seed=0):
    if u.shape != s.shape:
        raise ValueError('U/S shape mismatch')
    u_seed, s_seed = np.random.SeedSequence(seed).spawn(2)
    return (count_parts(u, parts, np.random.default_rng(u_seed)),
            count_parts(s, parts, np.random.default_rng(s_seed)))


@dataclass(frozen=True)
class LogStateProjection:
    """One coordinate map for state, raw-S velocity and mean-shift.

    Inputs must have the same upstream count normalization. This map does not
    estimate library-size derivatives; raw and normalized rates cannot be mixed.
    """
    mean: np.ndarray
    components: np.ndarray
    scale: float

    @classmethod
    def fit(cls, train_s, dimensions=50, seed=0):
        s = matrix(train_s)
        if (s < 0).any() or min(s.shape) < 2:
            raise ValueError('Invalid training S')
        pca = PCA(min(dimensions, len(s) - 1, s.shape[1]), svd_solver='full', random_state=seed)
        z = pca.fit_transform(np.log1p(s))
        scale = max(float(np.sqrt(np.square(z).mean())), 1e-8)
        return cls(pca.mean_.copy(), pca.components_.copy(), scale)

    def state(self, s):
        s = matrix(s)
        if (s < 0).any():
            raise ValueError('Negative S')
        return (np.log1p(s) - self.mean) @ self.components.T / self.scale

    def velocity(self, s, ds_dt):
        s, ds_dt = matrix(s), matrix(ds_dt)
        if s.shape != ds_dt.shape or (s < 0).any():
            raise ValueError('Invalid velocity/S alignment')
        return (ds_dt / (1 + s)) @ self.components.T / self.scale


def local_u_control(us, z, strata, *, seed=0, neighbors=10, match_depth=True):
    """Swap only U between disjoint mutual-kNN pairs within source strata.

    Positive U-depth cells are paired within fixed log2(S-depth) bins. Unmatched
    cells stay unchanged and are counted; no distant donor fallback is allowed.
    Optional scaling preserves each recipient's U depth, not integer UMIs.
    """
    us, z = matrix(us), matrix(z)
    strata = np.asarray(strata).astype(str)
    n, width = us.shape
    if width % 2 or n != len(z) or strata.shape != (n,) or neighbors < 1 or (us < 0).any():
        raise ValueError('Invalid U/S control input')
    genes = width // 2
    u, s = us[:, :genes], us[:, genes:]
    depth = u.sum(1)
    s_bins = np.floor(np.log2(np.maximum(s.sum(1), 1))).astype(int)
    order = np.arange(n)
    kth = np.zeros(n)
    rng = np.random.default_rng(seed)
    for label in np.unique(strata):
        for depth_bin in np.unique(s_bins[strata == label]):
            ids = np.flatnonzero((strata == label) & (s_bins == depth_bin) & (depth > 0))
            if len(ids) < 2:
                continue
            k = min(neighbors, len(ids) - 1)
            dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(z[ids]).kneighbors(z[ids])
            candidate = []
            for i in range(len(ids)):
                valid = idx[i] != i
                candidate.append(idx[i][valid][:k])
                kth[ids[i]] = float(dist[i][valid][:k][-1])
            sets = [set(v) for v in candidate]
            free = set(range(len(ids)))
            for i in rng.permutation(len(ids)):
                if i not in free:
                    continue
                possible = sorted(j for j in candidate[i] if j in free and i in sets[j])
                if not possible:
                    continue
                j = int(rng.choice(possible))
                order[ids[i]], order[ids[j]] = ids[j], ids[i]
                free.remove(i); free.remove(j)
    result = us.copy()
    result[:, :genes] = u[order]
    moved = order != np.arange(n)
    ratio = np.divide(depth[order], depth, out=np.ones(n), where=depth > 0)
    if match_depth:
        factor = np.divide(depth, depth[order], out=np.ones(n), where=depth[order] > 0)
        result[:, :genes] *= factor[:, None]
    distance = np.linalg.norm(z - z[order], axis=1)
    if not np.array_equal(result[:, genes:], s) or len(np.unique(order)) != n:
        raise AssertionError('U-only/bijective control violated')
    audit = dict(moved_fraction=float(moved.mean()) if n else 0.,
                 unchanged_cells=int((~moved).sum()), donor_distance_median=float(np.median(distance[moved])) if moved.any() else None,
                 donor_u_depth_ratio_median=float(np.median(ratio[moved])) if moved.any() else None,
                 all_moved_within_mutual_knn=bool(np.all(distance[moved] <= kth[moved] + 1e-9)),
                 s_unchanged=True, depth_matched=match_depth,
                 control_semantics='matched_perturbation_not_exact_conditional_randomization')
    return result, order, audit


def orthogonal_basis(x, tolerance=1e-10):
    x = matrix(x)
    if x.shape[1] == 0:
        return np.zeros((len(x), 0))
    u, values, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(values > tolerance * max(float(values[0]), 1.))) if len(values) else 0
    return u[:, :rank]


def conditional_score_test(y, x, controls, *, simulations=999, seed=0):
    """HC2 multiplier score bootstrap for a linear conditional-mean null.

    Directions/hyperparameters must be fixed independently of this test's Y.
    Rows must be independent units (e.g. clones, not cells within a clone).
    The two-sided joint statistic detects mixed-sign coefficients.
    """
    y, x, controls = matrix(y), matrix(x), matrix(controls)
    n = len(y)
    if y.shape[1] != 1 or len(x) != n or len(controls) != n or simulations < 99:
        raise ValueError('Invalid conditional-score dimensions or simulation budget')
    c = orthogonal_basis(np.column_stack((np.ones(n), controls)))
    xr = x - c @ (c.T @ x)
    q = orthogonal_basis(xr)
    if n - c.shape[1] - q.shape[1] < 20:
        raise ValueError('Insufficient residual degrees of freedom')
    residual = (y - c @ (c.T @ y)).ravel()
    if q.shape[1] == 0 or np.linalg.norm(residual) < 1e-10:
        return dict(p_value=1., statistic=0., added_rank=q.shape[1], n=n,
                    identifiable=False, method='HC2_multiplier_score_linear_null')
    leverage = np.square(c).sum(1)
    adjusted = residual / np.sqrt(np.maximum(1 - leverage, 1e-8))
    scores = q * adjusted[:, None]
    covariance = scores.T @ scores
    inv = np.linalg.pinv(covariance, rcond=1e-10)
    observed = residual @ q
    stat = float(observed @ inv @ observed)
    rng = np.random.default_rng(seed)
    null = []
    for start in range(0, simulations, 128):
        signs = rng.choice([-1., 1.], size=(min(128, simulations - start), n))
        b = signs @ scores
        null.extend(np.einsum('bi,ij,bj->b', b, inv, b).tolist())
    p = (1 + int(np.sum(np.asarray(null) >= stat))) / (simulations + 1)
    return dict(p_value=float(p), statistic=stat, added_rank=q.shape[1], n=n,
                identifiable=True, partial_r2=float(np.square(q.T @ residual).sum() / np.square(residual).sum()),
                method='HC2_multiplier_score_linear_null',
                assumptions='independent_units_correct_linear_null_mean_asymptotic_not_exact')


def two_stage_iv(y, noisy_state, extra, instrument):
    """2SLS with intercept and extra in BOTH stages; weak-IV diagnostic only.

    Instrument exclusion/measurement independence cannot be established by F.
    Return coefficients, not gene-iid inferential p-values.
    """
    y, s, u, w = [matrix(v) for v in (y, noisy_state, extra, instrument)]
    if any(v.shape != (len(y), 1) for v in (y, s, u, w)) or len(y) < 10:
        raise ValueError('IV requires aligned scalar vectors')
    exog = np.column_stack((np.ones(len(y)), u))
    design = np.column_stack((exog, s))
    instruments = np.column_stack((exog, w))
    if np.linalg.matrix_rank(instruments) != 3 or np.linalg.matrix_rank(design) != 3:
        raise ValueError('Rank-deficient IV design')
    q = orthogonal_basis(instruments)
    fitted_design = q @ (q.T @ design)
    normal = design.T @ fitted_design
    if np.linalg.cond(normal) > 1e12:
        raise ValueError('Unidentified/ill-conditioned IV system')
    beta = np.linalg.solve(normal, fitted_design.T @ y).ravel()
    reduced = s - exog @ np.linalg.lstsq(exog, s, rcond=None)[0]
    full = s - q @ (q.T @ s)
    rss0, rss1 = float(np.square(reduced).sum()), float(np.square(full).sum())
    first_f = max(0., (rss0 - rss1) / max(rss1 / (len(y) - 3), 1e-15))
    return dict(extra_coefficient=float(beta[1]), state_coefficient=float(beta[2]),
                intercept=float(beta[0]), first_stage_f=first_f, weak_instrument=first_f < 10,
                inference='coefficients_only_genes_not_assumed_independent')
