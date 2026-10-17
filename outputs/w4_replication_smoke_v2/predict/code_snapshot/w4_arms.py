"""Equal-budget W4 population predictors and TF-level decision rules."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


RIDGE_GRID = (.1, 1., 10., 100., 1000.)
CORE_ARMS = ('static', 'vel_pool', 'vel_mean', 'vel_pool_Uswap', 'pseudo_S', 'direct_U_pool')


def design(features, embeddings, arm):
    base = np.concatenate((features['state'], embeddings), axis=1)
    if arm == 'static':
        return base
    if arm not in CORE_ARMS or arm not in features:
        raise ValueError(f'Unimplemented arm (no silent fallback): {arm}')
    return np.concatenate((base, features[arm]), axis=1)


def fit_ridge_training_cv(x, y, grid=RIDGE_GRID, *, controls=None):
    """Inner leave-one-training-TF; controls stay in training, not CV units.

    Refit scaling and intercept inside EACH split. An exact dual eigensolve
    evaluates the alpha grid together; final model is sklearn Ridge.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    controls = np.zeros(len(x), dtype=bool) if controls is None else np.asarray(controls, dtype=bool)
    grid = np.asarray(grid, dtype=float)
    if len(x) != len(y) or controls.shape != (len(x),) or (~controls).sum() < 4:
        raise ValueError('At least four distinct training TFs are required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError('Nonfinite features/targets or invalid ridge grid')
    errors = []
    for i in np.flatnonzero(~controls):
        keep = np.arange(len(x)) != i
        scale = StandardScaler().fit(x[keep])
        a, test = scale.transform(x[keep]), scale.transform(x[i:i+1])
        # StandardScaler centers a. Center y separately for unpenalized intercept.
        ymean = y[keep].mean(0); centered = y[keep]-ymean
        eig, vec = np.linalg.eigh(a @ a.T)
        eig = np.maximum(eig, 0.)
        left, right = (test @ a.T) @ vec, vec.T @ centered
        predictions = ymean + np.einsum('i,ia,ij->aj', left[0], 1/(eig[:, None]+grid), right)
        errors.append(np.square(predictions-y[i]).sum(1))
    scores = np.mean(errors, axis=0)
    # Exact score ties prefer stronger regularization, as fixed in the draft.
    tied = np.flatnonzero(scores == scores.min())
    chosen = int(tied[np.argmax(grid[tied])])
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=float(grid[chosen])).fit(scaler.transform(x), y)
    return scaler, model, dict(alpha=float(grid[chosen]), grid=grid.tolist(),
        inner_TF_sse=scores.tolist(), inner_units=int((~controls).sum()),
        scaler_scope='refitted_on_each_inner_training_subset')


def predict_arms(train_features, test_features, train_target_means, embeddings_train, embeddings_test,
                 *, arms=CORE_ARMS, controls=None, grid=RIDGE_GRID):
    delta = np.asarray(train_target_means)-train_features['state']
    predictions, selections = {}, {}
    for arm in arms:
        x, xe = design(train_features, embeddings_train, arm), design(test_features, embeddings_test, arm)
        scaler, model, selection = fit_ridge_training_cv(x, delta, grid, controls=controls)
        predictions[arm] = test_features['state']+model.predict(scaler.transform(xe))
        selections[arm] = selection
    return predictions, selections


def paired_gain_test(real_error, ref_error, static_error, *, seed=0, repetitions=10000):
    """Rows are TFs, columns fixed time intervals, NOT independent cells/seeds."""
    real, ref, base = [np.asarray(a, dtype=float) for a in (real_error, ref_error, static_error)]
    if real.shape != ref.shape or real.shape != base.shape or real.ndim != 2 or len(real) < 2:
        raise ValueError('Paired TF-by-time error arrays required')
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (real, ref, base)):
        raise ValueError('Invalid squared errors')
    gains = (ref-real)/np.maximum(base, 1e-12)
    tf = gains.mean(1)
    rng = np.random.default_rng(seed)
    bootstrap = tf[rng.integers(len(tf), size=(repetitions, len(tf)))].mean(1)
    lo, hi = np.quantile(bootstrap, [.025, .975])
    p = 1. if np.all(tf == 0) else float(wilcoxon(tf, alternative='greater').pvalue)
    return dict(gain=float(tf.mean()), ci_low=float(lo), ci_high=float(hi), wilcoxon_p=p,
        by_interval=gains.mean(0).tolist(), independent_TFs=len(tf), denominator_floors=int((base < 1e-12).sum()),
        supported=bool(lo > 0 and p < .05 and (gains.mean(0) > 0).all()))


def average_predictions_then_errors(seed_predictions, truth):
    pred, truth = np.asarray(seed_predictions), np.asarray(truth)
    if pred.ndim != truth.ndim+1 or pred.shape[1:] != truth.shape:
        raise ValueError('Seed predictions and common-coordinate target mismatch')
    return np.square(pred.mean(0)-truth).sum(-1)
