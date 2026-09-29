"""Variance-based velocity diagnostics; not a test of conditional information."""
import numpy as np


def centered_velocity_r2(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.shape != predicted.shape or observed.ndim != 2 or not len(observed):
        raise ValueError('Velocity diagnostic needs aligned nonempty matrices')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite velocity diagnostic')
    residual = float(np.square(observed-predicted).sum())
    variance = float(np.square(observed-observed.mean(axis=0, keepdims=True)).sum())
    if variance <= 1e-12:
        return 1. if residual <= 1e-12 else 0.
    return 1.-residual/variance
