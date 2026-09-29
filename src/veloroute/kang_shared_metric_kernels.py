#!/usr/bin/env python
"""CPU sample-based diagnostics; see reports/multimodal_evaluation_20260914.md.

The peak scores describe fixed 1-D projections, NOT joint-distribution mode
counts. D4 eligibility and bootstrap stability must be established separately.
PRDC definitions: https://proceedings.mlr.press/v119/naeem20a.html
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite 2-D array')
    return value


def prdc(real, predicted, k=5, chunk_size=256):
    """Precision/recall/density/coverage, with bounded cross-distance memory.

    Strict ball membership matches the published reference implementation.
    Density is not a probability: it may exceed 1 and is not 'higher is better'.
    """
    real, predicted = _matrix(real, 'real'), _matrix(predicted, 'predicted')
    if real.shape[1] != predicted.shape[1]:
        raise ValueError('feature dimensions differ')
    if not isinstance(k, (int, np.integer)) or k < 1 or min(len(real), len(predicted)) <= k:
        raise ValueError('both sample counts must exceed positive integer k')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    rr = cKDTree(real).query(real, k=k + 1)[0][:, -1]
    rp = cKDTree(predicted).query(predicted, k=k + 1)[0][:, -1]
    precision_hits = np.zeros(len(predicted), dtype=bool)
    density_hits = np.zeros(len(predicted), dtype=np.int64)
    recall_hits = coverage_hits = 0
    for start in range(0, len(real), chunk_size):
        distances = cdist(real[start:start + chunk_size], predicted)
        in_real = distances < rr[start:start + chunk_size, None]
        precision_hits |= in_real.any(axis=0)
        density_hits += in_real.sum(axis=0)
        recall_hits += (distances < rp[None, :]).any(axis=1).sum()
        coverage_hits += in_real.any(axis=1).sum()
    return {
        'precision': float(precision_hits.mean()),
        'recall': float(recall_hits / len(real)),
        'density': float(density_hits.mean() / k),
        'coverage': float(coverage_hits / len(real)),
        'real_zero_radius_fraction': float((rr == 0).mean()),
        'predicted_zero_radius_fraction': float((rp == 0).mean()),
    }


def _kde(samples, grid, bandwidth):
    # Absolute bandwidth shared by reference, real test and generated samples.
    density = np.zeros(len(grid))
    for start in range(0, len(samples), 256):
        z = (grid[:, None] - samples[None, start:start + 256]) / bandwidth
        density += np.exp(-0.5 * z ** 2).sum(axis=1)
    return density / (len(samples) * bandwidth * np.sqrt(2 * np.pi))


def _match_peaks(real, predicted, tolerance):
    if len(real) == 0 or len(predicted) == 0:
        return []
    distances = np.abs(real[:, None] - predicted[None, :])
    # Prioritize maximum cardinality under the tolerance, then distance.
    penalty = max(len(real), len(predicted)) + 1.0
    costs = np.where(distances <= tolerance, distances / tolerance, penalty)
    rows, cols = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if distances[i, j] <= tolerance]


def projected_peak_metrics(reference, real, predicted, *, bandwidth=None,
                           grid_size=1024, prominence_fraction=0.05,
                           match_bandwidths=1.0, valley_height_fraction=0.5,
                           rare_mass_threshold=0.1, min_cells=50,
                           min_mode_cells=10):
    """Pilot diagnostics in one predeclared projection.

    A separate real-only reference fixes bandwidth, grid, peak threshold and
    basins. 'ok' is numerical eligibility, not a D4-positive declaration.
    """
    vectors = []
    for name, value in [('reference', reference), ('real', real), ('predicted', predicted)]:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be a nonempty finite 1-D array')
        vectors.append(value)
    reference, real, predicted = vectors
    if not (0 < prominence_fraction < 1 and 0 < valley_height_fraction < 1
            and 0 < rare_mass_threshold < 1 and match_bandwidths > 0):
        raise ValueError('invalid peak thresholds')
    if grid_size < 64 or min_cells < 2 or min_mode_cells < 1:
        raise ValueError('invalid grid or minimum sample size')
    result = {'status': 'insufficient_cells', 'n_reference': len(reference),
              'n_real': len(real), 'n_predicted': len(predicted)}
    if min(len(reference), len(real), len(predicted)) < min_cells:
        return result
    if bandwidth is None:
        bandwidth = np.std(reference, ddof=1) * len(reference) ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= np.finfo(float).eps:
        result['status'] = 'degenerate_reference'
        return result
    left, right = reference.min() - 4 * bandwidth, reference.max() + 4 * bandwidth
    grid = np.linspace(left, right, grid_size)
    ref_density = _kde(reference, grid, bandwidth)
    threshold = prominence_fraction * ref_density.max()
    spacing = max(1, int(np.ceil(bandwidth / (grid[1] - grid[0]))))

    def peaks(density):
        return find_peaks(density, prominence=threshold, distance=spacing)[0]

    ref_peaks = peaks(ref_density)
    true_density, pred_density = _kde(real, grid, bandwidth), _kde(predicted, grid, bandwidth)
    true_peaks, pred_peaks = peaks(true_density), peaks(pred_density)
    ref_positions, true_positions, pred_positions = grid[ref_peaks], grid[true_peaks], grid[pred_peaks]
    tolerance = match_bandwidths * bandwidth
    matches = _match_peaks(true_positions, pred_positions, tolerance)
    reference_matches = _match_peaks(ref_positions, true_positions, tolerance)
    n_true, n_pred = len(true_peaks), len(pred_peaks)
    recall = len(matches) / n_true if n_true else None
    precision = len(matches) / n_pred if n_pred else 0.0
    f1 = 2 * len(matches) / (n_true + n_pred) if n_true else None
    valley_indices = [int(a + np.argmin(ref_density[a:b + 1]))
                      for a, b in zip(ref_peaks[:-1], ref_peaks[1:])]
    edges = np.array([-np.inf, *grid[valley_indices], np.inf])
    ref_counts = np.histogram(reference, bins=edges)[0]
    real_counts = np.histogram(real, bins=edges)[0]
    pred_counts = np.histogram(predicted, bins=edges)[0]
    p, q = real_counts / len(real), pred_counts / len(predicted)
    stable = (len(ref_peaks) == n_true == len(reference_matches) and n_true > 0)
    enough_per_mode = bool((ref_counts >= min_mode_cells).all() and (real_counts >= min_mode_cells).all())
    status = 'ok' if stable and enough_per_mode else (
        'insufficient_cells_per_basin' if not enough_per_mode else 'reference_test_peak_disagreement')
    if n_true == 0 or len(ref_peaks) == 0:
        status = 'no_resolved_real_peak'
    # Reference basins cover the full line, including tails. Tail mass is
    # reported separately so points far beyond the fixed grid remain visible.
    basin_rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        xr, xp = real[(real >= lo) & (real < hi)], predicted[(predicted >= lo) & (predicted < hi)]
        basin_rows.append({
            'basin': i, 'n_real': len(xr), 'n_predicted': len(xp),
            'real_mass': float(p[i]), 'predicted_mass': float(q[i]),
            'w1': float(wasserstein_distance(xr, xp)) if len(xr) and len(xp) else None,
            'center_error_in_bandwidths': float(abs(xr.mean() - xp.mean()) / bandwidth)
            if len(xr) and len(xp) else None,
        })
    valley_rows = []
    for a, b, v in zip(ref_peaks[:-1], ref_peaks[1:], valley_indices):
        ceiling = valley_height_fraction * min(ref_density[a], ref_density[b])
        if ref_density[v] >= ceiling:
            continue
        lo = hi = v
        while lo > a and ref_density[lo - 1] <= ceiling:
            lo -= 1
        while hi < b and ref_density[hi + 1] <= ceiling:
            hi += 1
        bounds = (float(grid[lo]), float(grid[hi]))
        mass_real = float(((real >= bounds[0]) & (real <= bounds[1])).mean())
        mass_pred = float(((predicted >= bounds[0]) & (predicted <= bounds[1])).mean())
        valley_rows.append({'interval': bounds, 'real_mass': mass_real, 'predicted_mass': mass_pred,
                            'excess': max(0.0, mass_pred - mass_real)})
    rare_recall = None
    rare_count = 0
    if status == 'ok':
        rare = np.flatnonzero(p <= rare_mass_threshold)
        rare_count = len(rare)
        matched_true = {i for i, _ in matches}
        # Same count and 1-D distance matching imply order-preserving mapping.
        ref_to_true = dict(reference_matches)
        if rare_count:
            rare_recall = float(np.mean([ref_to_true[i] in matched_true for i in rare]))
    result.update({
        'status': status, 'bandwidth': float(bandwidth), 'grid_bounds': [float(left), float(right)],
        'reference_peak_positions': ref_positions.tolist(),
        'real_peak_positions': true_positions.tolist(), 'predicted_peak_positions': pred_positions.tolist(),
        'n_real_peaks': n_true, 'n_predicted_peaks': n_pred,
        'peak_recall': recall, 'peak_precision': precision, 'peak_f1': f1,
        'missing_peak_rate': 1 - recall if recall is not None else None,
        'spurious_peak_rate': 1 - precision if n_pred else None,
        'peak_count_error': n_pred - n_true, 'peak_count_absolute_error': abs(n_pred - n_true),
        'peak_location_error_in_bandwidths': float(np.mean([
            abs(true_positions[i] - pred_positions[j]) / bandwidth for i, j in matches])) if matches else None,
        'mode_mass_tv': float(np.abs(p - q).sum() / 2),
        'rare_peak_recall': rare_recall, 'n_eligible_rare_peaks': rare_count,
        'valley_mass_excess': float(sum(v['excess'] for v in valley_rows)) if valley_rows else None,
        'real_outside_grid_mass': float(((real < left) | (real > right)).mean()),
        'predicted_outside_grid_mass': float(((predicted < left) | (predicted > right)).mean()),
        'per_basin': basin_rows, 'valleys': valley_rows,
    })
    return result

# Vendored verbatim pure functions from the user-provided baseline evaluator.
UPSTREAM_PATH = "/home/yuchang/caixujia/scripts/evaluation/multimodal_metrics.py"
UPSTREAM_SHA256 = "6fb77c34a865ff032e935cf6af3289bbadd71eaf310ddb532d8c3ed6fb142e28"

