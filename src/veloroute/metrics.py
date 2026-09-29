"""Fixed distribution metric and condition-level paired uncertainty estimates."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .contracts import finite_matrix


def _mean_distances(x, y, block_size):
    total = 0.0
    for start in range(0, len(x), block_size):
        for other in range(0, len(y), block_size):
            total += cdist(x[start:start+block_size], y[other:other+block_size]).sum()
    return total / (len(x) * len(y))


def energy_distance(x, y, *, block_size=256):
    """V-statistic: 2 E||X-Y|| - E||X-X'|| - E||Y-Y'|| (NOT square-rooted)."""
    x, y = finite_matrix(x, "prediction"), finite_matrix(y, "target")
    if x.shape[1] != y.shape[1] or block_size < 1:
        raise ValueError("Metric dimensions/block size invalid")
    value = 2*_mean_distances(x, y, block_size) - _mean_distances(x, x, block_size) - _mean_distances(y, y, block_size)
    if value < -1e-8:
        raise ArithmeticError("Unexpected negative energy V-statistic")
    return float(max(0.0, value))


def paired_condition_bootstrap(rows, *, real_arm, control_arm, seeds, conditions,
                               n_bootstrap, seed, confidence=0.95, metric="energy_distance"):
    if len(conditions) < 2 or len(set(conditions)) != len(conditions):
        raise ValueError("Need at least two distinct preregistered conditions")
    if not seeds or len(set(seeds)) != len(seeds) or n_bootstrap < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap protocol")
    indexed = {}
    for row in rows:
        if row["arm"] not in {real_arm, control_arm}:
            continue
        key = (row["condition"], int(row["seed"]), row["arm"])
        if key[0] not in conditions or key[1] not in seeds:
            raise ValueError("Unregistered condition or seed; cannot silently drop observations")
        if key in indexed:
            raise ValueError("Duplicate condition/seed/arm metric")
        value = float(row[metric])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Error metrics must be finite and nonnegative")
        indexed[key] = value
    differences = []
    for condition in conditions:
        delta = []
        for run_seed in seeds:
            try:
                delta.append(indexed[condition, run_seed, control_arm] - indexed[condition, run_seed, real_arm])
            except KeyError as exc:
                raise ValueError(f"Incomplete paired result grid: {exc}") from exc
        differences.append(float(np.mean(delta)))
    differences = np.array(differences)
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))].mean(1)
    alpha = 1-confidence
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    return {"real_arm": real_arm, "control_arm": control_arm, "gain": float(differences.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "confidence": confidence,
            "n_conditions": len(conditions), "n_seeds": len(seeds),
            "resampling_unit": "condition_after_seed_averaging", "per_condition_gain": differences.tolist()}
