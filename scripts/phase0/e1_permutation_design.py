"""E1: permutation-design empirical test (RENGE).

Spec: ridge regression from (z, v) predicts the day5 displacement (pseudo-bulk).
Two zero-designs for v:
  * within-condition z-matched permutation -> velocity-specific gain must be ~0
    (validates the project's null-design theorem);
  * cross-condition z-matched permutation -> condition field destroyed; the real
    arm should retain a positive population-level gain.

Gain = error(permuted v) - error(real v); positive favours real velocity.
Condition bootstrap CIs. Two endpoints retained:
  * ridge: fit [z,v]->NN-barycenter displacement on training conditions;
  * raw  : z + scale*v with training-only robust scale.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.latent import load_pack

MIN_CELLS = 20


def z_matched_permute(v, z, conditions, k=10, seed=0, cross=False):
    rng = np.random.default_rng(seed)
    out = np.array(v, copy=True)
    pool = np.arange(len(z))
    for c in sorted(set(conditions)):
        m = np.flatnonzero(np.asarray(conditions) == c)
        if len(m) < 2:
            continue
        cand = pool if cross else m
        zc = z[cand]
        nn = NearestNeighbors(n_neighbors=min(k, len(cand))).fit(zc)
        idx = nn.kneighbors(z[m], return_distance=False)
        for i in range(len(m)):
            out[m[i]] = v[cand[idx[i][rng.integers(len(idx[i]))]]]
    return out


def robust_scale(v, displacement, conditions):
    ratios = []
    for c in sorted(set(conditions)):
        m = np.asarray(conditions) == c
        vm, dm = v[m].mean(0), displacement[m].mean(0)
        if np.linalg.norm(vm) > 1e-10:
            ratios.append(np.linalg.norm(dm) / np.linalg.norm(vm))
    return float(np.median(ratios)) if ratios else 1.0


def nn_displacement(source, target, neighbors=10):
    result = np.zeros_like(source['z'])
    for c in sorted(set(source['conditions'])):
        a = source['conditions'] == c
        b = target['conditions'] == c
        if not b.any():
            continue
        nn = NearestNeighbors(n_neighbors=min(neighbors, int(b.sum()))).fit(target['z'][b])
        idx = nn.kneighbors(source['z'][a], return_distance=False)
        result[a] = target['z'][b][idx].mean(1) - source['z'][a]
    return result


def energy(x, y):
    x, y = np.asarray(x, dtype='float64'), np.asarray(y, dtype='float64')
    return float(max(0., 2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()))


def errors_by_condition(source, target, endpoint_disp):
    rows = {}
    for c in sorted(set(source['conditions'])):
        a = source['conditions'] == c
        b = target['conditions'] == c
        if int(a.sum()) < MIN_CELLS or int(b.sum()) < MIN_CELLS:
            continue
        endpoint = source['z'][a] + endpoint_disp[a]
        y = target['z'][b]
        rows[str(c)] = dict(pseudobulk_mse=float(np.square(endpoint.mean(0) - y.mean(0)).mean()),
                            energy=energy(endpoint, y))
    return rows


def bootstrap_gain(gains, seed=20260923, n=10000):
    rng = np.random.default_rng(seed)
    gains = np.asarray(gains)
    boot = gains[rng.integers(len(gains), size=(n, len(gains)))].mean(1)
    return float(gains.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    ap.add_argument('--output', default='outputs/w1_e1_permutation_design.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    fold = Path(args.fold)
    train, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    train_target, _ = load_pack(fold / 'train_target.npz', expected_side='target')
    val, _ = load_pack(fold / 'validation_source.npz', expected_side='source')
    val_target, _ = load_pack(fold / 'validation_target.npz', expected_side='target')

    disp_train = nn_displacement(train, train_target)
    scale = robust_scale(train['velocity'], disp_train, train['conditions'])
    ridge = Ridge(alpha=1.).fit(np.concatenate([train['z'], train['velocity']], 1), disp_train)

    def ridge_disp(zv):
        return ridge.predict(zv)

    def raw_disp(z, v):
        return scale * v

    result = dict(tag='E1', fold=str(fold), scale=scale)
    for role, src, tgt in [('train', train, train_target), ('validation', val, val_target)]:
        real_v = src['velocity']
        within_v = z_matched_permute(real_v, src['z'], src['conditions'], k=10, seed=args.seed, cross=False)
        cross_v = z_matched_permute(real_v, src['z'], src['conditions'], k=10, seed=args.seed, cross=True)
        for model, disp_fn in [('ridge', ridge_disp), ('raw', raw_disp)]:
            def endpoint(vv):
                return disp_fn(np.concatenate([src['z'], vv], 1)) if model == 'ridge' else disp_fn(src['z'], vv)
            real = errors_by_condition(src, tgt, endpoint(real_v))
            within = errors_by_condition(src, tgt, endpoint(within_v))
            cross = errors_by_condition(src, tgt, endpoint(cross_v))
            for metric in ('energy', 'pseudobulk_mse'):
                common = sorted(set(real) & set(within) & set(cross))
                g_within = [within[c][metric] - real[c][metric] for c in common]
                g_cross = [cross[c][metric] - real[c][metric] for c in common]
                mw, lw, hw = bootstrap_gain(g_within)
                mc, lc, hc = bootstrap_gain(g_cross)
                result[f'{role}_{model}_{metric}'] = dict(
                    n_conditions=len(common),
                    within_gain=mw, within_ci=[lw, hw], within_crosses_zero=bool(lw <= 0 <= hw),
                    cross_gain=mc, cross_ci=[lc, hc], cross_crosses_zero=bool(lc <= 0 <= hc))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
