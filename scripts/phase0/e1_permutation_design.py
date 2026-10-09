"""E1: permutation-design empirical test (RENGE), corrected.

Permutation (both variants are proper bijections, no self-sampling):
  * within-condition: permute_local (within condition, depth bin, z-sorted blocks
    of ``block`` cells, cyclic roll);
  * cross-condition: same z-sorted depth-block roll but pooling all conditions.

Premise check: after permutation, the cross-fitted z->v R2 must stay ~= the
unpermuted value (permutation must preserve p(v|z)).

Models (ridge on training conditions, endpoint z + predicted displacement):
  [z], [z,e_c], [z,v], [z,v,e_c].
Gain = error(permuted v) - error(real v); positive favours real velocity.
Condition bootstrap CIs over 4 validation conditions; repeated over seeds.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.latent import load_pack
from veloroute.probes import permute_local
from veloroute.velocity_diagnostics import centered_velocity_r2

MIN_CELLS = 20


def cross_permute(v, source, block=8, seed=0):
    rng = np.random.default_rng(seed)
    order = np.arange(len(v))
    depth = np.floor(np.log2(np.maximum(source['depth'], 1))).astype(int)
    for db in np.unique(depth):
        group = np.flatnonzero(depth == db)
        group = group[np.argsort(source['z'][group, 0], kind='stable')]
        for start in range(0, len(group), block):
            blk = group[start:start + block]
            if len(blk) > 1:
                order[blk] = np.roll(blk, int(rng.integers(1, len(blk))))
    return np.asarray(v)[order].copy()


def z_kmeans_permute(v, z, conditions, per_cluster=6, seed=0, cross=False):
    """Bijective permutation within KMeans clusters in the FULL z space.

    Preserves p(v|z) far better than PC1-sorted blocks because matching uses all
    z dimensions. within: clusters are condition-restricted; cross: global.
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    n = len(z)
    order = np.arange(n)
    groups = [np.arange(n)] if cross else [np.flatnonzero(np.asarray(conditions) == c)
                                           for c in sorted(set(conditions))]
    for g in groups:
        if len(g) < 2:
            continue
        k = max(1, len(g) // per_cluster)
        lab = KMeans(n_clusters=min(k, len(g)), n_init=3, random_state=seed).fit_predict(z[g])
        for c in np.unique(lab):
            idx = g[lab == c]
            if len(idx) > 1:
                order[idx] = idx[rng.permutation(len(idx))]
    return np.asarray(v)[order].copy()


def crossfit_fhat(z, v, seed=0):
    pred = np.zeros_like(v)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(z):
        pred[te] = Ridge(alpha=1.).fit(z[tr], v[tr]).predict(z[te])
    return pred


def residual_permute(v, z, conditions, seed=0, cross=False):
    """Freedman-Lane: v* = f_hat(z) + permuted residuals. E[v|z] preserved by
    construction, so the premise R2(z->v*) ~= R2(z->v) holds."""
    rng = np.random.default_rng(seed)
    f = crossfit_fhat(z, v, seed=seed)
    r = v - f
    rp = r.copy()
    if cross:
        rp = r[rng.permutation(len(r))]
    else:
        for c in sorted(set(conditions)):
            idx = np.flatnonzero(np.asarray(conditions) == c)
            rp[idx] = r[rng.permutation(idx)]
    return f + rp


def cv_z_to_v_r2(z, v, seed=0):
    pred = np.zeros_like(v)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(z):
        pred[te] = Ridge(alpha=1.).fit(z[tr], v[tr]).predict(z[te])
    return float(centered_velocity_r2(v, pred))


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
    ap.add_argument('--conditions', default='data/renge/conditions/esm2_3b_v1/conditions.npz')
    ap.add_argument('--block', type=int, default=8)
    ap.add_argument('--perm-method', default='residual', choices=['residual', 'kmeans', 'local'])
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--velocity-file', default=None, help='npz with train__<label> and validation__<label> velocities')
    ap.add_argument('--velocity-label', default='GFG_joint')
    ap.add_argument('--output', default='outputs/w1_e1_permutation_design.json')
    args = ap.parse_args()
    fold = Path(args.fold)
    train, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    train_target, _ = load_pack(fold / 'train_target.npz', expected_side='target')
    val, _ = load_pack(fold / 'validation_source.npz', expected_side='source')
    val_target, _ = load_pack(fold / 'validation_target.npz', expected_side='target')
    if args.velocity_file:
        vd = np.load(args.velocity_file, allow_pickle=True)
        V = {'train': vd[f'train__{args.velocity_label}'].astype(np.float32),
             'validation': vd[f'validation__{args.velocity_label}'].astype(np.float32)}
    else:
        V = {'train': train['velocity'], 'validation': val['velocity']}
    with np.load(args.conditions, allow_pickle=False) as data:
        cond_names = [str(c) for c in data['conditions']]
        embeddings = np.asarray(data['embeddings'], dtype=np.float32)
    cond_lookup = {c: embeddings[i] for i, c in enumerate(cond_names)}

    disp_train = nn_displacement(train, train_target)
    seeds = [int(s) for s in args.seeds.split(',')]

    def within_perm(v, src, seed):
        if args.perm_method == 'residual':
            return residual_permute(v, src['z'], src['conditions'], seed=seed, cross=False)
        if args.perm_method == 'kmeans':
            return z_kmeans_permute(v, src['z'], src['conditions'], per_cluster=args.block, seed=seed, cross=False)
        return permute_local(v, src, seed=seed, neighbors=args.block)[0]

    def cross_perm(v, src, seed):
        if args.perm_method == 'residual':
            return residual_permute(v, src['z'], src['conditions'], seed=seed, cross=True)
        if args.perm_method == 'kmeans':
            return z_kmeans_permute(v, src['z'], src['conditions'], per_cluster=args.block, seed=seed, cross=True)
        return cross_permute(v, src, block=args.block, seed=seed)

    # premise check on the training set (cross-fitted z->v R2)
    prem = dict(velocity_label=args.velocity_label, perm_method=args.perm_method,
                unpermuted_ztov_r2=cv_z_to_v_r2(train['z'], V['train']))
    prem['within_permuted_ztov_r2'] = [cv_z_to_v_r2(train['z'], within_perm(V['train'], train, s)) for s in seeds]
    prem['cross_permuted_ztov_r2'] = [cv_z_to_v_r2(train['z'], cross_perm(V['train'], train, s)) for s in seeds]

    def ec(src):
        return np.vstack([cond_lookup[str(c)] for c in src['conditions']])

    def endpoint(model, src, v):
        z = src['z']; c = ec(src)
        if model == 'z':
            x = z
        elif model == 'z_ec':
            x = np.concatenate([z, c], 1)
        elif model == 'z_v':
            x = np.concatenate([z, v], 1)
        else:
            x = np.concatenate([z, v, c], 1)
        return model, x

    def fit_predict(model, tr_src, tr_v, eval_src, eval_v):
        _, xtr = endpoint(model, tr_src, tr_v)
        r = Ridge(alpha=1.).fit(xtr, disp_train)
        _, xe = endpoint(model, eval_src, eval_v)
        return r.predict(xe)

    result = dict(tag='E1', fold=str(fold), block=args.block, seeds=seeds, premise=prem)
    # baseline model comparison on validation conditions (real v)
    baseline = {}
    for model in ('z', 'z_ec', 'z_v', 'z_v_ec'):
        pred = fit_predict(model, train, V['train'], val, V['validation'])
        errs = {}
        for c in sorted(set(val['conditions'])):
            a = val['conditions'] == c
            b = val_target['conditions'] == c
            if int(a.sum()) < MIN_CELLS or int(b.sum()) < MIN_CELLS:
                continue
            ep = val['z'][a] + pred[a]
            y = val_target['z'][b]
            errs[str(c)] = dict(energy=energy(ep, y), pseudobulk_mse=float(np.square(ep.mean(0) - y.mean(0)).mean()))
        baseline[model] = {k: float(np.mean([e[k] for e in errs.values()])) for k in ('energy', 'pseudobulk_mse')}
    result['validation_baselines'] = baseline

    # permutation tests with real-arm ridge [z,v]
    for role, src, tgt in [('train', train, train_target), ('validation', val, val_target)]:
        for metric in ('energy', 'pseudobulk_mse'):
            gw, gc = [], []
            for s in seeds:
                real_pred = fit_predict('z_v', train, V['train'], src, V[role])
                within_v = within_perm(V[role], src, s)
                cross_v = cross_perm(V[role], src, s)
                wpred = fit_predict('z_v', train, V['train'], src, within_v)
                cpred = fit_predict('z_v', train, V['train'], src, cross_v)
                real = errors_by_condition(src, tgt, real_pred)
                wi = errors_by_condition(src, tgt, wpred)
                cr = errors_by_condition(src, tgt, cpred)
                common = sorted(set(real) & set(wi) & set(cr))
                gw.append(np.mean([wi[c][metric] - real[c][metric] for c in common]))
                gc.append(np.mean([cr[c][metric] - real[c][metric] for c in common]))
            mw, lw, hw = bootstrap_gain(gw, seed=20260923)
            mc, lc, hc = bootstrap_gain(gc, seed=20260923)
            result[f'{role}_{metric}'] = dict(
                n_seeds=len(seeds), seed_mean_within=float(np.mean(gw)),
                within_gain=mw, within_ci=[lw, hw], within_crosses_zero=bool(lw <= 0 <= hw),
                cross_gain=mc, cross_ci=[lc, hc], cross_crosses_zero=bool(lc <= 0 <= hc))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
