"""E4: population-level velocity readout on held-out RENGE TFs.

Compare ridge models predicting the day5 displacement (NN barycenter) and score
by PSEUDOBULK MSE per held-out condition:
  1. z
  2. global mean shift            (condition-agnostic constant)
  3. z + low-dim e_c  (ESM -> 8 PCA dims)
  4. z + condition-mean velocity E_c[v]
  5. z + covariance-matched noise (per-condition, cov matched to v)

If (4) beats (2) and (3) on unseen TFs, "velocity as a population readout of the
perturbation state" survives. Run for v_kin (steady-state) and GFG velocities.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform, load_pack, normalized_counts

MIN_CELLS = 20


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


def fit_gamma_tail(s, u, q=0.05, min_cells=10):
    ng = s.shape[1]
    gamma = np.zeros(ng)
    qlo = np.quantile(s, q, axis=0); qhi = np.quantile(s, 1 - q, axis=0)
    for g in range(ng):
        tail = (s[:, g] <= qlo[g]) | (s[:, g] >= qhi[g])
        if tail.sum() < min_cells:
            continue
        ss, uu = s[tail, g], u[tail, g]
        d = float((ss * ss).sum())
        if d > 1e-12:
            gamma[g] = float((ss * uu).sum() / d)
    return gamma


def vkin_for(fold, gene_dir, transform):
    values, source, _ = read_gene_input(gene_dir / 'train_gene_source.npz', fold / 'train_source.npz')
    ng = values.shape[1] // 2
    s, u = normalized_counts(values[:, ng:], values[:, :ng], transform.target_sum)
    gamma = fit_gamma_tail(s, u)
    return (np.where(gamma > 0, u - s * gamma, 0.0) @ transform.components.T) / transform.velocity_scale


def pseudobulk_errors(source, target, pred, conditions):
    errs = []
    for c in sorted(set(conditions)):
        a = source['conditions'] == c
        b = target['conditions'] == c
        if int(a.sum()) < MIN_CELLS or int(b.sum()) < MIN_CELLS:
            continue
        ep = (source['z'][a] + pred[a]).mean(0)
        y = target['z'][b].mean(0)
        errs.append(float(np.square(ep - y).mean()))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    ap.add_argument('--gene-dir', default='outputs/veloroute_gfg_inputs_20260914')
    ap.add_argument('--conditions', default='data/renge/conditions/esm2_3b_v1/conditions.npz')
    ap.add_argument('--velocity-file', default='/data/yuchang/veloroute_velocity_short_horizon_20260923_v2/frozen_source_vectors.npz')
    ap.add_argument('--velocity-label', default='GFG_native')
    ap.add_argument('--n-esm', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--output', default='outputs/w1_e4_population_readout.json')
    args = ap.parse_args()
    fold = Path(args.fold)

    train, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    train_target, _ = load_pack(fold / 'train_target.npz', expected_side='target')
    val, _ = load_pack(fold / 'validation_source.npz', expected_side='source')
    val_target, _ = load_pack(fold / 'validation_target.npz', expected_side='target')
    with np.load(args.conditions, allow_pickle=False) as data:
        cond_names = [str(c) for c in data['conditions']]
        emb = np.asarray(data['embeddings'], dtype=np.float32)
    esm8 = PCA(args.n_esm, random_state=0).fit_transform(emb)
    e8 = {c: esm8[i] for i, c in enumerate(cond_names)}

    transform = FrozenSplicingTransform.load(fold / 'transform.npz')
    gfg = np.load(args.velocity_file, allow_pickle=True)
    vels = {
        'v_kin': (vkin_for(fold, Path(args.gene_dir), transform), None),
        'GFG': (gfg['train__' + args.velocity_label].astype(np.float32),
                gfg['validation__' + args.velocity_label].astype(np.float32)),
    }
    disp_train = nn_displacement(train, train_target)
    gms = disp_train.mean(0)

    def ec8(src):
        return np.vstack([e8[str(c)] for c in src['conditions']])

    def cond_mean_vel(v_tr, v_va, src, role):
        m = {}
        vv = v_tr if role == 'train' else v_va
        for c in sorted(set(src['conditions'])):
            m[str(c)] = vv[np.asarray(src['conditions']) == c].mean(0)
        return np.vstack([m[str(c)] for c in src['conditions']])

    result = dict(tag='E4', velocity_label=args.velocity_label, n_esm=args.n_esm,
                  task='RENGE_day4_to_day5_heldout_TFs', metric='pseudobulk_mse', arms={})
    for vname, (v_tr, v_va) in vels.items():
        if v_va is None:
            v_va = v_tr  # v_kin has no validation-specific variant (same estimator)
        # recompute v_kin for validation source with the same transform
        if vname == 'v_kin':
            values, vsrc, _ = read_gene_input(Path(args.gene_dir) / 'validation_gene_source.npz', fold / 'validation_source.npz')
            ng = values.shape[1] // 2
            s, u = normalized_counts(values[:, ng:], values[:, :ng], transform.target_sum)
            gamma = fit_gamma_tail(s, u)
            v_va = (np.where(gamma > 0, u - s * gamma, 0.0) @ transform.components.T) / transform.velocity_scale
        Ev_tr = cond_mean_vel(v_tr, v_va, train, 'train')
        Ev_va = cond_mean_vel(v_tr, v_va, val, 'validation')
        cov = np.cov(v_tr.T)
        arms = {
            'z': (train['z'], val['z']),
            'z_ec8': (np.concatenate([train['z'], ec8(train)], 1), np.concatenate([val['z'], ec8(val)], 1)),
            'z_Ev': (np.concatenate([train['z'], Ev_tr], 1), np.concatenate([val['z'], Ev_va], 1)),
        }
        out = {}
        # arms that include z
        for arm, (xtr, xva) in arms.items():
            r = Ridge(alpha=1.).fit(xtr, disp_train)
            pred = r.predict(xva)
            out[arm] = float(np.mean(pseudobulk_errors(val, val_target, pred, val['conditions'])))
        # global mean shift (constant)
        out['global_mean_shift'] = float(np.mean(pseudobulk_errors(
            val, val_target, np.broadcast_to(gms, val['z'].shape), val['conditions'])))
        # covariance-matched noise (per condition), averaged over seeds
        rng = np.random.default_rng(0)
        noise_errs = []
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            nmap = {str(c): rng.multivariate_normal(np.zeros(v_tr.shape[1]), cov) for c in sorted(set(train['conditions']))}
            ntr = np.vstack([nmap[str(c)] for c in train['conditions']])
            nmap_va = {str(c): rng.multivariate_normal(np.zeros(v_tr.shape[1]), cov) for c in sorted(set(val['conditions']))}
            nva = np.vstack([nmap_va[str(c)] for c in val['conditions']])
            r = Ridge(alpha=1.).fit(np.concatenate([train['z'], ntr], 1), disp_train)
            pred = r.predict(np.concatenate([val['z'], nva], 1))
            noise_errs.append(float(np.mean(pseudobulk_errors(val, val_target, pred, val['conditions']))))
        out['z_noise'] = float(np.mean(noise_errs))
        result['arms'][vname] = out
    result['best_arm'] = {v: min(o, key=o.get) for v, o in result['arms'].items()}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
