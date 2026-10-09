"""E4 (condition level): does a condition's mean velocity predict its centroid
displacement deviation from the global mean, on held-out TFs?

prediction = global mean shift + correction; correction regressed on condition
features (PCs of E_c[v], or 8-dim e_c), LOO-CV over 14 train conditions, scored on
the 4 validation TFs by centroid-displacement MSE.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform, load_pack, normalized_counts


def fit_gamma_tail(s, u, q=0.05, min_cells=10):
    ng = s.shape[1]; gamma = np.zeros(ng)
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


def vkin(fold, gene_dir, role, transform):
    values, src, _ = read_gene_input(Path(gene_dir) / f'{role}_gene_source.npz', fold / f'{role}_source.npz')
    ng = values.shape[1] // 2
    s, u = normalized_counts(values[:, ng:], values[:, :ng], transform.target_sum)
    gamma = fit_gamma_tail(s, u)
    return (np.where(gamma > 0, u - s * gamma, 0.0) @ transform.components.T) / transform.velocity_scale


def cond_centroid_disp(src, tgt, conditions):
    d = {}
    for c in sorted(set(conditions)):
        a = src['conditions'] == c; b = tgt['conditions'] == c
        if a.any() and b.any():
            d[str(c)] = tgt['z'][b].mean(0) - src['z'][a].mean(0)
    return d


def cond_mean(feat, src):
    m = {}
    for c in sorted(set(src['conditions'])):
        m[str(c)] = feat[np.asarray(src['conditions']) == c].mean(0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    ap.add_argument('--gene-dir', default='outputs/veloroute_gfg_inputs_20260914')
    ap.add_argument('--conditions', default='data/renge/conditions/esm2_3b_v1/conditions.npz')
    ap.add_argument('--velocity-file', default='/data/yuchang/veloroute_velocity_short_horizon_20260923_v2/frozen_source_vectors.npz')
    ap.add_argument('--velocity-label', default='GFG_native')
    ap.add_argument('--n-vel-pc', type=int, default=5)
    ap.add_argument('--n-esm', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--output', default='outputs/w1_e4_condition_level.json')
    args = ap.parse_args()
    fold = Path(args.fold)
    train, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    train_t, _ = load_pack(fold / 'train_target.npz', expected_side='target')
    val, _ = load_pack(fold / 'validation_source.npz', expected_side='source')
    val_t, _ = load_pack(fold / 'validation_target.npz', expected_side='target')
    with np.load(args.conditions, allow_pickle=False) as data:
        names = [str(c) for c in data['conditions']]
        emb = np.asarray(data['embeddings'], dtype=np.float32)
    esm = PCA(args.n_esm, random_state=0).fit_transform(emb)
    e = {c: esm[i] for i, c in enumerate(names)}
    gfg = np.load(args.velocity_file, allow_pickle=True)
    transform = FrozenSplicingTransform.load(fold / 'transform.npz')
    vels = {
        'v_kin': (vkin(fold, args.gene_dir, 'train', transform), None),
        'GFG': (gfg['train__' + args.velocity_label].astype(np.float32),
                gfg['validation__' + args.velocity_label].astype(np.float32)),
    }
    dtr = cond_centroid_disp(train, train_t, train['conditions'])
    dva = cond_centroid_disp(val, val_t, val['conditions'])
    tc = sorted(dtr); vc = sorted(dva)
    gms = np.mean([dtr[c] for c in tc], 0)

    result = dict(tag='E4_condition', task='RENGE_heldout_TFs', metric='centroid_disp_mse', arms={})
    for vname, (v_tr, v_va) in vels.items():
        if v_va is None:
            v_va = vkin(fold, args.gene_dir, 'validation', transform)
        Ev_tr = cond_mean(v_tr, train); Ev_va = cond_mean(v_va, val)
        # velocity PCs fit across all 23 condition means
        allv = np.vstack([Ev_tr[c] for c in tc] + [Ev_va[c] for c in vc])
        pv = PCA(args.n_vel_pc, random_state=0).fit_transform(allv)
        k = len(tc)
        fE = {c: pv[i] for i, c in enumerate(tc)}; fE.update({c: pv[k + i] for i, c in enumerate(vc)})
        fe = {c: e[c] for c in tc + vc}
        delta = {c: dtr[c] - gms for c in tc}
        rng = np.random.default_rng(0)

        def loo_eval(feat):
            X = np.vstack([feat[c] for c in tc]); y = np.vstack([delta[c] for c in tc])
            pred_val = np.zeros((len(vc), y.shape[1]))
            for j, cv in enumerate(vc):
                errs = []
                for tr_i in range(k):
                    idx = [i for i in range(k) if i != tr_i]
                    r = Ridge(alpha=1.).fit(X[idx], y[idx])
                    errs.append(r.predict(feat[cv][None])[0])
                pred_val[j] = np.mean(errs, 0)  # average LOO models for stability
            out = []
            for j, cv in enumerate(vc):
                dhat = gms + pred_val[j]
                out.append(float(np.square(dhat - dva[cv]).mean()))
            return float(np.mean(out))

        gms_only = float(np.mean([np.square(gms - dva[c]).mean() for c in vc]))
        arms = {'global_mean_shift': gms_only,
                'gms_Ev': loo_eval(fE), 'gms_ec': loo_eval(fe)}
        noise_errs = []
        for s in range(args.seeds):
            rg = np.random.default_rng(s)
            nf = {c: rg.standard_normal(args.n_vel_pc) for c in tc + vc}
            noise_errs.append(loo_eval(nf))
        arms['gms_noise'] = float(np.mean(noise_errs))
        result['arms'][vname] = arms
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
