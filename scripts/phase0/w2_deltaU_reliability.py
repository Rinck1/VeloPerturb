"""W2 T3: DeltaU pseudobulk reliability + condition-level lead-lag (RENGE).

T3.1 pseudobulk per (condition, day): sum counts -> CPM+log1p; Delta = pb(c,d) - pb(ref,d),
     ref = same-day non-targeting control (CTRL/AAVS1).
T3.2 reliability: cell-level count-split into 3 (p=1/3): C picks response genes
     (top-200 |dS| per condition); A,B give correlations, Spearman-Brown x2.
     eta = dU - b*dS with IV b; eta reliability.
T3.3 condition-level lead-lag: partial corr(dS_day5, dU_day4[A] | dS_day4[B]) per
     condition, mean over conditions; null shuffles day5 condition pairing.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')

CONTROLS = ['CTRL', 'AAVS1']
VAL_TF = ['LIN28A', 'NANOG', 'POU5F1', 'ZIC3']


def log(m):
    print(f'[T3] {m}', flush=True)


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', '/home/yuchang/wangjiaxuan', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def split3_counts(mat, rng):
    a = mat.astype(np.int64).copy(); a.data = rng.binomial(a.data, 1 / 3)
    rest = mat.astype(np.int64) - a
    b = rest.copy(); b.data = rng.binomial(rest.data, 0.5)
    c = rest - b
    for x in (a, b, c):
        x.eliminate_zeros()
    return a.tocsr(), b.tocsr(), c.tocsr()


def cpm_logp(counts, lib=None):
    """counts: cells x genes sparse; pseudobulk = sum over cells then CPM+log1p."""
    total = np.asarray(counts.sum(1)).ravel()
    keep = total > 0
    return keep, None


def pb(counts, idx, library=None):
    """pseudobulk over cell subset idx: sum counts, CPM (per 1e4) + log1p."""
    sub = counts[idx]
    colsum = np.asarray(sub.sum(0)).ravel().astype(np.float64)
    if library is None:
        library = colsum.sum()
    if library <= 0:
        return np.zeros_like(colsum)
    return np.log1p(colsum / library * 1e4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day4', default='data/renge/processed_release_v1/day4/day4.h5ad')
    ap.add_argument('--day5', default='data/renge/processed_release_v1/day5/day5.h5ad')
    ap.add_argument('--n-hvg', type=int, default=2000)
    ap.add_argument('--n-response', type=int, default=200)
    ap.add_argument('--n-perm', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default='outputs/w2_t3_deltaU_reliability.json')
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    d4 = ad.read_h5ad(args.day4)
    d5 = ad.read_h5ad(args.day5)
    genes = np.array([str(g) for g in d4.var['gene_symbol']])
    conds = {str(c) for c in d4.obs['condition'].unique()} | {str(c) for c in d5.obs['condition'].unique()}
    train_tf = sorted([c for c in conds if c not in VAL_TF and c not in CONTROLS and c != 'unassigned'])
    val_tf = [c for c in VAL_TF if c in conds]
    log(f'conds: train={len(train_tf)} val={len(val_tf)} controls={[c for c in CONTROLS if c in conds]}')

    # HVG from control S (both days)
    c4 = d4.obs['condition'].isin(CONTROLS).to_numpy()
    c5 = d5.obs['condition'].isin(CONTROLS).to_numpy()
    S4 = sparse.csr_matrix(d4.layers['spliced']); S5 = sparse.csr_matrix(d5.layers['spliced'])
    ctrl_s = sparse.vstack([S4[c4], S5[c5]]).tocsr()
    logs = ctrl_s.astype(np.float32).copy(); logs.data = np.log1p(logs.data)
    expressed = np.asarray((ctrl_s > 0).sum(0)).ravel() >= 10
    var = np.asarray(logs.power(2).mean(0)).ravel() - np.asarray(logs.mean(0)).ravel() ** 2
    var[~expressed] = -1
    sel = np.sort(np.argsort(-var)[:args.n_hvg])
    S4 = S4[:, sel].tocsr(); S5 = S5[:, sel].tocsr()
    U4 = sparse.csr_matrix(d4.layers['unspliced'])[:, sel].tocsr()
    U5 = sparse.csr_matrix(d5.layers['unspliced'])[:, sel].tocsr()
    T_targets = train_tf + val_tf
    ref = [c for c in CONTROLS if c in conds]
    ref_day = {}
    for day, obs in ((4, d4.obs), (5, d5.obs)):
        present = [c for c in ref if (obs['condition'] == c).any()]
        ref_day[day] = present[0] if present else None
    ref_mode = 'same_day_nontargeting_control' if all(ref_day.values()) else 'leave_one_out_train_mean'

    def day_data(day):
        s, u, obs = (S4, U4, d4.obs) if day == 4 else (S5, U5, d5.obs)
        return s, u, obs

    # T3.1/T3.2 per condition (train+val) reliability of dU, dS, eta
    rows = []
    resp_genes = {}
    for day in (4, 5):
        s, u, obs = day_data(day)
        refc = ref_day[day]
        ref_idx = np.flatnonzero(obs['condition'].to_numpy() == refc) if refc else None
        # 3-way split of ALL cells of this day
        sa, sb, sc = split3_counts(s, rng)
        ua, ub, uc = split3_counts(u, rng)
        # reference library size per half from ref cells
        for cond in T_targets:
            m = (obs['condition'].to_numpy() == cond)
            if m.sum() < 20:
                continue
            def delta(hs, hu, hv, ref_idx):
                lib_c = np.asarray(hs[m].sum()).sum() + np.asarray(hu[m].sum()).sum()
                lib_r = np.asarray(hs[ref_idx].sum()).sum() + np.asarray(hu[ref_idx].sum()).sum() if ref_idx is not None else None
                pbc_u = pb(hu, np.flatnonzero(m)); pbc_s = pb(hs, np.flatnonzero(m))
                pbr_u = pb(hu, ref_idx); pbr_s = pb(hs, ref_idx)
                return (pbc_u - pbr_u), (pbc_s - pbr_s)
            dUA, dSA = delta(sa, ua, None, ref_idx)
            dUB, dSB = delta(sb, ub, None, ref_idx)
            dUC, dSC = delta(sc, uc, None, ref_idx)
            # response genes from C half
            top = np.argsort(-np.abs(dSC))[:args.n_response]
            resp_genes[(cond, day)] = top.tolist()
            # eta = dU - b*dS, IV b per condition
            num = (dUB * dSA).mean() - dUB.mean() * dSA.mean()
            den = (dSB * dSA).mean() - dSB.mean() * dSA.mean()
            b = num / den if abs(den) > 1e-12 else 0.0
            etaA = dUA - b * dSA; etaB = dUB - b * dSB
            def sb_corr(x, y, ids):
                xx, yy = x[ids], y[ids]
                xx = xx - xx.mean(); yy = yy - yy.mean()
                d = np.sqrt((xx ** 2).sum() * (yy ** 2).sum())
                r = float((xx * yy).sum() / d) if d > 1e-12 else 0.0
                r = np.clip(r, -0.999, 0.999)
                return 2 * r / (1 + r)
            allids = np.arange(len(sel))
            rows.append(dict(condition=cond, day=day, n_cells=int(m.sum()),
                             dU_rel_all=sb_corr(dUA, dUB, allids), dU_rel_resp=sb_corr(dUA, dUB, top),
                             dS_rel_all=sb_corr(dSA, dSB, allids), dS_rel_resp=sb_corr(dSA, dSB, top),
                             eta_rel_all=sb_corr(etaA, etaB, allids), eta_rel_resp=sb_corr(etaA, etaB, top)))
    result = dict(task='T3', git_commit=git_commit(), args=vars(args), ref_mode=ref_mode, rows=rows)

    # aggregate stats
    def agg(key, filt=None):
        v = [r[key] for r in rows if (filt is None or filt(r))]
        return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))
    result['summary'] = dict(
        dU_rel_resp=agg('dU_rel_resp'), eta_rel_resp=agg('eta_rel_resp'),
        dS_rel_resp=agg('dS_rel_resp'), dU_rel_all=agg('dU_rel_all'))
    log(f"dU_rel_resp median={result['summary']['dU_rel_resp']['median']:.3f} "
        f"eta_rel_resp={result['summary']['eta_rel_resp']['median']:.3f} dS_rel_resp={result['summary']['dS_rel_resp']['median']:.3f}")

    # T3.3 condition-level lead-lag on training conditions
    # per condition: partial corr(dS_day5, dU_day4[A] | dS_day4[B]) over genes
    ll_rows = {}
    d4s, d4u, obs4 = day_data(4)
    d5s, d5u, obs5 = day_data(5)
    # count-split day4 (A,B) and day5 (single, or split for library consistency)
    s4a, s4b, _ = split3_counts(d4s, rng)
    u4a, u4b, _ = split3_counts(d4u, rng)
    ref4 = ref_day[4]; ref5 = ref_day[5]
    def delta_day(s, u, obs, cond, refc):
        m = np.flatnonzero(obs['condition'].to_numpy() == cond)
        r = np.flatnonzero(obs['condition'].to_numpy() == refc)
        if len(m) < 20 or len(r) < 5:
            return None, None
        return (pb(u, m) - pb(u, r)), (pb(s, m) - pb(s, r))
    dU4 = {}; dS4 = {}; dS5 = {}
    for cond in train_tf + val_tf:
        a_u, a_s = delta_day(s4a, u4a, obs4, cond, ref4)
        b_u, b_s = delta_day(s4b, u4b, obs4, cond, ref4)
        _, dS5[cond] = delta_day(d5s, d5u, obs5, cond, ref5)
        dU4[cond] = (a_u, b_u); dS4[cond] = (a_s, b_s)
    for cond in train_tf:
        a_u, b_u = dU4[cond]; a_s, b_s = dS4[cond]
        y5 = dS5[cond]
        if y5 is None or a_u is None or b_u is None:
            continue
        def pcorr(Y, X, C):
            Cc = C - C.mean(); den = (Cc ** 2).sum()
            Yc = Y - Y.mean() - (Cc * (Y - Y.mean())).sum() / den * Cc if den > 1e-12 else Y - Y.mean()
            Xc = X - X.mean() - (Cc * (X - X.mean())).sum() / den * Cc if den > 1e-12 else X - X.mean()
            d = np.sqrt((Yc ** 2).sum() * (Xc ** 2).sum())
            return float((Yc * Xc).sum() / d) if d > 1e-12 else 0.0
        ll_rows[cond] = dict(obs=pcorr(y5, a_u, b_s), ctrl=pcorr(y5, a_s, b_s))
    obs_mean = float(np.mean([v['obs'] for v in ll_rows.values()])) if ll_rows else 0.0
    ctrl_mean = float(np.mean([v['ctrl'] for v in ll_rows.values()])) if ll_rows else 0.0
    conds_ll = list(ll_rows)
    rngp = np.random.default_rng(0)
    null = []
    for _ in range(args.n_perm):
        perm = rngp.permutation(len(conds_ll))
        vals = []
        for i, cond in enumerate(conds_ll):
            y5p = dS5[conds_ll[perm[i]]]
            a_u, b_u = dU4[cond]; a_s, b_s = dS4[cond]
            if y5p is None:
                continue
            Cc = b_s - b_s.mean(); den = (Cc ** 2).sum()
            Yc = y5p - y5p.mean() - (Cc * (y5p - y5p.mean())).sum() / den * Cc
            Xc = a_u - a_u.mean() - (Cc * (a_u - a_u.mean())).sum() / den * Cc
            d = np.sqrt((Yc ** 2).sum() * (Xc ** 2).sum())
            vals.append(float((Yc * Xc).sum() / d) if d > 1e-12 else 0.0)
        null.append(np.mean(vals))
    null = np.array(null)
    result['t3_3_leadlag'] = dict(n_conditions=len(ll_rows), obs=obs_mean, ctrl_dS=ctrl_mean,
                                  null95=float(np.percentile(null, 95)),
                                  p_value=float((1 + (null >= obs_mean).sum()) / (args.n_perm + 1)),
                                  passes=bool(obs_mean > np.percentile(null, 95) and obs_mean > ctrl_mean))
    log(f"T3.3 leadlag obs={obs_mean:.4f} ctrl={ctrl_mean:.4f} null95={result['t3_3_leadlag']['null95']:.4f} p={result['t3_3_leadlag']['p_value']:.3f}")
    result['runtime_sec'] = time.time() - t0
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
