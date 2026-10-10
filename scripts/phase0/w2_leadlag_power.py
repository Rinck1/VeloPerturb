"""W2 T1: lead-lag power control (MeRLin Day0 310 -> Day21 308).

T1.1 vectorize partial correlation + regression check against leadlag_meclin.py.
T1.2 separate permutation nulls for kappa0 and eps_S control (n_perm=2000).
T1.3 positive controls P1 (S0_B ~ S21), P2 (eps_S0_A ~ S21), P3 (=eps_S control).
T1.4 clone-level reliability (Day0 triple split A1/A2/B; Day21 double H1/H2).
T1.5 signal injection into kappa0 with lambda sweep -> MDE.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.preprocess import load_usa
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def log(m):
    print(f'[T1] {m}', flush=True)


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', '/home/yuchang/wangjiaxuan', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def count_split(mat, rng):
    a = mat.astype(np.int64).copy(); a.data = rng.binomial(a.data, 0.5)
    b = mat.astype(np.int64) - a
    a.eliminate_zeros(); b.eliminate_zeros()
    return a.tocsr(), b.tocsr()


def count_split3(mat, rng):
    a1 = mat.astype(np.int64).copy(); a1.data = rng.binomial(a1.data, 1 / 3)
    rest = mat.astype(np.int64) - a1
    a2 = rest.copy(); a2.data = rng.binomial(rest.data, 0.5)
    b = rest - a2
    for x in (a1, a2, b):
        x.eliminate_zeros()
    return a1.tocsr(), a2.tocsr(), b.tocsr()


def normalize_log(mat, target=10000.0):
    tot = np.asarray(mat.sum(1)).ravel().astype(np.float64)
    sc = np.where(tot > 0, target / np.maximum(tot, 1e-9), 0.0)
    out = mat.astype(np.float32).multiply(sc[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def knn_residual(target_log, z, k=30):
    n = len(z)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    kk = idx.shape[1]
    rows = np.repeat(np.arange(n), kk)
    W = sparse.csr_matrix((np.full(n * kk, 1.0 / kk), (rows, idx.ravel())), shape=(n, n))
    return (target_log - W @ target_log).astype(np.float32)


def iv_b(u_a, u_b, s_a, s_b):
    """b_g from independent count-split halves."""
    num = (u_b * s_a).mean(0) - u_b.mean(0) * s_a.mean(0)
    den = (s_b * s_a).mean(0) - s_b.mean(0) * s_a.mean(0)
    return np.where(np.abs(den) > 1e-12, num / den, 0.0)


def vec_partial_corr(Y, X, C):
    """Vectorized mean over columns of partial corr(Y, X | C). Y,X,C: clones x genes."""
    Cc = C - C.mean(0, keepdims=True)
    den = (Cc ** 2).sum(0)

    def resid(A):
        Ac = A - A.mean(0, keepdims=True)
        b = np.where(den > 1e-12, (Cc * Ac).sum(0) / np.where(den > 1e-12, den, 1.0), 0.0)
        return Ac - Cc * b
    ry, rx = resid(Y), resid(X)
    num = (ry * rx).sum(0)
    dnm = np.sqrt((ry ** 2).sum(0) * (rx ** 2).sum(0))
    valid = (np.std(Y, 0) > 1e-9) & (np.std(X, 0) > 1e-9) & (dnm > 1e-12)
    r = np.where(valid, num / np.where(dnm > 1e-12, dnm, 1.0), np.nan)
    return float(np.nanmean(r)), int(valid.sum())


def vec_partial_percol(Y, X, C):
    Cc = C - C.mean(0, keepdims=True)
    den = (Cc ** 2).sum(0)

    def resid(A):
        Ac = A - A.mean(0, keepdims=True)
        b = np.where(den > 1e-12, (Cc * Ac).sum(0) / np.where(den > 1e-12, den, 1.0), 0.0)
        return Ac - Cc * b
    ry, rx = resid(Y), resid(X)
    num = (ry * rx).sum(0); dnm = np.sqrt((ry ** 2).sum(0) * (rx ** 2).sum(0))
    return num / np.where(dnm > 1e-12, dnm, np.nan)


def orig_partial_corr_stat(Y, X, C):
    """Reference (non-vectorized) implementation copied from leadlag_meclin.py."""
    vals = []
    for g in range(Y.shape[1]):
        y, x, c = Y[:, g], X[:, g], C[:, g]
        if np.std(y) < 1e-9 or np.std(x) < 1e-9:
            continue
        def resid(a):
            cc = c - c.mean(); dd = (cc ** 2).sum()
            if dd < 1e-12:
                return a - a.mean()
            return a - a.mean() - (cc * (a - a.mean())).sum() / dd * cc
        ry, rx = resid(y), resid(x)
        if np.std(ry) < 1e-9 or np.std(rx) < 1e-9:
            continue
        vals.append(float(np.corrcoef(ry, rx)[0, 1]))
    return (float(np.mean(vals)) if vals else 0.0), len(vals)


def load_clones(path):
    m = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m[r['cell_barcode']] = r['clone_barcode']
    return m


def clmean(cell, lab, use):
    out = []
    for c in use:
        m = np.flatnonzero(lab == c)
        out.append(cell[m].mean(0))
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--day0', default='310')
    ap.add_argument('--day21', default='308')
    ap.add_argument('--top-hvg', type=int, default=2000)
    ap.add_argument('--min-cells', type=int, default=3)
    ap.add_argument('--n-perm', type=int, default=2000)
    ap.add_argument('--inject-reps', type=int, default=50)
    ap.add_argument('--inject-perm', type=int, default=500)
    ap.add_argument('--output', default='outputs/w2_t1_leadlag_power.json')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    root = Path(args.root)
    rng = np.random.default_rng(0)

    def load_run(tag):
        rows, genes, mats, meta = load_usa(root / f'quant_merlin{tag}' / 'af_quant')
        return rows, genes, mats[0].tocsr(), mats[1].tocsr()

    r0, genes0, s0, u0 = load_run(args.day0)
    r21, genes21, s21, u21 = load_run(args.day21)
    name_map = {}
    with (root / f'quant_merlin{args.day0}' / 'af_quant' / 'gene_id_to_name.tsv').open() as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 2:
                name_map[p[0]] = p[1]
    symbols = np.array([name_map.get(g, '') for g in genes0])
    combs = sparse.vstack([s0, s21]).tocsr()
    tot = np.asarray(combs.sum(1)).ravel(); keepc = tot >= 100
    logs = normalize_log(combs[keepc]).astype(np.float32)
    expressed = np.asarray((combs[keepc] > 0).sum(0)).ravel() >= 10
    var = np.asarray(logs.power(2).mean(0)).ravel() - np.asarray(logs.mean(0)).ravel() ** 2
    var[~expressed] = -1
    sel = np.sort(np.argsort(-var)[:args.top_hvg])
    s0 = s0[:, sel].tocsr(); u0 = u0[:, sel].tocsr(); s21 = s21[:, sel].tocsr()
    keep0 = np.asarray((s0 + u0).sum(1)).ravel() >= 100
    keep21 = np.asarray(s21.sum(1)).ravel() >= 50
    s0, u0, r0 = s0[keep0], u0[keep0], np.array(r0)[keep0]
    s21, r21 = s21[keep21], np.array(r21)[keep21]
    cmap0 = load_clones(root / f'clones/SRR33960{args.day0}' / 'cell_clone_assignments.csv')
    cmap21 = load_clones(root / f'clones/SRR33960{args.day21}' / 'cell_clone_assignments.csv')
    lab0 = np.array([cmap0.get(b, '') for b in r0], dtype=object)
    lab21 = np.array([cmap21.get(b, '') for b in r21], dtype=object)
    shared = set(cmap0.values()) & set(cmap21.values())
    vals0, cnt0 = np.unique(lab0[lab0 != ''], return_counts=True)
    c0 = set(vals0[cnt0 >= args.min_cells])
    c21 = set(lab21[lab21 != ''])
    use = sorted(shared & c0 & c21)
    log(f'usable clones={len(use)}')

    # Day0 two-way split: kappa/eps_S from A, S0 from B
    s0a, s0b = count_split(s0, rng); u0a, u0b = count_split(u0, rng)
    za = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s0a).todense(), dtype=np.float32))
    zb = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s0b).todense(), dtype=np.float32))
    eUa = knn_residual(np.asarray(normalize_log(u0a).todense(), dtype=np.float32), za)
    eSa = knn_residual(np.asarray(normalize_log(s0a).todense(), dtype=np.float32), za)
    eUb = knn_residual(np.asarray(normalize_log(u0b).todense(), dtype=np.float32), zb)
    eSb = knn_residual(np.asarray(normalize_log(s0b).todense(), dtype=np.float32), zb)
    b_iv = iv_b(eUa, eUb, eSa, eSb)
    kappa0 = eUa - eSa * b_iv
    S0c = np.asarray(normalize_log(s0b).todense(), dtype=np.float32)
    S21c = np.asarray(normalize_log(s21).todense(), dtype=np.float32)
    K0 = clmean(kappa0, lab0, use); ES0 = clmean(eSa, lab0, use); S0 = clmean(S0c, lab0, use); S21 = clmean(S21c, lab21, use)

    # modules
    modules = {'ALL': np.ones(len(sel), bool)}
    progs = json.loads((root / 'merlin_programs.json').read_text())['signatures']
    sym2idx = {s: i for i, s in enumerate(symbols[sel])}
    for k, p in progs.items():
        gs = [str(g).replace('*', '').strip() for g in (p.get('genes', p) if isinstance(p, dict) else p)]
        idx = np.array([sym2idx[g] for g in gs if g in sym2idx])
        if len(idx) >= 5:
            modules[k] = np.isin(np.arange(len(sel)), idx)

    # ---- T1.1 regression check ----
    reg = {}
    for mname, mask in modules.items():
        gidx = np.flatnonzero(mask)
        o_vec, _ = vec_partial_corr(S21[:, gidx], K0[:, gidx], S0[:, gidx])
        o_orig, _ = orig_partial_corr_stat(S21[:, gidx], K0[:, gidx], S0[:, gidx])
        c_vec, _ = vec_partial_corr(S21[:, gidx], ES0[:, gidx], S0[:, gidx])
        c_orig, _ = orig_partial_corr_stat(S21[:, gidx], ES0[:, gidx], S0[:, gidx])
        reg[mname] = dict(vec_kappa=o_vec, orig_kappa=o_orig, vec_epsS=c_vec, orig_epsS=c_orig)
    # per-gene check on 50 random genes
    rngc = np.random.default_rng(1)
    gi = rngc.choice(len(sel), 50, replace=False)
    pv = vec_partial_percol(S21[:, gi], K0[:, gi], S0[:, gi])
    po = np.array([orig_partial_corr_stat(S21[:, [g]], K0[:, [g]], S0[:, [g]])[0] for g in gi])
    per_gene_max_abs = float(np.nanmax(np.abs(pv - po)))
    log(f'T1.1 regression: ALL vec={reg["ALL"]["vec_kappa"]:.4f} orig={reg["ALL"]["orig_kappa"]:.4f}; '
        f'per-gene max|diff|={per_gene_max_abs:.2e}')
    result = dict(task='T1', git_commit=git_commit(), args=vars(args),
                  n_clones=len(use), t1_1_regression=reg, t1_1_per_gene_max_abs_diff=per_gene_max_abs)
    Path(args.output).write_text(json.dumps(result, indent=2))
    if args.check_only:
        return

    # ---- T1.2 separate nulls ----
    def null_dist(stat_fn, n_perm, seed=0):
        rr = np.random.default_rng(seed)
        vals = []
        for _ in range(n_perm):
            perm = rr.permutation(len(use))
            vals.append(stat_fn(perm))
        return np.array(vals)
    t1_2 = {}
    for mname, mask in modules.items():
        gidx = np.flatnonzero(mask)
        obs_k, _ = vec_partial_corr(S21[:, gidx], K0[:, gidx], S0[:, gidx])
        obs_s, _ = vec_partial_corr(S21[:, gidx], ES0[:, gidx], S0[:, gidx])
        nk = null_dist(lambda p: vec_partial_corr(S21[p][:, gidx], K0[:, gidx], S0[:, gidx])[0], args.n_perm, seed=0)
        ns = null_dist(lambda p: vec_partial_corr(S21[p][:, gidx], ES0[:, gidx], S0[:, gidx])[0], args.n_perm, seed=0)
        t1_2[mname] = dict(
            obs_kappa=obs_k, null95_kappa=float(np.percentile(nk, 95)),
            p_kappa=float((1 + (nk >= obs_k).sum()) / (args.n_perm + 1)),
            obs_epsS=obs_s, null95_epsS=float(np.percentile(ns, 95)),
            p_epsS=float((1 + (ns >= obs_s).sum()) / (args.n_perm + 1)))
        log(f'  T1.2 {mname}: kappa p={t1_2[mname]["p_kappa"]:.3f} obs={obs_k:.4f}; epsS p={t1_2[mname]["p_epsS"]:.3f} obs={obs_s:.4f}')
    result['t1_2_separate_nulls'] = t1_2

    # ---- T1.3 positive controls ----
    t1_3 = {}
    for mname, mask in modules.items():
        gidx = np.flatnonzero(mask)
        # P1: marginal corr(S0_B, S21)
        def marg(A, B):
            Ac = A - A.mean(0, keepdims=True); Bc = B - B.mean(0, keepdims=True)
            num = (Ac * Bc).sum(0); den = np.sqrt((Ac ** 2).sum(0) * (Bc ** 2).sum(0))
            return float(np.nanmean(num / np.where(den > 1e-12, den, np.nan)))
        p1 = marg(S0[:, gidx], S21[:, gidx])
        p2 = marg(ES0[:, gidx], S21[:, gidx])
        n1 = null_dist(lambda p: marg(S0[:, gidx], S21[p][:, gidx]), args.n_perm, seed=1)
        p1_p = float((1 + (n1 >= p1).sum()) / (args.n_perm + 1))
        n2 = null_dist(lambda p: marg(ES0[:, gidx], S21[p][:, gidx]), args.n_perm, seed=1)
        p2_p = float((1 + (n2 >= p2).sum()) / (args.n_perm + 1))
        t1_3[mname] = dict(P1_S0_S21=p1, P1_p=p1_p, P1_null95=float(np.percentile(n1, 95)),
                           P2_epsS0_S21=p2, P2_p=p2_p)
        log(f'  T1.3 {mname}: P1(S0~S21)={p1:.4f} p={p1_p:.4f} | P2(epsS0~S21)={p2:.4f} p={p2_p:.4f}')
    result['t1_3_positive_controls'] = t1_3

    # ---- T1.4 clone-level reliability ----
    def split3_kappa(s, u, rng):
        a1, a2, b = count_split3(s, rng)
        u1, u2, _ = count_split3(u, rng)
        z1 = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(a1).todense(), dtype=np.float32))
        z2 = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(a2).todense(), dtype=np.float32))
        eU1 = knn_residual(np.asarray(normalize_log(u1).todense(), dtype=np.float32), z1)
        eS1 = knn_residual(np.asarray(normalize_log(a1).todense(), dtype=np.float32), z1)
        eU2 = knn_residual(np.asarray(normalize_log(u2).todense(), dtype=np.float32), z2)
        eS2 = knn_residual(np.asarray(normalize_log(a2).todense(), dtype=np.float32), z2)
        k1 = eU1 - eS1 * iv_b(eU1, eU2, eS1, eS2)
        k2 = eU2 - eS2 * iv_b(eU2, eU1, eS2, eS1)
        return k1, k2, eS1, eS2, np.asarray(normalize_log(b).todense(), dtype=np.float32)

    rng4 = np.random.default_rng(2)
    k1, k2, eS1, eS2, S0b = split3_kappa(s0, u0, rng4)
    h1, h2 = count_split(s21, np.random.default_rng(2))
    S21h1 = np.asarray(normalize_log(h1).todense(), dtype=np.float32)
    S21h2 = np.asarray(normalize_log(h2).todense(), dtype=np.float32)

    def clone_corr(m1_cell, m2_cell, lab):
        K1 = clmean(m1_cell, lab, use); K2 = clmean(m2_cell, lab, use)
        K1 = K1 - K1.mean(0, keepdims=True); K2 = K2 - K2.mean(0, keepdims=True)
        num = (K1 * K2).sum(0); den = np.sqrt((K1 ** 2).sum(0) * (K2 ** 2).sum(0))
        return num / np.where(den > 1e-12, den, np.nan)

    r_k = clone_corr(k1, k2, lab0)
    r_s = clone_corr(eS1, eS2, lab0)
    r_21 = clone_corr(S21h1, S21h2, lab21)
    rel_k = 1.5 * r_k / (1 + 0.5 * r_k)
    rel_s = 1.5 * r_s / (1 + 0.5 * r_s)
    rel_21 = 2 * r_21 / (1 + r_21)

    def cell_icc(vals, lab):
        # between-clone variance fraction (cell level)
        d = {}
        for c in use:
            m = np.flatnonzero(lab == c)
            if len(m):
                d[c] = vals[m].mean(0)
        return np.nan  # placeholder; per-module below

    t1_4 = {}
    for mname, mask in modules.items():
        g = np.flatnonzero(mask)
        t1_4[mname] = dict(
            rel_kappa_median=float(np.nanmedian(rel_k[g])), rel_kappa_q25=float(np.nanpercentile(rel_k[g], 25)),
            rel_kappa_q75=float(np.nanpercentile(rel_k[g], 75)),
            rel_epsS_median=float(np.nanmedian(rel_s[g])),
            rel_S21_median=float(np.nanmedian(rel_21[g])))
        log(f'  T1.4 {mname}: rel_kappa={t1_4[mname]["rel_kappa_median"]:.3f} rel_epsS={t1_4[mname]["rel_epsS_median"]:.3f} rel_S21={t1_4[mname]["rel_S21_median"]:.3f}')
    result['t1_4_reliability'] = t1_4

    # ---- T1.5 signal injection ----
    lambdas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    R = args.inject_reps
    usable_cell = np.isin(lab0, use)
    modu = list(modules.items())
    detect_a = {l: 0 for l in lambdas}; detect_b = {l: 0 for l in lambdas}
    icc_inj = {l: l * l / (1 + l * l) for l in lambdas}
    cl_corr = {l: [] for l in lambdas}
    for rep in range(R):
        rr = np.random.default_rng(1000 + rep)
        s0a, s0b = count_split(s0, rr); u0a, u0b = count_split(u0, rr)
        za = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s0a).todense(), dtype=np.float32))
        zb = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s0b).todense(), dtype=np.float32))
        eUa = knn_residual(np.asarray(normalize_log(u0a).todense(), dtype=np.float32), za)
        eSa = knn_residual(np.asarray(normalize_log(s0a).todense(), dtype=np.float32), za)
        eUb = knn_residual(np.asarray(normalize_log(u0b).todense(), dtype=np.float32), zb)
        eSb = knn_residual(np.asarray(normalize_log(s0b).todense(), dtype=np.float32), zb)
        kap = eUa - eSa * iv_b(eUa, eUb, eSa, eSb)
        S0cell = np.asarray(normalize_log(s0b).todense(), dtype=np.float32)
        hh1, hh2 = count_split(s21, rr)
        H1 = np.asarray(normalize_log(hh1).todense(), dtype=np.float32)
        H2 = np.asarray(normalize_log(hh2).todense(), dtype=np.float32)
        S21_1 = clmean(H1, lab21, use); S21_2 = clmean(H2, lab21, use); S0m = clmean(S0cell, lab0, use)
        # target direction T = standardized residual S21_H1 ~ S0
        Cc = S0m - S0m.mean(0, keepdims=True); den = (Cc ** 2).sum(0)
        b = np.where(den > 1e-12, (Cc * (S21_1 - S21_1.mean(0, keepdims=True))).sum(0) / np.where(den > 1e-12, den, 1), 0)
        resid = (S21_1 - S21_1.mean(0, keepdims=True)) - Cc * b
        T = resid / (resid.std(0, keepdims=True) + 1e-12)
        sd_g = kap[usable_cell].std(0)
        clone_of_cell = {c: np.flatnonzero(lab0 == c) for c in use}
        # eps_S control obs (no injection) using H2
        ES = clmean(eSa, lab0, use)
        for lam in lambdas:
            kinj = kap.copy()
            if lam > 0:
                for ci, c in enumerate(use):
                    m = clone_of_cell[c]
                    kinj[m] = kinj[m] + lam * sd_g * T[ci]
            K0i = clmean(kinj, lab0, use)
            # primary statistic: ALL module
            ga = np.flatnonzero(modules['ALL'])
            obsa, _ = vec_partial_corr(S21_2[:, ga], K0i[:, ga], S0m[:, ga])
            ctrla, _ = vec_partial_corr(S21_2[:, ga], ES[:, ga], S0m[:, ga])
            rrp = np.random.default_rng(rep)
            na = np.array([vec_partial_corr(S21_2[p][:, ga], K0i[:, ga], S0m[:, ga])[0] for p in (rrp.permutation(len(use)) for _ in range(args.inject_perm))])
            pa = (1 + (na >= obsa).sum()) / (args.inject_perm + 1)
            detect_a[lam] += int(pa < 0.05)
            detect_b[lam] += int(pa < 0.05 and obsa > ctrla)
            if lam > 0:
                Kc = K0i - K0i.mean(0, keepdims=True); Tc = T - T.mean(0, keepdims=True)
                num = (Kc * Tc).sum(0); dd = np.sqrt((Kc ** 2).sum(0) * (Tc ** 2).sum(0))
                cl_corr[lam].append(float(np.nanmedian(num / np.where(dd > 1e-12, dd, np.nan))))
        if (rep + 1) % 5 == 0:
            log(f'  T1.5 rep {rep+1}/{R} done')
    t1_5 = {'lambdas': lambdas, 'R': R, 'detect_a_any': {l: detect_a[l] / R for l in lambdas},
            'detect_b_primary': {l: detect_b[l] / R for l in lambdas},
            'icc_inj': icc_inj, 'clonal_corr_median': {l: (float(np.median(cl_corr[l])) if cl_corr[l] else 0.0) for l in lambdas}}
    mde = None
    for l in lambdas:
        if l > 0 and t1_5['detect_a_any'][l] >= 0.8 and t1_5['detect_b_primary'][l] >= 0.8:
            mde = l; break
    if mde is None:
        for l in lambdas:
            if l > 0 and t1_5['detect_a_any'][l] >= 0.8:
                mde = l; break
    t1_5['MDE_lambda'] = mde
    t1_5['MDE_icc_inj'] = (mde * mde / (1 + mde * mde)) if mde else None
    t1_5['MDE_clonal_corr'] = t1_5['clonal_corr_median'].get(mde) if mde else None
    result['t1_5_injection'] = t1_5
    log(f'  T1.5 MDE lambda={mde} ICC_inj={t1_5["MDE_icc_inj"]}')

    result['runtime_sec'] = time.time() - t0
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2, default=str)[:4000])


if __name__ == '__main__':
    main()
