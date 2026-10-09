"""Lead-lag test (corrected): does Day0 unspliced innovation predict Day21 spliced
level, clone by clone, after removing the same-day coupling?

Design (per Rinck):
  * count-split Day0 (310) into halves A/B; kappa0 from half A, S0 from half B;
  * Day21 S21 from 308;
  * per gene, partial correlation of S21 with kappa0 controlling for S0 (across
    clones), aggregated per module;
  * null permutes ONLY the Day21 clone rows (S0 stays bound to kappa0);
  * controls: replace kappa0 by eps_S(A); report corr(kappa0, -S0);
  * only clones with >= min_cells Day0 cells.
Positive only if significant AND clearly stronger than the eps_S control.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.preprocess import load_usa
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def log(m):
    print(f'[LL] {m}', flush=True)


def count_split(mat, rng):
    a = mat.astype(np.int64).copy(); a.data = rng.binomial(a.data, 0.5)
    b = mat.astype(np.int64) - a
    a.eliminate_zeros(); b.eliminate_zeros()
    return a.tocsr(), b.tocsr()


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


def load_clones(path):
    m = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m[r['cell_barcode']] = r['clone_barcode']
    return m


def clone_means(mat, labels, keep_ids):
    """Return dict clone -> mean row for clones in keep_ids."""
    out = {}
    for c in keep_ids:
        m = np.flatnonzero(labels == c)
        if len(m):
            out[c] = np.asarray(mat[m].todense()).mean(0) if sparse.issparse(mat[m]) else mat[m].mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--day0', default='310')
    ap.add_argument('--day21', default='308')
    ap.add_argument('--top-hvg', type=int, default=2000)
    ap.add_argument('--min-cells', type=int, default=3)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--output', default='outputs/w1_leadlag_meclin.json')
    args = ap.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(0)

    def load_run(tag):
        rows, genes, mats, meta = load_usa(root / f'quant_merlin{tag}' / 'af_quant')
        return rows, genes, mats[0].tocsr(), mats[1].tocsr()

    r0, genes0, s0, u0 = load_run(args.day0)
    r21, genes21, s21, u21 = load_run(args.day21)
    assert genes0 == genes21
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
    log(f'selected {len(sel)} genes')

    s0 = s0[:, sel].tocsr(); u0 = u0[:, sel].tocsr(); s21 = s21[:, sel].tocsr()
    # drop cells with too few counts
    keep0 = np.asarray((s0 + u0).sum(1)).ravel() >= 100
    keep21 = np.asarray(s21.sum(1)).ravel() >= 50
    s0, u0, r0 = s0[keep0], u0[keep0], np.array(r0)[keep0]
    s21, r21 = s21[keep21], np.array(r21)[keep21]

    cmap0 = load_clones(root / f'clones/SRR33960{args.day0}' / 'cell_clone_assignments.csv')
    cmap21 = load_clones(root / f'clones/SRR33960{args.day21}' / 'cell_clone_assignments.csv')
    lab0 = np.array([cmap0.get(b, '') for b in r0], dtype=object)
    lab21 = np.array([cmap21.get(b, '') for b in r21], dtype=object)

    # Day0 count-split; kappa0 from half A, S0 from half B
    s0a, s0b = count_split(s0, rng); u0a, u0b = count_split(u0, rng)
    za = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s0a).todense(), dtype=np.float32))
    zb = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s0b).todense(), dtype=np.float32))
    eUa = knn_residual(np.asarray(normalize_log(u0a).todense(), dtype=np.float32), za)
    eSa = knn_residual(np.asarray(normalize_log(s0a).todense(), dtype=np.float32), za)
    eUb = knn_residual(np.asarray(normalize_log(u0b).todense(), dtype=np.float32), zb)
    eSb = knn_residual(np.asarray(normalize_log(s0b).todense(), dtype=np.float32), zb)
    # IV b (cross-half), kappa0 and epsS control from half A
    num = (eUb * eSa).mean(0) - eUb.mean(0) * eSa.mean(0)
    den = (eSb * eSa).mean(0) - eSb.mean(0) * eSa.mean(0)
    b_iv = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    kappa0_cell = eUa - eSa * b_iv
    S0_cell = np.asarray(normalize_log(s0b).todense(), dtype=np.float32)
    epsS0_cell = eSa
    S21_cell = np.asarray(normalize_log(s21).todense(), dtype=np.float32)
    log(f'b_iv_med={np.median(b_iv):.4f}')

    # clone-level means; only shared clones with >= min_cells on both days
    shared = set(cmap0.values()) & set(cmap21.values())
    def sel_clones(lab, minc):
        vals, counts = np.unique(lab[lab != ''], return_counts=True)
        return set(vals[counts >= minc])
    c0 = sel_clones(lab0, args.min_cells); c21 = sel_clones(lab21, 1)
    use = sorted(shared & c0 & c21)
    log(f'shared={len(shared)} usable(>= {args.min_cells} day0 cells)={len(use)}')

    def clmean(cell, lab):
        out = {}
        for c in use:
            m = np.flatnonzero(lab == c)
            if len(m):
                out[c] = cell[m].mean(0)
        return np.vstack([out[c] for c in use])
    K0 = clmean(kappa0_cell, lab0)      # clones x genes (Day0, half A)
    ES0 = clmean(epsS0_cell, lab0)      # eps_S half A control
    S0 = clmean(S0_cell, lab0)          # S0 half B
    S21 = clmean(S21_cell, lab21)       # Day21

    # modules
    modules = {'ALL': np.ones(len(sel), bool)}
    progs = json.loads((root / 'merlin_programs.json').read_text())['signatures']
    sym2idx = {s: i for i, s in enumerate(symbols[sel])}
    for k, p in progs.items():
        gs = [str(g).replace('*', '').strip() for g in (p.get('genes', p) if isinstance(p, dict) else p)]
        idx = np.array([sym2idx[g] for g in gs if g in sym2idx])
        if len(idx) >= 5:
            modules[k] = np.isin(np.arange(len(sel)), idx)

    def partial_corr_stat(Y, X, C):
        """mean over genes of partial corr(Y, X | C), across clones."""
        vals = []
        for g in range(Y.shape[1]):
            y, x, c = Y[:, g], X[:, g], C[:, g]
            if np.std(y) < 1e-9 or np.std(x) < 1e-9:
                continue
            # residualize y and x on c
            def resid(a):
                cc = c - c.mean()
                den = (cc ** 2).sum()
                if den < 1e-12:
                    return a - a.mean()
                return a - a.mean() - (cc * (a - a.mean())).sum() / den * cc
            ry, rx = resid(y), resid(x)
            if np.std(ry) < 1e-9 or np.std(rx) < 1e-9:
                continue
            vals.append(float(np.corrcoef(ry, rx)[0, 1]))
        return (float(np.mean(vals)) if vals else 0.0), len(vals)

    out = {}
    for mname, mask in modules.items():
        if mask.sum() < 5:
            continue
        gidx = np.flatnonzero(mask)
        obs, ng = partial_corr_stat(S21[:, gidx], K0[:, gidx], S0[:, gidx])
        ctrl, _ = partial_corr_stat(S21[:, gidx], ES0[:, gidx], S0[:, gidx])
        coupling = float(np.mean([np.corrcoef(K0[:, g], -S0[:, g])[0, 1]
                                  for g in gidx if np.std(K0[:, g]) > 1e-9 and np.std(S0[:, g]) > 1e-9]))
        rng2 = np.random.default_rng(0)
        null = []
        for _ in range(args.n_perm):
            perm = rng2.permutation(len(use))
            null.append(partial_corr_stat(S21[perm][:, gidx], K0[:, gidx], S0[:, gidx])[0])
        null = np.array(null)
        out[mname] = dict(n_genes=ng, n_clones=len(use),
                          obs_partial_corr=obs, epsS_control=ctrl,
                          corr_kappa_negS0=coupling,
                          null95=float(np.percentile(null, 95)),
                          p_value=float((1 + (null >= obs).sum()) / (args.n_perm + 1)),
                          passes=bool(obs > np.percentile(null, 95) and obs > ctrl))
        log(f'  {mname}: n={ng} obs={obs:+.4f} epsS_ctrl={ctrl:+.4f} corr(K,-S0)={coupling:+.3f} '
            f'null95={out[mname]["null95"]:+.4f} p={out[mname]["p_value"]:.3f} pass={out[mname]["passes"]}')
    result = dict(tag='LEADLAG_v2', day0=args.day0, day21=args.day21, min_cells=args.min_cells,
                  n_clones=len(use), modules=out)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
