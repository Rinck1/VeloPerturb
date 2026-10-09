"""A2: U-innovation reliability + clone ICC (master switch).

Question: is a single cell's U deviation a repeatable signal or count noise?
If repeatable, is it shared across cells of the same clone?

Method (per dataset):
  1. Binomial count-split integer S/U (p=0.5) -> halves A,B.
  2. Each half: z = PCA-50 of log1p normalized S; E[U|z] via k=30 kNN regression;
     eps = log1p(U) - E[U|z] (no smoothing).
  3. Module scores: top-20 PCs fit on eps_A, projected onto both halves.
  4. Reliability: per-module corr(score_A, score_B) across cells, Spearman-Brown.
  5. Clone ICC (MeRLin): clones with >=2 cells; null shuffles clone labels within
     z deciles preserving within-stratum clone-size multiset.

Judgement: corrected reliability >= 0.2 AND ICC lower CI > null 95th pct -> continue.
           reliability < 0.1 -> stop all per-cell claims.
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


def log(msg):
    print(f'[A2] {msg}', flush=True)


def count_split(mat, rng):
    a = mat.astype(np.int64).copy()
    a.data = rng.binomial(a.data, 0.5)
    b = mat.astype(np.int64) - a
    a.eliminate_zeros(); b.eliminate_zeros()
    return a.tocsr(), b.tocsr()


def normalize_log(mat, target=10000.0):
    totals = np.asarray(mat.sum(1)).ravel().astype(np.float64)
    scale = np.where(totals > 0, target / np.maximum(totals, 1e-9), 0.0)
    out = mat.astype(np.float32).multiply(scale[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def select_genes(s, u, top_hvg=2000, min_frac=0.005):
    n = s.shape[0]
    expressed = np.asarray((s > 0).sum(0)).ravel() >= max(10, min_frac * n)
    logs = s.astype(np.float32).copy()
    logs.data = np.log1p(logs.data)
    mean = np.asarray(logs.mean(0)).ravel()
    sq = np.asarray(logs.power(2).mean(0)).ravel()
    var = sq - mean ** 2
    var[~expressed] = -1
    order = np.argsort(-var)
    return np.sort(order[:min(top_hvg, int(expressed.sum()))])


def knn_innovation(u_log, z, k=30):
    n = len(z)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    kk = idx.shape[1]
    rows = np.repeat(np.arange(n), kk)
    W = sparse.csr_matrix((np.full(n * kk, 1.0 / kk), (rows, idx.ravel())), shape=(n, n))
    u = np.asarray(u_log.todense(), dtype=np.float32)
    pred = W @ u
    return (u - pred).astype(np.float32)


def module_pca(eps_a, eps_b, n_components=20):
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0)
    return pca.fit_transform(eps_a), pca.transform(eps_b)


def spearman_brown(r):
    r = np.clip(r, -0.999, 0.999)
    return 2 * r / (1 + r)


def permodule_corr(sa, sb):
    sa = sa - sa.mean(0, keepdims=True)
    sb = sb - sb.mean(0, keepdims=True)
    num = (sa * sb).sum(0)
    den = np.sqrt((sa ** 2).sum(0) * (sb ** 2).sum(0)) + 1e-12
    return num / den


def icc(scores, labels):
    labels = np.asarray(labels)
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n = len(scores)
    kk = len(uniq)
    if n <= kk:
        return 0.0
    sums = np.bincount(inv, weights=scores)
    means = sums / counts
    grand = scores.mean()
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(np.sum(scores.astype(np.float64) ** 2) - np.sum(counts * means ** 2))
    ms_between = ss_between / max(kk - 1, 1)
    ms_within = ss_within / max(n - kk, 1)
    denom = ms_between + (n - kk) / max(kk - 1, 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 1e-12 else 0.0


def icc_bootstrap(scores, labels, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([np.flatnonzero(labels == g) for g in pick])
        out.append(icc(scores[idx], labels[idx]))
    return float(np.percentile(out, 2.5))


def icc_null(scores, labels, z, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    qs = np.quantile(z[:, 0], np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(z[:, 0], qs)
    labels = np.asarray(labels)
    out = []
    for _ in range(n_perm):
        perm = labels.copy()
        for d in np.unique(dec):
            md = np.flatnonzero(dec == d)
            perm[md] = labels[rng.permutation(md)]
        out.append(icc(scores, perm))
    out = np.asarray(out)
    return float(np.percentile(out, 95)), float((1 + (out >= icc(scores, labels)).sum()) / (n_perm + 1))


def run_dataset(name, s_mat, u_mat, rng, top_hvg, result, clone_labels=None):
    log(f'{name}: raw S nnz={s_mat.nnz} U nnz={u_mat.nnz}')
    tot = np.asarray((s_mat + u_mat).sum(1)).ravel()
    keep = tot >= 100
    s_mat, u_mat = s_mat[keep].tocsr(), u_mat[keep].tocsr()
    if clone_labels is not None:
        clone_labels = np.asarray(clone_labels)[keep]
    log(f'{name}: {int(keep.sum())} cells pass total>=100')
    sel = select_genes(s_mat, u_mat, top_hvg=top_hvg)
    s_sel, u_sel = s_mat[:, sel].tocsr(), u_mat[:, sel].tocsr()
    s_a, s_b = count_split(s_sel, rng)
    u_a, u_b = count_split(u_sel, rng)
    z_a = PCA(n_components=50, svd_solver='randomized', random_state=0).fit_transform(
        np.asarray(normalize_log(s_a).todense(), dtype=np.float32))
    z_b = PCA(n_components=50, svd_solver='randomized', random_state=0).fit_transform(
        np.asarray(normalize_log(s_b).todense(), dtype=np.float32))
    eps_a = knn_innovation(normalize_log(u_a), z_a, k=30)
    eps_b = knn_innovation(normalize_log(u_b), z_b, k=30)
    sa, sb = module_pca(eps_a, eps_b, n_components=20)
    r = permodule_corr(sa, sb)
    r_sb = spearman_brown(r)
    result[name] = dict(
        n_cells=int(keep.sum()), n_genes=int(len(sel)),
        module_reliability_raw=[float(x) for x in r],
        module_reliability_spearman_brown=[float(x) for x in r_sb],
        median_reliability_sb=float(np.median(r_sb)),
        max_reliability_sb=float(np.max(r_sb)),
        frac_modules_reliability_ge_0p2=float((r_sb >= 0.2).mean()),
    )
    log(f'{name}: median SB reliability={np.median(r_sb):.3f} max={np.max(r_sb):.3f}')
    if clone_labels is not None:
        valid = clone_labels >= 0
        vals = np.unique(clone_labels[valid])
        _, counts = np.unique(clone_labels[valid], return_counts=True)
        multi_vals = vals[counts >= 2]
        multi = np.isin(clone_labels, multi_vals)
        log(f'{name}: clone cells(>=2)={int(multi.sum())} clones={len(multi_vals)}')
        iccs = [icc(sa[multi, j], clone_labels[multi]) for j in range(sa.shape[1])]
        result[name]['clone_icc_all_modules'] = [float(v) for v in iccs]
        result[name]['clone_icc_median'] = float(np.median(iccs))
        best = int(np.argmax(np.abs(iccs)))
        sc = sa[multi, best]
        lab = clone_labels[multi]
        null95, pval = icc_null(sc, lab, z_a[multi], n_perm=1000, seed=0)
        z_ctrl = icc(z_a[multi, 0], lab)
        z_null95, z_pval = icc_null(z_a[multi, 0], lab, z_a[multi], n_perm=200, seed=1)
        result[name]['clone_icc'] = dict(
            best_module=int(best), observed=float(icc(sc, lab)),
            null95=null95, p_value=pval, passes=bool(icc(sc, lab) > null95),
            z_control_observed=float(z_ctrl), z_control_null95=z_null95, z_control_p=z_pval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='merlin315,renge_day4')
    ap.add_argument('--merlin-root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--top-hvg', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default='outputs/w1_a2_u_innovation_reliability.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    mr = Path(args.merlin_root)
    result = dict(tag='A2', datasets=args.datasets, top_hvg=args.top_hvg)
    wanted = set(args.datasets.split(','))

    if 'merlin315' in wanted:
        rows, genes, mats, meta = load_usa(mr / 'quant_merlin315' / 'af_quant')
        s, u = mats[0].tocsr(), mats[1].tocsr()
        pos = {bc: i for i, bc in enumerate(rows)}
        clone_map, labels = {}, np.full(len(rows), -1, dtype=np.int64)
        with (mr / 'clones' / 'SRR33960315' / 'cell_clone_assignments.csv').open() as fh:
            for row in csv.DictReader(fh):
                j = pos.get(row['cell_barcode'])
                if j is None:
                    continue
                cb = row['clone_barcode']
                if cb not in clone_map:
                    clone_map[cb] = len(clone_map)
                labels[j] = clone_map[cb]
        run_dataset('merlin315', s, u, rng, args.top_hvg, result, clone_labels=labels)

    if 'merlin310' in wanted:
        rows, genes, mats, meta = load_usa(mr / 'quant_merlin310' / 'af_quant')
        run_dataset('merlin310', mats[0].tocsr(), mats[1].tocsr(), rng, args.top_hvg, result)

    if 'renge_day4' in wanted:
        import anndata as ad
        a = ad.read_h5ad('data/renge/processed_release_v1/day4/day4.h5ad')
        s = sparse.csr_matrix(a.layers['spliced'])
        u = sparse.csr_matrix(a.layers['unspliced'])
        run_dataset('renge_day4', s, u, rng, args.top_hvg, result)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
