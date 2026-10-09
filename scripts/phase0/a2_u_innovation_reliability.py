"""A2 (v2): U-innovation reliability + clone ICC, with reviewer controls.

Fixes vs v1:
  * clone-ICC null uses the max |ICC| over all modules per permutation
    (selection-bias-corrected), not a single pre-picked module;
  * S-residual control: same pipeline on log1p(S) predicted from a z' built on
    the other half of genes -> if ICC(eps_S) ~ ICC(eps_U), U is not special;
  * technical-covariate control: regress U/(U+S) and log total UMI out of module
    scores, then recompute reliability and ICC;
  * chromosome proxy: top-loading gene indices of the best module are tested for
    genomic clustering (copy-number signature).

Premise: count-split removes only Poisson noise; repeatable != dynamical.
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
    print(f'[A2] {m}', flush=True)


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


def select_genes(s, top_hvg=2000, min_frac=0.005):
    n = s.shape[0]
    expressed = np.asarray((s > 0).sum(0)).ravel() >= max(10, min_frac * n)
    logs = s.astype(np.float32).copy(); logs.data = np.log1p(logs.data)
    mean = np.asarray(logs.mean(0)).ravel()
    var = np.asarray(logs.power(2).mean(0)).ravel() - mean ** 2
    var[~expressed] = -1
    return np.sort(np.argsort(-var)[:min(top_hvg, int(expressed.sum()))])


def knn_residual(target_log, z, k=30):
    n = len(z)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    kk = idx.shape[1]
    rows = np.repeat(np.arange(n), kk)
    W = sparse.csr_matrix((np.full(n * kk, 1.0 / kk), (rows, idx.ravel())), shape=(n, n))
    return (target_log - W @ target_log).astype(np.float32)


def spearman_brown(r):
    r = np.clip(r, -0.999, 0.999)
    return 2 * r / (1 + r)


def permodule_corr(sa, sb):
    sa = sa - sa.mean(0, keepdims=True); sb = sb - sb.mean(0, keepdims=True)
    num = (sa * sb).sum(0)
    den = np.sqrt((sa ** 2).sum(0) * (sb ** 2).sum(0)) + 1e-12
    return num / den


def icc(scores, labels):
    labels = np.asarray(labels)
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n, kk = len(scores), len(uniq)
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


def icc_maxstat_null(scores_all, labels, z, n_perm=1000, seed=0):
    """scores_all: (n_cells, n_modules). Null distribution of max|ICC|."""
    rng = np.random.default_rng(seed)
    qs = np.quantile(z[:, 0], np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(z[:, 0], qs)
    labels = np.asarray(labels)
    obs = np.array([icc(scores_all[:, j], labels) for j in range(scores_all.shape[1])])
    obs_max = float(np.max(np.abs(obs)))
    null_max = []
    for _ in range(n_perm):
        perm = labels.copy()
        for d in np.unique(dec):
            md = np.flatnonzero(dec == d)
            perm[md] = labels[rng.permutation(md)]
        vals = [abs(icc(scores_all[:, j], perm)) for j in range(scores_all.shape[1])]
        null_max.append(max(vals))
    null_max = np.asarray(null_max)
    return dict(obs_per_module=obs.tolist(), obs_max=obs_max,
                null95=float(np.percentile(null_max, 95)),
                p_value=float((1 + (null_max >= obs_max).sum()) / (n_perm + 1)),
                passes=bool(obs_max > np.percentile(null_max, 95)))


def run_dataset(name, s_mat, u_mat, rng, top_hvg, result, clone_labels=None):
    log(f'{name}: S nnz={s_mat.nnz} U nnz={u_mat.nnz}')
    tot = np.asarray((s_mat + u_mat).sum(1)).ravel()
    keep = tot >= 100
    s_mat, u_mat = s_mat[keep].tocsr(), u_mat[keep].tocsr()
    if clone_labels is not None:
        clone_labels = np.asarray(clone_labels)[keep]
    log(f'{name}: {int(keep.sum())} cells >=100')
    sel = select_genes(s_mat, top_hvg=top_hvg)
    s_sel, u_sel = s_mat[:, sel].tocsr(), u_mat[:, sel].tocsr()
    s_a, s_b = count_split(s_sel, rng)
    u_a, u_b = count_split(u_sel, rng)
    sna, snb = normalize_log(s_a), normalize_log(s_b)
    una, unb = normalize_log(u_a), normalize_log(u_b)
    z_a = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(sna.todense(), dtype=np.float32))
    z_b = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(snb.todense(), dtype=np.float32))
    eps_a = knn_residual(np.asarray(una.todense(), dtype=np.float32), z_a)
    eps_b = knn_residual(np.asarray(unb.todense(), dtype=np.float32), z_b)
    pca = PCA(20, svd_solver='randomized', random_state=0).fit(eps_a)
    sa, sb = pca.transform(eps_a), pca.transform(eps_b)
    r_sb = spearman_brown(permodule_corr(sa, sb))
    result[name] = dict(n_cells=int(keep.sum()), n_genes=int(len(sel)),
                        median_reliability_sb=float(np.median(r_sb)), max_reliability_sb=float(np.max(r_sb)),
                        frac_modules_reliability_ge_0p2=float((r_sb >= 0.2).mean()))

    # ---- S-residual control (gene half split; z' from other half) ----
    nt = len(sel)
    idxA, idxB = np.arange(0, nt, 2), np.arange(1, nt, 2)
    sf = np.asarray(normalize_log(s_sel).todense(), dtype=np.float32)
    uf = np.asarray(normalize_log(u_sel).todense(), dtype=np.float32)
    zprime = PCA(50, svd_solver='randomized', random_state=0).fit_transform(sf[:, idxB])
    epsS = knn_residual(sf[:, idxA], zprime)
    epsU = knn_residual(uf[:, idxA], zprime)
    pcaS = PCA(20, svd_solver='randomized', random_state=0).fit(epsS)
    pcaU = PCA(20, svd_solver='randomized', random_state=0).fit(epsU)
    result[name]['S_residual_control'] = dict(
        note='eps_S vs eps_U, both on gene-half A with z-prime from gene-half B',
        epsU_module_var=[float(v) for v in pcaU.explained_variance_],
        epsS_module_var=[float(v) for v in pcaS.explained_variance_])
    if clone_labels is not None:
        valid = clone_labels >= 0
        vals = np.unique(clone_labels[valid]); _, counts = np.unique(clone_labels[valid], return_counts=True)
        multi = np.isin(clone_labels, vals[counts >= 2])
        lab = clone_labels[multi]
        su = pcaU.transform(epsU)[multi]
        ss = pcaS.transform(epsS)[multi]
        icu = np.array([icc(su[:, j], lab) for j in range(su.shape[1])])
        ics = np.array([icc(ss[:, j], lab) for j in range(ss.shape[1])])
        result[name]['S_residual_control'].update(
            icc_epsU_median=float(np.median(icu)), icc_epsS_median=float(np.median(ics)),
            icc_epsU_max=float(np.max(np.abs(icu))), icc_epsS_max=float(np.max(np.abs(ics))),
            U_not_special=bool(abs(np.max(np.abs(icu)) - np.max(np.abs(ics))) < 0.02))

    if clone_labels is not None:
        valid = clone_labels >= 0
        vals = np.unique(clone_labels[valid]); _, counts = np.unique(clone_labels[valid], return_counts=True)
        multi = np.isin(clone_labels, vals[counts >= 2])
        lab = clone_labels[multi]
        sa_m = sa[multi]
        log(f'{name}: clone multi cells={int(multi.sum())} clones={len(vals[counts>=2])}')

        # main ICC with max-statistic null
        ms = icc_maxstat_null(sa_m, lab, z_a[multi], n_perm=1000, seed=0)
        result[name]['clone_icc_maxstat'] = ms
        log(f'{name}: ICC max-stat obs={ms["obs_max"]:.4f} null95={ms["null95"]:.4f} p={ms["p_value"]:.4f} pass={ms["passes"]}')

        # technical-covariate control: regress U/(U+S) and log total out of module scores
        ufrac = np.asarray(u_sel.sum(1)).ravel() / np.maximum(np.asarray((u_sel + s_sel).sum(1)).ravel(), 1)
        ltot = np.log1p(np.asarray((u_sel + s_sel).sum(1)).ravel())
        Xc = np.column_stack([ufrac, ltot, np.ones(len(ufrac))])[multi]
        proj = Xc @ np.linalg.lstsq(Xc, sa_m, rcond=None)[0]
        sa_corr = sa_m - proj
        ms_corr = icc_maxstat_null(sa_corr, lab, z_a[multi], n_perm=300, seed=0)
        result[name]['clone_icc_after_covariates'] = ms_corr
        log(f'{name}: ICC after covariates obs={ms_corr["obs_max"]:.4f} null95={ms_corr["null95"]:.4f} pass={ms_corr["passes"]}')

        # chromosome proxy: top-loading gene indices of best module
        best = int(np.argmax(np.abs(ms['obs_per_module'])))
        load = pca.components_[best]
        top_idx = np.argsort(-np.abs(load))[:100]
        span = int(top_idx.max() - top_idx.min())
        expected_span = 100 / len(sel) * len(sel)
        result[name]['best_module_index_clustering'] = dict(
            best_module=best, top_index_min=int(top_idx.min()), top_index_max=int(top_idx.max()),
            span=int(span), expected_span_uniform=float(len(sel) * (1 - 1 / 100)),
            clustered=bool(span < 0.3 * len(sel)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='merlin315,merlin310,renge_day4')
    ap.add_argument('--merlin-root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--top-hvg', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default='outputs/w1_a2_u_innovation_reliability.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    mr = Path(args.merlin_root)
    result = dict(tag='A2', datasets=args.datasets, top_hvg=args.top_hvg)
    wanted = args.datasets.split(',')
    if 'merlin315' in wanted:
        rows, genes, mats, meta = load_usa(mr / 'quant_merlin315' / 'af_quant')
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
        run_dataset('merlin315', mats[0].tocsr(), mats[1].tocsr(), rng, args.top_hvg, result, clone_labels=labels)
    if 'merlin310' in wanted:
        rows, genes, mats, meta = load_usa(mr / 'quant_merlin310' / 'af_quant')
        run_dataset('merlin310', mats[0].tocsr(), mats[1].tocsr(), rng, args.top_hvg, result)
    if 'renge_day4' in wanted:
        import anndata as ad
        a = ad.read_h5ad('data/renge/processed_release_v1/day4/day4.h5ad')
        run_dataset('renge_day4', sparse.csr_matrix(a.layers['spliced']), sparse.csr_matrix(a.layers['unspliced']),
                    rng, args.top_hvg, result)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
