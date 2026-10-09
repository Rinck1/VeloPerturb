"""C1 (v2): RENGE directed-regulation quick test, corrected.

Fixes vs v1:
  * regulators are ONLY the 23 TFs (p=23), fit on single control cells (n~788),
    not 523 regulators on 26 metacells;
  * marginal regression coefficients cov(alpha_g, S_r)/var(S_r) replace
    underdetermined multivariate ridge;
  * DeltaS_r fixed to -1 (CRISPR knockdown mRNA sign is unreliable);
  * directed vs co-expression correlations are disattenuated by each predictor's
    split-half reliability.

Metric: per TF, Pearson(pred, observed effect) over top-200 DEGs (excluding r).
Directed beats co-expression in >=15/23 TFs and paired Wilcoxon p<0.05.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse
from scipy.stats import wilcoxon

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')

TFS = ['ETS2', 'ETV4', 'FOXH1', 'ID1', 'JARID2', 'LIN28A', 'MYC', 'MYCN', 'NANOG', 'NR5A2',
       'PDLIM1', 'POU5F1', 'PRDM14', 'RUNX1T1', 'SOX2', 'TRIM24', 'TRIM25', 'VENTX', 'ZIC2',
       'ZIC3', 'ZNF398', 'ZNF649', 'ZNF90']
CONTROLS = ['CTRL', 'AAVS1']


def log(m):
    print(f'[C1] {m}', flush=True)


def normalize(mat, target=10000.0):
    tot = np.asarray(mat.sum(1)).ravel().astype(np.float64)
    sc = np.where(tot > 0, target / np.maximum(tot, 1e-9), 0.0)
    out = mat.astype(np.float32).multiply(sc[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def marginal_coef(alpha, s):
    """alpha, s: (n_cells, n_genes). Returns cov(alpha_g, s_r)/var(s_r) per (g,r)."""
    ac = alpha - alpha.mean(0, keepdims=True)
    sc = s - s.mean(0, keepdims=True)
    denom = (sc ** 2).sum(0) + 1e-12
    return (ac.T @ sc) / denom[None, :]


def spear_brown(r):
    r = np.clip(r, -0.999, 0.999)
    return 2 * r / (1 + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-hvg', type=int, default=2000)
    ap.add_argument('--top-deg', type=int, default=200)
    ap.add_argument('--output', default='outputs/w1_c1_directed_regulation.json')
    args = ap.parse_args()

    d4 = ad.read_h5ad('data/renge/processed_release_v1/day4/day4.h5ad')
    d5 = ad.read_h5ad('data/renge/processed_release_v1/day5/day5.h5ad')
    genes = np.array([str(g) for g in d4.var['gene_symbol']])
    assert list(genes) == [str(g) for g in d5.var['gene_symbol']]

    def ctrl_mask(a):
        return a.obs['condition'].isin(CONTROLS).to_numpy()

    c4s = sparse.csr_matrix(d4.layers['spliced'])[ctrl_mask(d4)]
    c4u = sparse.csr_matrix(d4.layers['unspliced'])[ctrl_mask(d4)]
    c5s = sparse.csr_matrix(d5.layers['spliced'])[ctrl_mask(d5)]
    c5u = sparse.csr_matrix(d5.layers['unspliced'])[ctrl_mask(d5)]
    cs = sparse.vstack([c4s, c5s]).tocsr()
    cu = sparse.vstack([c4u, c5u]).tocsr()
    n = cs.shape[0]
    log(f'control cells: {int(ctrl_mask(d4).sum())}+{int(ctrl_mask(d5).sum())}={n}')

    # target genes = top HVG union TFs
    logs = normalize(cs)
    dense = np.asarray(logs.todense(), dtype=np.float32)
    var = dense.var(0)
    expressed = np.asarray((cs > 0).sum(0)).ravel() >= 10
    var[~expressed] = -1
    hvg = list(np.argsort(-var)[:args.n_hvg])
    tf_idx = {t: int(np.flatnonzero(genes == t)[0]) for t in TFS if (genes == t).any()}
    target = np.array(sorted(set(hvg) | set(tf_idx.values())))
    reg_pos = {t: int(np.flatnonzero(target == i)[0]) for t, i in tf_idx.items()}
    log(f'target genes={len(target)} TFs found={len(tf_idx)}')

    Sn = np.asarray(normalize(cs)[:, target].todense(), dtype=np.float32)
    Un = np.asarray(normalize(cu)[:, target].todense(), dtype=np.float32)
    reg_target_cols = np.array([reg_pos[t] for t in tf_idx])
    tf_list = list(tf_idx)
    reg_col_of_tf = {t: k for k, t in enumerate(tf_list)}
    S_reg = Sn[:, reg_target_cols]

    W = marginal_coef(Un, S_reg)      # (ngenes, nreg): alpha=U predicts
    C = marginal_coef(Sn, S_reg)      # (ngenes, nreg): S co-expression

    # split-half reliability of each coefficient column
    rng = np.random.default_rng(0)
    perm = rng.permutation(n); h1, h2 = perm[:n // 2], perm[n // 2:]
    W1 = marginal_coef(Un[h1], S_reg[h1]); W2 = marginal_coef(Un[h2], S_reg[h2])
    C1 = marginal_coef(Sn[h1], S_reg[h1]); C2 = marginal_coef(Sn[h2], S_reg[h2])
    def col_rel(A, B):
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
        num = (A * B).sum(0); den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)) + 1e-12
        return spear_brown(num / den)
    rel_W = col_rel(W1, W2)
    rel_C = col_rel(C1, C2)

    d5n = np.asarray(normalize(sparse.csr_matrix(d5.layers['spliced'])).todense(), dtype=np.float32)
    ctrl5 = ctrl_mask(d5)
    ctrl_mean = d5n[ctrl5].mean(0)

    rows = []
    for r, ti in tf_idx.items():
        if r not in reg_pos:
            continue
        ko = (d5.obs['condition'] == r).to_numpy()
        if int(ko.sum()) < 20:
            continue
        delta = d5n[ko].mean(0) - ctrl_mean
        true = delta[target]
        j = reg_col_of_tf[r]
        pred_dir = -W[:, j]
        pred_cov = -C[:, j]
        top = np.argsort(-np.abs(true))[:args.top_deg]
        top = top[target[top] != ti]
        def pearson(p):
            x, y = p[top], true[top]
            x = x - x.mean(); y = y - y.mean()
            den = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-12
            return float((x * y).sum() / den)
        raw_d, raw_c = pearson(pred_dir), pearson(pred_cov)
        corr_d = raw_d / np.sqrt(max(rel_W[j], 1e-3))
        corr_c = raw_c / np.sqrt(max(rel_C[j], 1e-3))
        rows.append(dict(tf=r, n_ko=int(ko.sum()), rel_W=float(rel_W[j]), rel_C=float(rel_C[j]),
                         raw_directed=raw_d, raw_coexpr=raw_c,
                         pearson_directed=float(corr_d), pearson_coexpr=float(corr_c)))
        log(f'  {r:8s} n={int(ko.sum()):4d} relW={rel_W[j]:+.2f} relC={rel_C[j]:+.2f} '
            f'dir={corr_d:+.3f} cov={corr_c:+.3f}')
    dd = np.array([x['pearson_directed'] for x in rows]); cc = np.array([x['pearson_coexpr'] for x in rows])
    wins = int((dd > cc).sum())
    try:
        p = float(wilcoxon(dd, cc).pvalue)
    except Exception:
        p = 1.0
    result = dict(tag='C1v2', n_tfs=len(rows), directed_wins=wins, wilcoxon_p=p,
                  passes=bool(wins >= 15 and p < 0.05),
                  mean_directed=float(dd.mean()) if len(dd) else None,
                  mean_coexpr=float(cc.mean()) if len(cc) else None, rows=rows)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
