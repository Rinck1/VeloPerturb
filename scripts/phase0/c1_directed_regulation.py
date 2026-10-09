"""C1: RENGE directed-regulation quick test.

Only control cells (CTRL/AAVS1, day4+day5). Build metacells (~30 cells) from S.
Transcription-rate proxy alpha_g = metacell U. Fit, per target gene g, a ridge
    alpha_g ~ S_r  (directed matrix W)   and   S_g ~ S_r (co-expression C),
regulators r = 23 TFs + top-500 HVGs.

For each TF r: observed effect Delta_r = log pseudo-bulk(KO r day5) - control day5.
Predictions W[:,r]*DeltaS_r and C[:,r]*DeltaS_r. Metric: Pearson over top-200 DEGs
(excluding r). Directed beats co-expression in >=15/23 TFs and paired Wilcoxon
p<0.05 -> directional signal.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-hvg', type=int, default=2000)
    ap.add_argument('--n-reg-hvg', type=int, default=500)
    ap.add_argument('--n-metacells', type=int, default=30)
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--top-deg', type=int, default=200)
    ap.add_argument('--output', default='outputs/w1_c1_directed_regulation.json')
    args = ap.parse_args()

    d4 = ad.read_h5ad('data/renge/processed_release_v1/day4/day4.h5ad')
    d5 = ad.read_h5ad('data/renge/processed_release_v1/day5/day5.h5ad')
    genes = np.array([str(g) for g in d4.var['gene_symbol']])
    assert list(genes) == [str(g) for g in d5.var['gene_symbol']]

    def ctrl_mask(a):
        return a.obs['condition'].isin(CONTROLS).to_numpy()

    def sel(a, mask):
        return sparse.csr_matrix(a.layers['spliced'])[mask], sparse.csr_matrix(a.layers['unspliced'])[mask]

    c4s, c4u = sel(d4, ctrl_mask(d4))
    c5s, c5u = sel(d5, ctrl_mask(d5))
    cs = sparse.vstack([c4s, c5s]).tocsr()
    cu = sparse.vstack([c4u, c5u]).tocsr()
    log(f'control cells: day4={c4s.shape[0]} day5={c5s.shape[0]} total={cs.shape[0]}')

    # HVG on control S
    logs = normalize(cs)
    dense = np.asarray(logs.todense(), dtype=np.float32)
    var = dense.var(0)
    expressed = np.asarray((cs > 0).sum(0)).ravel() >= 10
    var[~expressed] = -1
    hvg_idx = np.sort(np.argsort(-var)[:args.n_hvg])
    hvg_names = genes[hvg_idx]

    # regulators = 23 TF + top reg HVG (excluding those already TF)
    tf_idx = [int(np.flatnonzero(genes == t)[0]) for t in TFS if (genes == t).any()]
    tf_names = [genes[i] for i in tf_idx]
    reg_extra = [i for i in hvg_idx if genes[i] not in set(tf_names)][:args.n_reg_hvg]
    reg_idx = np.array(tf_idx + reg_extra)
    reg_names = genes[reg_idx]
    log(f'TF found={len(tf_idx)}/{len(TFS)} regulators={len(reg_idx)}')

    # metacells
    Sn = np.asarray(normalize(cs).todense(), dtype=np.float32)
    Un = np.asarray(normalize(cu).todense(), dtype=np.float32)
    z = PCA(n_components=30, svd_solver='randomized', random_state=0).fit_transform(Sn)
    km = KMeans(n_clusters=min(args.n_metacells, len(z) // 5), n_init=5, random_state=0).fit(z)
    labels = km.labels_
    Smc = np.vstack([Sn[labels == k].mean(0) for k in range(km.n_clusters)])
    Umc = np.vstack([Un[labels == k].mean(0) for k in range(km.n_clusters)])
    log(f'metacells={km.n_clusters} shape Smc={Smc.shape}')

    Sreg = Smc[:, reg_idx]               # (mc, nreg)
    W = np.zeros((args.n_hvg, len(reg_idx)), dtype=np.float32)
    C = np.zeros((args.n_hvg, len(reg_idx)), dtype=np.float32)
    for j, g in enumerate(hvg_idx):
        W[j] = Ridge(alpha=args.alpha).fit(Sreg, Umc[:, g]).coef_
        C[j] = Ridge(alpha=args.alpha).fit(Sreg, Smc[:, g]).coef_
    log('W/C fitted')

    # observed day5 KO vs control pseudo-bulk (all genes, normalized log)
    d5n = np.asarray(normalize(sparse.csr_matrix(d5.layers['spliced'])).todense(), dtype=np.float32)
    d5un = np.asarray(normalize(sparse.csr_matrix(d5.layers['unspliced'])).todense(), dtype=np.float32)
    ctrl5 = (d5.obs['condition'].isin(CONTROLS)).to_numpy()
    ctrl_mean = d5n[ctrl5].mean(0)
    reg_col = {name: k for k, name in enumerate(reg_names)}

    rows = []
    for r in tf_names:
        ko = (d5.obs['condition'] == r).to_numpy()
        if int(ko.sum()) < 20 or r not in reg_col:
            continue
        delta = d5n[ko].mean(0) - ctrl_mean
        k = reg_col[r]
        dS_r = float(delta[int(np.flatnonzero(genes == r)[0])])
        pred_dir = W[:, k] * dS_r
        pred_cov = C[:, k] * dS_r
        true_on_hvg = delta[hvg_idx]
        top = np.argsort(-np.abs(true_on_hvg))[:args.top_deg]
        top = top[hvg_names[top] != r]
        def pearson(p):
            x, y = p[top], true_on_hvg[top]
            x = x - x.mean(); y = y - y.mean()
            den = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-12
            return float((x * y).sum() / den)
        rows.append(dict(tf=r, deltaS=dS_r, n_ko=int(ko.sum()),
                         pearson_directed=pearson(pred_dir), pearson_coexpr=pearson(pred_cov)))
        log(f'  {r:8s} n={int(ko.sum()):4d} dS={dS_r:+.2f} dir={rows[-1]["pearson_directed"]:+.3f} cov={rows[-1]["pearson_coexpr"]:+.3f}')

    if len(rows) >= 5:
        dd = np.array([x['pearson_directed'] for x in rows])
        cc = np.array([x['pearson_coexpr'] for x in rows])
        wins = int((dd > cc).sum())
        try:
            stat, p = wilcoxon(dd, cc)
            p = float(p)
        except Exception:
            p = 1.0
    else:
        wins, p = 0, 1.0
    result = dict(tag='C1', n_tfs=len(rows), directed_wins=wins,
                  wilcoxon_p=p, passes=bool(wins >= 15 and p < 0.05),
                  mean_directed=float(np.mean([x['pearson_directed'] for x in rows])) if rows else None,
                  mean_coexpr=float(np.mean([x['pearson_coexpr'] for x in rows])) if rows else None,
                  rows=rows)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
