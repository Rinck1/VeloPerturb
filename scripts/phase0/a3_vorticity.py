"""A3: vorticity eta (does the velocity field carry non-equilibrium circulation?).

For each dataset (pancreas, bonemarrow, dentate, MeRLin315):
  1. kinetic velocity v_kin = U - gamma_g*S (steady-state; substituted for
     scVelo-stochastic, which is not installed) on raw normalized counts;
  2. z = PCA-20 of log1p normalized S;
  3. per subsampled cell, k=100 neighbourhood local linear map J (v ~ J(z-zbar)),
     neighbourhood covariance Sigma, M = J Sigma, eta = ||M-M^T||_F/(2||M||_F);
  4. dataset eta = median over cells.

Nulls (max of 95th percentiles): gamma random rescale [0.5,2]; mean-shift pseudo
velocity; binomial count-split two-half eta difference.
Judgement: eta > null 95th pct for >=2 datasets.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')


def log(m):
    print(f'[A3] {m}', flush=True)


def normalize_log(mat, target=10000.0):
    totals = np.asarray(mat.sum(1)).ravel().astype(np.float64)
    scale = np.where(totals > 0, target / np.maximum(totals, 1e-9), 0.0)
    out = mat.astype(np.float32).multiply(scale[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def select_genes(s, top=2000, min_frac=0.005):
    n = s.shape[0]
    expressed = np.asarray((s > 0).sum(0)).ravel() >= max(10, min_frac * n)
    logs = s.astype(np.float32).copy(); logs.data = np.log1p(logs.data)
    mean = np.asarray(logs.mean(0)).ravel()
    var = np.asarray(logs.power(2).mean(0)).ravel() - mean ** 2
    var[~expressed] = -1
    return np.sort(np.argsort(-var)[:min(top, int(expressed.sum()))])


def fit_gamma(s, u):
    ng = s.shape[1]
    gamma = np.zeros(ng)
    qlo = np.quantile(s, 0.05, axis=0)
    qhi = np.quantile(s, 0.95, axis=0)
    for g in range(ng):
        tail = (s[:, g] <= qlo[g]) | (s[:, g] >= qhi[g])
        if tail.sum() < 10:
            continue
        ss, uu = s[tail, g], u[tail, g]
        denom = float((ss * ss).sum())
        if denom > 1e-12:
            gamma[g] = float((ss * uu).sum() / denom)
    return gamma


def local_curl(z, v, k=100, n_cells=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(z)
    cells = rng.choice(n, min(n_cells, n), replace=False)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(z)
    nb_all = nn.kneighbors(z[cells], return_distance=False)[:, 1:]
    etas = []
    for nb in nb_all:
        Z = z[nb]; V = v[nb]
        Zc = Z - Z.mean(0)
        Jt, *_ = np.linalg.lstsq(Zc, V, rcond=None)
        J = Jt.T
        Sig = Zc.T @ Zc / len(nb)
        M = J @ Sig
        etas.append(np.linalg.norm(M - M.T) / (2 * np.linalg.norm(M) + 1e-12))
    return float(np.median(etas))


def analyze(name, s, u, out, n_cells, seed=0):
    rng = np.random.default_rng(seed)
    tot = np.asarray((s + u).sum(1)).ravel()
    keep = tot >= 100
    s, u = s[keep].tocsr(), u[keep].tocsr()
    sel = select_genes(s)
    s, u = s[:, sel].tocsr(), u[:, sel].tocsr()
    sn, un = normalize_log(s), normalize_log(u)
    pca = PCA(n_components=20, svd_solver='randomized', random_state=0)
    z = pca.fit_transform(np.asarray(sn.todense(), dtype=np.float32))
    gamma = fit_gamma(np.asarray(sn.todense(), dtype=np.float32), np.asarray(un.todense(), dtype=np.float32))
    sd = np.asarray(sn.todense(), dtype=np.float32); ud = np.asarray(un.todense(), dtype=np.float32)
    vgene = np.where(gamma > 0, ud - sd * gamma, 0.0)
    v = vgene @ pca.components_.T
    eta = local_curl(z, v, n_cells=n_cells, seed=seed)

    # null 1: gamma random rescale
    rescale = []
    for r in range(20):
        f = rng.uniform(0.5, 2.0, size=len(gamma))
        vr = (np.where(gamma > 0, ud - sd * (gamma * f), 0.0)) @ pca.components_.T
        rescale.append(local_curl(z, vr, n_cells=n_cells, seed=seed + r))
    # null 2: mean-shift pseudo velocity
    nn = NearestNeighbors(n_neighbors=min(100, len(z))).fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    vms = z[idx].mean(1) - z
    ms = local_curl(z, vms, n_cells=n_cells, seed=seed)
    # null 3: count-split two-half eta difference
    diffs = []
    for r in range(5):
        a = s.astype(np.int64).copy(); a.data = rng.binomial(a.data, 0.5)
        b = s.astype(np.int64) - a
        ua = u.astype(np.int64).copy(); ua.data = rng.binomial(ua.data, 0.5)
        ub = u.astype(np.int64) - ua
        def half_eta(sx, ux):
            snx, unx = normalize_log(sx), normalize_log(ux)
            zx = PCA(n_components=20, svd_solver='randomized', random_state=0).fit_transform(
                np.asarray(snx.todense(), dtype=np.float32))
            gx = fit_gamma(np.asarray(snx.todense(), dtype=np.float32), np.asarray(unx.todense(), dtype=np.float32))
            sxv, uxv = np.asarray(snx.todense(), dtype=np.float32), np.asarray(unx.todense(), dtype=np.float32)
            vx = np.where(gx > 0, uxv - sxv * gx, 0.0)
            # project with the full-data components for comparability
            return local_curl(zx, vx @ pca.components_.T, n_cells=n_cells, seed=seed + r)
        diffs.append(abs(half_eta(a, ua) - half_eta(b, ub)))
    nulls = dict(gamma_rescale_95=float(np.percentile(rescale, 95)),
                 mean_shift=float(ms),
                 count_split_95=float(np.percentile(diffs, 95)))
    threshold = max(nulls.values())
    out[name] = dict(n_cells=int(keep.sum()), eta=eta, nulls=nulls, threshold=threshold,
                     passes=bool(eta > threshold))
    log(f'{name}: eta={eta:.4f} threshold={threshold:.4f} passes={eta > threshold} nulls={nulls}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='pancreas,bonemarrow,dentate,merlin315')
    ap.add_argument('--root', default='/data/yuchang/veloroute_gainprobe_20260915')
    ap.add_argument('--merlin-root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--n-cells', type=int, default=3000)
    ap.add_argument('--output', default='outputs/w1_a3_vorticity.json')
    args = ap.parse_args()
    import anndata as ad
    out = dict(tag='A3', velocity='steady_state_u_minus_gamma_s')
    wanted = args.datasets.split(',')
    paths = {'pancreas': f'{args.root}/pancreas_scvelo.h5ad',
             'bonemarrow': f'{args.root}/bonemarrow_scvelo.h5ad',
             'dentate': f'{args.root}/dentate_scvelo.h5ad'}
    for name in wanted:
        if name in paths:
            a = ad.read_h5ad(paths[name])
            s = sparse.csr_matrix(a.layers['spliced'])
            u = sparse.csr_matrix(a.layers['unspliced'])
            analyze(name, s, u, out, args.n_cells)
        elif name == 'merlin315':
            from veloroute.preprocess import load_usa
            rows, genes, mats, meta = load_usa(Path(args.merlin_root) / 'quant_merlin315' / 'af_quant')
            analyze('merlin315', mats[0].tocsr(), mats[1].tocsr(), out, args.n_cells)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
