"""Lead-lag test: does Day0 unspliced innovation predict Day21 spliced change,
clone by clone? (MeRLin 310 Day0 -> 308 Day21)

Dynamics => U leads S. For each shared clone:
  kappa0_c = mean (within-clone) kappa = eps_U - b_iv*eps_S at Day0 (310)
  dS_c     = mean log1p(norm S) Day21 (308) - Day0 (310)
Per module (DTP program gene sets + whole selected set), clone-level Pearson
across module genes; observed = mean over clones; null shuffles clone
correspondence (Day0 kappa of clone i paired with Day21 dS of clone j).
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


def prep(name, s_mat, u_mat, clone_map, rng, sel):
    rows = np.array(list(clone_map), dtype=object)
    # align cells: keep those with total>=100
    tot = np.asarray((s_mat + u_mat).sum(1)).ravel()
    keep = tot >= 100
    s, u = s_mat[keep].tocsr(), u_mat[keep].tocsr()
    snf = normalize_log(s)
    unf = normalize_log(u)
    z = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(snf.todense(), dtype=np.float32))
    epsU = knn_residual(np.asarray(unf.todense(), dtype=np.float32), z)
    epsS = knn_residual(np.asarray(snf.todense(), dtype=np.float32), z)
    # IV b from count-split halves
    s_a, s_b = count_split(s, rng); u_a, u_b = count_split(u, rng)
    z_a = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s_a).todense(), dtype=np.float32))
    z_b = PCA(50, svd_solver='randomized', random_state=0).fit_transform(np.asarray(normalize_log(s_b).todense(), dtype=np.float32))
    eU_b = knn_residual(np.asarray(normalize_log(u_b).todense(), dtype=np.float32), z_b)
    eS_a = knn_residual(np.asarray(normalize_log(s_a).todense(), dtype=np.float32), z_a)
    eS_b = knn_residual(np.asarray(normalize_log(s_b).todense(), dtype=np.float32), z_b)
    num = (eU_b * eS_a).mean(0) - eU_b.mean(0) * eS_a.mean(0)
    den = (eS_b * eS_a).mean(0) - eS_b.mean(0) * eS_a.mean(0)
    b_iv = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    kappa = epsU - epsS * b_iv
    log(f'{name}: cells={keep.sum()} b_iv_med={np.median(b_iv):.3f}')
    return keep, snf, kappa, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--day0', default='310')
    ap.add_argument('--day21', default='308')
    ap.add_argument('--top-hvg', type=int, default=2000)
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
    ng = len(genes0)
    # gene symbol map
    name_map = {}
    with (root / f'quant_merlin{args.day0}' / 'af_quant' / 'gene_id_to_name.tsv').open() as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2:
                name_map[parts[0]] = parts[1]
    symbols = np.array([name_map.get(g, '') for g in genes0])

    # select genes: HVG on combined S
    combs = sparse.vstack([s0, s21]).tocsr()
    tot = np.asarray(combs.sum(1)).ravel(); keepc = tot >= 100
    logs = normalize_log(combs[keepc]).astype(np.float32)
    expressed = np.asarray((combs[keepc] > 0).sum(0)).ravel() >= 10
    var = np.asarray(logs.power(2).mean(0)).ravel() - np.asarray(logs.mean(0)).ravel() ** 2
    var[~expressed] = -1
    sel = np.sort(np.argsort(-var)[:args.top_hvg])
    log(f'selected {len(sel)} genes')

    cmap0 = load_clones(root / f'clones/SRR33960{args.day0}' / 'cell_clone_assignments.csv')
    cmap21 = load_clones(root / f'clones/SRR33960{args.day21}' / 'cell_clone_assignments.csv')
    shared = sorted(set(cmap0.values()) & set(cmap21.values()))
    log(f'shared clones={len(shared)}')

    # per-cell labels aligned: build clone index arrays for the raw quant rows
    lab0 = np.array([cmap0.get(bc, '') for bc in r0], dtype=object)
    lab21 = np.array([cmap21.get(bc, '') for bc in r21], dtype=object)

    # subset to selected genes before prep to save memory
    s0, u0 = s0[:, sel].tocsr(), u0[:, sel].tocsr()
    s21, u21 = s21[:, sel].tocsr(), u21[:, sel].tocsr()
    keep0, S0, K0, _ = prep(args.day0, s0, u0, cmap0, rng, sel)
    keep21, S21, K21, _ = prep(args.day21, s21, u21, cmap21, rng, sel)
    lab0 = lab0[keep0]; lab21 = lab21[keep21]

    # clone-mean kappa (Day0) and S (both days)
    k0_mean, S0_mean, S21_mean = {}, {}, {}
    for c in shared:
        m0 = np.flatnonzero(lab0 == c)
        m21 = np.flatnonzero(lab21 == c)
        if len(m0) == 0 or len(m21) == 0:
            continue
        k0_mean[c] = K0[m0].mean(0)
        S0_mean[c] = np.asarray(S0[m0].todense()).mean(0)
        S21_mean[c] = np.asarray(S21[m21].todense()).mean(0)
    cl = [c for c in shared if c in k0_mean]
    Kmat = np.vstack([k0_mean[c] for c in cl])           # clones x genes
    dSmat = np.vstack([S21_mean[c] - S0_mean[c] for c in cl])
    log(f'clones used={len(cl)} Kmat={Kmat.shape}')

    # center per gene across clones (focus on clone-specific patterns)
    Kc = Kmat - np.nanmean(Kmat, 0, keepdims=True)
    dSc = dSmat - np.nanmean(dSmat, 0, keepdims=True)

    # modules: DTP programs + whole set
    prog_path = root / 'merlin_programs.json'
    modules = {'ALL': np.ones(len(sel), bool)}
    if prog_path.exists():
        progs = json.loads(prog_path.read_text())['signatures']
        sym2idx = {s: i for i, s in enumerate(symbols[sel])}
        for k, p in progs.items():
            gs = p.get('genes', p) if isinstance(p, dict) else p
            gs = [str(g).replace('*', '').strip() for g in gs]
            idx = np.array([sym2idx[g] for g in gs if g in sym2idx])
            if len(idx) >= 5:
                modules[k] = np.isin(np.arange(len(sel)), idx)

    def module_corr(K, dS, mask):
        x = K[:, mask]; y = dS[:, mask]
        x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
        num = (x * y).sum(1)
        den = np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1)) + 1e-12
        return num / den

    out = {}
    for mname, mask in modules.items():
        if mask.sum() < 5:
            continue
        obs = module_corr(Kc, dSc, mask)
        obs_mean = float(np.mean(obs))
        # null: shuffle clone correspondence
        rng2 = np.random.default_rng(0)
        null = []
        for _ in range(args.n_perm):
            perm = rng2.permutation(len(cl))
            null.append(float(np.mean(module_corr(Kc, dSc[perm], mask))))
        null = np.array(null)
        out[mname] = dict(n_genes=int(mask.sum()), n_clones=len(cl),
                          obs_mean_corr=obs_mean, null_mean=float(null.mean()),
                          null95=float(np.percentile(null, 95)),
                          p_value=float((1 + (null >= obs_mean).sum()) / (args.n_perm + 1)),
                          passes=bool(obs_mean > np.percentile(null, 95)))
        log(f'  {mname}: n={int(mask.sum())} obs={obs_mean:.4f} null95={out[mname]["null95"]:.4f} p={out[mname]["p_value"]:.3f} pass={out[mname]["passes"]}')
    result = dict(tag='LEADLAG', day0=args.day0, day21=args.day21, n_shared_clones=len(cl), modules=out)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
