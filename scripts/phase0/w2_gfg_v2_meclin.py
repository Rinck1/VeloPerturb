"""W2: MeRLin downstream with GFG v2 velocity (corrected lead-lag design).

Train GFG v2 on MeRLin Day0 (310) U/S; take its gene-level velocity; test whether
the Day0 clone-mean velocity predicts the Day21 (308) S change beyond Day0 S
(partial corr controlling S0; null permutes only Day21; eps_S control).
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.preprocess import load_usa

spec = importlib.util.spec_from_file_location('g2', '/home/yuchang/wangjiaxuan/scripts/phase0/w2_gfg_v2.py')
g2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2)


def log(m):
    print(f'[MG2] {m}', flush=True)


def normalize_log(mat, target=10000.0):
    tot = np.asarray(mat.sum(1)).ravel().astype(np.float64)
    sc = np.where(tot > 0, target / np.maximum(tot, 1e-9), 0.0)
    out = mat.astype(np.float32).multiply(sc[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def load_clones(path):
    m = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m[r['cell_barcode']] = r['clone_barcode']
    return m


def clmean(cell, lab, use):
    return np.vstack([cell[np.flatnonzero(lab == c)].mean(0) for c in use])


def vec_partial_corr(Y, X, C):
    Cc = C - C.mean(0, keepdims=True); den = (Cc ** 2).sum(0)
    def resid(A):
        Ac = A - A.mean(0, keepdims=True)
        b = np.where(den > 1e-12, (Cc * Ac).sum(0) / np.where(den > 1e-12, den, 1.0), 0.0)
        return Ac - Cc * b
    ry, rx = resid(Y), resid(X)
    num = (ry * rx).sum(0); dnm = np.sqrt((ry ** 2).sum(0) * (rx ** 2).sum(0))
    valid = (np.std(Y, 0) > 1e-9) & (np.std(X, 0) > 1e-9) & (dnm > 1e-12)
    r = np.where(valid, num / np.where(dnm > 1e-12, dnm, 1.0), np.nan)
    return float(np.nanmean(r)), int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/yuchang/veloroute_ucheck_20260915')
    ap.add_argument('--day0', default='310')
    ap.add_argument('--day21', default='308')
    ap.add_argument('--top-hvg', type=int, default=2000)
    ap.add_argument('--min-cells', type=int, default=3)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-perm', type=int, default=2000)
    ap.add_argument('--output', default='outputs/w2_gfg_v2_meclin.json')
    args = ap.parse_args()
    t0 = time.time()
    root = Path(args.root)
    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

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
    keepc = np.asarray(combs.sum(1)).ravel() >= 100
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
    log(f'310 cells={s0.shape[0]} 308 cells={s21.shape[0]} genes={len(sel)}')

    # train GFG v2 on 310
    gu = torch.tensor(np.concatenate([np.asarray(u0.todense(), dtype=np.float32),
                                      np.asarray(s0.todense(), dtype=np.float32)], 1))
    from sklearn.decomposition import PCA
    # use project-style components: PCA-50 of raw S
    comp = torch.tensor(PCA(50, svd_solver='randomized', random_state=0).fit(np.asarray(s0.todense(), dtype=np.float32)).components_)
    model = g2.GFGv2(args.top_hvg, state_dim=50, dim=8, codes=32).to(device)
    model.prepare(gu.to(device), comp.to(device))
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gud = gu.to(device); n = len(gud); losses = []
    for step in range(args.steps):
        ix = torch.randint(0, n, (args.batch,))
        out, v = model(gud[ix])
        opt.zero_grad(); out['native'].backward(); opt.step(); losses.append(float(out['native']))
        if (step + 1) % 1000 == 0:
            log(f'step {step+1} native={np.mean(losses[-1000:]):.2f}')
    model.eval()
    # gene-level velocity for 310 (and S21 for target)
    vg = []
    with torch.no_grad():
        for i in range(0, len(gud), 64):
            out, _ = model(gud[i:i+64]); vg.append(out['vs'].cpu().numpy())
    vg = np.concatenate(vg)                          # (cells, genes) Day0 GFG v2 velocity
    S0c = np.asarray(normalize_log(s0).todense(), dtype=np.float32)
    S21c = np.asarray(normalize_log(s21).todense(), dtype=np.float32)
    # eps_S (Day0 S residual on z) as control
    z = PCA(50, svd_solver='randomized', random_state=0).fit_transform(S0c)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=31).fit(z); idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(len(z)), 30)
    W = sparse.csr_matrix((np.full(len(z) * 30, 1 / 30), (rows, idx.ravel())), shape=(len(z), len(z)))
    epsS = (S0c - W @ S0c)

    cmap0 = load_clones(root / f'clones/SRR33960{args.day0}' / 'cell_clone_assignments.csv')
    cmap21 = load_clones(root / f'clones/SRR33960{args.day21}' / 'cell_clone_assignments.csv')
    lab0 = np.array([cmap0.get(b, '') for b in r0], dtype=object)
    lab21 = np.array([cmap21.get(b, '') for b in r21], dtype=object)
    vals0, cnt0 = np.unique(lab0[lab0 != ''], return_counts=True)
    c0 = set(vals0[cnt0 >= args.min_cells]); c21 = set(lab21[lab21 != ''])
    shared = set(cmap0.values()) & set(cmap21.values())
    use = sorted(shared & c0 & c21)
    log(f'usable clones={len(use)}')
    V0 = clmean(vg, lab0, use); ES0 = clmean(epsS, lab0, use); S0 = clmean(S0c, lab0, use); S21 = clmean(S21c, lab21, use)

    modules = {'ALL': np.ones(len(sel), bool)}
    progs = json.loads((root / 'merlin_programs.json').read_text())['signatures']
    sym2idx = {s: i for i, s in enumerate(symbols[sel])}
    for k, p in progs.items():
        gs = [str(g).replace('*', '').strip() for g in (p.get('genes', p) if isinstance(p, dict) else p)]
        gi = np.array([sym2idx[g] for g in gs if g in sym2idx])
        if len(gi) >= 5:
            modules[k] = np.isin(np.arange(len(sel)), gi)

    def null(stat, seed=0):
        rr = np.random.default_rng(seed); v = []
        for _ in range(args.n_perm):
            p = rr.permutation(len(use)); v.append(stat(p))
        return np.array(v)

    out = {}
    for mname, mask in modules.items():
        gi = np.flatnonzero(mask)
        obs, ng = vec_partial_corr(S21[:, gi], V0[:, gi], S0[:, gi])
        ctrl, _ = vec_partial_corr(S21[:, gi], ES0[:, gi], S0[:, gi])
        # marginal velocity (no S0 control)
        marg = float(np.nanmean([np.corrcoef(S21[:, g], V0[:, g])[0, 1] for g in gi if np.std(V0[:, g]) > 1e-9 and np.std(S21[:, g]) > 1e-9]))
        nl = null(lambda p: vec_partial_corr(S21[p][:, gi], V0[:, gi], S0[:, gi])[0])
        out[mname] = dict(n_genes=ng, n_clones=len(use), obs_partial=obs, epsS_control=ctrl, marginal=marg,
                          null95=float(np.percentile(nl, 95)),
                          p=float((1 + (nl >= obs).sum()) / (args.n_perm + 1)),
                          passes=bool(obs > np.percentile(nl, 95) and obs > ctrl))
        log(f'  {mname}: n={ng} obs={obs:+.4f} ctrl={ctrl:+.4f} marginal={marg:+.4f} null95={out[mname]["null95"]:+.4f} p={out[mname]["p"]:.3f} pass={out[mname]["passes"]}')
    result = dict(task='MeRLin_310_to_308_GFGv2', final_native=float(np.mean(losses[-100:])),
                  n_clones=len(use), modules=out, runtime_sec=time.time() - t0)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2, default=str)[:2500])


if __name__ == '__main__':
    main()
