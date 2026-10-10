"""W2 downstream: does GFG v2 velocity add information beyond state on RENGE?

Task: RENGE day4 -> day5, held-out TFs. Train GFG v2 on the RENGE training-source
U/S (fold gene set), then a three-arm endpoint model predicts the day5 displacement:
  static  : ridge(z)
  gfg_v2  : ridge([z, scale*v_gfg])
  shuffled: ridge([z, scale*v_shuffled])   (U shuffled within condition/depth blocks)
  v_kin   : ridge([z, scale*v_kin])        (analytic steady-state reference)
Metric: pseudobulk MSE and energy per held-out condition; condition bootstrap.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform, load_pack, normalized_counts
from veloroute.probes import permute_local
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

spec = importlib.util.spec_from_file_location('g2', '/home/yuchang/wangjiaxuan/scripts/phase0/w2_gfg_v2.py')
g2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2)


def log(m):
    print(f'[DS] {m}', flush=True)


def nn_displacement(source, target, neighbors=10):
    result = np.zeros_like(source['z'])
    for c in sorted(set(source['conditions'])):
        a = source['conditions'] == c; b = target['conditions'] == c
        if not b.any():
            continue
        nn = NearestNeighbors(n_neighbors=min(neighbors, int(b.sum()))).fit(target['z'][b])
        idx = nn.kneighbors(source['z'][a], return_distance=False)
        result[a] = target['z'][b][idx].mean(1) - source['z'][a]
    return result


def energy(x, y):
    x, y = np.asarray(x, 'float64'), np.asarray(y, 'float64')
    return float(max(0., 2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()))


def fit_gamma_tail(s, u, q=0.05, min_cells=10):
    ng = s.shape[1]; gam = np.zeros(ng)
    qlo = np.quantile(s, q, 0); qhi = np.quantile(s, 1 - q, 0)
    for g in range(ng):
        t = (s[:, g] <= qlo[g]) | (s[:, g] >= qhi[g])
        if t.sum() < min_cells:
            continue
        d = float((s[t, g] ** 2).sum())
        if d > 1e-12:
            gam[g] = float((s[t, g] * u[t, g]).sum() / d)
    return gam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    ap.add_argument('--gene-dir', default='outputs/veloroute_gfg_inputs_20260914')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='outputs/w2_gfg_v2_downstream_renge.json')
    args = ap.parse_args()
    t0 = time.time()
    fold = Path(args.fold); gd = Path(args.gene_dir)
    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    transform = FrozenSplicingTransform.load(fold / 'transform.npz')
    comp = torch.tensor(transform.components, dtype=torch.float32)  # (state_dim, genes)
    ngenes = comp.shape[1]

    train, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    train_t, _ = load_pack(fold / 'train_target.npz', expected_side='target')
    val, _ = load_pack(fold / 'validation_source.npz', expected_side='source')
    val_t, _ = load_pack(fold / 'validation_target.npz', expected_side='target')
    vtr_vals, _, _ = read_gene_input(gd / 'train_gene_source.npz', fold / 'train_source.npz')
    vva_vals, _, _ = read_gene_input(gd / 'validation_gene_source.npz', fold / 'validation_source.npz')
    assert vtr_vals.shape[1] == 2 * ngenes

    gu_tr = torch.tensor(vtr_vals, dtype=torch.float32)
    gu_va = torch.tensor(vva_vals, dtype=torch.float32)
    model = g2.GFGv2(ngenes, state_dim=comp.shape[0], dim=8, codes=32).to(device)
    model.prepare(gu_tr.to(device), comp.to(device))
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gu_tr_dev = gu_tr.to(device)
    n = len(gu_tr_dev); losses = []
    for step in range(args.steps):
        ix = torch.randint(0, n, (args.batch,))
        out, v = model(gu_tr_dev[ix])
        opt.zero_grad(); out['native'].backward(); opt.step()
        losses.append(float(out['native']))
        if (step + 1) % 1000 == 0:
            log(f'step {step+1} native={np.mean(losses[-1000:]):.3f}')
    model.eval()

    def predict_v(x):
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                _, v = model(x[i:i+64].to(device))
                outs.append(v.cpu().numpy())
        return np.concatenate(outs)

    v_gfg_tr = predict_v(gu_tr); v_gfg_va = predict_v(gu_va)
    # v_kin analytic on the same genes
    def vkin(gene_us):
        ng = gene_us.shape[1] // 2
        s, u = normalized_counts(gene_us[:, ng:], gene_us[:, :ng], transform.target_sum)
        gam = fit_gamma_tail(s, u)
        return (np.where(gam > 0, u - s * gam, 0.0) @ transform.components.T) / transform.velocity_scale
    vk_tr = vkin(vtr_vals.astype(np.float64)); vk_va = vkin(vva_vals.astype(np.float64))
    # shuffled: shuffle U within condition/depth/z blocks, recompute GFG v2
    shuf_tr = permute_local(gu_tr.numpy(), train, seed=args.seed, neighbors=10)[0]
    shuf_va = permute_local(gu_va.numpy(), val, seed=args.seed, neighbors=10)[0]
    v_sh_tr = predict_v(torch.tensor(shuf_tr)); v_sh_va = predict_v(torch.tensor(shuf_va))

    disp = nn_displacement(train, train_t)

    def scale(v_tr, disp):
        r = []
        for c in sorted(set(train['conditions'])):
            m = train['conditions'] == c
            vm, dm = v_tr[m].mean(0), disp[m].mean(0)
            if np.linalg.norm(vm) > 1e-10:
                r.append(np.linalg.norm(dm) / np.linalg.norm(vm))
        return float(np.median(r)) if r else 1.0

    arms = {
        'static': (None, None),
        'gfg_v2': (v_gfg_tr, v_gfg_va),
        'shuffled': (v_sh_tr, v_sh_va),
        'v_kin': (vk_tr, vk_va),
    }
    result = dict(task='RENGE_day4_to_day5', gene_set='fold_2000', steps=args.steps,
                  final_native=float(np.mean(losses[-100:])), arms={})
    preds = {}
    for name, (vtr, vva) in arms.items():
        if vtr is None:
            X = train['z']; Xe = val['z']
        else:
            sc = scale(vtr, disp)
            X = np.concatenate([train['z'], sc * vtr], 1)
            Xe = np.concatenate([val['z'], sc * vva], 1)
        r = Ridge(alpha=1.).fit(X, disp)
        pred = r.predict(Xe)
        errs = {}
        for c in sorted(set(val['conditions'])):
            a = val['conditions'] == c; b = val_t['conditions'] == c
            ep = (val['z'][a] + pred[a])
            errs[str(c)] = dict(pseudobulk_mse=float(np.square(ep.mean(0) - val_t['z'][b].mean(0)).mean()),
                                energy=energy(ep, val_t['z'][b]))
        result['arms'][name] = {k: float(np.mean([e[k] for e in errs.values()])) for k in ('pseudobulk_mse', 'energy')}
        preds[name] = errs
    # gains vs static and vs shuffled (condition bootstrap)
    def boot(g):
        g = np.asarray(g); rng = np.random.default_rng(20260923)
        b = g[rng.integers(len(g), size=(10000, len(g)))].mean(1)
        return float(g.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))
    conds = sorted(preds['static'])
    for metric in ('pseudobulk_mse', 'energy'):
        for comp_name in ('static', 'shuffled'):
            g = [preds[comp_name][c][metric] - preds['gfg_v2'][c][metric] for c in conds]
            m, lo, hi = boot(g)
            result[f'gfg_v2_vs_{comp_name}_{metric}'] = dict(gain=m, ci=[lo, hi], crosses_zero=bool(lo <= 0 <= hi))
    result['runtime_sec'] = time.time() - t0
    Path(args.output).write_text(json.dumps(result, indent=2))
    log(json.dumps({k: v for k, v in result.items() if not k.startswith('_')}, indent=2, default=str)[:2500])


if __name__ == '__main__':
    main()
