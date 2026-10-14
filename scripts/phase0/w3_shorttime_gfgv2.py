"""W3 short-time prediction with GFG v2 as a velocity estimator (RENGE day4->day5).

- unified geometry via LogStateProjection (log1p S PCA);
- GFG v2 = FrozenInnovationControl (frozen innovation slope, source scope);
- three arms: static [z], gfg_v2 [z, v], shuffled [z, v_shuffled] with U-only
  local control (local_u_control: mutual-kNN pairs, S fixed, depth matched);
- metric: pseudobulk MSE and energy on held-out TFs; condition bootstrap;
- conditional score test of v beyond z on held-out conditions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
sys.path.insert(0, '/home/yuchang/wangjiaxuan/scripts/phase0')
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import load_pack, normalized_counts
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from w3_validation import LogStateProjection, local_u_control, conditional_score_test
from w3_gfg_v2_control import FrozenInnovationControl


def log(m):
    print(f'[ST] {m}', flush=True)


def nn_displacement(source, target, neighbors=10):
    d = np.zeros_like(source['z'])
    for c in sorted(set(source['conditions'])):
        a = source['conditions'] == c; b = target['conditions'] == c
        if not b.any():
            continue
        nn = NearestNeighbors(n_neighbors=min(neighbors, int(b.sum()))).fit(target['z'][b])
        idx = nn.kneighbors(source['z'][a], return_distance=False)
        d[a] = target['z'][b][idx].mean(1) - source['z'][a]
    return d


def energy(x, y):
    x, y = np.asarray(x, 'float64'), np.asarray(y, 'float64')
    return float(max(0., 2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    ap.add_argument('--gene-dir', default='outputs/veloroute_gfg_inputs_20260914')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='outputs/w3_shorttime_gfgv2_renge.json')
    args = ap.parse_args()
    t0 = time.time()
    fold = Path(args.fold); gd = Path(args.gene_dir)
    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    train, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    train_t, _ = load_pack(fold / 'train_target.npz', expected_side='target')
    val, _ = load_pack(fold / 'validation_source.npz', expected_side='source')
    val_t, _ = load_pack(fold / 'validation_target.npz', expected_side='target')
    vtr, _, _ = read_gene_input(gd / 'train_gene_source.npz', fold / 'train_source.npz')
    vva, _, _ = read_gene_input(gd / 'validation_gene_source.npz', fold / 'validation_source.npz')
    genes = vtr.shape[1] // 2
    us_tr = np.asarray(vtr, dtype=np.float64); us_va = np.asarray(vva, dtype=np.float64)

    # unified geometry from train day4 S
    proj = LogStateProjection.fit(us_tr[:, genes:], dimensions=50, seed=args.seed)
    z_tr = proj.state(us_tr[:, genes:]); z_va = proj.state(us_va[:, genes:])
    log(f'train={len(us_tr)} val={len(us_va)} genes={genes} scale={proj.scale:.3f}')

    # GFG v2 velocity estimator, trained on train day4
    model = FrozenInnovationControl(genes, state_dim=50, dim=8, codes=32).to(device)
    model.fit_source(torch.tensor(us_tr, dtype=torch.float32),
                     cell_ids=list(train['cell_ids']), allowed_ids=set(train['cell_ids']))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    gud = torch.tensor(us_tr, dtype=torch.float32, device=device)
    n = len(gud); losses = []
    torch.manual_seed(args.seed)
    for step in range(args.steps):
        ix = torch.randint(0, n, (args.batch,))
        out = model.core(gud[ix][:, :genes], gud[ix][:, genes:], model.u_mean, model.u_std,
                         model.s_mean, model.s_std, model.innovation(gud[ix][:, :genes], gud[ix][:, genes:]), model.gamma_ss)
        opt.zero_grad(); out['native'].backward(); opt.step(); losses.append(float(out['native']))
        if (step + 1) % 1000 == 0:
            log(f'step {step+1} native={np.mean(losses[-1000:]):.2f}')

    def velocity(us):
        u, s = us[:, :genes], us[:, genes:]
        vv = []
        with torch.no_grad():
            for i in range(0, len(us), 128):
                uu = torch.tensor(u[i:i+128], dtype=torch.float32, device=device)
                ss = torch.tensor(s[i:i+128], dtype=torch.float32, device=device)
                vs = model.core.vel_head(torch.stack(((uu - model.u_mean) / model.u_std,
                                                      (ss - model.s_mean) / model.s_std,
                                                      model.innovation(uu, ss) / model.u_std), -1)).squeeze(-1)
                vv.append(((vs / (1 + ss)) @ model.components.T / model.velocity_scale).cpu().numpy())
        return np.concatenate(vv)
    v_gfg_tr = velocity(us_tr); v_gfg_va = velocity(us_va)

    # U-only local control (S fixed)
    us_tr_sh, _, audit_tr = local_u_control(us_tr, z_tr, train['conditions'], seed=args.seed, neighbors=10)
    us_va_sh, _, audit_va = local_u_control(us_va, z_va, val['conditions'], seed=args.seed, neighbors=10)
    v_sh_tr = velocity(us_tr_sh); v_sh_va = velocity(us_va_sh)
    log(f'shuffle moved={audit_tr["moved_fraction"]:.3f} depth_ratio={audit_tr["donor_u_depth_ratio_median"]}')

    disp = nn_displacement(train, train_t)
    def scale_of(vt):
        r = []
        for c in sorted(set(train['conditions'])):
            m = train['conditions'] == c
            if np.linalg.norm(vt[m].mean(0)) > 1e-10:
                r.append(np.linalg.norm(disp[m].mean(0)) / np.linalg.norm(vt[m].mean(0)))
        return float(np.median(r)) if r else 1.0
    sc_gfg = scale_of(v_gfg_tr); sc_sh = scale_of(v_sh_tr)
    arms = {'static': (None, None, 1.0), 'gfg_v2': (v_gfg_tr, v_gfg_va, sc_gfg),
            'shuffled': (v_sh_tr, v_sh_va, sc_sh)}
    errs_all = {}
    for name, (vt, ve, sc) in arms.items():
        if vt is None:
            X, Xe = z_tr, z_va
        else:
            X = np.concatenate([z_tr, sc * vt], 1)
            Xe = np.concatenate([z_va, sc * ve], 1)
        r = Ridge(alpha=1.).fit(X, disp); pred = r.predict(Xe)
        e = {}
        for c in sorted(set(val['conditions'])):
            a = val['conditions'] == c; b = val_t['conditions'] == c
            ep = z_va[a] + pred[a]
            e[str(c)] = dict(pseudobulk_mse=float(np.square(ep.mean(0) - val_t['z'][b].mean(0)).mean()),
                             energy=energy(ep, val_t['z'][b]))
        errs_all[name] = e
    def boot(g, seed=20260923):
        g = np.asarray(g); rng = np.random.default_rng(seed)
        b = g[rng.integers(len(g), size=(10000, len(g)))].mean(1)
        return float(g.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))
    conds = sorted(errs_all['static'])
    result = dict(task='RENGE_shorttime_GFGv2', final_native=float(np.mean(losses[-100:])),
                  shuffle_audit=audit_tr, arms={}, comparisons={})
    for name in arms:
        result['arms'][name] = {k: float(np.mean([errs_all[name][c][k] for c in conds])) for k in ('pseudobulk_mse', 'energy')}
    for metric in ('pseudobulk_mse', 'energy'):
        for ctrl in ('static', 'shuffled'):
            g = [errs_all[ctrl][c][metric] - errs_all['gfg_v2'][c][metric] for c in conds]
            m, lo, hi = boot(g)
            result['comparisons'][f'gfg_v2_vs_{ctrl}_{metric}'] = dict(gain=m, ci=[lo, hi], crosses_zero=bool(lo <= 0 <= hi))
    # (conditional score test omitted: only 4 held-out conditions, insufficient units)
    result['n_conditions'] = len(conds)
    result['runtime_sec'] = time.time() - t0
    Path(args.output).write_text(json.dumps(result, indent=2))
    log(json.dumps(result, indent=2, default=str)[:2200])


if __name__ == '__main__':
    main()
