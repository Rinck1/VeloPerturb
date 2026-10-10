"""W2 T2: shuffle scale audit. Is "GFG does not read U" a real finding or a
dimensional artifact of permute_local (depth-mismatched, PC1-sorted blocks)?

T2.1 audit existing permute_local donor->receiver pairs (U total ratio, U/(U+S)
     diff, z-space distance and its quantile in receiver 10-NN distances).
T2.2 feed U variants to GFG (S fixed): V0 real, V1 permute_local, V2 V1+depth-match,
     V3 true full-z local + depth-match, V4 global same-condition + depth-match,
     V5/V6 U*0.5 / U*2, V7 U=0, V8 10-NN mean. Report cos(v_GFG,V0 vs variant),
     norm ratio, gene-level cos, and v_kin calibration cos.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.artifacts import load_config
from veloroute.contracts import FitScope
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import FullConfig, FullVeloRoute
from veloroute.gfg_experiments import read_gene_input
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.latent import FrozenSplicingTransform, load_pack, normalized_counts
from veloroute.probes import permute_local
from sklearn.neighbors import NearestNeighbors


def log(m):
    print(f'[T2] {m}', flush=True)


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', '/home/yuchang/wangjiaxuan', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def cos_rows(a, b):
    return (unit(a) * unit(b)).sum(1)


def fit_gamma_tail(s, u, q=0.05, min_cells=10):
    ng = s.shape[1]; gamma = np.zeros(ng)
    qlo = np.quantile(s, q, axis=0); qhi = np.quantile(s, 1 - q, axis=0)
    for g in range(ng):
        tail = (s[:, g] <= qlo[g]) | (s[:, g] >= qhi[g])
        if tail.sum() < min_cells:
            continue
        ss, uu = s[tail, g], u[tail, g]
        d = float((ss * ss).sum())
        if d > 1e-12:
            gamma[g] = float((ss * uu).sum() / d)
    return gamma


def build_variants(u_raw, source, seed):
    rng = np.random.default_rng(seed)
    n = u_raw.shape[0]
    rowsum = u_raw.sum(1)
    # permute_local donor order
    _, order = permute_local(u_raw, source, seed=seed, neighbors=10)
    # full-z kNN per condition
    z = source['z']; cond = source['conditions']
    local_donor = np.arange(n)
    global_donor = np.arange(n)
    for c in sorted(set(cond)):
        idx = np.flatnonzero(cond == c)
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(10, len(idx))).fit(z[idx])
        nnidx = nn.kneighbors(z[idx], return_distance=False)
        local_donor[idx] = idx[nnidx[np.arange(len(idx)), rng.integers(nnidx.shape[1], size=len(idx))]]
        global_donor[idx] = idx[rng.integers(len(idx), size=len(idx))]
    depth_factor = lambda d: (rowsum / np.maximum(rowsum[d], 1e-9))[:, None]
    V = {}
    V['V0_real'] = u_raw.copy()
    V['V1_permute_local'] = u_raw[order]
    V['V2_perm_local_depthmatch'] = u_raw[order] * depth_factor(order)
    V['V3_truelocal_depthmatch'] = u_raw[local_donor] * depth_factor(local_donor)
    V['V4_global_depthmatch'] = u_raw[global_donor] * depth_factor(global_donor)
    V['V5_u_half'] = u_raw * 0.5
    V['V6_u_double'] = u_raw * 2.0
    V['V7_zero'] = np.zeros_like(u_raw)
    # V8 10-NN mean (same condition, k=10)
    V8 = np.zeros_like(u_raw)
    for c in sorted(set(cond)):
        idx = np.flatnonzero(cond == c)
        if len(idx) < 2:
            V8[idx] = u_raw[idx]
            continue
        nn = NearestNeighbors(n_neighbors=min(10, len(idx))).fit(z[idx])
        nnidx = nn.kneighbors(z[idx], return_distance=False)
        V8[idx] = u_raw[idx][nnidx].mean(1)
    V['V8_nn_mean'] = V8
    return V, order, local_donor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    ap.add_argument('--gene-dir', default='outputs/veloroute_gfg_inputs_20260914')
    ap.add_argument('--conditions', default='data/renge/conditions/esm2_3b_v1/conditions.npz')
    ap.add_argument('--gfg-config', default='configs/veloroute_gfg_joint_20260914.yaml')
    ap.add_argument('--kang-config', default='configs/veloroute_kang_20260914.yaml')
    ap.add_argument('--n-cells', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='outputs/w2_t2_shuffle_scale_audit.json')
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    fold = Path(args.fold); gene_dir = Path(args.gene_dir)
    source, _ = load_pack(fold / 'train_source.npz', expected_side='source')
    n = len(source['z'])
    ix = np.sort(rng.choice(n, min(args.n_cells, n), replace=False))
    values_all, source_full, _ = read_gene_input(gene_dir / 'train_gene_source.npz', fold / 'train_source.npz')
    transform = FrozenSplicingTransform.load(fold / 'transform.npz')
    ng = values_all.shape[1] // 2
    u_raw = values_all[:, :ng]
    s_raw = values_all[:, ng:]
    V, order, local_donor = build_variants(u_raw, source_full, args.seed)

    # T2.1 audit
    rowsum = u_raw.sum(1); ufrac = rowsum / np.maximum(u_raw.sum(1) + s_raw.sum(1), 1)
    donors = order[ix]; recv = ix
    ratio = rowsum[ix] / np.maximum(rowsum[donors], 1e-9)
    ufrac_diff = ufrac[ix] - ufrac[donors]
    z = source_full['z']
    dist = np.linalg.norm(z[ix] - z[donors], axis=1)
    # z-distance quantile in receiver 10-NN distribution
    nn = NearestNeighbors(n_neighbors=11).fit(z)
    nnd = nn.kneighbors(z[ix], return_distance=True)[0][:, 1:]
    quant = (dist[:, None] > nnd).mean(1)
    def q(a):
        return dict(median=float(np.median(a)), p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))
    audit = dict(u_total_ratio=q(ratio), ufrac_diff=q(ufrac_diff), z_dist=q(dist), z_dist_quantile=q(quant))

    # load model
    import torch
    kang = load_config(args.kang_config); base = load_config(args.gfg_config)
    with np.load(args.conditions, allow_pickle=False) as data:
        esm_dim = int(data['embeddings'].shape[1])
    base['model'].update(esm_dim=esm_dim, hidden_dim=768, residual_blocks=6, max_experts=8, top_k=2,
        initial_active=1, use_adaptive_modes=True, use_intrinsic=True, use_gate=True, use_noise=True,
        support_start=0., support_end=1., gfg_genes=len(transform.selected),
        state_dim=transform.components.shape[0])
    enforce_gpu_policy(args.device); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = FullVeloRoute(FullConfig(**base['model'])).to(device)
    model.dynamics.core.load_pretrained(kang['training']['gfg_checkpoint'])
    model.eval()
    model.dynamics.prepare(torch.as_tensor(values_all, dtype=torch.float32, device=device),
        torch.as_tensor(transform.components, dtype=torch.float32, device=device),
        cell_ids=list(source_full['cell_ids']), scope=FitScope(frozenset(source_full['cell_ids'])))
    tokens, mask, _ = condition_token_sets(args.conditions, source['conditions'])

    def gfg_velocity(u_arr):
        vals = values_all.copy()
        vals[:, :ng] = u_arr
        chunks = []
        with torch.no_grad():
            for start in range(0, len(ix), 256):
                bi = ix[start:start + 256]
                ctx = model.make_context(
                    torch.as_tensor(source['z'][bi], dtype=torch.float32, device=device),
                    torch.as_tensor(vals[bi], dtype=torch.float32, device=device),
                    torch.as_tensor(tokens[bi], dtype=torch.float32, device=device),
                    torch.as_tensor(mask[bi], dtype=torch.bool, device=device), time_start=0.)
                chunks.append(ctx.velocity.cpu().numpy())
        return np.concatenate(chunks)

    def vkin(u_arr):
        s_n, u_n = normalized_counts(s_raw, u_arr, transform.target_sum)
        gamma = fit_gamma_tail(s_n, u_n)
        return (np.where(gamma > 0, u_n - s_n * gamma, 0.0) @ transform.components.T) / transform.velocity_scale

    v0 = gfg_velocity(V['V0_real'])
    k0 = vkin(V['V0_real'])[ix]
    out = {}
    for name, u_arr in V.items():
        vv = gfg_velocity(u_arr)
        kv = vkin(u_arr)[ix]
        cc = cos_rows(vv, v0)
        out[name] = dict(
            gfg_cos_vs_V0=dict(median=float(np.median(cc)), q25=float(np.percentile(cc, 25)),
                               q75=float(np.percentile(cc, 75))),
            gfg_norm_ratio=float(np.median(np.linalg.norm(vv, axis=1) / (np.linalg.norm(v0, axis=1) + 1e-9))),
            vkin_cos_vs_V0=float(np.median(cos_rows(kv, k0))))
    result = dict(task='T2', git_commit=git_commit(), args=vars(args), n_cells=int(len(ix)),
                  t2_1_audit=audit, t2_2_variants=out, runtime_sec=time.time() - t0)
    Path(args.output).write_text(json.dumps(result, indent=2))
    for name in V:
        o = out[name]
        log(f'  {name:26s} gfg_cos={o["gfg_cos_vs_V0"]["median"]:+.3f} norm_ratio={o["gfg_norm_ratio"]:.2f} vkin_cos={o["vkin_cos_vs_V0"]:+.3f}')
    log(f'audit U ratio med={audit["u_total_ratio"]["median"]:.2f} z_dist_quantile med={audit["z_dist_quantile"]["median"]:.2f}')
    print(json.dumps(result, indent=2, default=str)[:3000])


if __name__ == '__main__':
    main()
