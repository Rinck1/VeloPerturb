"""A1: GFG velocity direction source (N4 diagnostic).

Question: is the current GFG velocity direction driven by U/S dynamics, or by
density geometry (mean-shift)?

Reports per-cell cos(v_GFG, m), cos(v_GFG, v_kin), cos(v_kin, m) medians/IQR.
Judgement: cos(v_GFG, m) - cos(v_GFG, v_kin) >= 0.2 -> N4 holds (GFG reads geometry).
"""
import argparse
import json
import sys
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
from sklearn.neighbors import NearestNeighbors


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def cos_rows(a, b):
    return (unit(a) * unit(b)).sum(1)


def fit_gamma_tail(s, u, q=0.05, min_cells=10):
    """Per-gene steady-state gamma from the low/high q-quantile tails of s."""
    ng = s.shape[1]
    gamma = np.zeros(ng)
    for g in range(ng):
        sg = s[:, g]
        if (sg > 0).sum() < min_cells:
            continue
        lo = sg <= np.quantile(sg, q)
        hi = sg >= np.quantile(sg, 1 - q)
        tail = lo | hi
        if tail.sum() < min_cells:
            continue
        ss, uu = sg[tail], u[tail, g]
        var = float((ss * ss).sum())
        if var <= 1e-12:
            continue
        gamma[g] = float((ss * uu).sum() / var)
    return gamma


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
    ap.add_argument('--output', default='outputs/w1_a1_gfg_direction_source.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fold = Path(args.fold)
    gene_dir = Path(args.gene_dir)

    source, sm = load_pack(fold / 'train_source.npz', expected_side='source')
    z = source['z']
    n = len(z)
    ix = rng.choice(n, size=min(args.n_cells, n), replace=False)
    ix.sort()
    z = z[ix]

    # ---- v_kin from raw normalized U/S via the frozen transform ----
    transform = FrozenSplicingTransform.load(fold / 'transform.npz')
    values, source_full, _ = read_gene_input(gene_dir / 'train_gene_source.npz', fold / 'train_source.npz')
    ng = values.shape[1] // 2
    u_raw, s_raw = values[:, :ng], values[:, ng:]
    s_norm, u_norm = normalized_counts(s_raw, u_raw, transform.target_sum)
    gamma = fit_gamma_tail(s_norm, u_norm)
    supported = gamma > 0
    v_kin_gene = np.where(supported, u_norm - s_norm * gamma, 0.0)
    v_kin = (v_kin_gene @ transform.components.T) / transform.velocity_scale
    v_kin = v_kin[ix]
    # Controls for the "GFG reads U/S dynamics" claim: the -gamma*s term is fully
    # state-determined. Local U shuffle (within condition/depth, z-sorted blocks)
    # keeps that term but breaks any genuine U signal.
    from veloroute.probes import permute_local
    u_shuf, _ = permute_local(u_norm, source_full, seed=args.seed, neighbors=10)
    v_kin_shuf_gene = np.where(supported, u_shuf - s_norm * gamma, 0.0)
    v_kin_shuf = (v_kin_shuf_gene @ transform.components.T) / transform.velocity_scale
    v_kin_shuf = v_kin_shuf[ix]
    v_neg_gs = ((-s_norm * gamma) @ transform.components.T) / transform.velocity_scale
    v_neg_gs = v_neg_gs[ix]
    v_u_only = ((np.where(supported, u_norm, 0.0)) @ transform.components.T) / transform.velocity_scale
    v_u_only = v_u_only[ix]

    # ---- mean-shift m from z ----
    nn = NearestNeighbors(n_neighbors=min(20, len(z))).fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    m = z[idx].mean(1) - z

    # ---- v_GFG from frozen checkpoint ----
    import torch
    kang = load_config(args.kang_config)
    base = load_config(args.gfg_config)
    with np.load(args.conditions, allow_pickle=False) as data:
        esm_dim = int(data['embeddings'].shape[1])
    base['model'].update(esm_dim=esm_dim, hidden_dim=768, residual_blocks=6, max_experts=8, top_k=2,
        initial_active=1, use_adaptive_modes=True, use_intrinsic=True, use_gate=True, use_noise=True,
        support_start=0., support_end=1., gfg_genes=len(transform.selected),
        state_dim=transform.components.shape[0])
    enforce_gpu_policy(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = FullVeloRoute(FullConfig(**base['model'])).to(device)
    model.dynamics.core.load_pretrained(kang['training']['gfg_checkpoint'])
    model.eval()
    model.dynamics.prepare(torch.as_tensor(values, dtype=torch.float32, device=device),
        torch.as_tensor(transform.components, dtype=torch.float32, device=device),
        cell_ids=list(source_full['cell_ids']), scope=FitScope(frozenset(source_full['cell_ids'])))
    tokens, mask, _ = condition_token_sets(args.conditions, source['conditions'])
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ix), 256):
            bi = ix[start:start + 256]
            context = model.make_context(
                torch.as_tensor(source['z'][bi], dtype=torch.float32, device=device),
                torch.as_tensor(values[bi], dtype=torch.float32, device=device),
                torch.as_tensor(tokens[bi], dtype=torch.float32, device=device),
                torch.as_tensor(mask[bi], dtype=torch.bool, device=device), time_start=0.)
            chunks.append(context.velocity.cpu().numpy())
    v_gfg = np.concatenate(chunks)

    cos_gfg_m = cos_rows(v_gfg, m)
    cos_gfg_kin = cos_rows(v_gfg, v_kin)
    cos_kin_m = cos_rows(v_kin, m)
    cos_gfg_kin_shuf = cos_rows(v_gfg, v_kin_shuf)
    cos_gfg_neg_gs = cos_rows(v_gfg, v_neg_gs)
    cos_gfg_u_only = cos_rows(v_gfg, v_u_only)

    def stat(a):
        return dict(median=float(np.median(a)), q25=float(np.percentile(a, 25)),
                    q75=float(np.percentile(a, 75)), mean=float(a.mean()))

    gap = float(np.median(cos_gfg_m) - np.median(cos_gfg_kin))
    u_specific = float(np.median(cos_gfg_kin) - np.median(cos_gfg_kin_shuf))
    result = dict(tag='A1', fold=str(fold), n_cells=int(len(ix)),
                  cos_vGFG_meanshift=stat(cos_gfg_m),
                  cos_vGFG_vkin=stat(cos_gfg_kin),
                  cos_vkin_meanshift=stat(cos_kin_m),
                  cos_vGFG_vkin_Ushuffled=stat(cos_gfg_kin_shuf),
                  cos_vGFG_neg_gamma_s=stat(cos_gfg_neg_gs),
                  cos_vGFG_U_only=stat(cos_gfg_u_only),
                  gap_gfg_m_minus_gfg_kin=gap,
                  N4_holds=bool(gap >= 0.2),
                  U_specific_gap_gfg_kin_minus_Ushuf=u_specific,
                  GFG_reads_U_gap=bool(u_specific >= 0.1),
                  norm_vgfg=float(np.sqrt((v_gfg**2).mean())),
                  norm_vkin=float(np.sqrt((v_kin**2).mean())))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
