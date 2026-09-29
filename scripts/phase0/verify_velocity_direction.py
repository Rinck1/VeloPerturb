"""Generalized velocity direction-correctness check for prepared folds.

Usage:
  verify_velocity_direction.py --fold DIR [--gene-dir DIR] [--conditions PATH] --tag NAME --output X.json
Runs:
  * residual velocity test (skipped when the pack velocity is all zeros)
  * GFG decoder-JVP test (needs gene_source packs + transform)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.artifacts import load_config
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.probes import permute_local
from veloroute.gpu_policy import enforce_gpu_policy
from sklearn.neighbors import NearestNeighbors

MIN_CELLS = 20


def direction_cos(z_src, v_src, z_tgt, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(z_tgt))).fit(z_tgt)
    idx = nn.kneighbors(z_src, return_distance=False)
    d = z_tgt[idx] - z_src[:, None, :]
    d = d/np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
    d = d.mean(1)
    d = d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    v = v_src/np.maximum(np.linalg.norm(v_src, axis=1, keepdims=True), 1e-9)
    return (v*d).sum(1)


def coherence(v):
    v = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    if len(v) < 2:
        return 0.0
    gram = v @ v.T
    return float((gram.sum()-len(v))/(len(v)*(len(v)-1)))


def condition_rows(role, source, target, v, label):
    rows = []
    v_shuf = permute_local(v, source, seed=0, neighbors=10)[0]
    for cond in sorted(set(source['conditions'])):
        ms = np.asarray(source['conditions'] == cond)
        mt = np.asarray(target['conditions'] == cond)
        if int(ms.sum()) < MIN_CELLS or int(mt.sum()) < MIN_CELLS:
            continue
        cos = direction_cos(source['z'][ms], v[ms], target['z'][mt])
        cos_shuf = direction_cos(source['z'][ms], v_shuf[ms], target['z'][mt])
        rows.append(dict(velocity=label, role=role, condition=str(cond), n=int(ms.sum()),
            mean_cos=float(cos.mean()), mean_cos_shuffled=float(cos_shuf.mean()),
            velocity_coherence=coherence(v[ms])))
    return rows


def residual_test(fold, results):
    for role in ('train', 'validation'):
        try:
            source, sm = load_pack(fold/f'{role}_source.npz', expected_side='source')
        except Exception:
            continue
        if float(np.abs(source['velocity']).max()) == 0.:
            results.append(dict(velocity='residual', role=role, condition='ALL', n=0,
                                note='zero velocity in pack; skipped'))
            continue
        target, tm = load_pack(fold/f'{role}_target.npz', expected_side='target')
        results.extend(condition_rows(role, source, target, source['velocity'], 'residual'))


def gfg_test(fold, gene_dir, conditions_path, results, *, device='cpu', seed=0):
    import torch
    from veloroute.contracts import FitScope
    from veloroute.full_experiments import condition_token_sets
    from veloroute.full_model import FullConfig, FullVeloRoute
    from veloroute.gfg_experiments import read_gene_input
    kang = load_config('configs/veloroute_kang_20260914.yaml')
    base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    with np.load(Path(conditions_path), allow_pickle=False) as data:
        esm_dim = int(data['embeddings'].shape[1])
    base['model'].update(esm_dim=esm_dim, hidden_dim=768, residual_blocks=6, max_experts=8, top_k=2,
        initial_active=1, use_adaptive_modes=True, use_intrinsic=True, use_gate=True, use_noise=True,
        support_start=0., support_end=1., gfg_genes=len(transform.selected),
        state_dim=transform.components.shape[0])
    enforce_gpu_policy(device)
    torch.manual_seed(seed)
    device = torch.device(device)
    model = FullVeloRoute(FullConfig(**base['model'])).to(device)
    model.dynamics.core.load_pretrained(kang['training']['gfg_checkpoint'])
    model.eval()
    for role in ('train', 'validation'):
        gpath = Path(gene_dir)/f'{role}_gene_source.npz'
        spath = fold/f'{role}_source.npz'
        if not gpath.exists():
            results.append(dict(velocity='GFG', role=role, condition='ALL', n=0, note='no gene pack; skipped'))
            continue
        values, source, sm = read_gene_input(gpath, spath)
        target, tm = load_pack(fold/f'{role}_target.npz', expected_side='target')
        if role == 'train':
            model.dynamics.prepare(torch.as_tensor(values, dtype=torch.float32, device=device),
                torch.as_tensor(transform.components, dtype=torch.float32, device=device),
                cell_ids=list(source['cell_ids']), scope=FitScope(frozenset(source['cell_ids'])))
        elif not bool(model.dynamics.prepared):
            raise ValueError('Training source normalization must be available before validation')
        tokens, mask, _ = condition_token_sets(str(conditions_path), source['conditions'])
        chunks = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                ix = slice(start, start+256)
                context = model.make_context(
                    torch.as_tensor(source['z'][ix], dtype=torch.float32, device=device),
                    torch.as_tensor(values[ix], dtype=torch.float32, device=device),
                    torch.as_tensor(tokens[ix], dtype=torch.float32, device=device),
                    torch.as_tensor(mask[ix], dtype=torch.bool, device=device), time_start=0.)
                chunks.append(context.velocity.cpu().numpy())
        v_gfg = np.concatenate(chunks)
        results.extend(condition_rows(role, source, target, v_gfg, 'GFG'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--gene-dir')
    parser.add_argument('--conditions')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fold = Path(args.fold)
    gene_dir = Path(args.gene_dir) if args.gene_dir else fold
    conditions = args.conditions or str(fold/'conditions.npz')
    results = []
    residual_test(fold, results)
    try:
        gfg_test(fold, gene_dir, conditions, results)
    except Exception as error:
        results.append(dict(velocity='GFG', role='ALL', condition='ERROR', n=0,
                            note=f'{type(error).__name__}: {error}'))
    out = dict(tag=args.tag, fold=str(fold), rows=results)
    Path(args.output).write_text(json.dumps(out, indent=2))
    for row in results:
        if row.get('n'):
            print(f"{row['velocity']:9s} {row['role']:10s} {row['condition']:40s} n={row['n']:5d} "
                  f"cos={row['mean_cos']:+.3f} (shuf {row['mean_cos_shuffled']:+.3f})")
        else:
            print(f"{row['velocity']:9s} {row['role']:10s} {row['condition']}: {row.get('note')}")
