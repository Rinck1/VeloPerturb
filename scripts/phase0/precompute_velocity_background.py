"""Precompute background-subtracted velocity fields for a fold.

Backgrounds: E[v|z] and E[r|z] fitted on TRAINING source cells (ridge).
Saved per-role arrays are aligned with the source pack cell order, so trainers and
predictors can slice them by batch index.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.artifacts import load_config
from veloroute.contracts import FitScope
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import FullConfig, FullVeloRoute
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.velocity_background import BACKGROUND_FORMAT
from sklearn.linear_model import Ridge


def build_model(fold, conditions_path, *, device='cpu', seed=0):
    kang = load_config('configs/veloroute_kang_20260914.yaml')
    base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    transform = FrozenSplicingTransform.load(Path(fold)/'transform.npz')
    with np.load(conditions_path, allow_pickle=False) as data:
        esm_dim = int(data['embeddings'].shape[1])
    base['model'].update(esm_dim=esm_dim, hidden_dim=768, residual_blocks=6, max_experts=8, top_k=2,
        initial_active=1, use_adaptive_modes=True, use_intrinsic=True, use_gate=True, use_noise=True,
        support_start=0., support_end=1., gfg_genes=len(transform.selected),
        state_dim=transform.components.shape[0])
    import torch
    enforce_gpu_policy(device)
    torch.manual_seed(seed)
    device = torch.device(device)
    model = FullVeloRoute(FullConfig(**base['model'])).to(device)
    model.dynamics.core.load_pretrained(kang['training']['gfg_checkpoint'])
    model.eval()
    return model, transform, device


def contexts(fold, gene_dir, conditions_path, *, device='cpu', seed=0):
    import torch
    model, transform, device = build_model(fold, conditions_path, device=device, seed=seed)
    out = {}
    for role in ('train', 'validation'):
        values, source, sm = read_gene_input(Path(gene_dir)/f'{role}_gene_source.npz',
                                             Path(fold)/f'{role}_source.npz')
        if role == 'train':
            model.dynamics.prepare(torch.as_tensor(values, dtype=torch.float32, device=device),
                torch.as_tensor(transform.components, dtype=torch.float32, device=device),
                cell_ids=list(source['cell_ids']), scope=FitScope(frozenset(source['cell_ids'])))
        tokens, mask, _ = condition_token_sets(str(conditions_path), source['conditions'])
        rs, vs = [], []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                ix = slice(start, start+256)
                context = model.make_context(
                    torch.as_tensor(source['z'][ix], dtype=torch.float32, device=device),
                    torch.as_tensor(values[ix], dtype=torch.float32, device=device),
                    torch.as_tensor(tokens[ix], dtype=torch.float32, device=device),
                    torch.as_tensor(mask[ix], dtype=torch.bool, device=device), time_start=0.)
                rs.append(context.representation.cpu().numpy())
                vs.append(context.velocity.cpu().numpy())
        out[role] = dict(z=source['z'], r=np.concatenate(rs), v=np.concatenate(vs), cell_ids=source['cell_ids'])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--gene-dir')
    parser.add_argument('--conditions')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    fold = Path(args.fold)
    gene_dir = args.gene_dir or args.fold
    conditions = args.conditions or str(fold/'conditions.npz')
    if Path(args.output).exists():
        raise FileExistsError('Use a fresh background output; preserve the legacy artifact')
    data = contexts(fold, gene_dir, conditions, device=args.device, seed=args.seed)
    train = data['train']
    w_v = Ridge(alpha=1.0).fit(train['z'], train['v'])
    w_r = Ridge(alpha=1.0).fit(train['z'], train['r'])
    out = dict(format=np.asarray(BACKGROUND_FORMAT), seed=np.asarray(args.seed),
               estimator=np.asarray('fixed_pretraining_reference_not_online_joint_innovation'))
    for role in ('train', 'validation'):
        z = data[role]['z']
        out[f'{role}_v_bg'] = w_v.predict(z).astype('float32')
        out[f'{role}_r_bg'] = w_r.predict(z).astype('float32')
        out[f'{role}_cell_ids'] = data[role]['cell_ids']
        resid = data[role]['v']-out[f'{role}_v_bg']
        print(f"{role}: v_rms={np.sqrt(np.square(data[role]['v']).mean()):.3f} "
              f"resid_rms={np.sqrt(np.square(resid).mean()):.3f} "
              f"ratio={np.sqrt(np.square(resid).mean())/np.sqrt(np.square(data[role]['v']).mean()):.3f}")
    np.savez_compressed(args.output, **out)
    print('saved', args.output)


if __name__ == '__main__':
    main()
