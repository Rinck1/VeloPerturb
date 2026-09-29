"""Same innovation decomposition as velocity_innovation_diagnostic but for the
GFG decoder-JVP velocity (the one the main instrument uses)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.artifacts import load_config
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.probes import permute_local
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.velocity_diagnostics import centered_velocity_r2
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


def direction_cos(z_src, v_src, z_tgt, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(z_tgt))).fit(z_tgt)
    idx = nn.kneighbors(z_src, return_distance=False)
    d = z_tgt[idx] - z_src[:, None, :]
    d = d/np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
    d = d.mean(1)
    d = d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    v = v_src/np.maximum(np.linalg.norm(v_src, axis=1, keepdims=True), 1e-9)
    return (v*d).sum(1)


def gfg_velocity(fold, gene_dir, conditions_path, *, device='cpu', seed=0):
    import torch
    from veloroute.contracts import FitScope
    from veloroute.full_experiments import condition_token_sets
    from veloroute.full_model import FullConfig, FullVeloRoute
    from veloroute.gfg_experiments import read_gene_input
    kang = load_config('configs/veloroute_kang_20260914.yaml')
    base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    transform = FrozenSplicingTransform.load(Path(fold)/'transform.npz')
    with np.load(conditions_path, allow_pickle=False) as data:
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
    out = {}
    for role in ('train', 'validation'):
        values, source, sm = read_gene_input(Path(gene_dir)/f'{role}_gene_source.npz',
                                             Path(fold)/f'{role}_source.npz')
        if role == 'train':
            model.dynamics.prepare(torch.as_tensor(values, dtype=torch.float32, device=device),
                torch.as_tensor(transform.components, dtype=torch.float32, device=device),
                cell_ids=list(source['cell_ids']), scope=FitScope(frozenset(source['cell_ids'])))
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
        out[role] = (source, np.concatenate(chunks))
    return out


def analyze(tag, fold, gene_dir, conditions):
    print(f'=== {tag} (GFG velocity)', flush=True)
    data = gfg_velocity(fold, gene_dir, conditions)
    train_source, train_v = data['train']
    background = Ridge(alpha=1.).fit(train_source['z'], train_v)
    for role, (source, v) in data.items():
        target, tm = load_pack(Path(fold)/f'{role}_target.npz', expected_side='target')
        pred = np.zeros_like(v)
        if role == 'train':
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            for tr, te in kf.split(source['z']):
                pred[te] = Ridge(alpha=1.0).fit(source['z'][tr], v[tr]).predict(source['z'][te])
        else:
            pred = background.predict(source['z'])
        innov = v-pred
        r2 = centered_velocity_r2(v, pred)
        print(f'  {role}: z-redundancy R2={r2:.3f} |v|rms={float(np.sqrt(np.square(v).mean())):.3f}', flush=True)
        for cond in sorted(set(source['conditions'])):
            ms = np.asarray(source['conditions'] == cond)
            mt = np.asarray(target['conditions'] == cond)
            if ms.sum() < 20 or mt.sum() < 20:
                continue
            print(f'    {role:10s} {str(cond):12s} n={int(ms.sum()):4d} '
                  f'cos(v)={direction_cos(source["z"][ms], v[ms], target["z"][mt]).mean():+.3f} '
                  f'cos(E[v|z])={direction_cos(source["z"][ms], pred[ms], target["z"][mt]).mean():+.3f} '
                  f'cos(innov)={direction_cos(source["z"][ms], innov[ms], target["z"][mt]).mean():+.3f}', flush=True)


if __name__ == '__main__':
    analyze('RENGE', '/home/yuchang/wangjiaxuan/outputs/veloroute_real_pipeline_20260912_v2/fold',
            '/home/yuchang/wangjiaxuan/outputs/veloroute_gfg_inputs_20260914',
            'data/renge/conditions/esm2_3b_v1/conditions.npz')
    analyze('pancreas', '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways',
            '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways',
            '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways/conditions.npz')
