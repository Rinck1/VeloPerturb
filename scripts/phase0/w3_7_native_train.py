"""Call native GFG fit unchanged, then audit U dependence with a frozen S graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy import sparse

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.gpu_policy import enforce_gpu_policy
from w3_7_native_prepare import native_imports, scoped_artifact_root
from w3_validation import local_u_control


def row_normalization(raw, normalized):
    raw, normalized = np.asarray(raw, float), np.asarray(normalized, float)
    if raw.shape != normalized.shape or (raw < 0).any():
        raise ValueError('Invalid frozen normalization inputs')
    size, after = raw.sum(1), normalized.sum(1)
    factor = np.divide(after, size, out=np.ones(len(size)), where=size > 0)
    if not np.allclose(raw * factor[:, None], normalized, rtol=1e-5, atol=1e-5):
        raise ValueError('Native preprocessing is not the expected per-cell linear normalization')
    return factor


def raw_variants(u, s, state, strata, *, seed, neighbors):
    variants = {'V0_real': u.copy(), 'V7_zero_U': np.zeros_like(u)}
    local, order, audit = local_u_control(np.concatenate((u, s), axis=1), state, strata,
        neighbors=neighbors, seed=seed, match_depth=True)
    variants['V3_local_U'] = local[:, :u.shape[1]]
    # Global bijective rotation among positive-depth cells; preserve zero cells.
    depth = u.sum(1); positive = np.flatnonzero(depth > 0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(positive); global_order = np.arange(len(u))
    if len(perm) > 1:
        global_order[perm] = np.roll(perm, 1)
    swapped = u[global_order].copy()
    factor = np.divide(depth, depth[global_order], out=np.ones(len(u)), where=depth[global_order] > 0)
    variants['V4_global_U'] = swapped * factor[:, None]
    for name in ('V3_local_U', 'V4_global_U'):
        if not np.allclose(variants[name].sum(1), depth, atol=1e-5, rtol=1e-6):
            raise ValueError('Control failed to preserve recipient selected-gene U depth')
    return variants, dict(local=audit, local_order=order.tolist(), global_order=global_order.tolist(),
        global_moved_fraction=float(np.mean(global_order != np.arange(len(u)))),
        depth_scope='selected training gene panel before frozen library normalization')


def velocity_comparison(reference, candidate):
    a, b = np.asarray(reference, float), np.asarray(candidate, float)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    good = (na > 1e-10) & (nb > 1e-10)
    return dict(cos_median=float(np.median(np.sum(a[good] * b[good], 1) / (na[good] * nb[good]))) if good.any() else None,
        norm_ratio_median=float(np.median(nb[na > 1e-10] / na[na > 1e-10])) if (na > 1e-10).any() else None,
        valid_direction_fraction=float(good.mean()), relative_l2=float(np.linalg.norm(b-a) / max(np.linalg.norm(a), 1e-12)))


def infer(model, u, s, batch_size, device):
    import torch
    out = []
    for start in range(0, len(u), batch_size):
        inp = torch.as_tensor(np.concatenate((u[start:start+batch_size], s[start:start+batch_size]), axis=1), dtype=torch.float32)
        _, velocity = model.predict(inp, device=device)
        values = velocity.detach().cpu().numpy()[:, model.G:]
        if not np.isfinite(values).all():
            raise ArithmeticError('Nonfinite native velocity')
        out.append(values)
    return np.concatenate(out)


def train_one(c, dataset_name, seed, output, smoke=False):
    import anndata as ad
    import torch
    from torch.utils.data import DataLoader
    from model.Config import Config
    from model.dataset import Data
    from model.graph import GraphBatchSampler, build_root_directed_adjacency
    from model.model import VeloModel
    artifact_root = scoped_artifact_root(c) / dataset_name
    ready = json.loads((artifact_root / 'input_v1/ready.json').read_text())
    path = Path(ready['path'])
    if sha256(path) != ready['sha256'] or ready['dataset'] != dataset_name:
        raise ValueError('Prepared source artifact changed')
    adata = ad.read_h5ad(path)
    if smoke:
        adata = adata[:64].copy()
    native = Path(c['native_root'])
    native_files = sorted(native.glob('model/*.py')) + sorted(native.glob('tool/*.py')) + [native / 'preprocessing.py']
    destination = artifact_root / ('smoke_v1' if smoke else f'seed{seed}_v1')
    if destination.exists():
        raise ValueError('Refusing to overwrite native training artifacts')
    start = time.monotonic()
    config = dict(c, dataset=dataset_name, seed=seed, smoke=smoke,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        prepared_sha256=ready['sha256'])
    with Run(output, stage='W3_7_native_target_fit_and_U_sensitivity', kind='engineering' if smoke else 'development',
             config=config, inputs=[path, Path(__file__), Path(__file__).with_name('w3_7_native_prepare.py'),
                                    Path(__file__).with_name('w3_validation.py'), *native_files], seed=seed) as run:
        destination.mkdir(parents=True, exist_ok=False)
        native_hashes = {str(p): sha256(p) for p in native_files}
        save_json(run.directory / 'native_code.json', native_hashes)
        cfg = Config(adata, seed=seed)
        if smoke:
            cfg.config['train']['num_epochs'] = 1
            cfg.config['train']['batch_size'] = 4
        ds = Data(adata, spliced_key='Ms', unspliced_key='Mu')
        adjacency = adata.obsp['connectivities']
        directed, root_distance = build_root_directed_adjacency(adjacency, ds.root_mask)
        sampler = GraphBatchSampler(adjacency, batch_size=cfg.config['train']['batch_size'], seed=seed)
        loader = DataLoader(ds, batch_sampler=sampler)
        model = VeloModel(loader, cfg, device='cuda:0', cluster_edge=None)
        count = [0]
        def watch(module, inputs, outputs):
            if not module.training:
                return
            for value in (outputs[0], outputs[1], outputs[2][0]['L_ode']):
                if not torch.isfinite(value).all():
                    raise ArithmeticError('Nonfinite native reconstruction/velocity/ODE loss')
            count[0] += 1
            if count[0] % 5 == 0 or count[0] == 1:
                save_json(run.directory / 'status.json', dict(status='training', batches_forward=count[0],
                    epoch_budget=cfg.config['train']['num_epochs'], batches_per_epoch=len(loader),
                    native_ode_loss=float(outputs[2][0]['L_ode'].detach())))
        hook = model.register_forward_hook(watch)
        save_json(run.directory / 'native_config.json', cfg.config)
        model.fit(adjacency_matrix=adjacency, directed_adjacency_matrix=directed, device='cuda:0', save_path=str(destination))
        hook.remove()
        if any(not np.isfinite(v).all() for v in cfg.history.values()):
            raise ArithmeticError('Nonfinite native epoch losses')
        if any(sha256(p) != native_hashes[str(p)] for p in native_files):
            raise ValueError('External native code changed during the fit')
        np.savez_compressed(destination / 'normalization.npz',
            u_mean=ds.unspliced_state['mean'], u_std=ds.unspliced_state['std'],
            s_mean=ds.spliced_state['mean'], s_std=ds.spliced_state['std'], genes=adata.var_names.to_numpy(dtype=str))
        save_json(run.directory / 'epoch_losses.json', cfg.history)
        rows = []
        if not smoke:
            save_json(run.directory / 'status.json', dict(status='auditing_U_sensitivity'))
            u = adata.layers['raw_U'].toarray().astype(float)
            s = adata.layers['raw_S'].toarray().astype(float)
            normalized_u = sparse.csr_matrix(adata.layers['unspliced']).toarray()
            ufactor = row_normalization(u, normalized_u)
            operator = adata.obsp['w3_moment_operator']
            ms = np.asarray(adata.layers['Ms'], dtype=np.float32)
            strata = (adata.obs['condition'].astype(str).to_numpy() if dataset_name == 'renge'
                      else np.repeat('MeRLin_Day0', adata.n_obs))
            variants, audit = raw_variants(u, s, adata.obsm['X_pca'], strata,
                seed=c['control_seed'], neighbors=c['local_neighbors'])
            audit['strata'] = 'RENGE condition' if dataset_name == 'renge' else 'MeRLin Day0 sample; clone identity not a stratum'
            save_json(run.directory / 'controls.json', audit)
            mu0 = np.asarray(adata.layers['Mu'], dtype=np.float32)
            gamma = np.sum(mu0 * ms, axis=0) / np.maximum(np.sum(ms * ms, axis=0), 1e-12)
            velocities, analytic = {}, {}
            for name in c['variants']:
                mu = (operator @ sparse.csr_matrix(variants[name] * ufactor[:, None])).toarray().astype(np.float32)
                if name == 'V0_real' and not np.allclose(mu, mu0, rtol=1e-5, atol=1e-5):
                    raise ValueError('V0 recomputed moments differ from original training input')
                velocities[name] = infer(model, mu, ms, cfg.config['train']['batch_size'], 'cuda:0')
                analytic[name] = mu - gamma * ms
            for name in c['variants']:
                rows.append(dict(dataset=dataset_name, seed=seed, variant=name,
                    **velocity_comparison(velocities['V0_real'], velocities[name]),
                    analytic_cos_median=velocity_comparison(analytic['V0_real'], analytic[name])['cos_median']))
            np.savez_compressed(destination / 'train_source_velocities.npz', **velocities,
                source_ids=adata.obs_names.to_numpy(dtype=str), genes=adata.var_names.to_numpy(dtype=str), gamma=gamma)
            save_csv(run.directory / 'U_sensitivity.csv', rows)
        checkpoint = destination / 'final.pth'
        result = dict(status='NATIVE_SMOKE_PASSED' if smoke else 'NATIVE_TRAINING_AND_U_AUDIT_COMPLETE',
            dataset=dataset_name, seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            cells=adata.n_obs, genes=adata.n_vars, native_config=cfg.config,
            roots=int(ds.root_mask.sum()), unreachable_root_distances=int(np.isnan(root_distance).sum()),
            parameter_count=sum(p.numel() for p in model.parameters()), sensitivity=rows,
            downstream_increment_evaluated=False, sensitivity_scope='training source, frozen S graph and train normalizers',
            runtime_seconds=time.monotonic()-start)
        save_json(run.directory / 'summary.json', result)
        save_json(destination / 'ready.json', result)
        save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# Native GFG target retraining\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_7_native_gfg_20261014.json')
    ap.add_argument('--dataset', required=True, choices=['renge', 'merlin'])
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); c = load_config(args.config)
    enforce_gpu_policy('cuda:0'); native_imports(c)
    seeds = [int(x) for x in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(c['training_seeds']):
        raise ValueError('Seeds must be unique registered training seeds')
    if args.smoke and len(seeds) != 1:
        raise ValueError('Smoke is one bounded native training run')
    if not args.smoke:
        gate = scoped_artifact_root(c) / 'renge/smoke_v1/ready.json'
        if not gate.exists() or json.loads(gate.read_text())['status'] != 'NATIVE_SMOKE_PASSED':
            raise ValueError('Native GPU smoke has not passed')
    for seed in seeds:
        train_one(c, args.dataset, seed, Path(args.output) / f'seed{seed}', smoke=args.smoke)


if __name__ == '__main__':
    main()
