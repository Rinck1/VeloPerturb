"""Audit the existing denoising checkpoint in its native log-CPM format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from veloroute.artifacts import Run, save_csv, save_json, sha256
from veloroute.gpu_policy import enforce_gpu_policy
from w3_7_native_train import raw_variants, velocity_comparison

FOUNDATION = Path('/home/yuchang/GFG_velocity_foundation_mousebrain_v1')
CHECKPOINT = FOUNDATION / 'experiments/foundation_pilot_20260830/gpu6_denoise_pretrain/pretrain_final.pt'


def full_library_log_cpm(counts, full_library):
    counts = np.asarray(counts, dtype=np.float32)
    full_library = np.asarray(full_library, dtype=np.float32)
    if full_library.shape != (len(counts),) or not np.isfinite(counts).all() or (counts < 0).any() or (full_library < 0).any():
        raise ValueError('Invalid raw counts or source library depths')
    if (counts.sum(1) > full_library + np.maximum(1e-3, full_library * 1e-6)).any():
        raise ValueError('Selected counts exceed full source library')
    return np.log1p(counts * (1e4 / np.maximum(full_library, 1))[:, None])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True, choices=['renge', 'merlin'])
    ap.add_argument('--output', required=True); args = ap.parse_args()
    enforce_gpu_policy('cuda:0')
    sys.path.insert(0, str(Path('tools/w3-scvelo').resolve())); sys.path.insert(0, str(FOUNDATION))
    import anndata as ad
    import torch
    from foundation_pilot import PilotModelSpec, build_model
    from w3_7_native_train import infer
    root = Path('/data/yuchang/veloroute_w3_20261014/native_gfg') / args.dataset
    ready = json.loads((root / 'input_v1/ready.json').read_text()); source = Path(ready['path'])
    if sha256(source) != ready['sha256']:
        raise ValueError('Frozen source artifact changed')
    output_data = root / 'foundation_zero_v1'
    if output_data.exists():
        raise ValueError('Refusing to overwrite foundation audit')
    c = dict(dataset=args.dataset, checkpoint=str(CHECKPOINT), source_sha256=ready['sha256'], seed=0,
        batch_size=8, variants=['V0_real', 'V3_local_U', 'V4_global_U', 'V7_zero_U'], control_seed=20261014,
        preprocessing='log1p CPM1e4 using full pre-HVG source library depth; no moments, no z-score',
        gene_panel='2000 target-training HVGs, shared per-gene model, no identity vocabulary in this checkpoint',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    start = time.monotonic()
    with Run(args.output, stage='W3_7_foundation_native_zero_shot_U_audit', kind='development', config=c,
             inputs=[Path(__file__), Path(__file__).with_name('w3_7_native_train.py'),
                     Path(__file__).with_name('w3_validation.py'), CHECKPOINT, source,
                     FOUNDATION / 'foundation_pilot.py', *sorted((FOUNDATION / 'model').glob('*.py'))], seed=0) as run:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        if payload['variant'] != 'denoise':
            raise ValueError('Expected registered denoising checkpoint')
        weights = payload['model_state']
        vkeys = [k for k in weights if k.startswith('velocity_encoder.')]
        copied = bool(vkeys) and all(torch.equal(weights[k], weights['manifold_encoder.' + k[len('velocity_encoder.'):]]) for k in vkeys)
        save_json(run.directory / 'checkpoint_identity.json', dict(checkpoint_sha256=sha256(CHECKPOINT),
            variant=payload['variant'], model_spec=payload['model_spec'], training_steps=payload['step'],
            actual_cells_seen=payload['cells_seen'], velocity_encoder_equals_state_encoder=copied,
            source_code_saves_state_encoder_copy_into_velocity=True,
            directly_velocity_supervised=False, gene_identity_vocabulary_present=False))
        adata = ad.read_h5ad(source)
        model, cfg = build_model(adata.n_vars, PilotModelSpec(**payload['model_spec']), seed=0)
        model.load_state_dict(weights, strict=True); model.to('cuda:0').eval()
        output_data.mkdir(parents=True, exist_ok=False)
        u = adata.layers['raw_U'].toarray().astype(float)
        s = adata.layers['raw_S'].toarray().astype(float)
        strata = adata.obs['condition'].astype(str).to_numpy() if args.dataset == 'renge' else np.repeat('MeRLin_Day0', adata.n_obs)
        variants, audit = raw_variants(u, s, adata.obsm['X_pca'], strata, seed=c['control_seed'], neighbors=30)
        save_json(run.directory / 'controls.json', audit)
        uu = adata.obs['initial_size_unspliced'].to_numpy()
        ss = adata.obs['initial_size_spliced'].to_numpy()
        state = full_library_log_cpm(s, ss)
        velocity = {}; rows = []
        for name in c['variants']:
            save_json(run.directory / 'status.json', dict(status='inferencing', variant=name))
            velocity[name] = infer(model, full_library_log_cpm(variants[name], uu), state, c['batch_size'], 'cuda:0')
            rows.append(dict(dataset=args.dataset, variant=name, **velocity_comparison(velocity['V0_real'], velocity[name])))
        np.savez_compressed(output_data / 'train_source_velocities.npz', **velocity,
            source_ids=adata.obs_names.to_numpy(dtype=str), genes=adata.var_names.to_numpy(dtype=str))
        result = dict(status='FOUNDATION_ZERO_SHOT_U_AUDIT_COMPLETE', dataset=args.dataset, sensitivity=rows,
            checkpoint_sha256=sha256(CHECKPOINT), velocity_encoder_equals_state_encoder=copied,
            input_genes=adata.n_vars, input_cells=adata.n_obs, directly_velocity_supervised=False,
            frozen_model=True, downstream_increment_evaluated=False, runtime_seconds=time.monotonic()-start)
        save_csv(run.directory / 'U_sensitivity.csv', rows); save_json(run.directory / 'summary.json', result)
        save_json(output_data / 'ready.json', result); save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# Foundation checkpoint U dependence\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
