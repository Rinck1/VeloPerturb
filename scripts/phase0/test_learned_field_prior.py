"""Did the trained model learn to use the condition-adaptive field from velocity?

For a trained checkpoint, predict held-out conditions with three U variants:
real / zeroed / globally permuted. Compare predicted displacement direction
alignment against the true direction. If real > zero/permuted, the model uses
the condition's own velocity information.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.artifacts import load_config
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import load_full_checkpoint
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import load_pack
from sklearn.neighbors import NearestNeighbors


def true_direction(z_src, z_tgt, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(z_tgt))).fit(z_tgt)
    idx = nn.kneighbors(z_src, return_distance=False)
    d = z_tgt[idx] - z_src[:, None, :]
    d = d/np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
    return d.mean(1)


def unit(v):
    return v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def run(fold, gene_dir, conditions, checkpoint, output):
    device = 'cuda'
    torch.set_num_threads(4)
    values, source, sm = read_gene_input(Path(gene_dir)/'validation_gene_source.npz',
                                         Path(fold)/'validation_source.npz')
    target, tm = load_pack(Path(fold)/'validation_target.npz', expected_side='target')
    model, metadata = load_full_checkpoint(checkpoint, map_location=device)
    config = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    config.update(conditions=str(conditions))
    tokens, mask, _ = condition_token_sets(config['conditions'], source['conditions'])
    rng = np.random.default_rng(0)
    n_genes = values.shape[1]//2
    variants = {
        'real': values.copy(),
        'zero_u': values.copy(),
        'perm_u': values.copy(),
    }
    variants['zero_u'][:, :n_genes] = 0.
    variants['perm_u'][:, :n_genes] = values[rng.permutation(len(values)), :n_genes]
    results = {}
    for name, v in variants.items():
        zs = []
        with torch.no_grad():
            for start in range(0, len(v), 64):
                ix = slice(start, start+64)
                z0 = torch.as_tensor(source['z'][ix], dtype=torch.float32, device=device)
                us = torch.as_tensor(v[ix], dtype=torch.float32, device=device)
                token = torch.as_tensor(tokens[ix], dtype=torch.float32, device=device)
                padding = torch.as_tensor(mask[ix], dtype=torch.bool, device=device)
                pred = model.predict(z0, us, token, padding, t0=0., t1=1., max_step=.25,
                                     stochastic=False)
                zs.append(pred.endpoint.cpu().numpy())
        endpoint = np.concatenate(zs)
        for condition in sorted(set(source['conditions'])):
            ms = np.asarray(source['conditions'] == condition)
            mt = np.asarray(target['conditions'] == condition)
            if ms.sum() < 20 or mt.sum() < 20:
                continue
            disp = unit(endpoint[ms]-source['z'][ms]).mean(0)
            truth = true_direction(source['z'][ms], target['z'][mt]).mean(0)
            cos = float(disp @ truth/max(np.linalg.norm(disp)*np.linalg.norm(truth), 1e-9))
            results.setdefault(condition, {})[name] = cos
    for condition, row in results.items():
        print(condition, json.dumps({k: round(v, 4) for k, v in row.items()}))
    Path(output).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--gene-dir', default=None)
    parser.add_argument('--conditions', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.fold, args.gene_dir or args.fold, args.conditions or str(Path(args.fold)/'conditions.npz'),
        args.checkpoint, args.output)
