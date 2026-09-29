"""Exploratory gain screen: do velocity/U features help predict a held-out condition's target?

Same probe family as probes.run_g1 (OT pseudo-pairs, ridge map, empirical residual resampling)
but dataset-agnostic and explicitly outside the preregistered ER-short runner.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.coupling import sinkhorn_log
from veloroute.latent import load_pack
from veloroute.metrics import energy_distance, paired_condition_bootstrap
from veloroute.probes import _ridge_fit, _ridge_predict, arm_features, crossfit_u


def check_packs(parts, config):
    metas = {key: m for key, (p, m) in parts.items()}
    hashes = {m.get('transform_hash') for m in metas.values()}
    if len(hashes) != 1 or None in hashes:
        raise ValueError('Transform hash mismatch across probe packs')
    if any(m.get('kind') != 'exploratory_gain_probe' for m in metas.values()):
        raise ValueError('Probe packs must declare kind exploratory_gain_probe')
    source_train, target_train = parts[('train', 'source')], parts[('train', 'target')]
    source_valid, target_valid = parts[('validation', 'source')], parts[('validation', 'target')]
    for role in ('train', 'validation'):
        s, t = parts[(role, 'source')][0], parts[(role, 'target')][0]
        if set(s['conditions']) != set(t['conditions']):
            raise ValueError('Source/target condition coverage differs')
    if set(source_train[0]['conditions']) & set(source_valid[0]['conditions']):
        raise ValueError('Held-out conditions overlap training')
    # Consecutive transitions legitimately share boundary-day cells across roles,
    # but no held-out cell may appear anywhere in the fit split.
    train_ids = set(source_train[0]['cell_ids']) | set(target_train[0]['cell_ids'])
    valid_ids = set(source_valid[0]['cell_ids']) | set(target_valid[0]['cell_ids'])
    if train_ids & valid_ids:
        raise ValueError('Held-out cells leaked into the fit split')
    return source_train[0], target_train[0], source_valid[0], target_valid[0]


def _fit_dist_torch(x, y, xv, *, steps, lr, batch, seed):
    """Distribution-matching linear map: minimize energy distance between pushed
    source batch and target batch. No pairing anywhere."""
    import torch
    torch.manual_seed(seed)
    X = torch.as_tensor(x, dtype=torch.float32)
    Y = torch.as_tensor(y, dtype=torch.float32)
    Xv = torch.as_tensor(xv, dtype=torch.float32)
    W = torch.zeros(X.shape[1], Y.shape[1], requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        si = torch.as_tensor(rng.choice(len(X), batch, replace=False))
        ti = torch.as_tensor(rng.choice(len(Y), batch, replace=False))
        pred = X[si] @ W
        tgt = Y[ti]
        loss = 2*torch.cdist(pred, tgt).mean() - torch.cdist(pred, pred).mean() - torch.cdist(tgt, tgt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (Xv @ W).numpy()


def _smooth_velocity(pack, k, seed):
    """k-nearest-neighbor velocity mean (self included), neighbors in z space."""
    from sklearn.neighbors import NearestNeighbors
    z = pack['z']
    nn = NearestNeighbors(n_neighbors=k).fit(z)
    idx = nn.kneighbors(return_distance=False)
    pack['velocity'] = pack['velocity'][idx].mean(1).astype('float32')


def run_probe(fold, output, config_path):
    config = load_config(config_path)
    if config.get('research_status') != 'exploratory_gain_screen_not_preregistered':
        raise ValueError('Gain probe requires explicit exploratory status')
    seeds = config['seeds']
    if len(set(seeds)) < 3:
        raise ValueError('Gain probe requires three seeds')
    parts = {}
    for role in ('train', 'validation'):
        for side in ('source', 'target'):
            path = Path(fold)/f'{role}_{side}.npz'
            parts[(role, side)] = load_pack(path, expected_side=side)[:2]
    source, target, valid, future = check_packs(parts, config)
    # Condition embeddings: frozen one-hot prepared with the packs.
    with np.load(Path(fold)/'conditions.npz', allow_pickle=False) as data:
        labels, vectors = data['conditions'].astype(str), data['embeddings'].astype(np.float32)
    lookup = dict(zip(labels, vectors))
    for condition in set(source['conditions'])|set(valid['conditions']):
        if condition not in lookup:
            raise ValueError(f'Missing condition embedding: {condition}')
    projection = np.random.default_rng(config['condition_projection_seed']).normal(
        size=(vectors.shape[1], config['condition_projection_dim']))/np.sqrt(config['condition_projection_dim'])
    e = np.stack([lookup[c] for c in source['conditions']]) @ projection
    ev = np.stack([lookup[c] for c in valid['conditions']]) @ projection
    valid = {**valid, 'u_predicted': _ridge_predict(
        _ridge_fit(source['z'], source['u_features'], config['ridge_alpha']), valid['z'])}
    validation_conditions = sorted(str(c) for c in set(valid['conditions']))
    with Run(output, stage='exploratory_gain_probe', kind='engineering',
             config={**config, 'fold': str(fold), 'validation_conditions': validation_conditions},
             inputs=[Path(fold)/name for name in ('train_source.npz', 'train_target.npz',
                 'validation_source.npz', 'validation_target.npz', 'conditions.npz')]) as run:
        torch.set_num_threads(2)
        frozen_predictions = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            paired_source, paired_target = [], []
            for condition in sorted(set(source['conditions'])):
                si = np.flatnonzero(source['conditions'] == condition)
                ti = np.flatnonzero(target['conditions'] == condition)
                si = rng.choice(si, min(len(si), config['training_cells_per_condition']), replace=False)
                ti = rng.choice(ti, min(len(ti), config['training_cells_per_condition']), replace=False)
                cost = torch.cdist(torch.tensor(source['z'][si]), torch.tensor(target['z'][ti])).square()
                plan = sinkhorn_log(cost/cost.mean().clamp_min(1e-8), epsilon=config['ot_epsilon']).numpy()
                flat = rng.choice(plan.size, config['training_cells_per_condition'], p=(plan/plan.sum()).ravel())
                paired_source.extend(si[flat//len(ti)])
                paired_target.extend(ti[flat % len(ti)])
            ix, iy = np.array(paired_source), np.array(paired_target)
            u_crossfit = crossfit_u(source, seed, config['ridge_alpha'])
            dist_mode = config.get('fit_mode') == 'distribution_matching'
            if int(config.get('velocity_smoothing_k', 0)) > 1:
                _smooth_velocity(source, int(config['velocity_smoothing_k']), seed)
                _smooth_velocity(valid, int(config['velocity_smoothing_k']), seed)
            for arm in config['arms']:
                x = arm_features(source, e, arm, seed=seed, neighbors=config['permutation_block_size'], crossfit=u_crossfit)
                xv = arm_features(valid, ev, arm, seed=seed, neighbors=config['permutation_block_size'])
                if dist_mode:
                    prediction = _fit_dist_torch(x, target['z'], xv, steps=config['dist_steps'],
                        lr=config['dist_lr'], batch=config['dist_batch'], seed=seed)
                else:
                    fit = _ridge_fit(x[ix], target['z'][iy], config['ridge_alpha'])
                    residual = target['z'][iy] - _ridge_predict(fit, x[ix])
                    residual_rng = np.random.default_rng(seed)
                    prediction = _ridge_predict(fit, xv) + residual[residual_rng.integers(0, len(residual), len(xv))]
                path = run.directory/f'{arm}_seed{seed}_predictions.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, z=prediction, conditions=valid['conditions'])
                frozen_predictions.append({'arm': arm, 'seed': seed, 'path': str(path), 'sha256': sha256(path)})
        save_json(run.directory/'frozen_predictions.json', frozen_predictions)
        rows = []
        for record in frozen_predictions:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Prediction modified before evaluation')
            with np.load(record['path'], allow_pickle=False) as data:
                predictions, conditions = data['z'], data['conditions']
            for condition in validation_conditions:
                p, y = predictions[conditions == condition], future['z'][future['conditions'] == condition]
                rows.append({'condition': condition, 'seed': record['seed'], 'arm': record['arm'],
                             'energy_distance': energy_distance(p, y),
                             'variance_ratio': float(p.var(0).sum()/max(y.var(0).sum(), 1e-12))})
        save_csv(run.directory/'metrics.csv', rows)
        comparisons = []
        for real, control in [('A1', 'A0'), ('A1', 'A2'), ('B1', 'A0'), ('B1', 'B2'), ('B1', 'B3')]:
            comparisons.append(paired_condition_bootstrap(rows, real_arm=real, control_arm=control,
                seeds=seeds, conditions=validation_conditions,
                n_bootstrap=config['n_bootstrap'], seed=config['bootstrap_seed']))
        save_json(run.directory/'comparisons.json', comparisons)
        result = dict(status='EXPLORATORY_GAIN_PROBE_COMPLETE', fold=str(fold),
            validation_conditions=validation_conditions, arms=config['arms'],
            comparisons=comparisons, formal_G1='NOT_RUN', confirmation='SEALED',
            individual_fate_claim=False, note='Exploratory screen only; not an information bound.')
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Exploratory velocity gain probe\n\n'+json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--config', default='configs/veloroute_gain_probe_20260915.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(Path(args.fold), Path(args.output), args.config), indent=2))
