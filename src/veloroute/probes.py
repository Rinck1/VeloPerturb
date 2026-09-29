"""Fixed low-cost distribution probes, explicitly distinct from router training.

These probes use expression-OT pseudo-pairs and empirical residual resampling.
A negative result is specific to this estimator/probe, not an information bound.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from .artifacts import Run, load_config, save_csv, save_json, sha256
from .coupling import sinkhorn_log
from .latent import load_pack
from .metrics import energy_distance, paired_condition_bootstrap
from .protocol import protocol_errors


def permute_local(values, source, *, seed, neighbors=10):
    """Bijective negative-control shuffle within condition and nearby S/depth blocks.

The sorting is source-side only and is not fitted into the velocity estimator.
Returns the actual permutation, including singleton/nonmoving rows for audit.
"""
    if neighbors < 2 or len(values) != len(source['z']):
        raise ValueError('Invalid local permutation')
    rng = np.random.default_rng(seed)
    order = np.arange(len(values))
    for condition in sorted(set(source['conditions'])):
        index = np.flatnonzero(source['conditions'] == condition)
        # Coarse depth strata have fixed log2 boundaries, not future quantiles.
        depth_bins = np.floor(np.log2(np.maximum(source['depth'][index], 1))).astype(int)
        for depth_bin in np.unique(depth_bins):
            group = index[depth_bins == depth_bin]
            group = group[np.argsort(source['z'][group, 0], kind='stable')]
            for start in range(0, len(group), neighbors):
                block = group[start:start+neighbors]
                if len(block) > 1:
                    order[block] = np.roll(block, int(rng.integers(1, len(block))))
    return np.asarray(values)[order].copy(), order


def _ridge_fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    penalty = np.eye(design.shape[1])*alpha; penalty[0, 0] = 0
    coef = np.linalg.solve(design.T@design + penalty, design.T@y)
    return mean, scale, coef


def _ridge_predict(fit, x):
    mean, scale, coef = fit
    return np.column_stack((np.ones(len(x)), (x-mean)/scale)) @ coef


def crossfit_u(source, seed, alpha):
    output = np.zeros_like(source['u_features'])
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train, valid in splitter.split(source['z']):
        fit = _ridge_fit(source['z'][train], source['u_features'][train], alpha)
        output[valid] = _ridge_predict(fit, source['z'][valid])
    return output


def arm_features(source, embedding, arm, *, seed, neighbors, crossfit=None):
    z, v, u = source['z'], source['velocity'], source['u_features']
    if arm == 'A0':
        second = np.zeros_like(v)
    elif arm == 'A1':
        second = v
    elif arm == 'A2':
        second = permute_local(v, source, seed=seed, neighbors=neighbors)[0]
    elif arm == 'A3':
        second, z = v, np.zeros_like(z)
    elif arm == 'B1':
        second = u
    elif arm == 'B2':
        second = source['u_predicted'] if crossfit is None else crossfit
    elif arm == 'B3':
        predicted = source['u_predicted'] if crossfit is None else crossfit
        second = predicted + permute_local(u-predicted, source, seed=seed, neighbors=neighbors)[0]
    else:
        raise ValueError('Unknown diagnostic arm')
    return np.concatenate((z, second, embedding), axis=1).astype(np.float64)


def run_g1(frozen_path, train_source_path, train_target_path, validation_source_path, validation_target_path,
           condition_path, output, *, exploratory_config_path=None):
    from .experiments import condition_vectors, validate_pair, validate_real_artifacts
    from .artifacts import object_hash
    exploratory = exploratory_config_path is not None
    if exploratory:
        if frozen_path is not None:
            raise ValueError('Exploratory and formal G1 are distinct runs')
        config = load_config(exploratory_config_path)
        if config.get('research_status') != 'exploratory_not_preregistered' or len(set(config.get('seeds', []))) < 3:
            raise ValueError('Exploratory design must explicitly declare its status and three seeds')
        seeds = config['seeds']
        design_path = exploratory_config_path
    else:
        frozen = json.loads(Path(frozen_path).read_text())
        if object_hash({k: v for k, v in frozen.items() if k != 'freeze_hash'}) != frozen.get('freeze_hash'):
            raise ValueError('Invalid preregistration freeze hash')
        if protocol_errors(frozen['config']):
            raise ValueError('Formal diagnostic prerequisites incomplete')
        for value in frozen['inputs'].values():
            if sha256(value['path']) != value['sha256']:
                raise ValueError('Frozen diagnostic input changed')
        validate_real_artifacts(frozen, train_source_path, train_target_path, condition_path)
        declared = json.loads(Path(frozen['inputs']['data_provenance']['path']).read_text())
        for key, path in [('validation_source_sha256', validation_source_path), ('validation_target_sha256', validation_target_path)]:
            if declared.get(key) != sha256(path):
                raise ValueError(f'Validation artifact not hash-bound: {key}')
        config = load_config(frozen['inputs']['probe_config']['path'])
        seeds, design_path = frozen['config']['seeds'], frozen_path
    source, sm = load_pack(train_source_path, expected_side='source')
    target, tm = load_pack(train_target_path, expected_side='target')
    validate_pair(source, sm, target, tm, 'train')
    valid, vm = load_pack(validation_source_path, expected_side='source')
    if vm.get('role') != 'validation' or sm['kind'] != vm.get('kind'):
        raise ValueError('Development validation role/origin mismatch')
    if not exploratory and (sm['kind'] != 'development' or not vm.get('formal_ready')):
        raise ValueError('G1 accepts only approved development/validation packs')
    if set(valid['conditions']) != set(config['validation_conditions']):
        raise ValueError('Validation condition list must be frozen before diagnostics')
    if set(source['conditions']) & set(valid['conditions']):
        raise ValueError('Held-out conditions overlap training')
    if sm['transform_hash'] != vm.get('transform_hash'):
        raise ValueError('Validation transform mismatch')
    e, _ = condition_vectors(condition_path, source['conditions'])
    ev, _ = condition_vectors(condition_path, valid['conditions'])
    projection_dim = int(config['condition_projection_dim'])
    if not 1 <= projection_dim <= e.shape[1]:
        raise ValueError('Invalid predeclared condition projection size')
    projection = np.random.default_rng(config['condition_projection_seed']).normal(
        size=(e.shape[1], projection_dim)) / np.sqrt(projection_dim)
    e, ev = e @ projection, ev @ projection
    # B2/B3 use the same ridge U|S estimator family in cross-fit training and
    # validation. Do not mix the upstream full-fold auxiliary estimator with it.
    valid = {**valid, 'u_predicted': _ridge_predict(
        _ridge_fit(source['z'], source['u_features'], config['ridge_alpha']), valid['z'])}
    if config['arms'] != ['A0', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3']:
        raise ValueError('All preregistered A/B arms are required (B0 shares A0)')
    with Run(output, stage='exploratory_velocity_AB' if exploratory else 'G1', kind='engineering' if exploratory else 'development', config=config,
             inputs=[design_path, train_source_path, train_target_path, validation_source_path,
                     validation_target_path, condition_path]) as run:
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
                paired_target.extend(ti[flat%len(ti)])
            ix, iy = np.array(paired_source), np.array(paired_target)
            u_crossfit = crossfit_u(source, seed, config['ridge_alpha'])
            for arm in config['arms']:
                x = arm_features(source, e, arm, seed=seed, neighbors=config['permutation_block_size'], crossfit=u_crossfit)
                xv = arm_features(valid, ev, arm, seed=seed, neighbors=config['permutation_block_size'])
                fit = _ridge_fit(x[ix], target['z'][iy], config['ridge_alpha'])
                residual = target['z'][iy] - _ridge_predict(fit, x[ix])
                # Same resampling seed and generation budget for each arm.
                residual_rng = np.random.default_rng(seed)
                prediction = _ridge_predict(fit, xv) + residual[residual_rng.integers(0, len(residual), len(xv))]
                path = run.directory/f'{arm}_seed{seed}_predictions.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, z=prediction, conditions=valid['conditions'])
                frozen_predictions.append({'arm': arm, 'seed': seed, 'path': str(path), 'sha256': sha256(path)})
        save_json(run.directory/'frozen_predictions.json', frozen_predictions)
        # Only now load the independent future validation outcome.
        future, fm = load_pack(validation_target_path, expected_side='target')
        validate_pair(valid, vm, future, fm, 'validation')
        rows = []
        for record in frozen_predictions:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Prediction modified before evaluation')
            with np.load(record['path'], allow_pickle=False) as data:
                predictions, conditions = data['z'], data['conditions']
            for condition in config['validation_conditions']:
                p, y = predictions[conditions == condition], future['z'][future['conditions'] == condition]
                rows.append({'condition': condition, 'seed': record['seed'], 'arm': record['arm'],
                             'energy_distance': energy_distance(p, y),
                             'variance_ratio': float(p.var(0).sum()/max(y.var(0).sum(), 1e-12))})
        save_csv(run.directory/'metrics.csv', rows)
        comparisons = []
        for real, control in [('A1', 'A0'), ('A1', 'A2'), ('B1', 'A0'), ('B1', 'B2'), ('B1', 'B3')]:
            comparisons.append(paired_condition_bootstrap(rows, real_arm=real, control_arm=control,
                seeds=seeds, conditions=config['validation_conditions'],
                n_bootstrap=config['n_bootstrap'], seed=config['bootstrap_seed'],
                confidence=1-config['familywise_alpha']/5))
        save_json(run.directory/'comparisons.json', comparisons)
        if exploratory:
            decision = {'stage': 'exploratory_velocity_AB', 'kind': 'engineering', 'status': 'EXPLORATORY_COMPLETE',
                        'formal_G1': 'NOT_RUN', 'automatic_training_authorized': False,
                        'positive_zero_margin_comparisons': sum(c['ci_low'] > 0 for c in comparisons),
                        'reason': 'User-authorized development analysis before formal QC/effect-margin review. '
                                  'No useful-effect threshold, G1-GO, or confirmation claim is inferred.'}
        else:
            passed = all(c['ci_low'] > frozen['config']['minimum_useful_gain'] for c in comparisons)
            decision = {'stage': 'G1', 'kind': 'development', 'protocol_hash': frozen['freeze_hash'],
                        'status': 'REVIEW_REQUIRED' if passed else 'NO_GO_OR_INSUFFICIENT',
                        'numerical_A_B_gate_passed': passed, 'automatic_training_authorized': False,
                        'reason': 'Needs predeclared kinetic-quality, perturbation-specific/structural review and advisor decision; '
                                  'these fixed OT-paired probes are not an information upper bound.'}
        save_json(run.directory/'decision.json', decision)
        (run.directory/'RESULTS.md').write_text('# G1 diagnostic\n\n'+json.dumps(decision, indent=2)
            +'\n\n'+json.dumps(comparisons, indent=2)+'\n')
    return decision
