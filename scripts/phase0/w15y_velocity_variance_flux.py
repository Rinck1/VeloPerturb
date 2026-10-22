"""Train-KO-only probe of signed velocity/state covariance and future spread."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts
import w15x_u_increment_screen as increment

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('S', 'velocity_positive', 'velocity_free', 'velocity_reverse', 'direct_U', 'shuffle_0', 'shuffle_1', 'shuffle_2')


def covariance_diagonal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.mean((a-a.mean(0))*(b-b.mean(0)), axis=0)


def stratified_permutation(ids, seed, identity):
    libraries = np.array([str(i).split(':', 1)[0] for i in ids])
    order = np.arange(len(ids))
    for lib in sorted(set(libraries)):
        ix = np.flatnonzero(libraries == lib)
        if len(ix) > 1:
            order[ix] = ix[increment.derangement(len(ix), increment.keyed_seed(seed, identity+'|'+lib))]
    return order


def load_data(plan):
    from veloroute.latent import load_pack
    xp = json.loads(Path(plan['source_configuration']).read_text())
    source_data, c = increment.load_data(xp)
    data = {k: source_data[k] for k in ('labels', 'z0', 'esm', 'technical')}
    source, _ = load_pack(Path(c['fold'])/'train_source.npz', expected_side='source')
    target, _ = load_pack(Path(c['fold'])/'train_target.npz', expected_side='target')
    if set(source['conditions']) != set(data['labels']) or set(target['conditions']) != set(data['labels']):
        raise ValueError('Only original training KOs are allowed')
    local, variance, end_variance, raw_flux = [], [], [], []
    movement = []
    for ko in data['labels']:
        mask = source['conditions'] == ko
        z, v, u = [source[k][mask].astype(float) for k in ('z', 'velocity', 'u_features')]
        target_z = target['z'][target['conditions'] == ko].astype(float)
        var = np.maximum(z.var(0), 1e-8); centered = z-z.mean(0)
        skew = np.mean(centered**3, axis=0)/np.maximum(var**1.5, 1e-8)
        kurt = np.mean(centered**4, axis=0)/np.maximum(var**2, 1e-8)-3.
        local.append(np.stack((z.mean(0), z.mean(0)**2, np.log(var), np.log(var)**2, skew, kurt), axis=-1))
        variance.append(var); end_variance.append(np.maximum(target_z.var(0), 1e-8))
        fields = {'velocity_positive': covariance_diagonal(z, v), 'direct_U': covariance_diagonal(z, u)}
        for seed in plan['shuffle_seeds']:
            order = stratified_permutation(source['cell_ids'][mask], seed, str(ko))
            fields[f'shuffle_{seed}'] = covariance_diagonal(z, v[order])
            movement.append(dict(KO=str(ko), seed=seed, moved_fraction=float(np.mean(order != np.arange(len(order))))))
        raw_flux.append(fields)
    data['local'] = np.asarray(local); data['variance'] = np.asarray(variance)
    data['target_variance'] = np.asarray(end_variance)
    data['y'] = np.log(data['target_variance'])-np.log(data['variance'])
    data['flux'] = {name:np.stack([r[name] for r in raw_flux]) for name in raw_flux[0]}
    return data, c, xp, movement


def predict_fold(data, train, query, plan):
    """Query variance labels are not used until the caller scores predictions."""
    xt, xq = increment.designs(data, train, query, plan)
    names = ['velocity_positive', 'direct_U']+[f'shuffle_{s}' for s in plan['shuffle_seeds']]
    floor = plan['variance_floor_fraction']*data['variance'][train].mean(0)
    features = {name:data['flux'][name]/(data['variance']+floor) for name in names}
    targets = np.stack([data['y'][train]]+[features[n][train] for n in names], axis=-1)
    penalty = np.eye(xt.shape[-1])*plan['ridge']; penalty[0, 0] = 0.
    gram = np.einsum('ngp,ngq->gpq', xt, xt, optimize=True)+penalty[None]
    rhs = np.einsum('ngp,ngt->gpt', xt, targets, optimize=True)
    coef = np.linalg.solve(gram, rhs)
    inverse = np.linalg.pinv(xt.transpose(1, 0, 2), rcond=1e-10)
    coef[:, :, 1:] = np.einsum('gpn,ngt->gpt', inverse, targets[:, :, 1:], optimize=True)
    fitted = np.einsum('ngp,gpt->ngt', xt, coef, optimize=True)
    queried = np.einsum('ngp,gpt->ngt', xq, coef, optimize=True)
    residual_y = data['y'][train]-fitted[:, :, 0]
    prediction, alphas = {'S': queried[:, :, 0]}, {'S': 0.}
    for arm in ARMS:
        if arm == 'S': continue
        name = 'velocity_positive' if arm in ('velocity_free', 'velocity_reverse') else arm
        index = names.index(name)+1
        sign = -1. if arm == 'velocity_reverse' else 1.
        x = sign*(features[name][train]-fitted[:, :, index])
        xquery = sign*(features[name][query]-queried[:, :, index])
        scale2 = float(np.mean(x**2)); raw_scale2 = float(np.var(features[name][train], axis=0).mean())
        alpha = float(np.sum(x*residual_y)/(np.sum(x*x)+plan['ridge']*max(scale2, 1e-12)))
        if scale2 <= max(1e-12, plan['minimum_innovation_fraction']*raw_scale2): alpha = 0.
        if arm != 'velocity_free': alpha = max(alpha, 0.)
        prediction[arm] = queried[:, :, 0]+alpha*xquery; alphas[arm] = alpha
    return prediction, alphas


def program(configuration, output):
    from w7_router_scaling import write_csv
    plan = json.loads(Path(configuration).read_text())
    if tuple(plan['arms']) != ARMS: raise ValueError('Unregistered variance-flux grid')
    xp = json.loads(Path(plan['source_configuration']).read_text())
    c = json.loads(Path(xp['base_configuration']).read_text()); fold = Path(c['fold'])
    inputs = [Path(configuration), Path(__file__), ROOT/'PLANS/w15y_velocity_variance_flux_20261022.md',
        Path(plan['source_configuration']), ROOT/'scripts/phase0/w15x_u_increment_screen.py',
        Path(xp['base_configuration']), Path(xp['training_features']), Path(xp['training_counts']),
        Path(c['conditions']), Path(c['split']), fold/'train_source.npz', fold/'train_target.npz', fold/'transform.npz']
    run = Artifacts(output, seed=plan['seed'], inputs=inputs)
    data, c, xp, movement = load_data(plan)
    rows, coefficients = [], []
    for fold_index, (train, query) in enumerate(increment.split_indices(np.arange(len(data['labels'])), plan['outer_folds'], plan['seed'])):
        prediction, alphas = predict_fold(data, train, query, plan)
        np.savez_compressed(run.directory/f'fold{fold_index}_predictions.npz', query_indices=query, training_indices=train, **prediction)
        for arm, values in prediction.items():
            coefficients.append(dict(fold=fold_index, arm=arm, alpha=alphas[arm]))
            for i, ko in enumerate(data['labels'][query]):
                rows.append(dict(fold=fold_index, KO=str(ko), arm=arm, log_variance_mse=float(np.mean((values[i]-data['y'][query[i]])**2))))
        run.json('status.json', dict(status='outer_fold_complete', completed_outer_folds=fold_index+1))
    labels = list(data['labels']); by = {(r['KO'], r['arm']):r['log_variance_mse'] for r in rows}
    comparisons = []
    for real in ('velocity_positive', 'velocity_free'):
        for control in ARMS:
            if real == control: continue
            comparison = increment.bootstrap_comparison([by[(ko, real)] for ko in labels], [by[(ko, control)] for ko in labels], plan)
            comparisons.append(dict(real=real, control=control, **comparison))
    primary = [r for r in comparisons if r['real'] == 'velocity_positive' and r['control'] in plan['primary_controls']]
    alpha_support = sum(r['alpha'] > 0 for r in coefficients if r['arm'] == 'velocity_positive')
    passed = alpha_support >= 3 and all(r['ci_low'] > 0 for r in primary)
    result = dict(status='TRAINING_KO_VARIANCE_FLUX_COMPLETE', training_KOs=len(labels), axes=data['variance'].shape[1],
        comparisons=comparisons, positive_alpha_folds=alpha_support, variance_signal_screen_passed=passed,
        means={a:float(np.mean([r['log_variance_mse'] for r in rows if r['arm'] == a])) for a in ARMS},
        coefficients=coefficients, shuffled_moved_fraction=float(np.mean([r['moved_fraction'] for r in movement])),
        validation_targets_read=False, test_targets_read=False, GFG_trained=False, router_trained=False,
        multimodality_claim=False, instantaneous_ground_truth_claim=False)
    write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'comparisons.csv', comparisons)
    write_csv(run.directory/'coefficients.csv', coefficients); write_csv(run.directory/'shuffle_movement.csv', movement)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# Training-only velocity variance flux\n\n'+json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--configuration', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); print(json.dumps(program(args.configuration, args.output), indent=2), flush=True)
