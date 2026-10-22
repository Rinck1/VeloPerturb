"""Same signed variance-flux probe on the existing 0 -> 6h training fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts
import w15x_u_increment_screen as increment
import w15y_velocity_variance_flux as variance

ROOT = Path(__file__).resolve().parents[2]


def load_training_data(plan):
    from veloroute.latent import load_pack
    split = json.loads(Path(plan['split']).read_text())
    source, sm = load_pack(Path(plan['fold'])/'train_source.npz', expected_side='source')
    target, tm = load_pack(Path(plan['fold'])/'train_target.npz', expected_side='target')
    labels = np.asarray(sorted(split['train']), dtype=str)
    if set(source['conditions']) != set(labels) or set(target['conditions']) != set(labels):
        raise ValueError('Only registered training KOs are allowed')
    if (sm.get('day'), tm.get('day')) != (plan['source_hours'], plan['target_hours']):
        raise ValueError('Wrong short-window source/target time')
    with np.load(plan['conditions'], allow_pickle=False) as f:
        embeddings = {str(k):v for k, v in zip(f['conditions'], f['embeddings'])}
    libraries = np.array([str(v).split(':', 1)[0] for v in source['cell_ids']])
    library_names = sorted(set(libraries))
    local, starts, ends, means, technical, flux, movement = [], [], [], [], [], [], []
    for ko in labels:
        mask = source['conditions'] == ko
        z, v, u = [source[k][mask].astype(float) for k in ('z', 'velocity', 'u_features')]
        z1 = target['z'][target['conditions'] == ko].astype(float)
        var = np.maximum(z.var(0), 1e-8); centered = z-z.mean(0)
        skew = np.mean(centered**3, axis=0)/np.maximum(var**1.5, 1e-8)
        kurt = np.mean(centered**4, axis=0)/np.maximum(var**2, 1e-8)-3.
        local.append(np.stack((z.mean(0), z.mean(0)**2, np.log(var), np.log(var)**2, skew, kurt), axis=-1))
        starts.append(var); ends.append(np.maximum(z1.var(0), 1e-8)); means.append(z.mean(0))
        technical.append([np.log(source['depth'][mask]).mean(), np.log(mask.sum())]+
                         [float(np.mean(libraries[mask] == lib)) for lib in library_names[:2]])
        values = dict(velocity_positive=variance.covariance_diagonal(z, v), direct_U=variance.covariance_diagonal(z, u))
        for seed in plan['shuffle_seeds']:
            order = variance.stratified_permutation(source['cell_ids'][mask], seed, str(ko))
            values[f'shuffle_{seed}'] = variance.covariance_diagonal(z, v[order])
            movement.append(dict(KO=str(ko), seed=seed, moved_fraction=float(np.mean(order != np.arange(len(order))))))
        flux.append(values)
    starts, ends = np.asarray(starts), np.asarray(ends)
    return dict(labels=labels, local=np.asarray(local), variance=starts, target_variance=ends,
                y=np.log(ends)-np.log(starts), z0=np.asarray(means), technical=np.asarray(technical),
                esm=np.stack([embeddings[k] for k in labels]),
                flux={name:np.stack([v[name] for v in flux]) for name in flux[0]}), movement


def program(configuration, output):
    from w7_router_scaling import write_csv
    plan = json.loads(Path(configuration).read_text())
    if tuple(plan['arms']) != variance.ARMS: raise ValueError('Short-window recipe changed')
    inputs = [Path(configuration), Path(__file__), ROOT/'PLANS/w15z_short_variance_flux_20261022.md',
              ROOT/'scripts/phase0/w15y_velocity_variance_flux.py', ROOT/'scripts/phase0/w15x_u_increment_screen.py',
              Path(plan['split']), Path(plan['conditions']), Path(plan['fold'])/'train_source.npz',
              Path(plan['fold'])/'train_target.npz']
    run = Artifacts(output, seed=plan['seed'], inputs=inputs)
    data, movement = load_training_data(plan)
    rows, coefficients = [], []
    for fold, (train, query) in enumerate(increment.split_indices(np.arange(len(data['labels'])), plan['outer_folds'], plan['seed'])):
        predictions, alphas = variance.predict_fold(data, train, query, plan)
        np.savez_compressed(run.directory/f'fold{fold}_predictions.npz', training_indices=train, query_indices=query, **predictions)
        for arm, pred in predictions.items():
            coefficients.append(dict(fold=fold, arm=arm, alpha=alphas[arm]))
            for i, ko in enumerate(data['labels'][query]):
                rows.append(dict(fold=fold, KO=str(ko), arm=arm, log_variance_mse=float(np.mean((pred[i]-data['y'][query[i]])**2))))
        run.json('status.json', dict(status='outer_fold_complete', completed_outer_folds=fold+1))
    by = {(r['KO'], r['arm']):r['log_variance_mse'] for r in rows}
    comparisons = []
    for real in ('velocity_positive', 'velocity_free'):
        for control in variance.ARMS:
            if real == control: continue
            values = increment.bootstrap_comparison([by[(ko, real)] for ko in data['labels']],
                [by[(ko, control)] for ko in data['labels']], plan)
            comparisons.append(dict(real=real, control=control, **values))
    primary = [r for r in comparisons if r['real'] == 'velocity_positive' and r['control'] in plan['primary_controls']]
    positive_folds = sum(r['alpha'] > 0 for r in coefficients if r['arm'] == 'velocity_positive')
    result = dict(status='SHORT_WINDOW_VARIANCE_FLUX_COMPLETE', source_hours=plan['source_hours'], target_hours=plan['target_hours'],
        training_KOs=len(data['labels']), comparisons=comparisons, positive_alpha_folds=positive_folds,
        variance_signal_screen_passed=positive_folds >= 3 and all(r['ci_low'] > 0 for r in primary),
        means={a:float(np.mean([r['log_variance_mse'] for r in rows if r['arm'] == a])) for a in variance.ARMS},
        coefficients=coefficients, shuffled_moved_fraction=float(np.mean([r['moved_fraction'] for r in movement])),
        validation_targets_read=False, test_targets_read=False, GFG_trained=False, router_trained=False,
        multimodality_claim=False, independent_external_confirmation=False)
    write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'comparisons.csv', comparisons)
    write_csv(run.directory/'coefficients.csv', coefficients); write_csv(run.directory/'shuffle_movement.csv', movement)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# Short-window training-KO variance flux\n\n'+json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--configuration', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); print(json.dumps(program(args.configuration, args.output), indent=2), flush=True)
