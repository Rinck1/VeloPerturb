"""Train-KO-only effective splicing-rate calibration; no router or future val."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts

ROOT = Path(__file__).resolve().parents[2]
FOLD = ROOT/'outputs/w15_traxler_fold_6to24_v1'
COUNTS = Path('/data/yuchang/veloroute_w15v_target_estimator_20261022/fixed_estimator_v1/training_counts.npz')
RAW = Path('/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1')
ARMS = ('mean', 'steady_state_scaled', 'S_only', 'U_only', 'unconstrained_SU', 'positive_kinetic', 'ridge_S50', 'ridge_SU50')


def fit_coefficients(features, target, ridge=1., positive=False):
    """Gene-wise ridge with optional exact nonnegative two-variable solution."""
    features, target = np.asarray(features, float), np.asarray(target, float)
    mean, ymean = features.mean(0), target.mean(0)
    centered = features-mean
    scale = np.maximum(np.sqrt((centered**2).mean(0)), 1e-8)
    x, y = centered/scale, target-ymean
    gram = np.einsum('ngi,ngj->gij', x, x)+ridge*np.eye(x.shape[-1])[None]
    rhs = np.einsum('ngi,ng->gi', x, y)
    theta = np.linalg.solve(gram, rhs[..., None])[..., 0]
    if positive and x.shape[-1] == 1:
        theta = np.maximum(theta, 0.)
    elif positive and x.shape[-1] == 2:
        first, second = np.zeros_like(theta), np.zeros_like(theta)
        first[:, 0] = np.maximum(rhs[:, 0]/gram[:, 0, 0], 0.)
        second[:, 1] = np.maximum(rhs[:, 1]/gram[:, 1, 1], 0.)
        candidates = np.stack((np.maximum(theta, 0.), first, second, np.zeros_like(theta)))
        costs = np.einsum('agi,gij,agj->ag', candidates, gram, candidates)-2*np.einsum('agi,gi->ag', candidates, rhs)
        costs[0, (theta < 0).any(1)] = np.inf
        theta = candidates[np.argmin(costs, axis=0), np.arange(len(theta))]
    elif positive:
        raise ValueError('Exact positive solver supports one/two features only')
    return dict(mean=mean, ymean=ymean, coefficient=theta/scale)


def predict_coefficients(model, features):
    return model['ymean']+np.einsum('ngi,gi->ng', np.asarray(features)-model['mean'], model['coefficient'])


def target_log_means(pack, transform, labels):
    import h5py
    from w4_shorttime import read_count_rows
    totals = np.zeros((len(labels), len(transform.selected)))
    sizes = np.zeros(len(labels), dtype=int)
    ids = np.asarray(pack['cell_ids'], dtype=str)
    label_index = {k: i for i, k in enumerate(labels)}
    for library in sorted({k.split(':', 1)[0] for k in ids}):
        path = RAW/(library+'.h5ad')
        with h5py.File(path, 'r') as f:
            names, genes = f['obs']['_index'].asstr()[:], f['var']['_index'].asstr()[:]
        if tuple(genes) != transform.gene_ids: raise ValueError('Target gene order changed')
        lookup = {v: i for i, v in enumerate(names)}
        positions = np.flatnonzero(np.char.startswith(ids, library+':'))
        rows = np.array([lookup[ids[i]] for i in positions]); order = np.argsort(rows)
        rows, positions = rows[order], positions[order]
        s = read_count_rows(path, 'spliced', rows)  # No target U, no non-training row.
        depth = np.asarray(s.sum(1)).ravel()
        if (depth <= 0).any(): raise ValueError('Zero target S exposure')
        log_s = np.log1p(s[:, transform.selected].toarray()*transform.target_sum/depth[:, None])
        conditions = pack['conditions'][positions]
        for ko in set(conditions):
            mask = conditions == ko; index = label_index[str(ko)]
            totals[index] += log_s[mask].sum(0); sizes[index] += int(mask.sum())
    if (sizes == 0).any(): raise ValueError('Missing training target condition')
    return totals/sizes[:, None]


def program(output):
    from veloroute.latent import FrozenSplicingTransform, load_pack
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from w7_router_scaling import write_csv
    source, _ = load_pack(FOLD/'train_source.npz', expected_side='source')
    target, _ = load_pack(FOLD/'train_target.npz', expected_side='target')
    transform = FrozenSplicingTransform.load(FOLD/'transform.npz')
    labels = sorted(set(source['conditions']))
    with np.load(COUNTS, allow_pickle=False) as f:
        s, u, depth = f['S'], f['U'], f['depth']
        np.testing.assert_array_equal(f['cell_ids'], source['cell_ids'])
    libraries = sorted({str(v).split(':', 1)[0] for v in target['cell_ids']})
    inputs = [Path(__file__), ROOT/'PLANS/w15w_rate_calibration_probe_20261022.md', FOLD/'train_source.npz',
              FOLD/'train_target.npz', FOLD/'transform.npz', COUNTS]+[RAW/(v+'.h5ad') for v in libraries]
    run = Artifacts(output, seed=20261022, inputs=inputs)
    run.json('status.json', dict(status='reading_training_target_S_only'))
    s, u = s*transform.target_sum/depth[:, None], u*transform.target_sum/depth[:, None]
    ls = np.log1p(s)
    A = np.stack([(u/(1+s))[source['conditions'] == ko].mean(0) for ko in labels])
    B = np.stack([(-s/(1+s))[source['conditions'] == ko].mean(0) for ko in labels])
    log_start = np.stack([ls[source['conditions'] == ko].mean(0) for ko in labels])
    log_end = target_log_means(target, transform, labels)
    z0 = np.stack([source['z'][source['conditions'] == ko].astype(float).mean(0) for ko in labels])
    z1 = np.stack([target['z'][target['conditions'] == ko].astype(float).mean(0) for ko in labels])
    np.testing.assert_allclose((log_start-transform.mean)@transform.components.T, z0, rtol=0, atol=2e-6)
    np.testing.assert_allclose((log_end-transform.mean)@transform.components.T, z1, rtol=0, atol=2e-6)
    y = (log_end-log_start)/18.
    x = np.stack((A, B), axis=-1)
    ss = (A+B*transform.gamma)[..., None]
    U50 = np.stack([source['u_features'][source['conditions'] == ko].astype(float).mean(0) for ko in labels])
    rows, rates = [], []
    for fold, (train, val) in enumerate(KFold(5, shuffle=True, random_state=20261022).split(labels)):
        features = {'steady_state_scaled': ss, 'S_only': B[..., None], 'U_only': A[..., None],
                    'unconstrained_SU': x, 'positive_kinetic': x}
        predictions = {'mean': np.broadcast_to(y[train].mean(0), (len(val), y.shape[1]))}
        for arm, values in features.items():
            model = fit_coefficients(values[train], y[train], ridge=1., positive=arm in ('positive_kinetic', 'steady_state_scaled'))
            predictions[arm] = predict_coefficients(model, values[val])
            if arm == 'positive_kinetic':
                rates.append(dict(fold=fold, positive_beta_fraction=float((model['coefficient'][:, 0] > 0).mean()),
                                  positive_gamma_fraction=float((model['coefficient'][:, 1] > 0).mean())))
        zpred = {arm: z0[val]+18*values@transform.components.T for arm, values in predictions.items()}
        for arm, values in [('ridge_S50', z0), ('ridge_SU50', np.concatenate((z0, U50), axis=1))]:
            mean, scale = values[train].mean(0), np.maximum(values[train].std(0), 1e-6)
            ridge = Ridge(alpha=1.).fit((values[train]-mean)/scale, z1[train]-z0[train])
            zpred[arm] = z0[val]+ridge.predict((values[val]-mean)/scale)
        for arm, pred in zpred.items():
            for i, index in enumerate(val):
                rows.append(dict(fold=fold, KO=labels[index], arm=arm, mse=float(np.mean((pred[i]-z1[index])**2))))
    by = {(r['KO'], r['arm']): r['mse'] for r in rows}
    comparisons = []
    for control in ARMS:
        if control == 'positive_kinetic': continue
        values = [1-by[(ko, 'positive_kinetic')]/max(by[(ko, control)], 1e-12) for ko in labels]
        comparisons.append(dict(control=control, mse_gain=float(np.mean(values)), winning_KOs=sum(v > 0 for v in values)))
    lookup = {r['control']: r['mse_gain'] for r in comparisons}
    result = dict(status='TRAINING_KO_RATE_CALIBRATION_PROBE_COMPLETE', training_KOs=len(labels), internal_folds=5,
        comparisons=comparisons, means={a:float(np.mean([r['mse'] for r in rows if r['arm'] == a])) for a in ARMS},
        candidate_for_separate_downstream=all(lookup[k] > 0 for k in ('mean', 'steady_state_scaled', 'S_only')),
        rate_diagnostics=rates, validation_targets_read=False, test_targets_read=False, target_U_read=False,
        representation_fixed_on_outer_training_fold=True, velocity_identity='training_time_calibrated_effective_coefficients_not_instantaneous_truth',
        router_trained=False, JVP=False)
    write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'comparisons.csv', comparisons)
    np.savez_compressed(run.directory/'training_mean_features.npz', labels=np.asarray(labels), A=A, B=B, y=y,
                        z0=z0, z1=z1, log_start=log_start, log_end=log_end)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# Training-KO effective rate probe\n\n'+json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    print(json.dumps(program(args.output), indent=2), flush=True)
