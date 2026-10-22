"""Five fixed rolling readouts with an onset-aware pulse-label reference.

The reference is an explicit working model, not ground truth. Its baseline
rates and cross-assay observation mapping use 0-min cells only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts, sha256

HERE = Path(__file__).resolve().parent
ARMS = ('S_history', 'S_U_driver', 'S_velocity', 'S_reverse_velocity', 'S_label_reference')


def fit_label_baseline(raw_total, raw_new, normalized_S, target_sum, plan):
    total, new, sn = [np.asarray(v, dtype=float) for v in (raw_total, raw_new, normalized_S)]
    if total.shape != new.shape or sn.shape != total.shape or total.ndim != 1:
        raise ValueError('Aligned gene pools required')
    if not np.isfinite(total+new+sn).all() or (total < 0).any() or (new < 0).any() or (new > total).any():
        raise ValueError('Invalid new/total observations')
    # Caller passes selected genes and their normalization based on ALL genes.
    r0 = total*target_sum
    fraction = np.divide(new, total, out=np.zeros_like(new), where=total > 0)
    support = ((total >= plan['minimum_baseline_total_gene_counts']) &
        (new >= plan['minimum_baseline_new_gene_counts']) &
        (total-new >= plan['minimum_baseline_old_gene_counts']) &
        (fraction > 0) & (fraction < 1) & (sn > 0))
    gamma = np.zeros_like(total)
    gamma[support] = -np.log1p(-fraction[support])/plan['label_duration_minutes']
    mapping = np.divide(sn, r0, out=np.zeros_like(sn), where=support & (r0 > 0))
    return dict(gamma=gamma, support=support, R0=r0, baseline_fraction=fraction, mapping_to_S=mapping)


def onset_reference(total, new, baseline, minute, label_duration):
    """Recover net rate under steady baseline/fixed decay/constant-onset-alpha."""
    r, n = np.asarray(total, float), np.asarray(new, float)
    if minute <= 0 or minute > label_duration: raise ValueError('Source must be within the labeling window')
    if r.shape != baseline['gamma'].shape or n.shape != r.shape or not np.isfinite(r+n).all():
        raise ValueError('Aligned finite current label pools required')
    if (r < 0).any() or (n < 0).any() or (n > r+1e-9).any(): raise ValueError('Invalid current label observations')
    gamma = baseline['gamma']; support = baseline['support']
    decay = np.exp(-gamma*minute)
    prestimulus = baseline['R0']*(decay-np.exp(-gamma*label_duration))
    denominator = -np.expm1(-gamma*minute)
    factor = np.divide(gamma, denominator, out=np.full_like(gamma, 1./minute), where=denominator > 1e-12)
    raw_alpha = factor*(n-prestimulus)
    alpha = np.maximum(raw_alpha, 0.)
    rate = np.where(support, alpha-gamma*r, 0.)
    diagnostics = dict(minute=minute, supported_genes=int(support.sum()),
        negative_synthesis_fraction=float(np.mean(raw_alpha[support] < 0)) if support.any() else None,
        prestimulus_new_fraction=float(prestimulus[support].sum()/max(n[support].sum(), 1e-12)))
    return rate, diagnostics


def fields(population, baseline_population, label_baseline, gamma, components, minute, plan):
    from w5_scntseq_repair import velocity_feature
    v = velocity_feature(population, baseline_population, gamma, components, 'pool', 'baseline0')
    u = (population['pool_u']-baseline_population['pool_u']) @ components.T
    if minute == 0:
        label_rate = np.zeros_like(label_baseline['gamma']); diagnostics = dict(minute=0)
    else:
        label_rate, diagnostics = onset_reference(population['label_total'], population['label_new'],
            label_baseline, minute, plan['label_duration_minutes'])
    label = (label_baseline['mapping_to_S']*label_rate/(1+population['normalized_pool_s'])) @ components.T
    return dict(velocity=v, U_driver=u, label_reference=label), diagnostics


def design(arm, populations, vectors, minute, previous):
    offset = populations[minute]['state']-populations[0]['state']
    history = (populations[minute]['state']-populations[previous]['state'])/(minute-previous)
    if arm == 'S_history': return [offset, history], [False, False]
    if arm == 'S_U_driver': return [offset, vectors[minute]['U_driver']], [False, False]
    if arm == 'S_velocity': return [offset, vectors[minute]['velocity']], [False, True]
    if arm == 'S_reverse_velocity': return [offset, -vectors[minute]['velocity']], [False, True]
    if arm == 'S_label_reference': return [offset, vectors[minute]['label_reference']], [False, True]
    raise ValueError('Unknown fixed arm')


def program(configuration, output=None):
    from w4_shorttime import read_metadata
    from w4_streams import SourceTransform
    from w5_scntseq_minutes import load_population, cosine
    from w5_scntseq_repair import setup, sufficient_features, regularized_gamma, fit_readout, readout_rate, effective_minutes
    from w7_router_scaling import write_csv
    plan = json.loads(Path(configuration).read_text())
    if tuple(plan['arms']) != ARMS or plan['calibration_interval'] != [15, 30] or plan['prediction_intervals'] != [[30, 60], [60, 120]]:
        raise ValueError('Fixed rolling design changed')
    _, original, data, transform_path = setup(plan['original_repair_configuration'])
    code = [HERE/n for n in ('w16b_scntseq_label_reference.py', 'w5_scntseq_repair.py',
        'w5_scntseq_minutes.py', 'w4_streams.py', 'w4_shorttime.py', 'w4_artifacts.py')]
    inputs = [Path(configuration), Path(plan['protocol']), Path(plan['original_repair_configuration']),
        data, transform_path, *code]
    run = Artifacts(output or plan['root'], seed=plan['seed'], inputs=inputs)
    try:
        transform = SourceTransform.load(transform_path); obs, genes = read_metadata(data)
        np.testing.assert_array_equal(genes, transform.gene_names)
        populations = {}; counts = {}; raw_pools = {}; vectors = {}; diagnostic = []
        def load(minute):
            layers, ids = load_population(data, obs, minute, original, source=True)
            value = sufficient_features(transform, layers)
            total = np.asarray(layers['total'].sum(0)).ravel().astype(float)
            new = np.asarray(layers['new'].sum(0)).ravel().astype(float)
            if total.sum() <= 0: raise ValueError('Empty label pool')
            value['label_total'] = total[transform.genes]*(transform.target_sum/total.sum())
            value['label_new'] = new[transform.genes]*(transform.target_sum/total.sum())
            raw_pools[minute] = dict(total=total[transform.genes], new=new[transform.genes],
                total_to_normalized=transform.target_sum/total.sum())
            populations[minute] = value; counts[str(minute)] = len(ids)
            if minute == 0: np.testing.assert_array_equal(ids, transform.source_ids)
        for minute in (0, 15, 30): load(minute)
        raw0 = raw_pools[0]
        label_baseline = fit_label_baseline(raw0['total'], raw0['new'], populations[0]['normalized_pool_s'],
            raw0['total_to_normalized'], plan)
        if label_baseline['support'].sum() < plan['minimum_reference_genes']:
            raise ValueError('Too few baseline-supported genes for a reference; no prior-only rates invented')
        gamma, gamma_details = regularized_gamma(populations[0]['normalized_pool_s'], populations[0]['normalized_pool_u'],
            plan['prior_spliced_units'])
        np.savez_compressed(run.directory/'baseline_estimators.npz', gene_ids=genes[transform.genes],
            splicing_gamma=gamma, **label_baseline)
        for minute in (0, 15, 30):
            vectors[minute], details = fields(populations[minute], populations[0], label_baseline, gamma,
                transform.components, minute, plan); diagnostic.append(details)
        fits = {}; calibration_rate = (populations[30]['state']-populations[15]['state'])/15.
        for arm in ARMS:
            x, positive = design(arm, populations, vectors, 15, 0)
            fits[arm] = fit_readout(x, calibration_rate, positive, plan['readout_ridge'])
        run.json('calibration.json', dict(interval=[15, 30], fits=fits, label_rate_fit_minutes=[0],
            mapping_fit_minutes=[0], readout_fit_target_minutes=[30], prior_details=gamma_details,
            label_supported_genes=int(label_baseline['support'].sum()),
            reference_is_ground_truth=False, future_prediction_targets_used=False))
        names = ('identity', *ARMS); predictions = []; source_states = []
        for start, stop, previous in ((30, 60, 15), (60, 120, 30)):
            if start == 60:
                load(60)
                vectors[60], details = fields(populations[60], populations[0], label_baseline, gamma,
                    transform.components, 60, plan); diagnostic.append(details)
            state = populations[start]['state']; values = [state.copy()]
            for arm in ARMS:
                x, _ = design(arm, populations, vectors, start, previous)
                values.append(state+effective_minutes(stop-start, 15, plan['time_kernel'])*readout_rate(x, fits[arm]))
            values = np.stack(values); path = run.directory/f'prediction_{start}_to_{stop}.npz'
            np.savez_compressed(path, arms=np.asarray(names), predictions=values, source_state=state)
            run.json(f'interval_{start}_to_{stop}_frozen.json', dict(status='frozen', prediction_sha256=sha256(path),
                source_minutes_loaded=[0, 15, 30] if start == 30 else [0, 15, 30, 60],
                future_target_used=False, coefficients_refitted=False))
            predictions.append(values); source_states.append(state)
        combined = run.directory/'predictions.npz'
        np.savez_compressed(combined, arms=np.asarray(names), intervals=np.asarray(plan['prediction_intervals']),
            predictions=np.stack(predictions), source_states=np.stack(source_states))
        run.json('predictions_frozen.json', dict(status='all_predictions_frozen', sha256=sha256(combined),
            calibration_sha256=sha256(run.directory/'calibration.json'), target120_read=False,
            source_cells=counts, original_results_changed=False))
        rows = []; directions = []
        for index, (start, stop) in enumerate(plan['prediction_intervals']):
            if stop == 60: truth = populations[60]['state']
            else:
                layers, _ = load_population(data, obs, stop, original, source=False)
                truth = transform.state(layers['spliced']).mean(0)
            actual = truth-source_states[index]
            errors = {name:float(np.mean((p-truth)**2)) for name, p in zip(names, predictions[index])}
            for name in names:
                rows.append(dict(source_minutes=start, target_minutes=stop, arm=name, mse=errors[name],
                    gain_vs_identity=1-errors[name]/max(errors['identity'], 1e-12),
                    gain_vs_S=1-errors[name]/max(errors['S_history'], 1e-12),
                    gain_vs_U=1-errors[name]/max(errors['S_U_driver'], 1e-12)))
            for key, field in vectors[start].items():
                directions.append(dict(source_minutes=start, target_minutes=stop, representation=key,
                    cosine_to_future_displacement=cosine(field, actual), norm=float(np.linalg.norm(field))))
        write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'direction_metrics.csv', directions)
        write_csv(run.directory/'reference_diagnostics.csv', [r for r in diagnostic if r['minute'] > 0])
        comparison = {a:dict(mean_gain_vs_S=float(np.mean([r['gain_vs_S'] for r in rows if r['arm'] == a])),
            mean_gain_vs_U=float(np.mean([r['gain_vs_U'] for r in rows if r['arm'] == a])),
            both_intervals_better_than_S_and_U=all(r['gain_vs_S'] > 0 and r['gain_vs_U'] > 0 for r in rows if r['arm'] == a)) for a in ARMS}
        result = dict(status='SCNTSEQ_LABEL_REFERENCE_COMPLETE', arms=list(names), fitted_readouts=5,
            calibration_interval=[15, 30], prediction_intervals=plan['prediction_intervals'], metrics=rows,
            comparisons=comparison, coefficients=fits, reference_diagnostics=diagnostic,
            label_supported_genes=int(label_baseline['support'].sum()), PCA_dimensions=len(transform.components),
            previous_results_overwritten=False, independent_confirmation=False, biological_significance_test=False,
            reference_is_ground_truth=False, new_RNA_is_velocity=False,
            labeling_model='baseline_steady_fixed_decay_constant_effective_alpha_since_stimulus',
            cross_assay_mapping='baseline_gene_specific_positive_ratio_fixed_before_forecast',
            GFG=False, JVP=False, router_trained=False, gpu_used=False)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# scNT-seq onset-aware label reference\n\n'+json.dumps(result, indent=2)+'\n\n'
            'This is a model-supported relative-RNA reference, not observed instantaneous velocity or a strict oracle upper bound.\n')
        run.json('status.json', dict(status='complete')); return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--configuration', required=True)
    parser.add_argument('--output'); args = parser.parse_args()
    print(json.dumps(program(args.configuration, args.output), indent=2), flush=True)
