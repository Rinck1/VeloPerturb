"""Post-hoc diagnosis of the frozen minute-scale result, NOT new forecasts.

Unscaled direction and a retrospective 1-D oracle separate direction failure
from a zero calibration scale. Baseline-centered velocity variants are marked
exploratory; no primary prediction, score, or protocol is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_metadata
from w4_streams import SourceTransform
from w5_scntseq_minutes import (cosine, load_population, population_features, verify_config)


def oracle_direction(feature, displacement, minutes):
    alignment = cosine(feature, displacement)
    denominator = float(feature @ feature)
    scale = 0. if denominator <= 1e-12 else max(0., float(feature @ displacement)/(minutes*denominator))
    return dict(raw_direction_cosine=alignment, retrospective_nonnegative_scale=scale,
        retrospective_1d_gain_ceiling=0. if alignment is None else max(0., alignment)**2)


def audit(predictions, output):
    directory = Path(predictions)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    config_path = Path(manifest['config_path'])
    config, protocol, data = verify_config(config_path, manifest['freeze_commit'])
    transform_path = directory/'source_transform.npz'
    if sha256(transform_path) != manifest['transform_sha256']:
        raise ValueError('Frozen source transform changed')
    run = Artifacts(output, seed=config['seed'], inputs=[Path(__file__),
        Path(__file__).with_name('w5_scntseq_minutes.py'), config_path, protocol, data,
        transform_path, directory/'prediction_manifest.json', directory/'calibration.json'])
    transform = SourceTransform.load(transform_path)
    obs, genes = read_metadata(data)
    np.testing.assert_array_equal(genes, transform.gene_names)
    base_layers, _ = load_population(data, obs, 0, config, source=True)
    states, features = {}, {}
    states[0], features[0] = population_features(transform, base_layers)
    previous = 0
    for minute in (15,30,60):
        layers, _ = load_population(data, obs, minute, config, source=True)
        states[minute], features[minute] = population_features(transform, layers, baseline=features[0],
            previous_state=states[previous], elapsed_minutes=minute-previous)
        previous = minute
    last, _ = load_population(data, obs, 120, config, source=False)
    states[120] = transform.state(last['spliced']).mean(0)
    rows = []
    for start,stop in ((15,30),(30,60),(60,120)):
        candidates = {key:value for key,value in features[start].items() if key != 'identity'}
        for key in ('vel_mean','vel_pool'):
            candidates[f'{key}_minus_baseline0'] = features[start][key]-features[0][key]
        for name, feature in candidates.items():
            rows.append(dict(source_minute=start, target_minute=stop, feature=name,
                interval_role='calibration' if start == 15 else 'previously_scored_test',
                exploratory_variant=name.endswith('_minus_baseline0'),
                feature_norm=float(np.linalg.norm(feature)),
                **oracle_direction(feature, states[stop]-states[start], stop-start)))
    sn, un = transform.normalized(base_layers['spliced'], base_layers['unspliced'])
    reasons = dict(insufficient_positive_S=0, zero_tail_cross_product=0,
        other_zero_fit=0, U_never_observed_among_zero_gamma=0)
    for gene in np.flatnonzero(transform.gamma == 0):
        s, u = sn[:,gene], un[:,gene]
        if not np.any(u > 0):
            reasons['U_never_observed_among_zero_gamma'] += 1
        if np.count_nonzero(s > 0) < 10:
            reasons['insufficient_positive_S'] += 1
        else:
            low,high = np.quantile(s,[.05,.95]); selected = (s <= low) | (s >= high)
            reasons['zero_tail_cross_product' if s[selected] @ u[selected] == 0 else 'other_zero_fit'] += 1
    selected = [r for r in rows if r['feature'] in ('vel_mean','vel_pool',
        'vel_mean_minus_baseline0','vel_pool_minus_baseline0')]
    result = dict(status='POSTHOC_MINUTE_DIRECTION_AUDIT_COMPLETE', exploratory=True,
        original_predictions_changed=False, oracle_results_are_deployable=False,
        gamma_zero_genes=int((transform.gamma == 0).sum()), total_HVGs=len(transform.gamma),
        zero_gamma_reasons=reasons, velocity_directions=selected,
        baseline_field_norms={key:float(np.linalg.norm(features[0][key])) for key in ('vel_mean','vel_pool')})
    with (run.directory/'directions.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# Post-hoc minute-scale direction audit\n\n'
        'Oracle scales/ceilings use already observed future states and are NOT prediction results. '
        'Baseline-centered velocity is exploratory. Frozen primary forecasts remain unchanged.\n\n'
        +json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default='outputs/w5_scntseq_predictions_v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.predictions, args.output)
