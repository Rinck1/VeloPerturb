"""Calibrate the W4 implementation using synthetic, unpaired populations.

The independent test populations are never used to fit geometry or response.
This fixture tests detection/calibration, not the power of the RENGE protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts
from w4_arms import paired_gain_test, predict_arms
from w4_streams import SourceTransform, add_swap_feature, group_mean, population_features


def synthetic_case(seed, kind, *, train_groups=40, test_groups=24, cells=12, genes=12):
    rng = np.random.default_rng(seed)
    n = train_groups+test_groups
    labels = np.array([f'TF{i:03}' for i in range(n)])
    groups = np.repeat(labels, cells)
    ids = np.array([f'source_{i}' for i in range(len(groups))])
    state = np.exp(rng.normal(0, .15, (n, genes)))
    state /= state.sum(1, keepdims=True)
    s = np.repeat(state*10000, cells, axis=0)*np.exp(rng.normal(0, .02, (n*cells, genes)))
    s *= 10000/s.sum(1, keepdims=True)
    innovation = rng.normal(0, 90, (n, genes))
    innovation -= innovation.mean(1, keepdims=True)
    u = .8*s + np.repeat(innovation, cells, axis=0)
    if kind == 'S_determined_stress':
        # Strictly a function of OBSERVED per-cell S, not latent group S.
        u = .8*s + .15*(s-10000/genes)
    if (u < 0).any():
        raise ArithmeticError('Synthetic U went negative; do not clip silently')
    response = .5*innovation if kind == 'positive' else np.zeros_like(innovation)
    # Independent destructive sampling from the population: do not reuse the
    # realized source cells/noise to create the future snapshot.
    target_s = np.repeat(state*10000, cells, axis=0)*np.exp(rng.normal(0, .02, s.shape))
    target_s *= 10000/target_s.sum(1, keepdims=True)
    target_s += np.repeat(response, cells, axis=0)
    if (target_s < 0).any():
        raise ArithmeticError('Synthetic target went negative')
    training = np.arange(len(groups)) < train_groups*cells
    transform = SourceTransform.fit(s[training], u[training], cell_ids=ids[training],
        gene_names=[f'gene{i}' for i in range(genes)], allowed_ids=set(ids[training]),
        n_hvg=genes, dimensions=6, clusters=4, seed=seed)
    train = population_features(transform, s[training], u[training], groups[training], labels[:train_groups], ids[training])
    test = population_features(transform, s[~training], u[~training], groups[~training], labels[train_groups:], ids[~training])
    train, _ = add_swap_feature(transform, train, train, seed)
    test, _ = add_swap_feature(transform, test, train, seed)
    # Training future counts are legal response supervision, not geometry fit.
    target_train = group_mean(transform.state(target_s[training]), groups[training], labels[:train_groups])
    prediction, _ = predict_arms(train, test, target_train, np.zeros((train_groups, 1)), np.zeros((test_groups, 1)),
        arms=('static', 'vel_pool', 'vel_pool_Uswap'))
    truth = group_mean(transform.state(target_s[~training]), groups[~training], labels[train_groups:])
    errors = {arm: np.square(p-truth).sum(1)[:, None] for arm, p in prediction.items()}
    comparisons = {ref: paired_gain_test(errors['vel_pool'], errors[ref], errors['static'],
        seed=seed, repetitions=1000) for ref in ('static', 'vel_pool_Uswap')}
    return dict(seed=seed, kind=kind, detected=all(c['supported'] for c in comparisons.values()),
        gain_static=comparisons['static']['gain'], gain_swap=comparisons['vel_pool_Uswap']['gain'],
        p_static=comparisons['static']['wilcoxon_p'], p_swap=comparisons['vel_pool_Uswap']['wilcoxon_p'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--replicates', type=int, default=200)
    p.add_argument('--seed', type=int, default=20261017)
    args = p.parse_args()
    if args.replicates < 1:
        raise ValueError('Positive replicate count required')
    inputs = [Path(__file__), Path(__file__).with_name('w4_streams.py'),
              Path(__file__).with_name('w4_arms.py'), Path(__file__).with_name('w4_artifacts.py')]
    run = Artifacts(args.output, seed=args.seed, inputs=inputs)
    rows = []
    for kind in ('positive', 'null', 'S_determined_stress'):
        for i in range(args.replicates):
            row = synthetic_case(args.seed+i, kind)
            rows.append(row)
            if (i+1) % 20 == 0:
                run.json('status.json', dict(status='running', kind=kind, completed=i+1))
                print(json.dumps(dict(kind=kind, completed=i+1)), flush=True)
    with open(run.directory/'replicates.csv', 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rates = {kind: float(np.mean([r['detected'] for r in rows if r['kind'] == kind]))
             for kind in ('positive', 'null', 'S_determined_stress')}
    summary = dict(status='complete', replicates_per_kind=args.replicates,
        detection_rates=rates, positive_detected=rates['positive'] >= .8,
        null_false_positive_within_10_percent=rates['null'] <= .1,
        stress_is_not_exact_randomization=True, fixture='40_training_24_independent_test_populations',
        RENGE_power_estimate=False, independent_future_cells=True, strict_observed_S_determined_stress=True,
        csv_sha256=__import__('w4_artifacts').sha256(run.directory/'replicates.csv'))
    run.json('summary.json', summary); run.json('status.json', dict(status='complete'))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
