"""Report preregistered H7 from frozen H1 errors; no new count data or model fits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts
from w4_arms import CORE_ARMS, paired_gain_test


def report(directory, output):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    if summary['status'] != 'H1_REPLICATION_EVALUATED' or summary['kind'] != 'formal_replication':
        raise ValueError('Completed formal H1 errors required')
    error_path = directory/'TF_interval_errors.npz'
    run = Artifacts(output, seed=20261017, inputs=[Path(__file__), Path(__file__).with_name('w4_arms.py'),
        Path(__file__).with_name('w4_artifacts.py'), directory/'summary.json', error_path])
    with np.load(error_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['TFs'], summary['H1_TFs'])
        np.testing.assert_array_equal(saved['intervals'], summary['intervals'])
        errors = {arm:saved[arm].copy() for arm in CORE_ARMS}
    comparisons = dict(direct_U_vs_static=paired_gain_test(errors['direct_U_pool'], errors['static'], errors['static'],
        seed=20261017, repetitions=10000),
        velocity_vs_direct_U=paired_gain_test(errors['vel_pool'], errors['direct_U_pool'], errors['static'],
        seed=20261017, repetitions=10000))
    intervals = {arm:((errors['static']-error)/np.maximum(errors['static'], 1e-12)).mean(0).tolist()
                 for arm,error in errors.items()}
    result = dict(status='H7_AUXILIARY_REPORTED', n_TFs=len(summary['H1_TFs']),
        H1_supported=summary['H1_supported'], H1_success_rule_changed=False, comparisons=comparisons,
        interval_relative_gain_over_static=intervals, intervals=summary['intervals'],
        inference_unit='TF_after_interval_averaging', scope='prespecified_auxiliary_no_new_model_or_subset')
    run.json('summary.json', result)
    with (run.directory/'comparisons.csv').open('x', newline='') as stream:
        fields = ['comparison', 'gain', 'ci_low', 'ci_high', 'wilcoxon_p']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name,value in comparisons.items():
            writer.writerow(dict(comparison=name, **{k:value[k] for k in fields[1:]}))
    (run.directory/'RESULTS.md').write_text('# W4 H7: direct U versus kinetic features\n\n'
        'Prespecified auxiliary comparisons on the same frozen 16-TF, two-interval grid. '
        'These do not modify the H1 decision.\n\n'+json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='outputs/w4_h1_evaluation_v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report(args.evaluation, args.output)
