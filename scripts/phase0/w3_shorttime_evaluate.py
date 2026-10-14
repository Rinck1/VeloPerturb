"""Evaluate only after the entire fixed short-time seed grid is frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.latent import load_pack
from w3_shorttime_population import population_means


def evaluate(c, output):
    root = Path(c['output_root']); frozen = []
    for seed in c['seeds']:
        directory = root / f'seed{seed}'
        summary = json.loads((directory / 'summary.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        path = directory / 'validation_predictions.npz'
        if (summary['status'] != 'SHORTTIME_PREDICTIONS_FROZEN' or summary['future_validation_target_read']
            or not summary['no_JVP'] or provenance['status'] != 'complete'
            or sha256(path) != summary['prediction_sha256'] or sha256(path) != provenance['outputs'][path.name]):
            raise ValueError('Full prediction grid must be immutable, complete and source-only')
        frozen.append(dict(seed=seed, path=str(path), sha256=sha256(path)))
    # No validation target was read or hashed above this boundary.
    target_path = Path(c['fold']) / 'validation_target.npz'
    with Run(output, stage='shorttime_frozen_grid_development_evaluation', kind='development', config=c,
             inputs=[Path(__file__), target_path, *[Path(r['path']) for r in frozen]], seed=c['seeds']) as run:
        target, meta = load_pack(target_path, expected_side='target')
        if meta['role'] != 'validation' or meta['day'] != 5:
            raise ValueError('Expected one-day development target')
        metrics = []; gains = []
        save_json(run.directory / 'frozen_grid.json', frozen)
        for r in frozen:
            with np.load(r['path'], allow_pickle=False) as f:
                labels = f['conditions'].astype(str)
                truth = population_means(target['z'], target['conditions'], labels)
                errors = {}
                for arm in c['arms']:
                    errors[arm] = np.square(f[arm]-truth).mean(1)
                    for i, label in enumerate(labels):
                        metrics.append(dict(seed=r['seed'], condition=label, arm=arm,
                            pseudobulk_mse=float(errors[arm][i])))
                for control in ('static', 'gfg_population_shuffled_U', 'gfg_mean_cell_velocity', 'analytic_population'):
                    for i, label in enumerate(labels):
                        gains.append(dict(seed=r['seed'], condition=label, control=control,
                            gain=float(errors[control][i]-errors['gfg_population'][i])))
        arm_means = {arm: float(np.mean([r['pseudobulk_mse'] for r in metrics if r['arm'] == arm])) for arm in c['arms']}
        comparisons = []
        for control in ('static', 'gfg_population_shuffled_U', 'gfg_mean_cell_velocity', 'analytic_population'):
            per_tf = [float(np.mean([r['gain'] for r in gains if r['condition'] == tf and r['control'] == control]))
                      for tf in sorted(set(target['conditions']))]
            comparisons.append(dict(control=control, mean_gain=float(np.mean(per_tf)),
                positive_TFs=int(np.sum(np.array(per_tf)>0)), n_TFs=len(per_tf),
                seed_averaged_TF_gains=per_tf, formal_significance_claim=False))
        summary = dict(status='SHORTTIME_DEVELOPMENT_GRID_EVALUATED', task=c['primary_task'],
            arm_mean_pseudobulk_mse=arm_means, comparisons=comparisons, inference_unit='TF after seed averaging',
            n_independent_TFs=4, seeds_are_biological_replicates=False,
            verdict='DEVELOPMENT_DIRECTION_POSITIVE' if all(v['mean_gain']>0 for v in comparisons[:2]) else 'VELOCITY_INCREMENT_NOT_SUPPORTED_BY_DEVELOPMENT_DIRECTION',
            final_confirmation=False, population_future_prediction=True, individual_fate_prediction=False)
        save_csv(run.directory / 'metrics.csv', metrics); save_csv(run.directory / 'gains.csv', gains)
        save_json(run.directory / 'summary.json', summary)
        (run.directory / 'RESULTS.md').write_text('# One-day population forecast\n\n' + json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/w3_shorttime_20261014.json')
    ap.add_argument('--output', required=True); ap.add_argument('--wait-hours', type=float, default=2)
    args = ap.parse_args(); c = load_config(args.config); deadline = time.monotonic()+args.wait_hours*3600
    root = Path(c['output_root'])
    while not all((root / f'seed{s}/provenance.json').is_file() and
                  json.loads((root / f'seed{s}/provenance.json').read_text())['status']=='complete' for s in c['seeds']):
        failed = [s for s in c['seeds'] if (root/f'seed{s}/provenance.json').is_file() and
                  json.loads((root/f'seed{s}/provenance.json').read_text())['status']=='failed']
        if failed:
            raise RuntimeError(f'Short-time seed runs failed: {failed}')
        if time.monotonic()>deadline:
            raise TimeoutError('Fixed short-time grid is incomplete; target unopened')
        print(json.dumps(dict(status='waiting_for_entire_prediction_grid')), flush=True); time.sleep(15)
    evaluate(c, args.output)


if __name__ == '__main__':
    main()
