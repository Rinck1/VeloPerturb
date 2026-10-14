"""Evaluate the frozen no-JVP RENGE day4->day5 group-mean prediction grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.latent import load_pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_7_short_horizon_no_jvp_20261015.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    c = load_config(args.config)
    artifact_root = Path(c['artifact_root'])
    expected_conditions = {'LIN28A', 'NANOG', 'POU5F1', 'ZIC3'}
    arms = list(c['arms'])

    # Freeze and validate the complete prediction grid before opening future labels.
    frozen = []
    for seed in c['seeds']:
        directory = artifact_root / f'seed{seed}_v1'
        manifest_path = directory / 'frozen_prediction_manifest.json'
        prediction_path = directory / 'frozen_source_predictions.npz'
        model_path = directory / 'no_jvp_model.pt'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'frozen_source_only' or manifest.get('future_target_read') is not False:
            raise ValueError(f'Seed {seed} is not a source-only frozen prediction')
        if manifest.get('validation_target_read') is not False:
            raise ValueError(f'Seed {seed} claims validation target access')
        if manifest.get('validation_conditions') is None or set(manifest['validation_conditions']) != expected_conditions:
            raise ValueError(f'Seed {seed} validation condition set mismatch')
        if manifest.get('arms') != arms:
            raise ValueError(f'Seed {seed} registered arms mismatch')
        if sha256(prediction_path) != manifest.get('source_sha256') or sha256(model_path) != manifest.get('model_sha256'):
            raise ValueError(f'Seed {seed} frozen artifact hash mismatch')
        if manifest.get('protocol_hash') != sha256(Path(args.config)):
            raise ValueError(f'Seed {seed} protocol hash mismatch')
        with np.load(prediction_path, allow_pickle=False) as data:
            if set(data['conditions'].astype(str)) != expected_conditions or data['arms'].astype(str).tolist() != arms:
                raise ValueError(f'Seed {seed} prediction keys mismatch')
            payload = {key: data[key].copy() for key in data.files}
        frozen.append((seed, manifest, payload))

    # This is the first permitted read of held-out future observations.
    fold = Path(c['fold'])
    target, target_meta = load_pack(fold / 'validation_target.npz', expected_side='target')
    if target_meta['transform_hash'] != load_pack(fold / 'validation_source.npz', expected_side='source')[1]['transform_hash']:
        raise ValueError('Validation source/target transform mismatch')
    if set(target['conditions']) != expected_conditions:
        raise ValueError('Held-out target condition set mismatch')

    rows = []
    for seed, manifest, payload in frozen:
        conditions = payload['conditions'].astype(str)
        for arm in arms:
            predicted = payload[f'{arm}_predicted_future_mean']
            if predicted.shape != (len(conditions), 50):
                raise ValueError(f'Unexpected prediction shape for seed={seed}, arm={arm}: {predicted.shape}')
            for i, condition in enumerate(conditions):
                truth = target['z'][target['conditions'] == condition].mean(0)
                mse = float(np.mean((predicted[i] - truth) ** 2))
                rows.append(dict(seed=seed, condition=condition, arm=arm, group_mean_mse=mse,
                                 target_cells=int(np.sum(target['conditions'] == condition)),
                                 prediction_sha256=manifest['source_sha256']))

    summary = []
    for arm in arms:
        vals = [r['group_mean_mse'] for r in rows if r['arm'] == arm]
        summary.append(dict(arm=arm, mean_condition_seed_mse=float(np.mean(vals)), n_conditions=len(expected_conditions),
                            n_seeds=len(c['seeds']), inference_unit='heldout_TF; seeds are repeats, not biological units'))
    bykey = {(r['seed'], r['condition'], r['arm']): r['group_mean_mse'] for r in rows}
    contrasts = []
    for arm in arms[1:]:
        paired = [bykey[(seed, condition, 'S_only')] - bykey[(seed, condition, arm)]
                  for seed in c['seeds'] for condition in sorted(expected_conditions)]
        mean_improvement = float(np.mean(paired))
        per_condition = {condition: float(np.mean([bykey[(seed, condition, 'S_only')]-
                                                    bykey[(seed, condition, arm)] for seed in c['seeds']]))
                         for condition in sorted(expected_conditions)}
        contrasts.append(dict(arm=arm, improvement_vs_S_only=float(np.mean(paired)),
                              relative_improvement=float(mean_improvement /
                                  np.mean([bykey[(seed, condition, 'S_only')] for seed in c['seeds']
                                           for condition in sorted(expected_conditions)])),
                              heldout_conditions_won=sum(value > 0 for value in per_condition.values()),
                              per_condition_improvement=per_condition,
                              condition_seed_differences=paired,
                              interpretation='positive means lower heldout group-mean MSE than S_only; descriptive with four TFs'))

    output = Path(args.output)
    with Run(output, stage='W3_7_no_JVP_frozen_short_horizon_evaluation', kind='development', config=c,
             inputs=[Path(args.config), *[artifact_root / f'seed{s}_v1' / 'frozen_prediction_manifest.json' for s in c['seeds']],
                     *[artifact_root / f'seed{s}_v1' / 'no_jvp_model.pt' for s in c['seeds']],
                     fold / 'validation_source.npz', fold / 'validation_target.npz',
                     Path(__file__), Path('scripts/phase0/w3_7_short_horizon_no_jvp.py'),
                     Path('scripts/phase0/w3_gfg_v2_control.py'), Path('scripts/phase0/w2_gfg_v2.py'),
                     Path('scripts/phase0/w3_validation.py')]) as run:
        save_csv(run.directory / 'condition_seed_metrics.csv', rows)
        save_csv(run.directory / 'arm_summary.csv', summary)
        result = dict(status='NO_JVP_DAY4_TO_DAY5_EVALUATED', dataset='RENGE_day4_to_day5', arms=summary,
                      contrasts=contrasts, validation_target_opened_after_full_grid_freeze=True,
                      heldout_TFs=sorted(expected_conditions), claim_scope='descriptive heldout-condition population mean prediction; not cell fate')
        save_json(run.directory / 'summary.json', result)
        (run.directory / 'RESULTS.md').write_text('# RENGE no-JVP day4→day5 evaluation\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
