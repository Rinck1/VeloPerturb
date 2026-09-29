"""Freeze the entire predetermined GFG pilot before reading development targets."""
import argparse
import json
from pathlib import Path

import numpy as np

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.experiments import evaluate_prediction
from veloroute.metrics import paired_condition_bootstrap


def evaluate_grid(config, root, output):
    root, fold = Path(root), Path(config['fold'])
    arms, seeds = config['pilot']['arms'], config['pilot']['seeds']
    frozen = []
    # Do not open ANY target until every arm/seed prediction is complete and bound.
    for seed in seeds:
        for arm in arms:
            directory = root/f'{arm}_seed{seed}'
            selection_path = directory/'selection.json'
            if selection_path.exists():
                selection = json.loads(selection_path.read_text())
                training, prediction = Path(selection['training']), Path(selection['prediction'])
                if selection['arm'] != arm or selection['seed'] != seed:
                    raise ValueError('Recovery selection arm/seed mismatch')
                if sha256(training/'model.pt') != selection['model_sha256'] or sha256(prediction/'predictions.npz') != selection['prediction_sha256']:
                    raise ValueError('Recovered experiment artifact hash mismatch')
            else:
                training, prediction = directory/'train', directory/'predict'
            for phase in (training, prediction):
                provenance = json.loads((phase/'provenance.json').read_text())
                if provenance['status'] != 'complete':
                    raise ValueError(f'Incomplete fixed grid: {phase}')
            manifest_path = prediction/'prediction_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest['seed'] != seed or manifest['velocity_arm'] != arm or manifest['future_target_read']:
                raise ValueError('Prediction arm/seed/source-only contract mismatch')
            path = prediction/'predictions.npz'
            if sha256(path) != manifest['prediction_sha256']:
                raise ValueError('Prediction modified after freezing')
            for name in ('predictions.npz', 'prediction_manifest.json'):
                if provenance['outputs'][name] != sha256(prediction/name):
                    raise ValueError('Prediction provenance mismatch')
            frozen.append(dict(arm=arm, seed=seed, path=str(path), sha256=sha256(path)))
    protocol = dict(arms=arms, seeds=seeds, comparisons=['static', 'gfg_joint_shuffled', 'gfg_frozen'],
        metrics=['energy_distance', 'log_variance_error', 'condition_centered_gene_mean_mse'],
        bootstrap_repetitions=10000, bootstrap_seed=20260914, confidence=1-.05/9,
        scope='all_four_development_TFs_no_posthoc_subgroup', confirmation='SEALED',
        shuffle='conditional_U_permutation_before_GFG_within_condition_depth_local_S_blocks',
        frozen_arm_limit='same_A_warmup_but_no_native_GFG_updates_during_C',
        population_inference='exploratory_four_conditions_not_confirmatory')
    # Run creation hashes source predictions only. Frozen grid is written before targets are opened.
    with Run(output, stage='GFG_fixed_grid_development_evaluation', kind='development',
             config=protocol, inputs=[r['path'] for r in frozen], seed=seeds) as run:
        save_json(run.directory/'frozen_grid.json', frozen)
        rows = []
        for record in frozen:
            metrics = evaluate_prediction(Path(record['path']).parent, fold/'validation_target.npz',
                run.directory/f"{record['arm']}_seed{record['seed']}",
                target_genes_path=fold/'validation_target_genes.npz')
            for row in metrics:
                row['log_variance_error'] = float(abs(np.log(max(row['variance_ratio'], 1e-12))))
            rows.extend(metrics)
        conditions = sorted({r['condition'] for r in rows})
        if len(conditions) != 4:
            raise ValueError('Expected the fixed four development TFs')
        comparisons = []
        for metric in protocol['metrics']:
            for control in protocol['comparisons']:
                comparison = paired_condition_bootstrap(rows, real_arm='gfg_joint', control_arm=control,
                    seeds=seeds, conditions=conditions, n_bootstrap=protocol['bootstrap_repetitions'],
                    seed=protocol['bootstrap_seed'], confidence=protocol['confidence'], metric=metric)
                comparison['metric'] = metric
                comparisons.append(comparison)
        means = {arm: {metric: float(np.mean([r[metric] for r in rows if r['arm'] == arm]))
                       for metric in protocol['metrics']} for arm in arms}
        primary = [r for r in comparisons if r['control_arm'] in ('static', 'gfg_joint_shuffled')]
        result = dict(status='GFG_PILOT_EVALUATED', conditions=conditions, means=means,
            comparisons=comparisons, all_primary_intervals_positive=all(r['ci_low'] > 0 for r in primary),
            formal_G1='NOT_RUN', confirmation='SEALED', real_multimodal_advantage='NOT_ESTABLISHED')
        save_csv(run.directory/'metrics.csv', rows)
        save_csv(run.directory/'comparisons.csv', comparisons)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# GFG fixed-budget pilot\n\n'
            +json.dumps(result, indent=2, ensure_ascii=False)
            +'\n\nGain = control error minus joint GFG error; positive is better. '
             'Intervals resample conditions after averaging seeds, with Bonferroni correction across '
             'nine planned comparisons. Four repeatedly inspected development TFs cannot establish '
             'confirmatory generalization. No CellFlow official 15-metric claim.\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/veloroute_gfg_joint_20260914.yaml')
    parser.add_argument('--root', default='outputs/veloroute_gfg_pilot_20260914')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_grid(load_config(args.config), args.root, args.output)))
