"""Fixed-budget developer-side router comparisons. Freeze all predictions first."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import Run, load_config, save_csv, save_json, sha256
from .experiments import evaluate_prediction, predict_model, train_model
from .latent import FrozenSplicingTransform, load_pack
from .metrics import paired_condition_bootstrap
from .model import load_checkpoint


def baseline_prediction(source_path, shift, transform_path, output, seed, arm):
    source, sm = load_pack(source_path, expected_side='source')
    if sm['role'] != 'validation' or sm['day'] != 4:
        raise ValueError('Only development source baselines are allowed')
    transform = FrozenSplicingTransform.load(transform_path)
    with Run(output, stage='baseline_prediction', kind=sm['kind'], config={'arm': arm, 'seed': seed, 'source_only': True},
             inputs=[source_path, transform_path], seed=seed) as run:
        z = source['z']+shift
        with (run.directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, z=z, q=np.ones((len(z), 1)), modes=np.zeros(len(z), dtype=int),
                conditions=source['conditions'], source_ids=source['cell_ids'], gene_logspliced=transform.decode(z),
                gene_ids=np.array(transform.gene_ids)[transform.selected])
        save_json(run.directory/'prediction_manifest.json', {'prediction_sha256': sha256(run.directory/'predictions.npz'),
            'transform_hash': sm['transform_hash'], 'role': sm['role'], 'kind': sm['kind'], 'seed': seed,
            'task': 'ER-short', 'velocity_arm': arm, 'future_target_read': False, 'gene_space_decoded': True,
            'research_status': 'exploratory_not_preregistered'})


def run_router_increment(config_path, output):
    config = load_config(config_path)
    if config['research_status'] != 'exploratory_not_preregistered':
        raise ValueError('Router increment entry is an explicit exploratory analysis')
    fold, condition = Path(config['fold']), Path(config['conditions'])
    with Run(output, stage='exploratory_router_increment', kind='engineering', config=config,
             inputs=[config_path, condition, fold/'provenance.json']) as run:
        source, _ = load_pack(fold/'train_source.npz', expected_side='source')
        target, _ = load_pack(fold/'train_target.npz', expected_side='target')
        shifts = [target['z'][target['conditions'] == label].mean(0)-source['z'][source['conditions'] == label].mean(0)
                  for label in sorted(set(source['conditions']))]
        shift = np.stack(shifts).mean(0)
        np.save(run.directory/'training_global_shift.npy', shift, allow_pickle=False)
        records, parameters = [], []
        for seed in config['seeds']:
            for arm in config['arms']:
                name = arm['name']
                model_config = {'model': {**config['model'], 'n_experts': arm['experts'], 'top_k': min(2, arm['experts'])},
                    **{key: config[key] for key in ('steps', 'warmup_steps', 'batch_size', 'learning_rate', 'device', 'cpu_threads')},
                    'velocity_arm': arm['velocity'], 'lambda_dynamic': 0., 'lambda_pair_dynamic': arm['pair_dynamic']}
                directory = run.directory/f'{name}_seed{seed}'
                save_json(run.directory/'status.json', {'status': 'training', 'arm': name, 'seed': seed})
                train_model(fold/'train_source.npz', fold/'train_target.npz', condition, model_config,
                            directory/'train', seed=seed, exploratory_real=True)
                checkpoint = directory/'train/model.pt'
                model, _ = load_checkpoint(checkpoint)
                parameters.append({'arm': name, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters())})
                predict_model(checkpoint, fold/'validation_source.npz', condition, directory/'predict', seed=seed,
                              transform_path=fold/'transform.npz')
                records.append({'arm': name, 'seed': seed, 'directory': str(directory/'predict'),
                                'sha256': sha256(directory/'predict/predictions.npz')})
            for name in config['baselines']:
                directory = run.directory/f'{name}_seed{seed}'/'predict'
                baseline_prediction(fold/'validation_source.npz', shift if name == 'training_global_mean_shift' else 0.,
                                    fold/'transform.npz', directory, seed, name)
                records.append({'arm': name, 'seed': seed, 'directory': str(directory), 'sha256': sha256(directory/'predictions.npz')})
        save_json(run.directory/'frozen_predictions.json', records)
        save_csv(run.directory/'parameters.csv', parameters)
        # No held-out future expression was loaded until the complete experiment grid was predicted.
        rows = []
        for record in records:
            directory = Path(record['directory'])
            if sha256(directory/'predictions.npz') != record['sha256']:
                raise ValueError('Frozen router prediction changed')
            values = evaluate_prediction(directory, fold/'validation_target.npz', directory.parent/'evaluate',
                                          target_genes_path=fold/'validation_target_genes.npz')
            for value in values:
                value['arm'] = record['arm']
            if {v['condition'] for v in values} != set(config['validation_conditions']):
                raise ValueError('Unexpected development condition coverage')
            rows.extend(values)
        save_csv(run.directory/'metrics.csv', rows)
        comparisons = [paired_condition_bootstrap(rows, real_arm=real, control_arm=control, seeds=config['seeds'],
            conditions=config['validation_conditions'], n_bootstrap=config['n_bootstrap'], seed=config['bootstrap_seed'],
            confidence=1-config['familywise_alpha']/len(config['comparisons'])) for real, control in config['comparisons']]
        save_json(run.directory/'comparisons.json', comparisons)
        summary = {'status': 'EXPLORATORY_ROUTER_COMPARISONS_COMPLETE', 'learned_runs': len(parameters),
                   'prediction_runs': len(records), 'metric_rows': len(rows), 'confirmation_evaluated': False,
                   'formal_G1': 'NOT_RUN', 'velocity_router_vs_static': comparisons[0],
                   'velocity_router_vs_shuffled': comparisons[1]}
        save_json(run.directory/'summary.json', summary)
        save_json(run.directory/'status.json', {'status': 'complete'})
        (run.directory/'RESULTS.md').write_text('# Exploratory router increment\n\n'+json.dumps(summary, indent=2)
            +'\n\n'+json.dumps(comparisons, indent=2)+'\n\nEffects are development-only; no architecture search or outcome-based early stopping was used.\n')
    return summary
