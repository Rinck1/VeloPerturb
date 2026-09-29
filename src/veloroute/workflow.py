"""One entry point from completed counts to frozen packs and gated experiments."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .artifacts import Run, load_config, save_csv, save_json, sha256, utc_now
from .latent import load_pack, prepare_fold
from .protocol import assert_real_training_allowed, protocol_errors


def run_workflow(config_path, output):
    config = load_config(config_path)
    if config.get('formal_training_requires_G1') is not True:
        raise ValueError('This workflow cannot bypass the formal G1 gate')
    deadline = time.monotonic()+config['wait_seconds']
    with Run(output, stage='data_to_experiment_workflow', kind='engineering', config=config, inputs=[config_path]) as run:
        def state(message):
            save_json(run.directory/'status.json', {'status': message, 'pid': os.getpid(), 'updated_utc': utc_now()})

        def wait_artifact(path):
            path = Path(path)
            state(f'waiting_for_{path}')
            while True:
                provenance_path = path.parent/'provenance.json'
                if provenance_path.is_file():
                    record = json.loads(provenance_path.read_text())
                    if record.get('status') == 'failed':
                        raise RuntimeError(f'Upstream artifact failed: {provenance_path}; {record.get("error")}')
                    if record.get('status') == 'complete':
                        if not path.is_file() or record['outputs'].get(path.name) != sha256(path):
                            raise ValueError(f'Completed artifact missing or modified: {path}')
                        return path
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Waiting for completed upstream artifact: {path}')
                time.sleep(5)

        try:
            source = wait_artifact(config['source_counts'])
            target = wait_artifact(config['target_counts'])
            formal = bool(config.get('frozen_protocol') and config.get('g1_decision'))
            if formal:
                if not config.get('frozen_fold'):
                    raise ValueError('Formal continuation requires the exact frozen_fold used for G1; no refitting')
                fold = Path(config['frozen_fold']).resolve()
                wait_artifact(fold/'transform.npz')
                summary = json.loads((fold/'summary.json').read_text())
                if summary['kind'] != 'development':
                    raise ValueError('Formal continuation cannot promote an engineering fold')
            else:
                state('fitting_training_only_S_U_transforms')
                fold = run.directory/'fold'
                summary = prepare_fold(source, target, load_config(config['latent_config']), fold, kind='engineering')
            conditions = wait_artifact(config['conditions'])
            evidence = {'stage': 'data_pack_binding', 'formal_ready': formal,
                        'source_counts_sha256': sha256(source), 'target_counts_sha256': sha256(target),
                        'condition_embeddings_sha256': sha256(conditions),
                        'transform_sha256': sha256(fold/'transform.npz'),
                        **{f'{role}_{side}_sha256': sha256(fold/f'{role}_{side}.npz')
                           for role in ('train', 'validation', 'confirmation') for side in ('source', 'target')}}
            save_json(run.directory/'data_provenance.json', evidence)
            save_json(run.directory/'velocity_provenance.json', {'transform_sha256': summary['transform_hash'],
                      'estimator': summary['estimator'], 'fit_scope': 'training_TFs_only_day4_and_day5',
                      'no_condition_input': True, 'no_heldout_neighbors': True, 'reference_GFG_VeloVI_used': False,
                      'incremental_value_verified': False})
            coverage = []
            for role in ('train', 'validation'):
                for side in ('source', 'target'):
                    values, metadata = load_pack(fold/f'{role}_{side}.npz', expected_side=side)
                    for label in sorted(set(values['conditions'])):
                        coverage.append({'role': role, 'side': side, 'condition': str(label),
                                         'cells': int((values['conditions'] == label).sum())})
            save_csv(run.directory/'development_coverage.csv', coverage)
            if not formal:
                result = {'status': 'ENGINEERING_PIPELINE_READY_FOR_PROTOCOL_REVIEW',
                          'real_counts_and_frozen_packs': True, 'frozen_ESM2_conditions': True,
                          'G1': 'NOT_RUN', 'real_model_training': 'NOT_RUN', 'confirmation_evaluation': 'SEALED',
                          'protocol_readiness_errors': protocol_errors(load_config(config['draft_protocol']))}
                save_json(run.directory/'summary.json', result)
                state('complete_engineering_review_required_before_G1')
                (run.directory/'RESULTS.md').write_text('# Data-to-experiment workflow\n\n'+json.dumps(result, indent=2)
                    +'\n\nCounts, training-only velocity/PCA, split packs and public ESM2 conditions are connected. '
                     'No formal G1 or router training is inferred. See data_provenance.json, velocity_provenance.json '
                     'and development_coverage.csv for the protocol-review handoff.\n')
                return result
            frozen, decision = [json.loads(Path(config[k]).read_text()) for k in ('frozen_protocol', 'g1_decision')]
            assert_real_training_allowed(frozen, decision)
            # The hash-bound packs used for G1 cannot be regenerated and silently
            # substituted here: train_model verifies the exact frozen pack hashes.
            from .experiments import evaluate_prediction, predict_model, train_model
            results = []
            for seed in config['seeds']:
                for arm in config['arms']:
                    model_config = {**load_config(config['model_config']), 'velocity_arm': arm}
                    if arm == 'static':
                        model_config['lambda_dynamic'] = 0.
                    arm_root = run.directory/f'{arm}_seed{seed}'
                    train_dir, pred_dir, eval_dir = [arm_root/stage for stage in ('train', 'predict', 'evaluate')]
                    state(f'train_{arm}_seed{seed}')
                    train_model(fold/'train_source.npz', fold/'train_target.npz', conditions, model_config, train_dir,
                                seed=seed, frozen_path=config['frozen_protocol'], decision_path=config['g1_decision'])
                    predict_model(train_dir/'model.pt', fold/'validation_source.npz', conditions, pred_dir, seed=seed,
                                  transform_path=fold/'transform.npz')
                    results.extend(evaluate_prediction(pred_dir, fold/'validation_target.npz', eval_dir,
                                                       target_genes_path=fold/'validation_target_genes.npz'))
            save_csv(run.directory/'metrics.csv', results)
            result = {'status': 'DEVELOPMENT_EXPERIMENTS_COMPLETE', 'confirmation_evaluation': 'SEALED', 'rows': len(results)}
            save_json(run.directory/'summary.json', result)
            state('complete')
            (run.directory/'RESULTS.md').write_text('# Development experiments\n\n'+json.dumps(result, indent=2)+'\n')
            return result
        except BaseException as error:
            state(f'failed: {type(error).__name__}: {error}')
            (run.directory/'RESULTS.md').write_text(f'# Workflow stopped\n\n{type(error).__name__}: {error}\n\nCompleted artifacts are preserved.\n')
            raise
