"""File-level end-to-end acceptance using unmistakably synthetic S/U counts."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import Run, save_csv, save_json
from .experiments import evaluate_prediction, predict_model, train_model
from .latent import prepare_fold


def run_pipeline_smoke(output, *, seed=20260912, steps=12):
    config = {'seed': seed, 'steps': steps, 'data_origin': 'veloroute_synthetic_counts_v1', 'real_G1': 'NOT_RUN'}
    with Run(output, stage='file_pipeline_smoke', kind='synthetic', config=config, seed=seed) as run:
        rng = np.random.default_rng(seed)
        conditions = np.array([f'TOY_TF_{i}' for i in range(7)])
        roles = {c: 'train' if i < 3 else 'validation' if i < 5 else 'confirmation' for i, c in enumerate(conditions)}
        genes = [f'TOY_GENE_{i}' for i in range(24)]
        for day in (4, 5):
            names = np.repeat(conditions, 20)
            s = rng.poisson(20, (len(names), len(genes))) + (day-4)*rng.poisson(2, (len(names), len(genes)))
            u = rng.poisson(6, (len(names), len(genes)))
            obs = pd.DataFrame({'condition': names, 'role': [roles[c] for c in names], 'day': day,
                                'technical_qc_pass': True, 'guide_call_pass': True},
                               index=[f'TOY_day{day}:cell{i}' for i in range(len(names))])
            data = ad.AnnData(sparse.csr_matrix(s), obs=obs, var=pd.DataFrame(index=genes),
                              layers={'spliced': sparse.csr_matrix(s), 'unspliced': sparse.csr_matrix(u)})
            data.uns['veloroute'] = {'data_origin': 'veloroute_synthetic_counts_v1', 'formal_ready': False}
            data.write_h5ad(run.directory/f'toy_day{day}.h5ad')
        latent_config = dict(seed=seed, n_genes=24, n_components=6, target_sum=10000., upper_quantile=.8, minimum_u_cells=3)
        fold = run.directory/'fold'
        prepare_fold(run.directory/'toy_day4.h5ad', run.directory/'toy_day5.h5ad', latent_config, fold, kind='synthetic')
        condition_file = run.directory/'synthetic_conditions.npz'
        with condition_file.open('xb') as stream:
            np.savez_compressed(stream, conditions=conditions, embeddings=np.eye(7, dtype=np.float32),
                metadata_json=np.array(json.dumps({'source': 'synthetic_fixture_only_not_ESM', 'kind': 'synthetic', 'frozen': True})))
        model_config = {'model': {'hidden_dim': 32, 'router_hidden_dim': 32, 'time_hidden_dim': 16,
                                  'residual_blocks': 1, 'expert_rank': 8, 'n_experts': 2, 'top_k': 2, 'use_velocity': True},
                        'steps': steps, 'warmup_steps': max(1, steps//3), 'batch_size': 16, 'learning_rate': .001,
                        'lambda_dynamic': 0., 'velocity_arm': 'real', 'cpu_threads': 2, 'device': 'cpu'}
        rows = []
        for arm in ('real', 'static', 'shuffled'):
            arm_config = {**model_config, 'velocity_arm': arm}
            train_model(fold/'train_source.npz', fold/'train_target.npz', condition_file, arm_config,
                        run.directory/f'train_{arm}', seed=seed, engineering=True)
            predict_model(run.directory/f'train_{arm}'/'model.pt', fold/'validation_source.npz', condition_file,
                          run.directory/f'predict_{arm}', seed=seed, transform_path=fold/'transform.npz')
            rows.extend(evaluate_prediction(run.directory/f'predict_{arm}', fold/'validation_target.npz',
                                            run.directory/f'evaluate_{arm}', target_genes_path=fold/'validation_target_genes.npz'))
        save_csv(run.directory/'metrics.csv', rows)
        summary = {'status': 'ENGINEERING_PASS', 'raw_input': 'synthetic_S_U_AnnData',
                   'stages': ['training_only_transform', 'sealed_role_packs', 'unpaired_OT_training',
                              'router_experts', 'checkpoint', 'source_only_prediction', 'frozen_hash', 'separate_evaluation'],
                   'arms': ['real', 'static', 'shuffled'], 'evaluation_rows': len(rows),
                   'confirmation_evaluated': False, 'G1': 'NOT_RUN', 'real_data_scientific_evidence': False}
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# File-level pipeline engineering test\n\n'+json.dumps(summary, indent=2)
            +'\n\nToy counts and toy one-hot conditions are synthetic fixtures, not RENGE/ESM. '
             'Real/negative-control metric differences here are not scientific success criteria.\n')
    return summary
