"""Configured full architecture: synthetic raw S/U -> full training -> evaluation."""
from __future__ import annotations

import json
from dataclasses import asdict

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import Run, save_json
from .experiments import evaluate_prediction
from .full_experiments import predict_full_model, train_full_model
from .full_model import FullConfig
from .latent import prepare_fold


def run_full_smoke(config, output, *, config_path=None, seed=20260913, device='cpu'):
    cfg = FullConfig(**config['model'])
    training = {**config['training'], 'stage_a_steps': 2, 'stage_b_steps': 3, 'stage_c_steps': 12,
                'batch_size': 8, 'e_interval': 1, 'activation_interval': 1, 'reference_interval': 4,
                'usage_interval': 12, 'corruption_interval': 4, 'checkpoint_interval': 6, 'seed': seed}
    execution = {**config, 'model': asdict(cfg), 'training': training, 'device': device, 'cpu_threads': 2}
    with Run(output, stage='configured_full_file_smoke', kind='synthetic', config=execution,
             inputs=[config_path] if config_path else [], seed=seed) as run:
        rng = np.random.default_rng(seed)
        conditions = ['TOY_A', 'TOY_B', 'TOY_AB', 'TOY_C', 'TOY_D']
        roles = dict(zip(conditions, ['train', 'train', 'train', 'validation', 'confirmation']))
        genes = max(24, cfg.state_dim*2)
        per_condition = max(12, cfg.state_dim//3+2)
        for day in (4, 5):
            names = np.repeat(conditions, per_condition)
            s = rng.poisson(20+(day-4), (len(names), genes))
            u = rng.poisson(6, s.shape)
            obs = pd.DataFrame({'condition': names, 'role': [roles[name] for name in names], 'day': day,
                                'technical_qc_pass': True, 'guide_call_pass': True},
                               index=[f'TOY:{day}:{i}' for i in range(len(names))])
            data = ad.AnnData(sparse.csr_matrix(s), obs=obs,
                var=pd.DataFrame(index=[f'TOY_GENE_{i}' for i in range(genes)]),
                layers={'spliced': sparse.csr_matrix(s), 'unspliced': sparse.csr_matrix(u)})
            data.uns['veloroute'] = {'data_origin': 'veloroute_synthetic_counts_v1', 'formal_ready': False}
            data.write_h5ad(run.directory/f'day{day}.h5ad')
        fold = run.directory/'fold'
        prepare_fold(run.directory/'day4.h5ad', run.directory/'day5.h5ad', dict(seed=seed, n_genes=genes,
            n_components=cfg.state_dim, target_sum=10000., upper_quantile=.8, minimum_u_cells=3), fold, kind='synthetic')
        conditions_path = run.directory/'toy_protein_embeddings.npz'
        with conditions_path.open('xb') as stream:
            np.savez_compressed(stream, conditions=np.array(['TOY_A', 'TOY_B', 'TOY_C', 'TOY_D']),
                embeddings=rng.normal(0, .1, (4, cfg.esm_dim)).astype(np.float32),
                metadata_json=np.array(json.dumps({'source': 'synthetic_fixture_NOT_ESM_weights', 'frozen': True, 'kind': 'synthetic'})))
        combination = run.directory/'combinations.json'
        save_json(combination, {'TOY_AB': ['TOY_A', 'TOY_B']})
        summary = train_full_model(fold/'train_source.npz', fold/'train_target.npz', conditions_path,
            execution, run.directory/'train', transform_path=fold/'transform.npz', combination_path=combination,
            synthetic_engineering=True)
        prediction = predict_full_model(run.directory/'train/model.pt', fold/'validation_source.npz', conditions_path,
            run.directory/'predict', transform_path=fold/'transform.npz', combination_path=combination,
            seed=seed, device=device, batch_size=16)
        metrics = evaluate_prediction(run.directory/'predict', fold/'validation_target.npz', run.directory/'evaluate',
                                      target_genes_path=fold/'validation_target_genes.npz')
        if not summary['teacher_updates'] or summary['corruption_batches'] != 3 or not prediction['gene_space_decoded']:
            raise RuntimeError('Full smoke did not exercise all required training/inference stages')
        result = {'status': 'FULL_ENGINEERING_PASS', 'configured_state_dim': cfg.state_dim,
                  'configured_ESM_dimension': cfg.esm_dim, 'maximum_experts': cfg.max_experts,
                  'teacher_updates': summary['teacher_updates'], 'corruption_batches': summary['corruption_batches'],
                  'deployment_parameters': summary['deployment_parameters_including_frozen_fallback'],
                  'evaluation_conditions': len(metrics), 'combination_attention_used': True,
                  'checkpoint_and_gene_decoding': True, 'confirmation_evaluated': False,
                  'real_data_used': False, 'device': device, 'G1': 'NOT_RUN'}
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Full-model engineering acceptance\n\n'+json.dumps(result, indent=2)
            +'\n\nSynthetic counts and synthetic protein tokens exercise the full configured architecture. '
             'This is not evidence of real velocity benefit or predictive quality.\n')
    return result
