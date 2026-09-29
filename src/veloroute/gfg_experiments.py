"""Gene-level GFG inputs and joint full-model runs, with sealed confirmation."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .contracts import FitScope
from .full_experiments import condition_token_sets
from .full_model import FullConfig, FullVeloRoute, load_full_checkpoint, save_full_checkpoint
from .full_training import FullTrainConfig, FullTrainer
from .latent import FrozenSplicingTransform, load_pack
from .gpu_policy import enforce_gpu_policy


def prepare_gfg_inputs(config, output):
    fold, path = Path(config['fold']), Path(config['source_counts'])
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    with Run(output, stage='GFG_gene_input_preparation', kind='engineering', config=config,
             inputs=[path, fold/'transform.npz', fold/'train_source.npz', fold/'validation_source.npz']) as run:
        data = ad.read_h5ad(path, backed='r')
        if tuple(data.var_names) != transform.gene_ids:
            raise ValueError('GFG input gene order differs from frozen PCA')
        for role in ('train', 'validation'):
            pack, meta = load_pack(fold/f'{role}_source.npz', expected_side='source')
            indices = data.obs_names.get_indexer(pack['cell_ids'])
            if (indices < 0).any() or not (data.obs.iloc[indices]['role'].to_numpy() == role).all():
                raise ValueError('GFG barcode/role mismatch')
            sfull = data.layers['spliced'][indices]
            depth = np.asarray(sfull.sum(1)).ravel()
            np.testing.assert_allclose(depth, pack['depth'])
            factor = transform.target_sum/depth[:, None]
            s = sfull[:, transform.selected].toarray()*factor
            u = data.layers['unspliced'][indices][:, transform.selected].toarray()*factor
            z = (np.log1p(s)-transform.mean)@transform.components.T
            np.testing.assert_allclose(z, pack['z'], rtol=1e-4, atol=1e-5)
            with (run.directory/f'{role}_gene_source.npz').open('xb') as stream:
                np.savez_compressed(stream, gene_us=np.concatenate((u, s), 1).astype('float32'),
                    cell_ids=pack['cell_ids'], gene_ids=np.array(transform.gene_ids)[transform.selected],
                    metadata_json=np.array(json.dumps(dict(role=role, day=4, side='source_gene_US',
                        transform_hash=meta['transform_hash'], source_hash=sha256(fold/f'{role}_source.npz'),
                        normalization='source_S_library_10000_unlogged_US_no_neighbor_smoothing'))))
        data.file.close()
        summary = dict(status='GFG_GENE_INPUTS_READY', genes=len(transform.selected),
                       confirmation_processed=False, reference_estimator='GFG_decoder_JVP')
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# GFG gene input preparation\n\n'+json.dumps(summary, indent=2))
    return summary


def read_gene_input(path, source_path):
    pack, meta = load_pack(source_path, expected_side='source')
    if meta['role'] == 'confirmation':
        raise ValueError('Confirmation remains sealed')
    with np.load(path, allow_pickle=False) as data:
        gm = json.loads(str(data['metadata_json']))
        from .gfg_inputs import validate_gene_source
        validate_gene_source(data, pack, meta, gm, source_path)
        if gm['source_hash'] != sha256(source_path) or gm['transform_hash'] != meta['transform_hash']:
            raise ValueError('GFG gene input provenance mismatch')
        if gm['role'] != meta['role'] or not np.array_equal(pack['cell_ids'], data['cell_ids']):
            raise ValueError('GFG gene input barcode/role mismatch')
        return data['gene_us'].copy(), pack, meta


def training_gene_us(values, source, arm, seed):
    if arm == 'gfg_joint_shuffled':
        from .probes import permute_local
        values = values.copy()
        genes = values.shape[1]//2
        values[:, :genes] = permute_local(values[:, :genes], source, seed=seed, neighbors=10)[0]
    return values


def train_gfg(config, inputs, output, *, arm='gfg_joint', seed=0, pilot=False):
    enforce_gpu_policy(config['device'])
    fold, inputs = Path(config['fold']), Path(inputs)
    values, source, sm = read_gene_input(inputs/'train_gene_source.npz', fold/'train_source.npz')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    if sm['role'] != 'train' or tm['role'] != 'train' or sm['fit_ids_hash'] != tm['fit_ids_hash']:
        raise ValueError('GFG training requires matched training-fold source/target packs')
    if arm not in {'static', 'gfg_frozen', 'gfg_joint', 'gfg_joint_shuffled'}:
        raise ValueError('Unknown GFG experiment arm')
    values = training_gene_us(values, source, arm, seed)
    model_config = {**config['model'],
                    # static is a strict S-only router: keep the GFG module in
                    # the checkpoint schema for parameter matching, but do not
                    # expose it to the optimizer or the forward task path.
                    'joint_dynamics': arm not in {'gfg_frozen', 'static'},
                    'use_dynamics': arm != 'static',
                    'static_uses_trained_field': arm == 'static'}
    train_config = {**config['training'], 'seed': seed,
                    'implementation_revision': 'gradient_audit_20260921_v2'}
    if pilot:
        model_config.update(hidden_dim=256, residual_blocks=3, max_experts=2, initial_active=2,
                            use_adaptive_modes=False, use_intrinsic=False, use_gate=False, use_noise=False)
        train_config.update({k: config['pilot'][k] for k in ('stage_a_steps', 'stage_b_steps', 'stage_c_steps')})
        train_config.update(e_interval=50, reference_interval=200, checkpoint_interval=200, usage_interval=400)
    effective = {**config, 'model': model_config, 'training': train_config, 'arm': arm, 'pilot_run': pilot}
    c = FullConfig(**model_config)
    t = FullTrainConfig(**train_config)
    device = torch.device(config['device'])
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(seed)
    model = FullVeloRoute(c).to(device)
    checkpoint_hash = model.dynamics.core.load_pretrained(config['gfg_checkpoint'])
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    gene_us = torch.as_tensor(values, device=device)
    if arm != 'static':
        model.dynamics.prepare(gene_us, torch.as_tensor(transform.components, dtype=torch.float32, device=device),
            cell_ids=source['cell_ids'].tolist(), scope=FitScope(frozenset(source['cell_ids'])))
    tokens, mask, _ = condition_token_sets(config['conditions'], source['conditions'])
    z, y, protein = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in (source['z'], target['z'], tokens)]
    padding = torch.as_tensor(mask, dtype=torch.bool, device=device)
    trainer = FullTrainer(model, t)
    metadata = dict(architecture='GFG_joint_VeloRoute', kind=sm['kind'], task=sm['task'], arm=arm, seed=seed,
        research_status='exploratory_GFG_joint_NOT_formal_G1', transform_hash=sm['transform_hash'],
        fit_ids_hash=sm['fit_ids_hash'], condition_hash=sha256(config['conditions']),
        GFG_checkpoint_used=True, GFG_checkpoint_sha256=checkpoint_hash, GFG_joint=c.joint_dynamics,
        velocity_reference_estimator='GFG_decoder_JVP_joint' if c.joint_dynamics else 'GFG_decoder_JVP_frozen',
        GFG_pretraining='local_MouseBrain_seed0_not_foundation_release', condition_input_to_GFG=False,
        GFG_uncertainty='source_RNA_reconstruction_error_proxy_not_posterior_velocity_uncertainty',
        configuration=effective, confirmation='SEALED', teacher_deployed=False)
    input_paths = [inputs/'train_gene_source.npz', fold/'train_source.npz', fold/'train_target.npz',
                   fold/'transform.npz', config['conditions'], config['gfg_checkpoint']]
    if t.velocity_background:
        input_paths.append(t.velocity_background)
    with Run(output, stage='GFG_joint_training', kind='engineering', config=effective,
             inputs=input_paths, seed=seed) as run:
        def progress(row):
            if row['step'] % 10 == 0:
                save_json(run.directory/'status.json', {'status': 'running', **row})
        trainer.progress_callback = progress
        def checkpoint(current):
            directory = run.directory/'checkpoints'/f'step_{current.completed_c_steps:07d}'
            directory.mkdir(parents=True, exist_ok=False)
            save_full_checkpoint(directory/'model.pt', model, metadata=metadata)
            with (directory/'training_state.pt').open('xb') as stream:
                torch.save(current.state_dict(), stream)
            save_json(directory/'manifest.json', dict(config_hash=object_hash(effective),
                inputs={str(p): sha256(p) for p in input_paths},
                files={name: sha256(directory/name) for name in ('model.pt', 'training_state.pt')}))
        trace = trainer.fit(z, gene_us, torch.zeros_like(z), y, protein, padding,
            source_conditions=source['conditions'].tolist(), target_conditions=target['conditions'].tolist(),
            source_ids=source['cell_ids'].tolist(), target_ids=target['cell_ids'].tolist(),
            scope=FitScope(frozenset(list(source['cell_ids'])+list(target['cell_ids']))), checkpoint_callback=checkpoint)
        save_full_checkpoint(run.directory/'model.pt', model, metadata=metadata)
        save_csv(run.directory/'training_trace.csv', trace)
        gradients = [row['gfg_task_gradient_norm'] for row in trace if str(row['stage']).startswith('C')]
        summary = dict(status='GFG_TRAINING_COMPLETE', **metadata, completed_steps=trainer.completed_c_steps,
            gfg_task_gradient_max=max(gradients, default=0),
            gfg_task_gradient_median=float(np.median(gradients)) if gradients else 0.0,
            deployment_parameters=sum(p.numel() for p in model.parameters()),
            teacher_updates=trainer.teacher_updates, corruption_batches=trainer.corruptions)
        save_json(run.directory/'summary.json', summary)
        save_json(run.directory/'status.json', {'status': 'complete'})
        (run.directory/'RESULTS.md').write_text('# GFG joint training\n\n'+json.dumps(summary, indent=2))
    return summary


def predict_gfg(checkpoint, config, inputs, output, *, seed=0):
    enforce_gpu_policy(config['device'])
    fold = Path(config['fold'])
    values, source, sm = read_gene_input(Path(inputs)/'validation_gene_source.npz', fold/'validation_source.npz')
    model, metadata = load_full_checkpoint(checkpoint, map_location=config['device'])
    model.eval()
    if sm['transform_hash'] != metadata['transform_hash'] or sha256(config['conditions']) != metadata['condition_hash']:
        raise ValueError('GFG prediction transform/condition hash mismatch')
    values = training_gene_us(values, source, metadata['arm'], seed)
    tokens, mask, _ = condition_token_sets(config['conditions'], source['conditions'])
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    generator = torch.Generator(device=config['device']).manual_seed(seed)
    records = {key: [] for key in ('z', 'q', 'modes', 'gate')}
    with Run(output, stage='GFG_source_only_prediction', kind='engineering',
        config=dict(seed=seed, arm=metadata['arm'], source_only=True),
        inputs=[checkpoint, Path(inputs)/'validation_gene_source.npz', fold/'validation_source.npz', config['conditions']], seed=seed) as run:
        for start in range(0, len(values), 8):
            ix = slice(start, start+8)
            z, us, protein = [torch.as_tensor(v[ix], dtype=torch.float32, device=config['device'])
                              for v in (source['z'], values, tokens)]
            padding = torch.as_tensor(mask[ix], dtype=torch.bool, device=config['device'])
            result = model.predict(z, us, protein, padding, generator=generator,
                t0=metadata['configuration']['training']['time_start'],
                t1=metadata['configuration']['training']['time_end'],
                stochastic=not metadata['configuration']['pilot_run'],
                force_gate=config.get('prediction',{}).get('force_gate',1. if metadata['configuration']['pilot_run'] else None))
            for name, value in (('z', result.endpoint), ('q', result.probabilities), ('modes', result.modes), ('gate', result.gate)):
                records[name].append(value.detach().cpu().numpy())
        payload = {key: np.concatenate(value) for key, value in records.items()}
        payload.update(conditions=source['conditions'], source_ids=source['cell_ids'],
            gene_logspliced=transform.decode(payload['z']).astype('float32'),
            gene_ids=np.array(transform.gene_ids)[transform.selected])
        with (run.directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, **payload)
        save_json(run.directory/'prediction_manifest.json', dict(prediction_sha256=sha256(run.directory/'predictions.npz'),
            transform_hash=sm['transform_hash'], role='validation', kind=sm['kind'], seed=seed,
            task=sm['task'], velocity_arm=metadata['arm'], future_target_read=False, gene_space_decoded=True))
        (run.directory/'RESULTS.md').write_text('# GFG source-only prediction\n\nFuture targets were not loaded.\n')
    return {'status': 'GFG_SOURCE_PREDICTED', 'cells': len(values)}
