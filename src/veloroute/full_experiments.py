"""Full-model file runners, including explicit exploratory development training.

Exploratory is NOT a G1-GO or a preregistered result. Confirmation stays sealed.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .contracts import FitScope
from .experiments import condition_vectors, validate_pair, validate_real_artifacts
from .full_model import FullConfig, FullVeloRoute, load_full_checkpoint, save_full_checkpoint
from .full_training import FullTrainConfig, FullTrainer
from .latent import FrozenSplicingTransform, load_pack
from .protocol import assert_real_training_allowed


def condition_token_sets(path, labels, combination_path=None):
    """Explicit component map; never guess gene names by splitting label strings."""
    mapping = json.loads(Path(combination_path).read_text()) if combination_path else {}
    combinations = []
    for label in labels:
        genes = mapping.get(str(label), [str(label)])
        if not isinstance(genes, list) or not genes or any(not isinstance(g, str) or not g for g in genes) or len(set(genes)) != len(genes):
            raise ValueError('Combination manifest must contain nonempty distinct gene-name lists')
        combinations.append(sorted(genes))
    genes = sorted({gene for combination in combinations for gene in combination})
    vectors, meta = condition_vectors(path, genes)
    lookup = dict(zip(genes, vectors))
    width = max(map(len, combinations))
    tokens = np.zeros((len(labels), width, vectors.shape[1]), dtype=np.float32)
    mask = np.zeros((len(labels), width), dtype=bool)
    for i, combination in enumerate(combinations):
        for j, gene in enumerate(combination):
            tokens[i, j], mask[i, j] = lookup[gene], True
    return tokens, mask, meta


def _transform_for_pack(path, metadata, *, real):
    if path is None:
        if real:
            raise ValueError('Real full-model training needs the exact frozen transform, including the U projection origin')
        return None
    if sha256(path) != metadata['transform_hash']:
        raise ValueError('Frozen S/U/PCA transform hash mismatch')
    return FrozenSplicingTransform.load(path)


def train_full_model(source_path, target_path, condition_path, config, output, *, transform_path=None,
                     combination_path=None, frozen_path=None, decision_path=None,
                     synthetic_engineering=False, exploratory_real=False, resume=None):
    if config.get('model', {}).get('dynamics_backend') == 'gfg':
        raise ValueError('GFG requires the gene-level run_gfg_joint.py entry; PCA U packs are not GFG inputs')
    if synthetic_engineering and exploratory_real:
        raise ValueError('Synthetic engineering and real exploratory are distinct modes')
    source, sm = load_pack(source_path, expected_side='source')
    target, tm = load_pack(target_path, expected_side='target')
    validate_pair(source, sm, target, tm, 'train')
    tokens, mask, embedding_meta = condition_token_sets(condition_path, source['conditions'], combination_path)
    inputs = [source_path, target_path, condition_path]+([combination_path] if combination_path else [])
    if synthetic_engineering:
        if sm['kind'] != 'synthetic' or embedding_meta.get('kind') != 'synthetic':
            raise ValueError('Synthetic engineering cannot authorize real full-model training')
        kind, research_status, protocol_hash = 'synthetic', 'synthetic_engineering_only', None
    elif exploratory_real:
        if sm['kind'] not in {'engineering', 'development'} or embedding_meta.get('kind') == 'synthetic':
            raise ValueError('Exploratory real training requires real training-role packs and public protein priors')
        kind, research_status, protocol_hash = sm['kind'], 'exploratory_not_preregistered_G1_NOT_RUN', None
    else:
        if not frozen_path or not decision_path or sm['kind'] != 'development' or not sm.get('formal_ready'):
            raise ValueError('Formal full training requires G1-GO; exploratory training must be explicitly requested')
        frozen, decision = [json.loads(Path(p).read_text()) for p in (frozen_path, decision_path)]
        assert_real_training_allowed(frozen, decision)
        validate_real_artifacts(frozen, source_path, target_path, condition_path)
        evidence = json.loads(Path(frozen['inputs']['data_provenance']['path']).read_text())
        if combination_path and evidence.get('combination_manifest_sha256') != sha256(combination_path):
            raise ValueError('Formal combination definitions must be bound before G1')
        kind, research_status, protocol_hash = 'development', 'G1_approved_development', frozen['freeze_hash']
        inputs += [frozen_path, decision_path]
    transform = _transform_for_pack(transform_path, sm, real=not synthetic_engineering)
    if transform_path:
        inputs.append(transform_path)
    cfg = FullConfig(**{**config['model'], 'state_dim': source['z'].shape[1], 'esm_dim': tokens.shape[-1]})
    training = FullTrainConfig(**config['training'])
    if (training.time_start, training.time_end) != (4., 5.):
        raise ValueError('Current file packs declare ER-short day4 -> day5; other time tasks require their own packs')
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in this environment; no silent budget/device change')
    torch.set_num_threads(config.get('cpu_threads', 2))
    torch.manual_seed(training.seed)
    binding = {'source_sha256': sha256(source_path), 'target_sha256': sha256(target_path),
               'condition_sha256': sha256(condition_path), 'combination_sha256': sha256(combination_path) if combination_path else None,
               'transform_sha256': sha256(transform_path) if transform_path else sm['transform_hash']}
    metadata = {'architecture': 'veloroute_full_v1', 'kind': kind, 'research_status': research_status,
                'protocol_hash': protocol_hash, 'transform_hash': sm['transform_hash'], 'fit_ids_hash': sm['fit_ids_hash'],
                'condition_hash': binding['condition_sha256'], 'combination_hash': binding['combination_sha256'],
                'input_binding': binding, 'task': 'ER-short', 'seed': training.seed, 'velocity_arm': 'full',
                'configuration': config, 'reference_fit': 'training_source_cells_only',
                'velocity_reference_estimator': sm['estimator'], 'GFG_checkpoint_used': False,
                'confirmation': 'SEALED', 'teacher_deployed': False,
                'u_input_space': 'uncentered_log1p_U_PCA_projection' if transform else 'synthetic_projected_U',
                'trainable_state_semantics': 'identity_anchored_PCA_denoiser_frozen_after_A'}
    if resume:
        resume = Path(resume)
        manifest = json.loads((resume/'checkpoint_manifest.json').read_text())
        if manifest['input_binding'] != binding or manifest['config_hash'] != object_hash(config):
            raise ValueError('Resume input/config hash mismatch')
        for name in ('model.pt', 'training_state.pt'):
            if sha256(resume/name) != manifest['files'][name]:
                raise ValueError('Resume checkpoint was modified or incomplete')
        model, old_meta = load_full_checkpoint(resume/'model.pt', map_location=device)
        if old_meta['research_status'] != research_status:
            raise ValueError('Cannot promote exploratory/synthetic checkpoints through resume')
        trainer = FullTrainer(model, training)
        trainer.load_state_dict(torch.load(resume/'training_state.pt', map_location=device, weights_only=True))
        inputs += [resume/'checkpoint_manifest.json', resume/'model.pt', resume/'training_state.pt']
    else:
        model = FullVeloRoute(cfg).to(device)
        trainer = FullTrainer(model, training)
    # Reintroduce the frozen U projection origin so zero-U corruption means log U=0.
    unspliced = source['u_features'] + (transform.u_mean if transform else 0)
    convert = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    s, u, v, y, protein = map(convert, (source['z'], unspliced, source['velocity'], target['z'], tokens))
    padding = torch.as_tensor(mask, dtype=torch.bool, device=device)
    scope = FitScope(frozenset(list(source['cell_ids'])+list(target['cell_ids'])))
    with Run(output, stage='full_model_training', kind=kind, config=config, inputs=inputs, seed=training.seed) as run:
        def progress(row):
            if row['step'] % 25 == 0:
                save_json(run.directory/'status.json', {'status': 'training', **row,
                          'research_status': research_status})
        trainer.progress_callback = progress
        def checkpoint(current):
            directory = run.directory/'checkpoints'/f'step_{current.completed_c_steps:07d}'
            directory.mkdir(parents=True, exist_ok=False)
            save_full_checkpoint(directory/'model.pt', current.model, metadata=metadata)
            with (directory/'training_state.pt').open('xb') as stream:
                torch.save(current.state_dict(), stream)
            save_json(directory/'checkpoint_manifest.json', {'input_binding': binding, 'config_hash': object_hash(config),
                'files': {name: sha256(directory/name) for name in ('model.pt', 'training_state.pt')},
                'completed_c_steps': current.completed_c_steps, 'status': 'complete'})
            save_json(run.directory/'status.json', {'status': 'training', 'completed_c_steps': current.completed_c_steps,
                      'total_c_steps': training.stage_c_steps, 'checkpoint': str(directory)})
        trace = trainer.fit(s, u, v, y, protein, padding, source_conditions=source['conditions'].tolist(),
            target_conditions=target['conditions'].tolist(), source_ids=source['cell_ids'].tolist(),
            target_ids=target['cell_ids'].tolist(), scope=scope, checkpoint_callback=checkpoint)
        if trainer.completed_c_steps % training.checkpoint_interval:
            checkpoint(trainer)
        save_full_checkpoint(run.directory/'model.pt', model, metadata=metadata)
        save_csv(run.directory/'training_trace.csv', trace)
        summary = {**metadata, 'status': 'FULL_TRAINING_COMPLETE', 'completed_c_steps': trainer.completed_c_steps,
                   'teacher_updates': trainer.teacher_updates, 'E_rounds': trainer.e_round,
                   'corruption_batches': trainer.corruptions, 'active_modes': int(model.active_modes.sum()),
                   'retired_modes': torch.where(model.mode_controller.retired)[0].cpu().tolist(),
                   'deployment_parameters_including_frozen_fallback': sum(p.numel() for p in model.parameters()),
                   'frozen_fallback_parameters': sum(p.numel() for p in model.static_field.parameters())+
                                                 sum(p.numel() for p in model.static_condition_encoder.parameters()),
                   'teacher_training_only_parameters': sum(p.numel() for p in trainer.teacher.parameters()),
                   'reference_cells': len(model.reference.positions), 'device': str(device)}
        save_json(run.directory/'summary.json', summary)
        save_json(run.directory/'status.json', {'status': 'complete', 'completed_c_steps': trainer.completed_c_steps})
        (run.directory/'RESULTS.md').write_text('# Full VeloRoute training\n\n'+json.dumps(summary, indent=2)
            +'\n\nA full-model optimization run is not evidence of velocity/router benefit. '
             'The frozen static fallback is an additional parameter/storage cost. No confirmation target was read.\n')
    return summary


def predict_full_model(checkpoint, source_path, condition_path, output, *, transform_path=None, combination_path=None,
                       seed=0, repeats=1, batch_size=256, device='cpu', stochastic=True, force_gate=None):
    if force_gate not in (None, 0., 1.):
        raise ValueError('Gate intervention is only learned, frozen-static, or all-dynamic')
    source, sm = load_pack(source_path, expected_side='source')
    if sm.get('role') == 'confirmation':
        raise ValueError('Confirmation remains sealed; full engineering is not a final release approval')
    model, metadata = load_full_checkpoint(checkpoint, map_location=device)
    if (sm.get('day'), sm.get('task'), sm.get('kind'), sm.get('fit_ids_hash')) != (
            4, 'ER-short', metadata['kind'], metadata['fit_ids_hash']):
        raise ValueError('Full prediction source day/task/kind/fit-scope mismatch')
    if metadata['transform_hash'] != sm['transform_hash'] or metadata['condition_hash'] != sha256(condition_path):
        raise ValueError('Full prediction frozen input hash mismatch')
    if metadata['combination_hash'] != (sha256(combination_path) if combination_path else None):
        raise ValueError('Combination component definitions changed after training')
    if repeats < 1 or batch_size < 1:
        raise ValueError('Need a positive prediction budget')
    transform = _transform_for_pack(transform_path, sm, real=metadata['kind'] != 'synthetic')
    tokens, mask, _ = condition_token_sets(condition_path, source['conditions'], combination_path)
    u = source['u_features']+(transform.u_mean if transform else 0)
    indices = np.tile(np.arange(len(source['z'])), repeats)
    inputs = [checkpoint, source_path, condition_path]+([transform_path] if transform_path else [])+([combination_path] if combination_path else [])
    config = {'seed': seed, 'repeats': repeats, 'batch_size': batch_size, 'stochastic': stochastic,
              'force_gate': force_gate, 'source_only': True}
    with Run(output, stage='full_frozen_prediction', kind=metadata['kind'], config=config, inputs=inputs, seed=seed) as run:
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = {name: [] for name in ('z', 'q', 'modes', 'gate')}
        displacement = {name: [] for name in ('static', 'intrinsic', 'perturbation', 'within_mode_noise')}
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            s, up, protein = [torch.as_tensor(value[ix], device=device, dtype=torch.float32) for value in (source['z'], u, tokens)]
            padding = torch.as_tensor(mask[ix], device=device, dtype=torch.bool)
            prediction = model.predict(s, up, protein, padding, t0=4., t1=5., generator=generator,
                                       stochastic=stochastic, force_gate=force_gate)
            for name, value in [('z', prediction.endpoint), ('q', prediction.probabilities), ('modes', prediction.modes), ('gate', prediction.gate)]:
                outputs[name].append(value.cpu().numpy())
            for name in displacement:
                displacement[name].append(prediction.contributions[name].cpu().numpy())
        payload = {name: np.concatenate(values) for name, values in outputs.items()}
        payload.update({f'displacement_{name}': np.concatenate(values) for name, values in displacement.items()})
        payload.update(conditions=source['conditions'][indices], source_ids=source['cell_ids'][indices])
        if transform:
            payload['gene_logspliced'] = transform.decode(payload['z']).astype(np.float32)
            payload['gene_ids'] = np.array(transform.gene_ids)[transform.selected]
        with (run.directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, **payload)
        manifest = {'prediction_sha256': sha256(run.directory/'predictions.npz'), 'transform_hash': sm['transform_hash'],
                    'checkpoint_sha256': sha256(checkpoint), 'role': sm['role'], 'kind': metadata['kind'],
                    'seed': seed, 'task': 'ER-short',
                    'velocity_arm': 'full' if force_gate is None else 'full_frozen_static' if force_gate == 0 else 'full_dynamic',
                    'future_target_read': False,
                    'gene_space_decoded': transform is not None,
                    'gene_space': 'log1p_S_normalized_by_S_library' if transform else None,
                    'research_status': metadata['research_status'], 'teacher_deployed': False}
        save_json(run.directory/'prediction_manifest.json', manifest)
        (run.directory/'RESULTS.md').write_text('# Full source-only prediction\n\nPrediction and attribution saved before separate target evaluation.\n')
    return manifest
