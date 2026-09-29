"""Unpaired training -> source-only frozen prediction -> separate evaluation.

Real training is G1-gated. Engineering execution accepts only synthetic packs.
No target input is accepted by the prediction function.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from .artifacts import Run, load_config, object_hash, save_csv, save_json, sha256
from .coupling import build_responsibilities, sinkhorn_log
from .latent import FrozenSplicingTransform, load_pack
from .metrics import energy_distance
from .model import ModelConfig, VeloRoute, flow_matching_loss, load_checkpoint, routing_loss, save_checkpoint
from .protocol import assert_real_training_allowed


def condition_vectors(path, conditions):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'conditions', 'embeddings', 'metadata_json'}:
            raise ValueError('Condition file requires named embeddings and provenance metadata')
        labels, vectors = data['conditions'].astype(str), data['embeddings'].astype(np.float32)
        metadata = json.loads(str(data['metadata_json']))
    if len(set(labels)) != len(labels) or vectors.ndim != 2 or len(vectors) != len(labels) or not np.isfinite(vectors).all():
        raise ValueError('Invalid condition embedding matrix')
    lookup = dict(zip(labels, vectors))
    if set(conditions)-set(lookup):
        raise ValueError(f'Missing condition embeddings: {sorted(set(conditions)-set(lookup))}; no zero/random fallback')
    if not metadata.get('source') or not metadata.get('frozen'):
        raise ValueError('Condition embeddings require a declared frozen source')
    return np.stack([lookup[c] for c in conditions]), metadata


def validate_pair(source, source_meta, target, target_meta, role):
    if source_meta.get('role') != role or target_meta.get('role') != role:
        raise ValueError(f'Expected {role} packs; cannot fit/evaluate another split')
    for key in ('transform_hash', 'fit_ids_hash', 'task', 'kind'):
        if not source_meta.get(key) or source_meta.get(key) != target_meta.get(key):
            raise ValueError(f'Source/target provenance mismatch: {key}')
    if (source_meta.get('day'), target_meta.get('day'), source_meta.get('task')) != (4, 5, 'ER-short'):
        raise ValueError('This runner is preregistered for ER-short only')
    if set(source['cell_ids']) & set(target['cell_ids']):
        raise ValueError('Source/target cell ID collision')
    if set(source['conditions']) != set(target['conditions']):
        raise ValueError('Source/target condition coverage differs; do not silently drop a condition')


def validate_real_artifacts(frozen, source_path, target_path, condition_path):
    evidence = json.loads(Path(frozen['inputs']['data_provenance']['path']).read_text())
    for key, path in [('train_source_sha256', source_path), ('train_target_sha256', target_path),
                      ('condition_embeddings_sha256', condition_path)]:
        if evidence.get(key) != sha256(path):
            raise ValueError(f'Real experiment inputs not bound by frozen data provenance: {key}')


def train_model(source_path, target_path, condition_path, config, output, *, seed=0,
                frozen_path=None, decision_path=None, engineering=False, exploratory_real=False):
    if engineering and exploratory_real:
        raise ValueError('Synthetic engineering and real exploration are distinct run modes')
    source, sm = load_pack(source_path, expected_side='source')
    target, tm = load_pack(target_path, expected_side='target')
    validate_pair(source, sm, target, tm, 'train')
    embeddings, embedding_meta = condition_vectors(condition_path, source['conditions'])
    input_paths = [source_path, target_path, condition_path]
    if engineering:
        if sm['kind'] != 'synthetic' or tm['kind'] != 'synthetic' or embedding_meta.get('kind') != 'synthetic':
            raise ValueError('Engineering training accepts synthetic packs only; real data needs G1')
        kind, protocol_hash = 'synthetic', 'synthetic_not_G1'
    elif exploratory_real:
        if sm['kind'] not in {'engineering', 'development'} or embedding_meta.get('kind') == 'synthetic':
            raise ValueError('Exploratory real models require real training packs and public protein priors')
        kind, protocol_hash = sm['kind'], 'exploratory_NOT_G1'
    else:
        if not frozen_path or not decision_path or sm['kind'] != 'development' or not sm.get('formal_ready'):
            raise ValueError('Formal training needs approved development data and G1 decision')
        frozen, decision = [json.loads(Path(p).read_text()) for p in (frozen_path, decision_path)]
        assert_real_training_allowed(frozen, decision)
        validate_real_artifacts(frozen, source_path, target_path, condition_path)
        kind, protocol_hash = 'development', frozen['freeze_hash']
        input_paths += [frozen_path, decision_path]
    cfg = ModelConfig(**{**config['model'], 'state_dim': source['z'].shape[1],
                         'velocity_dim': source['velocity'].shape[1], 'condition_dim': embeddings.shape[1]})
    if not 1 <= config['warmup_steps'] < config['steps'] or config['batch_size'] < 2:
        raise ValueError('Need bounded warmup, training budget and at least two samples per batch')
    if config.get('velocity_arm', 'real') not in {'real', 'static', 'shuffled'}:
        raise ValueError('Unknown velocity arm')
    if config['velocity_arm'] == 'static' and (config.get('lambda_dynamic', 0) or config.get('lambda_pair_dynamic', 0)):
        raise ValueError('Static baseline cannot use a dynamic coupling cost')
    torch.set_num_threads(config.get('cpu_threads', 2))
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; do not silently change execution budget')
    with Run(output, stage='unpaired_train', kind=kind, config=config, inputs=input_paths, seed=seed) as run:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = VeloRoute(cfg).to(device)
        field_optimizer = torch.optim.AdamW(model.field.parameters(), lr=config['learning_rate'])
        router_optimizer = torch.optim.AdamW(model.router.parameters(), lr=config['learning_rate'])
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        z, target_z, v, e = map(tensor, (source['z'], target['z'], source['velocity'], embeddings))
        if config['velocity_arm'] == 'static':
            v.zero_()
        elif config['velocity_arm'] == 'shuffled':
            from .probes import permute_local
            v = tensor(permute_local(source['velocity'], source, seed=seed, neighbors=10)[0])
        labels = sorted(set(source['conditions']))
        source_groups = {c: np.flatnonzero(source['conditions'] == c) for c in labels}
        target_groups = {c: np.flatnonzero(target['conditions'] == c) for c in labels}
        rates = []
        # Training-only initial expression OT, followed by displacement clustering.
        # These clusters are computational modes, never fate labels.
        for condition in labels:
            si = rng.choice(source_groups[condition], size=min(128, len(source_groups[condition])), replace=False)
            ti = rng.choice(target_groups[condition], size=min(128, len(target_groups[condition])), replace=False)
            cost = torch.cdist(z[si], target_z[ti]).square()
            plan = sinkhorn_log(cost/cost.mean().clamp_min(1e-8), epsilon=.1)
            sample = torch.multinomial(plan.flatten(), 128, replacement=True)
            ii, jj = sample//len(ti), sample%len(ti)
            rates.append((target_z[ti][jj]-z[si][ii]).cpu().numpy())
        centres = KMeans(n_clusters=cfg.n_experts, n_init=10, random_state=seed).fit(np.concatenate(rates)).cluster_centers_
        centres = tensor(centres)
        np.save(run.directory/'initial_displacement_centres.npy', centres.cpu().numpy(), allow_pickle=False)
        trace = []
        for step in range(config['steps']):
            condition = labels[step % len(labels)]
            si = rng.choice(source_groups[condition], config['batch_size'], replace=True)
            ti = rng.choice(target_groups[condition], config['batch_size'], replace=True)
            zs, vs, es, zt = z[si], v[si], e[si], target_z[ti]
            with torch.no_grad():
                if step < config['warmup_steps']:
                    cost = torch.cdist(zs, zt).square()
                    plan = sinkhorn_log(cost/cost.mean().clamp_min(1e-8), epsilon=.1)
                    rate = zt[None, :, :] - zs[:, None, :]
                    nearest = ((rate[:, :, None, :]-centres[None, None, :, :])**2).sum(-1).argmin(-1)
                    gamma = plan[:, :, None] * torch.nn.functional.one_hot(nearest, cfg.n_experts)
                    weights = gamma.sum(1); weights /= weights.sum(-1, keepdim=True).clamp_min(1e-12)
                else:
                    coupling = build_responsibilities(model, zs, zt, vs, es,
                        source_conditions=[condition]*len(zs), target_conditions=[condition]*len(zt),
                        lambda_dynamic=config['lambda_dynamic'], lambda_pair_dynamic=config.get('lambda_pair_dynamic', 0.))
                    gamma, weights = coupling['gamma'], coupling['source_responsibilities']
                draw = torch.multinomial(gamma.flatten(), config['batch_size'], replacement=True)
                kk = draw % cfg.n_experts
                jj = (draw//cfg.n_experts) % len(zt)
                ii = draw//(cfg.n_experts*len(zt))
                assignment = torch.nn.functional.one_hot(kk, cfg.n_experts).to(z.dtype)
            field_optimizer.zero_grad(set_to_none=True)
            fm = flow_matching_loss(model, zs[ii], zt[jj], es[ii], assignment, t0=4., t1=5.)
            fm.backward()
            torch.nn.utils.clip_grad_norm_(model.field.parameters(), 1.)
            field_optimizer.step()
            router_optimizer.zero_grad(set_to_none=True)
            ce = routing_loss(model, zs, vs, es, weights)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.)
            router_optimizer.step()
            if not torch.isfinite(fm+ce):
                raise FloatingPointError('Nonfinite training loss')
            trace.append({'step': step, 'condition': condition, 'field_loss': float(fm.detach()),
                          'router_loss': float(ce.detach()), 'phase': 'warmup' if step < config['warmup_steps'] else 'coupling'})
        metadata = {'kind': kind, 'protocol_hash': protocol_hash, 'transform_hash': sm['transform_hash'],
                    'research_status': 'exploratory_not_preregistered' if exploratory_real else 'synthetic' if engineering else 'G1_approved',
                    'fit_ids_hash': sm['fit_ids_hash'], 'task': 'ER-short', 'condition_hash': sha256(condition_path),
                    'seed': seed, 'velocity_arm': config['velocity_arm'], 'configuration': config,
                    'mode_semantics': 'training_OT_displacement_components_not_fate_truth',
                    'train_source_hash': sha256(source_path), 'train_target_hash': sha256(target_path)}
        save_checkpoint(run.directory/'model.pt', model, metadata=metadata)
        save_csv(run.directory/'training_trace.csv', trace)
        save_json(run.directory/'summary.json', metadata)
        (run.directory/'RESULTS.md').write_text('# Unpaired training\n\n'+json.dumps(metadata, indent=2)
            +'\n\nFixed training budget; no validation or confirmation target was used for fitting or early stopping.\n')
    return metadata


def predict_model(checkpoint, source_path, condition_path, output, *, seed=0, repeats=1, batch_size=256, transform_path=None):
    source, sm = load_pack(source_path, expected_side='source')
    if sm.get('role') == 'confirmation':
        raise ValueError('Confirmation requires the separately frozen multi-task release; development runner stays sealed')
    model, metadata = load_checkpoint(checkpoint)
    if (sm.get('day'), sm.get('task'), sm.get('kind'), sm.get('fit_ids_hash')) != (
            4, 'ER-short', metadata['kind'], metadata['fit_ids_hash']):
        raise ValueError('Source day/task/data-origin/fit-scope mismatch')
    if metadata['transform_hash'] != sm.get('transform_hash') or metadata['condition_hash'] != sha256(condition_path):
        raise ValueError('Checkpoint and source/condition transforms mismatch')
    if repeats < 1 or batch_size < 1:
        raise ValueError('Invalid generation budget')
    embedding, _ = condition_vectors(condition_path, source['conditions'])
    transform = None
    if transform_path:
        if sha256(transform_path) != sm['transform_hash']:
            raise ValueError('Inverse PCA transform does not match frozen source coordinates')
        transform = FrozenSplicingTransform.load(transform_path)
    elif metadata['kind'] != 'synthetic':
        raise ValueError('Real-data predictions require an explicit frozen inverse PCA transform')
    indices = np.tile(np.arange(len(source['z'])), repeats)
    velocity = source['velocity'].copy()
    if metadata['velocity_arm'] == 'static':
        velocity[:] = 0
    elif metadata['velocity_arm'] == 'shuffled':
        from .probes import permute_local
        velocity = permute_local(velocity, source, seed=seed, neighbors=10)[0]
    config = {'seed': seed, 'repeats': repeats, 'batch_size': batch_size, 'source_only': True}
    inputs = [checkpoint, source_path, condition_path] + ([transform_path] if transform_path else [])
    with Run(output, stage='frozen_prediction', kind=metadata['kind'], config=config, inputs=inputs, seed=seed) as run:
        model.eval()
        generator = torch.Generator().manual_seed(seed)
        predictions, probabilities, modes = [], [], []
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            z, v, e = [torch.tensor(x[ix], dtype=torch.float32) for x in (source['z'], velocity, embedding)]
            pred = model.predict(z, v, e, t0=4., t1=5., generator=generator)
            predictions.append(pred.endpoint.numpy()); probabilities.append(pred.probabilities.numpy()); modes.append(pred.modes.numpy())
        payload = dict(z=np.concatenate(predictions), q=np.concatenate(probabilities), modes=np.concatenate(modes),
                       conditions=source['conditions'][indices], source_ids=source['cell_ids'][indices])
        if transform is not None:
            payload['gene_logspliced'] = transform.decode(payload['z']).astype(np.float32)
            payload['gene_ids'] = np.array(transform.gene_ids)[transform.selected]
        with (run.directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, **payload)
        frozen = {'prediction_sha256': sha256(run.directory/'predictions.npz'), 'transform_hash': metadata['transform_hash'],
                  'checkpoint_sha256': sha256(checkpoint), 'role': sm['role'], 'kind': metadata['kind'],
                  'seed': seed, 'task': 'ER-short', 'velocity_arm': metadata['velocity_arm'], 'future_target_read': False,
                  'gene_space_decoded': transform is not None, 'gene_space': 'log1p_S_normalized_by_S_library' if transform else None}
        save_json(run.directory/'prediction_manifest.json', frozen)
        (run.directory/'RESULTS.md').write_text('# Frozen prediction\n\nSource-only predictions saved and hashed before evaluation.\n')
    return frozen


def evaluate_prediction(prediction_directory, target_path, output, *, target_genes_path=None):
    prediction_directory = Path(prediction_directory)
    provenance = json.loads((prediction_directory/'provenance.json').read_text())
    if provenance.get('status') != 'complete':
        raise ValueError('Prediction run must complete before any evaluation')
    manifest = json.loads((prediction_directory/'prediction_manifest.json').read_text())
    for filename in ('predictions.npz', 'prediction_manifest.json'):
        if provenance['outputs'].get(filename) != sha256(prediction_directory/filename):
            raise ValueError('Frozen prediction/manifest was modified')
    if sha256(prediction_directory/'predictions.npz') != manifest['prediction_sha256']:
        raise ValueError('Prediction hash mismatch')
    target, tm = load_pack(target_path, expected_side='target')
    if tm.get('kind') != manifest['kind'] or tm.get('task') != manifest['task']:
        raise ValueError('Evaluation data origin/task mismatch')
    if tm['role'] != manifest['role'] or tm['transform_hash'] != manifest['transform_hash'] or tm.get('day') != 5:
        raise ValueError('Target split/time/transform mismatch')
    if tm['role'] == 'confirmation':
        raise ValueError('Confirmation evaluation is sealed pending final multi-task preregistration')
    with np.load(prediction_directory/'predictions.npz', allow_pickle=False) as data:
        pred = {key: data[key] for key in data.files}
    if set(pred['source_ids']) & set(target['cell_ids']):
        raise ValueError('Prediction source IDs overlap future target IDs')
    if set(pred['conditions']) != set(target['conditions']):
        raise ValueError('Prediction/target condition coverage differs')
    gene_target = None
    if target_genes_path:
        with np.load(target_genes_path, allow_pickle=False) as data:
            gene_meta = json.loads(str(data['metadata_json']))
            gene_target = {key: data[key] for key in data.files if key != 'metadata_json'}
        if (gene_meta.get('side') != 'target_gene_space' or gene_meta.get('role') != tm['role']
                or gene_meta.get('transform_hash') != tm['transform_hash'] or not manifest['gene_space_decoded']):
            raise ValueError('Gene-space target metadata/decoder mismatch')
        for key, reference in [('cell_ids', target['cell_ids']), ('conditions', target['conditions']), ('gene_ids', pred['gene_ids'])]:
            if not np.array_equal(gene_target[key], reference):
                raise ValueError(f'Gene-space evaluation alignment mismatch: {key}')
        if not np.isfinite(gene_target['gene_logspliced']).all():
            raise ValueError('Invalid future gene-space values')
    inputs = [prediction_directory/'provenance.json', prediction_directory/'predictions.npz', target_path]
    if target_genes_path:
        inputs.append(target_genes_path)
    with Run(output, stage='distribution_evaluation', kind=manifest['kind'], config=manifest,
             inputs=inputs, seed=manifest['seed']) as run:
        rows, mode_rows, gene_means = [], [], []
        for condition in sorted(set(target['conditions'])):
            pi, ti = pred['conditions'] == condition, target['conditions'] == condition
            p, y = pred['z'][pi], target['z'][ti]
            rows.append({'condition': str(condition), 'seed': manifest['seed'], 'arm': manifest['velocity_arm'],
                         'energy_distance': energy_distance(p, y), 'mean_squared_error': float(np.mean((p.mean(0)-y.mean(0))**2)),
                         'variance_ratio': float(np.var(p, axis=0).sum()/max(np.var(y, axis=0).sum(), 1e-12)),
                         'source_generated_n': len(p), 'target_n': len(y)})
            for k in range(pred['q'].shape[1]):
                mode_rows.append({'condition': str(condition), 'mode': k, 'soft_fraction': float(pred['q'][pi, k].mean()),
                                  'hard_fraction': float((pred['modes'][pi] == k).mean()), 'fate_truth_available': False})
            if gene_target is not None:
                gp, gy = pred['gene_logspliced'][pi].mean(0), gene_target['gene_logspliced'][ti].mean(0)
                denominator = float(np.sum((gy-gy.mean())**2))
                rows[-1]['gene_mean_mse'] = float(np.mean((gp-gy)**2))
                rows[-1]['gene_mean_r2'] = float(1-np.sum((gp-gy)**2)/denominator) if denominator > 1e-12 else None
                gene_means.append((gp, gy))
        if gene_means:
            pm, ym = np.stack([v[0] for v in gene_means]), np.stack([v[1] for v in gene_means])
            centered_errors = ((pm-pm.mean(0))-(ym-ym.mean(0)))**2
            for row, value in zip(rows, centered_errors.mean(1)):
                row['condition_centered_gene_mean_mse'] = float(value)
        save_csv(run.directory/'metrics.csv', rows)
        save_csv(run.directory/'mode_usage.csv', mode_rows)
        (run.directory/'RESULTS.md').write_text('# Distribution evaluation\n\n'+json.dumps(rows, indent=2)
            +'\n\nMetrics use the shared frozen PCA space. This is not the official CellFlow 15-metric suite. '
             'No cell-to-cell fate accuracy is claimed.\n')
    return rows
