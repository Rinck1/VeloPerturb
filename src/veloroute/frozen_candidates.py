"""One fixed-budget follow-up: learn expression candidates, freeze, then route."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

from .artifacts import Run, load_config, save_csv, save_json, sha256
from .coupling import build_responsibilities, sinkhorn_log
from .experiments import condition_vectors, evaluate_prediction, predict_model, validate_pair
from .latent import load_pack
from .metrics import paired_condition_bootstrap
from .model import ModelConfig, VeloRoute, flow_matching_loss, routing_loss, save_checkpoint
from .probes import permute_local


def run_frozen_candidates(config_path, output):
    config = load_config(config_path)
    if config['research_status'] != 'exploratory_not_preregistered' or config['arms'] != ['static', 'real', 'shuffled', 'uniform']:
        raise ValueError('This is a fixed four-arm exploratory follow-up')
    if config['seeds'] != [0, 1, 2]:
        raise ValueError('This follow-up requires the fixed three seeds [0, 1, 2]')
    if config.get('confirmation') != 'sealed':
        raise ValueError('Confirmation must remain sealed')
    if any(config[name] < 1 for name in ('candidate_steps', 'router_steps', 'training_OT_samples_per_condition')):
        raise ValueError('Candidate/router budgets must be positive')
    fold, conditions_path = Path(config['fold']), Path(config['conditions'])
    source, sm = load_pack(fold/'train_source.npz', expected_side='source')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    validate_pair(source, sm, target, tm, 'train')
    embeddings, embedding_meta = condition_vectors(conditions_path, source['conditions'])
    if sm['kind'] not in {'engineering', 'development'} or embedding_meta.get('kind') == 'synthetic':
        raise ValueError('Expected real development-only training packs and public protein priors')
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; do not silently change budget')
    torch.set_num_threads(config['cpu_threads'])
    convert = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    z, y, velocity, e = map(convert, (source['z'], target['z'], source['velocity'], embeddings))
    labels = sorted(set(source['conditions']))
    sg = {label: np.flatnonzero(source['conditions'] == label) for label in labels}
    tg = {label: np.flatnonzero(target['conditions'] == label) for label in labels}
    cfg = ModelConfig(**{**config['model'], 'state_dim': z.shape[1], 'velocity_dim': velocity.shape[1], 'condition_dim': e.shape[1]})
    with Run(output, stage='exploratory_frozen_candidate_router', kind='engineering', config=config,
             inputs=[config_path, fold/'train_source.npz', fold/'train_target.npz', conditions_path], seed=None) as run:
        records, separation_rows = [], []
        for seed in config['seeds']:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = VeloRoute(cfg).to(device)
            initial_router = copy.deepcopy(model.router.state_dict())
            pairs, residual_rates = {}, []
            with torch.no_grad():
                for label in labels:
                    si = rng.choice(sg[label], min(len(sg[label]), config['training_OT_samples_per_condition']), replace=False)
                    ti = rng.choice(tg[label], min(len(tg[label]), config['training_OT_samples_per_condition']), replace=False)
                    cost = torch.cdist(z[si], y[ti]).square()
                    plan = sinkhorn_log(cost/cost.mean().clamp_min(1e-8), epsilon=config['epsilon'], iterations=config['sinkhorn_iterations'])
                    draw = torch.multinomial(plan.flatten(), config['training_OT_samples_per_condition'], replacement=True)
                    ii, jj = np.asarray(si)[(draw//len(ti)).cpu().numpy()], np.asarray(ti)[(draw % len(ti)).cpu().numpy()]
                    condition_mean = y[tg[label]].mean(0)-z[sg[label]].mean(0)
                    residual = y[jj]-z[ii]-condition_mean
                    pairs[label] = {'si': ii, 'ti': jj}
                    residual_rates.append(residual.cpu().numpy())
            clustering = KMeans(n_clusters=cfg.n_experts, n_init=10, random_state=seed).fit(np.concatenate(residual_rates))
            for label, residual in zip(labels, residual_rates):
                pairs[label]['mode'] = clustering.predict(residual)
            candidate_config = {**config, 'phase': 'training_expression_candidates', 'seed': seed}
            directory = run.directory/f'candidates_seed{seed}'
            with Run(directory, stage='frozen_candidate_pretraining', kind=sm['kind'], config=candidate_config,
                     inputs=[fold/'train_source.npz', fold/'train_target.npz', conditions_path], seed=seed) as candidate_run:
                optimizer = torch.optim.AdamW(model.field.parameters(), lr=config['learning_rate'])
                trace = []
                for step in range(config['candidate_steps']):
                    label = labels[step % len(labels)]
                    pair = pairs[label]
                    selected = rng.integers(0, len(pair['si']), config['batch_size'])
                    ii, jj = pair['si'][selected], pair['ti'][selected]
                    weights = F.one_hot(torch.as_tensor(pair['mode'][selected], dtype=torch.long, device=device), cfg.n_experts).float()
                    optimizer.zero_grad(set_to_none=True)
                    loss = flow_matching_loss(model, z[ii], y[jj], e[ii], weights, t0=4., t1=5.)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite fixed-candidate field loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.field.parameters(), 1.)
                    optimizer.step()
                    trace.append({'step': step, 'condition': label, 'field_loss': float(loss.detach())})
                save_csv(candidate_run.directory/'training_trace.csv', trace)
                with (candidate_run.directory/'cluster_centres.npy').open('xb') as stream:
                    np.save(stream, clustering.cluster_centers_, allow_pickle=False)
                candidate_metadata = {'kind': sm['kind'], 'research_status': 'exploratory_frozen_candidates', 'seed': seed,
                    'transform_hash': sm['transform_hash'], 'fit_ids_hash': sm['fit_ids_hash'], 'condition_hash': sha256(conditions_path),
                    'task': 'ER-short', 'velocity_arm': 'real', 'candidate_labels': config['candidate_labels'],
                    'confirmation': 'SEALED', 'protocol_hash': 'exploratory_NOT_G1'}
                save_checkpoint(candidate_run.directory/'model.pt', model, metadata=candidate_metadata)
                (candidate_run.directory/'RESULTS.md').write_text('# Frozen expression candidates\n\nTraining-only residual displacement clusters; no velocity used for candidate pretraining.\n')
            model.field.requires_grad_(False)
            for parameter in model.field.parameters():
                parameter.grad = None
            field_reference = {key: value.clone() for key, value in model.field.state_dict().items()}
            with torch.no_grad():
                field_values = model.field(z[:256], 4., e[:256])
                separation_rows.append({'seed': seed, 'between_expert_rms': float(field_values.var(1, unbiased=False).mean().sqrt()),
                    'between_over_total_rms': float(field_values.var(1, unbiased=False).mean().sqrt()/field_values.square().mean().sqrt().clamp_min(1e-10))})
            for arm in config['arms']:
                model.router.load_state_dict(initial_router)
                torch.manual_seed(seed)
                rng_arm = np.random.default_rng(seed)
                v = velocity
                if arm == 'static':
                    v = torch.zeros_like(velocity)
                elif arm == 'shuffled':
                    v = convert(permute_local(source['velocity'], source, seed=seed, neighbors=10)[0])
                arm_directory = run.directory/f'{arm}_seed{seed}'
                arm_config = {**config, 'phase': 'router_only', 'arm': arm, 'seed': seed}
                save_json(run.directory/'status.json', {'status': 'router_training', 'seed': seed, 'arm': arm})
                with Run(arm_directory/'train', stage='frozen_expert_router_training', kind=sm['kind'], config=arm_config,
                         inputs=[directory/'model.pt', fold/'train_source.npz', fold/'train_target.npz', conditions_path], seed=seed) as arm_run:
                    optimizer = torch.optim.AdamW(model.router.parameters(), lr=config['learning_rate'])
                    trace = []
                    if arm == 'uniform':
                        with torch.no_grad():
                            for parameter in model.router.parameters():
                                parameter.zero_()
                    else:
                        for step in range(config['router_steps']):
                            label = labels[step % len(labels)]
                            si = rng_arm.choice(sg[label], config['batch_size'], replace=True)
                            ti = rng_arm.choice(tg[label], config['batch_size'], replace=True)
                            result = build_responsibilities(model, z[si], y[ti], v[si], e[si], source_conditions=[label]*len(si),
                                target_conditions=[label]*len(ti), lambda_dynamic=0. if arm == 'static' else config['lambda_dynamic'],
                                temperature=config['temperature'], epsilon=config['epsilon'], iterations=config['sinkhorn_iterations'])
                            optimizer.zero_grad(set_to_none=True)
                            loss = routing_loss(model, z[si], v[si], e[si], result['source_responsibilities'])
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.)
                            optimizer.step()
                            posterior = result['source_responsibilities']
                            trace.append({'step': step, 'condition': label, 'router_loss': float(loss.detach()),
                                'posterior_entropy': float(-(posterior*posterior.clamp_min(1e-12).log()).sum(-1).mean())})
                    for key, reference in field_reference.items():
                        if not torch.equal(reference, model.field.state_dict()[key]):
                            raise RuntimeError('Frozen expression experts changed while training the router')
                    metadata = {**candidate_metadata, 'velocity_arm': arm if arm != 'uniform' else 'real',
                                'routing_policy': arm, 'experts_frozen': True, 'lambda_dynamic': 0. if arm in {'static', 'uniform'} else config['lambda_dynamic']}
                    save_checkpoint(arm_run.directory/'model.pt', model, metadata=metadata)
                    save_csv(arm_run.directory/'training_trace.csv', trace, fields=['step', 'condition', 'router_loss', 'posterior_entropy'])
                    (arm_run.directory/'RESULTS.md').write_text('# Frozen-expert router\n\nSame frozen expression experts; no validation outcomes or free parameter search.\n')
                predict_model(arm_directory/'train/model.pt', fold/'validation_source.npz', conditions_path,
                              arm_directory/'predict', seed=seed, transform_path=fold/'transform.npz')
                records.append({'arm': arm, 'seed': seed, 'directory': str(arm_directory/'predict'),
                                'sha256': sha256(arm_directory/'predict/predictions.npz')})
        save_json(run.directory/'frozen_predictions.json', records)
        save_csv(run.directory/'candidate_separation.csv', separation_rows)
        rows = []
        for record in records:
            directory = Path(record['directory'])
            if sha256(directory/'predictions.npz') != record['sha256']:
                raise ValueError('Frozen candidate prediction modified before evaluation')
            metrics = evaluate_prediction(directory, fold/'validation_target.npz', directory.parent/'evaluate',
                                           target_genes_path=fold/'validation_target_genes.npz')
            for metric in metrics:
                metric['arm'] = record['arm']
            rows.extend(metrics)
        save_csv(run.directory/'metrics.csv', rows)
        comparisons = [paired_condition_bootstrap(rows, real_arm='real', control_arm=control, seeds=config['seeds'],
            conditions=config['validation_conditions'], n_bootstrap=config['n_bootstrap'], seed=config['bootstrap_seed'],
            confidence=1-config['familywise_alpha']/3) for control in ('static', 'shuffled', 'uniform')]
        save_json(run.directory/'comparisons.json', comparisons)
        summary = {'status': 'FROZEN_CANDIDATE_FOLLOWUP_COMPLETE', 'training_seeds': 3, 'prediction_runs': len(records),
                   'experts_verified_frozen': True, 'confirmation_evaluated': False, 'comparisons': comparisons,
                   'research_status': 'exploratory_followup_after_development_inspection'}
        save_json(run.directory/'summary.json', summary)
        save_json(run.directory/'status.json', {'status': 'complete'})
        (run.directory/'RESULTS.md').write_text('# Fixed-budget frozen-candidate follow-up\n\n'+json.dumps(summary, indent=2)
            +'\n\nThis follows a training-only expert-separation diagnosis, but the development set has been viewed before. '
             'It is not a new independent confirmation or a fate-label experiment.\n')
    return summary
