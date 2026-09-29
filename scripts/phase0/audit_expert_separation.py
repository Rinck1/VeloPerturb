"""Training-only expert separation and routing sensitivity; no held-out targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from veloroute.artifacts import Run, object_hash, save_csv, save_json
from veloroute.coupling import build_responsibilities
from veloroute.experiments import condition_vectors
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import load_full_checkpoint
from veloroute.full_training import DelayedTeacher, FullTrainConfig, full_responsibilities
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.model import load_checkpoint
from veloroute.probes import permute_local


def separation(values, baseline):
    between = values.var(1, unbiased=False).mean().sqrt()
    denominator = baseline.square().mean().sqrt().clamp_min(1e-10)
    return float(between), float(between/denominator)


def entropy(p):
    return float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--router-root', required=True)
    parser.add_argument('--full-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    fold, router_root, full_root = map(Path, (args.fold, args.router_root, args.full_root))
    source, sm = load_pack(fold/'train_source.npz', expected_side='source')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    if sm['role'] != 'train' or tm['role'] != 'train' or set(source['conditions']) != set(target['conditions']):
        raise ValueError('Expert audit requires matching training-role condition packs')
    config = {**vars(args), 'seed': 20260914, 'cells_per_training_condition': 16,
              'reference': 'training_only', 'confirmation_evaluated': False}
    checkpoints = [router_root/f'velocity_router_seed{seed}/train/model.pt' for seed in (0, 1, 2)]+[full_root/'model.pt']
    with Run(args.output, stage='training_expert_separation_audit', kind='engineering', config=config,
             inputs=[__file__, fold/'train_source.npz', fold/'train_target.npz', args.conditions, *checkpoints]) as run:
        rng = np.random.default_rng(config['seed'])
        selected = np.concatenate([rng.choice(np.flatnonzero(source['conditions'] == label), size=16, replace=False)
                                   for label in sorted(set(source['conditions']))])
        save_csv(run.directory/'audit_cell_ids.csv', [{'cell_id': str(source['cell_ids'][ix]), 'role': 'train'} for ix in selected])
        perm = permute_local(source['velocity'], source, seed=config['seed'], neighbors=10)[1]
        embeddings, _ = condition_vectors(args.conditions, source['conditions'])
        tensor = lambda x: torch.as_tensor(x, device=args.device, dtype=torch.float32)
        s, v, e = [tensor(x[selected]) for x in (source['z'], source['velocity'], embeddings)]
        rows, posterior_rows = [], []
        with torch.no_grad():
            for seed in (0, 1, 2):
                model, metadata = load_checkpoint(checkpoints[seed], map_location=args.device)
                model.eval()
                fields = model.field(s, 4., e)
                q = model.router(s, v, e)
                q_shuffled = model.router(s, tensor(source['velocity'][perm][selected]), e)
                endpoints = torch.stack([model.predict(s, v, e, modes=torch.full((len(s),), k, device=args.device, dtype=torch.long)).endpoint
                                         for k in range(model.config.n_experts)], 1)
                rms, relative = separation(fields, fields)
                endpoint_rms, endpoint_relative = separation(endpoints, endpoints-s[:, None, :])
                static, _ = load_checkpoint(router_root/f'static_router_seed{seed}/train/model.pt', map_location=args.device)
                maximum_field_parameter_difference = max(float((value-static.field.state_dict()[name]).abs().max())
                                                         for name, value in model.field.state_dict().items())
                rows.append({'model': 'minimal_velocity_router', 'seed': seed, 'experts': model.config.n_experts,
                    'field_between_expert_rms': rms, 'field_between_over_total_rms': relative,
                    'endpoint_between_expert_rms': endpoint_rms, 'endpoint_between_over_displacement_rms': endpoint_relative,
                    'router_entropy': entropy(q), 'source_shuffle_q_L1': float((q-q_shuffled).abs().sum(-1).mean()),
                    'max_field_parameter_diff_vs_matched_static': maximum_field_parameter_difference})
                for label in sorted(set(source['conditions'])):
                    ix = np.flatnonzero(source['conditions'][selected] == label)
                    iy = rng.choice(np.flatnonzero(target['conditions'] == label), 16, replace=False)
                    result = build_responsibilities(model, s[ix], tensor(target['z'][iy]), v[ix], e[ix],
                        source_conditions=[label]*len(ix), target_conditions=[label]*len(iy), iterations=500)
                    labels = result['source_responsibilities']
                    posterior_rows.append({'model': 'minimal_velocity_router', 'seed': seed, 'condition': label,
                        'normalized_posterior_entropy': entropy(labels)/np.log(model.config.n_experts),
                        'max_source_posterior_deviation_from_uniform': float((labels-1/model.config.n_experts).abs().max())})
            full, metadata = load_full_checkpoint(full_root/'model.pt', map_location=args.device)
            expected = object_hash(sorted(source['cell_ids'].tolist()))
            if full.reference.fit_ids_hash != expected:
                raise ValueError('Full reference memory is not the declared training source cohort')
            transform = FrozenSplicingTransform.load(fold/'transform.npz')
            unspliced = source['u_features']+transform.u_mean
            tokens, mask, _ = condition_token_sets(args.conditions, source['conditions'])
            up, protein = tensor(unspliced[selected]), tensor(tokens[selected])
            padding = torch.as_tensor(mask[selected], device=args.device, dtype=torch.bool)
            context = full.make_context(s, up, protein, padding)
            fields = full.field(context.z0, 4., context.condition)[:, full.active_modes]
            active = torch.where(full.active_modes)[0].tolist()
            endpoints = torch.stack([full.predict(s, up, protein, padding, modes=torch.full((len(s),), k, device=args.device,
                dtype=torch.long), stochastic=False, force_gate=1.).endpoint for k in active], 1)
            changed = full.make_context(s, tensor(unspliced[perm][selected]), protein, padding)
            rms, relative = separation(fields, fields)
            endpoint_rms, endpoint_relative = separation(endpoints, endpoints-context.z0[:, None, :])
            rows.append({'model': 'full', 'seed': 0, 'experts': len(active), 'field_between_expert_rms': rms,
                'field_between_over_total_rms': relative, 'endpoint_between_expert_rms': endpoint_rms,
                'endpoint_between_over_displacement_rms': endpoint_relative, 'router_entropy': entropy(context.probabilities),
                'source_shuffle_q_L1': float((context.probabilities-changed.probabilities).abs().sum(-1).mean()),
                'max_field_parameter_diff_vs_matched_static': None})
            trainer_file = full_root/'checkpoints/step_0012000/training_state.pt'
            training_state = torch.load(trainer_file, map_location=args.device, weights_only=True)
            teacher = DelayedTeacher(full.config).to(args.device)
            teacher.load_state_dict(training_state['teacher'])
            train_config = FullTrainConfig(**training_state['config'])
            for label in sorted(set(source['conditions'])):
                ix = np.flatnonzero(source['conditions'][selected] == label)
                iy = rng.choice(np.flatnonzero(target['conditions'] == label), 16, replace=False)
                context = full.make_context(s[ix], up[ix], protein[ix], padding[ix])
                result = full_responsibilities(full, context, tensor(target['z'][iy]), train_config,
                                              teacher=teacher, target_encoder=full.state_encoder)
                labels = result['labels'][:, full.active_modes]
                posterior_rows.append({'model': 'full', 'seed': 0, 'condition': label,
                    'normalized_posterior_entropy': entropy(labels)/np.log(len(active)) if len(active) > 1 else 0.,
                    'max_source_posterior_deviation_from_uniform': float((labels-1/len(active)).abs().max())})
        save_csv(run.directory/'expert_separation.csv', rows)
        save_csv(run.directory/'training_posterior_entropy.csv', posterior_rows)
        summary = {'status': 'TRAINING_EXPERT_AUDIT_COMPLETE', 'models': rows,
                   'median_normalized_posterior_entropy_by_model': {name: float(np.median([r['normalized_posterior_entropy'] for r in posterior_rows if r['model'] == name]))
                        for name in ('minimal_velocity_router', 'full')},
                   'confirmation_evaluated': False,
                   'scope': 'Training-only component diagnostics; not an independent fate or multimodality test.'}
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# Training-only expert separation\n\n'+json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
